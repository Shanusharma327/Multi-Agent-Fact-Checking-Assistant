"""Emit the oracle, sensitivity and power macros the revised paper cites.

Kept separate from condition_tables.py because these quantities compare
two families of runs rather than describing one, and because the paper
must label every oracle number as diagnostic. Appending them to the same
macro file keeps one place for the prose to draw on.

    python src/analysis/oracle_macros.py --live results/av250 \\
        --oracle results/av250_oracle --cache cache/av250.jsonl \\
        --out results/macros/condition_numbers.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_conditions import METRICS, load_conditions, rows_for  # noqa: E402
from ablation import apply_policy, make_deterministic  # noqa: E402
from error_attribution import attribute_run  # noqa: E402
from outage_sensitivity import compare, degraded_claims  # noqa: E402
from stats import (  # noqa: E402
    mcnemar,
    DEFAULT_RESAMPLES,
    bootstrap_metric,
    holm_correct,
    minimum_detectable_difference,
    observed_discordance,
)

from fact_checker.cache import PassageCache  # noqa: E402
from fact_checker.retrieval_metrics import score_citations  # noqa: E402

KEY = {"B0": "Bzero", "B1": "Bone", "B4": "Bfour", "MAFCA": "Mafca",
       "ORACLE_B1": "OracleBone", "ORACLE_MAFCA": "OracleMafca"}


def _records_for(source, cond, ids):
    """Records for one condition, deriving B4 where it was not run.

    The oracle sweep ran B1 and MAFCA only. B4 is MAFCA with the
    deterministic policy applied to the same stored audit, so it is
    recovered by replaying that policy rather than by substituting a
    different configuration, which would mismatch the live profile it
    is compared against.
    """
    if cond in source:
        return [source[cond][c] for c in ids]
    if cond != "B4" or "MAFCA" not in source:
        return None
    wrapped = [{"multi": source["MAFCA"][c]} for c in ids]
    replayed = apply_policy(wrapped, make_deterministic(1.0, gate=True))
    return [r["multi"] for r in replayed]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=20240101)
    args = parser.parse_args(argv)

    live = load_conditions(args.live)
    oracle = load_conditions(args.oracle)
    cache = PassageCache.load(args.cache)
    empty, partial = degraded_claims(cache)
    degraded = empty | partial

    ids = sorted(set.intersection(
        *(set(c) for c in live.values()), *(set(c) for c in oracle.values())
    ))
    clean = [c for c in ids if c not in degraded]

    arms: Dict[str, Sequence[dict]] = {
        n: rows_for(r, ids) for n, r in live.items()
    }
    arms.update({f"ORACLE_{n}": rows_for(r, ids) for n, r in oracle.items()})
    clean_arms: Dict[str, Sequence[dict]] = {
        n: rows_for(r, clean) for n, r in live.items()
    }
    clean_arms.update({f"ORACLE_{n}": rows_for(r, clean) for n, r in oracle.items()})

    macros: Dict[str, str] = {}

    def put(key: str, value: str) -> None:
        macros[key] = value

    # --- oracle condition figures -----------------------------------------
    for name in ("ORACLE_B1", "ORACLE_MAFCA"):
        if name not in arms:
            continue
        key = KEY[name]
        for metric, suffix in (("accuracy", "Acc"), ("macro_f1", "FOne"),
                               ("decidable_acc", "Dec")):
            interval = bootstrap_metric(arms[name], METRICS[metric],
                                        resamples=args.resamples, seed=args.seed)
            put(f"Cond{key}{suffix}", f"{100 * interval.estimate:.1f}")
            put(f"Cond{key}{suffix}Lo", f"{100 * interval.low:.1f}")
            put(f"Cond{key}{suffix}Hi", f"{100 * interval.high:.1f}")
        put(f"Cond{key}Ece", f"{METRICS['ece10'](arms[name]):.3f}")
        put(f"Cond{key}Brier", f"{METRICS['brier'](arms[name]):.3f}")
        source = oracle[name.replace("ORACLE_", "")]
        citations = score_citations([{"multi": source[c]} for c in ids])
        put(f"Cond{key}Cited", f"{100 * citations.grounded_rate:.1f}")
        put(f"Cond{key}Support", f"{100 * citations.support_precision:.1f}")

    # --- paired contrasts, with Holm across the oracle family --------------
    family = [("ORACLE_B1", "B1"), ("ORACLE_MAFCA", "MAFCA"),
              ("ORACLE_MAFCA", "ORACLE_B1")]
    raw: List[float] = []
    for a, b in family:
        _, _, _, p = compare(arms[a], arms[b], "accuracy", args.resamples, args.seed)
        raw.append(p)
    adjusted = holm_correct(raw)

    for (a, b), p_adj in zip(family, adjusted):
        d, lo, hi, p = compare(arms[a], arms[b], "accuracy",
                               args.resamples, args.seed)
        tag = f"{KEY[a]}Vs{KEY[b]}"
        put(f"CondD{tag}", f"{100 * d:+.1f}")
        put(f"CondD{tag}Lo", f"{100 * lo:.1f}")
        put(f"CondD{tag}Hi", f"{100 * hi:.1f}")
        put(f"CondD{tag}P", f"{p:.3f}")
        put(f"CondD{tag}Holm", f"{p_adj:.3f}")
        # On the clean subset, which is the sensitivity check.
        dc, lc, hc, _ = compare(clean_arms[a], clean_arms[b], "accuracy",
                                args.resamples, args.seed)
        put(f"CondD{tag}Clean", f"{100 * dc:+.1f}")
        put(f"CondD{tag}CleanLo", f"{100 * lc:.1f}")
        put(f"CondD{tag}CleanHi", f"{100 * hc:.1f}")

    # --- live contrasts the assignment requires first ----------------------
    for a, b in (("MAFCA", "B0"), ("MAFCA", "B1"), ("B4", "B1"), ("B4", "B0")):
        if a not in arms or b not in arms:
            continue
        d, lo, hi, p = compare(arms[a], arms[b], "accuracy",
                               args.resamples, args.seed)
        tag = f"{KEY[a]}Vs{KEY[b]}"
        put(f"CondD{tag}", f"{100 * d:+.1f}")
        put(f"CondD{tag}Lo", f"{100 * lo:.1f}")
        put(f"CondD{tag}Hi", f"{100 * hi:.1f}")
        put(f"CondD{tag}P", f"{p:.3f}")
        dm, lm, hm, _ = compare(arms[a], arms[b], "macro_f1",
                                args.resamples, args.seed)
        put(f"CondD{tag}F", f"{100 * dm:+.1f}")
        put(f"CondD{tag}FLo", f"{100 * lm:.1f}")
        put(f"CondD{tag}FHi", f"{100 * hm:.1f}")

    # --- sensitivity ------------------------------------------------------
    put("CondDegraded", str(len(degraded & set(ids))))
    put("CondDegradedEmpty", str(len(empty & set(ids))))
    put("CondDegradedPartial", str(len(partial & set(ids))))
    put("CondCleanN", str(len(clean)))

    # Per-claim win and loss against the single LLM. The ratio is a
    # plainer statement of the same evidence as the paired test, and it
    # survives being quoted without an interval attached, which a point
    # estimate does not.
    for name, key in (("MAFCA", "Mafca"), ("B4", "Bfour")):
        if name not in arms:
            continue
        _, wins, losses = mcnemar(
            [r["correct"] for r in arms[name]],
            [r["correct"] for r in arms["B0"]],
        )
        put(f"Cond{key}OnlyRight", str(wins))
        put(f"Cond{key}BzeroOnlyRight", str(losses))
        put(f"Cond{key}WinRatio", f"{wins / losses:.1f}" if losses else "n/a")

    # Prose rather than a number, but it belongs with the figures it
    # qualifies: every mention of the sample must carry its provenance.
    put(
        "CondCondTrainNote",
        "as the first \\CondN{} claims of a seeded, label-stratified draw "
        "across the AVeriTeC development and training splits "
        "(\\CondTrain{} from train, \\CondDev{} from dev)",
    )

    # --- error attribution, live and oracle --------------------------------
    for tag, source in (("Live", live), ("Oracle", oracle)):
        # The primary configuration is the computed one, so its profile
        # carries the plain macro names; the agent-reported variant is
        # suffixed. Quoting one beside a headline drawn from the other
        # mismatches the configuration.
        for cond, suffix in (("B4", ""), ("MAFCA", "Reported")):
            rows = _records_for(source, cond, ids)
            if rows is None:
                continue
            summary = attribute_run(rows)
            for stage in ("retrieval", "verification", "aggregation"):
                put(f"CondErr{tag}{stage.capitalize()}{suffix}",
                    f"{100 * summary['shares'][stage]:.1f}")
            put(f"CondErr{tag}Wrong{suffix}", str(summary["n_wrong"]))
    clean_summary = attribute_run([live["B4"][c] for c in clean])
    put("CondErrCleanRetrieval",
        f"{100 * clean_summary['shares']['retrieval']:.1f}")

    # --- power -------------------------------------------------------------
    # One pair of figures per contrast. Quoting the power of one
    # comparison beside the estimate of another understates how large a
    # difference the sample could actually have resolved: discordance
    # differs between contrasts, and a higher discordance needs a larger
    # difference to detect.
    for a, b, tag in (("B4", "B0", ""),          # the headline contrast
                      ("MAFCA", "B1", "MafcaBone")):
        if a not in arms or b not in arms:
            continue
        disc = observed_discordance(
            [r["correct"] for r in arms[a]], [r["correct"] for r in arms[b]]
        )
        power = minimum_detectable_difference(len(ids), discordance=disc)
        put(f"CondMdd{tag}", f"{100 * power['mdd']:.1f}")
        put(f"CondDiscordance{tag}", f"{100 * disc:.1f}")
        put(f"CondInformativePairs{tag}",
            f"{power['informative_pairs']:.0f}")

    existing = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
    addition = "\n".join(
        ["", "% --- oracle, sensitivity and power (analysis/oracle_macros.py) ---"]
        + [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
        + [""]
    )
    args.out.write_text(existing.rstrip("\n") + "\n" + addition, encoding="utf-8")
    print(f"appended {len(macros)} macros -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
