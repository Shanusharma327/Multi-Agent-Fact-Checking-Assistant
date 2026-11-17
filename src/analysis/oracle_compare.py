"""Compare oracle retrieval against live retrieval, paired claim by claim.

Error attribution puts most of MAFCA's failures on retrieval. The oracle
condition is what says how much of that is recoverable: it substitutes
the evidence a human fact-checker actually used for whatever search
returned, holding the model, the prompts and the claim set fixed.

Two families of comparison, and they answer different questions:

* **oracle-MAFCA vs oracle-B1** asks whether decomposition helps *when
  retrieval is not the bottleneck*. If the decomposition only pays off
  on good evidence, this is where it would show.
* **each condition against its own live counterpart** asks how much
  headroom retrieval is costing. It is the size of the prize for
  fixing retrieval, measured rather than asserted.

Every comparison is paired on the same claims, so a paired bootstrap and
exact McNemar apply. A difference whose interval spans zero is reported
as no difference demonstrated.

    python src/analysis/oracle_compare.py --live results/av250 \\
        --oracle results/av250_oracle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_conditions import (  # noqa: E402
    METRICS,
    abstention_rate,
    accuracy_when_committing,
    load_conditions,
    rows_for,
)
from stats import (  # noqa: E402
    DEFAULT_RESAMPLES,
    bootstrap_metric,
    mcnemar,
    minimum_detectable_difference,
    observed_discordance,
    paired_bootstrap_difference,
)

from fact_checker.retrieval_metrics import score_citations  # noqa: E402


def fmt(interval, places: int = 1, scale: float = 100.0) -> str:
    return (
        f"{interval.estimate * scale:>6.{places}f} "
        f"[{interval.low * scale:>6.{places}f},{interval.high * scale:>7.{places}f}]"
    )


def paired_line(
    name: str,
    rows_a: Sequence[dict],
    rows_b: Sequence[dict],
    metric: str,
    resamples: int,
    seed: int,
) -> str:
    """One paired difference, with its interval and (for accuracy) a test."""
    function = METRICS[metric]
    pairs = list(zip(rows_a, rows_b))
    difference = paired_bootstrap_difference(
        pairs,
        lambda sample: function([a for a, _ in sample]),
        lambda sample: function([b for _, b in sample]),
        resamples=resamples,
        seed=seed,
    )
    if metric == "accuracy":
        p_value, a_only, b_only = mcnemar(
            [a["correct"] for a, _ in pairs], [b["correct"] for _, b in pairs]
        )
        test = f"p={p_value:.3f}  ({a_only} vs {b_only} discordant)"
    else:
        test = ""
    flag = "no difference demonstrated" if not difference.excludes_zero else "CI clears 0"
    return f"  {name:<44}{fmt(difference)}  {test:<28}{flag}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=20240101)
    args = parser.parse_args(argv)

    live_raw = load_conditions(args.live)
    oracle_raw = load_conditions(args.oracle)
    if not live_raw or not oracle_raw:
        print("Missing condition files.", file=sys.stderr)
        return 1

    # Pair on claims present in every arm of both families.
    shared = set.intersection(
        *(set(c) for c in live_raw.values()),
        *(set(c) for c in oracle_raw.values()),
    )
    claim_ids = sorted(shared)
    if not claim_ids:
        print("No claim is present in every condition.", file=sys.stderr)
        return 1

    arms: Dict[str, Sequence[dict]] = {}
    for name, records in live_raw.items():
        arms[name] = rows_for(records, claim_ids)
    for name, records in oracle_raw.items():
        arms[f"ORACLE_{name}"] = rows_for(records, claim_ids)

    label = {
        "B0": "B0 single-LLM, no retrieval",
        "B1": "B1 single-call RAG (live)",
        "MAFCA": "MAFCA (live)",
        "B4": "B4 computed verdict (live)",
        "MAJORITY": "Majority class",
        "RANDOM": "Uniform random",
        "ORACLE_B1": "B1 single-call RAG (oracle)",
        "ORACLE_MAFCA": "MAFCA (oracle)",
    }
    order = ["MAJORITY", "RANDOM", "B0", "B1", "MAFCA", "B4",
             "ORACLE_B1", "ORACLE_MAFCA"]
    order = [n for n in order if n in arms]

    print(f"\nPaired on {len(claim_ids)} claims present in every condition.")
    first = next(iter(live_raw.values()))
    counts: Dict[str, int] = {}
    for claim_id in claim_ids:
        gold = first[claim_id].get("gold", "?")
        counts[gold] = counts.get(gold, 0) + 1
    print(f"  class counts  {dict(sorted(counts.items()))}")

    print("\n" + "=" * 96)
    print(f"  CLASSIFICATION  |  {args.resamples:,}-resample bootstrap 95% CI")
    print("=" * 96)
    print(f"  {'condition':<34}{'accuracy':>22}{'macro-F1':>22}{'decidable':>22}")
    print("  " + "-" * 92)
    for name in order:
        cells = "".join(
            f"{fmt(bootstrap_metric(arms[name], METRICS[m], resamples=args.resamples, seed=args.seed)):>22}"
            for m in ("accuracy", "macro_f1", "decidable_acc")
        )
        print(f"  {label.get(name, name):<34}{cells}")

    # --- the two comparisons that matter -----------------------------------
    print("\n" + "=" * 96)
    print("  ORACLE RETRIEVAL: DOES DECOMPOSITION HELP WHEN EVIDENCE IS GOOD?")
    print("=" * 96)
    if "ORACLE_MAFCA" in arms and "ORACLE_B1" in arms:
        for metric in ("accuracy", "macro_f1"):
            print(
                paired_line(
                    f"oracle-MAFCA - oracle-B1 ({metric})",
                    arms["ORACLE_MAFCA"], arms["ORACLE_B1"],
                    metric, args.resamples, args.seed,
                )
            )

    print("\n" + "=" * 96)
    print("  HEADROOM: WHAT IS LIVE RETRIEVAL COSTING?")
    print("=" * 96)
    for oracle_name, live_name in (("ORACLE_MAFCA", "MAFCA"), ("ORACLE_B1", "B1")):
        if oracle_name in arms and live_name in arms:
            for metric in ("accuracy", "macro_f1"):
                print(
                    paired_line(
                        f"{label[oracle_name]} - {label[live_name]} ({metric})",
                        arms[oracle_name], arms[live_name],
                        metric, args.resamples, args.seed,
                    )
                )

    # --- behaviour ----------------------------------------------------------
    print("\n" + "=" * 96)
    print("  BEHAVIOUR")
    print("=" * 96)
    print(f"  {'condition':<34}{'abstains':>10}{'acc|commit':>12}"
          f"{'ECE@10':>10}{'Brier':>9}{'cited':>8}{'support':>9}")
    print("  " + "-" * 92)
    for name in order:
        rows = arms[name]
        source = oracle_raw if name.startswith("ORACLE_") else live_raw
        key = name.replace("ORACLE_", "") if name.startswith("ORACLE_") else name
        records = [source[key][c] for c in claim_ids]
        citations = score_citations([{"multi": r} for r in records])
        print(
            f"  {label.get(name, name):<34}"
            f"{100 * abstention_rate(rows):>10.1f}"
            f"{100 * accuracy_when_committing(rows):>12.1f}"
            f"{METRICS['ece10'](rows):>10.3f}"
            f"{METRICS['brier'](rows):>9.3f}"
            f"{100 * citations.grounded_rate:>8.1f}"
            f"{100 * citations.support_precision:>9.1f}"
        )

    # --- power --------------------------------------------------------------
    print("\n" + "=" * 96)
    print("  POWER")
    print("=" * 96)
    if "ORACLE_MAFCA" in arms and "ORACLE_B1" in arms:
        discordance = observed_discordance(
            [r["correct"] for r in arms["ORACLE_MAFCA"]],
            [r["correct"] for r in arms["ORACLE_B1"]],
        )
        power = minimum_detectable_difference(len(claim_ids), discordance=discordance)
        print(
            f"  oracle-MAFCA vs oracle-B1: observed discordance "
            f"{100 * discordance:.1f}% over {len(claim_ids)} claims.\n"
            f"  Smallest paired accuracy difference detectable at alpha=0.05 "
            f"with 80% power: {100 * power['mdd']:.1f} percentage points\n"
            f"  ({power['informative_pairs']:.0f} informative pairs). A smaller "
            "difference cannot be told from chance here,\n  whatever the point "
            "estimate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
