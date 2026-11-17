"""Sensitivity: do the conclusions survive dropping outage-degraded claims?

Eleven of the ninety cached claims were retrieved while the search host
was failing. Two returned no passages at all and nine returned fewer
than they otherwise would have, because one of their two queries timed
out. Those claims carry less evidence than the protocol intended, and
every evidence-using condition saw the same impoverished set.

That cuts both ways and neither direction is obvious in advance: thin
evidence could flatter the ungrounded baseline, which loses nothing by
it, or flatter the grounded systems, if the passages that survived were
the easy ones. The only way to know whether a conclusion rests on them
is to remove them and look.

    python src/analysis/outage_sensitivity.py --live results/av250 \\
        --oracle results/av250_oracle --cache cache/av250.jsonl

Reports the headline comparisons on all claims and on the clean subset
side by side, and says for each whether the conclusion changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_conditions import METRICS, load_conditions, rows_for  # noqa: E402
from error_attribution import attribute_run  # noqa: E402
from stats import (  # noqa: E402
    DEFAULT_RESAMPLES,
    apply_holm,
    bootstrap_metric,
    holm_correct,
    mcnemar,
    minimum_detectable_difference,
    observed_discordance,
    paired_bootstrap_difference,
)

from fact_checker.cache import PassageCache  # noqa: E402

# A warning naming one of these means the search host failed on that
# query, so the claim's evidence is thinner than the protocol intended.
OUTAGE_WORDS = (
    "timed out", "timeout", "throttled", "connection reset",
    "ratelimit", "dnserror", "search failed",
)


def degraded_claims(cache: PassageCache) -> Tuple[Set[str], Set[str]]:
    """Claims whose retrieval hit a search failure, split by severity.

    Returns ``(empty, partial)``: those that returned nothing, and those
    that returned something but lost at least one query to a failure.
    """
    empty: Set[str] = set()
    partial: Set[str] = set()
    for entry in cache:
        text = " ".join(entry.retrieval_warnings).lower()
        hit = any(word in text for word in OUTAGE_WORDS)
        if entry.is_empty:
            empty.add(entry.claim_id)
        elif hit:
            partial.add(entry.claim_id)
    return empty, partial


def compare(
    rows_a: Sequence[dict],
    rows_b: Sequence[dict],
    metric: str,
    resamples: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    """``(difference, low, high, p)`` for one paired contrast."""
    function = METRICS[metric]
    pairs = list(zip(rows_a, rows_b))
    interval = paired_bootstrap_difference(
        pairs,
        lambda s: function([a for a, _ in s]),
        lambda s: function([b for _, b in s]),
        resamples=resamples,
        seed=seed,
    )
    if metric == "accuracy":
        p_value, _, _ = mcnemar(
            [a["correct"] for a, _ in pairs], [b["correct"] for _, b in pairs]
        )
    else:
        p_value = float("nan")
    return interval.estimate, interval.low, interval.high, p_value


def verdict_of(low: float, high: float) -> str:
    return "clears 0" if (low > 0 or high < 0) else "spans 0"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, default=None)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=20240101)
    args = parser.parse_args(argv)

    cache = PassageCache.load(args.cache)
    empty, partial = degraded_claims(cache)
    degraded = empty | partial

    live = load_conditions(args.live)
    oracle = load_conditions(args.oracle) if args.oracle else {}

    pools = [set(c) for c in live.values()] + [set(c) for c in oracle.values()]
    all_ids = sorted(set.intersection(*pools)) if pools else []
    clean_ids = [c for c in all_ids if c not in degraded]

    print(f"Outage-degraded claims: {len(degraded & set(all_ids))}")
    print(f"  returned nothing           {len(empty & set(all_ids))}")
    print(f"  lost a query to a failure  {len(partial & set(all_ids))}")
    print(f"\nall claims   n = {len(all_ids)}")
    print(f"clean subset n = {len(clean_ids)}")

    def arms(ids: Sequence[str]) -> Dict[str, Sequence[dict]]:
        out = {name: rows_for(recs, ids) for name, recs in live.items()}
        out.update({f"ORACLE_{n}": rows_for(r, ids) for n, r in oracle.items()})
        return out

    full, clean = arms(all_ids), arms(clean_ids)

    # --- point estimates ---------------------------------------------------
    print("\n" + "=" * 94)
    print("  ACCURACY: ALL CLAIMS vs CLEAN SUBSET")
    print("=" * 94)
    print(f"  {'condition':<32}{'all (n=%d)' % len(all_ids):>24}"
          f"{'clean (n=%d)' % len(clean_ids):>24}{'shift':>10}")
    print("  " + "-" * 90)
    order = [n for n in ("MAJORITY", "RANDOM", "B0", "B1", "MAFCA", "B4",
                         "ORACLE_B1", "ORACLE_MAFCA") if n in full]
    for name in order:
        a = bootstrap_metric(full[name], METRICS["accuracy"],
                             resamples=args.resamples, seed=args.seed)
        c = bootstrap_metric(clean[name], METRICS["accuracy"],
                             resamples=args.resamples, seed=args.seed)
        print(
            f"  {name:<32}"
            f"{100 * a.estimate:>8.1f} [{100 * a.low:>5.1f},{100 * a.high:>6.1f}]"
            f"{100 * c.estimate:>8.1f} [{100 * c.low:>5.1f},{100 * c.high:>6.1f}]"
            f"{100 * (c.estimate - a.estimate):>+10.1f}"
        )

    # --- headline contrasts -------------------------------------------------
    contrasts = [("MAFCA", "B0"), ("MAFCA", "B1"), ("B4", "B1")]
    if "ORACLE_B1" in full:
        contrasts += [("ORACLE_MAFCA", "ORACLE_B1"),
                      ("ORACLE_B1", "B1"), ("ORACLE_MAFCA", "MAFCA")]

    print("\n" + "=" * 94)
    print("  PAIRED CONTRASTS: DOES ANY CONCLUSION CHANGE?")
    print("=" * 94)
    print(f"  {'contrast':<34}{'all claims':>26}{'clean subset':>26}{'same?':>7}")
    print("  " + "-" * 90)
    changed: List[str] = []
    for a_name, b_name in contrasts:
        if a_name not in full or b_name not in full:
            continue
        da, la, ha, _ = compare(full[a_name], full[b_name], "accuracy",
                                args.resamples, args.seed)
        dc, lc, hc, _ = compare(clean[a_name], clean[b_name], "accuracy",
                                args.resamples, args.seed)
        same = verdict_of(la, ha) == verdict_of(lc, hc)
        if not same:
            changed.append(f"{a_name} - {b_name}")
        print(
            f"  {a_name + ' - ' + b_name:<34}"
            f"{100 * da:>+7.1f} [{100 * la:>5.1f},{100 * ha:>6.1f}] {verdict_of(la, ha):<9}"
            f"{100 * dc:>+7.1f} [{100 * lc:>5.1f},{100 * hc:>6.1f}] {verdict_of(lc, hc):<9}"
            f"{'yes' if same else 'NO':>5}"
        )

    print()
    if changed:
        print(f"  CONCLUSION CHANGED for: {', '.join(changed)}")
    else:
        print("  No conclusion changes. Every contrast keeps the same verdict on")
        print("  whether its interval excludes zero, so none of them rests on the")
        print("  outage-degraded claims.")

    # --- error attribution --------------------------------------------------
    if "MAFCA" in live:
        print("\n" + "=" * 94)
        print("  ERROR ATTRIBUTION (MAFCA, live retrieval)")
        print("=" * 94)
        for tag, ids in (("all claims", all_ids), ("clean subset", clean_ids)):
            summary = attribute_run([live["MAFCA"][c] for c in ids])
            shares = summary["shares"]
            print(
                f"  {tag:<16}n={len(ids):<5}wrong={summary['n_wrong']:<4}"
                f"retrieval {100 * shares['retrieval']:5.1f}%   "
                f"verification {100 * shares['verification']:5.1f}%   "
                f"aggregation {100 * shares['aggregation']:5.1f}%"
            )

    # --- Holm across the oracle-vs-live family ------------------------------
    if "ORACLE_B1" in full:
        print("\n" + "=" * 94)
        print("  HOLM CORRECTION ACROSS THE ORACLE-vs-LIVE FAMILY")
        print("=" * 94)
        family = [("ORACLE_B1", "B1"), ("ORACLE_MAFCA", "MAFCA"),
                  ("ORACLE_MAFCA", "ORACLE_B1")]
        raw: List[float] = []
        rows: List[Tuple[str, float, float, float, float]] = []
        for a_name, b_name in family:
            d, lo, hi, p = compare(full[a_name], full[b_name], "accuracy",
                                   args.resamples, args.seed)
            raw.append(p)
            rows.append((f"{a_name} - {b_name}", d, lo, hi, p))
        adjusted = holm_correct(raw)
        print(f"  {'contrast':<38}{'difference':>22}{'p':>9}{'p(Holm)':>10}  survives")
        print("  " + "-" * 90)
        for (name, d, lo, hi, p), p_adj in zip(rows, adjusted):
            survives = "yes" if p_adj < 0.05 else "no"
            print(
                f"  {name:<38}{100 * d:>+7.1f} [{100 * lo:>5.1f},{100 * hi:>6.1f}]"
                f"{p:>9.3f}{p_adj:>10.3f}  {survives}"
            )
        print(
            f"\n  Family size {len(family)}; Holm multiplies the smallest raw p by "
            f"{len(family)}."
        )

    # --- power on the clean subset ------------------------------------------
    if "MAFCA" in clean and "B1" in clean:
        discordance = observed_discordance(
            [r["correct"] for r in clean["MAFCA"]],
            [r["correct"] for r in clean["B1"]],
        )
        power = minimum_detectable_difference(len(clean_ids), discordance=discordance)
        print(f"\n  Clean subset power: discordance {100 * discordance:.1f}%, "
              f"minimum detectable difference {100 * power['mdd']:.1f} pp.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
