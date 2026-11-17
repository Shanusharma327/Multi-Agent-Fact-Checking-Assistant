"""Compare experimental conditions with intervals, tests and diagnostics.

Reads the per-condition JSONL files written by ``run_conditions.py`` and
produces the tables the revision needs: classification quality with
bootstrap intervals, paired differences with exact tests and Holm
correction, abstention and risk-coverage behaviour, calibration, citation
quality, the Eq. (13) span diagnostic, and cost.

The comparison order is deliberate. **MAFCA against B1 comes first**,
because that is the only pairing in which the evidence is held fixed and
the decomposition is the variable. MAFCA against B0 follows, labelled for
what it is: a comparison that varies retrieval and decomposition together
and therefore cannot attribute a difference to either.

    python src/analysis/compare_conditions.py --dir results/smoke60

Every number printed here is computed from the record files; nothing is
carried over from a previous run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reweight import AVERITEC_DEV_PRIOR, print_reweighted  # noqa: E402
from stats import (  # noqa: E402
    DEFAULT_RESAMPLES,
    Interval,
    PairedTest,
    apply_holm,
    bootstrap_metric,
    mcnemar,
    minimum_detectable_difference,
    observed_discordance,
    paired_bootstrap_difference,
)

from fact_checker.groundedness import aggregate_span_reports  # noqa: E402
from fact_checker.metrics import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    per_label_scores,
)
from fact_checker.retrieval_metrics import score_citations, score_retrieval  # noqa: E402
from fact_checker.schemas import VERDICT_VALUES  # noqa: E402

DECIDABLE = ("SUPPORTS", "REFUTES")

# Printed in this order wherever conditions are listed.
ORDER = ["MAJORITY", "RANDOM", "B0", "B1", "B2", "MAFCA", "B3", "B4", "B6"]

LABELS = {
    "MAJORITY": "Majority class",
    "RANDOM": "Uniform random",
    "B0": "B0 single-LLM, no retrieval",
    "B1": "B1 single-call RAG",
    "B2": "B2 MAFCA - verification",
    "MAFCA": "MAFCA (agent-reported)",
    "B3": "B3 MAFCA - post-conditions",
    "B4": "B4 MAFCA, computed verdict",
    "B6": "B6 MAFCA, per-passage calls",
}


# ---------------------------------------------------------------------------
# Loading and alignment
# ---------------------------------------------------------------------------

def load_conditions(directory: Path) -> Dict[str, Dict[str, dict]]:
    """Load every ``<condition>.jsonl``, keyed by condition then claim id."""
    out: Dict[str, Dict[str, dict]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        usable = {str(r["id"]): r for r in records if r.get("verdict")}
        if usable:
            out[path.stem] = usable
    return out


def common_claims(conditions: Dict[str, Dict[str, dict]]) -> List[str]:
    """Claim ids every condition completed, so comparisons stay paired.

    Scoring each condition on its own completed subset would let a
    condition that failed on hard claims look better than one that did not.
    """
    if not conditions:
        return []
    shared = set.intersection(*(set(c) for c in conditions.values()))
    return sorted(shared)


class Rows(list):
    """Aligned per-claim rows: ``(gold, verdict, confidence, correct)``."""


def rows_for(records: Dict[str, dict], claim_ids: Sequence[str]) -> Rows:
    rows = Rows()
    for claim_id in claim_ids:
        record = records[claim_id]
        gold = record.get("gold", "")
        verdict = record.get("verdict", "")
        rows.append(
            {
                "id": claim_id,
                "gold": gold,
                "verdict": verdict,
                "confidence": float(record.get("confidence", 0.0)),
                "correct": gold == verdict,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Metrics as functions of rows, so the bootstrap can resample them
# ---------------------------------------------------------------------------

def accuracy(rows: Sequence[dict]) -> float:
    return sum(1 for r in rows if r["correct"]) / len(rows) if rows else 0.0


def macro_f1(rows: Sequence[dict]) -> float:
    if not rows:
        return 0.0
    scores = per_label_scores([r["gold"] for r in rows], [r["verdict"] for r in rows])
    return statistics.mean(s.f1 for s in scores)


def decidable_accuracy(rows: Sequence[dict]) -> float:
    subset = [r for r in rows if r["gold"] in DECIDABLE]
    return accuracy(subset)


def abstention_rate(rows: Sequence[dict]) -> float:
    """Share of claims answered NOT_ENOUGH_INFO, whatever the gold label."""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["verdict"] == "NOT_ENOUGH_INFO") / len(rows)


def accuracy_when_committing(rows: Sequence[dict]) -> float:
    """Accuracy restricted to claims the system did not abstain on.

    The counterpart to abstention: a system can raise overall accuracy by
    declining more often, and this is the number that says whether the
    answers it does give are any good.
    """
    committed = [r for r in rows if r["verdict"] != "NOT_ENOUGH_INFO"]
    return accuracy(committed)


def ece(rows: Sequence[dict], bins: int = 10) -> float:
    return expected_calibration_error(
        [r["correct"] for r in rows], [r["confidence"] for r in rows], bins
    )


def brier(rows: Sequence[dict]) -> float:
    return brier_score([r["correct"] for r in rows], [r["confidence"] for r in rows])


METRICS: Dict[str, Callable[[Sequence[dict]], float]] = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "decidable_acc": decidable_accuracy,
    "abstention": abstention_rate,
    "acc_committing": accuracy_when_committing,
    "ece10": lambda rows: ece(rows, 10),
    "ece15": lambda rows: ece(rows, 15),
    "brier": brier,
}

# Metrics where a larger value is better, for reading the difference tables.
HIGHER_IS_BETTER = {"accuracy", "macro_f1", "decidable_acc", "acc_committing"}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(
    rows_a: Sequence[dict],
    rows_b: Sequence[dict],
    name: str,
    reference: str,
    metric: str,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20240101,
) -> PairedTest:
    """One paired comparison of ``name`` against ``reference`` on a metric.

    Accuracy uses exact McNemar, which is the right test for a paired
    binary outcome. Every other metric uses the paired bootstrap, since no
    closed-form paired test applies to macro-F1 or ECE.
    """
    function = METRICS[metric]
    paired = list(zip(rows_a, rows_b))

    if metric == "accuracy":
        correct_a = [a["correct"] for a, _ in paired]
        correct_b = [b["correct"] for _, b in paired]
        p_value, a_only, b_only = mcnemar(correct_a, correct_b)
        test = "exact McNemar"
        discordant = a_only + b_only
    else:
        p_value, test, discordant = float("nan"), "paired bootstrap", 0

    difference = paired_bootstrap_difference(
        paired,
        lambda sample: function([a for a, _ in sample]),
        lambda sample: function([b for _, b in sample]),
        resamples=resamples,
        seed=seed,
    )

    if metric != "accuracy":
        # Bootstrap p-value: the share of replicates on the far side of
        # zero, doubled for a two-sided test. Floored at 1/resamples,
        # because zero replicates past the null means "smaller than this
        # bootstrap can resolve", not "impossible".
        p_value = _bootstrap_p(difference, resamples)

    return PairedTest(
        name=name,
        reference=reference,
        n=len(paired),
        estimate_a=function([a for a, _ in paired]),
        estimate_b=function([b for _, b in paired]),
        difference=difference,
        p_value=p_value,
        test=test,
        n_discordant=discordant,
    )


def _bootstrap_p(difference: Interval, resamples: int) -> float:
    """Two-sided p-value implied by a percentile interval.

    Approximated from the interval's position rather than recomputed, so
    the expensive resampling is done once. A difference whose interval
    excludes zero gets a p-value below the complement of its level.
    """
    if difference.resamples == 0:
        return 1.0
    if difference.low > 0 or difference.high < 0:
        # The interval clears zero at the 95% level; report the strongest
        # statement this resolution supports rather than a spurious digit.
        return max(1.0 / resamples, 1.0 - difference.level)
    return 1.0


def comparison_family(
    conditions: Dict[str, Rows], reference: str, metric: str, **kwargs
) -> List[PairedTest]:
    """Compare every condition against one reference, Holm-corrected.

    Correction is applied within the family sharing a reference, which is
    the family in which the multiple-comparison problem actually arises.
    """
    tests = [
        compare(rows, conditions[reference], name, reference, metric, **kwargs)
        for name, rows in conditions.items()
        if name != reference
    ]
    return apply_holm(tests)


# ---------------------------------------------------------------------------
# Diagnostics that need the raw records
# ---------------------------------------------------------------------------

def _evidence_bearing(
    raw: Dict[str, Dict[str, dict]]
) -> Optional[Dict[str, dict]]:
    """Pick a condition whose records describe the retrieved evidence.

    B0 and the trivial baselines consume no evidence and record none, so a
    diagnostic about retrieval must not be computed from them. The evidence
    is identical across the conditions that do use it -- the fingerprint
    assertion guarantees that -- so any one of them will do.
    """
    for name in ("MAFCA", "B4", "B3", "B6", "B1", "B2"):
        if name in raw:
            return raw[name]
    return None


def risk_coverage(records: Dict[str, dict], claim_ids: Sequence[str]) -> List[dict]:
    """Accuracy as a function of coverage, sweeping a confidence threshold.

    Answers whether a condition's confidence is usable for triage: if
    accepting only the most confident half of the verdicts does not raise
    accuracy, the confidence carries no ranking information and abstention
    is just a change in verdict mix.
    """
    scored = sorted(
        (
            {
                "confidence": float(records[c].get("confidence", 0.0)),
                "correct": records[c].get("verdict") == records[c].get("gold"),
            }
            for c in claim_ids
        ),
        key=lambda r: -r["confidence"],
    )
    points = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        k = max(1, int(round(fraction * len(scored))))
        head = scored[:k]
        points.append(
            {
                "coverage": k / len(scored) if scored else 0.0,
                "n": k,
                "accuracy": sum(1 for r in head if r["correct"]) / k,
                "threshold": head[-1]["confidence"] if head else 0.0,
            }
        )
    return points


def nei_analysis(records: Dict[str, dict], claim_ids: Sequence[str]) -> dict:
    """NEI recall split by whether retrieval returned anything (issue 6).

    The concern is that a large NEI gain is an artifact of retrieval
    failure rather than of better verification. Splitting recall by the
    empty/non-empty partition is what distinguishes the two.
    """
    gold_nei = [c for c in claim_ids if records[c].get("gold") == "NOT_ENOUGH_INFO"]
    empty = [c for c in gold_nei if int(records[c].get("n_evidence", 0)) == 0]
    full = [c for c in gold_nei if int(records[c].get("n_evidence", 0)) > 0]

    def recall(subset: Sequence[str]) -> float:
        if not subset:
            return float("nan")
        hits = sum(
            1 for c in subset if records[c].get("verdict") == "NOT_ENOUGH_INFO"
        )
        return hits / len(subset)

    retrieved = [c for c in claim_ids if int(records[c].get("n_evidence", 0)) > 0]
    acc_retrieved = (
        sum(1 for c in retrieved if records[c].get("verdict") == records[c].get("gold"))
        / len(retrieved)
        if retrieved
        else float("nan")
    )

    return {
        "n_gold_nei": len(gold_nei),
        "n_gold_nei_empty": len(empty),
        "n_gold_nei_nonempty": len(full),
        "nei_recall_all": recall(gold_nei),
        "nei_recall_empty": recall(empty),
        "nei_recall_nonempty": recall(full),
        "n_with_evidence": len(retrieved),
        "accuracy_with_evidence": acc_retrieved,
    }


def cost_summary(records: Dict[str, dict], claim_ids: Sequence[str]) -> dict:
    """Latency and model-call counts for the cost table."""
    seconds = [float(records[c].get("seconds", 0.0)) for c in claim_ids]
    calls = [int(records[c].get("llm_calls", 0)) for c in claim_ids]
    ordered = sorted(seconds)
    return {
        "mean_s": statistics.mean(seconds) if seconds else 0.0,
        "median_s": statistics.median(seconds) if seconds else 0.0,
        "p95_s": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else 0.0,
        "mean_calls": statistics.mean(calls) if calls else 0.0,
        "total_calls": sum(calls),
    }


def span_summary(records: Dict[str, dict], claim_ids: Sequence[str]) -> Optional[dict]:
    """Aggregate the Eq. (13) diagnostic, where the condition produced one."""
    reports = [records[c].get("span_check") for c in claim_ids]
    reports = [r for r in reports if r]
    return aggregate_span_reports(reports) if reports else None


def retrieval_summary(records: Dict[str, dict], claim_ids: Sequence[str]) -> Optional[dict]:
    """Recall@k / Precision@k against gold URLs, where both are present."""
    pairs = [
        (records[c].get("retrieved_urls"), records[c].get("gold_urls"))
        for c in claim_ids
    ]
    pairs = [(r, g) for r, g in pairs if r is not None and g]
    if not pairs:
        return None
    retrieved = [r for r, _ in pairs]
    gold = [g for _, g in pairs]
    return {
        "exact": score_retrieval(retrieved, gold, domain=False).model_dump(),
        "domain": score_retrieval(retrieved, gold, domain=True).model_dump(),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt_interval(interval: Interval, places: int = 1, scale: float = 100.0) -> str:
    return (
        f"{interval.estimate * scale:>5.{places}f} "
        f"[{interval.low * scale:>5.{places}f},{interval.high * scale:>6.{places}f}]"
    )


def print_headline(
    conditions: Dict[str, Rows], claim_ids: Sequence[str], resamples: int, seed: int
) -> None:
    print("\n" + "=" * 92)
    print(f"  CLASSIFICATION  ·  n = {len(claim_ids)} claims  ·  "
          f"{resamples:,}-resample bootstrap 95% CI")
    print("=" * 92)
    header = (
        f"  {'condition':<30}{'accuracy':>21}{'macro-F1':>21}{'decidable acc':>21}"
    )
    print(header)
    print("  " + "-" * 88)
    for name in ORDER:
        if name not in conditions:
            continue
        rows = conditions[name]
        cells = [
            _fmt_interval(bootstrap_metric(rows, METRICS[m], resamples=resamples, seed=seed))
            for m in ("accuracy", "macro_f1", "decidable_acc")
        ]
        print(f"  {LABELS.get(name, name):<30}" + "".join(f"{c:>21}" for c in cells))


def print_differences(
    conditions: Dict[str, Rows],
    reference: str,
    metric: str,
    title: str,
    resamples: int,
    seed: int,
) -> None:
    if reference not in conditions:
        return
    tests = comparison_family(
        conditions, reference, metric, resamples=resamples, seed=seed
    )
    tests = [t for t in tests if t.name in ORDER]
    tests.sort(key=lambda t: ORDER.index(t.name))

    print(f"\n  {title}")
    print(f"  {'condition':<30}{'difference (pp)':>24}{'p':>10}{'p(Holm)':>10}  verdict")
    print("  " + "-" * 88)
    for test in tests:
        flag = "inconclusive (CI spans 0)" if test.inconclusive else (
            "significant" if test.significant else "CI clears 0, p n.s."
        )
        print(
            f"  {LABELS.get(test.name, test.name):<30}"
            f"{_fmt_interval(test.difference):>24}"
            f"{test.p_value:>10.4f}{test.p_adjusted:>10.4f}  {flag}"
        )


def print_calibration(conditions: Dict[str, Rows], resamples: int, seed: int) -> None:
    print("\n" + "=" * 92)
    print("  CALIBRATION  (lower is better)")
    print("=" * 92)
    print(f"  {'condition':<30}{'ECE@10':>21}{'ECE@15':>21}{'Brier':>21}")
    print("  " + "-" * 88)
    for name in ORDER:
        if name not in conditions:
            continue
        rows = conditions[name]
        cells = [
            _fmt_interval(
                bootstrap_metric(rows, METRICS[m], resamples=resamples, seed=seed),
                places=3, scale=1.0,
            )
            for m in ("ece10", "ece15", "brier")
        ]
        print(f"  {LABELS.get(name, name):<30}" + "".join(f"{c:>21}" for c in cells))


def print_abstention(
    conditions: Dict[str, Rows],
    raw: Dict[str, Dict[str, dict]],
    claim_ids: Sequence[str],
) -> None:
    print("\n" + "=" * 92)
    print("  ABSTENTION AND EVIDENCE  (issue 6)")
    print("=" * 92)
    print(f"  {'condition':<30}{'abstains':>10}{'acc|commit':>12}"
          f"{'NEI rec':>9}{'NEI rec':>9}{'NEI rec':>9}{'acc w/':>9}")
    print(f"  {'':<30}{'(%)':>10}{'(%)':>12}{'all':>9}{'empty':>9}"
          f"{'non-emp':>9}{'evid':>9}")
    print("  " + "-" * 88)
    for name in ORDER:
        if name not in conditions:
            continue
        rows = conditions[name]
        nei = nei_analysis(raw[name], claim_ids)

        def pct(value: float) -> str:
            return "   n/a" if value != value else f"{100 * value:6.1f}"

        print(
            f"  {LABELS.get(name, name):<30}"
            f"{100 * abstention_rate(rows):>10.1f}"
            f"{100 * accuracy_when_committing(rows):>12.1f}"
            f"{pct(nei['nei_recall_all']):>9}"
            f"{pct(nei['nei_recall_empty']):>9}"
            f"{pct(nei['nei_recall_nonempty']):>9}"
            f"{pct(nei['accuracy_with_evidence']):>9}"
        )
    # Count from a condition that actually consumed evidence. B0 and the
    # trivial baselines record n_evidence = 0 by construction, so reading
    # one of them would report total retrieval failure on every run.
    source = _evidence_bearing(raw)
    if source is None:
        return
    empties = sum(1 for c in claim_ids if int(source[c].get("n_evidence", 0)) == 0)
    print(f"\n  claims with empty evidence: {empties} of {len(claim_ids)}")


def print_risk_coverage(raw: Dict[str, Dict[str, dict]], claim_ids: Sequence[str]) -> None:
    print("\n" + "=" * 92)
    print("  RISK-COVERAGE  ·  accuracy when accepting the most confident X%")
    print("=" * 92)
    fractions = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    print(f"  {'condition':<30}" + "".join(f"{int(100 * f):>9}%" for f in fractions))
    print("  " + "-" * 88)
    for name in ORDER:
        if name not in raw:
            continue
        points = risk_coverage(raw[name], claim_ids)
        print(
            f"  {LABELS.get(name, name):<30}"
            + "".join(f"{100 * p['accuracy']:>9.1f}" for p in points)
        )
    print("\n  A flat row means confidence carries no ranking information:")
    print("  accepting only the most confident verdicts does not improve accuracy.")


def print_citations_and_spans(
    raw: Dict[str, Dict[str, dict]], claim_ids: Sequence[str]
) -> None:
    print("\n" + "=" * 92)
    print("  CITATIONS AND THE EQ. (13) SPAN DIAGNOSTIC")
    print("=" * 92)
    print(f"  {'condition':<30}{'cited':>9}{'validity':>10}{'support':>9}"
          f"{'quoted':>9}{'span viol':>11}")
    print(f"  {'':<30}{'(%)':>9}{'(%)':>10}{'(%)':>9}{'(%)':>9}{'(%)':>11}")
    print("  " + "-" * 88)
    for name in ORDER:
        if name not in raw:
            continue
        records = [raw[name][c] for c in claim_ids]
        citations = score_citations([{"multi": r} for r in records])
        spans = span_summary(raw[name], claim_ids)
        quoted = f"{100 * spans['quote_rate']:>9.1f}" if spans else "      n/a"
        violated = f"{100 * spans['violation_rate']:>11.1f}" if spans else "        n/a"
        print(
            f"  {LABELS.get(name, name):<30}"
            f"{100 * citations.grounded_rate:>9.1f}"
            f"{100 * citations.validity:>10.1f}"
            f"{100 * citations.support_precision:>9.1f}"
            f"{quoted}{violated}"
        )
    print("\n  Validity is guaranteed by construction wherever the citation")
    print("  post-condition is on, so it is not an empirical finding; support")
    print("  precision and the span violation rate are.")


def print_cost(raw: Dict[str, Dict[str, dict]], claim_ids: Sequence[str]) -> None:
    print("\n" + "=" * 92)
    print("  COST  (retrieval is cached, so calls count inference only)")
    print("=" * 92)
    print(f"  {'condition':<30}{'calls':>8}{'mean s':>10}{'median s':>10}{'p95 s':>10}")
    print("  " + "-" * 88)
    for name in ORDER:
        if name not in raw:
            continue
        cost = cost_summary(raw[name], claim_ids)
        print(
            f"  {LABELS.get(name, name):<30}{cost['mean_calls']:>8.1f}"
            f"{cost['mean_s']:>10.1f}{cost['median_s']:>10.1f}{cost['p95_s']:>10.1f}"
        )
    print(
        "\n  Model calls per claim is the portable cost measure. The seconds\n"
        "  columns are wall-clock under whatever concurrency the run used, and\n"
        "  a single local GPU serialises requests: the same call measured 6.3s\n"
        "  alone and 21.2s with three workers in flight. Compare latency across\n"
        "  conditions only when they ran at the same concurrency, and never\n"
        "  read it as a per-call service time."
    )


def print_power(conditions: Dict[str, Rows], claim_ids: Sequence[str]) -> None:
    print("\n" + "=" * 92)
    print("  POWER")
    print("=" * 92)
    if "MAFCA" in conditions and "B1" in conditions:
        discordance = observed_discordance(
            [r["correct"] for r in conditions["MAFCA"]],
            [r["correct"] for r in conditions["B1"]],
        )
    else:
        discordance = 0.35
    power = minimum_detectable_difference(len(claim_ids), discordance=discordance)
    print(
        f"  At n = {power['n']} with an observed discordance of "
        f"{100 * discordance:.1f}% (MAFCA vs B1), the smallest paired accuracy\n"
        f"  difference detectable at alpha=0.05 with 80% power is "
        f"{100 * power['mdd']:.1f} percentage points."
    )
    print(
        f"  Informative (discordant) pairs: {power['informative_pairs']:.0f}. "
        "A difference smaller than\n  that threshold cannot be distinguished "
        "from chance by this sample, whatever the point estimate."
    )


def print_sample(raw: Dict[str, Dict[str, dict]], claim_ids: Sequence[str]) -> None:
    first = next(iter(raw.values()))
    labels: Dict[str, int] = {}
    splits: Dict[str, int] = {}
    for claim_id in claim_ids:
        record = first[claim_id]
        labels[record.get("gold", "?")] = labels.get(record.get("gold", "?"), 0) + 1
        key = record.get("split") or "(unrecorded)"
        splits[key] = splits.get(key, 0) + 1
    print(f"\n  Evaluated sample: {len(claim_ids)} claims")
    print(f"    class counts  {dict(sorted(labels.items()))}")
    print(f"    split origin  {dict(sorted(splits.items()))}")
    majority = max(labels.values()) / len(claim_ids) if claim_ids else 0.0
    print(f"    majority-class accuracy on this sample: {100 * majority:.1f}%")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True,
                        help="Directory of <condition>.jsonl files.")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument(
        "--reweight", action="store_true",
        help="Also estimate metrics under AVeriTeC dev's natural label prior.")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Also write the computed numbers as JSON.")
    args = parser.parse_args(argv)

    raw = load_conditions(args.dir)
    if not raw:
        print(f"No condition files in {args.dir}", file=sys.stderr)
        return 1

    claim_ids = common_claims(raw)
    if not claim_ids:
        print("No claim is present in every condition; cannot pair.", file=sys.stderr)
        return 1

    dropped = {name: len(recs) - len(claim_ids) for name, recs in raw.items()}
    conditions = {name: rows_for(recs, claim_ids) for name, recs in raw.items()}

    print(f"Conditions: {', '.join(sorted(raw))}")
    if any(dropped.values()):
        print(f"  restricted to claims all conditions completed; dropped {dropped}")
    print_sample(raw, claim_ids)

    print_headline(conditions, claim_ids, args.resamples, args.seed)

    print("\n" + "=" * 92)
    print("  PAIRED DIFFERENCES")
    print("=" * 92)
    print("\n  MAFCA vs B1 is the controlled comparison: identical cached evidence,")
    print("  identical output schema, identical model. Only the decomposition differs.")
    print("  MAFCA vs B0 varies retrieval AND decomposition and cannot attribute either.")

    for reference, title in (
        ("B1", "Against B1 (single-call RAG) - accuracy"),
        ("B0", "Against B0 (no retrieval) - accuracy"),
    ):
        print_differences(conditions, reference, "accuracy", title,
                          args.resamples, args.seed)
    print_differences(conditions, "B1", "macro_f1",
                      "Against B1 (single-call RAG) - macro-F1",
                      args.resamples, args.seed)

    print_calibration(conditions, args.resamples, args.seed)
    print_abstention(conditions, raw, claim_ids)
    print_risk_coverage(raw, claim_ids)
    print_citations_and_spans(raw, claim_ids)
    print_cost(raw, claim_ids)
    print_power(conditions, claim_ids)

    if args.reweight:
        print_reweighted(conditions, ORDER, LABELS, AVERITEC_DEV_PRIOR)

    source = _evidence_bearing(raw)
    retrieval = retrieval_summary(source, claim_ids) if source else None
    if retrieval:
        print("\n" + "=" * 92)
        print("  RETRIEVAL vs GOLD EVIDENCE URLs  (shared by all conditions)")
        print("=" * 92)
        for granularity in ("exact", "domain"):
            score = retrieval[granularity]
            print(
                f"  {granularity:<8} Recall@k {100 * score['recall_at_k']:5.1f}%   "
                f"Precision@k {100 * score['precision_at_k']:5.1f}%   "
                f"hit rate {100 * score['hit_rate']:5.1f}%   "
                f"(n={score['n_claims']})"
            )

    if args.json_out:
        payload = {
            "n": len(claim_ids),
            "conditions": sorted(raw),
            "metrics": {
                name: {
                    m: bootstrap_metric(
                        rows, METRICS[m], resamples=args.resamples, seed=args.seed
                    ).model_dump()
                    for m in METRICS
                }
                for name, rows in conditions.items()
            },
            "retrieval": retrieval,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nJSON -> {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
