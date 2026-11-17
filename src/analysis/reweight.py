"""Re-express balanced-sample results under a corpus's natural label prior.

The primary sample is label-balanced (133 per class), which is the right
design for measuring per-class behaviour: AVeriTeC's development split
holds only 35 NOT_ENOUGH_INFO claims, and a natural-distribution draw
would leave that class with too few claims to say anything about.

But a balanced sample does not describe deployment. AVeriTeC dev is
roughly 66% REFUTES, and a system's accuracy there depends on how it
behaves on the class that dominates. Rather than throw away data by
subsampling to the natural prior -- which would discard most of the
SUPPORTS and NOT_ENOUGH_INFO claims and widen every interval -- the
balanced results are *reweighted* to the target prior.

**The assumption this rests on, stated plainly.** Reweighting scales each
gold class's confusion row by ``p_target / p_observed``. It is exact if
the system's class-conditional behaviour -- how it responds to a claim
whose true label is c -- does not depend on the prior. That holds here
because each claim is processed independently, with no calibration to a
batch and no knowledge of the sample composition. It would *not* hold for
a system that normalised over a batch or tuned a threshold on the
evaluated distribution.

Reweighted numbers are labelled as estimates wherever they are reported,
and never mixed into a table of measured ones.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Sequence

from pydantic import BaseModel, Field

# AVeriTeC development split, computed by analysis/dataset_stats.py from
# the 462 usable claims (305 REFUTES / 122 SUPPORTS / 35 NOT_ENOUGH_INFO).
AVERITEC_DEV_PRIOR: Dict[str, float] = {
    "SUPPORTS": 122 / 462,
    "REFUTES": 305 / 462,
    "NOT_ENOUGH_INFO": 35 / 462,
}


class ReweightedScore(BaseModel):
    """Metrics re-expressed under a target label prior."""

    prior: Dict[str, float]
    observed_prior: Dict[str, float]
    accuracy: float
    macro_f1: float
    per_label_recall: Dict[str, float] = Field(default_factory=dict)
    per_label_precision: Dict[str, float] = Field(default_factory=dict)
    per_label_f1: Dict[str, float] = Field(default_factory=dict)
    effective_n: float = Field(
        description="Sum of reweighted counts; equals the real n by construction."
    )
    min_class_support: int = Field(
        description="Smallest real class count behind the estimate."
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def observed_prior(rows: Sequence[dict]) -> Dict[str, float]:
    """Empirical label distribution of the evaluated sample."""
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["gold"]] = counts.get(row["gold"], 0) + 1
    total = sum(counts.values())
    return {label: count / total for label, count in counts.items()} if total else {}


def reweight(
    rows: Sequence[dict], prior: Dict[str, float] = None
) -> ReweightedScore:
    """Re-express a sample's metrics under ``prior``.

    Args:
        rows: Per-claim dicts carrying ``gold`` and ``verdict``.
        prior: Target label distribution. Defaults to the AVeriTeC dev
            split's natural distribution.

    Returns:
        A :class:`ReweightedScore`. ``min_class_support`` is carried
        because a class the sample barely covers cannot be reweighted into
        precision: scaling 3 claims up to 66% of the corpus magnifies their
        noise by the same factor.
    """
    prior = prior or AVERITEC_DEV_PRIOR
    rows = list(rows)
    if not rows:
        return ReweightedScore(
            prior=prior, observed_prior={}, accuracy=0.0, macro_f1=0.0,
            effective_n=0.0, min_class_support=0,
        )

    labels = sorted({r["gold"] for r in rows} | {r["verdict"] for r in rows} | set(prior))
    counts: Dict[str, int] = {label: 0 for label in labels}
    for row in rows:
        counts[row["gold"]] = counts.get(row["gold"], 0) + 1

    total = len(rows)
    observed = {label: _safe_div(counts[label], total) for label in labels}

    # One weight per gold class: how much to scale that row of the
    # confusion matrix so the sample looks drawn from the target prior.
    weights = {
        label: _safe_div(prior.get(label, 0.0), observed[label]) if observed[label] else 0.0
        for label in labels
    }

    # Reweighted confusion: cell[gold][pred] = count * weight(gold).
    cell: Dict[str, Dict[str, float]] = {
        g: {p: 0.0 for p in labels} for g in labels
    }
    for row in rows:
        cell[row["gold"]][row["verdict"]] += weights[row["gold"]]

    weighted_total = sum(sum(c.values()) for c in cell.values())
    correct = sum(cell[label][label] for label in labels)

    recall: Dict[str, float] = {}
    precision: Dict[str, float] = {}
    f1: Dict[str, float] = {}
    for label in labels:
        gold_mass = sum(cell[label].values())
        pred_mass = sum(cell[g][label] for g in labels)
        hit = cell[label][label]
        recall[label] = _safe_div(hit, gold_mass)
        precision[label] = _safe_div(hit, pred_mass)
        f1[label] = _safe_div(
            2 * precision[label] * recall[label], precision[label] + recall[label]
        )

    return ReweightedScore(
        prior=prior,
        observed_prior=observed,
        accuracy=_safe_div(correct, weighted_total),
        macro_f1=statistics.mean(f1[label] for label in labels) if labels else 0.0,
        per_label_recall=recall,
        per_label_precision=precision,
        per_label_f1=f1,
        effective_n=weighted_total,
        min_class_support=min(counts.values()) if counts else 0,
    )


def format_reweighted(name: str, score: ReweightedScore) -> str:
    """One console line for a reweighted condition."""
    return (
        f"  {name:<30}{100 * score.accuracy:>10.1f}{100 * score.macro_f1:>10.1f}"
        + "".join(
            f"{100 * score.per_label_recall.get(label, 0.0):>9.1f}"
            for label in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")
        )
    )


def print_reweighted(
    conditions: Dict[str, Sequence[dict]],
    order: Sequence[str],
    labels: Dict[str, str],
    prior: Dict[str, float] = None,
) -> None:
    """Print a reweighted table, labelled as an estimate throughout."""
    prior = prior or AVERITEC_DEV_PRIOR
    print("\n" + "=" * 92)
    print("  REWEIGHTED TO THE NATURAL LABEL DISTRIBUTION  (estimates, not measurements)")
    print("=" * 92)
    pretty = ", ".join(f"{k.split('_')[0][:3]} {100 * v:.0f}%" for k, v in prior.items())
    print(f"  target prior: {pretty}")
    print(f"  {'condition':<30}{'accuracy':>10}{'macro-F1':>10}"
          f"{'R(SUP)':>9}{'R(REF)':>9}{'R(NEI)':>9}")
    print("  " + "-" * 88)

    support = 0
    for name in order:
        if name not in conditions:
            continue
        score = reweight(conditions[name], prior)
        support = score.min_class_support
        print(format_reweighted(labels.get(name, name), score))

    print(
        "\n  These are estimates obtained by scaling each gold class's confusion\n"
        "  row to the target prior, not measurements on a natural-distribution\n"
        "  sample. They are exact only if class-conditional behaviour does not\n"
        f"  depend on the prior, which holds for a per-claim system. The\n"
        f"  smallest class behind them holds {support} real claims, and the\n"
        "  majority class is scaled up the most, so these carry wider real\n"
        "  uncertainty than the balanced numbers they are derived from."
    )


__all__ = [
    "AVERITEC_DEV_PRIOR",
    "ReweightedScore",
    "format_reweighted",
    "observed_prior",
    "print_reweighted",
    "reweight",
]
