"""Scoring for the FEVER benchmark: classification quality and calibration.

Calibration is scored alongside accuracy on purpose. The central claim of this
project is that an ungrounded model is *confidently* wrong, and accuracy alone
cannot show that -- two systems with identical accuracy are not equivalent if
one of them says 95% when it is wrong. Expected Calibration Error and Brier
score are what make that difference measurable.

Implemented with the standard library only: no numpy or scikit-learn.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from pydantic import BaseModel, Field

from .schemas import VERDICT_VALUES


class LabelScore(BaseModel):
    """Precision / recall / F1 for a single label."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int = Field(description="Number of gold instances of this label.")


class CalibrationBin(BaseModel):
    """One confidence bucket used to compute Expected Calibration Error."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


class Scorecard(BaseModel):
    """Everything measured for one system (multi-agent or baseline)."""

    system: str
    n: int
    accuracy: float
    macro_f1: float
    per_label: List[LabelScore] = Field(default_factory=list)
    confusion: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="confusion[gold][predicted] = count.",
    )
    mean_confidence: float = 0.0
    ece: float = Field(default=0.0, description="Expected Calibration Error.")
    brier: float = Field(default=0.0, description="Brier score on correctness.")
    overconfidence: float = Field(
        default=0.0,
        description="mean_confidence - accuracy. Positive means overconfident.",
    )
    mean_confidence_when_wrong: float = 0.0
    mean_confidence_when_right: float = 0.0
    mean_seconds: float = 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 rather than raising on an empty denominator."""
    return numerator / denominator if denominator else 0.0


def confusion_matrix(
    gold: Sequence[str], predicted: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    """Build ``matrix[gold_label][predicted_label] = count``."""
    matrix = {
        g: {p: 0 for p in VERDICT_VALUES} for g in VERDICT_VALUES
    }
    for gold_label, predicted_label in zip(gold, predicted):
        if gold_label in matrix and predicted_label in matrix[gold_label]:
            matrix[gold_label][predicted_label] += 1
    return matrix


def per_label_scores(
    gold: Sequence[str], predicted: Sequence[str]
) -> List[LabelScore]:
    """Compute precision, recall and F1 for every label."""
    scores: List[LabelScore] = []
    for label in VERDICT_VALUES:
        true_positive = sum(
            1 for g, p in zip(gold, predicted) if g == label and p == label
        )
        false_positive = sum(
            1 for g, p in zip(gold, predicted) if g != label and p == label
        )
        false_negative = sum(
            1 for g, p in zip(gold, predicted) if g == label and p != label
        )
        precision = _safe_div(true_positive, true_positive + false_positive)
        recall = _safe_div(true_positive, true_positive + false_negative)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        scores.append(
            LabelScore(
                label=label,
                precision=precision,
                recall=recall,
                f1=f1,
                support=true_positive + false_negative,
            )
        )
    return scores


def calibration_bins(
    correct: Sequence[bool], confidence: Sequence[float], bins: int = 10
) -> List[CalibrationBin]:
    """Bucket predictions by confidence and report accuracy within each bucket.

    A perfectly calibrated system has ``accuracy == mean_confidence`` in every
    bucket.
    """
    buckets: List[List[Tuple[bool, float]]] = [[] for _ in range(bins)]
    for is_correct, conf in zip(correct, confidence):
        clamped = min(1.0, max(0.0, conf))
        # `min` keeps confidence == 1.0 inside the final bucket.
        index = min(bins - 1, int(clamped * bins))
        buckets[index].append((is_correct, clamped))

    result: List[CalibrationBin] = []
    for index, bucket in enumerate(buckets):
        lower = index / bins
        upper = (index + 1) / bins
        if not bucket:
            result.append(
                CalibrationBin(
                    lower=lower, upper=upper, count=0, mean_confidence=0.0, accuracy=0.0
                )
            )
            continue
        result.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=len(bucket),
                mean_confidence=sum(c for _, c in bucket) / len(bucket),
                accuracy=sum(1 for ok, _ in bucket if ok) / len(bucket),
            )
        )
    return result


def expected_calibration_error(
    correct: Sequence[bool], confidence: Sequence[float], bins: int = 10
) -> float:
    """Weighted mean gap between confidence and accuracy across bins."""
    total = len(correct)
    if not total:
        return 0.0
    return sum(
        (b.count / total) * abs(b.accuracy - b.mean_confidence)
        for b in calibration_bins(correct, confidence, bins)
        if b.count
    )


def brier_score(correct: Sequence[bool], confidence: Sequence[float]) -> float:
    """Mean squared error between confidence and correctness. Lower is better."""
    if not correct:
        return 0.0
    return sum(
        (min(1.0, max(0.0, c)) - (1.0 if ok else 0.0)) ** 2
        for ok, c in zip(correct, confidence)
    ) / len(correct)


def score_system(
    system: str,
    gold: Sequence[str],
    predicted: Sequence[str],
    confidence: Sequence[float],
    seconds: Sequence[float] = (),
    *,
    bins: int = 10,
) -> Scorecard:
    """Compute the full scorecard for one system.

    Args:
        system: Display name, e.g. ``"multi-agent"``.
        gold: Ground-truth labels.
        predicted: Predicted labels, aligned with ``gold``.
        confidence: Confidence scores, aligned with ``gold``.
        seconds: Optional per-claim latency.
        bins: Number of calibration buckets.

    Returns:
        A populated :class:`Scorecard`.

    Raises:
        ValueError: If the aligned sequences have differing lengths.
    """
    if not (len(gold) == len(predicted) == len(confidence)):
        raise ValueError(
            "gold, predicted and confidence must be the same length "
            f"(got {len(gold)}, {len(predicted)}, {len(confidence)})"
        )

    total = len(gold)
    correct = [g == p for g, p in zip(gold, predicted)]
    accuracy = _safe_div(sum(correct), total)
    labels = per_label_scores(gold, predicted)
    macro_f1 = _safe_div(sum(s.f1 for s in labels), len(labels))
    mean_conf = _safe_div(sum(confidence), total)

    right = [c for ok, c in zip(correct, confidence) if ok]
    wrong = [c for ok, c in zip(correct, confidence) if not ok]

    return Scorecard(
        system=system,
        n=total,
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_label=labels,
        confusion=confusion_matrix(gold, predicted),
        mean_confidence=mean_conf,
        ece=expected_calibration_error(correct, confidence, bins),
        brier=brier_score(correct, confidence),
        overconfidence=mean_conf - accuracy,
        mean_confidence_when_right=_safe_div(sum(right), len(right)),
        mean_confidence_when_wrong=_safe_div(sum(wrong), len(wrong)),
        mean_seconds=_safe_div(sum(seconds), len(seconds)) if seconds else 0.0,
    )


def score_subset(
    system: str,
    gold: Sequence[str],
    predicted: Sequence[str],
    confidence: Sequence[float],
    keep: Sequence[str],
) -> Scorecard:
    """Score only the claims whose gold label is in ``keep``.

    Used to report the decidable subset (SUPPORTS / REFUTES) separately from
    NOT_ENOUGH_INFO. On FEVER the NEI label is corpus-relative -- it means
    "unverifiable from a 2017 Wikipedia snapshot" -- so folding it into a
    single headline accuracy conflates a system-quality question with a
    benchmark-construction artefact.
    """
    keep_set = set(keep)
    rows = [
        (g, p, c)
        for g, p, c in zip(gold, predicted, confidence)
        if g in keep_set
    ]
    if not rows:
        return Scorecard(system=system, n=0, accuracy=0.0, macro_f1=0.0)
    g, p, c = zip(*rows)
    return score_system(system, list(g), list(p), list(c))


def format_scorecard(card: Scorecard) -> str:
    """Render a scorecard as a fixed-width text block for the console."""
    lines = [
        f"  {card.system}  (n={card.n})",
        f"    accuracy            {card.accuracy:6.1%}",
        f"    macro F1            {card.macro_f1:6.1%}",
        f"    mean confidence     {card.mean_confidence:6.1%}",
        f"    overconfidence      {card.overconfidence:+6.1%}   (confidence - accuracy)",
        f"    ECE                 {card.ece:6.3f}   (lower is better)",
        f"    Brier               {card.brier:6.3f}   (lower is better)",
        f"    conf. when right    {card.mean_confidence_when_right:6.1%}",
        f"    conf. when wrong    {card.mean_confidence_when_wrong:6.1%}",
    ]
    if card.mean_seconds:
        lines.append(f"    mean latency        {card.mean_seconds:6.2f}s")
    lines.append("    per-label:")
    for score in card.per_label:
        lines.append(
            f"      {score.label:<16} P {score.precision:5.1%}  "
            f"R {score.recall:5.1%}  F1 {score.f1:5.1%}  (n={score.support})"
        )
    return "\n".join(lines)


def format_confusion(card: Scorecard) -> str:
    """Render the confusion matrix as a small aligned table."""
    short = {"SUPPORTS": "SUP", "REFUTES": "REF", "NOT_ENOUGH_INFO": "NEI"}
    header = "      gold \\ pred   " + "".join(f"{short[p]:>6}" for p in VERDICT_VALUES)
    rows = [header]
    for gold_label in VERDICT_VALUES:
        cells = "".join(
            f"{card.confusion[gold_label][p]:>6}" for p in VERDICT_VALUES
        )
        rows.append(f"      {short[gold_label]:<12}  {cells}")
    return "\n".join(rows)
