"""Deterministic specificity-weighted aggregation of a per-passage audit.

The paper specifies the adjudication policy as Equations (18)-(19): weight
each per-passage judgement by whether it quotes something specific, sum the
weights per label, and take the argmax over the decidable labels. In the
original implementation that policy was conveyed to the verdict agent as a
*prompt instruction*, which has two consequences the paper itself flags:
the weight :math:`\\beta` is an emphasis rather than a parameter, and
compliance cannot be verified from the output.

This module executes the policy in code instead. The aggregation becomes
auditable -- for any claim the weights, the per-label scores and the margin
that produced the verdict can be printed and checked -- and the adjudication
rules stop depending on a 3B model's willingness to follow them. Agent 3
still writes the natural-language reasoning; it no longer decides the label.

**Confidence.** The elicited self-report the agent produces is a known weak
point: it is generated in the same forward pass as the verdict and is not
read from model internals. The replacement here is a Laplace-smoothed
posterior over the weighted label mass,

.. math:: \\kappa = \\frac{W_{\\hat y} + \\alpha}{W_S + W_R + W_N + 3\\alpha}

with a symmetric Dirichlet prior :math:`\\alpha = 1`. Both constants are
fixed a priori -- :math:`\\beta = 1` makes a specific passage count double,
:math:`\\alpha = 1` is the standard Laplace prior -- and neither was tuned
against evaluation labels. The quantity is therefore a declared function of
the audit, not a fitted calibration curve.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

from pydantic import BaseModel, Field

from .schemas import Alignment, AuditReport, VERDICT_VALUES, Verdict

# Equation (18): a justification quoting something specific counts double.
DEFAULT_BETA = 1.0

# Symmetric Dirichlet prior for the confidence posterior.
DEFAULT_ALPHA = 1.0

# Confidence returned when no evidence was retrieved at all. Matches the
# paper's stated floor, and is deliberately below anything the posterior can
# produce from real evidence.
NO_EVIDENCE_CONFIDENCE = 0.1

DECIDABLE: Tuple[str, ...] = ("SUPPORTS", "REFUTES")

# Markers of evidential specificity: a figure, a date, a quantity, or a named
# official record. These are what Equation (18)'s indicator s_j detects.
_SPECIFICITY_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{4}\b"),                                  # a year
    re.compile(r"\b\d+(?:[.,]\d+)*\s*(?:%|percent)\b", re.I),   # a percentage
    re.compile(r"[$£€₹]\s?\d"),                                 # a monetary amount
    re.compile(r"\b\d+(?:[.,]\d+)+\b"),                         # a decimal figure
    re.compile(
        r"\b(?:january|february|march|april|may|june|july|august|september"
        r"|october|november|december)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:official|officially|according to|statistics?|census|report(?:ed)?"
        r"|record(?:s|ed)?|data|survey|study|court|ruling|legislation|act of)\b",
        re.I,
    ),
    re.compile(r"\b\d{1,3}(?:,\d{3})+\b"),                      # a grouped integer
)


def is_specific(justification: str) -> bool:
    """Equation (18)'s indicator: does the justification quote a specific?

    A justification citing a figure, a date, a monetary amount, or an official
    record is treated as more probative than one that gestures at a topic.
    """
    text = justification or ""
    return any(pattern.search(text) for pattern in _SPECIFICITY_PATTERNS)


class AggregationTrace(BaseModel):
    """The arithmetic behind one aggregated verdict, recorded for audit.

    This object is the point of executing the policy in code: it makes the
    decision reconstructible without re-running the model.
    """

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    weights: Dict[str, float] = Field(
        default_factory=dict, description="Per-source weight w_j from Eq. (18)."
    )
    scores: Dict[str, float] = Field(
        default_factory=dict, description="Weighted mass Score(y) per label."
    )
    margin: float = Field(
        default=0.0,
        description="Normalised lead of the winning decidable label over the other.",
    )
    n_specific: int = Field(default=0, description="Justifications quoting a specific.")
    n_passages: int = Field(default=0)
    rule: str = Field(default="", description="Which branch of Eq. (19) fired.")
    replaced_verdict: str = Field(
        default="",
        description=(
            "The label the model proposed before the computed policy "
            "overrode it. Recorded so the two aggregation strategies remain "
            "comparable from a single run."
        ),
    )
    replaced_confidence: float | None = Field(
        default=None, description="The model's own self-reported confidence."
    )

    def explain(self) -> str:
        """One-line human-readable account of how the verdict was reached."""
        parts = ", ".join(
            f"{label}={self.scores.get(label, 0.0):.1f}" for label in VERDICT_VALUES
        )
        return (
            f"{self.verdict} (kappa={self.confidence:.2f}) from weighted mass [{parts}] "
            f"over {self.n_passages} passage(s), {self.n_specific} quoting specifics; "
            f"rule: {self.rule}"
        )


def weigh(
    alignments: Sequence[Alignment], *, beta: float = DEFAULT_BETA
) -> Dict[str, float]:
    """Equation (18): per-source weight ``w_j = 1 + beta * s_j``."""
    return {
        alignment.source_id: 1.0 + beta * (1.0 if is_specific(alignment.justification) else 0.0)
        for alignment in alignments
    }


def label_scores(
    alignments: Sequence[Alignment], weights: Dict[str, float]
) -> Dict[str, float]:
    """Equation (18): weighted mass ``Score(y)`` for every label."""
    scores = {label: 0.0 for label in VERDICT_VALUES}
    for alignment in alignments:
        scores[alignment.verdict] += weights.get(alignment.source_id, 1.0)
    return scores


def aggregate(
    audit: AuditReport,
    *,
    beta: float = DEFAULT_BETA,
    alpha: float = DEFAULT_ALPHA,
) -> AggregationTrace:
    """Execute Equations (18)-(19) over a reconciled audit.

    Args:
        audit: Agent 2's reconciled per-passage judgements.
        beta: Specificity bonus. ``beta = 0`` reduces to unweighted voting,
            which is the natural ablation of this component.
        alpha: Dirichlet prior strength for the confidence posterior.

    Returns:
        An :class:`AggregationTrace` carrying the verdict, a computed
        confidence, and the arithmetic that produced both.
    """
    alignments = list(audit.alignments)
    if not alignments:
        return AggregationTrace(
            verdict="NOT_ENOUGH_INFO",
            confidence=NO_EVIDENCE_CONFIDENCE,
            weights={},
            scores={label: 0.0 for label in VERDICT_VALUES},
            margin=0.0,
            n_specific=0,
            n_passages=0,
            rule="no evidence retrieved",
        )

    weights = weigh(alignments, beta=beta)
    scores = label_scores(alignments, weights)

    supports = scores["SUPPORTS"]
    refutes = scores["REFUTES"]
    decidable_mass = supports + refutes

    if decidable_mass <= 0.0:
        # Equation (19), second case. Absence of evidence is not refutation:
        # an audit that establishes nothing cannot yield a substantive verdict.
        verdict: Verdict = "NOT_ENOUGH_INFO"
        rule = "no decidable mass (Eq. 19b): absence is not refutation"
        margin = 0.0
    else:
        verdict = "SUPPORTS" if supports > refutes else "REFUTES"
        if supports == refutes:
            # A genuine tie after specificity weighting means the evidence
            # conflicts and nothing breaks it. Declining is the honest move;
            # picking a side would manufacture a decision the audit does not
            # contain.
            verdict = "NOT_ENOUGH_INFO"
            rule = "tie on weighted mass: unresolved conflict"
            margin = 0.0
        else:
            rule = "argmax over decidable mass (Eq. 19a)"
            margin = abs(supports - refutes) / decidable_mass

    total_mass = sum(scores.values())
    confidence = (scores[verdict] + alpha) / (total_mass + 3.0 * alpha)

    return AggregationTrace(
        verdict=verdict,
        confidence=min(1.0, max(0.0, confidence)),
        weights=weights,
        scores=scores,
        margin=margin,
        n_specific=sum(1 for a in alignments if is_specific(a.justification)),
        n_passages=len(alignments),
        rule=rule,
    )


def consistent_citations(
    cited: Iterable[str],
    audit: AuditReport,
    verdict: str,
    valid_ids: Iterable[str],
) -> List[str]:
    """Keep only citations the audit supports in the verdict's direction.

    Validating that a citation names a *retrieved* passage is the guarantee
    the pipeline already had; it establishes that the reference is real, not
    that it bears on the verdict. This adds the stronger condition: for a
    decidable verdict, a cited passage must be one the verification stage
    labelled in that same direction.

    A NOT_ENOUGH_INFO verdict has no direction, so for it the rule would be
    vacuous -- and a citation attached to "the evidence settles nothing" is
    asserting the opposite of the verdict. Such citations are dropped.

    Args:
        cited: Identifiers the model emitted.
        audit: The reconciled per-passage audit.
        verdict: The final verdict.
        valid_ids: Identifiers of genuinely retrieved passages.

    Returns:
        The retained identifiers, de-duplicated and in the original order.
    """
    valid = set(valid_ids)
    by_id = {a.source_id: a.verdict for a in audit.alignments}
    if verdict not in DECIDABLE:
        return []
    return list(
        dict.fromkeys(
            source_id
            for source_id in cited
            if source_id in valid and by_id.get(source_id) == verdict
        )
    )


def supporting_ids(audit: AuditReport, verdict: str) -> List[str]:
    """Every audited passage whose label matches ``verdict``.

    Used as the citation set when the model emitted none but the audit
    contains passages that genuinely decided the verdict. Under deterministic
    aggregation the verdict *is* a function of those passages, so naming them
    is a statement of what drove the decision rather than a guess at it.
    """
    if verdict not in DECIDABLE:
        return []
    return [a.source_id for a in audit.alignments if a.verdict == verdict]


__all__ = [
    "AggregationTrace",
    "DECIDABLE",
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "NO_EVIDENCE_CONFIDENCE",
    "aggregate",
    "consistent_citations",
    "is_specific",
    "label_scores",
    "supporting_ids",
    "weigh",
]
