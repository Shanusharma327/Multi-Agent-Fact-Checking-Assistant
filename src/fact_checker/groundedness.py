"""Equation (13): are the justification's quoted tokens really in the passage?

The paper presents Eq. (13) --- ``Quoted(j_j) ⊆ Tokens(e_j)`` --- as the
enforcement mechanism for the closed-world restriction of Eq. (12): a
justification that cannot ground itself in its own passage is observable
evidence that the model reached outside the passage. Nothing in the original
implementation checked it.

This module checks it, and the honest description of what it provides is a
**diagnostic, not an enforcement mechanism**. The distinction matters:

* It cannot be enforced by rejection. A violation means the justification is
  ungrounded, not that the *label* is wrong, and discarding the label would
  throw away a judgement that may well be correct.
* It is a lower bound on groundedness. A model can write a perfectly
  grounded-looking justification by quoting the passage while having decided
  the label from parametric memory. Passing the check is necessary for
  groundedness, not sufficient.
* It has a false-positive floor. A model that paraphrases rather than quotes
  is penalised even when faithful, which is why the extraction below only
  treats explicitly delimited spans as quotations.

What it does establish is a rate: how often the stated justification contains
a span that demonstrably does not appear in the passage it claims to quote.
That rate is reportable, comparable across conditions, and was previously
unmeasured.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Sequence

from pydantic import BaseModel, Field

from .schemas import Alignment, AuditReport, ScoutReport

# Spans the model marked as quotation. Straight and typographic doubles,
# typographic singles. Bare apostrophes are excluded: "the government's
# figure" is possessive, not a quotation, and treating it as one would
# manufacture violations.
_QUOTE_PATTERNS = (
    re.compile(r'"([^"]{3,200})"'),
    re.compile(r"“([^”]{3,200})”"),
    re.compile(r"‘([^’]{3,200})’"),
)

# A numeric or date-like token, checked even when unquoted. These are the
# spans whose fabrication matters most for verification -- a justification
# citing "1971" when the passage says no such thing is exactly the failure
# Eq. (12) is meant to make visible.
_FIGURE = re.compile(
    r"\b\d[\d,.]*%?\b"
    r"|\b(?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\b",
    re.IGNORECASE,
)

_PUNCTUATION = re.compile(r"[^\w\s%]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")

# Negation, and the single largest source of false positives in this check.
# A justification reading "the snippet does not provide the 40% drop" names a
# figure in order to report its ABSENCE. Naive containment testing flags that
# "40" as fabricated, inverting what the justification actually said. Manual
# inspection of 30 flagged spans from the 60-claim run found 29 of this kind
# and one genuine violation, so the filter is not a refinement -- without it
# the measurement is almost entirely noise.
_NEGATION = re.compile(
    r"\b(?:does not|doesn't|do not|don't|did not|didn't|is not|isn't|are not|"
    r"aren't|was not|wasn't|were not|weren't|cannot|can't|not|no|none|without|"
    r"nor|never|fails? to|failed to|lacks?|lacking|absent|silent on|unrelated|"
    r"neither|makes no|contains no|provides no|offers no)\b",
    re.IGNORECASE,
)

# Sentence boundaries, so negation is sought within the clause that holds the
# span rather than anywhere in a multi-sentence justification.
_SENTENCE = re.compile(r"(?<=[.;!?])\s+")


def normalise(text: str) -> str:
    """Fold a string to the form used for containment testing.

    Unicode-normalises, lowercases, strips punctuation and collapses
    whitespace, so that a quotation differing from its source only in
    curly quotes, hyphenation or spacing is not counted as fabricated.
    """
    folded = unicodedata.normalize("NFKC", text or "").lower()
    folded = _PUNCTUATION.sub(" ", folded)
    return _WHITESPACE.sub(" ", folded).strip()


def extract_quotes(justification: str) -> List[str]:
    """Return the explicitly delimited spans in ``justification``."""
    spans: List[str] = []
    for pattern in _QUOTE_PATTERNS:
        spans.extend(match.group(1).strip() for match in pattern.finditer(justification or ""))
    return [s for s in spans if s]


def extract_figures(justification: str) -> List[str]:
    """Return numeric and month tokens, quoted or not."""
    return [m.group(0) for m in _FIGURE.finditer(justification or "")]


class SpanCheck(BaseModel):
    """Whether one justification's spans appear in its own passage."""

    source_id: str
    n_quotes: int = 0
    n_quotes_found: int = 0
    n_figures: int = 0
    n_figures_found: int = 0
    missing: List[str] = Field(
        default_factory=list,
        description="Spans absent from the passage and not under negation.",
    )
    negated: List[str] = Field(
        default_factory=list,
        description=(
            "Spans absent from the passage but named inside a negation, so "
            "the justification was reporting their absence rather than "
            "quoting them. Retained for audit, excluded from the rate."
        ),
    )

    @property
    def has_quote(self) -> bool:
        """Whether the justification quoted anything at all."""
        return self.n_quotes > 0

    @property
    def violates(self) -> bool:
        """Whether any checked span is absent from the passage."""
        return bool(self.missing)


class SpanReport(BaseModel):
    """Eq. (13) outcomes aggregated over one claim's audit."""

    checks: List[SpanCheck] = Field(default_factory=list)
    n_alignments: int = 0
    n_checkable: int = Field(
        default=0, description="Alignments carrying at least one checkable span."
    )
    n_violating: int = 0
    n_quoting: int = Field(
        default=0, description="Alignments that quoted anything at all."
    )
    n_negated_spans: int = Field(
        default=0,
        description="Spans excluded as negated mentions rather than quotations.",
    )

    @property
    def violation_rate(self) -> float:
        """Share of checkable alignments with a fabricated span."""
        return self.n_violating / self.n_checkable if self.n_checkable else 0.0

    @property
    def quote_rate(self) -> float:
        """Share of alignments that quoted, as the prompt requires."""
        return self.n_quoting / self.n_alignments if self.n_alignments else 0.0


def is_negated(justification: str, span: str) -> bool:
    """Whether ``span`` sits in a clause asserting the passage lacks it.

    Searched within the sentence containing the span, not the whole
    justification: a two-sentence justification can legitimately negate one
    thing and quote another.
    """
    for sentence in _SENTENCE.split(justification or ""):
        if span in sentence:
            return bool(_NEGATION.search(sentence))
    return bool(_NEGATION.search(justification or ""))


def check_alignment(alignment: Alignment, passage: str) -> SpanCheck:
    """Test one justification's quoted and numeric spans against its passage.

    A span is counted as a violation only when it is absent from the passage
    *and* the clause containing it is not a negation. The second condition
    is what separates "the model attributed a figure to this passage that is
    not in it" -- the failure Eq. (13) exists to detect -- from "the model
    correctly reported that this passage does not mention the figure".
    """
    haystack = normalise(passage)
    quotes = extract_quotes(alignment.justification)
    figures = extract_figures(alignment.justification)

    missing: List[str] = []
    negated: List[str] = []

    found_quotes = 0
    for span in quotes:
        if normalise(span) and normalise(span) in haystack:
            found_quotes += 1
        elif is_negated(alignment.justification, span):
            negated.append(span)
        else:
            missing.append(span)

    found_figures = 0
    for span in figures:
        if normalise(span) in haystack:
            found_figures += 1
        elif is_negated(alignment.justification, span):
            negated.append(span)
        else:
            missing.append(span)

    return SpanCheck(
        source_id=alignment.source_id,
        n_quotes=len(quotes),
        n_quotes_found=found_quotes,
        n_figures=len(figures),
        n_figures_found=found_figures,
        missing=missing,
        negated=negated,
    )


def audit_span_report(audit: AuditReport, scout: ScoutReport) -> SpanReport:
    """Run Eq. (13) over every alignment in one claim's audit.

    Alignments naming a passage that is not present are skipped rather than
    counted as violations: a missing passage is an identifier problem, which
    audit reconciliation already handles, and conflating the two would
    inflate the rate reported here.
    """
    passages = {item.source_id: item.snippet for item in scout.evidence}
    checks: List[SpanCheck] = []
    for alignment in audit.alignments:
        passage = passages.get(alignment.source_id)
        if passage is None:
            continue
        checks.append(check_alignment(alignment, passage))

    checkable = [c for c in checks if c.n_quotes or c.n_figures]
    return SpanReport(
        checks=checks,
        n_alignments=len(checks),
        n_checkable=len(checkable),
        n_violating=sum(1 for c in checkable if c.violates),
        n_quoting=sum(1 for c in checks if c.has_quote),
        n_negated_spans=sum(len(c.negated) for c in checks),
    )


def aggregate_span_reports(reports: Sequence[dict]) -> dict:
    """Pool per-claim span reports into run-level rates.

    Args:
        reports: ``SpanReport`` dictionaries as stored in result records.

    Returns:
        Counts and rates across the run, plus a sample of fabricated spans
        for qualitative inspection.
    """
    alignments = checkable = violating = quoting = negated = 0
    examples: List[str] = []
    for report in reports:
        if not report:
            continue
        alignments += int(report.get("n_alignments", 0))
        checkable += int(report.get("n_checkable", 0))
        violating += int(report.get("n_violating", 0))
        quoting += int(report.get("n_quoting", 0))
        negated += int(report.get("n_negated_spans", 0))
        for check in report.get("checks") or []:
            for span in check.get("missing") or []:
                if len(examples) < 20:
                    examples.append(span)

    return {
        "n_alignments": alignments,
        "n_checkable": checkable,
        "n_violating": violating,
        "n_quoting": quoting,
        "n_negated_spans": negated,
        "violation_rate": violating / checkable if checkable else 0.0,
        "quote_rate": quoting / alignments if alignments else 0.0,
        "example_missing_spans": examples,
    }


__all__ = [
    "SpanCheck",
    "SpanReport",
    "aggregate_span_reports",
    "audit_span_report",
    "check_alignment",
    "extract_figures",
    "extract_quotes",
    "normalise",
]
