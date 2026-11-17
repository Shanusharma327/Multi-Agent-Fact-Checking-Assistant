"""Pydantic v2 contracts for every agent boundary in the pipeline.

These models serve double duty: they are the JSON schemas handed to the model
via ``with_structured_output`` *and* the runtime validation layer for whatever
comes back. Keeping one definition for both is what makes the multi-agent and
baseline paths directly comparable -- Agent 3 and the baseline emit the exact
same :class:`Adjudication` object, so the UI can diff them field by field.
"""

from __future__ import annotations

from typing import Annotated, List, Literal

from pydantic import BaseModel, Field, field_validator

# The closed label set shared by Agent 2 (per-snippet) and Agent 3 (aggregate).
Verdict = Literal["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

VERDICT_VALUES: tuple[str, ...] = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")


class SearchQueries(BaseModel):
    """Agent 1's *generative* step: the queries, before any searching happens.

    The schema asks for exactly two because the retrieval design depends on
    two independent angles on the claim, not on however many the model felt
    like writing. The lower bound is 1 rather than 2 so that a model emitting
    one usable query still validates; the caller then completes the pair
    rather than silently running a single-perspective retrieval.
    """

    queries: Annotated[List[str], Field(min_length=1, max_length=4)] = Field(
        description=(
            "Exactly two neutral, keyword-style web search queries that "
            "investigate the claim without presupposing its truth."
        )
    )

    @field_validator("queries")
    @classmethod
    def _clean(cls, queries: List[str]) -> List[str]:
        """Strip blanks and case-insensitive duplicates, preserving order."""
        seen: set[str] = set()
        cleaned: List[str] = []
        for query in queries:
            text = query.strip().strip('"')
            if text and text.lower() not in seen:
                seen.add(text.lower())
                cleaned.append(text)
        return cleaned


class EvidenceItem(BaseModel):
    """One retrieved snippet, addressable by a stable ``doc_N`` identifier.

    ``title`` and ``url`` are retrieval metadata (used by the UI to hyperlink a
    citation); the agents only ever reason over ``source_id`` and ``snippet``.
    """

    source_id: str = Field(description="Stable identifier, e.g. 'doc_1'.")
    snippet: str = Field(description="Verbatim evidence text from the search result.")
    title: str = Field(default="", description="Result title, for display only.")
    url: str = Field(default="", description="Result URL, for display only.")
    query: str = Field(default="", description="Query that surfaced this result.")


class ScoutReport(BaseModel):
    """Agent 1 output: the queries it ran and the evidence it brought back."""

    queries: List[str] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)

    def source_ids(self) -> List[str]:
        return [item.source_id for item in self.evidence]

    def by_id(self) -> dict[str, EvidenceItem]:
        return {item.source_id: item for item in self.evidence}


class Alignment(BaseModel):
    """Agent 2's NLI judgement for a single snippet against the claim."""

    source_id: str = Field(description="The doc_N identifier being judged.")
    verdict: Verdict = Field(
        description="SUPPORTS, REFUTES, or NOT_ENOUGH_INFO for this snippet alone."
    )
    justification: str = Field(
        description=(
            "One sentence quoting the exact words or numbers in the snippet that "
            "drove the label."
        )
    )


class AuditReport(BaseModel):
    """Agent 2 output: exactly one :class:`Alignment` per retrieved snippet."""

    alignments: List[Alignment] = Field(default_factory=list)

    def tally(self) -> dict[str, int]:
        """Count alignments per label, including labels with zero hits."""
        counts = {value: 0 for value in VERDICT_VALUES}
        for alignment in self.alignments:
            counts[alignment.verdict] += 1
        return counts


class Adjudication(BaseModel):
    """Final verdict schema -- emitted by BOTH Agent 3 and the baseline.

    Identical shape on both paths is deliberate: it is what lets the dashboard
    put a grounded verdict and an ungrounded one side by side on equal terms.
    """

    claim: str = Field(description="The claim under evaluation, restated.")
    final_verdict: Verdict = Field(description="The unified verdict.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Calibrated confidence from 0.0 to 1.0."
    )
    detailed_reasoning: str = Field(
        description="Step-by-step account of how the verdict was reached."
    )
    cited_sources: List[str] = Field(
        default_factory=list,
        description="doc_N identifiers actually used. Empty for the baseline.",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: object) -> object:
        """Clamp instead of rejecting.

        Models occasionally emit 1.2 or a percentage-like 85. A hard validation
        error would throw away an otherwise-good verdict, so squeeze the value
        into range and let the rest of the object through.

        The two failure modes need different treatment: a small overshoot
        (1.0 < v < 5) is a model rounding past the ceiling and clamps to 1.0,
        while a value in [5, 100] is a percentage and is rescaled. Rescaling
        the overshoot instead would silently turn "certain" into 1.4%.
        """
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if numeric > 1.0:
                numeric = numeric / 100.0 if 5.0 <= numeric <= 100.0 else 1.0
            return min(1.0, max(0.0, numeric))
        return value


class StageTiming(BaseModel):
    """Wall-clock cost of one pipeline stage, surfaced in the UI."""

    name: str
    seconds: float


class MultiAgentResult(BaseModel):
    """Everything the three-agent path produced, including diagnostics."""

    claim: str
    scout: ScoutReport
    audit: AuditReport
    adjudication: Adjudication
    timings: List[StageTiming] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    aggregation: object | None = Field(
        default=None,
        description=(
            "The arithmetic behind a deterministically aggregated verdict "
            "(an AggregationTrace), or None when Agent 3 decided the label "
            "itself. Typed loosely to keep schemas.py free of a dependency "
            "on the aggregation module, which imports from here."
        ),
    )

    @property
    def total_seconds(self) -> float:
        return sum(timing.seconds for timing in self.timings)


class BaselineResult(BaseModel):
    """The monolithic single-LLM control run."""

    claim: str
    adjudication: Adjudication
    seconds: float = 0.0
