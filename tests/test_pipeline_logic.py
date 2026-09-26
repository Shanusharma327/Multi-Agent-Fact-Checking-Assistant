"""Offline tests for the deterministic glue around the LLM calls.

None of these hit the network or the Anthropic API: they cover the validation,
reconciliation, and de-duplication logic that keeps a sloppy model response
from corrupting the final verdict.
"""

from __future__ import annotations

import pytest

from fact_checker.agents.adjudicator import _sanitise_citations
from fact_checker.agents.auditor import _reconcile
from fact_checker.schemas import (
    Adjudication,
    Alignment,
    AuditReport,
    EvidenceItem,
    ScoutReport,
    SearchQueries,
)


def _scout(count: int) -> ScoutReport:
    return ScoutReport(
        queries=["q1", "q2"],
        evidence=[
            EvidenceItem(source_id=f"doc_{i}", snippet=f"snippet {i}")
            for i in range(1, count + 1)
        ],
    )


class TestSearchQueries:
    def test_strips_blanks_and_duplicates(self) -> None:
        plan = SearchQueries(queries=["  Eiffel height ", "EIFFEL HEIGHT", "", "burj"])
        assert plan.queries == ["Eiffel height", "burj"]

    def test_requires_at_least_one_query(self) -> None:
        with pytest.raises(Exception):
            SearchQueries(queries=[])


class TestConfidenceClamping:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(0.75, 0.75), (1.4, 1.0), (85, 0.85), (140, 1.0), (-0.3, 0.0), (0.0, 0.0)],
    )
    def test_out_of_range_values_are_coerced(self, raw: float, expected: float) -> None:
        verdict = Adjudication(
            claim="c",
            final_verdict="SUPPORTS",
            confidence=raw,
            detailed_reasoning="r",
        )
        assert verdict.confidence == pytest.approx(expected)


class TestAuditReconciliation:
    def test_missing_snippets_are_backfilled_as_not_enough_info(self) -> None:
        report = AuditReport(
            alignments=[
                Alignment(source_id="doc_1", verdict="SUPPORTS", justification="j")
            ]
        )
        reconciled = _reconcile(report, ["doc_1", "doc_2", "doc_3"])
        assert [a.source_id for a in reconciled.alignments] == ["doc_1", "doc_2", "doc_3"]
        assert reconciled.alignments[1].verdict == "NOT_ENOUGH_INFO"

    def test_hallucinated_ids_are_dropped(self) -> None:
        report = AuditReport(
            alignments=[
                Alignment(source_id="doc_9", verdict="REFUTES", justification="j"),
                Alignment(source_id="doc_1", verdict="SUPPORTS", justification="j"),
            ]
        )
        reconciled = _reconcile(report, ["doc_1"])
        assert len(reconciled.alignments) == 1
        assert reconciled.alignments[0].source_id == "doc_1"

    def test_duplicate_judgements_keep_the_first(self) -> None:
        report = AuditReport(
            alignments=[
                Alignment(source_id="doc_1", verdict="SUPPORTS", justification="first"),
                Alignment(source_id="doc_1", verdict="REFUTES", justification="second"),
            ]
        )
        reconciled = _reconcile(report, ["doc_1"])
        assert reconciled.alignments[0].justification == "first"

    def test_tally_reports_all_labels(self) -> None:
        report = AuditReport(
            alignments=[
                Alignment(source_id="doc_1", verdict="SUPPORTS", justification="j"),
                Alignment(source_id="doc_2", verdict="SUPPORTS", justification="j"),
            ]
        )
        assert report.tally() == {"SUPPORTS": 2, "REFUTES": 0, "NOT_ENOUGH_INFO": 0}


class TestCitationSanitising:
    def test_unknown_and_duplicate_citations_are_removed(self) -> None:
        verdict = Adjudication(
            claim="c",
            final_verdict="SUPPORTS",
            confidence=0.8,
            detailed_reasoning="r",
            cited_sources=["doc_1", "doc_7", "doc_1", "doc_2"],
        )
        cleaned = _sanitise_citations(verdict, {"doc_1", "doc_2"})
        assert cleaned.cited_sources == ["doc_1", "doc_2"]

    def test_valid_citations_pass_through_untouched(self) -> None:
        verdict = Adjudication(
            claim="c",
            final_verdict="REFUTES",
            confidence=0.9,
            detailed_reasoning="r",
            cited_sources=["doc_2", "doc_1"],
        )
        cleaned = _sanitise_citations(verdict, {"doc_1", "doc_2"})
        assert cleaned.cited_sources == ["doc_2", "doc_1"]


class TestScoutReport:
    def test_lookup_helpers(self) -> None:
        scout = _scout(3)
        assert scout.source_ids() == ["doc_1", "doc_2", "doc_3"]
        assert scout.by_id()["doc_2"].snippet == "snippet 2"


class TestCitationRecovery:
    """Small models often cite inline but leave cited_sources empty."""

    def test_recovers_ids_mentioned_in_the_reasoning(self) -> None:
        verdict = Adjudication(
            claim="c",
            final_verdict="REFUTES",
            confidence=0.9,
            detailed_reasoning="[doc_3] is the only snippet that refutes the claim.",
            cited_sources=[],
        )
        cleaned = _sanitise_citations(verdict, {"doc_1", "doc_3"})
        assert cleaned.cited_sources == ["doc_3"]

    def test_recovered_ids_are_still_validated(self) -> None:
        """A doc_N in prose that was never retrieved must not be resurrected."""
        verdict = Adjudication(
            claim="c",
            final_verdict="REFUTES",
            confidence=0.9,
            detailed_reasoning="According to doc_9 and doc_42 the claim is false.",
            cited_sources=[],
        )
        cleaned = _sanitise_citations(verdict, {"doc_1"})
        assert cleaned.cited_sources == []

    def test_explicit_citations_win_over_recovery(self) -> None:
        verdict = Adjudication(
            claim="c",
            final_verdict="SUPPORTS",
            confidence=0.9,
            detailed_reasoning="doc_2 also mentions this.",
            cited_sources=["doc_1"],
        )
        cleaned = _sanitise_citations(verdict, {"doc_1", "doc_2"})
        assert cleaned.cited_sources == ["doc_1"]

    def test_recovery_preserves_order_and_dedupes(self) -> None:
        verdict = Adjudication(
            claim="c",
            final_verdict="SUPPORTS",
            confidence=0.9,
            detailed_reasoning="doc_2 says X; doc_1 agrees; doc_2 repeats it.",
            cited_sources=[],
        )
        cleaned = _sanitise_citations(verdict, {"doc_1", "doc_2"})
        assert cleaned.cited_sources == ["doc_2", "doc_1"]

    def test_no_mentions_stays_empty(self) -> None:
        verdict = Adjudication(
            claim="c",
            final_verdict="NOT_ENOUGH_INFO",
            confidence=0.1,
            detailed_reasoning="No evidence was retrieved.",
            cited_sources=[],
        )
        assert _sanitise_citations(verdict, {"doc_1"}).cited_sources == []
