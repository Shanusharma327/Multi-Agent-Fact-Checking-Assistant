"""Tests for the deterministic specificity-weighted verdict aggregation."""

from __future__ import annotations

import pytest

from fact_checker.aggregation import (
    NO_EVIDENCE_CONFIDENCE,
    aggregate,
    consistent_citations,
    is_specific,
    label_scores,
    supporting_ids,
    weigh,
)
from fact_checker.schemas import Alignment, AuditReport


def align(index: int, verdict: str, justification: str = "no specifics here") -> Alignment:
    return Alignment(
        source_id=f"doc_{index}", verdict=verdict, justification=justification
    )


def report(*alignments: Alignment) -> AuditReport:
    return AuditReport(alignments=list(alignments))


class TestIsSpecific:
    @pytest.mark.parametrize(
        "text",
        [
            "the snippet says 1971, not 1969",
            "turnout was 62.4% of registered voters",
            "the budget was $4.2 million",
            "signed in March of that year",
            "according to official census data",
            "the figure was 1,250,000 people",
        ],
    )
    def test_specific_justifications_are_detected(self, text):
        assert is_specific(text)

    @pytest.mark.parametrize(
        "text",
        [
            "the snippet is about a related topic",
            "it broadly agrees with the claim",
            "",
            "this passage says nothing useful",
        ],
    )
    def test_vague_justifications_are_not(self, text):
        assert not is_specific(text)


class TestWeights:
    def test_a_specific_justification_counts_double_at_beta_one(self):
        audit = report(align(1, "SUPPORTS", "official data from 2019"), align(2, "SUPPORTS"))
        weights = weigh(audit.alignments, beta=1.0)
        assert weights == {"doc_1": 2.0, "doc_2": 1.0}

    def test_beta_zero_reduces_to_unweighted_voting(self):
        audit = report(align(1, "SUPPORTS", "official data from 2019"), align(2, "SUPPORTS"))
        assert set(weigh(audit.alignments, beta=0.0).values()) == {1.0}

    def test_scores_sum_weight_per_label(self):
        audit = report(
            align(1, "SUPPORTS", "the 2019 report"),
            align(2, "REFUTES"),
            align(3, "NOT_ENOUGH_INFO"),
        )
        scores = label_scores(audit.alignments, weigh(audit.alignments))
        assert scores == {"SUPPORTS": 2.0, "REFUTES": 1.0, "NOT_ENOUGH_INFO": 1.0}


class TestAggregate:
    def test_empty_audit_declines_at_the_stated_floor(self):
        trace = aggregate(report())
        assert trace.verdict == "NOT_ENOUGH_INFO"
        assert trace.confidence == NO_EVIDENCE_CONFIDENCE
        assert trace.rule == "no evidence retrieved"

    def test_absence_of_evidence_is_not_refutation(self):
        """Eq. (19b): an all-NEI audit cannot yield a substantive verdict."""
        trace = aggregate(report(align(1, "NOT_ENOUGH_INFO"), align(2, "NOT_ENOUGH_INFO")))
        assert trace.verdict == "NOT_ENOUGH_INFO"
        assert trace.scores["SUPPORTS"] == 0.0
        assert trace.scores["REFUTES"] == 0.0

    def test_unanimous_support_is_carried(self):
        trace = aggregate(report(align(1, "SUPPORTS"), align(2, "SUPPORTS")))
        assert trace.verdict == "SUPPORTS"
        assert trace.margin == 1.0

    def test_specificity_outweighs_an_equal_count(self):
        """One passage quoting a figure beats one that merely gestures."""
        trace = aggregate(
            report(
                align(1, "REFUTES", "the official record says 1971"),
                align(2, "SUPPORTS", "the article broadly agrees"),
            )
        )
        assert trace.verdict == "REFUTES"

    def test_an_unbroken_tie_declines_rather_than_guessing(self):
        trace = aggregate(report(align(1, "SUPPORTS"), align(2, "REFUTES")))
        assert trace.verdict == "NOT_ENOUGH_INFO"
        assert "tie" in trace.rule

    def test_confidence_rises_with_agreeing_evidence(self):
        thin = aggregate(report(align(1, "SUPPORTS")))
        thick = aggregate(report(*[align(i, "SUPPORTS") for i in range(1, 6)]))
        assert thick.confidence > thin.confidence

    def test_confidence_stays_a_probability(self):
        trace = aggregate(report(*[align(i, "SUPPORTS", "2019 data") for i in range(1, 30)]))
        assert 0.0 <= trace.confidence <= 1.0

    def test_a_contested_verdict_is_less_confident_than_a_clean_one(self):
        clean = aggregate(report(align(1, "SUPPORTS"), align(2, "SUPPORTS")))
        contested = aggregate(
            report(align(1, "SUPPORTS"), align(2, "SUPPORTS"), align(3, "REFUTES"))
        )
        assert contested.confidence < clean.confidence

    def test_beta_zero_lets_volume_win(self):
        """The specificity term is what stops counting from deciding alone."""
        audit = report(
            align(1, "REFUTES", "the official record says 1971"),
            align(2, "SUPPORTS"),
            align(3, "SUPPORTS"),
        )
        assert aggregate(audit, beta=0.0).verdict == "SUPPORTS"
        assert aggregate(audit, beta=3.0).verdict == "REFUTES"

    def test_trace_records_the_arithmetic(self):
        trace = aggregate(report(align(1, "SUPPORTS", "2019 figures")))
        assert trace.n_passages == 1
        assert trace.n_specific == 1
        assert "SUPPORTS" in trace.explain()


class TestCitationGating:
    def test_only_citations_agreeing_with_the_verdict_survive(self):
        audit = report(align(1, "REFUTES"), align(2, "SUPPORTS"), align(3, "NOT_ENOUGH_INFO"))
        kept = consistent_citations(
            ["doc_1", "doc_2", "doc_3"], audit, "REFUTES", ["doc_1", "doc_2", "doc_3"]
        )
        assert kept == ["doc_1"]

    def test_fabricated_identifiers_are_dropped(self):
        audit = report(align(1, "REFUTES"))
        assert consistent_citations(["doc_9"], audit, "REFUTES", ["doc_1"]) == []

    def test_an_undecided_verdict_cites_nothing(self):
        """A citation on 'the evidence settles nothing' asserts the opposite."""
        audit = report(align(1, "NOT_ENOUGH_INFO"))
        assert consistent_citations(["doc_1"], audit, "NOT_ENOUGH_INFO", ["doc_1"]) == []

    def test_order_is_preserved_and_duplicates_removed(self):
        audit = report(align(1, "SUPPORTS"), align(2, "SUPPORTS"))
        kept = consistent_citations(
            ["doc_2", "doc_1", "doc_2"], audit, "SUPPORTS", ["doc_1", "doc_2"]
        )
        assert kept == ["doc_2", "doc_1"]

    def test_supporting_ids_names_what_decided_the_verdict(self):
        audit = report(align(1, "SUPPORTS"), align(2, "REFUTES"), align(3, "SUPPORTS"))
        assert supporting_ids(audit, "SUPPORTS") == ["doc_1", "doc_3"]

    def test_supporting_ids_is_empty_for_an_undecided_verdict(self):
        assert supporting_ids(report(align(1, "NOT_ENOUGH_INFO")), "NOT_ENOUGH_INFO") == []
