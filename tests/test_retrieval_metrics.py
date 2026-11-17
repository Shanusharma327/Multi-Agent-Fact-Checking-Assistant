"""Tests for retrieval scoring and citation quality measurement."""

from __future__ import annotations

import pytest

from fact_checker.retrieval_metrics import (
    score_citations,
    score_one,
    score_retrieval,
)


class TestScoreOne:
    def test_a_perfect_retrieval(self):
        row = score_one(["https://a.com/1"], ["http://www.a.com/1"], domain=False)
        assert row["recall"] == 1.0
        assert row["precision"] == 1.0

    def test_a_complete_miss(self):
        row = score_one(["https://b.com/1"], ["https://a.com/1"], domain=False)
        assert row["recall"] == 0.0
        assert row["precision"] == 0.0

    def test_partial_recall(self):
        row = score_one(
            ["https://a.com/1", "https://c.com/9"],
            ["https://a.com/1", "https://b.com/2"],
            domain=False,
        )
        assert row["recall"] == 0.5
        assert row["precision"] == 0.5

    def test_domain_matching_is_more_forgiving_than_exact(self):
        """A different page on the cited source still names the right source."""
        retrieved = ["https://bbc.co.uk/news/other-story"]
        gold = ["https://bbc.co.uk/news/the-story"]
        assert score_one(retrieved, gold, domain=False)["recall"] == 0.0
        assert score_one(retrieved, gold, domain=True)["recall"] == 1.0

    def test_archived_gold_matches_a_live_retrieval(self):
        retrieved = ["https://bbc.co.uk/news/x"]
        gold = ["https://web.archive.org/web/2020/https://www.bbc.co.uk/news/x"]
        assert score_one(retrieved, gold, domain=False)["recall"] == 1.0

    def test_duplicate_retrievals_cannot_double_count(self):
        row = score_one(
            ["https://a.com/1", "http://www.a.com/1/"], ["https://a.com/1"], domain=False
        )
        assert row["n_retrieved"] == 1
        assert row["precision"] == 1.0

    def test_a_claim_without_gold_is_marked_unscorable(self):
        assert score_one(["https://a.com/1"], [], domain=False)["n_gold"] == 0.0


class TestScoreRetrieval:
    def test_macro_average_over_claims(self):
        score = score_retrieval(
            [["https://a.com/1"], ["https://z.com/9"]],
            [["https://a.com/1"], ["https://b.com/2"]],
        )
        assert score.n_claims == 2
        assert score.recall_at_k == 0.5
        assert score.hit_rate == 0.5

    def test_claims_without_gold_are_excluded_not_zeroed(self):
        """Counting an unannotated claim as a miss penalises an annotation gap."""
        score = score_retrieval(
            [["https://a.com/1"], ["https://b.com/2"]],
            [["https://a.com/1"], []],
        )
        assert score.n_claims == 1
        assert score.recall_at_k == 1.0

    def test_empty_input_is_safe(self):
        score = score_retrieval([], [])
        assert score.n_claims == 0
        assert score.recall_at_k == 0.0

    def test_misaligned_input_is_rejected(self):
        with pytest.raises(ValueError, match="align"):
            score_retrieval([["a"]], [])

    def test_granularity_is_reported(self):
        assert score_retrieval([], [], domain=True).granularity == "domain"
        assert score_retrieval([], [], domain=False).granularity == "exact"


def record(verdict, cited, evidence_ids, audit):
    return {
        "multi": {
            "verdict": verdict,
            "cited_sources": cited,
            "evidence_ids": evidence_ids,
            "audit": [{"source_id": s, "verdict": v} for s, v in audit],
        }
    }


class TestScoreCitations:
    def test_validity_counts_only_retrieved_identifiers(self):
        rows = [record("SUPPORTS", ["doc_1", "doc_9"], ["doc_1"], [("doc_1", "SUPPORTS")])]
        score = score_citations(rows)
        assert score.n_citations == 2
        assert score.validity == 0.5

    def test_support_precision_requires_the_audit_to_agree(self):
        rows = [
            record(
                "REFUTES",
                ["doc_1", "doc_2"],
                ["doc_1", "doc_2"],
                [("doc_1", "REFUTES"), ("doc_2", "NOT_ENOUGH_INFO")],
            )
        ]
        score = score_citations(rows)
        assert score.validity == 1.0
        assert score.support_precision == 0.5

    def test_undecided_verdicts_are_excluded_from_support(self):
        """NOT_ENOUGH_INFO has no direction for a citation to agree with."""
        rows = [
            record("NOT_ENOUGH_INFO", ["doc_1"], ["doc_1"], [("doc_1", "NOT_ENOUGH_INFO")])
        ]
        score = score_citations(rows)
        assert score.n_supportable == 0
        assert score.support_precision == 0.0

    def test_grounded_rate_counts_verdicts_not_citations(self):
        rows = [
            record("SUPPORTS", ["doc_1", "doc_2"], ["doc_1", "doc_2"], []),
            record("REFUTES", [], ["doc_1"], []),
        ]
        score = score_citations(rows)
        assert score.n_verdicts == 2
        assert score.n_with_citations == 1
        assert score.grounded_rate == 0.5

    def test_records_without_a_verdict_are_skipped(self):
        assert score_citations([{"multi": None}, {"multi": {}}]).n_verdicts == 0
