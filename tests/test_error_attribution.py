"""Tests for the error-attribution cascade.

Attribution assigns each wrong verdict to the earliest stage that failed.
The ordering matters: a later stage must not be blamed for working on
input an earlier one spoiled. These tests pin each branch and the
cascade's precedence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "analysis"))

from error_attribution import STAGES, attribute, attribute_run, format_attribution  # noqa: E402


def record(
    gold="SUPPORTS",
    verdict="REFUTES",
    audit=(),
    retrieved=("https://other.example/1",),
    gold_urls=("https://gold.example/a",),
    n_evidence=None,
):
    return {
        "id": "c1",
        "gold": gold,
        "verdict": verdict,
        "audit": [
            {"source_id": s, "verdict": v, "justification": j}
            for s, v, j in audit
        ],
        "retrieved_urls": list(retrieved),
        "gold_urls": list(gold_urls),
        "n_evidence": len(retrieved) if n_evidence is None else n_evidence,
    }


class TestRetrievalFailures:
    def test_empty_evidence_is_a_retrieval_failure(self):
        result = attribute(record(retrieved=(), n_evidence=0))
        assert result.stage == "retrieval"
        assert "nothing" in result.reason

    def test_nothing_usable_retrieved_is_a_retrieval_failure(self):
        """No gold-direction label and no gold-domain hit: nothing to work with."""
        result = attribute(
            record(
                gold="SUPPORTS",
                verdict="NOT_ENOUGH_INFO",
                audit=(("doc_1", "NOT_ENOUGH_INFO", "off topic"),),
            )
        )
        assert result.stage == "retrieval"

    def test_a_gold_domain_hit_rescues_retrieval_from_blame(self):
        """If the right source was found, retrieval did its job."""
        result = attribute(
            record(
                gold="SUPPORTS",
                verdict="NOT_ENOUGH_INFO",
                audit=(("doc_1", "NOT_ENOUGH_INFO", "vague"),),
                retrieved=("https://gold.example/different-page",),
            )
        )
        assert result.stage == "verification"
        assert result.gold_source_retrieved


class TestVerificationFailures:
    def test_the_gold_source_was_found_but_mislabelled(self):
        """Retrieval did its job; the per-passage judgement went the wrong way."""
        result = attribute(
            record(
                gold="REFUTES",
                verdict="SUPPORTS",
                audit=(("doc_1", "SUPPORTS", "the passage agrees"),),
                retrieved=("https://gold.example/the-page",),
            )
        )
        assert result.stage == "verification"
        assert result.gold_source_retrieved
        assert not result.gold_in_audit

    def test_an_audit_pointing_away_from_gold_is_a_verification_failure(self):
        result = attribute(
            record(
                gold="SUPPORTS",
                verdict="REFUTES",
                audit=(
                    ("doc_1", "REFUTES", "contradicted by the 2019 record"),
                    ("doc_2", "REFUTES", "also contradicted"),
                ),
                retrieved=("https://gold.example/a", "https://other.example/2"),
            )
        )
        assert result.stage == "verification"
        assert result.aggregated_label == "REFUTES"


class TestAggregationFailures:
    def test_a_correct_audit_with_a_wrong_verdict(self):
        result = attribute(
            record(
                gold="SUPPORTS",
                verdict="REFUTES",
                audit=(
                    ("doc_1", "SUPPORTS", "census data 45,000"),
                    ("doc_2", "SUPPORTS", "confirmed by the 2019 report"),
                ),
            )
        )
        assert result.stage == "aggregation"
        assert result.aggregated_label == "SUPPORTS"

    def test_aggregation_is_not_blamed_when_the_audit_is_also_wrong(self):
        """Precedence: the earlier failure owns the error."""
        result = attribute(
            record(
                gold="SUPPORTS",
                verdict="REFUTES",
                audit=(("doc_1", "REFUTES", "the record says otherwise"),),
            )
        )
        assert result.stage != "aggregation"


class TestRunSummary:
    def test_only_wrong_verdicts_are_attributed(self):
        records = [
            record(gold="SUPPORTS", verdict="SUPPORTS"),
            record(gold="SUPPORTS", verdict="REFUTES"),
        ]
        summary = attribute_run(records)
        assert summary["n_scored"] == 2
        assert summary["n_wrong"] == 1

    def test_shares_sum_to_one_when_there_are_errors(self):
        records = [record(gold="SUPPORTS", verdict="REFUTES") for _ in range(4)]
        shares = attribute_run(records)["shares"]
        assert sum(shares.values()) == 1.0

    def test_every_stage_appears_in_the_breakdown(self):
        summary = attribute_run([record()])
        assert set(summary["counts"]) == set(STAGES)

    def test_a_perfect_run_attributes_nothing(self):
        summary = attribute_run([record(gold="SUPPORTS", verdict="SUPPORTS")])
        assert summary["n_wrong"] == 0
        assert sum(summary["shares"].values()) == 0.0

    def test_records_without_a_verdict_are_not_scored(self):
        assert attribute_run([{"gold": "SUPPORTS"}])["n_scored"] == 0

    def test_the_breakdown_renders(self):
        text = format_attribution(attribute_run([record()]), "MAFCA")
        assert "MAFCA" in text
        assert "retrieval" in text
