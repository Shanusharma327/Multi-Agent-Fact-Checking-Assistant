"""Replay must reproduce a live deterministic run exactly.

The aggregation ablation replays policies over stored audits rather than
re-running the pipeline per condition, because live web search would make
the conditions differ in their evidence as well as in their aggregation.
That design is only sound if replay is *identical* to having run the policy
live. These tests pin that equivalence, so the ablation cannot quietly drift
away from the system it claims to describe.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "analysis"))

from ablation import make_deterministic, policy_as_run, rebuild_audit  # noqa: E402

from fact_checker.agents.adjudicator import apply_deterministic  # noqa: E402
from fact_checker.aggregation import aggregate  # noqa: E402
from fact_checker.schemas import Adjudication, Alignment, AuditReport  # noqa: E402


def make_audit(*pairs):
    return AuditReport(
        alignments=[
            Alignment(source_id=s, verdict=v, justification=j) for s, v, j in pairs
        ]
    )


def make_block(audit: AuditReport, cited, verdict="SUPPORTS", confidence=0.9):
    """A stored result block of the shape ``evaluate.py`` writes."""
    return {
        "verdict": verdict,
        "confidence": confidence,
        "cited_sources": list(cited),
        "evidence_ids": [a.source_id for a in audit.alignments],
        "audit": [
            {"source_id": a.source_id, "verdict": a.verdict, "justification": a.justification}
            for a in audit.alignments
        ],
    }


AUDITS = [
    make_audit(
        ("doc_1", "REFUTES", "the official record gives 1971"),
        ("doc_2", "SUPPORTS", "broadly agrees"),
        ("doc_3", "NOT_ENOUGH_INFO", "off topic"),
    ),
    make_audit(
        ("doc_1", "SUPPORTS", "census data: 45,000"),
        ("doc_2", "SUPPORTS", "a second report confirms 45,000"),
    ),
    make_audit(("doc_1", "NOT_ENOUGH_INFO", "unrelated")),
    make_audit(),
]


class TestRebuildAudit:
    def test_a_stored_audit_round_trips(self):
        for audit in AUDITS:
            rebuilt = rebuild_audit(make_block(audit, []))
            assert [(a.source_id, a.verdict, a.justification) for a in rebuilt.alignments] == [
                (a.source_id, a.verdict, a.justification) for a in audit.alignments
            ]

    def test_entries_without_an_identifier_are_dropped(self):
        block = {"audit": [{"source_id": "", "verdict": "SUPPORTS", "justification": "x"}]}
        assert rebuild_audit(block).alignments == []


class TestReplayMatchesLive:
    def test_verdict_and_confidence_are_identical(self):
        """Replaying a policy equals having applied it during the run."""
        for audit in AUDITS:
            for beta in (0.0, 1.0, 2.0):
                live, _ = apply_deterministic(
                    Adjudication(
                        claim="c",
                        final_verdict="SUPPORTS",
                        confidence=0.9,
                        detailed_reasoning="r",
                        cited_sources=["doc_1"],
                    ),
                    audit,
                    {a.source_id for a in audit.alignments},
                    beta=beta,
                    gate_citations=True,
                )
                replayed = make_deterministic(beta, gate=True)(
                    make_block(audit, ["doc_1"])
                )
                assert replayed[0] == live.final_verdict
                assert replayed[1] == live.confidence
                assert replayed[2] == live.cited_sources

    def test_gating_off_also_matches(self):
        audit = AUDITS[0]
        live, _ = apply_deterministic(
            Adjudication(
                claim="c",
                final_verdict="SUPPORTS",
                confidence=0.9,
                detailed_reasoning="r",
                cited_sources=["doc_1", "doc_9"],
            ),
            audit,
            {a.source_id for a in audit.alignments},
            beta=1.0,
            gate_citations=False,
        )
        replayed = make_deterministic(1.0, gate=False)(
            make_block(audit, ["doc_1", "doc_9"])
        )
        assert replayed[:2] == (live.final_verdict, live.confidence)
        assert replayed[2] == live.cited_sources

    def test_a_stored_trace_is_reproducible_from_its_audit(self):
        """The recorded arithmetic can be regenerated without the model."""
        audit = AUDITS[0]
        original = aggregate(audit, beta=1.0)
        block = make_block(audit, [])
        regenerated = aggregate(rebuild_audit(block), beta=1.0)
        assert regenerated.scores == original.scores
        assert regenerated.verdict == original.verdict
        assert regenerated.confidence == original.confidence


class TestPolicyAsRun:
    def test_the_reference_policy_is_a_passthrough(self):
        block = make_block(AUDITS[0], ["doc_1"], verdict="REFUTES", confidence=0.42)
        assert policy_as_run(block) == ("REFUTES", 0.42, ["doc_1"])


class TestOverrideIsRecorded:
    def test_the_replaced_label_is_kept_for_comparison(self):
        _, trace = apply_deterministic(
            Adjudication(
                claim="c",
                final_verdict="SUPPORTS",
                confidence=0.95,
                detailed_reasoning="r",
                cited_sources=[],
            ),
            AUDITS[2],
            {"doc_1"},
        )
        assert trace.replaced_verdict == "SUPPORTS"
        assert trace.replaced_confidence == 0.95
        assert trace.verdict == "NOT_ENOUGH_INFO"
