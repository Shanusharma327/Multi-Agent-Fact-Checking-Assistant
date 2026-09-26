"""End-to-end pipeline test with the LLM and search layers stubbed out.

Exercises the real orchestration path -- Agent 1 -> 2 -> 3, timings, warnings,
reconciliation and citation sanitising -- without a network call or API key.
"""

from __future__ import annotations

import pytest

import fact_checker.agents.adjudicator as adjudicator_mod
import fact_checker.agents.auditor as auditor_mod
import fact_checker.agents.baseline as baseline_mod
import fact_checker.agents.scout as scout_mod
import fact_checker.search as search_module
from fact_checker import Settings, run_multi_agent
from fact_checker.agents import run_baseline
from fact_checker.schemas import Adjudication, Alignment, AuditReport, SearchQueries

CLAIM = "The Eiffel Tower is still the tallest structure in the world."


@pytest.fixture
def stubbed(monkeypatch):
    """Replace every outbound call with a deterministic canned response."""
    monkeypatch.setattr(scout_mod, "build_llm", lambda *a, **k: object())
    monkeypatch.setattr(auditor_mod, "build_llm", lambda *a, **k: object())
    monkeypatch.setattr(adjudicator_mod, "build_llm", lambda *a, **k: object())
    monkeypatch.setattr(baseline_mod, "build_llm", lambda *a, **k: object())

    monkeypatch.setattr(
        scout_mod,
        "structured_call",
        lambda *a, **k: SearchQueries(
            queries=["eiffel tower height record", "tallest structure world current"]
        ),
    )
    monkeypatch.setattr(
        search_module,
        "_raw_search",
        lambda query, max_results: [
            {
                "title": "Eiffel Tower",
                "href": "https://a.example",
                "body": "Completed in 1889 at 300 m, it was the tallest until 1930.",
            },
            {
                "title": "Burj Khalifa",
                "href": "https://b.example",
                "body": "The Burj Khalifa stands 828 m and is the tallest structure.",
            },
        ],
    )

    def fake_audit(llm, schema, system, user, *, stage):
        # Deliberately imperfect: one hallucinated ID, one snippet omitted.
        return AuditReport(
            alignments=[
                Alignment(
                    source_id="doc_1",
                    verdict="REFUTES",
                    justification="It says 'tallest until 1930'.",
                ),
                Alignment(
                    source_id="doc_99",
                    verdict="SUPPORTS",
                    justification="Hallucinated source.",
                ),
            ]
        )

    monkeypatch.setattr(auditor_mod, "structured_call", fake_audit)

    def fake_adjudicate(llm, schema, system, user, *, stage):
        return Adjudication(
            claim="a paraphrase that should be overwritten",
            final_verdict="REFUTES",
            confidence=0.94,
            detailed_reasoning="doc_1 dates the record to 1930; doc_2 names Burj Khalifa.",
            cited_sources=["doc_1", "doc_2", "doc_77"],
        )

    monkeypatch.setattr(adjudicator_mod, "structured_call", fake_adjudicate)

    monkeypatch.setattr(
        baseline_mod,
        "structured_call",
        lambda *a, **k: Adjudication(
            claim="x",
            final_verdict="SUPPORTS",
            confidence=0.88,
            detailed_reasoning="From memory.",
            cited_sources=["doc_1", "made_up"],
        ),
    )


def test_full_pipeline(stubbed) -> None:
    stages: list[str] = []
    result = run_multi_agent(CLAIM, Settings(api_key="k"), on_progress=stages.append)

    # Agent 1 retrieved and numbered the evidence.
    assert result.scout.queries == [
        "eiffel tower height record",
        "tallest structure world current",
    ]
    assert result.scout.source_ids() == ["doc_1", "doc_2"]

    # Agent 2's hallucinated ID was dropped and the missing snippet backfilled.
    assert [a.source_id for a in result.audit.alignments] == ["doc_1", "doc_2"]
    assert result.audit.alignments[1].verdict == "NOT_ENOUGH_INFO"

    # Agent 3's verdict kept only real citations and the user's exact wording.
    assert result.adjudication.final_verdict == "REFUTES"
    assert result.adjudication.cited_sources == ["doc_1", "doc_2"]
    assert result.adjudication.claim == CLAIM

    # Diagnostics are populated.
    assert len(result.timings) == 3
    assert result.total_seconds >= 0.0
    assert len(stages) == 3


def test_baseline_citations_are_discarded(stubbed) -> None:
    """The baseline has no sources, so anything it cites is fabricated."""
    result = run_baseline(CLAIM, Settings(api_key="k"))

    assert result.adjudication.final_verdict == "SUPPORTS"
    assert result.adjudication.cited_sources == []
    assert result.adjudication.claim == CLAIM
    assert result.seconds >= 0.0


def test_empty_evidence_skips_the_auditor(stubbed, monkeypatch) -> None:
    """Total search failure must degrade, not crash."""
    monkeypatch.setattr(
        search_module,
        "_raw_search",
        lambda q, m: (_ for _ in ()).throw(RuntimeError("Ratelimit 202")),
    )

    result = run_multi_agent(CLAIM, Settings(api_key="k"))

    assert result.scout.evidence == []
    assert result.audit.alignments == []
    assert result.adjudication.cited_sources == []
    assert result.warnings, "a retrieval failure should surface a warning"
