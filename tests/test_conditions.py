"""Tests for the experimental conditions and B1's schema conformance.

B1 is the primary comparison, so its contract matters as much as its
prompt: it must emit the same object MAFCA emits, validate its citations
the same way, and consume the cached evidence unchanged. A silent
divergence in any of those would reintroduce the confound B1 exists to
remove.
"""

from __future__ import annotations

from typing import List

import pytest

from fact_checker.agents.rag_baseline import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    _format_evidence,
    run_rag_baseline,
)
from fact_checker.cache import CachedClaim, CachedPassage
from fact_checker.conditions import (
    CONDITIONS,
    LLM_CALLS,
    condition_majority,
    condition_random,
    majority_label,
)
from fact_checker.config import Settings
from fact_checker.schemas import Adjudication, BaselineResult


def passage(index: int, snippet: str = "evidence text") -> CachedPassage:
    return CachedPassage(
        source_id=f"doc_{index}",
        url=f"https://example.com/{index}",
        title=f"t{index}",
        snippet=snippet,
        query="q",
    )


def entry(claim_id="c1", gold="SUPPORTS", n=2) -> CachedClaim:
    return CachedClaim(
        claim_id=claim_id,
        dataset="averitec",
        split="dev",
        claim="a claim under test",
        gold=gold,
        gold_urls=["https://gold.example/1"],
        passages=[passage(i) for i in range(1, n + 1)],
        queries=["q1", "q2"],
        retrieved_at="2026-01-01T00:00:00+00:00",
    )


class StubLLM:
    """Stands in for a chat model, returning a scripted Adjudication."""

    def __init__(self, payload: Adjudication) -> None:
        self.payload = payload
        self.prompts: List[tuple] = []

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        self.prompts.append(tuple(messages))
        return self.payload


@pytest.fixture
def settings() -> Settings:
    return Settings(provider="ollama", model="stub")


class TestB1Prompt:
    def test_the_prompt_imposes_the_closed_world_rule(self):
        """B1 must be held to the same restriction as the auditor."""
        assert "CLOSED-WORLD RULE" in SYSTEM_PROMPT
        assert "no outside memory" in SYSTEM_PROMPT

    def test_the_prompt_carries_the_adjudication_rules(self):
        """A weak prompt would manufacture MAFCA's advantage."""
        for rule in (
            "Direct evidence beats volume",
            "Absence of evidence is never evidence of refutation",
        ):
            assert rule in SYSTEM_PROMPT

    def test_the_prompt_carries_the_same_confidence_rubric(self):
        assert "probability that your verdict is correct" in SYSTEM_PROMPT
        assert "0.90-1.00" in SYSTEM_PROMPT

    def test_the_prompt_instructs_citation_of_identifiers(self):
        assert "cited_sources" in SYSTEM_PROMPT
        assert "Never invent an identifier" in SYSTEM_PROMPT

    def test_evidence_is_formatted_exactly_as_the_auditor_formats_it(self):
        """A formatting difference would be a second uncontrolled variable."""
        from fact_checker.agents.auditor import _format_evidence as audit_format

        scout = entry(n=3).to_scout_report()
        assert _format_evidence(scout) == audit_format(scout)

    def test_an_empty_evidence_set_is_stated_not_omitted(self):
        assert "No evidence" in _format_evidence(entry(n=0).to_scout_report())

    def test_the_user_template_exposes_the_valid_identifiers(self):
        assert "{ids}" in USER_TEMPLATE
        assert "{evidence_block}" in USER_TEMPLATE


class TestB1Conformance:
    def _run(self, monkeypatch, settings, payload: Adjudication):
        stub = StubLLM(payload)
        monkeypatch.setattr(
            "fact_checker.agents.rag_baseline.build_llm", lambda *a, **k: stub
        )
        result, seconds = run_rag_baseline(
            "a claim under test", entry(n=2).to_scout_report(), settings
        )
        return result, seconds, stub

    def test_it_emits_the_shared_adjudication_schema(self, monkeypatch, settings):
        payload = Adjudication(
            claim="restated differently",
            final_verdict="SUPPORTS",
            confidence=0.8,
            detailed_reasoning="because doc_1 says so",
            cited_sources=["doc_1"],
        )
        result, _, _ = self._run(monkeypatch, settings, payload)
        assert isinstance(result, BaselineResult)
        assert isinstance(result.adjudication, Adjudication)

    def test_the_claim_is_pinned_to_the_original_wording(self, monkeypatch, settings):
        """A paraphrase would silently change what was adjudicated."""
        payload = Adjudication(
            claim="a model paraphrase",
            final_verdict="REFUTES",
            confidence=0.5,
            detailed_reasoning="r",
            cited_sources=[],
        )
        result, _, _ = self._run(monkeypatch, settings, payload)
        assert result.adjudication.claim == "a claim under test"

    def test_fabricated_citations_are_dropped(self, monkeypatch, settings):
        """The same post-condition Agent 3 gets; withholding it would flatter MAFCA."""
        payload = Adjudication(
            claim="c", final_verdict="SUPPORTS", confidence=0.7,
            detailed_reasoning="r", cited_sources=["doc_1", "doc_99"],
        )
        result, _, _ = self._run(monkeypatch, settings, payload)
        assert result.adjudication.cited_sources == ["doc_1"]

    def test_duplicate_citations_collapse(self, monkeypatch, settings):
        payload = Adjudication(
            claim="c", final_verdict="SUPPORTS", confidence=0.7,
            detailed_reasoning="r", cited_sources=["doc_1", "doc_1", "doc_2"],
        )
        result, _, _ = self._run(monkeypatch, settings, payload)
        assert result.adjudication.cited_sources == ["doc_1", "doc_2"]

    def test_the_passages_reach_the_prompt_verbatim(self, monkeypatch, settings):
        payload = Adjudication(
            claim="c", final_verdict="SUPPORTS", confidence=0.7,
            detailed_reasoning="r", cited_sources=[],
        )
        _, _, stub = self._run(monkeypatch, settings, payload)
        user_message = stub.prompts[0][1][1]
        assert "doc_1" in user_message
        assert "evidence text" in user_message

    def test_it_makes_exactly_one_model_call(self, monkeypatch, settings):
        payload = Adjudication(
            claim="c", final_verdict="SUPPORTS", confidence=0.7,
            detailed_reasoning="r", cited_sources=[],
        )
        _, _, stub = self._run(monkeypatch, settings, payload)
        assert len(stub.prompts) == 1


class TestTrivialBaselines:
    def test_majority_is_taken_from_the_evaluated_sample(self):
        """Not from the corpus: a stratified draw has a different majority."""
        entries = [entry(f"c{i}", gold="REFUTES") for i in range(5)]
        entries += [entry(f"d{i}", gold="SUPPORTS") for i in range(2)]
        assert majority_label(entries) == "REFUTES"

    def test_majority_predicts_one_label_throughout(self):
        entries = [entry(f"c{i}", gold="REFUTES") for i in range(4)]
        records = condition_majority(entries)
        assert {r["verdict"] for r in records} == {"REFUTES"}
        assert all(r["llm_calls"] == 0 for r in records)

    def test_majority_ties_break_deterministically(self):
        entries = [entry("a", gold="SUPPORTS"), entry("b", gold="REFUTES")]
        assert majority_label(entries) == majority_label(list(reversed(entries)))

    def test_random_is_reproducible_under_a_seed(self):
        entries = [entry(f"c{i}") for i in range(20)]
        first = [r["verdict"] for r in condition_random(entries, seed=7)]
        second = [r["verdict"] for r in condition_random(entries, seed=7)]
        assert first == second

    def test_random_differs_across_seeds(self):
        entries = [entry(f"c{i}") for i in range(40)]
        a = [r["verdict"] for r in condition_random(entries, seed=1)]
        b = [r["verdict"] for r in condition_random(entries, seed=2)]
        assert a != b

    def test_trivial_baselines_carry_no_evidence_fingerprint(self):
        """They consumed no evidence, so asserting identity would be false."""
        entries = [entry("c1")]
        for records in (condition_majority(entries), condition_random(entries)):
            assert "evidence_fingerprint" not in records[0]

    def test_their_confidence_is_the_uninformative_prior(self):
        """Using the sample base rate would leak the label distribution."""
        records = condition_majority([entry("c1")])
        assert records[0]["confidence"] == pytest.approx(1 / 3)


class TestConditionRegistry:
    def test_every_registered_condition_has_a_call_count(self):
        for name in CONDITIONS:
            if name != "B6":  # n-dependent, read from the records
                assert name in LLM_CALLS

    def test_the_decomposed_conditions_cost_more_calls_than_the_flat_ones(self):
        assert LLM_CALLS["MAFCA"] > LLM_CALLS["B1"] == LLM_CALLS["B0"]

    def test_replayed_conditions_are_registered(self):
        for name in ("B0", "B1", "B2", "B3", "B4", "MAFCA"):
            assert name in CONDITIONS
