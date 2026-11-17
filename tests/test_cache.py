"""Tests for the passage cache and the evidence-identity guarantee.

The whole condition-comparison design rests on one property: every
evidence-using condition sees byte-identical passages for a given claim.
These tests pin that property, including the failure mode where it stops
holding.
"""

from __future__ import annotations

import json

import pytest

from fact_checker.cache import (
    CachedClaim,
    CachedPassage,
    EvidenceMismatch,
    PassageCache,
    assert_identical_evidence,
    utc_now,
)


def passage(index: int, snippet: str = "text", url: str = "") -> CachedPassage:
    return CachedPassage(
        source_id=f"doc_{index}",
        url=url or f"https://example.com/{index}",
        title=f"title {index}",
        snippet=snippet,
        query="q",
    )


def entry(claim_id: str = "c1", n: int = 2, **kwargs) -> CachedClaim:
    defaults = dict(
        claim_id=claim_id,
        dataset="averitec",
        split="dev",
        claim="a claim",
        gold="SUPPORTS",
        gold_urls=["https://gold.example/a"],
        fact_check_url="https://factcheck.example/x",
        queries=["q1", "q2"],
        passages=[passage(i) for i in range(1, n + 1)],
        retrieved_at=utc_now(),
    )
    defaults.update(kwargs)
    return CachedClaim(**defaults)


class TestFingerprint:
    def test_identical_evidence_fingerprints_alike(self):
        assert entry().fingerprint() == entry().fingerprint()

    def test_a_changed_snippet_changes_the_fingerprint(self):
        other = entry()
        other.passages[0].snippet = "different"
        assert other.fingerprint() != entry().fingerprint()

    def test_a_changed_url_changes_the_fingerprint(self):
        other = entry()
        other.passages[0].url = "https://elsewhere.example/1"
        assert other.fingerprint() != entry().fingerprint()

    def test_reordering_changes_the_fingerprint(self):
        """Order matters: identifiers are positional in the prompt."""
        other = entry()
        other.passages.reverse()
        assert other.fingerprint() != entry().fingerprint()

    def test_metadata_no_agent_reads_does_not_change_it(self):
        """Title and query are retrieval metadata, not evidence."""
        other = entry()
        other.passages[0].title = "a totally different title"
        other.passages[0].query = "another query"
        assert other.fingerprint() == entry().fingerprint()

    def test_an_empty_evidence_set_has_a_stable_fingerprint(self):
        assert entry(n=0).fingerprint() == entry(n=0).fingerprint()


class TestScoutReconstruction:
    def test_passages_become_an_agent_one_report(self):
        report = entry(n=3).to_scout_report()
        assert report.source_ids() == ["doc_1", "doc_2", "doc_3"]
        assert report.queries == ["q1", "q2"]

    def test_snippets_survive_the_round_trip(self):
        source = entry(n=1)
        source.passages[0].snippet = "the exact words"
        assert source.to_scout_report().evidence[0].snippet == "the exact words"

    def test_empty_evidence_is_preserved_not_invented(self):
        assert entry(n=0).to_scout_report().evidence == []
        assert entry(n=0).is_empty


class TestPersistence:
    def test_round_trip_through_jsonl(self, tmp_path):
        cache = PassageCache()
        cache.add(entry("c1"))
        cache.add(entry("c2", n=3))
        path = cache.save(tmp_path / "cache.jsonl")

        loaded = PassageCache.load(path)
        assert len(loaded) == 2
        assert loaded.get("c2").fingerprint() == cache.get("c2").fingerprint()

    def test_loading_a_missing_cache_fails_loudly(self, tmp_path):
        """A silently empty cache would look like total retrieval failure."""
        with pytest.raises(FileNotFoundError, match="No passage cache"):
            PassageCache.load(tmp_path / "absent.jsonl")

    def test_subset_preserves_requested_order(self):
        cache = PassageCache()
        for name in ("a", "b", "c"):
            cache.add(entry(name))
        assert cache.subset(["c", "a"]).claim_ids() == ["c", "a"]

    def test_subset_ignores_unknown_ids(self):
        cache = PassageCache()
        cache.add(entry("a"))
        assert cache.subset(["a", "zzz"]).claim_ids() == ["a"]


class TestFromResults:
    def _record(self, claim_id="c1", evidence=None):
        return {
            "id": claim_id,
            "dataset": "averitec",
            "claim": "a claim",
            "gold": "REFUTES",
            "gold_urls": ["https://gold.example/x"],
            "fact_check_url": "https://fc.example/y",
            "multi": {
                "queries": ["q1"],
                "warnings": [],
                "evidence": evidence
                if evidence is not None
                else [
                    {"source_id": "doc_1", "url": "https://a.example/1",
                     "title": "t", "snippet": "s", "query": "q1"}
                ],
            },
        }

    def test_a_results_file_becomes_a_cache(self):
        cache = PassageCache.from_results([self._record()])
        assert len(cache) == 1
        assert cache.get("c1").passages[0].snippet == "s"

    def test_records_without_a_stored_trace_are_skipped(self):
        """An absent trace is not an empty evidence set, and must not become one."""
        record = self._record()
        del record["multi"]["evidence"]
        assert len(PassageCache.from_results([record])) == 0

    def test_a_genuinely_empty_evidence_set_is_kept(self):
        cache = PassageCache.from_results([self._record(evidence=[])])
        assert len(cache) == 1
        assert cache.get("c1").is_empty

    def test_a_stamp_can_be_applied_to_unstamped_records(self):
        cache = PassageCache.from_results(
            [self._record()], retrieved_at="2026-01-01T00:00:00+00:00"
        )
        assert cache.get("c1").retrieved_at == "2026-01-01T00:00:00+00:00"


class TestSummary:
    def test_counts_are_reported_for_interpretation(self):
        cache = PassageCache()
        cache.add(entry("a", n=2, gold="SUPPORTS", split="dev"))
        cache.add(entry("b", n=0, gold="REFUTES", split="train"))
        summary = cache.summary()
        assert summary["claims"] == 2
        assert summary["empty_evidence"] == 1
        assert summary["total_passages"] == 2
        assert summary["labels"] == {"SUPPORTS": 1, "REFUTES": 1}
        assert summary["splits"] == {"dev": 1, "train": 1}


class TestEvidenceIdentity:
    def _records(self, fingerprint: str, claim_ids=("c1", "c2")):
        return [
            {"id": c, "evidence_fingerprint": fingerprint, "verdict": "SUPPORTS"}
            for c in claim_ids
        ]

    def test_matching_conditions_pass(self):
        checked = assert_identical_evidence(
            {"B1": self._records("abc"), "MAFCA": self._records("abc")}
        )
        assert checked == 2

    def test_a_mismatch_fails_the_run(self):
        with pytest.raises(EvidenceMismatch, match="different evidence"):
            assert_identical_evidence(
                {"B1": self._records("abc"), "MAFCA": self._records("xyz")}
            )

    def test_the_failure_names_the_offending_claim(self):
        bad = [
            {"id": "c1", "evidence_fingerprint": "abc"},
            {"id": "c2", "evidence_fingerprint": "zzz"},
        ]
        good = self._records("abc")
        with pytest.raises(EvidenceMismatch, match="c2"):
            assert_identical_evidence({"A": bad, "B": good})

    def test_conditions_without_evidence_are_skipped_not_failed(self):
        """B0 retrieves nothing, so it carries no fingerprint by design."""
        b0 = [{"id": "c1", "verdict": "SUPPORTS"}, {"id": "c2", "verdict": "REFUTES"}]
        checked = assert_identical_evidence(
            {"B0": b0, "B1": self._records("abc"), "MAFCA": self._records("abc")}
        )
        assert checked == 2

    def test_a_claim_in_only_one_condition_is_not_checked(self):
        checked = assert_identical_evidence(
            {"A": [{"id": "lonely", "evidence_fingerprint": "abc"}]}
        )
        assert checked == 0

    def test_an_empty_family_is_safe(self):
        assert assert_identical_evidence({}) == 0
