"""Tests for evidence assembly, using a stubbed search backend."""

from __future__ import annotations

import fact_checker.search as search_module
from fact_checker.search import gather_evidence


def test_ids_are_sequential_and_duplicates_collapse(monkeypatch) -> None:
    """The same URL surfacing twice must not become two 'independent' sources."""
    rows = {
        "query one": [
            {"title": "A", "href": "https://a.example", "body": "alpha body text"},
            {"title": "B", "href": "https://b.example", "body": "beta body text"},
        ],
        "query two": [
            # Same URL as above -> dropped.
            {"title": "A", "href": "https://a.example", "body": "alpha body text"},
            {"title": "C", "href": "https://c.example", "body": "gamma body text"},
        ],
    }
    monkeypatch.setattr(
        search_module, "_raw_search", lambda query, max_results: rows[query]
    )

    outcome = gather_evidence(["query one", "query two"], results_per_query=2)

    assert [item.source_id for item in outcome.evidence] == ["doc_1", "doc_2", "doc_3"]
    assert [item.title for item in outcome.evidence] == ["A", "B", "C"]
    assert outcome.warnings == []


def test_rate_limit_degrades_gracefully(monkeypatch) -> None:
    """A throttled search returns a warning, never an exception."""

    def boom(query: str, max_results: int):
        raise RuntimeError("Ratelimit: 202 from duckduckgo")

    monkeypatch.setattr(search_module, "_raw_search", boom)

    outcome = gather_evidence(["anything"], results_per_query=3)

    assert outcome.evidence == []
    assert len(outcome.warnings) == 1
    assert "throttled" in outcome.warnings[0]


def test_snippets_are_truncated(monkeypatch) -> None:
    monkeypatch.setattr(
        search_module,
        "_raw_search",
        lambda query, max_results: [
            {"title": "T", "href": "https://x.example", "body": "x" * 5000}
        ],
    )

    outcome = gather_evidence(["q"], snippet_chars=100)

    assert len(outcome.evidence[0].snippet) <= 101  # 100 chars + ellipsis


def test_max_evidence_is_respected(monkeypatch) -> None:
    monkeypatch.setattr(
        search_module,
        "_raw_search",
        lambda query, max_results: [
            {"title": f"T{i}", "href": f"https://x{i}.example", "body": f"body {i}"}
            for i in range(10)
        ],
    )

    outcome = gather_evidence(["q1", "q2"], results_per_query=10, max_evidence=4)

    assert len(outcome.evidence) == 4
