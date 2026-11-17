"""Tests for timeout-aware backoff and the search circuit breaker.

A run once looped hundreds of claims against a search host that had
stopped answering, filling a cache with empty evidence that was
indistinguishable from a genuine retrieval result. The breaker exists so
an outage halts the run and says so, rather than being recorded as data.
"""

from __future__ import annotations

import pytest

from fact_checker import search as search_module
from fact_checker.search import (
    SearchUnavailable,
    _BACKOFF_SECONDS,
    _looks_rate_limited,
    breaker_state,
    gather_evidence,
    reset_breaker,
)


@pytest.fixture(autouse=True)
def clean_breaker():
    """Each test starts from a closed breaker and leaves one behind."""
    reset_breaker()
    yield
    reset_breaker()


class TestRateLimitDetection:
    @pytest.mark.parametrize(
        "message",
        [
            "Ratelimit: 202 from duckduckgo",
            "429 too many requests",
            "operation timed out",
            "error sending request: timed out",
            "connection reset by peer",
        ],
    )
    def test_pushback_is_recognised(self, message):
        """A timeout is how this host throttles, so it counts as throttling."""
        assert _looks_rate_limited(RuntimeError(message))

    def test_an_ordinary_fault_is_not(self):
        assert not _looks_rate_limited(RuntimeError("no results parsed"))


class TestBackoff:
    def test_the_schedule_is_longer_than_it_was(self):
        """The short schedule turned a transient throttle into a stall."""
        assert _BACKOFF_SECONDS[0] >= 4.0
        assert max(_BACKOFF_SECONDS) >= 30.0

    def test_the_schedule_increases(self):
        assert list(_BACKOFF_SECONDS) == sorted(_BACKOFF_SECONDS)


class TestBreaker:
    def test_consecutive_failures_eventually_halt_the_run(self, monkeypatch):
        def always_fail(query, max_results):
            raise RuntimeError("operation timed out")

        monkeypatch.setattr(search_module, "_raw_search", always_fail)
        queries = [f"q{i}" for i in range(40)]

        with pytest.raises(SearchUnavailable, match="consecutive search failures"):
            gather_evidence(queries, results_per_query=2, max_evidence=50)

    def test_the_halt_says_to_pause_rather_than_retry(self, monkeypatch):
        def always_fail(query, max_results):
            raise RuntimeError("operation timed out")

        monkeypatch.setattr(search_module, "_raw_search", always_fail)
        with pytest.raises(SearchUnavailable, match="Pause the run"):
            gather_evidence([f"q{i}" for i in range(40)], max_evidence=50)

    def test_a_success_closes_the_breaker(self, monkeypatch):
        calls = {"n": 0}

        def flaky(query, max_results):
            calls["n"] += 1
            if calls["n"] % 2:
                raise RuntimeError("operation timed out")
            return [{"title": "t", "href": "https://a.example/1", "body": "text"}]

        monkeypatch.setattr(search_module, "_raw_search", flaky)
        # Alternating failure and success must never trip the breaker.
        outcome = gather_evidence([f"q{i}" for i in range(20)], max_evidence=50)
        assert outcome.evidence
        assert breaker_state() == 0

    def test_a_few_failures_degrade_gracefully_as_before(self, monkeypatch):
        """Below the threshold the old contract holds: warn, do not raise."""
        def always_fail(query, max_results):
            raise RuntimeError("operation timed out")

        monkeypatch.setattr(search_module, "_raw_search", always_fail)
        outcome = gather_evidence(["q1", "q2"], results_per_query=2)
        assert outcome.evidence == []
        assert len(outcome.warnings) == 2

    def test_reset_clears_the_count(self, monkeypatch):
        def always_fail(query, max_results):
            raise RuntimeError("operation timed out")

        monkeypatch.setattr(search_module, "_raw_search", always_fail)
        gather_evidence(["q1", "q2"], results_per_query=2)
        assert breaker_state() > 0
        reset_breaker()
        assert breaker_state() == 0
