"""Tests for the windowed failure-rate rule in the search breaker.

The consecutive-failure rule did not catch the failure it was written
for. With the search host mostly refusing but occasionally answering,
every rare success reset the counter, so the breaker never tripped while
retrieval advanced by zero claims in thirty-four minutes. These tests
pin that the windowed rule catches that state, that the consecutive rule
still catches an outright outage, and that a merely flaky service is not
mistaken for a dead one.
"""

from __future__ import annotations

import pytest

from fact_checker import search as search_module
from fact_checker.search import (
    _BREAKER_THRESHOLD,
    _BREAKER_WINDOW,
    _BREAKER_WINDOW_RATE,
    SearchUnavailable,
    _Breaker,
    breaker_state,
    breaker_window_rate,
    gather_evidence,
    reset_breaker,
)


@pytest.fixture(autouse=True)
def clean_breaker():
    reset_breaker()
    yield
    reset_breaker()


class TestWindowedRule:
    def test_the_interleaved_pattern_that_defeated_the_old_rule_trips(self):
        """Mostly failing, with just enough successes to reset a counter."""
        breaker = _Breaker()
        for index in range(40):
            if index % 5 == 4:
                breaker.record_success()
            else:
                breaker.record_failure()
        assert breaker.consecutive < _BREAKER_THRESHOLD  # the old rule misses it
        assert breaker.tripped                            # the new rule does not

    def test_the_reason_names_the_windowed_rule(self):
        breaker = _Breaker()
        for index in range(40):
            breaker.record_success() if index % 5 == 4 else breaker.record_failure()
        assert "of the last" in breaker.reason()

    def test_a_flaky_but_working_service_does_not_trip(self):
        """A 20% failure rate is annoying, not an outage."""
        breaker = _Breaker()
        for index in range(60):
            breaker.record_failure() if index % 5 == 4 else breaker.record_success()
        assert not breaker.tripped

    def test_a_short_burst_of_failures_does_not_trip_on_rate(self):
        """Judging a rate before the window fills would halt healthy runs."""
        breaker = _Breaker()
        for _ in range(_BREAKER_THRESHOLD - 1):
            breaker.record_failure()
        assert len(breaker.recent) < _BREAKER_WINDOW
        assert not breaker.tripped

    def test_the_window_forgets_old_history(self):
        """Recovery must be possible without an explicit reset."""
        breaker = _Breaker()
        for _ in range(_BREAKER_WINDOW):
            breaker.record_failure()
        assert breaker.tripped
        for _ in range(_BREAKER_WINDOW):
            breaker.record_success()
        assert not breaker.tripped
        assert breaker.window_failure_rate == 0.0

    def test_the_rate_is_computed_over_the_window_not_all_history(self):
        breaker = _Breaker()
        for _ in range(100):
            breaker.record_success()
        for _ in range(_BREAKER_WINDOW):
            breaker.record_failure()
        assert breaker.window_failure_rate == 1.0


class TestConsecutiveRuleStillWorks:
    def test_an_outright_outage_trips_before_the_window_fills(self):
        breaker = _Breaker()
        for _ in range(_BREAKER_THRESHOLD):
            breaker.record_failure()
        assert breaker.tripped
        assert "consecutive" in breaker.reason()

    def test_a_success_clears_the_consecutive_count(self):
        breaker = _Breaker()
        for _ in range(_BREAKER_THRESHOLD - 1):
            breaker.record_failure()
        breaker.record_success()
        assert breaker.consecutive == 0


class TestIntegration:
    def test_an_interleaved_stream_halts_gather_evidence(self, monkeypatch):
        calls = {"n": 0}

        def mostly_failing(query, max_results):
            calls["n"] += 1
            if calls["n"] % 5 == 0:
                return [{"title": "t", "href": "https://a.example/1", "body": "text"}]
            raise RuntimeError("operation timed out")

        monkeypatch.setattr(search_module, "_raw_search", mostly_failing)
        with pytest.raises(SearchUnavailable, match="of the last"):
            gather_evidence([f"q{i}" for i in range(60)], max_evidence=500)

    def test_the_halt_says_retrieval_is_not_restarted_automatically(self, monkeypatch):
        def always_fail(query, max_results):
            raise RuntimeError("operation timed out")

        monkeypatch.setattr(search_module, "_raw_search", always_fail)
        with pytest.raises(SearchUnavailable, match="never restarted"):
            gather_evidence([f"q{i}" for i in range(40)], max_evidence=500)

    def test_reset_clears_both_signals(self, monkeypatch):
        def always_fail(query, max_results):
            raise RuntimeError("operation timed out")

        monkeypatch.setattr(search_module, "_raw_search", always_fail)
        gather_evidence(["q1", "q2"], results_per_query=2)
        assert breaker_state() > 0
        reset_breaker()
        assert breaker_state() == 0
        assert breaker_window_rate() == 0.0


class TestThresholds:
    def test_the_window_rate_demands_a_clear_majority_of_failures(self):
        assert 0.5 < _BREAKER_WINDOW_RATE <= 1.0

    def test_the_window_is_longer_than_the_consecutive_threshold(self):
        """Otherwise the rate rule could never fire before the other."""
        assert _BREAKER_WINDOW > _BREAKER_THRESHOLD
