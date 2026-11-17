"""Tests for the residual-leakage diagnostic.

The default leakage filter removes a claim's own originating fact-check.
It does not remove a *different* organisation's fact-check of the same
claim, which states the same verdict. Measuring that residual is what keeps
the control's limits visible rather than implied.
"""

from __future__ import annotations

from fact_checker.retrieval_metrics import fact_check_domains, leakage_exposure


def record(fact_check_url, retrieved, warnings=()):
    return {
        "fact_check_url": fact_check_url,
        "multi": {
            "verdict": "SUPPORTS",
            "retrieved_urls": list(retrieved),
            "warnings": list(warnings),
        },
    }


class TestFactCheckDomains:
    def test_domains_are_derived_from_the_corpus(self):
        rows = [
            record("https://www.politifact.com/a", []),
            record("https://leadstories.com/b", []),
            record("", []),
        ]
        assert fact_check_domains(rows) == {"politifact.com", "leadstories.com"}

    def test_an_archived_fact_check_resolves_to_its_real_domain(self):
        rows = [record("https://web.archive.org/web/2020/https://snopes.com/x", [])]
        assert fact_check_domains(rows) == {"snopes.com"}


class TestLeakageExposure:
    def test_a_sibling_fact_check_counts_as_exposure(self):
        """A different checker's page on the same claim states the verdict too."""
        rows = [
            record("https://leadstories.com/a", ["https://politifact.com/other"]),
            record("https://politifact.com/b", ["https://bbc.com/news"]),
        ]
        exposure = leakage_exposure(rows)
        assert exposure["n_claims"] == 2
        assert exposure["n_exposed"] == 1
        assert exposure["rate"] == 0.5

    def test_clean_retrieval_reports_no_exposure(self):
        rows = [record("https://politifact.com/a", ["https://bbc.com/1", "https://cdc.gov/2"])]
        assert leakage_exposure(rows)["n_exposed"] == 0

    def test_filter_firings_are_counted_separately_from_exposure(self):
        rows = [
            record(
                "https://politifact.com/a",
                ["https://bbc.com/1"],
                warnings=["excluded 1 result(s) naming the originating fact-check"],
            )
        ]
        exposure = leakage_exposure(rows)
        assert exposure["n_filtered_claims"] == 1
        assert exposure["n_exposed"] == 0

    def test_records_without_retrieval_are_skipped(self):
        assert leakage_exposure([{"multi": {"verdict": "SUPPORTS"}}])["n_claims"] == 0

    def test_empty_input_is_safe(self):
        assert leakage_exposure([])["rate"] == 0.0
