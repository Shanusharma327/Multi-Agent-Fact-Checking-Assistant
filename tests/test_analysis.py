"""Tests for the dataset analysis that produces REPORT.md Tables 3-8.

These guard the arithmetic behind published numbers, so a silent regression in
the stats script cannot put wrong figures into the report.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from fact_checker.datasets.base import BenchmarkClaim

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "dataset_stats", ROOT / "analysis" / "dataset_stats.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stats_mod():
    return _load_module()


@pytest.fixture
def claims() -> list[BenchmarkClaim]:
    # Lengths: 1, 2, 3 words for SUPPORTS; 4, 5 for REFUTES; 6 for NEI.
    return [
        BenchmarkClaim(id="1", claim="a", label="SUPPORTS"),
        BenchmarkClaim(id="2", claim="a b", label="SUPPORTS"),
        BenchmarkClaim(id="3", claim="a b c", label="SUPPORTS"),
        BenchmarkClaim(id="4", claim="a b c d", label="REFUTES"),
        BenchmarkClaim(id="5", claim="a b c d e", label="REFUTES"),
        BenchmarkClaim(id="6", claim="a b c d e f", label="NOT_ENOUGH_INFO"),
    ]


class TestCommonStats:
    def test_counts_and_shares(self, stats_mod, claims) -> None:
        stats = stats_mod.compute_common(claims)
        assert stats["total"] == 6
        assert stats["label_counts"] == {
            "SUPPORTS": 3,
            "REFUTES": 2,
            "NOT_ENOUGH_INFO": 1,
        }
        assert stats["label_pct"]["SUPPORTS"] == pytest.approx(50.0)
        assert stats["label_pct"]["NOT_ENOUGH_INFO"] == pytest.approx(100 / 6)

    def test_claim_length_statistics(self, stats_mod, claims) -> None:
        words = stats_mod.compute_common(claims)["claim_words"]
        assert words["mean"] == pytest.approx(3.5)     # (1+2+3+4+5+6)/6
        assert words["median"] == pytest.approx(3.5)
        assert words["min"] == 1
        assert words["max"] == 6

    def test_length_is_broken_out_per_label(self, stats_mod, claims) -> None:
        by_label = stats_mod.compute_common(claims)["words_by_label"]
        assert by_label["SUPPORTS"]["mean"] == pytest.approx(2.0)   # 1,2,3
        assert by_label["REFUTES"]["mean"] == pytest.approx(4.5)    # 4,5
        assert by_label["NOT_ENOUGH_INFO"]["mean"] == pytest.approx(6.0)

    def test_shares_sum_to_one_hundred(self, stats_mod, claims) -> None:
        stats = stats_mod.compute_common(claims)
        assert sum(stats["label_pct"].values()) == pytest.approx(100.0)


class TestAVeriTeCStats:
    def test_composition_is_summarised(self, stats_mod) -> None:
        from fact_checker.datasets.averitec import AVeriTeCClaim

        claims = [
            AVeriTeCClaim(
                id="a1", claim="one two", label="REFUTES", dataset="averitec",
                claim_date="31-10-2020", reporting_source="Facebook",
                claim_types=["Numerical Claim"], n_questions=2, n_answers=3,
            ),
            AVeriTeCClaim(
                id="a2", claim="one two three", label="SUPPORTS", dataset="averitec",
                claim_date="01-01-2021", reporting_source="Facebook",
                claim_types=["Numerical Claim", "Causal Claim"],
                n_questions=4, n_answers=4,
            ),
        ]
        stats = stats_mod.compute_averitec(claims, {"total": 3, "kept": 2, "conflicting": 1})

        assert stats["claim_types"]["Numerical Claim"] == 2
        assert stats["claim_types"]["Causal Claim"] == 1
        assert stats["reporting_sources"]["Facebook"] == 2
        assert stats["claim_years"] == {"2020": 1, "2021": 1}
        assert stats["questions_per_claim"]["mean"] == pytest.approx(3.0)
        assert stats["answers_per_claim"]["mean"] == pytest.approx(3.5)
        assert stats["load_report"]["conflicting"] == 1

    def test_claim_types_are_ordered_most_common_first(self, stats_mod) -> None:
        from fact_checker.datasets.averitec import AVeriTeCClaim

        claims = [
            AVeriTeCClaim(id=f"a{i}", claim="x", label="REFUTES",
                          claim_types=["Common"] if i < 3 else ["Rare"])
            for i in range(4)
        ]
        stats = stats_mod.compute_averitec(claims, {"total": 4, "kept": 4})
        assert list(stats["claim_types"]) == ["Common", "Rare"]


class TestFeverStats:
    def test_gold_page_and_verifiable_breakdown(self, stats_mod) -> None:
        from fact_checker.datasets.fever import FeverClaim

        claims = [
            FeverClaim(id="1", claim="a b", label="SUPPORTS",
                       verifiable="VERIFIABLE", gold_pages=["P1"]),
            FeverClaim(id="2", claim="a b", label="REFUTES",
                       verifiable="VERIFIABLE", gold_pages=["P1", "P2"]),
            FeverClaim(id="3", claim="a b", label="NOT_ENOUGH_INFO",
                       verifiable="NOT VERIFIABLE", gold_pages=[]),
        ]
        stats = stats_mod.compute_fever(claims)

        assert stats["verifiable"] == {"VERIFIABLE": 2, "NOT VERIFIABLE": 1}
        assert stats["gold_pages_by_label"]["SUPPORTS"]["mean"] == pytest.approx(1.0)
        assert stats["gold_pages_by_label"]["REFUTES"]["mean"] == pytest.approx(2.0)
        # The defining property of FEVER NEI: no annotated evidence at all.
        assert stats["gold_pages_by_label"]["NOT_ENOUGH_INFO"]["zero_pct"] == 100.0
