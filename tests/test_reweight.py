"""Tests for prior-shift reweighting of balanced-sample results.

Reweighting lets a balanced sample speak about a skewed deployment
distribution without discarding data. These tests pin the properties that
make it trustworthy: it is exact for a known case, it leaves a balanced
target untouched, it conserves mass, and it reports how little real data
sits behind a heavily upweighted class.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "analysis"))

from reweight import (  # noqa: E402
    AVERITEC_DEV_PRIOR,
    observed_prior,
    reweight,
)

BALANCED = {"SUPPORTS": 1 / 3, "REFUTES": 1 / 3, "NOT_ENOUGH_INFO": 1 / 3}


def rows(**per_class) -> list:
    """Build rows from ``{gold: {verdict: count}}``-style keyword args."""
    out = []
    for gold, predictions in per_class.items():
        for verdict, count in predictions.items():
            out.extend({"gold": gold, "verdict": verdict} for _ in range(count))
    return out


class TestObservedPrior:
    def test_a_balanced_sample_reports_thirds(self):
        sample = rows(
            SUPPORTS={"SUPPORTS": 10}, REFUTES={"REFUTES": 10},
            NOT_ENOUGH_INFO={"NOT_ENOUGH_INFO": 10},
        )
        assert observed_prior(sample)["SUPPORTS"] == pytest.approx(1 / 3)

    def test_an_empty_sample_is_safe(self):
        assert observed_prior([]) == {}


class TestReweight:
    def test_a_refutes_only_system_scores_the_refutes_share(self):
        """Exact case: predicting REFUTES always scores P(REFUTES)."""
        sample = rows(
            SUPPORTS={"REFUTES": 20}, REFUTES={"REFUTES": 20},
            NOT_ENOUGH_INFO={"REFUTES": 20},
        )
        score = reweight(sample, AVERITEC_DEV_PRIOR)
        assert score.accuracy == pytest.approx(AVERITEC_DEV_PRIOR["REFUTES"], abs=1e-9)

    def test_reweighting_to_the_observed_prior_changes_nothing(self):
        sample = rows(
            SUPPORTS={"SUPPORTS": 7, "REFUTES": 3},
            REFUTES={"REFUTES": 6, "SUPPORTS": 4},
            NOT_ENOUGH_INFO={"NOT_ENOUGH_INFO": 5, "REFUTES": 5},
        )
        measured = sum(1 for r in sample if r["gold"] == r["verdict"]) / len(sample)
        assert reweight(sample, BALANCED).accuracy == pytest.approx(measured)

    def test_recall_is_invariant_under_reweighting(self):
        """Recall is within-class, so a change of prior cannot move it."""
        sample = rows(
            SUPPORTS={"SUPPORTS": 6, "REFUTES": 4},
            REFUTES={"REFUTES": 8, "SUPPORTS": 2},
            NOT_ENOUGH_INFO={"NOT_ENOUGH_INFO": 5, "REFUTES": 5},
        )
        balanced = reweight(sample, BALANCED)
        skewed = reweight(sample, AVERITEC_DEV_PRIOR)
        for label in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"):
            assert skewed.per_label_recall[label] == pytest.approx(
                balanced.per_label_recall[label]
            )

    def test_precision_does_move_under_reweighting(self):
        """Precision depends on the prediction mix, which the prior changes."""
        sample = rows(
            SUPPORTS={"SUPPORTS": 10},
            REFUTES={"SUPPORTS": 10},
            NOT_ENOUGH_INFO={"NOT_ENOUGH_INFO": 10},
        )
        balanced = reweight(sample, BALANCED)
        skewed = reweight(sample, AVERITEC_DEV_PRIOR)
        assert skewed.per_label_precision["SUPPORTS"] != pytest.approx(
            balanced.per_label_precision["SUPPORTS"]
        )

    def test_mass_is_conserved(self):
        sample = rows(
            SUPPORTS={"SUPPORTS": 10}, REFUTES={"REFUTES": 10},
            NOT_ENOUGH_INFO={"NOT_ENOUGH_INFO": 10},
        )
        assert reweight(sample, AVERITEC_DEV_PRIOR).effective_n == pytest.approx(30.0)

    def test_a_perfect_system_stays_perfect(self):
        sample = rows(
            SUPPORTS={"SUPPORTS": 9}, REFUTES={"REFUTES": 9},
            NOT_ENOUGH_INFO={"NOT_ENOUGH_INFO": 9},
        )
        assert reweight(sample, AVERITEC_DEV_PRIOR).accuracy == pytest.approx(1.0)

    def test_the_smallest_real_class_is_reported(self):
        """A class scaled up from few claims must declare how few."""
        sample = rows(
            SUPPORTS={"SUPPORTS": 30}, REFUTES={"REFUTES": 30},
            NOT_ENOUGH_INFO={"NOT_ENOUGH_INFO": 3},
        )
        assert reweight(sample, AVERITEC_DEV_PRIOR).min_class_support == 3

    def test_the_observed_prior_is_carried_for_comparison(self):
        sample = rows(SUPPORTS={"SUPPORTS": 10}, REFUTES={"REFUTES": 30})
        score = reweight(sample, AVERITEC_DEV_PRIOR)
        assert score.observed_prior["SUPPORTS"] == pytest.approx(0.25)

    def test_an_empty_sample_is_safe(self):
        assert reweight([], AVERITEC_DEV_PRIOR).accuracy == 0.0

    def test_a_class_absent_from_the_sample_contributes_nothing(self):
        """Its weight is undefined, so it must not silently become 1."""
        sample = rows(SUPPORTS={"SUPPORTS": 10}, REFUTES={"REFUTES": 10})
        score = reweight(sample, AVERITEC_DEV_PRIOR)
        assert score.per_label_recall["NOT_ENOUGH_INFO"] == 0.0
        assert 0.0 <= score.accuracy <= 1.0


class TestPrior:
    def test_the_averitec_prior_sums_to_one(self):
        assert sum(AVERITEC_DEV_PRIOR.values()) == pytest.approx(1.0)

    def test_refutes_dominates_as_the_corpus_reports(self):
        assert AVERITEC_DEV_PRIOR["REFUTES"] == pytest.approx(0.66, abs=0.01)
