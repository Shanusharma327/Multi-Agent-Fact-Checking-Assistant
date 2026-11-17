"""Tests for the benchmark scoring, including the calibration measures."""

from __future__ import annotations

import pytest

from fact_checker.metrics import (
    brier_score,
    calibration_bins,
    confusion_matrix,
    expected_calibration_error,
    format_confusion,
    format_scorecard,
    per_label_scores,
    score_system,
)

S, R, N = "SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"


class TestClassification:
    def test_perfect_classifier(self) -> None:
        gold = [S, R, N, S]
        card = score_system("x", gold, gold, [0.9] * 4)
        assert card.accuracy == 1.0
        assert card.macro_f1 == 1.0
        assert all(s.f1 == 1.0 for s in card.per_label)

    def test_always_wrong(self) -> None:
        card = score_system("x", [S, S], [R, R], [0.9, 0.9])
        assert card.accuracy == 0.0
        assert card.macro_f1 == 0.0

    def test_macro_f1_is_unweighted_mean(self) -> None:
        # 3 SUPPORTS all correct, 1 REFUTES wrong -> F1: S=0.857, R=0, N=0
        card = score_system("x", [S, S, S, R], [S, S, S, S], [0.5] * 4)
        f1s = {s.label: s.f1 for s in card.per_label}
        assert f1s[S] == pytest.approx(6 / 7, abs=1e-6)
        assert f1s[R] == 0.0
        assert card.macro_f1 == pytest.approx((6 / 7) / 3, abs=1e-6)

    def test_support_counts_gold_not_predictions(self) -> None:
        card = score_system("x", [S, S, R], [S, R, R], [0.5] * 3)
        support = {s.label: s.support for s in card.per_label}
        assert support[S] == 2
        assert support[R] == 1
        assert support[N] == 0

    def test_confusion_matrix_orientation(self) -> None:
        matrix = confusion_matrix([S, S, R], [S, R, R])
        assert matrix[S][S] == 1
        assert matrix[S][R] == 1  # gold SUPPORTS, predicted REFUTES
        assert matrix[R][R] == 1
        assert matrix[R][S] == 0

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            score_system("x", [S, R], [S], [0.5])

    def test_empty_inputs_do_not_divide_by_zero(self) -> None:
        card = score_system("x", [], [], [])
        assert card.accuracy == 0.0
        assert card.n == 0


class TestCalibration:
    def test_perfect_calibration_has_zero_ece(self) -> None:
        # 100% confident and always right.
        correct = [True] * 10
        confidence = [1.0] * 10
        assert expected_calibration_error(correct, confidence) == pytest.approx(0.0)

    def test_confidently_wrong_has_maximal_ece(self) -> None:
        correct = [False] * 10
        confidence = [1.0] * 10
        assert expected_calibration_error(correct, confidence) == pytest.approx(1.0)

    def test_ece_is_weighted_by_bin_population(self) -> None:
        # 9 claims perfectly calibrated at 1.0, 1 claim wrong at 1.0 -> gap 0.1
        correct = [True] * 9 + [False]
        confidence = [1.0] * 10
        assert expected_calibration_error(correct, confidence) == pytest.approx(0.1)

    def test_confidence_of_one_lands_in_final_bin(self) -> None:
        bins = calibration_bins([True], [1.0], bins=10)
        assert bins[-1].count == 1
        assert sum(b.count for b in bins) == 1

    def test_brier_rewards_honest_uncertainty(self) -> None:
        # Being wrong at 50% beats being wrong at 100%.
        assert brier_score([False], [0.5]) < brier_score([False], [1.0])
        assert brier_score([True], [1.0]) == pytest.approx(0.0)

    def test_overconfidence_is_confidence_minus_accuracy(self) -> None:
        card = score_system("x", [S, S], [S, R], [0.9, 0.9])
        assert card.accuracy == pytest.approx(0.5)
        assert card.mean_confidence == pytest.approx(0.9)
        assert card.overconfidence == pytest.approx(0.4)

    def test_split_confidence_by_correctness(self) -> None:
        card = score_system("x", [S, S], [S, R], [0.8, 0.6])
        assert card.mean_confidence_when_right == pytest.approx(0.8)
        assert card.mean_confidence_when_wrong == pytest.approx(0.6)

    def test_out_of_range_confidence_is_clamped_by_scoring(self) -> None:
        assert brier_score([True], [1.5]) == pytest.approx(0.0)
        assert brier_score([True], [-0.5]) == pytest.approx(1.0)


class TestFormatting:
    def test_scorecard_renders(self) -> None:
        card = score_system("multi-agent", [S, R, N], [S, R, S], [0.9, 0.8, 0.4], [1.0, 2.0, 3.0])
        text = format_scorecard(card)
        assert "multi-agent" in text
        assert "accuracy" in text
        assert "SUPPORTS" in text
        assert "mean latency" in text

    def test_confusion_renders_all_labels(self) -> None:
        card = score_system("x", [S, R, N], [S, R, N], [0.9] * 3)
        text = format_confusion(card)
        assert "SUP" in text and "REF" in text and "NEI" in text

    def test_per_label_covers_every_label_even_when_absent(self) -> None:
        scores = per_label_scores([S], [S])
        assert {s.label for s in scores} == {S, R, N}
