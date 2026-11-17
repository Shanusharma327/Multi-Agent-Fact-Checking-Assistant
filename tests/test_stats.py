"""Tests for the bootstrap, exact tests and multiplicity correction.

These are the numbers that decide which claims the paper may make, so the
properties pinned here are the ones a wrong result would quietly violate:
that a known interval is recovered, that a paired test uses only discordant
pairs, that Holm is monotone, and that the power calculation moves the
right way with n.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "analysis"))

from stats import (  # noqa: E402
    Interval,
    PairedTest,
    apply_holm,
    bootstrap_metric,
    exact_binomial_two_sided,
    holm_correct,
    mcnemar,
    minimum_detectable_difference,
    observed_discordance,
    paired_bootstrap_difference,
)

MEAN = lambda rows: sum(rows) / len(rows) if rows else 0.0  # noqa: E731


class TestBootstrap:
    def test_the_point_estimate_is_the_observed_value(self):
        rows = [1.0] * 60 + [0.0] * 40
        assert bootstrap_metric(rows, MEAN, resamples=500).estimate == pytest.approx(0.6)

    def test_the_interval_brackets_the_estimate(self):
        rows = [1.0] * 60 + [0.0] * 40
        interval = bootstrap_metric(rows, MEAN, resamples=2000)
        assert interval.low <= interval.estimate <= interval.high

    def test_a_larger_sample_gives_a_tighter_interval(self):
        small = bootstrap_metric([1.0] * 30 + [0.0] * 30, MEAN, resamples=2000)
        large = bootstrap_metric([1.0] * 300 + [0.0] * 300, MEAN, resamples=2000)
        assert (large.high - large.low) < (small.high - small.low)

    def test_a_constant_sample_has_a_degenerate_interval(self):
        interval = bootstrap_metric([1.0] * 50, MEAN, resamples=500)
        assert interval.low == interval.high == 1.0

    def test_the_same_seed_reproduces_the_interval(self):
        rows = [1.0] * 40 + [0.0] * 60
        first = bootstrap_metric(rows, MEAN, resamples=1000, seed=7)
        second = bootstrap_metric(rows, MEAN, resamples=1000, seed=7)
        assert (first.low, first.high) == (second.low, second.high)

    def test_a_single_row_cannot_be_resampled(self):
        assert bootstrap_metric([1.0], MEAN, resamples=100).resamples == 0

    def test_an_empty_sample_is_safe(self):
        assert bootstrap_metric([], MEAN).estimate == 0.0


class TestPairedBootstrap:
    def test_a_known_difference_is_recovered(self):
        # A is right on 70, B on 50, of 100 paired claims.
        pairs = [(1.0, 1.0)] * 50 + [(1.0, 0.0)] * 20 + [(0.0, 0.0)] * 30
        difference = paired_bootstrap_difference(
            pairs,
            lambda s: sum(a for a, _ in s) / len(s),
            lambda s: sum(b for _, b in s) / len(s),
            resamples=2000,
        )
        assert difference.estimate == pytest.approx(0.2)
        assert difference.excludes_zero

    def test_no_difference_gives_an_interval_spanning_zero(self):
        pairs = [(1.0, 1.0)] * 50 + [(0.0, 0.0)] * 50
        difference = paired_bootstrap_difference(
            pairs,
            lambda s: sum(a for a, _ in s) / len(s),
            lambda s: sum(b for _, b in s) / len(s),
            resamples=1000,
        )
        assert difference.estimate == 0.0
        assert not difference.excludes_zero

    def test_pairing_is_preserved_under_resampling(self):
        """Perfectly correlated conditions must show a zero-width interval."""
        pairs = [(float(i % 2), float(i % 2)) for i in range(100)]
        difference = paired_bootstrap_difference(
            pairs,
            lambda s: sum(a for a, _ in s) / len(s),
            lambda s: sum(b for _, b in s) / len(s),
            resamples=1000,
        )
        assert difference.low == difference.high == 0.0


class TestExactBinomial:
    def test_a_fair_split_is_not_significant(self):
        assert exact_binomial_two_sided(5, 10) == pytest.approx(1.0)

    def test_a_lopsided_split_is(self):
        assert exact_binomial_two_sided(10, 12) < 0.05

    def test_an_extreme_split_is_strongly_so(self):
        assert exact_binomial_two_sided(20, 20) < 1e-5

    def test_zero_trials_is_safe(self):
        assert exact_binomial_two_sided(0, 0) == 1.0

    def test_the_result_is_symmetric(self):
        assert exact_binomial_two_sided(3, 12) == pytest.approx(
            exact_binomial_two_sided(9, 12)
        )


class TestMcNemar:
    def test_only_discordant_pairs_carry_information(self):
        """Adding claims both conditions get right must not change p."""
        a = [True] * 10 + [False] * 2
        b = [False] * 10 + [True] * 2
        p_small, _, _ = mcnemar(a, b)
        p_large, _, _ = mcnemar(a + [True] * 500, b + [True] * 500)
        assert p_small == pytest.approx(p_large)

    def test_discordant_counts_are_reported(self):
        a = [True] * 7 + [False] * 3
        b = [False] * 7 + [True] * 3
        p, a_only, b_only = mcnemar(a, b)
        assert (a_only, b_only) == (7, 3)

    def test_identical_conditions_are_not_significant(self):
        p, a_only, b_only = mcnemar([True, False] * 10, [True, False] * 10)
        assert p == 1.0
        assert (a_only, b_only) == (0, 0)

    def test_misaligned_sequences_are_rejected(self):
        with pytest.raises(ValueError, match="align"):
            mcnemar([True], [True, False])


class TestHolm:
    def test_the_smallest_p_takes_the_largest_multiplier(self):
        assert holm_correct([0.01, 0.04, 0.03])[0] == pytest.approx(0.03)

    def test_adjusted_values_are_monotone_in_the_raw_order(self):
        adjusted = holm_correct([0.001, 0.01, 0.02, 0.04])
        ordered = [adjusted[i] for i in sorted(range(4), key=lambda i: [0.001, 0.01, 0.02, 0.04][i])]
        assert ordered == sorted(ordered)

    def test_adjustment_never_decreases_a_p_value(self):
        raw = [0.01, 0.2, 0.5]
        assert all(a >= r for a, r in zip(holm_correct(raw), raw))

    def test_values_are_capped_at_one(self):
        assert all(p <= 1.0 for p in holm_correct([0.6, 0.7, 0.8]))

    def test_holm_is_no_more_conservative_than_bonferroni(self):
        raw = [0.01, 0.02, 0.03]
        bonferroni = [min(1.0, 3 * p) for p in raw]
        assert all(h <= b for h, b in zip(holm_correct(raw), bonferroni))

    def test_an_empty_family_is_safe(self):
        assert holm_correct([]) == []

    def test_apply_holm_attaches_to_tests(self):
        tests = [
            PairedTest(
                name=f"c{i}", reference="B1", n=10,
                estimate_a=0.5, estimate_b=0.4,
                difference=Interval(estimate=0.1, low=0.0, high=0.2),
                p_value=p,
            )
            for i, p in enumerate([0.01, 0.04])
        ]
        adjusted = apply_holm(tests)
        assert adjusted[0].p_adjusted == pytest.approx(0.02)
        assert all(not math.isnan(t.p_adjusted) for t in adjusted)


class TestPower:
    def test_a_larger_sample_detects_a_smaller_difference(self):
        small = minimum_detectable_difference(60, discordance=0.35)
        large = minimum_detectable_difference(400, discordance=0.35)
        assert large["mdd"] < small["mdd"]

    def test_the_detectable_difference_scales_as_one_over_root_n(self):
        a = minimum_detectable_difference(100, discordance=0.35)["mdd"]
        b = minimum_detectable_difference(400, discordance=0.35)["mdd"]
        assert a / b == pytest.approx(2.0, rel=0.01)

    def test_higher_discordance_needs_a_larger_difference(self):
        low = minimum_detectable_difference(200, discordance=0.1)
        high = minimum_detectable_difference(200, discordance=0.5)
        assert high["mdd"] > low["mdd"]

    def test_zero_claims_detects_nothing(self):
        assert minimum_detectable_difference(0)["mdd"] == 1.0

    def test_observed_discordance_counts_disagreements(self):
        a = [True, True, False, False]
        b = [True, False, True, False]
        assert observed_discordance(a, b) == 0.5

    def test_observed_discordance_of_identical_conditions_is_zero(self):
        assert observed_discordance([True, False], [True, False]) == 0.0


class TestInterval:
    def test_an_interval_above_zero_excludes_it(self):
        assert Interval(estimate=0.1, low=0.02, high=0.18).excludes_zero

    def test_an_interval_below_zero_also_excludes_it(self):
        assert Interval(estimate=-0.1, low=-0.2, high=-0.01).excludes_zero

    def test_an_interval_spanning_zero_does_not(self):
        assert not Interval(estimate=0.01, low=-0.05, high=0.07).excludes_zero

    def test_rendering_scales_to_percentage_points(self):
        rendered = Interval(estimate=0.123, low=0.1, high=0.15).render(scale=100.0)
        assert rendered == "12.3 [10.0, 15.0]"
