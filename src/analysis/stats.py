"""Uncertainty quantification for the condition comparisons.

The reviewer's first objection was that headline gains of a few percentage
points on n=60 correspond to roughly two claims changing label, with no
interval or test attached. Nothing in this module makes the samples larger;
it makes the uncertainty visible, so a difference that the data cannot
support is reported as such rather than asserted.

Three choices worth stating:

* **Paired throughout.** Every condition is evaluated on the same claims
  with the same cached evidence, so the comparisons are paired. A paired
  bootstrap resamples *claims* and recomputes both conditions on the
  resample, which preserves the correlation between them and gives a far
  tighter interval than treating the two as independent samples.
* **McNemar for accuracy.** For a paired binary outcome the informative
  quantity is the discordant pairs -- claims where exactly one condition is
  right. The exact binomial form is used rather than the chi-square
  approximation, because discordant counts here are small.
* **Holm for multiplicity.** Several conditions are compared against the
  same reference, so the family-wise error rate needs controlling. Holm is
  uniformly more powerful than Bonferroni and makes no independence
  assumption, which matters for comparisons sharing a reference arm.

Implemented with the standard library plus ``statistics``; no SciPy, so the
project keeps its existing dependency set.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Sequence, Tuple

from pydantic import BaseModel, Field

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20240101


class Interval(BaseModel):
    """A point estimate with a bootstrap percentile interval."""

    estimate: float
    low: float
    high: float
    level: float = 0.95
    resamples: int = DEFAULT_RESAMPLES

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies wholly on one side of zero."""
        return self.low > 0.0 or self.high < 0.0

    def render(self, places: int = 1, scale: float = 1.0) -> str:
        """Format as ``estimate [low, high]`` for a table cell."""
        return (
            f"{self.estimate * scale:.{places}f} "
            f"[{self.low * scale:.{places}f}, {self.high * scale:.{places}f}]"
        )


class PairedTest(BaseModel):
    """A paired difference between two conditions on the same claims."""

    name: str
    reference: str
    n: int
    estimate_a: float
    estimate_b: float
    difference: Interval
    p_value: float = Field(description="Uncorrected two-sided p-value.")
    p_adjusted: float = Field(default=float("nan"), description="Holm-corrected.")
    test: str = Field(default="", description="Which test produced p_value.")
    n_discordant: int = Field(default=0, description="Claims where exactly one is right.")

    @property
    def significant(self) -> bool:
        """Whether the Holm-corrected p-value clears 0.05."""
        p = self.p_adjusted if not math.isnan(self.p_adjusted) else self.p_value
        return p < 0.05

    @property
    def inconclusive(self) -> bool:
        """Whether the interval for the difference includes zero."""
        return not self.difference.excludes_zero


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile of a sorted-able sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_metric(
    rows: Sequence,
    metric: Callable[[Sequence], float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Percentile bootstrap interval for ``metric`` over ``rows``.

    Args:
        rows: One element per claim. Resampling is over claims, which is the
            unit the sample was drawn in.
        metric: Maps a resample to a scalar.
        resamples: Bootstrap replicates.
        level: Interval coverage.
        seed: RNG seed, so an interval is reproducible.

    Returns:
        The point estimate on the observed data, with its interval.
    """
    rows = list(rows)
    estimate = metric(rows) if rows else 0.0
    if len(rows) < 2:
        return Interval(estimate=estimate, low=estimate, high=estimate,
                        level=level, resamples=0)

    rng = random.Random(seed)
    n = len(rows)
    replicates: List[float] = []
    for _ in range(resamples):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        replicates.append(metric(sample))

    tail = (1.0 - level) / 2.0
    return Interval(
        estimate=estimate,
        low=_percentile(replicates, tail),
        high=_percentile(replicates, 1.0 - tail),
        level=level,
        resamples=resamples,
    )


def paired_bootstrap_difference(
    rows: Sequence,
    metric_a: Callable[[Sequence], float],
    metric_b: Callable[[Sequence], float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Bootstrap interval for ``metric_a - metric_b`` on paired rows.

    Both metrics are recomputed on the *same* resample, so the correlation
    between the conditions is preserved. Resampling them independently would
    inflate the interval and understate a real difference.
    """
    rows = list(rows)
    estimate = (metric_a(rows) - metric_b(rows)) if rows else 0.0
    if len(rows) < 2:
        return Interval(estimate=estimate, low=estimate, high=estimate,
                        level=level, resamples=0)

    rng = random.Random(seed)
    n = len(rows)
    replicates: List[float] = []
    for _ in range(resamples):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        replicates.append(metric_a(sample) - metric_b(sample))

    tail = (1.0 - level) / 2.0
    return Interval(
        estimate=estimate,
        low=_percentile(replicates, tail),
        high=_percentile(replicates, 1.0 - tail),
        level=level,
        resamples=resamples,
    )


# ---------------------------------------------------------------------------
# Exact tests
# ---------------------------------------------------------------------------

def _binomial_pmf(k: int, n: int, p: float = 0.5) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def exact_binomial_two_sided(successes: int, trials: int, p: float = 0.5) -> float:
    """Two-sided exact binomial p-value by the method of small p-values.

    Sums the probability of every outcome no more likely than the observed
    one. Exact rather than normal-approximate because the discordant counts
    this is applied to are routinely under 30.
    """
    if trials <= 0:
        return 1.0
    observed = _binomial_pmf(successes, trials, p)
    # Floating-point slack, so an outcome equally likely by symmetry is not
    # excluded by a last-bit difference.
    tolerance = observed * (1 + 1e-9)
    return min(1.0, sum(
        _binomial_pmf(k, trials, p)
        for k in range(trials + 1)
        if _binomial_pmf(k, trials, p) <= tolerance
    ))


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Tuple[float, int, int]:
    """Exact McNemar test on paired correctness.

    Args:
        correct_a: Whether condition A was right, per claim.
        correct_b: Whether condition B was right, on the same claims.

    Returns:
        ``(p_value, n_a_only, n_b_only)``. Concordant pairs carry no
        information about which condition is better and are excluded, which
        is the whole point of the test.

    Raises:
        ValueError: If the sequences differ in length.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"paired sequences must align (got {len(correct_a)}, {len(correct_b)})"
        )
    a_only = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
    b_only = sum(1 for a, b in zip(correct_a, correct_b) if b and not a)
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0, 0, 0
    return exact_binomial_two_sided(a_only, discordant), a_only, b_only


def holm_correct(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjustment, preserving input order.

    Uniformly more powerful than Bonferroni and valid without an
    independence assumption, which matters because these comparisons share
    a reference arm and are therefore correlated.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        # Step-down enforces monotonicity: an adjusted p-value may never
        # fall below one assigned to a smaller raw p-value.
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def apply_holm(tests: Sequence[PairedTest]) -> List[PairedTest]:
    """Attach Holm-corrected p-values across a family of comparisons."""
    adjusted = holm_correct([t.p_value for t in tests])
    return [t.model_copy(update={"p_adjusted": p}) for t, p in zip(tests, adjusted)]


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

def minimum_detectable_difference(
    n: int,
    *,
    baseline_accuracy: float = 0.5,
    discordance: float = 0.35,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, float]:
    """The smallest paired accuracy difference this sample could detect.

    For McNemar the relevant quantity is the discordant rate: only claims
    where exactly one condition is right carry information. With ``n``
    claims and a discordant proportion ``d``, roughly ``n*d`` informative
    pairs remain, and the detectable difference scales as
    ``(z_alpha/2 + z_beta) * sqrt(d/n)``.

    Args:
        n: Claims evaluated.
        baseline_accuracy: Reported for context only.
        discordance: Expected share of claims where the two conditions
            disagree in correctness. Measure it from data where possible;
            the default is a middling value for a three-way task.
        alpha: Two-sided significance level.
        power: Target power.

    Returns:
        The minimum detectable difference in accuracy, with its inputs.
    """
    if n <= 0:
        return {"n": 0, "mdd": 1.0, "discordance": discordance,
                "alpha": alpha, "power": power, "informative_pairs": 0}

    # Normal quantiles for the conventional levels, hardcoded to avoid a
    # SciPy dependency for two constants.
    z_alpha = {0.05: 1.959964, 0.01: 2.575829, 0.10: 1.644854}.get(alpha, 1.959964)
    z_beta = {0.80: 0.841621, 0.90: 1.281552, 0.95: 1.644854}.get(power, 0.841621)

    mdd = (z_alpha + z_beta) * math.sqrt(discordance / n)
    return {
        "n": n,
        "mdd": min(1.0, mdd),
        "discordance": discordance,
        "informative_pairs": n * discordance,
        "baseline_accuracy": baseline_accuracy,
        "alpha": alpha,
        "power": power,
    }


def observed_discordance(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> float:
    """Share of claims on which exactly one of two conditions is correct."""
    if not correct_a:
        return 0.0
    return sum(1 for a, b in zip(correct_a, correct_b) if a != b) / len(correct_a)


__all__ = [
    "DEFAULT_RESAMPLES",
    "Interval",
    "PairedTest",
    "apply_holm",
    "bootstrap_metric",
    "exact_binomial_two_sided",
    "holm_correct",
    "mcnemar",
    "minimum_detectable_difference",
    "observed_discordance",
    "paired_bootstrap_difference",
]
