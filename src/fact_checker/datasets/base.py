"""Shared types and sampling utilities for benchmark corpora.

Both FEVER and AVeriTeC are normalised onto :class:`BenchmarkClaim` so the
evaluation harness is dataset-agnostic: it only ever reads ``id``, ``claim``,
and ``label``, while each loader keeps its own corpus-specific metadata.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Sequence

from pydantic import BaseModel, Field

from ..schemas import VERDICT_VALUES, Verdict

logger = logging.getLogger(__name__)


class BenchmarkClaim(BaseModel):
    """One labelled claim, normalised to this project's three-way label set."""

    id: str = Field(description="Identifier, unique within its dataset.")
    claim: str = Field(description="The claim text as presented to the system.")
    label: Verdict = Field(description="Gold label, normalised to our verdicts.")
    dataset: str = Field(default="", description="Source corpus name.")
    split: str = Field(
        default="",
        description=(
            "Official split this claim came from. Recorded because a "
            "label-balanced sample larger than the development split's "
            "minority class has to be topped up from the training split, "
            "and a reader cannot interpret such a sample without knowing "
            "which claims came from where."
        ),
    )


def sample_balanced(
    claims: Sequence[BenchmarkClaim],
    per_label: int,
    *,
    seed: int = 20240101,
) -> List[BenchmarkClaim]:
    """Draw a label-balanced, reproducible sample.

    Stratification matters at the sample sizes this harness actually runs: an
    unstratified draw of 30 claims fluctuates enough to skew accuracy by
    chance, even from a corpus that is balanced at full size.

    Args:
        claims: Population to sample from.
        per_label: How many claims to draw per label.
        seed: RNG seed, so a given (population, per_label, seed) is repeatable.

    Returns:
        Up to ``per_label * 3`` claims, shuffled so labels are interleaved.
    """
    buckets: Dict[str, List[BenchmarkClaim]] = {value: [] for value in VERDICT_VALUES}
    for claim in claims:
        buckets[claim.label].append(claim)

    rng = random.Random(seed)
    selected: List[BenchmarkClaim] = []
    for label in VERDICT_VALUES:
        pool = buckets[label]
        if len(pool) <= per_label:
            if pool and len(pool) < per_label:
                logger.warning(
                    "Only %d claims available for %s (requested %d)",
                    len(pool),
                    label,
                    per_label,
                )
            selected.extend(pool)
        else:
            selected.extend(rng.sample(pool, per_label))

    # Interleave so a partial/interrupted run still covers all three labels.
    rng.shuffle(selected)
    return selected


def label_distribution(claims: Sequence[BenchmarkClaim]) -> Dict[str, int]:
    """Count claims per gold label, including labels with zero instances."""
    counts = {value: 0 for value in VERDICT_VALUES}
    for claim in claims:
        counts[claim.label] += 1
    return counts
