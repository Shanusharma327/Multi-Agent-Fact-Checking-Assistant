"""Benchmark datasets, behind one dataset-agnostic interface.

Two corpora are supported, and they serve different purposes:

* **FEVER** [Thorne et al., 2018] provides *comparability*. It is the canonical
  benchmark, so its numbers situate this work against prior art.
* **AVeriTeC** [Schlichtkrull et al., 2023] provides *validity*. Its claims are
  real-world and open-web, which is what this system actually targets, and it
  post-dates FEVER by five years so contamination is less likely.

They disagree by construction, and that disagreement is informative rather than
a problem to be smoothed over -- see ``REPORT.md`` §7.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from .averitec import (
    AVeriTeCClaim,
    AVeriTeCDownloadError,
    download_averitec,
    load_averitec,
)
from .base import BenchmarkClaim, label_distribution, sample_balanced
from .fever import (
    FeverClaim,
    FeverDownloadError,
    download_fever,
    iter_fever,
    load_fever,
)

DATASETS: Tuple[str, ...] = ("fever", "averitec")

DEFAULT_SPLITS: Dict[str, str] = {"fever": "dev", "averitec": "dev"}

DEFAULT_CACHE_DIRS: Dict[str, str] = {
    "fever": "data/fever",
    "averitec": "data/averitec",
}

SPLIT_CHOICES: Dict[str, Tuple[str, ...]] = {
    "fever": ("dev", "train", "paper_dev", "paper_test"),
    "averitec": ("dev", "train", "test"),
}


class UnknownDatasetError(ValueError):
    """Raised when a dataset name is not recognised."""


def load_benchmark(
    dataset: str,
    split: str = "",
    cache_dir: str = "",
    *,
    download: bool = True,
) -> Tuple[List[BenchmarkClaim], Dict[str, int]]:
    """Load any supported benchmark, normalised to the three-way label set.

    Args:
        dataset: ``"fever"`` or ``"averitec"``.
        split: Split name; defaults to that dataset's dev split.
        cache_dir: Local cache directory; defaults per dataset.
        download: Fetch the split if it is not already cached.

    Returns:
        ``(claims, report)``. ``report`` always carries ``total`` and ``kept``,
        and AVeriTeC additionally reports how many claims were excluded for
        carrying its fourth label.

    Raises:
        UnknownDatasetError: If ``dataset`` is not recognised.
    """
    name = (dataset or "").strip().lower()
    if name not in DATASETS:
        raise UnknownDatasetError(
            f"Unknown dataset {dataset!r}. Choose from: {', '.join(DATASETS)}"
        )

    split = split or DEFAULT_SPLITS[name]
    cache_dir = cache_dir or DEFAULT_CACHE_DIRS[name]

    if name == "averitec":
        return load_averitec(split, cache_dir, download=download)

    claims = load_fever(split, cache_dir, download=download)
    return list(claims), {"total": len(claims), "kept": len(claims)}


def download_benchmark(dataset: str, split: str = "", cache_dir: str = ""):
    """Download a benchmark split and return the cached path."""
    name = (dataset or "").strip().lower()
    if name not in DATASETS:
        raise UnknownDatasetError(
            f"Unknown dataset {dataset!r}. Choose from: {', '.join(DATASETS)}"
        )
    split = split or DEFAULT_SPLITS[name]
    cache_dir = cache_dir or DEFAULT_CACHE_DIRS[name]
    if name == "averitec":
        return download_averitec(split, cache_dir)
    return download_fever(split, cache_dir)


__all__ = [
    "DATASETS",
    "DEFAULT_SPLITS",
    "DEFAULT_CACHE_DIRS",
    "SPLIT_CHOICES",
    "UnknownDatasetError",
    "load_benchmark",
    "download_benchmark",
    "BenchmarkClaim",
    "FeverClaim",
    "FeverDownloadError",
    "AVeriTeCClaim",
    "AVeriTeCDownloadError",
    "download_fever",
    "download_averitec",
    "iter_fever",
    "load_fever",
    "load_averitec",
    "sample_balanced",
    "label_distribution",
]
