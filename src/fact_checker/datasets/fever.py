"""FEVER dataset loader.

FEVER (Fact Extraction and VERification, Thorne et al. 2018) provides ~185k
claims labelled SUPPORTS / REFUTES / NOT ENOUGH INFO -- the same three-way
label set this pipeline emits, which is what makes it a drop-in ground truth.

Downloads stream from fever.ai into a local cache; the HuggingFace ``datasets``
package is not required.

IMPORTANT -- what a FEVER score here does and does not mean:

FEVER's labels are defined *relative to a June 2017 Wikipedia dump*. This
pipeline retrieves from the open web instead. That mismatch is not a bug, but
it does change the interpretation:

* NOT ENOUGH INFO means "unverifiable from that Wikipedia snapshot". Live
  search can legitimately find evidence for such a claim, so the pipeline may
  answer SUPPORTS where FEVER says NEI and be *right* about the world while
  scoring as wrong. Expect depressed NEI recall.
* Claims are decontextualised mutations of Wikipedia sentences ("Tilda Swinton
  is a vegan."), often about obscure entities with ambiguous referents.
* This is therefore "FEVER-labelled claims under open-web retrieval", not
  official FEVER scoring -- the official metric additionally requires
  retrieving the gold Wikipedia sentences, which this pipeline never sees.

The comparison it *does* support cleanly is the one this project exists for:
the grounded pipeline and the ungrounded baseline face identical claims under
identical conditions.
"""

from __future__ import annotations

import json
import logging
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

from pydantic import Field

from ..schemas import Verdict  # noqa: F401  (re-exported for callers)
from .base import BenchmarkClaim, label_distribution, sample_balanced

logger = logging.getLogger(__name__)

BASE_URL = "https://fever.ai/download/fever"

# Split name -> remote filename. `dev` is the standard labelled evaluation set.
SPLIT_FILES: Dict[str, str] = {
    "dev": "shared_task_dev.jsonl",
    "train": "train.jsonl",
    "paper_dev": "paper_dev.jsonl",
    "paper_test": "paper_test.jsonl",
}

# FEVER writes the neutral label with a space; our schema uses an underscore.
_LABEL_MAP: Dict[str, str] = {
    "SUPPORTS": "SUPPORTS",
    "REFUTES": "REFUTES",
    "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
    "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
}

DEFAULT_CACHE_DIR = Path("data/fever")

_USER_AGENT = "multi-agent-fact-checker/1.0 (+evaluation harness)"


class FeverDownloadError(RuntimeError):
    """Raised when the FEVER split cannot be downloaded or read."""


class FeverClaim(BenchmarkClaim):
    """One labelled FEVER claim, normalised to this project's label set."""

    verifiable: str = Field(default="", description="FEVER's VERIFIABLE flag.")
    gold_pages: List[str] = Field(
        default_factory=list,
        description="Wikipedia page titles FEVER annotated as evidence.",
    )


def _normalise_label(raw: str) -> Optional[str]:
    """Map a FEVER label onto our verdict vocabulary, or None if unknown."""
    return _LABEL_MAP.get((raw or "").strip().upper())


def _extract_gold_pages(evidence: object) -> List[str]:
    """Pull the Wikipedia page titles out of FEVER's nested evidence lists.

    FEVER nests evidence as ``[[[annot_id, evidence_id, page, sent_id], ...]]``
    where ``page`` is None for NOT ENOUGH INFO claims.
    """
    pages: List[str] = []
    if not isinstance(evidence, list):
        return pages
    for group in evidence:
        if not isinstance(group, list):
            continue
        for item in group:
            if isinstance(item, list) and len(item) >= 3 and isinstance(item[2], str):
                # FEVER escapes parentheses as -LRB- / -RRB- and spaces as _.
                title = item[2].replace("-LRB-", "(").replace("-RRB-", ")")
                title = title.replace("_", " ").strip()
                if title and title not in pages:
                    pages.append(title)
    return pages


def local_path(split: str, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> Path:
    """Return the local cache path for ``split`` (no download performed)."""
    if split not in SPLIT_FILES:
        raise ValueError(
            f"Unknown split {split!r}. Choose from: {', '.join(sorted(SPLIT_FILES))}"
        )
    return Path(cache_dir) / SPLIT_FILES[split]


def download_fever(
    split: str = "dev",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    force: bool = False,
) -> Path:
    """Download a FEVER split to the local cache, skipping if already present.

    Args:
        split: One of ``dev``, ``train``, ``paper_dev``, ``paper_test``.
        cache_dir: Directory to cache the JSONL in.
        force: Re-download even if a cached copy exists.

    Returns:
        Path to the cached JSONL file.

    Raises:
        FeverDownloadError: If the download fails.
        ValueError: If ``split`` is not recognised.
    """
    target = local_path(split, cache_dir)
    if target.exists() and target.stat().st_size > 0 and not force:
        logger.info("FEVER %s already cached at %s", split, target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{SPLIT_FILES[split]}"
    logger.info("Downloading FEVER %s from %s", split, url)

    # Stream to a .part file so an interrupted download cannot masquerade as a
    # complete cache entry on the next run.
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    handle.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise FeverDownloadError(f"Could not download FEVER {split} from {url}: {exc}") from exc

    partial.replace(target)
    logger.info("Cached FEVER %s (%.1f MB)", split, target.stat().st_size / 1e6)
    return target


def iter_fever(path: Path | str, split: str = "") -> Iterator[FeverClaim]:
    """Yield labelled claims from a FEVER JSONL file, skipping unusable rows.

    ``split`` is recorded on each claim so that a sample pooled across
    splits stays interpretable; it is metadata only and never reaches a
    model.
    """
    path = Path(path)
    if not path.exists():
        raise FeverDownloadError(f"FEVER file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at %s:%d", path.name, line_number)
                continue

            label = _normalise_label(record.get("label", ""))
            claim = str(record.get("claim", "")).strip()
            # Unlabelled test splits have no usable label; skip rather than guess.
            if not label or not claim:
                continue

            yield FeverClaim(
                id=str(record.get("id", line_number)),
                claim=claim,
                label=label,  # type: ignore[arg-type]
                dataset="fever",
                split=split,
                verifiable=str(record.get("verifiable", "")),
                gold_pages=_extract_gold_pages(record.get("evidence")),
            )


def load_fever(
    split: str = "dev",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    download: bool = True,
) -> List[FeverClaim]:
    """Load every labelled claim from a FEVER split.

    Args:
        split: Which split to load.
        cache_dir: Local cache directory.
        download: Fetch the split if it is not already cached.

    Returns:
        All labelled claims in the split.

    Raises:
        FeverDownloadError: If the split is missing and ``download`` is False,
            or if the download fails.
    """
    path = local_path(split, cache_dir)
    if not path.exists():
        if not download:
            raise FeverDownloadError(
                f"FEVER {split} is not cached at {path}. "
                "Run with download=True or `python evaluate.py --download`."
            )
        path = download_fever(split, cache_dir)
    return list(iter_fever(path, split))


__all__ = [
    "FeverClaim",
    "FeverDownloadError",
    "SPLIT_FILES",
    "download_fever",
    "iter_fever",
    "load_fever",
    "local_path",
    "sample_balanced",
    "label_distribution",
]
