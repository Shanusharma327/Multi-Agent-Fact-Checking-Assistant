"""AVeriTeC dataset loader.

AVeriTeC (Schlichtkrull et al., 2023) contains real-world claims collected from
fact-checking organisations, each annotated with question-answer evidence
decomposition drawn from the open web.

**Why this corpus is used alongside FEVER.** FEVER supplies comparability -- it
is the canonical benchmark and a reviewer knows what its numbers mean -- but it
is a poor fit for an open-web, LLM-based verifier on four counts:

1. *Contamination.* Published in 2018, FEVER's claims and labels have been
   public for years and are plausibly in the pre-training data of the very
   models being evaluated. The ungrounded baseline may therefore be recalling
   the benchmark rather than reasoning about the world, which biases exactly
   the quantity this project measures.
2. *Claim-only artefacts.* Schuster et al. showed classifiers score well above
   chance on FEVER with the evidence removed, because the claim-mutation
   process left surface giveaways.
3. *Corpus-relative NEI.* FEVER's NOT ENOUGH INFO means "unverifiable from a
   June 2017 Wikipedia snapshot", a definition an open-web system does not
   implement.
4. *Distribution.* FEVER claims are short mutated Wikipedia sentences about
   entities, not the political, health, and scientific misinformation that
   motivates fact-checking.

AVeriTeC addresses all four: claims are naturally occurring and dated, evidence
is open-web, and it post-dates FEVER by five years. It is smaller (a few
thousand claims), so the two are complementary rather than substitutes.

**The fourth label.** AVeriTeC is 4-way; its
``Conflicting Evidence/Cherrypicking`` class has no honest equivalent in a
three-way scheme. It is *excluded* by default rather than silently folded into
NOT_ENOUGH_INFO, and the excluded count is always reported. Pass
``map_conflicting=True`` to map it to NOT_ENOUGH_INFO instead.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import Field

from .base import BenchmarkClaim, label_distribution, sample_balanced

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data"

SPLIT_FILES: Dict[str, str] = {
    "dev": "dev.json",
    "train": "train.json",
    "test": "test.json",
}

# AVeriTeC label -> our three-way vocabulary. The fourth class maps to None,
# meaning "no honest equivalent"; see the module docstring.
_LABEL_MAP: Dict[str, Optional[str]] = {
    "supported": "SUPPORTS",
    "refuted": "REFUTES",
    "not enough evidence": "NOT_ENOUGH_INFO",
    "conflicting evidence/cherrypicking": None,
}

CONFLICTING_LABEL = "Conflicting Evidence/Cherrypicking"

DEFAULT_CACHE_DIR = Path("data/averitec")

_USER_AGENT = "multi-agent-fact-checker/1.0 (+evaluation harness)"


class AVeriTeCDownloadError(RuntimeError):
    """Raised when the AVeriTeC split cannot be downloaded or read."""


class AVeriTeCClaim(BenchmarkClaim):
    """One AVeriTeC claim, normalised to this project's label set."""

    claim_date: str = Field(default="", description="Date the claim was made.")
    speaker: str = Field(default="", description="Who made the claim, if known.")
    reporting_source: str = Field(default="", description="Where it was reported.")
    claim_types: List[str] = Field(default_factory=list)
    country_code: str = Field(default="", description="ISO country code, if known.")
    n_questions: int = Field(
        default=0, description="Number of annotated evidence questions."
    )
    n_answers: int = Field(default=0, description="Total annotated answers.")
    justification: str = Field(default="", description="Annotator rationale.")
    fact_checking_article: str = Field(default="")


def _normalise_label(raw: str) -> Optional[str]:
    """Map an AVeriTeC label onto our vocabulary, or None if it has no match."""
    return _LABEL_MAP.get((raw or "").strip().lower())


def local_path(split: str, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> Path:
    """Return the local cache path for ``split`` (no download performed)."""
    if split not in SPLIT_FILES:
        raise ValueError(
            f"Unknown split {split!r}. Choose from: {', '.join(sorted(SPLIT_FILES))}"
        )
    return Path(cache_dir) / SPLIT_FILES[split]


def download_averitec(
    split: str = "dev",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    force: bool = False,
) -> Path:
    """Download an AVeriTeC split to the local cache, skipping if present.

    Raises:
        AVeriTeCDownloadError: If the download fails.
        ValueError: If ``split`` is not recognised.
    """
    target = local_path(split, cache_dir)
    if target.exists() and target.stat().st_size > 0 and not force:
        logger.info("AVeriTeC %s already cached at %s", split, target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{SPLIT_FILES[split]}"
    logger.info("Downloading AVeriTeC %s from %s", split, url)

    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with partial.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    handle.write(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        partial.unlink(missing_ok=True)
        raise AVeriTeCDownloadError(
            f"Could not download AVeriTeC {split} from {url}: {exc}"
        ) from exc

    partial.replace(target)
    logger.info("Cached AVeriTeC %s (%.1f MB)", split, target.stat().st_size / 1e6)
    return target


def load_averitec(
    split: str = "dev",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    download: bool = True,
    map_conflicting: bool = False,
) -> Tuple[List[AVeriTeCClaim], Dict[str, int]]:
    """Load an AVeriTeC split, normalised to the three-way label set.

    Args:
        split: Which split to load.
        cache_dir: Local cache directory.
        download: Fetch the split if it is not already cached.
        map_conflicting: Map ``Conflicting Evidence/Cherrypicking`` to
            NOT_ENOUGH_INFO instead of excluding it.

    Returns:
        ``(claims, report)`` where ``report`` counts what was kept, excluded,
        and skipped -- so an exclusion can never pass unnoticed.

    Raises:
        AVeriTeCDownloadError: If the split is missing or unreadable.
    """
    path = local_path(split, cache_dir)
    if not path.exists():
        if not download:
            raise AVeriTeCDownloadError(
                f"AVeriTeC {split} is not cached at {path}. "
                "Run `python evaluate.py --dataset averitec --download`."
            )
        path = download_averitec(split, cache_dir)

    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise AVeriTeCDownloadError(f"Could not read {path}: {exc}") from exc

    if not isinstance(records, list):
        raise AVeriTeCDownloadError(
            f"Expected a JSON array in {path}, got {type(records).__name__}"
        )

    claims: List[AVeriTeCClaim] = []
    report = {"total": len(records), "kept": 0, "conflicting": 0, "unusable": 0}

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            report["unusable"] += 1
            continue

        text = str(record.get("claim", "")).strip()
        raw_label = str(record.get("label", ""))
        label = _normalise_label(raw_label)

        if raw_label.strip().lower() == CONFLICTING_LABEL.lower():
            report["conflicting"] += 1
            if not map_conflicting:
                continue
            label = "NOT_ENOUGH_INFO"

        if not text or not label:
            report["unusable"] += 1
            continue

        questions = record.get("questions") or []
        n_answers = sum(
            len(q.get("answers") or []) for q in questions if isinstance(q, dict)
        )

        claims.append(
            AVeriTeCClaim(
                id=f"averitec_{split}_{index}",  # the corpus carries no id field
                claim=text,
                label=label,  # type: ignore[arg-type]
                dataset="averitec",
                claim_date=str(record.get("claim_date") or ""),
                speaker=str(record.get("speaker") or ""),
                reporting_source=str(record.get("reporting_source") or ""),
                claim_types=[str(t) for t in (record.get("claim_types") or [])],
                country_code=str(record.get("location_ISO_code") or ""),
                n_questions=len(questions),
                n_answers=n_answers,
                justification=str(record.get("justification") or ""),
                fact_checking_article=str(record.get("fact_checking_article") or ""),
            )
        )

    report["kept"] = len(claims)
    return claims, report


__all__ = [
    "AVeriTeCClaim",
    "AVeriTeCDownloadError",
    "SPLIT_FILES",
    "CONFLICTING_LABEL",
    "download_averitec",
    "load_averitec",
    "local_path",
    "label_distribution",
    "sample_balanced",
]
