"""Per-claim passage cache: the substrate every condition replays against.

Open-web retrieval is not reproducible. Two conditions run as separate live
benchmarks differ in the documents they saw as well as in the thing under
test, which makes any difference between them uninterpretable. Caching the
retrieved passages once and replaying them removes that confound: every
condition that uses evidence sees byte-identical evidence for a given claim,
and a difference between conditions is attributable to the condition.

The cache is the unit of release, too. A reader cannot re-run our searches --
the ranking function is undisclosed and the web has moved on -- but with the
cache they can re-run every condition over exactly the evidence we had.

Identity is enforced rather than assumed. :meth:`CachedClaim.fingerprint`
hashes the ordered ``(source_id, url, snippet)`` triples, every result record
carries that fingerprint, and :func:`assert_identical_evidence` fails the run
if two conditions ever disagree about what a claim's evidence was.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from pydantic import BaseModel, Field

from .schemas import EvidenceItem, ScoutReport


def utc_now() -> str:
    """Current time as an ISO-8601 UTC string, to the second."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class CachedPassage(BaseModel):
    """One retrieved passage, exactly as the verification stage saw it."""

    source_id: str = Field(description="Stable identifier, e.g. 'doc_1'.")
    url: str = Field(default="", description="Result URL, as returned.")
    title: str = Field(default="", description="Result title.")
    snippet: str = Field(description="Verbatim snippet text, post-truncation.")
    query: str = Field(default="", description="Query that surfaced this result.")


class CachedClaim(BaseModel):
    """Everything about one claim that a replayed condition may consume.

    The gold fields are carried for scoring convenience; they are never
    supplied to an agent. ``retrieved_at`` records when the evidence was
    fetched, which is what makes a cache interpretable years later -- the
    same query against the live web will not return this again.
    """

    claim_id: str
    dataset: str = ""
    split: str = Field(default="", description="Official split the claim came from.")
    claim: str = ""
    gold: str = ""
    gold_urls: List[str] = Field(default_factory=list)
    fact_check_url: str = ""
    queries: List[str] = Field(default_factory=list)
    passages: List[CachedPassage] = Field(default_factory=list)
    retrieved_at: str = Field(default="", description="ISO-8601 UTC fetch time.")
    retrieval_warnings: List[str] = Field(default_factory=list)
    n_leakage_filtered: int = 0

    def fingerprint(self) -> str:
        """A short hash of the evidence set, used to prove identity.

        Covers the ordered identifier, URL and snippet of every passage.
        Title and query are excluded: they are retrieval metadata that no
        agent reads, so a difference in them is not a difference in evidence.
        """
        digest = hashlib.sha256()
        for passage in self.passages:
            digest.update(passage.source_id.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(passage.url.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(passage.snippet.encode("utf-8"))
            digest.update(b"\x1e")
        return digest.hexdigest()[:16]

    def to_scout_report(self) -> ScoutReport:
        """Rebuild the Agent-1 output a live run would have produced."""
        return ScoutReport(
            queries=list(self.queries),
            evidence=[
                EvidenceItem(
                    source_id=p.source_id,
                    snippet=p.snippet,
                    title=p.title,
                    url=p.url,
                    query=p.query,
                )
                for p in self.passages
            ],
        )

    @property
    def is_empty(self) -> bool:
        """Whether retrieval returned nothing for this claim."""
        return not self.passages


class PassageCache(BaseModel):
    """A claim-id-keyed collection of :class:`CachedClaim`, stored as JSONL."""

    entries: Dict[str, CachedClaim] = Field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, claim_id: object) -> bool:
        return str(claim_id) in self.entries

    def __iter__(self) -> Iterator[CachedClaim]:  # type: ignore[override]
        return iter(self.entries.values())

    def get(self, claim_id: str) -> Optional[CachedClaim]:
        return self.entries.get(str(claim_id))

    def add(self, entry: CachedClaim) -> None:
        self.entries[entry.claim_id] = entry

    def claim_ids(self) -> List[str]:
        return list(self.entries)

    def subset(self, claim_ids: Iterable[str]) -> "PassageCache":
        """A cache restricted to ``claim_ids``, preserving their order."""
        wanted = [str(c) for c in claim_ids]
        return PassageCache(
            entries={c: self.entries[c] for c in wanted if c in self.entries}
        )

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Write the cache as JSONL, one claim per line, sorted by id."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for claim_id in sorted(self.entries):
                handle.write(
                    json.dumps(self.entries[claim_id].model_dump(), ensure_ascii=False)
                    + "\n"
                )
        return path

    @classmethod
    def load(cls, path: Path) -> "PassageCache":
        """Read a cache written by :meth:`save`.

        Raises:
            FileNotFoundError: If ``path`` does not exist. A silently empty
                cache would make every replayed condition look like a
                retrieval failure, which is worse than failing here.
        """
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No passage cache at {path}. Build one with "
                "`python -m fact_checker.cache --from-results <results.jsonl>` "
                "or run evaluate.py, which writes one."
            )
        cache = cls()
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                cache.add(CachedClaim.model_validate(json.loads(line)))
        return cache

    # -- construction ------------------------------------------------------

    @classmethod
    def from_results(
        cls, records: Sequence[dict], *, retrieved_at: str = ""
    ) -> "PassageCache":
        """Build a cache from result records that carry a stored trace.

        Lets an existing benchmark run become the substrate for new
        conditions without paying for retrieval again. Records written
        before the trace was captured are skipped rather than cached as
        empty, since an empty evidence set is a meaningful state and must
        not be manufactured.
        """
        cache = cls()
        for record in records:
            block = record.get("multi") or {}
            evidence = block.get("evidence")
            if evidence is None:
                continue
            cache.add(
                CachedClaim(
                    claim_id=str(record.get("id", "")),
                    dataset=str(record.get("dataset", "")),
                    split=str(record.get("split", "")),
                    claim=str(record.get("claim", "")),
                    gold=str(record.get("gold", "")),
                    gold_urls=list(record.get("gold_urls") or []),
                    fact_check_url=str(record.get("fact_check_url", "")),
                    queries=list(block.get("queries") or []),
                    passages=[
                        CachedPassage(
                            source_id=str(item.get("source_id", "")),
                            url=str(item.get("url", "")),
                            title=str(item.get("title", "")),
                            snippet=str(item.get("snippet", "")),
                            query=str(item.get("query", "")),
                        )
                        for item in evidence
                    ],
                    retrieved_at=str(record.get("retrieved_at") or retrieved_at),
                    retrieval_warnings=list(block.get("warnings") or []),
                )
            )
        return cache

    # -- reporting ---------------------------------------------------------

    def summary(self) -> Dict[str, object]:
        """Counts a reader needs to interpret anything replayed over this."""
        passages = [len(entry.passages) for entry in self]
        stamps = sorted({e.retrieved_at for e in self if e.retrieved_at})
        splits: Dict[str, int] = {}
        for entry in self:
            splits[entry.split or "(unrecorded)"] = (
                splits.get(entry.split or "(unrecorded)", 0) + 1
            )
        labels: Dict[str, int] = {}
        for entry in self:
            labels[entry.gold or "(none)"] = labels.get(entry.gold or "(none)", 0) + 1
        return {
            "claims": len(self),
            "empty_evidence": sum(1 for n in passages if n == 0),
            "mean_passages": (sum(passages) / len(passages)) if passages else 0.0,
            "total_passages": sum(passages),
            "splits": splits,
            "labels": labels,
            "retrieved_between": (stamps[0], stamps[-1]) if stamps else ("", ""),
        }


class EvidenceMismatch(AssertionError):
    """Raised when two conditions did not see the same evidence for a claim.

    This is a hard failure by design. A silent mismatch would invalidate
    every between-condition comparison in the paper while leaving the
    numbers looking perfectly reasonable.
    """


def assert_identical_evidence(
    condition_records: Dict[str, Sequence[dict]],
    *,
    key: str = "evidence_fingerprint",
) -> int:
    """Verify every evidence-using condition saw identical passages per claim.

    Args:
        condition_records: ``{condition_name: [result records]}``. Conditions
            whose records carry no fingerprint are skipped, which is how
            B0 (no retrieval) is excluded without special-casing its name.
        key: Record field holding the evidence fingerprint.

    Returns:
        The number of claims checked across at least two conditions.

    Raises:
        EvidenceMismatch: On the first claim whose fingerprints disagree.
    """
    seen: Dict[str, Dict[str, str]] = {}
    for condition, records in condition_records.items():
        for record in records:
            fingerprint = record.get(key)
            if not fingerprint:
                continue
            seen.setdefault(str(record.get("id", "")), {})[condition] = str(fingerprint)

    checked = 0
    for claim_id, by_condition in sorted(seen.items()):
        if len(by_condition) < 2:
            continue
        checked += 1
        distinct = set(by_condition.values())
        if len(distinct) > 1:
            detail = ", ".join(f"{c}={f}" for c, f in sorted(by_condition.items()))
            raise EvidenceMismatch(
                f"Claim {claim_id!r} was evaluated against different evidence "
                f"by different conditions ({detail}). Every replayed condition "
                "must consume the same cached passages; re-run the replay from "
                "a single cache."
            )
    return checked


__all__ = [
    "CachedClaim",
    "CachedPassage",
    "EvidenceMismatch",
    "PassageCache",
    "assert_identical_evidence",
    "utc_now",
]
