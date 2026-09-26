"""DuckDuckGo evidence retrieval with backoff and graceful degradation.

DuckDuckGo throttles aggressively and without warning. The contract here is
that a throttled or broken search **never** raises into the pipeline: it
returns whatever it managed to collect plus a human-readable warning, so the
run continues and Agent 2/3 correctly conclude NOT_ENOUGH_INFO instead of the
app dying with a stack trace.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .schemas import EvidenceItem

logger = logging.getLogger(__name__)

# The package was renamed from `duckduckgo-search` to `ddgs`. Support both so
# the app runs against whichever one is already installed.
try:  # pragma: no cover - import shape depends on the installed package.
    from ddgs import DDGS  # type: ignore[import-not-found]

    _SEARCH_BACKEND = "ddgs"
except ImportError:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS  # type: ignore[import-not-found]

        _SEARCH_BACKEND = "duckduckgo_search"
    except ImportError:
        DDGS = None  # type: ignore[assignment]
        _SEARCH_BACKEND = "unavailable"

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.5, 4.0)
_RATE_LIMIT_MARKERS = ("ratelimit", "rate limit", "202", "429", "too many requests")


@dataclass(slots=True)
class RetrievalOutcome:
    """Evidence gathered across all queries, plus any non-fatal problems."""

    evidence: List[EvidenceItem]
    warnings: List[str]


def search_backend_name() -> str:
    """Return which DuckDuckGo client is installed ('unavailable' if none)."""
    return _SEARCH_BACKEND


def _looks_rate_limited(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _RATE_LIMIT_MARKERS)


def _raw_search(query: str, max_results: int) -> List[dict]:
    """Execute a single DuckDuckGo text search with bounded retries.

    Raises:
        RuntimeError: If every attempt fails; the caller degrades gracefully.
    """
    if DDGS is None:
        raise RuntimeError(
            "No DuckDuckGo client installed. Run: pip install ddgs"
        )

    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with DDGS() as client:
                return list(
                    client.text(
                        query,
                        region="wt-wt",
                        safesearch="moderate",
                        max_results=max_results,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - client raises a wide variety of errors.
            last_error = exc
            if attempt < _MAX_ATTEMPTS - 1:
                delay = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
                # Throttling needs a longer cool-off than a transient network blip.
                if _looks_rate_limited(exc):
                    delay *= 2
                logger.warning(
                    "search attempt %d for %r failed (%s); retrying in %.1fs",
                    attempt + 1,
                    query,
                    exc,
                    delay,
                )
                time.sleep(delay)

    raise RuntimeError(str(last_error) if last_error else "unknown search failure")


def _normalise(row: dict) -> tuple[str, str, str]:
    """Pull (title, url, body) out of a result row across client versions."""
    title = str(row.get("title") or "").strip()
    url = str(row.get("href") or row.get("url") or row.get("link") or "").strip()
    body = str(row.get("body") or row.get("snippet") or row.get("description") or "").strip()
    return title, url, body


def gather_evidence(
    queries: Sequence[str],
    *,
    results_per_query: int = 4,
    max_evidence: int = 8,
    snippet_chars: int = 700,
) -> RetrievalOutcome:
    """Run every query, de-duplicate results, and assign stable ``doc_N`` IDs.

    De-duplication is by URL first and then by the opening of the snippet text,
    because the same story syndicated across mirrors would otherwise let one
    source masquerade as independent corroboration.

    Args:
        queries: Search queries produced by Agent 1.
        results_per_query: Results requested per query.
        max_evidence: Hard ceiling on total snippets returned.
        snippet_chars: Truncation length for each snippet.

    Returns:
        A :class:`RetrievalOutcome`; ``evidence`` may be empty on total failure.
    """
    evidence: List[EvidenceItem] = []
    warnings: List[str] = []
    seen_urls: set[str] = set()
    seen_bodies: set[str] = set()

    for query in queries:
        if len(evidence) >= max_evidence:
            break
        try:
            rows: Iterable[dict] = _raw_search(query, results_per_query)
        except RuntimeError as exc:
            hint = (
                "DuckDuckGo rate-limited the request"
                if _looks_rate_limited(exc)
                else "search failed"
            )
            warnings.append(f"{hint} for query {query!r}: {exc}")
            continue

        for row in rows:
            if len(evidence) >= max_evidence:
                break
            title, url, body = _normalise(row)
            if not body:
                continue
            body_key = body[:120].lower()
            if (url and url in seen_urls) or body_key in seen_bodies:
                continue
            if url:
                seen_urls.add(url)
            seen_bodies.add(body_key)

            snippet = body if len(body) <= snippet_chars else body[:snippet_chars].rstrip() + "…"
            evidence.append(
                EvidenceItem(
                    source_id=f"doc_{len(evidence) + 1}",
                    snippet=snippet,
                    title=title,
                    url=url,
                    query=query,
                )
            )

    if not evidence and not warnings:
        warnings.append("Searches returned no usable results for this claim.")

    return RetrievalOutcome(evidence=evidence, warnings=warnings)
