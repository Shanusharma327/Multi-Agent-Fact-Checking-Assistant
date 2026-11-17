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
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Sequence

from .schemas import EvidenceItem
from .urls import canonical_url, filter_leakage

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

# Raised from (1.5, 4.0) after a run drove the endpoint into refusing every
# request: two benchmark processes retrying on a short backoff turned a
# transient throttle into a total stall that never recovered. A retry
# schedule that is too eager is not merely wasteful, it is the cause.
_BACKOFF_SECONDS = (4.0, 12.0, 30.0)

_RATE_LIMIT_MARKERS = ("ratelimit", "rate limit", "202", "429", "too many requests")

# A timeout from the search host is the shape rate limiting takes here --
# the endpoint stops answering rather than returning an error code -- so it
# earns the same long cool-off as an explicit throttle.
_TIMEOUT_MARKERS = ("operation timed out", "timed out", "timeout", "connection reset")

# Circuit breaker. Two rules, because the first one alone proved
# insufficient in practice.
#
# The consecutive rule catches an outright outage: nothing is answering.
#
# The windowed rule catches what the consecutive rule missed, and the
# distinction cost half an hour of a benchmark run. With the host mostly
# refusing but occasionally answering, every rare success reset the
# consecutive counter, so the breaker never tripped while retrieval
# advanced by zero claims. A failure *rate* over a sliding window
# describes that state; a consecutive count cannot, because the state is
# precisely "failures interleaved with just enough successes".
_BREAKER_THRESHOLD = 8
_BREAKER_WINDOW = 20
_BREAKER_WINDOW_RATE = 0.75
_BREAKER_PAUSE_SECONDS = 180.0


class SearchUnavailable(RuntimeError):
    """Raised when consecutive failures indicate the service is down.

    Distinct from a single failed query, which the caller absorbs. This
    says: stop asking. Looping through hundreds of claims against a dead
    endpoint produces a cache full of empty evidence that looks like a
    retrieval result but is an outage, and the two must not be confused.
    """


class _Breaker:
    """Tracks search health by consecutive failures and by windowed rate."""

    def __init__(
        self,
        threshold: int = _BREAKER_THRESHOLD,
        window: int = _BREAKER_WINDOW,
        window_rate: float = _BREAKER_WINDOW_RATE,
    ) -> None:
        self.threshold = threshold
        self.window = window
        self.window_rate = window_rate
        self.consecutive = 0
        self.recent: Deque[bool] = deque(maxlen=window)

    def record_failure(self) -> None:
        self.consecutive += 1
        self.recent.append(False)

    def record_success(self) -> None:
        self.consecutive = 0
        self.recent.append(True)

    @property
    def window_failure_rate(self) -> float:
        """Share of the last ``window`` attempts that failed."""
        if not self.recent:
            return 0.0
        return sum(1 for ok in self.recent if not ok) / len(self.recent)

    @property
    def tripped(self) -> bool:
        if self.consecutive >= self.threshold:
            return True
        # Only judge a rate once the window is full; a short burst of
        # failures at the start of a run is not evidence of an outage.
        return (
            len(self.recent) >= self.window
            and self.window_failure_rate >= self.window_rate
        )

    def reason(self) -> str:
        """Why the breaker tripped, for a message the operator can act on."""
        if self.consecutive >= self.threshold:
            return f"{self.consecutive} consecutive search failures"
        return (
            f"{self.window_failure_rate:.0%} of the last {len(self.recent)} "
            "search attempts failed"
        )


_breaker = _Breaker()


def breaker_state() -> int:
    """Consecutive search failures so far, for callers that want to report it."""
    return _breaker.consecutive


def breaker_window_rate() -> float:
    """Failure rate over the recent window, for progress reporting."""
    return _breaker.window_failure_rate


def reset_breaker() -> None:
    """Clear both failure signals, after a deliberate pause."""
    _breaker.consecutive = 0
    _breaker.recent.clear()


@dataclass(slots=True)
class RetrievalOutcome:
    """Evidence gathered across all queries, plus any non-fatal problems."""

    evidence: List[EvidenceItem]
    warnings: List[str]
    n_leakage_filtered: int = 0
    """Results discarded for naming the claim's originating fact-check."""


def search_backend_name() -> str:
    """Return which DuckDuckGo client is installed ('unavailable' if none)."""
    return _SEARCH_BACKEND


def _looks_rate_limited(exc: Exception) -> bool:
    """Whether the service is pushing back, by code or by silence.

    A timeout counts. When this endpoint throttles it stops answering
    rather than returning 429, so treating timeouts as ordinary network
    faults gives them a short retry and makes the throttling worse.
    """
    message = str(exc).lower()
    return any(
        marker in message
        for marker in _RATE_LIMIT_MARKERS + _TIMEOUT_MARKERS
    )


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
    exclude_url: str = "",
    blocked_domains: Sequence[str] = (),
) -> RetrievalOutcome:
    """Run every query, de-duplicate results, and assign stable ``doc_N`` IDs.

    De-duplication is by URL first and then by the opening of the snippet text,
    because the same story syndicated across mirrors would otherwise let one
    source masquerade as independent corroboration. URLs are compared after
    canonicalisation, so the same page reached through an archive wrapper or
    carrying tracking parameters is recognised as a duplicate rather than as
    independent corroboration.

    Args:
        queries: Search queries produced by Agent 1.
        results_per_query: Results requested per query.
        max_evidence: Hard ceiling on total snippets returned.
        snippet_chars: Truncation length for each snippet.
        exclude_url: The claim's originating fact-check article. Any result
            naming it is discarded, because that article states the verdict
            and retrieving it would let the pipeline read the answer instead
            of deriving it. Empty in normal operation, where no such article
            is known.
        blocked_domains: Stricter leakage variant excluding whole
            fact-checking domains. Offered as a sensitivity check; it also
            removes genuine gold references, so it is not the default.

    Returns:
        A :class:`RetrievalOutcome`; ``evidence`` may be empty on total failure.
    """
    evidence: List[EvidenceItem] = []
    warnings: List[str] = []
    seen_urls: set[str] = set()
    seen_bodies: set[str] = set()
    n_filtered = 0

    for query in queries:
        if len(evidence) >= max_evidence:
            break
        if _breaker.tripped:
            # Every further query would fail and deepen the throttle.
            raise SearchUnavailable(
                f"{_breaker.reason()}: the service is not answering reliably. "
                "Pause the run and resume later rather than looping; a cache "
                "filled against a failing endpoint records an outage as though "
                "it were a retrieval result. Retrieval is never restarted "
                "automatically, because the right wait is a judgement about "
                "the service, not something this process can measure."
            )

        try:
            rows: Iterable[dict] = _raw_search(query, results_per_query)
            _breaker.record_success()
        except RuntimeError as exc:
            _breaker.record_failure()
            hint = (
                "search service throttled or timed out"
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

            if (exclude_url or blocked_domains) and url:
                leaks, = filter_leakage(
                    [url], exclude_url, blocked_domains=blocked_domains
                )
                if leaks:
                    n_filtered += 1
                    continue

            body_key = body[:120].lower()
            # Canonicalise before the duplicate check so the same document
            # served two ways collapses to one entry.
            url_key = canonical_url(url) or url
            if (url_key and url_key in seen_urls) or body_key in seen_bodies:
                continue
            if url_key:
                seen_urls.add(url_key)
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

    if n_filtered:
        # Never let a leakage exclusion pass silently: it changes what the
        # retriever was allowed to see, and the count belongs in the record.
        warnings.append(
            f"excluded {n_filtered} result(s) naming the originating fact-check"
        )

    if not evidence and not warnings:
        warnings.append("Searches returned no usable results for this claim.")

    return RetrievalOutcome(
        evidence=evidence, warnings=warnings, n_leakage_filtered=n_filtered
    )
