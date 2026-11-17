"""URL canonicalisation and comparison for retrieval scoring.

Retrieval quality is only measurable if a URL a search engine returns can be
compared against a URL a human annotator recorded. Those two rarely match
character for character even when they name the same page: a third of
AVeriTeC's gold references are served through the Wayback Machine, hosts
differ by a leading ``www.``, and live search results arrive carrying
tracking parameters.

Canonicalisation here is deliberately conservative -- it removes only what
cannot change which document is named. Query parameters are *kept* (apart
from a known tracking list) because ``?id=123`` frequently selects the
article rather than describing how the reader arrived at it.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set
from urllib.parse import parse_qsl, urlsplit, urlunsplit

# Archive services wrap the original URL inside their own path. The capture
# timestamp is variable, so matching requires unwrapping to the target.
_ARCHIVE_PATTERNS = (
    re.compile(r"^https?://web\.archive\.org/web/[^/]+/(?P<target>https?://.+)$", re.I),
    re.compile(
        r"^https?://(?:www\.)?webarchive\.org\.uk/wayback/archive/[^/]+/(?P<target>https?://.+)$",
        re.I,
    ),
    re.compile(r"^https?://archive\.(?:ph|is|today|li|fo)/[^/]+/(?P<target>https?://.+)$", re.I),
)

# Parameters that identify a referral, not a document.
_TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_name", "utm_reader", "utm_brand", "utm_social",
        "fbclid", "gclid", "dclid", "msclkid", "igshid", "mc_cid", "mc_eid",
        "ref", "ref_src", "ref_url", "_ga", "yclid", "spm", "scid",
        "cmpid", "ncid", "sr_share", "__twitter_impression",
    }
)

# Public-suffix fragments that take two labels rather than one. A full public
# suffix list is overkill here; this covers the hosts AVeriTeC actually cites.
_TWO_LABEL_SUFFIXES = frozenset(
    {
        "co.uk", "org.uk", "gov.uk", "ac.uk", "me.uk", "net.uk", "sch.uk",
        "com.au", "net.au", "org.au", "gov.au", "edu.au",
        "co.in", "net.in", "org.in", "gov.in", "ac.in", "nic.in",
        "co.za", "org.za", "gov.za", "co.nz", "org.nz", "govt.nz",
        "com.br", "gov.br", "org.br", "com.mx", "gob.mx",
        "co.jp", "or.jp", "ne.jp", "go.jp", "com.sg", "gov.sg",
        "com.ng", "gov.ng", "com.pk", "gov.pk", "com.ph", "gov.ph",
        "co.ke", "go.ke", "com.tr", "gov.tr", "co.id", "go.id",
    }
)

_DEFAULT_PORTS = {"http": "80", "https": "443"}

# A plausible DNS name: dot-separated alphanumeric labels, or a bare IPv4.
_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_HOSTNAME = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$|^\d{1,3}(?:\.\d{1,3}){3}$"
)


def unwrap_archive(url: str) -> str:
    """Recover the original target from an archive wrapper URL.

    Applied repeatedly, because an archived URL is occasionally itself an
    archive link. Returns ``url`` unchanged when it is not a wrapper.
    """
    current = (url or "").strip()
    for _ in range(3):  # bounded: a genuine chain is never deeper than this.
        for pattern in _ARCHIVE_PATTERNS:
            match = pattern.match(current)
            if match:
                current = match.group("target")
                break
        else:
            return current
    return current


def canonical_url(url: str) -> str:
    """Reduce ``url`` to a comparable canonical form.

    Unwraps archive wrappers, lowercases the scheme and host, strips a leading
    ``www.``, a default port, the fragment, tracking parameters, and a
    trailing slash. Returns ``""`` for anything unparseable, which never
    matches -- a missing URL must not be scored as a retrieval hit.
    """
    raw = unwrap_archive(url)
    if not raw:
        return ""

    # A bare "example.com/x" gives urlsplit no scheme to find.
    if "//" not in raw[:10]:
        raw = "http://" + raw.lstrip("/")

    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return ""

    # A free-text string parses into a "hostname" that no DNS name could be.
    # Rejecting it keeps junk out of the comparison sets, where it would
    # otherwise be a token that can never match but still inflates counts.
    if not host or not _HOSTNAME.match(host):
        return ""
    if host.startswith("www."):
        host = host[4:]

    netloc = host
    if port is not None and str(port) != _DEFAULT_PORTS.get(parts.scheme.lower(), ""):
        netloc = f"{host}:{port}"

    query = "&".join(
        f"{key}={value}"
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS
    )

    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"

    # The scheme is normalised away: http and https serve the same document.
    return urlunsplit(("https", netloc, path, query, ""))


def registered_domain(url: str) -> str:
    """Return the registrable domain of ``url`` (``""`` if indeterminate).

    Domain-level matching is reported alongside exact-URL matching because a
    retrieval system may legitimately surface a different page on the same
    source than the annotator happened to cite.
    """
    canonical = canonical_url(url)
    if not canonical:
        return ""
    host = urlsplit(canonical).hostname or ""
    # A bare IP has no registrable domain; splitting it on dots would yield a
    # meaningless fragment like "1.1".
    if _IPV4.match(host):
        return host
    labels = host.split(".")
    if len(labels) < 2:
        return host
    if ".".join(labels[-2:]) in _TWO_LABEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def canonical_set(urls: Iterable[str]) -> Set[str]:
    """Canonicalise an iterable of URLs, dropping blanks."""
    return {c for c in (canonical_url(u) for u in urls) if c}


def domain_set(urls: Iterable[str]) -> Set[str]:
    """Registrable domains of an iterable of URLs, dropping blanks."""
    return {d for d in (registered_domain(u) for u in urls) if d}


def same_document(left: str, right: str) -> bool:
    """Whether two URLs name the same document after canonicalisation."""
    canonical_left = canonical_url(left)
    return bool(canonical_left) and canonical_left == canonical_url(right)


def filter_leakage(
    urls: Sequence[str],
    fact_check_url: str = "",
    *,
    blocked_domains: Iterable[str] = (),
) -> List[bool]:
    """Flag which of ``urls`` must be discarded as verdict leakage.

    Every AVeriTeC claim carries the URL of the fact-checking article it was
    drawn from, and that article states the verdict. Retrieving it would let
    the pipeline read the answer rather than derive it.

    Args:
        urls: Candidate result URLs.
        fact_check_url: The claim's originating fact-check article.
        blocked_domains: Optional stricter variant excluding whole
            fact-checking domains. More conservative against near-duplicate
            republication, but it also removes genuine gold references, so it
            is offered as a sensitivity check rather than the default.

    Returns:
        A list of booleans, ``True`` where the corresponding URL leaks.
    """
    target = canonical_url(fact_check_url)
    blocked = {
        d for d in (registered_domain(b) or str(b).lower() for b in blocked_domains) if d
    }
    flags: List[bool] = []
    for url in urls:
        leaks = bool(target) and canonical_url(url) == target
        if not leaks and blocked:
            leaks = registered_domain(url) in blocked
        flags.append(leaks)
    return flags


__all__ = [
    "canonical_set",
    "canonical_url",
    "domain_set",
    "filter_leakage",
    "registered_domain",
    "same_document",
    "unwrap_archive",
]
