"""Retrieval quality scoring against annotated gold evidence URLs.

The paper's research question RQ2 asks whether explicit retrieval improves
*evidence groundedness*, which cannot be answered from verdict accuracy
alone: a system can reach the right verdict from the wrong source, and a
system that retrieves well can still reason badly. Separating the two is the
point of measuring retrieval on its own.

AVeriTeC makes this measurable because every annotated answer carries a
``source_url``. Scores are reported at two granularities:

* **exact** -- the canonical URL matches a gold reference;
* **domain** -- the registrable domain matches.

Domain-level recall is the more forgiving and the more honest of the two for
live web retrieval, because a search engine may surface a different page on
the same source than the annotator happened to cite. Both are reported;
neither is presented alone.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from pydantic import BaseModel, Field

from .urls import canonical_set, domain_set, registered_domain


class RetrievalScore(BaseModel):
    """Recall@k / Precision@k at one matching granularity."""

    granularity: str = Field(description="'exact' (canonical URL) or 'domain'.")
    n_claims: int = Field(description="Claims with at least one gold URL.")
    recall_at_k: float = Field(description="Mean per-claim share of gold found.")
    precision_at_k: float = Field(description="Mean per-claim share of hits retrieved.")
    hit_rate: float = Field(
        description="Share of claims where at least one gold reference was found."
    )
    mean_retrieved: float = Field(description="Mean retrieved items per claim.")
    mean_gold: float = Field(description="Mean gold references per claim.")


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def score_one(retrieved: Sequence[str], gold: Sequence[str], *, domain: bool) -> Dict[str, float]:
    """Recall and precision for a single claim at one granularity.

    Sets rather than lists are compared, so a duplicate retrieved URL cannot
    count twice toward either score.

    Returns:
        ``{"recall", "precision", "hits", "n_retrieved", "n_gold"}``. A claim
        with no gold references yields zeros and is excluded by the caller.
    """
    project = domain_set if domain else canonical_set
    retrieved_set = project(retrieved)
    gold_set = project(gold)
    if not gold_set:
        return {"recall": 0.0, "precision": 0.0, "hits": 0.0,
                "n_retrieved": float(len(retrieved_set)), "n_gold": 0.0}

    hits = retrieved_set & gold_set
    return {
        "recall": _safe_div(len(hits), len(gold_set)),
        "precision": _safe_div(len(hits), len(retrieved_set)),
        "hits": float(len(hits)),
        "n_retrieved": float(len(retrieved_set)),
        "n_gold": float(len(gold_set)),
    }


def score_retrieval(
    retrieved_per_claim: Sequence[Sequence[str]],
    gold_per_claim: Sequence[Sequence[str]],
    *,
    domain: bool = False,
) -> RetrievalScore:
    """Macro-average Recall@k and Precision@k over claims.

    Claims carrying no gold reference are skipped rather than scored as
    zero-recall: there is nothing for the retriever to have found, and
    counting them would penalise the system for an annotation gap.

    Args:
        retrieved_per_claim: Retrieved URLs, one list per claim.
        gold_per_claim: Gold evidence URLs, aligned with ``retrieved_per_claim``.
        domain: Match on registrable domain instead of canonical URL.

    Returns:
        A :class:`RetrievalScore` averaged over the scorable claims.

    Raises:
        ValueError: If the two sequences have differing lengths.
    """
    if len(retrieved_per_claim) != len(gold_per_claim):
        raise ValueError(
            "retrieved_per_claim and gold_per_claim must align "
            f"(got {len(retrieved_per_claim)} and {len(gold_per_claim)})"
        )

    rows: List[Dict[str, float]] = []
    for retrieved, gold in zip(retrieved_per_claim, gold_per_claim):
        row = score_one(retrieved, gold, domain=domain)
        if row["n_gold"]:
            rows.append(row)

    total = len(rows)
    return RetrievalScore(
        granularity="domain" if domain else "exact",
        n_claims=total,
        recall_at_k=_safe_div(sum(r["recall"] for r in rows), total),
        precision_at_k=_safe_div(sum(r["precision"] for r in rows), total),
        hit_rate=_safe_div(sum(1 for r in rows if r["hits"] > 0), total),
        mean_retrieved=_safe_div(sum(r["n_retrieved"] for r in rows), total),
        mean_gold=_safe_div(sum(r["n_gold"] for r in rows), total),
    )


class CitationScore(BaseModel):
    """Citation quality, separating validity from evidential support.

    Validity is the weaker property the pipeline guarantees by construction:
    every emitted identifier names a passage that was genuinely retrieved.
    *Support* is the stronger one it does not get for free -- that the cited
    passage was judged by the verification stage to bear on the verdict in
    the direction the verdict went. Reporting validity alone overstates what
    a citation establishes, which is why both appear here.
    """

    n_verdicts: int
    n_with_citations: int
    grounded_rate: float = Field(description="Share of verdicts carrying >=1 citation.")
    n_citations: int
    validity: float = Field(description="Share of citations naming retrieved passages.")
    support_precision: float = Field(
        description="Share of citations whose audit label matches the verdict."
    )
    n_supportable: int = Field(
        description="Citations on decidable verdicts, the denominator for support."
    )


def score_citations(
    records: Sequence[Dict],
    *,
    key: str = "multi",
) -> CitationScore:
    """Score citation validity and evidential support across a run.

    Support precision is computed only over citations attached to a decidable
    verdict (SUPPORTS or REFUTES). A citation on a NOT_ENOUGH_INFO verdict has
    no direction to agree with, so including it would measure something
    different from what the name says.

    Args:
        records: Result records carrying ``cited_sources``, ``evidence_ids``
            and ``audit`` entries.
        key: Which system's block to read.

    Returns:
        A populated :class:`CitationScore`.
    """
    n_verdicts = 0
    n_with = 0
    total_citations = 0
    valid_citations = 0
    supportable = 0
    supporting = 0

    for record in records:
        block = record.get(key) or {}
        if not block.get("verdict"):
            continue
        n_verdicts += 1

        cited = list(block.get("cited_sources") or [])
        valid_ids = set(block.get("evidence_ids") or [])
        audit = {
            entry.get("source_id"): entry.get("verdict")
            for entry in (block.get("audit") or [])
            if isinstance(entry, dict)
        }

        if cited:
            n_with += 1
        total_citations += len(cited)
        valid_citations += sum(1 for c in cited if c in valid_ids)

        if block["verdict"] in ("SUPPORTS", "REFUTES"):
            for citation in cited:
                if citation not in valid_ids:
                    continue
                supportable += 1
                if audit.get(citation) == block["verdict"]:
                    supporting += 1

    return CitationScore(
        n_verdicts=n_verdicts,
        n_with_citations=n_with,
        grounded_rate=_safe_div(n_with, n_verdicts),
        n_citations=total_citations,
        validity=_safe_div(valid_citations, total_citations),
        support_precision=_safe_div(supporting, supportable),
        n_supportable=supportable,
    )


def fact_check_domains(records: Sequence[Dict]) -> set:
    """Registrable domains of the originating fact-check articles in a run.

    Derived from the corpus rather than hardcoded, so the set reflects the
    actual claims being evaluated.
    """
    return {
        d
        for d in (registered_domain(str(r.get("fact_check_url") or "")) for r in records)
        if d
    }


def leakage_exposure(records: Sequence[Dict], *, key: str = "multi") -> Dict[str, float]:
    """How often retrieval surfaced a fact-checking site despite the filter.

    The default leakage filter removes only the claim's *own* originating
    article. A different organisation's fact-check of the same claim states
    the same verdict and is not excluded, so the control is weaker than
    "the system never reads an answer". This quantifies the residual: the
    share of claims where some fact-checking domain still appeared.

    Returns:
        ``{"n_claims", "n_exposed", "rate", "n_filtered_claims"}``.
    """
    domains = fact_check_domains(records)
    n_claims = 0
    n_exposed = 0
    n_filtered = 0

    for record in records:
        block = record.get(key) or {}
        urls = block.get("retrieved_urls")
        if urls is None:
            continue
        n_claims += 1
        if any(registered_domain(u) in domains for u in urls):
            n_exposed += 1
        if any(
            "originating fact-check" in w for w in (block.get("warnings") or [])
        ):
            n_filtered += 1

    return {
        "n_claims": float(n_claims),
        "n_exposed": float(n_exposed),
        "rate": _safe_div(n_exposed, n_claims),
        "n_filtered_claims": float(n_filtered),
        "n_domains": float(len(domains)),
    }


def format_retrieval(score: RetrievalScore) -> str:
    """Render a retrieval scorecard as a fixed-width console block."""
    return "\n".join(
        [
            f"    {score.granularity:<8} (n={score.n_claims})",
            f"      Recall@k        {score.recall_at_k:6.1%}",
            f"      Precision@k     {score.precision_at_k:6.1%}",
            f"      hit rate        {score.hit_rate:6.1%}   (>=1 gold reference found)",
            f"      mean retrieved  {score.mean_retrieved:6.2f}",
            f"      mean gold       {score.mean_gold:6.2f}",
        ]
    )


def format_citations(score: CitationScore) -> str:
    """Render a citation scorecard as a fixed-width console block."""
    return "\n".join(
        [
            f"      verdicts cited  {score.n_with_citations}/{score.n_verdicts}"
            f"  ({score.grounded_rate:.1%})",
            f"      citations       {score.n_citations}",
            f"      validity        {score.validity:6.1%}   (names a retrieved passage)",
            f"      support prec.   {score.support_precision:6.1%}"
            f"   (audit agrees with verdict, n={score.n_supportable})",
        ]
    )


__all__ = [
    "CitationScore",
    "RetrievalScore",
    "fact_check_domains",
    "leakage_exposure",
    "format_citations",
    "format_retrieval",
    "score_citations",
    "score_one",
    "score_retrieval",
]
