"""Attribute each wrong verdict to the stage that caused it.

A 50%-accurate system is not informative until one knows *where* it fails.
The decomposition makes that answerable in a way a monolithic system does
not: because retrieval, per-passage verification and aggregation leave
separate records, a wrong verdict can be traced to the first stage that
went wrong.

The attribution is a cascade, assigned to the earliest stage that failed,
because a later stage cannot be blamed for working on bad input:

**retrieval**
    No retrieved passage was judged to bear on the claim in the gold
    direction, and none came from a gold-annotated source. There was
    nothing in the evidence set for the later stages to get right.

**verification**
    A passage *was* retrieved from a gold source, or was labelled in the
    gold direction, but the audit as a whole points the wrong way -- the
    evidence was there and the per-passage judgements mishandled it.

**aggregation**
    The audit's weighted mass points at the gold label, and the verdict
    does not. The evidence was retrieved and correctly labelled, and the
    step that combined those labels lost it.

**unattributed**
    A wrong verdict none of the above explains, most often where the gold
    label is NOT_ENOUGH_INFO and the system committed. Reported rather
    than forced into a category.

The retrieval test is deliberately generous to the retriever: a claim
counts as a retrieval failure only when the evidence contains nothing
usable at all. That biases the breakdown *against* blaming retrieval, so
a large retrieval share is a conservative finding.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fact_checker.aggregation import aggregate  # noqa: E402
from fact_checker.schemas import Alignment, AuditReport  # noqa: E402
from fact_checker.urls import domain_set  # noqa: E402

STAGES = ("retrieval", "verification", "aggregation", "unattributed")


class Attribution(BaseModel):
    """Why one claim's verdict was wrong."""

    claim_id: str
    gold: str
    verdict: str
    stage: str
    reason: str
    n_evidence: int = 0
    gold_in_audit: bool = Field(
        default=False, description="Some passage was labelled in the gold direction."
    )
    gold_source_retrieved: bool = Field(
        default=False, description="Some retrieved URL shares a domain with gold."
    )
    aggregated_label: str = Field(
        default="", description="What the weighted audit mass pointed at."
    )


def _rebuild_audit(record: dict) -> AuditReport:
    return AuditReport(
        alignments=[
            Alignment(
                source_id=str(entry.get("source_id", "")),
                verdict=entry.get("verdict", "NOT_ENOUGH_INFO"),
                justification=str(entry.get("justification", "")),
            )
            for entry in (record.get("audit") or [])
            if entry.get("source_id")
        ]
    )


def attribute(record: dict, *, beta: float = 1.0) -> Attribution:
    """Assign one wrong verdict to the earliest stage that failed."""
    gold = str(record.get("gold", ""))
    verdict = str(record.get("verdict", ""))
    audit = _rebuild_audit(record)
    n_evidence = int(record.get("n_evidence", 0))

    gold_in_audit = any(a.verdict == gold for a in audit.alignments)
    retrieved_domains = domain_set(record.get("retrieved_urls") or [])
    gold_domains = domain_set(record.get("gold_urls") or [])
    gold_source_retrieved = bool(retrieved_domains & gold_domains)

    aggregated = aggregate(audit, beta=beta).verdict if audit.alignments else ""

    common = dict(
        claim_id=str(record.get("id", "")),
        gold=gold,
        verdict=verdict,
        n_evidence=n_evidence,
        gold_in_audit=gold_in_audit,
        gold_source_retrieved=gold_source_retrieved,
        aggregated_label=aggregated,
    )

    if n_evidence == 0:
        return Attribution(
            stage="retrieval", reason="retrieval returned nothing", **common
        )

    if not gold_in_audit and not gold_source_retrieved:
        return Attribution(
            stage="retrieval",
            reason="no retrieved passage was judged in the gold direction, and "
                   "none came from a gold-annotated source",
            **common,
        )

    if aggregated == gold and verdict != gold:
        return Attribution(
            stage="aggregation",
            reason=f"the weighted audit pointed at {gold} but the verdict was {verdict}",
            **common,
        )

    if gold_in_audit or gold_source_retrieved:
        return Attribution(
            stage="verification",
            reason="usable evidence was present but the audit did not carry it",
            **common,
        )

    return Attribution(stage="unattributed", reason="no stage explains this", **common)


def attribute_run(records: Sequence[dict], *, beta: float = 1.0) -> Dict[str, object]:
    """Attribute every wrong verdict in a run and summarise the breakdown."""
    wrong = [
        r for r in records
        if r.get("verdict") and r.get("gold") and r["verdict"] != r["gold"]
    ]
    attributions = [attribute(r, beta=beta) for r in wrong]

    counts = {stage: 0 for stage in STAGES}
    for item in attributions:
        counts[item.stage] += 1

    total_scored = sum(
        1 for r in records if r.get("verdict") and r.get("gold")
    )
    return {
        "n_scored": total_scored,
        "n_wrong": len(wrong),
        "counts": counts,
        "shares": {
            stage: (counts[stage] / len(wrong) if wrong else 0.0) for stage in STAGES
        },
        "attributions": [a.model_dump() for a in attributions],
    }


def format_attribution(summary: Dict[str, object], name: str = "") -> str:
    """Render the breakdown as a console block."""
    counts: Dict[str, int] = summary["counts"]  # type: ignore[assignment]
    shares: Dict[str, float] = summary["shares"]  # type: ignore[assignment]
    lines = [
        f"  {name or 'condition'}: {summary['n_wrong']} wrong of "
        f"{summary['n_scored']} scored"
    ]
    for stage in STAGES:
        if counts[stage] or stage != "unattributed":
            lines.append(
                f"    {stage:<14}{counts[stage]:>5}  ({100 * shares[stage]:5.1f}% of errors)"
            )
    return "\n".join(lines)


def examples(summary: Dict[str, object], stage: str, limit: int = 3) -> List[dict]:
    """A few attributed claims from one stage, for qualitative inspection."""
    return [
        a for a in summary["attributions"]  # type: ignore[union-attr]
        if a["stage"] == stage
    ][:limit]


__all__ = [
    "Attribution",
    "STAGES",
    "attribute",
    "attribute_run",
    "examples",
    "format_attribution",
]
