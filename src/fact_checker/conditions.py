"""The experimental conditions, each replayed over one cached evidence set.

Every condition here is a function of ``(CachedClaim, Settings)`` returning a
result record. Holding the evidence fixed is what makes the set a controlled
comparison rather than a collection of separate experiments:

====  ===================================================================
B0    Single-LLM, no retrieval. The original control. Varies retrieval and
      decomposition together, so it cannot isolate either.
B1    Single-call RAG over the cached passages. **The primary comparison**:
      same evidence, same schema, same model; only the decomposition differs.
B2    MAFCA without per-passage verification. Agent 3 reads the raw passages
      instead of an audit. Isolates the verification stage.
B3    MAFCA without post-conditions. Same two model calls, but audit
      reconciliation and citation validation are disabled. Isolates the
      deterministic repairs.
B4    MAFCA with the verdict computed in code from the audit (Eqs. 18-19).
      Replayed, no inference.
MAJ   Always the majority class of the evaluated sample.
RAND  Uniform random over the three labels, seeded.
====  ===================================================================

Costs differ and are reported rather than hidden: B0, B1 and B2 make one
model call per claim, MAFCA and B3 make two (verification and adjudication,
retrieval having been cached), and B4, MAJ and RAND make none.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Dict, List, Optional, Sequence

from .agents import run_adjudicator, run_auditor, run_baseline
from .agents.adjudicator import run_adjudicator_on_passages
from .agents.rag_baseline import run_rag_baseline
from .aggregation import aggregate
from .cache import CachedClaim
from .config import Settings
from .groundedness import audit_span_report
from .llm import AgentError, ConfigurationError
from .schemas import AuditReport, ScoutReport, VERDICT_VALUES

# A condition maps a cached claim to the fields of a result record.
Condition = Callable[[CachedClaim, Settings], dict]


def _base_record(entry: CachedClaim, condition: str) -> dict:
    """The fields every condition's record carries, however it was produced."""
    return {
        "id": entry.claim_id,
        "condition": condition,
        "claim": entry.claim,
        "gold": entry.gold,
        "dataset": entry.dataset,
        "split": entry.split,
        "gold_urls": list(entry.gold_urls),
        "fact_check_url": entry.fact_check_url,
        "retrieved_at": entry.retrieved_at,
        "evidence_fingerprint": entry.fingerprint(),
        "n_evidence": len(entry.passages),
        "retrieved_urls": [p.url for p in entry.passages],
        "evidence_ids": [p.source_id for p in entry.passages],
        "queries": list(entry.queries),
        "errors": [],
    }


def _verdict_fields(
    adjudication, *, seconds: float, llm_calls: int, **extra
) -> dict:
    """Normalise one system's output into the shape the scorers expect."""
    return {
        "verdict": adjudication.final_verdict,
        "confidence": adjudication.confidence,
        "cited_sources": list(adjudication.cited_sources),
        "reasoning": adjudication.detailed_reasoning,
        "seconds": seconds,
        "llm_calls": llm_calls,
        **extra,
    }


# ---------------------------------------------------------------------------
# B0 -- ungrounded single call
# ---------------------------------------------------------------------------

def condition_b0(entry: CachedClaim, settings: Settings) -> dict:
    """Single-LLM, no retrieval. Carries no fingerprint: it saw no evidence."""
    record = _base_record(entry, "B0")
    # B0 is excluded from the evidence-identity assertion by construction,
    # because it consumed none. Recording a fingerprint would assert a
    # relationship to evidence that does not exist.
    record.pop("evidence_fingerprint")
    record["n_evidence"] = 0
    record["retrieved_urls"] = []
    record["evidence_ids"] = []
    try:
        result = run_baseline(entry.claim, settings)
        record.update(
            _verdict_fields(result.adjudication, seconds=result.seconds, llm_calls=1)
        )
    except (AgentError, ConfigurationError) as exc:
        record["errors"].append(f"B0: {exc}")
    return record


# ---------------------------------------------------------------------------
# B1 -- single-call RAG. The primary comparison.
# ---------------------------------------------------------------------------

def condition_b1(entry: CachedClaim, settings: Settings) -> dict:
    """One prompt over the cached passages, same schema as MAFCA."""
    record = _base_record(entry, "B1")
    try:
        result, seconds = run_rag_baseline(
            entry.claim, entry.to_scout_report(), settings
        )
        record.update(
            _verdict_fields(result.adjudication, seconds=seconds, llm_calls=1)
        )
    except (AgentError, ConfigurationError) as exc:
        record["errors"].append(f"B1: {exc}")
    return record


# ---------------------------------------------------------------------------
# MAFCA and its ablations
# ---------------------------------------------------------------------------

def _run_mafca(
    entry: CachedClaim,
    settings: Settings,
    *,
    condition: str,
    reconcile: bool,
    validate_citations: bool,
    deterministic: bool,
    per_passage: bool = False,
) -> dict:
    """Shared body for MAFCA and the ablations that differ only in switches.

    Retrieval is replayed from the cache, so the two model calls are
    verification and adjudication. ``reconcile`` and ``validate_citations``
    are the two deterministic post-conditions; disabling both gives B3.
    """
    record = _base_record(entry, condition)
    scout: ScoutReport = entry.to_scout_report()
    started = time.perf_counter()
    calls = 0

    try:
        if not scout.evidence:
            # The empty-evidence short circuit: no model call, and the
            # verdict is insufficiency. Recorded explicitly so the issue-6
            # analysis can separate these from substantive judgements.
            audit = AuditReport(alignments=[])
            record["short_circuit"] = True
        else:
            audit = run_auditor(
                entry.claim,
                scout,
                settings,
                reconcile=reconcile,
                per_passage=per_passage,
            )
            calls += len(scout.evidence) if per_passage else 1
            record["short_circuit"] = False

        adjudication, trace = run_adjudicator(
            entry.claim,
            scout,
            audit,
            settings,
            deterministic=deterministic,
            beta=settings.aggregation_beta,
            gate_citations=settings.gate_citations,
            validate_citations=validate_citations,
        )
        calls += 1

        seconds = time.perf_counter() - started
        record.update(
            _verdict_fields(
                adjudication,
                seconds=seconds,
                llm_calls=calls,
                audit=[
                    {
                        "source_id": a.source_id,
                        "verdict": a.verdict,
                        "justification": a.justification,
                    }
                    for a in audit.alignments
                ],
                audit_tally=audit.tally(),
                aggregation_trace=trace.model_dump() if trace is not None else None,
                span_check=audit_span_report(audit, scout).model_dump(),
            )
        )
    except (AgentError, ConfigurationError) as exc:
        record["errors"].append(f"{condition}: {exc}")
    return record


def condition_mafca(entry: CachedClaim, settings: Settings) -> dict:
    """MAFCA as published: agent-reported verdict, post-conditions on."""
    return _run_mafca(
        entry, settings, condition="MAFCA",
        reconcile=True, validate_citations=True, deterministic=False,
    )


def condition_b2(entry: CachedClaim, settings: Settings) -> dict:
    """MAFCA with the verification stage removed, keeping Agent 3's framing.

    Agent 3 receives the raw passages where it would normally receive the
    audit, and is otherwise prompted exactly as in MAFCA -- same role, same
    four adjudication rules, same confidence rubric, same citation
    instruction.

    The distinction from B1 is worth stating, because the two are easy to
    conflate. B1 is a purpose-written single-call RAG system: the strongest
    flat baseline we can build. B2 is MAFCA *minus one stage*, which keeps
    the three-agent prompt architecture but deletes the per-passage
    judgement. Running B2 with B1's prompt would make the two conditions
    byte-identical at temperature 0 and the ablation vacuous; running it
    with Agent 3's prompt asks the question the ablation is for -- what does
    the verification stage contribute, holding the adjudicator's framing
    fixed.
    """
    record = _base_record(entry, "B2")
    scout = entry.to_scout_report()
    started = time.perf_counter()
    try:
        adjudication = run_adjudicator_on_passages(entry.claim, scout, settings)
        record.update(
            _verdict_fields(
                adjudication,
                seconds=time.perf_counter() - started,
                llm_calls=1,
            )
        )
        record["short_circuit"] = not scout.evidence
    except (AgentError, ConfigurationError) as exc:
        record["errors"].append(f"B2: {exc}")
    return record


def condition_b3(entry: CachedClaim, settings: Settings) -> dict:
    """MAFCA with both deterministic post-conditions disabled.

    Same two model calls and the same prompts as MAFCA; only audit
    reconciliation and citation validation are off. Any difference is
    attributable to the post-conditions alone, which is what issue 17 asks.
    """
    return _run_mafca(
        entry, settings, condition="B3",
        reconcile=False, validate_citations=False, deterministic=False,
    )


def condition_b4(entry: CachedClaim, settings: Settings) -> dict:
    """MAFCA with the verdict computed in code from the audit."""
    return _run_mafca(
        entry, settings, condition="B4",
        reconcile=True, validate_citations=True, deterministic=True,
    )


def condition_b6(entry: CachedClaim, settings: Settings) -> dict:
    """MAFCA with one verification call per passage (Tier 2, issue 4).

    Makes the independence of Eq. (12) a property of the execution rather
    than a request in a prompt, at a cost of n calls instead of one.
    """
    return _run_mafca(
        entry, settings, condition="B6",
        reconcile=True, validate_citations=True, deterministic=False,
        per_passage=True,
    )


# ---------------------------------------------------------------------------
# Trivial reference points
# ---------------------------------------------------------------------------

def majority_label(entries: Sequence[CachedClaim]) -> str:
    """The most frequent gold label in the evaluated sample.

    Computed from the sample actually evaluated, not from the corpus: a
    stratified draw and its parent split have different majorities, and the
    reference point that matters is the one for these claims.
    """
    counts: Dict[str, int] = {}
    for entry in entries:
        if entry.gold:
            counts[entry.gold] = counts.get(entry.gold, 0) + 1
    if not counts:
        return VERDICT_VALUES[0]
    return max(sorted(counts), key=lambda label: counts[label])


def condition_majority(
    entries: Sequence[CachedClaim], label: Optional[str] = None
) -> List[dict]:
    """Always predict the sample's majority class. No model, no evidence."""
    chosen = label or majority_label(entries)
    records = []
    for entry in entries:
        record = _base_record(entry, "MAJORITY")
        record.pop("evidence_fingerprint")
        record.update(
            {
                "verdict": chosen,
                # A constant predictor has no evidential basis for any
                # confidence; the sample's base rate would leak the label
                # distribution into a "calibration" result.
                "confidence": 1.0 / len(VERDICT_VALUES),
                "cited_sources": [],
                "reasoning": f"Constant majority-class predictor ({chosen}).",
                "seconds": 0.0,
                "llm_calls": 0,
            }
        )
        records.append(record)
    return records


def condition_random(
    entries: Sequence[CachedClaim], *, seed: int = 20240101
) -> List[dict]:
    """Uniform random over the three labels, seeded for reproducibility."""
    rng = random.Random(seed)
    records = []
    for entry in entries:
        record = _base_record(entry, "RANDOM")
        record.pop("evidence_fingerprint")
        record.update(
            {
                "verdict": rng.choice(list(VERDICT_VALUES)),
                "confidence": 1.0 / len(VERDICT_VALUES),
                "cited_sources": [],
                "reasoning": "Uniform random predictor.",
                "seconds": 0.0,
                "llm_calls": 0,
            }
        )
        records.append(record)
    return records


# Conditions that run per claim and may call a model.
CONDITIONS: Dict[str, Condition] = {
    "B0": condition_b0,
    "B1": condition_b1,
    "MAFCA": condition_mafca,
    "B2": condition_b2,
    "B3": condition_b3,
    "B4": condition_b4,
    "B6": condition_b6,
}

# Conditions computed over the whole sample at once, with no model.
SAMPLE_CONDITIONS = ("MAJORITY", "RANDOM")

# Per-claim model calls, for the cost table. Retrieval is cached, so these
# count inference only; B6 is n-dependent and reported from the records.
LLM_CALLS = {"B0": 1, "B1": 1, "B2": 1, "MAFCA": 2, "B3": 2, "B4": 2,
             "MAJORITY": 0, "RANDOM": 0}


__all__ = [
    "CONDITIONS",
    "LLM_CALLS",
    "SAMPLE_CONDITIONS",
    "condition_b0",
    "condition_b1",
    "condition_b2",
    "condition_b3",
    "condition_b4",
    "condition_b6",
    "condition_mafca",
    "condition_majority",
    "condition_random",
    "majority_label",
]
