"""Agent 3 -- Synthesis Adjudicator.

Weighs Agent 2's per-snippet alignments into a single calibrated verdict.
"""

from __future__ import annotations

import re

from ..config import Settings
from ..llm import build_llm, structured_call
from ..schemas import Adjudication, AuditReport, ScoutReport

SYSTEM_PROMPT = """You are the Synthesis Adjudicator, the final authority in a \
fact-checking pipeline.

You receive a claim and a list of per-snippet audits produced by an upstream \
NLI agent. You do not see the raw web pages and you must not introduce facts \
from your own memory. Your verdict is a synthesis of the audits in front of you.

How to weigh the audits:
1. Direct evidence beats volume. One snippet that squarely settles the claim \
outranks four snippets that merely touch the topic.
2. On genuine conflict between SUPPORTS and REFUTES, prefer the side whose \
justification quotes specific figures, dates, or official records over the side \
that is vague or general. Say in your reasoning which side you discounted and why.
3. If the only signal is NOT_ENOUGH_INFO, the verdict is NOT_ENOUGH_INFO. \
Absence of evidence is never evidence of refutation.
4. A claim that is partially true but wrong in its specific assertion (wrong \
number, wrong date, wrong superlative) is REFUTES, not SUPPORTS.
5. With no evidence at all, return NOT_ENOUGH_INFO at confidence 0.1 and say \
plainly that retrieval produced nothing.

Calibrate confidence honestly -- it is a probability that your verdict is \
correct, not a measure of your enthusiasm:
- 0.90-1.00  Multiple independent snippets agree and quote specifics.
- 0.70-0.89  Clear evidence from one strong snippet, or agreement without specifics.
- 0.50-0.69  Evidence leans one way but is indirect, dated, or thin.
- 0.25-0.49  Conflicting or mostly tangential evidence.
- 0.00-0.24  Effectively no usable evidence.

detailed_reasoning must walk through your actual steps: what the evidence \
showed, how you resolved any conflict, and what remains uncertain.

cited_sources must list ONLY the doc_N identifiers that genuinely drove your \
verdict. Never invent an identifier, and do not pad the list with snippets you \
ignored. If you cite nothing, return an empty list."""

USER_TEMPLATE = """Claim under evaluation:
\"\"\"{claim}\"\"\"

Available source_ids: {ids}

Per-snippet audit from the Alignment Auditor:
{audit_block}

Label tally: {tally}

Synthesise the unified verdict."""

EMPTY_AUDIT_BLOCK = "(No evidence was retrieved. The audit list is empty.)"


def _format_audit(audit: AuditReport) -> str:
    """Render the audit list for the prompt."""
    if not audit.alignments:
        return EMPTY_AUDIT_BLOCK
    return "\n".join(
        f"- [{item.source_id}] {item.verdict}: {item.justification}"
        for item in audit.alignments
    )


_DOC_REFERENCE = re.compile(r"\bdoc_\d+\b")


def _recover_citations(reasoning: str, valid_ids: set[str]) -> list[str]:
    """Pull ``doc_N`` references out of the reasoning prose.

    Smaller models routinely cite a snippet inline ("[doc_3] states that...")
    while leaving ``cited_sources`` empty. Those are citations the model really
    made, so recovering them is repair rather than invention -- and every id is
    still checked against genuinely retrieved evidence before it is kept.
    """
    found = _DOC_REFERENCE.findall(reasoning or "")
    return list(dict.fromkeys(doc for doc in found if doc in valid_ids))


def _sanitise_citations(adjudication: Adjudication, valid_ids: set[str]) -> Adjudication:
    """Drop fabricated identifiers, and recover ones stranded in the prose."""
    kept = [source_id for source_id in adjudication.cited_sources if source_id in valid_ids]
    # De-duplicate while preserving the model's ordering.
    deduped = list(dict.fromkeys(kept))

    if not deduped:
        deduped = _recover_citations(adjudication.detailed_reasoning, valid_ids)

    if deduped != adjudication.cited_sources:
        return adjudication.model_copy(update={"cited_sources": deduped})
    return adjudication


def run_adjudicator(
    claim: str,
    scout: ScoutReport,
    audit: AuditReport,
    settings: Settings,
) -> Adjudication:
    """Synthesise a final verdict from the per-snippet audit.

    Args:
        claim: The user-submitted claim.
        scout: Agent 1's report, used to validate citations.
        audit: Agent 2's per-snippet alignments.
        settings: Resolved runtime settings.

    Returns:
        A validated :class:`Adjudication` whose ``claim`` is the user's exact
        wording and whose citations are guaranteed to reference real evidence.

    Raises:
        AgentError: If the LLM call fails.
        ConfigurationError: If no API key is configured.
    """
    valid_ids = set(scout.source_ids())
    llm = build_llm(settings, max_tokens=4096)
    adjudication = structured_call(
        llm,
        Adjudication,
        SYSTEM_PROMPT,
        USER_TEMPLATE.format(
            claim=claim.strip(),
            ids=", ".join(sorted(valid_ids)) or "(none)",
            audit_block=_format_audit(audit),
            tally=audit.tally(),
        ),
        stage="Agent 3 (Synthesis Adjudicator)",
    )

    # Pin the claim to the user's exact wording; a restatement would quietly
    # change what the verdict is about.
    adjudication = adjudication.model_copy(update={"claim": claim.strip()})
    return _sanitise_citations(adjudication, valid_ids)
