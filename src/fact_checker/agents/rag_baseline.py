"""B1 -- single-call RAG baseline. The primary comparison for MAFCA.

MAFCA against the ungrounded baseline (B0) varies two things at once:
whether the system has evidence, and whether the task is decomposed. Any
difference between them is therefore uninterpretable as a result about
decomposition, which is what the paper claims to be testing.

B1 removes the confound. It receives the *same cached passages* MAFCA's
verification stage received, emits the *same* ``Adjudication`` schema, and
runs on the same model at the same temperature. The only remaining
difference is the decomposition: B1 retrieves, verifies, weighs and explains
in one forward pass, where MAFCA splits those across three.

**This prompt is deliberately strong.** A weak RAG prompt would manufacture
the result the paper wants, so the instructions below give B1 every
advantage the pipeline gives its own agents: the closed-world restriction
from the Alignment Auditor, the four adjudication rules from the Synthesis
Adjudicator, the same confidence rubric anchored to evidential conditions,
and an explicit instruction to cite the identifiers it used. The only thing
withheld is the thing under test -- being made to label each passage
separately before deciding.
"""

from __future__ import annotations

import time
from typing import Tuple

from ..config import Settings
from ..llm import build_llm, structured_call
from ..schemas import Adjudication, BaselineResult, ScoutReport

SYSTEM_PROMPT = """You are a fact-checking assistant. You are given a claim \
and a numbered set of evidence passages retrieved from the web, and you must \
decide whether the evidence establishes the claim.

CLOSED-WORLD RULE (absolute): base your verdict ONLY on the passages \
supplied below. You have no outside memory. If you personally believe the \
claim is true but no passage says so, the verdict is NOT_ENOUGH_INFO. Your \
own world knowledge is not evidence and must never appear in your reasoning.

Work through the passages before deciding. For each one, consider whether it \
states or directly entails that the claim is true, states or directly \
entails that it is false, or fails to settle the claim. Watch for these traps:
- Partial overlap. A passage confirming the subject exists does NOT support a \
claim about that subject's specific attribute.
- Near-miss numbers and dates. "1971" does not support a claim that says \
1969; that is REFUTES, not SUPPORTS.
- Superseded facts. If a passage describes a past state and the claim asserts \
a present one, that is usually NOT_ENOUGH_INFO unless the passage gives the \
current state.

Then decide, weighing the passages as follows:
1. Direct evidence beats volume. One passage that squarely settles the claim \
outranks four that merely touch the topic.
2. On genuine conflict, prefer the side whose passage quotes specific \
figures, dates, or official records over the side that is vague or general. \
Say in your reasoning which side you discounted and why.
3. Absence of evidence is never evidence of refutation. If no passage settles \
the claim, the verdict is NOT_ENOUGH_INFO.
4. A claim that is partially true but wrong in its specific assertion (wrong \
number, wrong date, wrong superlative) is REFUTES, not SUPPORTS.
5. With no usable passages at all, return NOT_ENOUGH_INFO at confidence 0.1 \
and say plainly that retrieval produced nothing.

Assign one verdict:
- SUPPORTS         The passages establish that the claim is true.
- REFUTES          The passages establish that the claim is false.
- NOT_ENOUGH_INFO  The passages do not settle the claim either way.

Calibrate confidence honestly -- it is a probability that your verdict is \
correct, not a measure of your enthusiasm:
- 0.90-1.00  Multiple independent passages agree and quote specifics.
- 0.70-0.89  Clear evidence from one strong passage, or agreement without specifics.
- 0.50-0.69  Evidence leans one way but is indirect, dated, or thin.
- 0.25-0.49  Conflicting or mostly tangential evidence.
- 0.00-0.24  Effectively no usable evidence.

detailed_reasoning must walk through what the passages showed, how you \
resolved any conflict, and what remains uncertain. Quote the exact words or \
numbers that decided it.

cited_sources must list ONLY the doc_N identifiers that genuinely drove your \
verdict. Never invent an identifier, and do not pad the list with passages \
you ignored. If you cite nothing, return an empty list."""

USER_TEMPLATE = """Claim under evaluation:
\"\"\"{claim}\"\"\"

Evidence passages ({count} total):

{evidence_block}

Available source_ids: {ids}

Give your verdict."""

EMPTY_EVIDENCE_BLOCK = "(No evidence was retrieved for this claim.)"


def _format_evidence(scout: ScoutReport) -> str:
    """Render the passages exactly as the Alignment Auditor sees them.

    Byte-identical formatting matters: a difference in how the evidence is
    presented would be a second uncontrolled variable between B1 and MAFCA.
    """
    if not scout.evidence:
        return EMPTY_EVIDENCE_BLOCK
    return "\n\n".join(
        f"[{item.source_id}]\n{item.snippet}" for item in scout.evidence
    )


def run_rag_baseline(
    claim: str, scout: ScoutReport, settings: Settings
) -> Tuple[BaselineResult, float]:
    """Decide ``claim`` from ``scout``'s passages in a single model call.

    Args:
        claim: The claim to verify.
        scout: The cached evidence set, identical to MAFCA's.
        settings: Resolved runtime settings.

    Returns:
        ``(result, seconds)``. Citations are validated against the retrieved
        identifiers, exactly as Agent 3's are, so neither system can score a
        citation it invented.

    Raises:
        AgentError: If the LLM call fails.
        ConfigurationError: If no backend is configured.
    """
    started = time.perf_counter()
    valid_ids = set(scout.source_ids())

    llm = build_llm(settings, max_tokens=4096)
    adjudication: Adjudication = structured_call(
        llm,
        Adjudication,
        SYSTEM_PROMPT,
        USER_TEMPLATE.format(
            claim=claim.strip(),
            count=len(scout.evidence),
            evidence_block=_format_evidence(scout),
            ids=", ".join(sorted(valid_ids)) or "(none)",
        ),
        stage="B1 (single-call RAG)",
    )
    elapsed = time.perf_counter() - started

    # Same citation post-condition as Agent 3: an identifier that names no
    # retrieved passage is dropped. Applying it to one system and not the
    # other would hand MAFCA a citation-validity advantage it did not earn.
    citations = [c for c in adjudication.cited_sources if c in valid_ids]
    adjudication = adjudication.model_copy(
        update={
            "claim": claim.strip(),
            "cited_sources": list(dict.fromkeys(citations)),
        }
    )
    return (
        BaselineResult(claim=claim.strip(), adjudication=adjudication, seconds=elapsed),
        elapsed,
    )


__all__ = ["run_rag_baseline", "SYSTEM_PROMPT", "USER_TEMPLATE"]
