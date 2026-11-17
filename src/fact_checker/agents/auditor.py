"""Agent 2 -- Alignment Auditor (natural language inference).

Labels each retrieved snippet independently against the claim under a strict
closed-world assumption: only the snippet's own text may justify a label.
"""

from __future__ import annotations

from ..config import Settings
from ..llm import build_llm, structured_call
from ..schemas import Alignment, AuditReport, ScoutReport

SYSTEM_PROMPT = """You are the Alignment Auditor, a strict natural language \
inference engine inside a fact-checking pipeline.

CLOSED-WORLD RULE (absolute): judge each snippet using ONLY the words in that \
snippet. You have no outside memory. If you personally know the claim is true \
but the snippet does not say so, the label is NOT_ENOUGH_INFO. Your own world \
knowledge is not evidence and must never appear in a justification.

Judge every snippet INDEPENDENTLY. Snippet doc_3 may not borrow context from \
doc_1. Do not let a majority among the snippets pull an individual label.

Assign exactly one label per snippet:
- SUPPORTS         The snippet states, or directly entails, that the claim is true.
- REFUTES          The snippet states, or directly entails, that the claim is false, \
including when it asserts a different value, date, name, or ranking than the claim does.
- NOT_ENOUGH_INFO  The snippet is about a related topic but does not settle the \
claim, is too vague, or is off-topic entirely.

Watch for these traps:
- Partial overlap. A snippet confirming the subject exists does NOT support a \
claim about that subject's specific attribute.
- Near-miss numbers and dates. "1971" does not support a claim that says 1969; \
that is REFUTES, not SUPPORTS.
- Superseded facts. If the snippet describes a past state and the claim asserts \
a present one, that is usually NOT_ENOUGH_INFO unless the snippet gives the \
current state.

Every justification must be ONE sentence and must quote the exact words or \
numbers from the snippet that decided the label.

Produce exactly one alignment for every source_id given to you -- no more, no \
fewer, and never invent an identifier."""

USER_TEMPLATE = """Claim under evaluation:
\"\"\"{claim}\"\"\"

Snippets to audit ({count} total). Judge each one on its own:

{evidence_block}

Return one alignment for each of these source_ids: {ids}"""


def _format_evidence(scout: ScoutReport) -> str:
    """Render the evidence list into a stable, clearly delimited block."""
    return "\n\n".join(
        f"[{item.source_id}]\n{item.snippet}" for item in scout.evidence
    )


def _reconcile(report: AuditReport, expected_ids: list[str]) -> AuditReport:
    """Force a one-to-one mapping between snippets and alignments.

    Hallucinated identifiers are dropped and omitted snippets are backfilled as
    NOT_ENOUGH_INFO, so Agent 3 always receives a complete, honest audit.
    """
    by_id: dict[str, Alignment] = {}
    for alignment in report.alignments:
        if alignment.source_id in expected_ids and alignment.source_id not in by_id:
            by_id[alignment.source_id] = alignment

    reconciled = [
        by_id.get(
            source_id,
            Alignment(
                source_id=source_id,
                verdict="NOT_ENOUGH_INFO",
                justification="The auditor returned no judgement for this snippet.",
            ),
        )
        for source_id in expected_ids
    ]
    return AuditReport(alignments=reconciled)


def _audit_one_call(claim: str, scout: ScoutReport, settings: Settings) -> AuditReport:
    """Judge every passage in a single call, all of them sharing a context.

    This is the original implementation, and it is worth being precise about
    what it does and does not guarantee. The prompt instructs the model to
    judge each snippet independently, but every snippet occupies the same
    context window, so independence is *requested* rather than enforced.
    Equation (12) therefore describes an intended property of this path, not
    a structural one; :func:`_audit_per_passage` is the variant that makes it
    structural, at n times the cost.
    """
    expected_ids = scout.source_ids()
    llm = build_llm(settings, max_tokens=4096)
    return structured_call(
        llm,
        AuditReport,
        SYSTEM_PROMPT,
        USER_TEMPLATE.format(
            claim=claim.strip(),
            count=len(expected_ids),
            evidence_block=_format_evidence(scout),
            ids=", ".join(expected_ids),
        ),
        stage="Agent 2 (Alignment Auditor)",
    )


def _audit_per_passage(
    claim: str, scout: ScoutReport, settings: Settings
) -> AuditReport:
    """Judge each passage in its own call, so no two can influence each other.

    This makes the independence of Equation (12) a property of the execution
    rather than of the prompt: a passage cannot be swayed by a majority among
    its neighbours when the model never sees them. The cost is one call per
    passage instead of one per claim.
    """
    llm = build_llm(settings, max_tokens=1024)
    alignments: list[Alignment] = []

    for item in scout.evidence:
        single = ScoutReport(queries=scout.queries, evidence=[item])
        report = structured_call(
            llm,
            AuditReport,
            SYSTEM_PROMPT,
            USER_TEMPLATE.format(
                claim=claim.strip(),
                count=1,
                evidence_block=_format_evidence(single),
                ids=item.source_id,
            ),
            stage=f"Agent 2 (Alignment Auditor, {item.source_id})",
        )
        # The call saw one passage, so any alignment it returned is about
        # that passage whatever identifier it chose to write.
        if report.alignments:
            alignments.append(
                report.alignments[0].model_copy(update={"source_id": item.source_id})
            )

    return AuditReport(alignments=alignments)


def run_auditor(
    claim: str,
    scout: ScoutReport,
    settings: Settings,
    *,
    reconcile: bool = True,
    per_passage: bool = False,
) -> AuditReport:
    """Label every snippet in ``scout`` against ``claim``.

    Args:
        claim: The user-submitted claim.
        scout: Agent 1's report, providing the snippets to audit.
        settings: Resolved runtime settings.
        reconcile: Apply the audit-reconciliation post-condition, which drops
            hallucinated identifiers and backfills omitted ones. Disabled by
            the B3 ablation to isolate what the post-conditions contribute.
        per_passage: Issue one call per passage instead of one per claim,
            making Equation (12)'s independence structural rather than
            requested.

    Returns:
        An :class:`AuditReport`. With ``reconcile`` set there is exactly one
        alignment per snippet. An empty evidence list yields an empty report
        without an LLM call.

    Raises:
        AgentError: If the LLM call fails.
        ConfigurationError: If no API key is configured.
    """
    if not scout.evidence:
        return AuditReport(alignments=[])

    report = (
        _audit_per_passage(claim, scout, settings)
        if per_passage
        else _audit_one_call(claim, scout, settings)
    )
    return _reconcile(report, scout.source_ids()) if reconcile else report
