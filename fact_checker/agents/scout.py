"""Agent 1 -- Evidence Scout.

Turns a claim into two neutral search queries, runs them, and returns the
retrieved snippets under stable ``doc_N`` identifiers.
"""

from __future__ import annotations

from ..config import Settings
from ..llm import build_llm, structured_call
from ..schemas import ScoutReport, SearchQueries
from ..search import RetrievalOutcome, gather_evidence

SYSTEM_PROMPT = """You are the Evidence Scout in a fact-checking pipeline.

Your ONLY job is to write web search queries that will surface the evidence \
needed to evaluate a claim. You never judge the claim yourself.

Rules for the queries you write:
1. Write EXACTLY 2 queries.
2. Stay neutral. Never phrase a query so that it presupposes the claim is true \
or false. Search for the underlying fact, not for confirmation.
   - Bad:  "proof that X is the worst Y ever"
   - Good: "X Y ranking official figures"
3. Query 1 targets the core entity, event, or number in the claim.
4. Query 2 approaches the same fact from an independent angle -- an official \
record, a primary source, a statistic, or the current status of the entity -- \
so that a single biased corner of the web cannot dominate the evidence.
5. Use short keyword phrases as a person would type into a search engine. No \
quotation marks, no boolean operators, no site: filters.
6. If the claim is time-sensitive (uses "current", "latest", "now", "still", \
or a year), put the relevant time anchor into at least one query.

Return only the two queries."""

USER_TEMPLATE = """Claim to investigate:
\"\"\"{claim}\"\"\"

Write the 2 neutral search queries."""


def run_scout(claim: str, settings: Settings) -> tuple[ScoutReport, list[str]]:
    """Generate queries and retrieve supporting evidence for ``claim``.

    Args:
        claim: The user-submitted claim.
        settings: Resolved runtime settings.

    Returns:
        A tuple of the :class:`ScoutReport` and any non-fatal warnings raised
        during retrieval (for example, search rate limiting).

    Raises:
        AgentError: If query generation fails.
        ConfigurationError: If no API key is configured.
    """
    llm = build_llm(settings, max_tokens=1024)
    plan = structured_call(
        llm,
        SearchQueries,
        SYSTEM_PROMPT,
        USER_TEMPLATE.format(claim=claim.strip()),
        stage="Agent 1 (Evidence Scout)",
    )

    # Fall back to the raw claim if the model returned nothing usable, so that
    # retrieval still gets a chance rather than the run ending empty-handed.
    queries = plan.queries[:2] or [claim.strip()]

    outcome: RetrievalOutcome = gather_evidence(
        queries,
        results_per_query=settings.results_per_query,
        max_evidence=settings.max_evidence,
        snippet_chars=settings.snippet_chars,
    )

    report = ScoutReport(queries=queries, evidence=outcome.evidence)
    return report, outcome.warnings
