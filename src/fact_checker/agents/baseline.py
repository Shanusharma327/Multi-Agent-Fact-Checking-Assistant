"""Baseline control -- monolithic single-LLM fact checker.

The experimental control for the whole project: one zero-shot call, no search,
no sub-agents, no evidence. It emits the same :class:`Adjudication` schema as
Agent 3 so the two verdicts can be compared field by field.
"""

from __future__ import annotations

import time

from ..config import Settings
from ..llm import build_llm, structured_call
from ..schemas import Adjudication, BaselineResult

SYSTEM_PROMPT = """You are a fact-checking assistant operating entirely from \
your own internal knowledge.

You have NO search tools, NO retrieved documents, and no way to look anything \
up. Judge the claim zero-shot, using only what you already know.

Assign one verdict:
- SUPPORTS         You believe the claim is true.
- REFUTES          You believe the claim is false.
- NOT_ENOUGH_INFO  You cannot determine this from what you know.

Be honest about the limits of parametric knowledge. If the claim depends on \
events after your training cutoff, on a fast-changing fact (a current \
officeholder, price, record, or ranking), or on details you are not confident \
you memorised correctly, say so and lower your confidence accordingly rather \
than guessing at a plausible-sounding answer.

Calibrate confidence as a probability that your verdict is correct:
- 0.90-1.00  A stable, well-established fact you are certain of.
- 0.70-0.89  Confident, but the detail is specific enough to misremember.
- 0.50-0.69  A plausible recollection you cannot verify.
- 0.25-0.49  Genuinely unsure.
- 0.00-0.24  Outside your knowledge, or past your cutoff.

detailed_reasoning must state what you are drawing on and flag any part that \
may be stale or half-remembered.

Set cited_sources to an empty list -- you have no sources to cite."""

USER_TEMPLATE = """Claim to evaluate from internal knowledge only:
\"\"\"{claim}\"\"\"

Give your zero-shot verdict."""


def run_baseline(claim: str, settings: Settings) -> BaselineResult:
    """Evaluate ``claim`` with a single ungrounded LLM call.

    Args:
        claim: The user-submitted claim.
        settings: Resolved runtime settings.

    Returns:
        A :class:`BaselineResult` wrapping an :class:`Adjudication` with an
        empty citation list and the wall-clock duration of the call.

    Raises:
        AgentError: If the LLM call fails.
        ConfigurationError: If no API key is configured.
    """
    started = time.perf_counter()
    llm = build_llm(settings, max_tokens=4096)
    adjudication: Adjudication = structured_call(
        llm,
        Adjudication,
        SYSTEM_PROMPT,
        USER_TEMPLATE.format(claim=claim.strip()),
        stage="Baseline (single LLM)",
    )
    elapsed = time.perf_counter() - started

    # The baseline is ungrounded by construction: any citation it produced is
    # fabricated, so it is discarded rather than displayed.
    adjudication = adjudication.model_copy(
        update={"claim": claim.strip(), "cited_sources": []}
    )
    return BaselineResult(claim=claim.strip(), adjudication=adjudication, seconds=elapsed)
