"""Orchestration for the three-agent pipeline and the baseline control.

The UI drives the stages one at a time (so it can render progress between
them), while :func:`run_multi_agent` offers the same sequence as a single
call for scripted or headless use.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional

from .agents import run_adjudicator, run_auditor, run_baseline, run_scout
from .config import Settings
from .llm import AgentError, ConfigurationError
from .schemas import (
    Adjudication,
    AuditReport,
    BaselineResult,
    MultiAgentResult,
    ScoutReport,
    StageTiming,
)

logger = logging.getLogger(__name__)

# Called with a short status string as each stage begins.
ProgressCallback = Callable[[str], None]


@contextmanager
def _timed(name: str, sink: List[StageTiming]) -> Iterator[None]:
    """Record the wall-clock duration of a stage into ``sink``."""
    started = time.perf_counter()
    try:
        yield
    finally:
        sink.append(StageTiming(name=name, seconds=time.perf_counter() - started))


class FactCheckPipeline:
    """Stage-by-stage access to the fact-checking agents.

    Each method is independently callable so a caller (the Streamlit app) can
    interleave rendering between stages; :meth:`run` chains them end to end.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.timings: List[StageTiming] = []
        self.warnings: List[str] = []

    def reset(self) -> None:
        """Clear timings and warnings before a fresh run."""
        self.timings = []
        self.warnings = []

    def scout(self, claim: str) -> ScoutReport:
        """Stage 1: generate queries and retrieve evidence."""
        with _timed("Agent 1 · Evidence Scout", self.timings):
            report, warnings = run_scout(claim, self.settings)
        self.warnings.extend(warnings)
        return report

    def audit(self, claim: str, scout: ScoutReport) -> AuditReport:
        """Stage 2: label each snippet against the claim."""
        with _timed("Agent 2 · Alignment Auditor", self.timings):
            return run_auditor(claim, scout, self.settings)

    def adjudicate(
        self, claim: str, scout: ScoutReport, audit: AuditReport
    ) -> Adjudication:
        """Stage 3: synthesise the unified verdict."""
        with _timed("Agent 3 · Synthesis Adjudicator", self.timings):
            return run_adjudicator(claim, scout, audit, self.settings)

    def baseline(self, claim: str) -> BaselineResult:
        """Control: single ungrounded LLM call."""
        return run_baseline(claim, self.settings)

    def run(
        self, claim: str, on_progress: Optional[ProgressCallback] = None
    ) -> MultiAgentResult:
        """Execute all three agents in sequence.

        Args:
            claim: The claim to verify.
            on_progress: Optional callback invoked with a status string as each
                stage starts.

        Returns:
            A :class:`MultiAgentResult` with the full trace, timings, and any
            non-fatal warnings.

        Raises:
            AgentError: If any agent's LLM call fails.
            ConfigurationError: If no API key is configured.
        """
        self.reset()
        claim = claim.strip()

        def notify(message: str) -> None:
            if on_progress is not None:
                on_progress(message)

        notify("Agent 1 · generating queries and retrieving evidence")
        scout = self.scout(claim)

        notify("Agent 2 · auditing snippets against the claim")
        audit = self.audit(claim, scout)

        notify("Agent 3 · synthesising the final verdict")
        adjudication = self.adjudicate(claim, scout, audit)

        return MultiAgentResult(
            claim=claim,
            scout=scout,
            audit=audit,
            adjudication=adjudication,
            timings=list(self.timings),
            warnings=list(self.warnings),
        )


def run_multi_agent(
    claim: str,
    settings: Settings,
    on_progress: Optional[ProgressCallback] = None,
) -> MultiAgentResult:
    """Convenience wrapper: run the full three-agent pipeline in one call."""
    return FactCheckPipeline(settings).run(claim, on_progress=on_progress)


__all__ = [
    "FactCheckPipeline",
    "run_multi_agent",
    "run_baseline",
    "AgentError",
    "ConfigurationError",
]
