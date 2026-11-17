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
from .aggregation import AggregationTrace
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
        self.trace: Optional[AggregationTrace] = None

    def reset(self) -> None:
        """Clear timings, warnings and the aggregation trace before a run."""
        self.timings = []
        self.warnings = []
        self.trace = None

    def scout(self, claim: str, *, exclude_url: str = "") -> ScoutReport:
        """Stage 1: generate queries and retrieve evidence."""
        with _timed("Agent 1 · Evidence Scout", self.timings):
            report, warnings = run_scout(
                claim, self.settings, exclude_url=exclude_url
            )
        self.warnings.extend(warnings)
        return report

    def audit(self, claim: str, scout: ScoutReport) -> AuditReport:
        """Stage 2: label each snippet against the claim."""
        with _timed("Agent 2 · Alignment Auditor", self.timings):
            return run_auditor(claim, scout, self.settings)

    def adjudicate(
        self, claim: str, scout: ScoutReport, audit: AuditReport
    ) -> Adjudication:
        """Stage 3: synthesise the unified verdict.

        When ``settings.deterministic_aggregation`` is set, the label and
        confidence are computed from the audit rather than taken from the
        model; the resulting arithmetic is kept on ``self.trace``.
        """
        with _timed("Agent 3 · Synthesis Adjudicator", self.timings):
            adjudication, trace = run_adjudicator(
                claim,
                scout,
                audit,
                self.settings,
                deterministic=self.settings.deterministic_aggregation,
                beta=self.settings.aggregation_beta,
                gate_citations=self.settings.gate_citations,
            )
        self.trace = trace
        return adjudication

    def baseline(self, claim: str) -> BaselineResult:
        """Control: single ungrounded LLM call."""
        return run_baseline(claim, self.settings)

    def run(
        self,
        claim: str,
        on_progress: Optional[ProgressCallback] = None,
        scout_override: Optional[ScoutReport] = None,
        exclude_url: str = "",
    ) -> MultiAgentResult:
        """Execute all three agents in sequence.

        Args:
            claim: The claim to verify.
            on_progress: Optional callback invoked with a status string as each
                stage starts.
            scout_override: A pre-built evidence set to use instead of running
                Agent 1. Used by the oracle-retrieval evaluation condition to
                score verification independently of retrieval; it is never set
                in normal operation, where no gold evidence exists.
            exclude_url: Originating fact-check article to keep out of the
                retrieved set. Set only by the evaluation harness.

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

        if scout_override is not None:
            notify("Agent 1 · bypassed (oracle retrieval)")
            scout = scout_override
            self.timings.append(StageTiming(name="Agent 1 · oracle", seconds=0.0))
        else:
            notify("Agent 1 · generating queries and retrieving evidence")
            scout = self.scout(claim, exclude_url=exclude_url)

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
            aggregation=self.trace,
        )


def run_multi_agent(
    claim: str,
    settings: Settings,
    on_progress: Optional[ProgressCallback] = None,
    scout_override: Optional[ScoutReport] = None,
    exclude_url: str = "",
) -> MultiAgentResult:
    """Convenience wrapper: run the full three-agent pipeline in one call."""
    return FactCheckPipeline(settings).run(
        claim,
        on_progress=on_progress,
        scout_override=scout_override,
        exclude_url=exclude_url,
    )


__all__ = [
    "FactCheckPipeline",
    "run_multi_agent",
    "run_baseline",
    "AgentError",
    "ConfigurationError",
]
