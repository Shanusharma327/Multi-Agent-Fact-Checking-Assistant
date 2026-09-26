"""Multi-Agent Fact-Checking Assistant.

A three-agent pipeline (Evidence Scout -> Alignment Auditor -> Synthesis
Adjudicator) that grounds claim verification in live web evidence, plus a
monolithic single-LLM baseline for side-by-side comparison.
"""

from .config import DEFAULT_MODEL, SUPPORTED_MODELS, Settings
from .llm import AgentError, ConfigurationError
from .pipeline import FactCheckPipeline, run_multi_agent
from .schemas import (
    Adjudication,
    Alignment,
    AuditReport,
    BaselineResult,
    EvidenceItem,
    MultiAgentResult,
    ScoutReport,
)

__version__ = "1.0.0"

__all__ = [
    "Settings",
    "DEFAULT_MODEL",
    "SUPPORTED_MODELS",
    "FactCheckPipeline",
    "run_multi_agent",
    "AgentError",
    "ConfigurationError",
    "Adjudication",
    "Alignment",
    "AuditReport",
    "BaselineResult",
    "EvidenceItem",
    "MultiAgentResult",
    "ScoutReport",
]
