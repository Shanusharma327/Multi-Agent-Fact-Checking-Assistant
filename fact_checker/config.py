"""Runtime settings, resolved from explicit args -> environment -> defaults.

Supports two model backends:

* ``anthropic`` -- the hosted Claude API (needs an API key, costs money).
* ``ollama``    -- a model running locally via Ollama (free, no key, no network
  egress for inference; web search still hits the internet).

The pipeline code is identical either way; only :mod:`fact_checker.llm` knows
which client to build.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:  # python-dotenv is convenient locally but must not be a hard dependency.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - absence of .env is entirely normal.
    pass

PROVIDERS: Tuple[str, ...] = ("anthropic", "ollama")

# Anthropic's current frontier model. Use the exact ID -- never append a date
# suffix. Swap via FACTCHECK_MODEL or the sidebar.
DEFAULT_MODEL = "claude-opus-5"

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Suggestions only -- any locally pulled tag works. Sizes are the q4 defaults.
PROVIDER_MODELS: Dict[str, Tuple[str, ...]] = {
    "anthropic": ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"),
    "ollama": (
        "qwen2.5:7b",      # ~4.7 GB - strongest JSON adherence in this size class
        "llama3.1:8b",     # ~4.9 GB
        "mistral:7b",      # ~4.4 GB
        "qwen2.5:3b",      # ~1.9 GB - for <6 GB VRAM or faster runs
        "gemma2:2b",       # ~1.6 GB - fastest, weakest reasoning
    ),
}

DEFAULT_MODELS: Dict[str, str] = {
    "anthropic": DEFAULT_MODEL,
    "ollama": DEFAULT_OLLAMA_MODEL,
}

# Backwards-compatible alias used by older imports.
SUPPORTED_MODELS: Tuple[str, ...] = PROVIDER_MODELS["anthropic"]

# Ollama defaults to a 2048-token context, which silently truncates the
# auditor prompt once several snippets are in play. Always set it explicitly.
DEFAULT_NUM_CTX = 8192


def _env_int(name: str, fallback: int) -> int:
    """Read an int from the environment, falling back on anything unparseable."""
    try:
        return int(os.getenv(name, "").strip() or fallback)
    except ValueError:
        return fallback


def default_provider() -> str:
    """Pick a backend: explicit env var, else API key presence, else local."""
    configured = (os.getenv("FACTCHECK_PROVIDER") or "").strip().lower()
    if configured in PROVIDERS:
        return configured
    return "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "ollama"


def ollama_status(host: str = DEFAULT_OLLAMA_HOST, timeout: float = 3.0) -> Tuple[bool, List[str]]:
    """Check whether an Ollama server is reachable and list its models.

    Args:
        host: Base URL of the Ollama server.
        timeout: Seconds to wait before giving up.

    Returns:
        ``(reachable, model_names)``. Never raises -- a dead server is a
        normal, reportable state rather than an error.
    """
    url = f"{host.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return False, []

    models = [
        str(entry.get("name", ""))
        for entry in payload.get("models", [])
        if entry.get("name")
    ]
    return True, sorted(models)


@dataclass(slots=True)
class Settings:
    """Everything the pipeline needs to run, in one injectable object."""

    api_key: Optional[str] = None
    provider: str = field(default_factory=default_provider)
    model: str = ""
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
    )
    num_ctx: int = field(default_factory=lambda: _env_int("FACTCHECK_NUM_CTX", DEFAULT_NUM_CTX))
    results_per_query: int = field(
        default_factory=lambda: _env_int("FACTCHECK_RESULTS_PER_QUERY", 4)
    )
    max_evidence: int = field(default_factory=lambda: _env_int("FACTCHECK_MAX_EVIDENCE", 8))
    snippet_chars: int = 700
    request_timeout: float = 120.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        self.provider = (self.provider or "").strip().lower()
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unknown provider {self.provider!r}. Choose from: {', '.join(PROVIDERS)}"
            )

        # An explicitly passed key wins; otherwise fall back to the environment.
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY") or None

        if not self.model:
            self.model = os.getenv("FACTCHECK_MODEL") or DEFAULT_MODELS[self.provider]

        self.results_per_query = max(1, min(10, self.results_per_query))
        self.max_evidence = max(1, min(20, self.max_evidence))
        self.num_ctx = max(2048, self.num_ctx)

        # Local models are slower per token; give them a longer leash.
        if self.provider == "ollama" and self.request_timeout < 300.0:
            self.request_timeout = 300.0

    @property
    def is_local(self) -> bool:
        """True when inference runs on this machine rather than a hosted API."""
        return self.provider == "ollama"

    @property
    def requires_api_key(self) -> bool:
        """Whether the selected backend needs an Anthropic key at all."""
        return self.provider == "anthropic"

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    @property
    def is_configured(self) -> bool:
        """Whether this backend has what it needs to attempt a call."""
        return self.has_api_key if self.requires_api_key else True
