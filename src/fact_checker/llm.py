"""Chat model factory and the structured-output call wrapper.

Every LLM call in this project goes through :func:`structured_call`, which
centralises what is easy to get wrong across two very different backends:
schema binding, context sizing, retry policy, and error translation.
"""

from __future__ import annotations

import logging
from typing import Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from .config import Settings

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Local models fail schema adherence more often than hosted ones, so give them
# a second chance before declaring the stage a failure.
_LOCAL_ATTEMPTS = 3
_HOSTED_ATTEMPTS = 1

# A dead server is a configuration problem, not a flaky response: retrying it
# just multiplies the wait, and blaming the schema sends the user chasing the
# wrong fix. WinError 10061 is Windows' connection-refused.
_CONNECTION_MARKERS = (
    "10061",
    "connection refused",
    "actively refused",
    "failed to establish",
    "max retries exceeded",
    "connecterror",
    "cannot connect",
    "connection error",
    "no connection could be made",
    "remote end closed",
)


def _is_connection_failure(exc: BaseException) -> bool:
    """Whether ``exc`` means 'nothing answered', rather than 'bad answer'."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _CONNECTION_MARKERS)


class ConfigurationError(RuntimeError):
    """Raised when the app cannot construct a usable model client."""


class AgentError(RuntimeError):
    """Raised when an agent's LLM call fails or returns an unusable object."""


def _build_anthropic(settings: Settings, max_tokens: int) -> BaseChatModel:
    """Construct a hosted Claude client."""
    if not settings.has_api_key:
        raise ConfigurationError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY in your "
            "environment, paste a key into the sidebar, or switch the provider "
            "to 'ollama' to run a local model instead."
        )

    from langchain_anthropic import ChatAnthropic

    # No temperature/top_p/top_k: the Opus 5 generation rejects sampling
    # parameters with a 400. Determinism comes from strict schemas and
    # closed-world prompts instead.
    return ChatAnthropic(
        model=settings.model,
        api_key=settings.api_key,
        max_tokens=max_tokens,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )


def _build_ollama(settings: Settings, max_tokens: int) -> BaseChatModel:
    """Construct a local Ollama client.

    Two settings matter disproportionately here:

    * ``num_ctx`` -- Ollama defaults to a 2048-token context and silently
      truncates anything longer. The auditor prompt carrying several snippets
      blows past that, so it is set explicitly.
    * ``temperature=0`` -- unlike the Opus family, Ollama accepts sampling
      parameters, and pinning it keeps local runs reproducible.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:  # pragma: no cover - depends on the install.
        raise ConfigurationError(
            "langchain-ollama is not installed. Run: pip install langchain-ollama"
        ) from exc

    return ChatOllama(
        model=settings.model,
        base_url=settings.ollama_host,
        temperature=0.0,
        num_predict=max_tokens,
        num_ctx=settings.num_ctx,
    )


def build_llm(settings: Settings, max_tokens: int = 8000) -> BaseChatModel:
    """Construct a chat client for one agent role.

    Args:
        settings: Resolved runtime settings, including the provider choice.
        max_tokens: Output ceiling for this role's responses.

    Returns:
        A configured chat model for the selected backend.

    Raises:
        ConfigurationError: If the backend cannot be constructed.
    """
    if settings.provider == "ollama":
        return _build_ollama(settings, max_tokens)
    return _build_anthropic(settings, max_tokens)


def _bind_schema(llm: BaseChatModel, schema: Type[SchemaT]):
    """Bind a response schema using the best method for this backend.

    For Ollama, ``json_schema`` uses the server's native constrained decoding,
    which a 7B model follows far more reliably than emulated tool calling.
    """
    if llm.__class__.__name__ == "ChatOllama":
        return llm.with_structured_output(schema, method="json_schema")
    return llm.with_structured_output(schema)


def _connection_help(llm: BaseChatModel, is_local: bool, exc: BaseException) -> str:
    """Build an actionable message for a server that did not answer."""
    if not is_local:
        return (
            "Could not reach the Anthropic API. Check your network connection "
            f"and that the API key is valid. Underlying error: {exc}"
        )

    host = getattr(llm, "base_url", None) or "http://localhost:11434"
    model = getattr(llm, "model", "qwen2.5:7b")
    return (
        f"No Ollama server is answering at {host}.\n\n"
        "Fix it with one of:\n"
        "  1. Install Ollama from https://ollama.com (if you have not yet), then\n"
        f"     run:  ollama pull {model}\n"
        "  2. If it is installed, start the server:  ollama serve\n"
        "  3. Check it is up:  ollama list\n\n"
        "Or switch the sidebar's Model backend to the Anthropic API and supply "
        "a key."
    )


def structured_call(
    llm: BaseChatModel,
    schema: Type[SchemaT],
    system_prompt: str,
    user_prompt: str,
    *,
    stage: str,
) -> SchemaT:
    """Run one schema-constrained turn and return a validated model instance.

    Args:
        llm: The chat client to call.
        schema: Pydantic model the response must conform to.
        system_prompt: Role instructions for the agent.
        user_prompt: The task payload.
        stage: Human-readable stage name, used in error messages.

    Returns:
        A validated instance of ``schema``.

    Raises:
        AgentError: If every attempt fails or returns an uncoercible response.
    """
    structured = _bind_schema(llm, schema)
    is_local = llm.__class__.__name__ == "ChatOllama"
    attempts = _LOCAL_ATTEMPTS if is_local else _HOSTED_ATTEMPTS
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            result = structured.invoke(
                [("system", system_prompt), ("human", user_prompt)]
            )
        except Exception as exc:  # noqa: BLE001 - provider errors vary widely.
            # Nothing answered: retrying cannot help, and a schema hint would
            # send the user after the wrong problem. Fail fast and say why.
            if _is_connection_failure(exc):
                raise ConfigurationError(_connection_help(llm, is_local, exc)) from exc
            last_error = exc
            logger.warning("%s: attempt %d/%d failed (%s)", stage, attempt, attempts, exc)
            continue

        if isinstance(result, schema):
            return result

        # Some provider/parser combinations hand back a plain dict instead of
        # the model instance; validate it rather than failing on type alone.
        if isinstance(result, dict):
            try:
                return schema.model_validate(result)
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "%s: attempt %d/%d returned an invalid object (%s)",
                    stage,
                    attempt,
                    attempts,
                    exc,
                )
                continue

        last_error = TypeError(f"unexpected response type: {type(result)!r}")
        logger.warning("%s: attempt %d/%d %s", stage, attempt, attempts, last_error)

    hint = ""
    if is_local:
        hint = (
            " A smaller local model may struggle with this schema -- try "
            "'qwen2.5:7b', raise FACTCHECK_NUM_CTX, or lower --max-evidence."
        )
    raise AgentError(f"{stage} failed after {attempts} attempt(s): {last_error}.{hint}")
