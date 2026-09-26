"""Tests for the pluggable model backend (Anthropic hosted vs. Ollama local).

No server and no API key are needed: clients are constructed but never invoked.
"""

from __future__ import annotations

import pytest

from fact_checker.config import (
    DEFAULT_MODELS,
    DEFAULT_NUM_CTX,
    PROVIDER_MODELS,
    Settings,
    default_provider,
    ollama_status,
)
from fact_checker.llm import ConfigurationError, build_llm


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Isolate every test from the developer's real environment."""
    for name in (
        "ANTHROPIC_API_KEY",
        "FACTCHECK_PROVIDER",
        "FACTCHECK_MODEL",
        "FACTCHECK_NUM_CTX",
        "OLLAMA_HOST",
    ):
        monkeypatch.delenv(name, raising=False)


class TestProviderSelection:
    def test_defaults_to_ollama_without_a_key(self) -> None:
        assert default_provider() == "ollama"

    def test_defaults_to_anthropic_with_a_key(self, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        assert default_provider() == "anthropic"

    def test_env_var_overrides_key_detection(self, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        monkeypatch.setenv("FACTCHECK_PROVIDER", "ollama")
        assert default_provider() == "ollama"

    def test_unknown_provider_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            Settings(provider="gpt4all")


class TestSettingsDefaults:
    def test_model_defaults_per_provider(self) -> None:
        assert Settings(provider="ollama").model == DEFAULT_MODELS["ollama"]
        assert Settings(provider="anthropic", api_key="k").model == DEFAULT_MODELS["anthropic"]

    def test_explicit_model_wins(self) -> None:
        assert Settings(provider="ollama", model="llama3.1:8b").model == "llama3.1:8b"

    def test_local_backend_needs_no_key(self) -> None:
        settings = Settings(provider="ollama")
        assert settings.is_local
        assert not settings.requires_api_key
        assert settings.is_configured

    def test_hosted_backend_is_unconfigured_without_a_key(self) -> None:
        settings = Settings(provider="anthropic")
        assert settings.requires_api_key
        assert not settings.is_configured

    def test_local_runs_get_a_longer_timeout(self) -> None:
        """A 7B model on consumer hardware is far slower than a hosted call."""
        assert Settings(provider="ollama").request_timeout >= 300.0
        assert Settings(provider="anthropic", api_key="k").request_timeout == 120.0

    def test_num_ctx_has_a_floor(self) -> None:
        """Ollama's 2048 default silently truncates the auditor prompt."""
        assert Settings(provider="ollama").num_ctx == DEFAULT_NUM_CTX
        assert Settings(provider="ollama", num_ctx=128).num_ctx == 2048

    def test_num_ctx_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("FACTCHECK_NUM_CTX", "16384")
        assert Settings(provider="ollama").num_ctx == 16384

    def test_ollama_host_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
        assert Settings(provider="ollama").ollama_host == "http://gpu-box:11434"


class TestClientConstruction:
    def test_builds_an_ollama_client_without_a_key(self) -> None:
        llm = build_llm(Settings(provider="ollama", model="qwen2.5:7b"), max_tokens=512)
        assert llm.__class__.__name__ == "ChatOllama"
        assert llm.model == "qwen2.5:7b"
        assert llm.num_ctx == DEFAULT_NUM_CTX
        assert llm.num_predict == 512
        # Local models accept sampling params; pin them for reproducibility.
        assert llm.temperature == 0.0

    def test_builds_an_anthropic_client_with_a_key(self) -> None:
        llm = build_llm(
            Settings(provider="anthropic", api_key="sk-ant-x", model="claude-opus-5"),
            max_tokens=512,
        )
        assert llm.__class__.__name__ == "ChatAnthropic"
        # Opus 5 rejects sampling parameters with a 400, so none are sent.
        assert llm.temperature is None

    def test_anthropic_without_a_key_points_at_the_local_option(self) -> None:
        with pytest.raises(ConfigurationError, match="ollama"):
            build_llm(Settings(provider="anthropic"))


class TestStructuredBinding:
    def test_ollama_binds_with_native_json_schema(self) -> None:
        """json_schema is Ollama's constrained decoding -- far more reliable
        than emulated tool calling for a small model."""
        from fact_checker.llm import _bind_schema
        from fact_checker.schemas import Adjudication

        llm = build_llm(Settings(provider="ollama"))
        assert _bind_schema(llm, Adjudication) is not None

    def test_anthropic_binds_with_tool_calling(self) -> None:
        from fact_checker.llm import _bind_schema
        from fact_checker.schemas import Adjudication

        llm = build_llm(Settings(provider="anthropic", api_key="sk-ant-x"))
        assert _bind_schema(llm, Adjudication) is not None


class TestOllamaStatus:
    def test_unreachable_host_reports_cleanly(self) -> None:
        """A dead server is a reportable state, never an exception."""
        reachable, models = ollama_status("http://localhost:1", timeout=1.0)
        assert reachable is False
        assert models == []

    def test_suggested_models_are_non_empty(self) -> None:
        assert PROVIDER_MODELS["ollama"]
        assert DEFAULT_MODELS["ollama"] in PROVIDER_MODELS["ollama"]


class TestRetryPolicy:
    def test_local_backend_retries_more_than_hosted(self, monkeypatch) -> None:
        """Small models miss schemas more often, so they get more attempts."""
        from fact_checker import llm as llm_mod
        from fact_checker.schemas import SearchQueries

        calls = {"n": 0}

        class FlakyRunnable:
            def invoke(self, _messages):
                calls["n"] += 1
                raise RuntimeError("bad json")

        monkeypatch.setattr(llm_mod, "_bind_schema", lambda l, s: FlakyRunnable())

        local = build_llm(Settings(provider="ollama"))
        with pytest.raises(llm_mod.AgentError, match="attempt"):
            llm_mod.structured_call(local, SearchQueries, "s", "u", stage="test")
        assert calls["n"] == llm_mod._LOCAL_ATTEMPTS

        calls["n"] = 0
        hosted = build_llm(Settings(provider="anthropic", api_key="k"))
        with pytest.raises(llm_mod.AgentError):
            llm_mod.structured_call(hosted, SearchQueries, "s", "u", stage="test")
        assert calls["n"] == llm_mod._HOSTED_ATTEMPTS

    def test_local_failure_message_suggests_a_fix(self, monkeypatch) -> None:
        from fact_checker import llm as llm_mod
        from fact_checker.schemas import SearchQueries

        class Broken:
            def invoke(self, _messages):
                raise RuntimeError("nope")

        monkeypatch.setattr(llm_mod, "_bind_schema", lambda l, s: Broken())
        with pytest.raises(llm_mod.AgentError, match="qwen2.5:7b"):
            llm_mod.structured_call(
                build_llm(Settings(provider="ollama")),
                SearchQueries,
                "s",
                "u",
                stage="test",
            )

    def test_dict_response_is_validated(self, monkeypatch) -> None:
        from fact_checker import llm as llm_mod
        from fact_checker.schemas import SearchQueries

        class DictReturner:
            def invoke(self, _messages):
                return {"queries": ["a", "b"]}

        monkeypatch.setattr(llm_mod, "_bind_schema", lambda l, s: DictReturner())
        result = llm_mod.structured_call(
            build_llm(Settings(provider="ollama")),
            SearchQueries,
            "s",
            "u",
            stage="test",
        )
        assert result.queries == ["a", "b"]


class TestConnectionFailures:
    """A dead server must not be reported as a schema problem."""

    def _dead_runnable(self):
        class Dead:
            def invoke(self, _messages):
                raise OSError(
                    "[WinError 10061] No connection could be made because the "
                    "target machine actively refused it"
                )

        return Dead()

    def test_connection_error_raises_configuration_not_agent_error(self, monkeypatch):
        from fact_checker import llm as llm_mod
        from fact_checker.schemas import SearchQueries

        monkeypatch.setattr(llm_mod, "_bind_schema", lambda l, s: self._dead_runnable())
        with pytest.raises(llm_mod.ConfigurationError) as excinfo:
            llm_mod.structured_call(
                build_llm(Settings(provider="ollama")),
                SearchQueries, "s", "u", stage="Agent 1",
            )

        message = str(excinfo.value)
        assert "No Ollama server" in message
        assert "ollama.com" in message
        assert "ollama pull" in message
        # The schema hint must NOT appear - it sends the user after the wrong fix.
        assert "struggle with this schema" not in message
        assert "NUM_CTX" not in message

    def test_connection_error_is_not_retried(self, monkeypatch):
        """Retrying a refused connection just multiplies the wait."""
        from fact_checker import llm as llm_mod
        from fact_checker.schemas import SearchQueries

        calls = {"n": 0}

        class Dead:
            def invoke(self, _messages):
                calls["n"] += 1
                raise ConnectionRefusedError("[WinError 10061] actively refused it")

        monkeypatch.setattr(llm_mod, "_bind_schema", lambda l, s: Dead())
        with pytest.raises(llm_mod.ConfigurationError):
            llm_mod.structured_call(
                build_llm(Settings(provider="ollama")),
                SearchQueries, "s", "u", stage="Agent 1",
            )
        assert calls["n"] == 1, "connection failures must fail fast, not retry"

    def test_hosted_connection_error_gets_its_own_message(self, monkeypatch):
        from fact_checker import llm as llm_mod
        from fact_checker.schemas import SearchQueries

        monkeypatch.setattr(llm_mod, "_bind_schema", lambda l, s: self._dead_runnable())
        with pytest.raises(llm_mod.ConfigurationError, match="Anthropic API"):
            llm_mod.structured_call(
                build_llm(Settings(provider="anthropic", api_key="k")),
                SearchQueries, "s", "u", stage="Agent 1",
            )

    @pytest.mark.parametrize(
        "message",
        [
            "[WinError 10061] No connection could be made",
            "Connection refused",
            "Max retries exceeded with url",
            "httpx.ConnectError: All connection attempts failed",
            "cannot connect to host",
        ],
    )
    def test_recognises_connection_shapes(self, message):
        from fact_checker.llm import _is_connection_failure

        assert _is_connection_failure(RuntimeError(message))

    def test_schema_errors_are_still_treated_as_schema_errors(self, monkeypatch):
        """A genuine validation failure must keep retrying and keep its hint."""
        from fact_checker import llm as llm_mod
        from fact_checker.schemas import SearchQueries

        calls = {"n": 0}

        class BadJson:
            def invoke(self, _messages):
                calls["n"] += 1
                raise ValueError("Expected a JSON object, got prose")

        monkeypatch.setattr(llm_mod, "_bind_schema", lambda l, s: BadJson())
        with pytest.raises(llm_mod.AgentError, match="qwen2.5:7b"):
            llm_mod.structured_call(
                build_llm(Settings(provider="ollama")),
                SearchQueries, "s", "u", stage="Agent 1",
            )
        assert calls["n"] == llm_mod._LOCAL_ATTEMPTS
