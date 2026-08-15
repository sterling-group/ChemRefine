"""The embedded agent: same tools, human-gated mutations, any model — proven offline.

``ALLOW_MODEL_REQUESTS = False`` at import, so nothing in here (or anywhere the suite
wanders) can reach a real endpoint: the harness runs against PydanticAI's ``TestModel``
(schema-driven synthetic calls) and ``FunctionModel`` (scripted turns, for exact
code-path control). What must hold: provider resolution is flags → environment →
preset with honest errors; every shared tool is registered under its own name with the
mutating ones behind the confirmation callback; a declined mutation returns a
structured refusal the model continues from, and the mutation did not happen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic_ai import models
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from chemrefine import agent_tools
from chemrefine.agent import harness
from chemrefine.agent.providers import ProviderConfig
from chemrefine.errors import ConfigError

models.ALLOW_MODEL_REQUESTS = False  # the whole suite stays offline, permanently


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def test_provider_resolution_is_flags_env_preset(monkeypatch: pytest.MonkeyPatch):
    for var in ("CHEMREFINE_LLM_MODEL", "CHEMREFINE_LLM_BASE_URL", "CHEMREFINE_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    ollama = ProviderConfig.resolve("ollama", model="qwen3")
    assert (ollama.base_url, ollama.api_key) == ("http://localhost:11434/v1", "ollama")

    monkeypatch.setenv("CHEMREFINE_LLM_MODEL", "env-model")
    monkeypatch.setenv("CHEMREFINE_LLM_BASE_URL", "http://env:1/v1")
    monkeypatch.setenv("CHEMREFINE_LLM_API_KEY", "env-key")
    resolved = ProviderConfig.resolve("ollama")
    assert (resolved.model, resolved.base_url, resolved.api_key) == (
        "env-model",
        "http://env:1/v1",
        "env-key",
    )
    assert ProviderConfig.resolve("ollama", model="flag-model").model == "flag-model"


def test_provider_refusals_name_the_fix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CHEMREFINE_LLM_MODEL", raising=False)
    with pytest.raises(ConfigError, match="unknown provider"):
        ProviderConfig.resolve("skynet")
    with pytest.raises(ConfigError, match="CHEMREFINE_LLM_MODEL"):
        ProviderConfig.resolve("custom")


def test_build_model_pins_an_endpoint_or_passes_the_string_through(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("CHEMREFINE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("CHEMREFINE_LLM_API_KEY", raising=False)
    from pydantic_ai.models.openai import OpenAIChatModel

    pinned = ProviderConfig.resolve("ollama", model="qwen3").build_model()
    assert isinstance(pinned, OpenAIChatModel)

    inferred = ProviderConfig.resolve("custom", model="openai:gpt-5-mini").build_model()
    assert inferred == "openai:gpt-5-mini"


# ---------------------------------------------------------------------------
# Harness: registration + instructions
# ---------------------------------------------------------------------------


def _allow(_tool: str, _args: str) -> bool:
    return True


def test_every_shared_tool_is_registered_verbatim():
    """TestModel sees the toolset the agent carries — it must be TOOLS, name for name."""
    seen: dict[str, Any] = {}

    def spy(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["tools"] = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart("ok")])

    agent = harness.build_agent(FunctionModel(spy), confirm=_allow)
    agent.run_sync("hello")
    assert seen["tools"] == sorted(t.__name__ for t in agent_tools.TOOLS)


def test_instructions_carry_the_guide_and_the_session_config():
    text = harness.instructions("/tmp/proj/input.yaml")
    assert "operating guide" in text
    assert "/tmp/proj/input.yaml" in text
    assert "confirmation" in text
    assert "input.yaml" not in harness.instructions(None).replace("input.yaml`", "")


def test_a_real_tool_answers_through_the_model_loop():
    """End to end offline: the model calls get_schema and the answer is the schema."""
    agent = harness.build_agent(TestModel(call_tools=["get_schema"]), confirm=_allow)
    result = agent.run_sync("what does a config look like?")
    assert "chemrefine_version" in result.output


# ---------------------------------------------------------------------------
# The confirmation gate
# ---------------------------------------------------------------------------


def _scripted_scaffold(config_path: Path) -> FunctionModel:
    """A model that asks for one scaffold, then reports what came back."""

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart("scaffold_templates", {"config_path": str(config_path)})]
            )
        return ModelResponse(parts=[TextPart("finished")])

    return FunctionModel(script)


def _orca_config(tmp_path: Path) -> Path:
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump({"steps": [{"step": 1, "engine": "orca"}]}), "utf-8")
    return path


def test_an_allowed_mutation_runs_and_names_itself_to_the_human(tmp_path: Path):
    config = _orca_config(tmp_path)
    asked: list[tuple[str, str]] = []

    def confirm(tool: str, rendered: str) -> bool:
        asked.append((tool, rendered))
        return True

    agent = harness.build_agent(_scripted_scaffold(config), confirm=confirm)
    result = agent.run_sync("scaffold my templates")
    assert result.output == "finished"
    [(tool, rendered)] = asked
    assert tool == "scaffold_templates"
    assert str(config) in rendered
    assert (tmp_path / "templates" / "step1.inp").is_file()


def test_a_declined_mutation_does_not_happen_and_the_loop_survives(tmp_path: Path):
    config = _orca_config(tmp_path)
    agent = harness.build_agent(
        _scripted_scaffold(config), confirm=lambda _tool, _args: False
    )
    result = agent.run_sync("scaffold my templates")
    assert result.output == "finished"  # the refusal was an answer, not a crash
    assert not (tmp_path / "templates").exists()
    denials = [
        part
        for message in result.all_messages()
        for part in message.parts
        if getattr(part, "part_kind", "") == "tool-return" and "denied" in str(part.content)
    ]
    assert denials, "the model must be told the user said no"


def test_read_only_tools_are_never_gated():
    asked: list[str] = []

    def confirm(tool: str, _args: str) -> bool:
        asked.append(tool)
        return True

    agent = harness.build_agent(TestModel(call_tools=["get_schema"]), confirm=confirm)
    agent.run_sync("schema please")
    assert asked == []


# ---------------------------------------------------------------------------
# REPL + CLI
# ---------------------------------------------------------------------------


def test_chat_confirm_names_the_tool_and_defaults_to_no(monkeypatch: pytest.MonkeyPatch):
    import typer

    from chemrefine.agent import chat

    asked: list[tuple[str, bool]] = []

    def fake_confirm(message: str, default: bool) -> bool:
        asked.append((message, default))
        return False

    monkeypatch.setattr(typer, "confirm", fake_confirm)
    assert chat._confirm("start_run", '{"config_path": "x"}') is False
    [(message, default)] = asked
    assert "start_run" in message
    assert default is False  # the safe answer is the default answer


def test_chat_repl_threads_history_until_exit(monkeypatch: pytest.MonkeyPatch):
    import typer

    from chemrefine.agent import chat

    monkeypatch.setenv("CHEMREFINE_LLM_MODEL", "test-model")
    monkeypatch.setattr(
        ProviderConfig, "build_model", lambda self: TestModel(call_tools=[])
    )
    prompts = iter(["hello there", "exit"])
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: next(prompts))
    echoed: list[str] = []
    monkeypatch.setattr(typer, "echo", echoed.append)

    chat.main(provider="custom")
    assert any("ChemRefine agent" in line for line in echoed)
    assert len(echoed) >= 2  # banner + at least one model reply


def test_chat_repl_ends_on_eof(monkeypatch: pytest.MonkeyPatch):
    import typer

    from chemrefine.agent import chat

    monkeypatch.setenv("CHEMREFINE_LLM_MODEL", "test-model")
    monkeypatch.setattr(
        ProviderConfig, "build_model", lambda self: TestModel(call_tools=[])
    )

    def eof(*args: Any, **kwargs: Any) -> str:
        raise EOFError

    monkeypatch.setattr(typer, "prompt", eof)
    chat.main(provider="custom")  # returns instead of crashing


def test_cli_agent_hands_off_and_maps_config_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from typer.testing import CliRunner

    from chemrefine.agent import chat
    from chemrefine.cli import app

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(chat, "main", lambda **kwargs: calls.append(kwargs))
    config = _orca_config(tmp_path)
    result = CliRunner().invoke(
        app, ["agent", str(config), "--provider", "ollama", "--model", "qwen3"]
    )
    assert result.exit_code == 0
    assert calls == [
        {
            "provider": "ollama",
            "model": "qwen3",
            "base_url": None,
            "config_path": str(config),
        }
    ]

    def refuse(**kwargs: Any) -> None:
        raise ConfigError("no model configured")

    monkeypatch.setattr(chat, "main", refuse)
    failed = CliRunner().invoke(app, ["agent"])
    assert failed.exit_code == 2


def test_cli_agent_names_the_missing_extra(monkeypatch: pytest.MonkeyPatch):
    import builtins

    from typer.testing import CliRunner

    from chemrefine.cli import app

    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("chemrefine.agent") and not name.startswith("chemrefine.agent_"):
            raise ImportError("No module named 'pydantic_ai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    result = CliRunner().invoke(app, ["agent"])
    assert result.exit_code == 1
