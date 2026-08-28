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

import contextlib
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
    # The key follows the same three tiers, for the GUI panel that holds one per session.
    assert ProviderConfig.resolve("ollama", api_key="flag-key").api_key == "flag-key"
    assert ProviderConfig.resolve("ollama", api_key="").api_key == "env-key"  # blank falls through


def test_a_supplied_key_reaches_openai_and_hijacks_nothing_else(monkeypatch: pytest.MonkeyPatch):
    """The one branch where a key changes which model object gets built.

    ``_PRESETS["openai"]`` carries no base URL, so ``build_model`` returned the bare name
    and PydanticAI read the *server's* ``OPENAI_API_KEY`` — a key the GUI's user cannot
    set, which made the panel's key box inert for the provider most likely to need it.
    ``OpenAIProvider`` defaults to api.openai.com, so a key alone is enough.

    Two things it must NOT do, which is why the branch is narrow: hijack a
    ``provider:model`` string (that spelling *is* the instruction to let PydanticAI
    resolve the provider, credentials included), and hijack a native provider for anyone
    who merely has ``CHEMREFINE_LLM_API_KEY`` set.
    """
    for var in ("CHEMREFINE_LLM_MODEL", "CHEMREFINE_LLM_BASE_URL", "CHEMREFINE_LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    keyed = ProviderConfig.resolve("openai", model="gpt-5-mini", api_key="sk-panel")
    assert keyed.keyed_openai is True
    assert not isinstance(keyed.build_model(), str)  # pinned to OpenAI with our key

    for spelling in (
        ProviderConfig.resolve("openai", model="openai:gpt-5-mini", api_key="sk-panel"),
        ProviderConfig.resolve("custom", model="anthropic:claude-x", api_key="sk-panel"),
        ProviderConfig.resolve("openai", model="gpt-5-mini"),
    ):
        assert spelling.keyed_openai is False
        assert isinstance(spelling.build_model(), str)  # handed to PydanticAI verbatim

    # probe_url and build_model share one predicate, so the preflight can never validate
    # a key the chat then declines to use.
    for config in (keyed, *[ProviderConfig.resolve("openai", model="gpt-5-mini")]):
        assert (config.probe_url is not None) == (not isinstance(config.build_model(), str))


def test_provider_refusals_name_the_fix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CHEMREFINE_LLM_MODEL", raising=False)
    with pytest.raises(ConfigError, match="unknown provider"):
        ProviderConfig.resolve("skynet")
    with pytest.raises(ConfigError, match="CHEMREFINE_LLM_MODEL"):
        ProviderConfig.resolve("custom")


def test_resolve_refuses_a_non_http_base_url(monkeypatch: pytest.MonkeyPatch):
    """The one resolver every caller shares refuses odd schemes before a key rides them.

    The GUI's chat endpoint feeds a request-supplied base URL into resolve, where it is
    paired with ``CHEMREFINE_LLM_API_KEY`` — the refusal has to live here, not only in
    the ``--check`` preflight.
    """
    monkeypatch.delenv("CHEMREFINE_LLM_BASE_URL", raising=False)
    with pytest.raises(ConfigError, match="not HTTP"):
        ProviderConfig.resolve("custom", model="m", base_url="ftp://somewhere/v1")


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
# --check: verification, never provisioning
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _listing_server(payload: bytes, status: int = 200) -> Any:
    """A one-endpoint OpenAI-compatible ``/models`` stub on a kernel-assigned port.

    ``seen`` collects the request headers so a test can assert on what we sent, not only
    on what we did with the answer.
    """
    import http.server
    import threading

    seen: list[dict[str, str]] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen.append(dict(self.headers))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            """Keep the test output quiet."""

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", seen
    finally:
        server.shutdown()
        server.server_close()  # both halves, or the listening socket trips filterwarnings=error


def _cfg(base_url: str | None, model: str = "m1") -> ProviderConfig:
    return ProviderConfig(model=model, base_url=base_url, api_key="k")


def test_check_reports_a_served_model_usable():
    from chemrefine.agent.providers import check

    with _listing_server(b'{"data": [{"id": "m1"}, {"id": "m2"}]}') as (base, _seen):
        report = check(_cfg(base))
    assert report.ok is True
    assert "m1" in report.findings[0]


def test_the_turn_timeout_defaults_and_can_be_overridden(monkeypatch: pytest.MonkeyPatch):
    """Sized for the slowest supported endpoint, not the fastest.

    ``ollama`` and ``vllm`` are presets here and run on whatever hardware the user has; a
    small model on CPU can spend minutes on prompt evaluation alone, and neither harness
    streams, so nothing arrives until the turn ends. The OpenAI client's own ten-minute
    default sits below that floor and reported the overrun as "model endpoint failed",
    which names the endpoint for what is really a clock.
    """
    from chemrefine.agent.providers import (
        CHAT_TIMEOUT_ENV,
        DEFAULT_CHAT_TIMEOUT_SECONDS,
        chat_timeout_seconds,
    )

    monkeypatch.delenv(CHAT_TIMEOUT_ENV, raising=False)
    assert chat_timeout_seconds() == DEFAULT_CHAT_TIMEOUT_SECONDS
    assert DEFAULT_CHAT_TIMEOUT_SECONDS > 600  # the client default this exists to clear

    monkeypatch.setenv(CHAT_TIMEOUT_ENV, "90")
    assert chat_timeout_seconds() == 90.0


@pytest.mark.parametrize("bad", ["soon", "", "0", "-30", "nan", "inf", "-inf"])
def test_an_unusable_turn_timeout_is_refused_not_ignored(bad: str, monkeypatch: pytest.MonkeyPatch):
    """A mistyped timeout that silently reverted to the default is found by waiting an hour."""
    from chemrefine.agent.providers import CHAT_TIMEOUT_ENV, chat_timeout_seconds

    monkeypatch.setenv(CHAT_TIMEOUT_ENV, bad)
    with pytest.raises(ConfigError, match=CHAT_TIMEOUT_ENV):
        chat_timeout_seconds()


def test_both_harnesses_carry_the_same_turn_timeout(monkeypatch: pytest.MonkeyPatch):
    """One home for the value, so the REPL and the web panel cannot drift apart."""
    from chemrefine.agent.providers import CHAT_TIMEOUT_ENV

    def timeout_of(agent: Any) -> float:
        # `model_settings` is typed as settings-or-callable-or-None; the harness always
        # sets the mapping form, and narrowing says so rather than indexing past the union.
        settings = agent.model_settings
        assert isinstance(settings, dict)
        return float(settings["timeout"])

    monkeypatch.setenv(CHAT_TIMEOUT_ENV, "1234")
    assert timeout_of(harness.build_agent(TestModel(), confirm=_allow)) == 1234.0
    assert timeout_of(harness.build_web_agent(TestModel())) == 1234.0


def test_check_identifies_chemrefine_to_the_endpoint():
    """The probe must carry a real ``User-Agent``, not urllib's default.

    Not politeness: urllib announces ``Python-urllib/3.x`` unless told otherwise, and
    Groq's edge answers that token with a flat 403 — so this preflight reported a valid
    API key as "authentication rejected", and the GUI's chat panel (whose Send button is
    gated on the preflight) could not be used with Groq at all, while the chat itself
    worked fine through the OpenAI SDK and its own product token.

    ``get`` rather than ``["User-Agent"]`` because urllib title-cases what it sends.
    """
    from chemrefine import USER_AGENT, __version__
    from chemrefine.agent.providers import check

    with _listing_server(b'{"data": [{"id": "m1"}]}') as (base, seen):
        check(_cfg(base))
    sent = {k.lower(): v for k, v in seen[0].items()}
    assert sent["user-agent"] == USER_AGENT
    assert "ChemRefine" in sent["user-agent"]
    assert __version__ in sent["user-agent"]
    assert "urllib" not in sent["user-agent"]


def test_check_names_the_missing_model_and_what_is_served():
    from chemrefine.agent.providers import check

    with _listing_server(b'{"data": [{"id": "other"}]}') as (base, _seen):
        report = check(_cfg(base, model="qwen3:4b"))
    assert report.ok is False
    assert "qwen3:4b" in report.findings[0]
    assert "other" in report.findings[0]


def test_check_maps_auth_rejection_to_the_key_fix():
    from chemrefine.agent.providers import check

    with _listing_server(b"{}", status=401) as (base, _seen):
        report = check(_cfg(base))
    assert report.ok is False
    assert any("CHEMREFINE_LLM_API_KEY" in line for line in report.findings)


def test_check_reports_other_http_statuses_plainly():
    from chemrefine.agent.providers import check

    with _listing_server(b"{}", status=500) as (base, _seen):
        report = check(_cfg(base))
    assert report.ok is False
    assert "HTTP 500" in report.findings[0]


def test_check_reports_an_unreachable_endpoint():
    from chemrefine.agent.providers import check

    with _listing_server(b"{}") as (base, _seen):
        pass  # the context closed the server — the port now refuses connections
    report = check(_cfg(base), timeout=2.0)
    assert report.ok is False
    assert "unreachable" in report.findings[0]


def test_check_reports_an_endpoint_that_breaks_the_protocol(monkeypatch: pytest.MonkeyPatch):
    """A truncated reply (IncompleteRead) is a finding, not a traceback.

    `http.client.HTTPException` subclasses neither OSError nor ValueError, so an endpoint
    that answered and then died mid-body escaped the function whose whole contract is
    that a bad endpoint becomes a finding.
    """
    import urllib.request
    from http.client import IncompleteRead

    from chemrefine.agent.providers import check

    def die_mid_body(request, timeout):
        raise IncompleteRead(b"{")

    monkeypatch.setattr(urllib.request, "urlopen", die_mid_body)
    report = check(_cfg("http://127.0.0.1:1/v1"))
    assert report.ok is False
    assert "unreachable" in report.findings[0]


def test_check_passes_native_strings_through_unprobed():
    from chemrefine.agent.providers import check

    report = check(_cfg(None, model="openai:gpt-5-mini"))
    assert report.ok is True
    assert "not probed" in report.findings[0]


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("an HTML page", b"<html><body>Sign in</body></html>"),
        ("a bare list", b"[]"),
        ("a JSON scalar", b'"ok"'),
        ("data of non-objects", b'{"data": ["a", "b"]}'),
    ],
)
def test_check_reports_a_reply_that_is_not_a_model_listing(label: str, body: bytes):
    """A 200 that is not a listing is a finding, because that is what --check is for.

    Every one of these raised out of `check()` — the first as a JSONDecodeError, the rest
    as AttributeError from `.get` on the wrong type — and `cli.py` catches only
    ChemRefineError, so the preflight printed a traceback. They are not exotic: a base URL
    pointing at a web app or a proxy's login page answers 200 with HTML, and `custom` is
    the default provider.
    """
    from chemrefine.agent.providers import check

    with _listing_server(body) as (base, _seen):
        report = check(_cfg(base))
    assert report.ok is False
    assert "not an OpenAI-style model listing" in report.findings[0], label


def test_check_refuses_a_non_http_url():
    from chemrefine.agent.providers import check

    report = check(_cfg("ftp://somewhere/v1"))
    assert report.ok is False
    assert "not HTTP" in report.findings[0]


def test_check_fixes_name_ollamas_own_commands():
    """The conventional port earns the verbatim one-command fixes; suggestions only."""
    from chemrefine.agent.providers import _fixes

    assert _fixes("http://127.0.0.1:11434/v1/models", "unreachable") == (
        "start it with: ollama serve",
    )
    assert "ollama pull" in _fixes("http://127.0.0.1:11434/v1/models", "missing-model")[0]
    assert "endpoint URL" in _fixes("https://api.groq.com/openai/v1/models", "unreachable")[0]


def test_cli_check_exits_by_verdict(monkeypatch: pytest.MonkeyPatch):
    from typer.testing import CliRunner

    from chemrefine.agent import providers
    from chemrefine.cli import app

    monkeypatch.setenv("CHEMREFINE_LLM_MODEL", "m1")
    monkeypatch.setattr(
        providers, "check", lambda cfg: providers.CheckReport(ok=True, findings=("fine",))
    )
    good = CliRunner().invoke(app, ["agent", "--check"])
    assert good.exit_code == 0
    assert "fine" in good.stdout

    monkeypatch.setattr(
        providers, "check", lambda cfg: providers.CheckReport(ok=False, findings=("broken",))
    )
    bad = CliRunner().invoke(app, ["agent", "--check"])
    assert bad.exit_code == 1
    assert "broken" in bad.stdout


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
    text = harness.instructions("/tmp/proj/input.yaml", gate=harness.TERMINAL_GATE)
    assert "operating guide" in text
    assert "/tmp/proj/input.yaml" in text
    assert "confirmation" in text
    bare = harness.instructions(None, gate=harness.TERMINAL_GATE)
    assert "input.yaml" not in bare.replace("input.yaml`", "")


def _prompt_seen_by(build: Any) -> str:
    """The instructions a harness installs, read back from the model that receives them.

    Asserting on ``harness.instructions(...)`` would only prove the helper composes what
    it was handed; the question is which gate each *builder* passes it, so the prompt is
    read where the model gets it.
    """
    seen: dict[str, str] = {}

    def spy(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["text"] = info.instructions or ""
        return ModelResponse(parts=[TextPart("ok")])

    build(FunctionModel(spy)).run_sync("hello")
    return seen["text"]


def test_each_harness_is_told_the_gate_it_actually_enforces():
    """The two harnesses enforce one guarantee through different machinery.

    The terminal wrapper returns ``{"denied": …}``; the web harness's denial is the SDK's
    ``ToolDenied``, whose message is "The tool call was denied." — no ``denied`` key
    anywhere. One shared instruction told *both* models they were in a terminal chat and
    to watch for the terminal payload, so the web model was given a marker it could never
    see and the "never retry the same call" rule beside it had nothing to key on.
    """
    terminal = _prompt_seen_by(lambda model: harness.build_agent(model, confirm=_allow))
    web = _prompt_seen_by(harness.build_web_agent)

    assert "terminal chat" in terminal
    assert "terminal chat" not in web
    assert "chat panel" in web

    # The denial marker each model will really receive, and only that one.
    assert "`denied` key" in terminal
    assert "`denied` key" not in web
    assert "denied" in web  # it is still told denials happen — just not their spelling

    # Whatever else differs, the shared halves stay shared.
    for text in (terminal, web):
        assert "operating guide" in text
        assert "wait for a go-ahead before start_run" in text


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
    agent = harness.build_agent(_scripted_scaffold(config), confirm=lambda _tool, _args: False)
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
    """The second turn carries the first turn's messages — the web twin's assertion.

    Two model turns before ``exit``, scripted through a ``FunctionModel`` that counts
    the messages each turn receives, exactly as ``test_gui.py``'s
    ``test_chat_turns_thread_history`` does for the panel. One turn plus a banner count
    — this test's old shape — stayed green with the REPL's ``message_history=``
    threading deleted entirely, because no second successful turn ever ran.
    """
    import typer

    from chemrefine.agent import chat

    monkeypatch.setenv("CHEMREFINE_LLM_MODEL", "test-model")
    seen: list[int] = []

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(len(messages))
        return ModelResponse(parts=[TextPart(f"reply {len(seen)}")])

    monkeypatch.setattr(ProviderConfig, "build_model", lambda self: FunctionModel(script))
    prompts = iter(["hello there", "and after that?", "exit"])
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: next(prompts))
    echoed: list[str] = []
    monkeypatch.setattr(typer, "echo", echoed.append)

    chat.main(provider="custom")
    assert any("ChemRefine agent" in line for line in echoed)
    assert "reply 1" in echoed and "reply 2" in echoed
    assert seen[1] > seen[0], "the second turn must carry the first turn's messages"


def test_chat_repl_ends_on_eof(monkeypatch: pytest.MonkeyPatch):
    import typer

    from chemrefine.agent import chat

    monkeypatch.setenv("CHEMREFINE_LLM_MODEL", "test-model")
    monkeypatch.setattr(ProviderConfig, "build_model", lambda self: TestModel(call_tools=[]))

    def eof(*args: Any, **kwargs: Any) -> str:
        raise EOFError

    monkeypatch.setattr(typer, "prompt", eof)
    chat.main(provider="custom")  # returns instead of crashing


def test_chat_repl_survives_a_dead_endpoint(monkeypatch: pytest.MonkeyPatch):
    """A failed model turn is an answer, not an exit — the next turn still runs.

    Endpoint-down is the likeliest failure this feature meets (it is why ``--check``
    exists), and it used to take the whole conversation down as a traceback. The web
    harness answers the same failure per turn (``test_gui``'s 502 test); the REPL now
    mirrors that posture.
    """
    import typer

    from chemrefine.agent import chat

    turns = iter([ConnectionError("connection refused"), "recovered"])

    def flaky(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        turn = next(turns)
        if isinstance(turn, Exception):
            raise turn
        return ModelResponse(parts=[TextPart(turn)])

    monkeypatch.setenv("CHEMREFINE_LLM_MODEL", "test-model")
    monkeypatch.setattr(ProviderConfig, "build_model", lambda self: FunctionModel(flaky))
    prompts = iter(["hello", "again", "exit"])
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: next(prompts))
    echoed: list[str] = []
    monkeypatch.setattr(typer, "echo", lambda m, **k: echoed.append(str(m)))

    chat.main(provider="custom")
    failed = next(i for i, line in enumerate(echoed) if "model endpoint failed" in line)
    assert any("recovered" in line for line in echoed[failed + 1 :])


def test_chat_repl_keeps_a_tool_errors_documented_shape(monkeypatch: pytest.MonkeyPatch):
    """A ``ChemRefineError`` out of a turn prints its name and message, and the loop lives."""
    import typer

    from chemrefine.agent import chat

    def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ConfigError("no step matches 'x'")

    monkeypatch.setenv("CHEMREFINE_LLM_MODEL", "test-model")
    monkeypatch.setattr(ProviderConfig, "build_model", lambda self: FunctionModel(broken))
    prompts = iter(["hello", "exit"])
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: next(prompts))
    echoed: list[str] = []
    monkeypatch.setattr(typer, "echo", lambda m, **k: echoed.append(str(m)))

    chat.main(provider="custom")
    assert any("ConfigError: no step matches" in line for line in echoed)


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


def test_cli_agent_names_the_missing_extra(without_extra, caplog):
    """Without PydanticAI the command names the extra and exits 1 — not a traceback.

    Like its GUI twin, this used to raise on ``chemrefine.agent`` rather than on the SDK,
    so it passed while the guard could not fire: PydanticAI was imported inside
    ``build_agent``, well past the ``except ImportError``.
    """
    from typer.testing import CliRunner

    from chemrefine.cli import app

    without_extra("pydantic_ai", purge=("chemrefine.agent.chat", "chemrefine.agent.harness"))
    result = CliRunner().invoke(app, ["agent", "--model", "openai:gpt-5-mini"])
    assert result.exit_code == 1
    assert "chemrefine[agent]" in caplog.text


def test_agent_check_works_before_the_extra_is_installed(
    without_extra, monkeypatch: pytest.MonkeyPatch
):
    """``--check`` is a preflight, so it must run in the environment it diagnoses.

    ``providers`` states that it imports no SDK at runtime, and ``cli.py`` depends on that
    by routing ``--check`` through ``chemrefine.agent.providers`` rather than through
    ``chat``. Nothing held it: one import added to ``agent/__init__.py``, or an eager SDK
    import in ``providers``, would break the preflight silently — and the preflight exists
    for people who have not installed the extra yet. A provider-native model string
    returns before any network call, so this stays offline.
    """
    from typer.testing import CliRunner

    from chemrefine.cli import app

    for name in ("CHEMREFINE_LLM_MODEL", "CHEMREFINE_LLM_BASE_URL", "CHEMREFINE_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    without_extra("pydantic_ai", purge=("chemrefine.agent.chat", "chemrefine.agent.harness"))
    result = CliRunner().invoke(app, ["agent", "--check", "--model", "openai:gpt-5-mini"])
    assert result.exit_code == 0
    assert "not probed" in result.stdout
