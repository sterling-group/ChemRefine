"""The GUI backend: token-gated JSON endpoints, one YAML implementation, no surprises.

Driven through Flask's ``test_client`` against the real app factory — the same route as
the browser, minus the socket. What must hold: every ``/api/*`` endpoint refuses a
missing/wrong token (the static page stays open — it holds no secrets), the YAML
emit/parse pair round-trips a real config (server-side PyYAML is the *single* YAML
implementation, so the form pane and text pane cannot drift), library errors surface as
the documented ``{error, exit_code}`` shape, and ``serve.launch`` binds loopback on the
stable per-user port (kernel-assigned when that one is taken) with a fresh token and the
effective port in the URL it opens — printing the SSH forwarding recipe, never launching
a text browser, whenever nothing here would open a *window*. A display is not that
question: ``ssh -X`` sets one on nodes whose only browser is lynx.
"""

from __future__ import annotations

import logging
import types
import webbrowser
from pathlib import Path
from typing import Any

import pytest
import yaml

import chemrefine.gui.serve as serve_mod
from chemrefine.gui.app import create_app

TOKEN = "test-token"


@pytest.fixture
def client(tmp_path: Path):
    """A test client for an app launched without a preloaded config."""
    return create_app(token=TOKEN).test_client()


def _get(client: Any, path: str, **kwargs: Any) -> Any:
    return client.get(path, headers={"X-ChemRefine-Token": TOKEN}, **kwargs)


def _post(client: Any, path: str, payload: dict[str, Any]) -> Any:
    return client.post(path, json=payload, headers={"X-ChemRefine-Token": TOKEN})


# ---------------------------------------------------------------------------
# Gate + page
# ---------------------------------------------------------------------------


def test_the_page_is_open_but_the_api_is_gated(client: Any):
    page = client.get("/")
    assert page.status_code == 200
    page.close()  # the file-backed response must be closed under filterwarnings=error
    assert client.get("/api/bootstrap").status_code == 401  # no token
    bad = client.get("/api/bootstrap", headers={"X-ChemRefine-Token": "wrong"})
    assert bad.status_code == 401
    # Non-ASCII is still just a wrong token. Werkzeug decodes headers as latin-1, and
    # `compare_digest` refuses a non-ASCII str, so this used to be a 500 with a traceback
    # from inside the auth gate — the one place that should answer plainly.
    exotic = client.get("/api/bootstrap", headers={"X-ChemRefine-Token": "wröng"})
    assert exotic.status_code == 401
    assert _get(client, "/api/bootstrap").status_code == 200


def test_every_route_is_behind_the_gate(client: Any):
    """Walk the route table: everything but the page and its assets answers 401 bare.

    The gate is ``request.path.startswith("/api/")`` — a spelling convention. A future
    route registered outside the prefix would ship unauthenticated while every
    hand-written 401 test stayed green; enumerating the url_map turns the convention
    into an invariant.
    """
    ungated = {"index", "static"}  # the same files the docs site publishes openly
    rules = [r for r in client.application.url_map.iter_rules() if r.endpoint not in ungated]
    assert len(rules) > 10  # the walk really covers the API surface
    for rule in rules:
        assert "<" not in rule.rule, f"{rule.endpoint}: a parameterized route needs its own probe"
        method = "GET" if "GET" in rule.methods else "POST"
        response = client.open(rule.rule, method=method)
        assert response.status_code == 401, f"{rule.endpoint} answers {response.status_code} bare"


def test_a_tokenless_app_is_the_unit_test_affordance(tmp_path: Path):
    open_app = create_app(token=None).test_client()
    assert open_app.get("/api/bootstrap").status_code == 200


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_carries_the_schema_and_no_initial_without_a_config(client: Any):
    import socket

    data = _get(client, "/api/bootstrap").get_json()
    assert "config" in data["schema"]
    assert "engines" in data["schema"]
    assert data["initial"] is None
    # Over SSH forwarding the address bar always says 127.0.0.1 — this is how the UI
    # can say where Save… actually writes.
    assert data["host"] == socket.gethostname()


def test_bootstrap_preloads_a_launched_config(tmp_path: Path):
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump({"steps": [{"step": 1, "engine": "fake"}]}), "utf-8")
    app = create_app(token=TOKEN, config_path=config).test_client()
    initial = _get(app, "/api/bootstrap").get_json()["initial"]
    assert initial["path"] == str(config)
    assert "steps" in initial["yaml_text"]


# ---------------------------------------------------------------------------
# YAML round-trip + validation
# ---------------------------------------------------------------------------


def test_yaml_emit_and_parse_round_trip(client: Any):
    config = {
        "template_dir": "./templates",
        "steps": [{"step": 1, "engine": "orca", "sample": {"method": "min", "count": 2}}],
    }
    emitted = _post(client, "/api/yaml", {"config": config}).get_json()["yaml_text"]
    parsed = _post(client, "/api/parse", {"yaml_text": emitted}).get_json()["config"]
    assert parsed == config


def test_yaml_emits_settings_first_and_steps_last(client: Any):
    """Click order must not leak into the file: a fresh session builds cfg from
    ``{steps: []}``, but the YAML reads like the shipped examples — settings above."""
    config = {
        "steps": [{"engine": "orca", "name": "refine", "step": 1}],
        "output_dir": "./outputs",
        "template_dir": "./templates",
        "unknown_key": 1,
    }
    emitted = _post(client, "/api/yaml", {"config": config}).get_json()["yaml_text"]
    lines = [line.split(":")[0] for line in emitted.splitlines() if line and line[0] != " "]
    assert lines.index("template_dir") < lines.index("output_dir") < lines.index("steps")
    assert lines.index("steps") < lines.index("unknown_key")  # junk sorts last, validation's job
    # Step keys follow StepConfig order: step before name before engine.
    body = emitted[emitted.index("steps:") :]
    assert body.index("step:") < body.index("name:") < body.index("engine:")

    # Degenerate raw-edit shapes pass through the emitter unharmed — complaining
    # about them is validation's job.
    non_dict = _post(client, "/api/yaml", {"config": ["not", "a", "mapping"]}).get_json()
    assert "not" in non_dict["yaml_text"]
    odd_steps = _post(client, "/api/yaml", {"config": {"steps": "tbd"}}).get_json()
    assert "tbd" in odd_steps["yaml_text"]


def test_parse_reports_bad_yaml_as_400(client: Any):
    bad = _post(client, "/api/parse", {"yaml_text": "steps: [unclosed"})
    assert bad.status_code == 400
    assert "malformed YAML" in bad.get_json()["error"]
    not_mapping = _post(client, "/api/parse", {"yaml_text": "- a\n- list\n"})
    assert not_mapping.status_code == 400


def test_validate_returns_the_structured_report(client: Any, tmp_path: Path):
    report = _post(
        client,
        "/api/validate",
        {
            "yaml_text": yaml.safe_dump({"steps": [{"step": 1, "engine": "nope"}]}),
            "base_dir": str(tmp_path),
        },
    ).get_json()
    assert report["ok"] is False
    assert report["issues"][0]["kind"] == "engine"


# ---------------------------------------------------------------------------
# Filesystem: browse + save
# ---------------------------------------------------------------------------


def test_browse_lists_directories_first_and_skips_dotfiles(client: Any, tmp_path: Path):
    (tmp_path / "zeta").mkdir()
    (tmp_path / "alpha.yaml").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden").write_text("x", encoding="utf-8")
    data = _get(client, f"/api/browse?path={tmp_path}").get_json()
    assert [e["name"] for e in data["entries"]] == ["zeta", "alpha.yaml"]
    assert data["parent"] == str(tmp_path.parent)
    assert _get(client, f"/api/browse?path={tmp_path}/alpha.yaml").status_code == 400


def test_browse_defaults_to_home(client: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert _get(client, "/api/browse").get_json()["path"] == str(tmp_path)


def test_browse_answers_an_unreadable_directory_with_a_400(client: Any, tmp_path: Path):
    """A picker walks wherever the filesystem leads — an unreadable stop is a 400, not a 500."""
    import os

    if os.geteuid() == 0:
        pytest.skip("root reads everything; the permission wall cannot be built")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        response = _get(client, f"/api/browse?path={locked}")
    finally:
        locked.chmod(0o755)
    assert response.status_code == 400
    assert "cannot list" in response.get_json()["error"]


def test_save_writes_the_artifact(client: Any, tmp_path: Path):
    destination = tmp_path / "proj" / "input.yaml"
    saved = _post(
        client, "/api/save", {"path": str(destination), "yaml_text": "steps: []\n"}
    ).get_json()
    assert saved["path"] == str(destination)
    assert destination.read_text(encoding="utf-8") == "steps: []\n"


def test_save_answers_an_unwritable_destination_with_a_400(client: Any, tmp_path: Path):
    """Save-as into a directory the user cannot write is a plain 400, like any bad input."""
    import os

    if os.geteuid() == 0:
        pytest.skip("root writes everywhere; the permission wall cannot be built")
    fortress = tmp_path / "fortress"
    fortress.mkdir()
    fortress.chmod(0o555)
    try:
        response = _post(
            client, "/api/save", {"path": str(fortress / "input.yaml"), "yaml_text": "steps: []\n"}
        )
    finally:
        fortress.chmod(0o755)
    assert response.status_code == 400
    assert "cannot write" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# Library seams: scaffold, template editor, summary, error shape
# ---------------------------------------------------------------------------


def _saved_config(tmp_path: Path) -> Path:
    path = tmp_path / "input.yaml"
    path.write_text(
        yaml.safe_dump({"steps": [{"step": 1, "name": "refine", "engine": "orca"}]}), "utf-8"
    )
    return path


def test_scaffold_then_template_roundtrip(client: Any, tmp_path: Path):
    config = _saved_config(tmp_path)
    scaffolded = _post(client, "/api/scaffold", {"config_path": str(config)}).get_json()
    assert any(p.endswith("step1.inp") for p in scaffolded["written"])

    read = _get(client, f"/api/template?config_path={config}&step=1").get_json()
    assert "%pal" in read["text"]

    _post(
        client,
        "/api/template",
        {"config_path": str(config), "step": "refine", "text": "! Mine\n"},
    )
    again = _get(client, f"/api/template?config_path={config}&step=refine").get_json()
    assert again["text"] == "! Mine\n"


def test_summary_mirrors_the_config(client: Any, tmp_path: Path):
    config = _saved_config(tmp_path)
    summary = _post(client, "/api/summary", {"config_path": str(config)}).get_json()
    assert summary["steps"][0]["engine"] == "orca"


def test_library_errors_carry_the_exit_code_shape(client: Any, tmp_path: Path):
    config = _saved_config(tmp_path)
    missing = _get(client, f"/api/template?config_path={config}&step=9")
    assert missing.status_code == 400
    body = missing.get_json()
    assert "no step matches" in body["error"]
    assert body["exit_code"] == 2


@pytest.mark.parametrize("step", ["²", "①"])
def test_a_digit_that_is_not_a_number_is_still_a_400(client: Any, tmp_path: Path, step: str):
    """``"²".isdigit()`` is True and ``int("²")`` raises — so the guard let one through.

    The selector was routed to ``int()`` by ``isdigit``, which accepts the Unicode ``No``
    category ``int`` rejects. The bare ``ValueError`` is neither an ``HTTPException`` nor
    a ``ChemRefineError``, so the error handler re-raised it: a 500 and a logged traceback
    where every other unusable selector is the documented 400.
    """
    config = _saved_config(tmp_path)
    response = _get(client, f"/api/template?config_path={config}&step={step}")
    assert response.status_code == 400
    assert "no step matches" in response.get_json()["error"]


# ---------------------------------------------------------------------------
# Run dashboard endpoints — thin over agent_tools, against pipeline-writer fixtures
# ---------------------------------------------------------------------------


def _reported_tree(tmp_path: Path) -> Path:
    """A saved config whose output tree carries pipeline-written status artifacts."""
    from chemrefine import io
    from chemrefine.cache import save_failure_records
    from chemrefine.state import FailureKind, FailureRecord

    path = tmp_path / "input.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "steps": [
                    {"step": 1, "name": "screen", "engine": "fake"},
                    {"step": 2, "engine": "fake"},
                ]
            }
        ),
        encoding="utf-8",
    )
    outputs = tmp_path / "outputs"
    io.save_step_csv(
        energies_hartree=[-1.0, -0.9], structure_ids=["0", "1"], step_number=1, output_dir=outputs
    )
    save_failure_records(
        outputs / "step1_screen",
        [FailureRecord(structure_id="2", kind=FailureKind.MISSING_OUTPUT, reason="no output")],
    )
    return path


def test_dashboard_status_results_failures(client: Any, tmp_path: Path):
    config = _reported_tree(tmp_path)

    status = _post(client, "/api/status", {"config_path": str(config)}).get_json()
    assert status["running"] is False
    assert status["steps"][0]["reported_survivors"] == 2
    assert status["steps"][0]["failures"] == 1

    results = _post(
        client, "/api/results", {"config_path": str(config), "step": 1, "limit": 1}
    ).get_json()
    assert results["total"] == 2
    assert len(results["rows"]) == 1

    failures = _post(client, "/api/failures", {"config_path": str(config)}).get_json()
    assert failures["failures"][0]["structure_id"] == "2"
    assert failures["suggested_action"] == "rerun-errors"


def test_dashboard_run_launches_detached(
    client: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import subprocess

    calls: list[dict[str, Any]] = []

    class _Recorded:
        def __init__(self, argv: list[str], **kwargs: Any) -> None:
            self.pid = 4242
            calls.append({"argv": argv, **kwargs})

    monkeypatch.setattr(subprocess, "Popen", _Recorded)
    config = _reported_tree(tmp_path)
    started = _post(
        client, "/api/run", {"config_path": str(config), "action": "rerun-errors"}
    ).get_json()
    assert started["pid"] == 4242
    [call] = calls
    assert call["argv"][3] == "rerun-errors"
    assert call["start_new_session"] is True


def test_dashboard_run_surfaces_a_held_lock_as_exit_code_10(client: Any, tmp_path: Path):
    import json as jsonlib
    import os
    import socket

    from chemrefine import pipeline

    config = _reported_tree(tmp_path)
    outputs = tmp_path / "outputs"
    (outputs / pipeline.RUN_LOCK_NAME).write_text(
        jsonlib.dumps({"host": socket.gethostname(), "pid": os.getpid(), "started": "now"}),
        encoding="utf-8",
    )
    refused = _post(client, "/api/run", {"config_path": str(config)})
    assert refused.status_code == 400
    assert refused.get_json()["exit_code"] == 10


# ---------------------------------------------------------------------------
# Agent chat endpoints — deferred approvals over HTTP, proven offline
# ---------------------------------------------------------------------------


def _script_model(config_path: Path) -> Any:
    """A model that asks to scaffold ``config_path`` once, then reports done."""
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart("scaffold_templates", {"config_path": str(config_path)})]
            )
        return ModelResponse(parts=[TextPart("finished")])

    return FunctionModel(script)


@pytest.fixture
def chat_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setenv("CHEMREFINE_LLM_MODEL", "scripted")
    return monkeypatch


def _inject_model(monkeypatch: pytest.MonkeyPatch, model: Any) -> None:
    from chemrefine.agent.providers import ProviderConfig

    monkeypatch.setattr(ProviderConfig, "build_model", lambda self: model)


def test_agent_availability_reports_the_three_states(client: Any, monkeypatch: pytest.MonkeyPatch):
    import sys

    monkeypatch.setenv("CHEMREFINE_LLM_MODEL", "some-model")
    ready = _get(client, "/api/agent/availability").get_json()
    assert ready == {"installed": True, "configured": True, "detail": "some-model"}

    monkeypatch.delenv("CHEMREFINE_LLM_MODEL")
    unconfigured = _get(client, "/api/agent/availability").get_json()
    assert unconfigured["installed"] is True
    assert unconfigured["configured"] is False

    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    missing = _get(client, "/api/agent/availability").get_json()
    assert missing["installed"] is False
    assert "chemrefine[agent]" in missing["detail"]


def test_chat_turns_thread_history(client: Any, chat_env: pytest.MonkeyPatch):
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    seen: list[int] = []

    def script(messages: Any, info: Any) -> ModelResponse:
        seen.append(len(messages))
        return ModelResponse(parts=[TextPart("reply " + str(len(seen)))])

    _inject_model(chat_env, FunctionModel(script))
    first = _post(client, "/api/agent/chat", {"message": "hello"}).get_json()
    assert first == {"reply": "reply 1", "pending": None}
    _post(client, "/api/agent/chat", {"message": "again"})
    assert seen[1] > seen[0]  # the second turn carried the first turn's messages


def test_chat_gates_mutations_behind_http_approvals(
    client: Any, chat_env: pytest.MonkeyPatch, tmp_path: Path
):
    config = _saved_config(tmp_path)
    _inject_model(chat_env, _script_model(config))

    suspended = _post(client, "/api/agent/chat", {"message": "scaffold it"}).get_json()
    [request_card] = suspended["pending"]
    assert request_card["tool"] == "scaffold_templates"
    assert str(config) in str(request_card["args"])
    assert not (tmp_path / "templates").exists()  # suspended means nothing ran

    resumed = _post(client, "/api/agent/chat", {"approvals": {request_card["id"]: True}}).get_json()
    assert resumed["reply"] == "finished"
    assert (tmp_path / "templates" / "step1.inp").is_file()


def test_chat_denial_keeps_the_disk_untouched(
    client: Any, chat_env: pytest.MonkeyPatch, tmp_path: Path
):
    config = _saved_config(tmp_path)
    _inject_model(chat_env, _script_model(config))
    suspended = _post(client, "/api/agent/chat", {"message": "scaffold it"}).get_json()
    [request_card] = suspended["pending"]
    resumed = _post(
        client, "/api/agent/chat", {"approvals": {request_card["id"]: False}}
    ).get_json()
    assert resumed["reply"] == "finished"
    assert not (tmp_path / "templates").exists()


def test_chat_refuses_approvals_with_nothing_pending(client: Any, chat_env: pytest.MonkeyPatch):
    from pydantic_ai.models.test import TestModel

    _inject_model(chat_env, TestModel(call_tools=[]))
    refused = _post(client, "/api/agent/chat", {"approvals": {"x": True}})
    assert refused.status_code == 400
    assert "no suspended run" in refused.get_json()["error"]


def test_chat_keeps_tool_errors_in_the_documented_shape(client: Any, chat_env: pytest.MonkeyPatch):
    """A ChemRefineError from a tool is a 400 with its exit code — never a 502."""
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart("summarize_config", {"config_path": "/absent.yaml"})]
            )
        return ModelResponse(parts=[TextPart("unreached")])

    _inject_model(chat_env, FunctionModel(script))
    failed = _post(client, "/api/agent/chat", {"message": "summarize"})
    assert failed.status_code == 400
    assert failed.get_json()["exit_code"] == 2


def test_chat_maps_endpoint_failures_to_502(client: Any, chat_env: pytest.MonkeyPatch):
    from pydantic_ai.models.function import FunctionModel

    def explode(messages: Any, info: Any) -> Any:
        raise RuntimeError("connection refused by nobody:11434")

    _inject_model(chat_env, FunctionModel(explode))
    failed = _post(client, "/api/agent/chat", {"message": "hi"})
    assert failed.status_code == 502
    assert "model endpoint failed" in failed.get_json()["error"]


def test_chat_reset_forgets_the_conversation(client: Any, chat_env: pytest.MonkeyPatch):
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    seen: list[int] = []

    def script(messages: Any, info: Any) -> ModelResponse:
        seen.append(len(messages))
        return ModelResponse(parts=[TextPart("ok")])

    _inject_model(chat_env, FunctionModel(script))
    _post(client, "/api/agent/chat", {"message": "one"})
    reset = _post(client, "/api/agent/chat", {"reset": True}).get_json()
    assert reset["reset"] is True
    _post(client, "/api/agent/chat", {"message": "two"})
    assert seen[0] == seen[1]  # the second conversation started fresh


def test_non_chemrefine_errors_are_not_swallowed(client: Any):
    """Only ChemRefineError gets the JSON shape; a genuine bug must stay a loud bug."""
    with pytest.raises(KeyError):
        _post(client, "/api/validate", {})  # missing yaml_text → KeyError, not a 400


def test_routine_http_errors_stay_routine(client: Any):
    """A stray URL is a plain 404 response, never a logged traceback.

    Browsers probe /favicon.ico on every visit; re-raising the NotFound through the
    catch-all handler printed a full traceback per page load.
    """
    response = client.get("/favicon.ico")
    assert response.status_code == 404
    response.close()


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def test_cli_gui_hands_off_to_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from typer.testing import CliRunner

    from chemrefine.cli import app as cli_app
    from chemrefine.gui import serve

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        serve,
        "launch",
        lambda config, *, port, open_browser: calls.append(
            {"config": config, "port": port, "open_browser": open_browser}
        ),
    )
    config = tmp_path / "input.yaml"
    config.write_text("steps: []\n", encoding="utf-8")
    result = CliRunner().invoke(cli_app, ["gui", str(config), "--port", "8123", "--no-browser"])
    assert result.exit_code == 0
    assert calls == [{"config": config, "port": 8123, "open_browser": False}]


def test_cli_gui_omitting_port_asks_for_the_personal_default(monkeypatch: pytest.MonkeyPatch):
    """No --port hands ``None`` to launch — the port policy lives in serve, not here."""
    from typer.testing import CliRunner

    from chemrefine.cli import app as cli_app
    from chemrefine.gui import serve

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        serve,
        "launch",
        lambda config, *, port, open_browser: calls.append(
            {"config": config, "port": port, "open_browser": open_browser}
        ),
    )
    result = CliRunner().invoke(cli_app, ["gui", "--no-browser"])
    assert result.exit_code == 0
    assert calls == [{"config": None, "port": None, "open_browser": False}]


def test_cli_gui_reports_a_taken_port(monkeypatch: pytest.MonkeyPatch, caplog):
    """A busy --port exits 1 with the fix named — not a waitress traceback."""
    from typer.testing import CliRunner

    from chemrefine.cli import app as cli_app
    from chemrefine.gui import serve

    def taken(config: Any, *, port: int | None, open_browser: bool) -> None:
        raise OSError(98, "Address already in use")

    monkeypatch.setattr(serve, "launch", taken)
    result = CliRunner().invoke(cli_app, ["gui", "--port", "8123"])
    assert result.exit_code == 1
    assert "--port 0" in caplog.text


def test_cli_gui_reports_a_bind_failure_on_the_default_port(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    """An OSError with no --port set must not reach the ``%d`` formatter with ``None``."""
    from typer.testing import CliRunner

    from chemrefine.cli import app as cli_app
    from chemrefine.gui import serve

    def refused(config: Any, *, port: int | None, open_browser: bool) -> None:
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(serve, "launch", refused)
    result = CliRunner().invoke(cli_app, ["gui"])
    assert result.exit_code == 1
    assert "could not serve the GUI" in caplog.text


def test_cli_gui_names_the_missing_extra(without_extra, caplog):
    """Without Flask/waitress the command names the extra and exits 1 — not a traceback.

    This passed for years while the guard was dead: it raised on ``chemrefine.gui``, our
    own subpackage, rather than on the server the extra provides. ``chemrefine gui`` on a
    bare install actually printed a traceback past the handler written to prevent it.
    """
    from typer.testing import CliRunner

    from chemrefine.cli import app as cli_app

    without_extra(
        "flask", "waitress", "werkzeug", purge=("chemrefine.gui.app", "chemrefine.gui.serve")
    )
    result = CliRunner().invoke(cli_app, ["gui"])
    assert result.exit_code == 1
    assert "chemrefine[gui]" in caplog.text
    assert "waitress" in caplog.text


# ---------------------------------------------------------------------------
# serve.launch
# ---------------------------------------------------------------------------


class _FakeServer:
    """Stands in for waitress: records that it ran instead of blocking."""

    def __init__(self, record: dict[str, Any]) -> None:
        self._record = record

    def run(self) -> None:
        self._record["ran"] = True


def _fake_socket_module(
    binds: list[tuple[str, int]],
    *,
    refuse: frozenset[int] = frozenset(),
    effective: int = 43210,
) -> Any:
    """A stand-in ``socket`` module: records binds, refuses named ports, opens no fd."""

    class _Sock:
        def bind(self, addr: tuple[str, int]) -> None:
            binds.append(addr)
            if addr[1] in refuse:
                raise OSError(98, "Address already in use")

        def getsockname(self) -> tuple[str, int]:
            return ("127.0.0.1", effective)

    return types.SimpleNamespace(
        AF_INET=0, SOCK_STREAM=0, socket=lambda *a: _Sock(), gethostname=lambda: "login03"
    )


def _fake_browser(opened: list[str], *, result: bool = True, windowed: bool = True) -> Any:
    """A stand-in ``webbrowser`` that records calls and answers *what kind* of browser is here.

    ``get()`` returns a real controller instance, not a mock, because that is the whole
    question ``_opens_a_window`` asks: the console browsers register as
    :class:`~webbrowser.GenericBrowser` and block, every windowed launcher registers as
    :class:`~webbrowser.BackgroundBrowser` and does not. Constructing either is inert —
    the class only stores a name until something calls ``open`` on it.

    ``windowed=False`` is the X11-forwarded browserless login node: a display is set, so
    the old ``DISPLAY`` check said "not headless", and lynx got the terminal.
    """
    controller = (
        webbrowser.BackgroundBrowser("xdg-open") if windowed else webbrowser.GenericBrowser("lynx")
    )

    def _open(url: str) -> bool:
        opened.append(url)
        return result

    return types.SimpleNamespace(
        open=_open,
        get=lambda *a: controller,
        BackgroundBrowser=webbrowser.BackgroundBrowser,
        Error=webbrowser.Error,
    )


def test_personal_port_is_stable_and_in_range(monkeypatch: pytest.MonkeyPatch):
    """Same user, same port, every session — what makes a one-time forwarding stanza work."""
    monkeypatch.setattr(serve_mod.getpass, "getuser", lambda: "ada")
    first = serve_mod._personal_port()
    assert first == serve_mod._personal_port()
    assert serve_mod._PORT_BASE <= first < serve_mod._PORT_BASE + serve_mod._PORT_SPAN
    monkeypatch.setattr(serve_mod.getpass, "getuser", lambda: "grace")
    assert serve_mod._personal_port() != first


def test_personal_port_survives_a_passwdless_environment(monkeypatch: pytest.MonkeyPatch):
    """``getpass.getuser`` raises under an arbitrary UID; the numeric UID stands in."""

    def no_passwd_entry() -> str:
        raise OSError("no passwd entry")

    monkeypatch.setattr(serve_mod.getpass, "getuser", no_passwd_entry)
    monkeypatch.setattr(serve_mod.os, "getuid", lambda: 4242)
    port = serve_mod._personal_port()
    assert serve_mod._PORT_BASE <= port < serve_mod._PORT_BASE + serve_mod._PORT_SPAN


def test_launch_binds_loopback_with_a_fresh_token(monkeypatch: pytest.MonkeyPatch, caplog):
    created: dict[str, Any] = {}
    opened: list[str] = []
    binds: list[tuple[str, int]] = []

    def fake_create_server(app: Any, *, sockets: list[Any]) -> _FakeServer:
        created["sockets"] = sockets
        return _FakeServer(created)

    # Patched on `serve`, not on `waitress.server`: the name is bound at import time
    # (module scope, so the CLI's ImportError guard can fire), so patching the origin
    # after the fact leaves `launch` holding the real one — which binds a real socket and
    # runs a real server, and the suite then hangs on waitress's handler threads.
    monkeypatch.setattr(serve_mod, "create_server", fake_create_server)
    monkeypatch.setattr(serve_mod.getpass, "getuser", lambda: "ada")
    personal = serve_mod._personal_port()
    monkeypatch.setattr(serve_mod, "socket", _fake_socket_module(binds, effective=personal))
    monkeypatch.setattr(serve_mod, "webbrowser", _fake_browser(opened))
    with caplog.at_level(logging.INFO, logger="chemrefine.gui.serve"):
        serve_mod.launch(None, open_browser=True)

    assert binds == [("127.0.0.1", personal)]  # the default is the personal port
    assert created["ran"] is True
    assert len(created["sockets"]) == 1  # waitress serves the pre-bound socket
    [url] = opened
    assert url.startswith(f"http://127.0.0.1:{personal}/?token=")
    assert "ssh -L" not in caplog.text  # a browser opened; nobody needs the recipe


def test_a_taken_personal_port_falls_back_to_a_kernel_one(monkeypatch: pytest.MonkeyPatch, caplog):
    """A squatted personal port degrades to a free one, and the URL names the real port."""
    created: dict[str, Any] = {}
    binds: list[tuple[str, int]] = []
    monkeypatch.setattr(serve_mod, "create_server", lambda app, *, sockets: _FakeServer(created))
    monkeypatch.setattr(serve_mod.getpass, "getuser", lambda: "ada")
    personal = serve_mod._personal_port()
    monkeypatch.setattr(
        serve_mod,
        "socket",
        _fake_socket_module(binds, refuse=frozenset({personal}), effective=51423),
    )
    with caplog.at_level(logging.INFO, logger="chemrefine.gui.serve"):
        serve_mod.launch(None, open_browser=False)
    assert binds == [("127.0.0.1", personal), ("127.0.0.1", 0)]
    assert "http://127.0.0.1:51423/?token=" in caplog.text


def test_an_explicit_taken_port_is_not_second_guessed(monkeypatch: pytest.MonkeyPatch):
    """A port the user named raises OSError to the CLI — no silent fallback."""
    binds: list[tuple[str, int]] = []
    monkeypatch.setattr(serve_mod, "socket", _fake_socket_module(binds, refuse=frozenset({8123})))
    with pytest.raises(OSError):
        serve_mod.launch(None, port=8123, open_browser=False)
    assert binds == [("127.0.0.1", 8123)]


def test_launch_can_keep_the_browser_closed(monkeypatch: pytest.MonkeyPatch, caplog):
    opened: list[str] = []
    binds: list[tuple[str, int]] = []
    created: dict[str, Any] = {}

    monkeypatch.setattr(serve_mod, "create_server", lambda app, *, sockets: _FakeServer(created))
    monkeypatch.setattr(serve_mod, "socket", _fake_socket_module(binds, effective=1))
    monkeypatch.setattr(serve_mod, "webbrowser", _fake_browser(opened))
    with caplog.at_level(logging.INFO, logger="chemrefine.gui.serve"):
        serve_mod.launch(None, port=0, open_browser=False)
    assert opened == []
    assert binds == [("127.0.0.1", 0)]  # an explicit 0 still means "the kernel picks"
    assert "ssh -L" not in caplog.text  # --no-browser prints the URL alone


def test_a_headless_session_gets_the_recipe_not_a_text_browser(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    """A DISPLAY-less POSIX session never calls webbrowser — lynx in the terminal is no
    browser — and the recipe names the address the user's own ssh client connected to."""
    created: dict[str, Any] = {}
    opened: list[str] = []
    binds: list[tuple[str, int]] = []
    monkeypatch.setattr(serve_mod, "create_server", lambda app, *, sockets: _FakeServer(created))
    monkeypatch.setattr(serve_mod, "socket", _fake_socket_module(binds, effective=21244))
    monkeypatch.setattr(serve_mod, "webbrowser", _fake_browser(opened))
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.5 50000 192.0.2.7 22")
    with caplog.at_level(logging.INFO, logger="chemrefine.gui.serve"):
        serve_mod.launch(None, port=0, open_browser=True)
    assert opened == []  # never invoked, not even to fail
    assert "ssh -L 21244:127.0.0.1:21244 192.0.2.7" in caplog.text
    assert "LocalForward 21244 127.0.0.1:21244" in caplog.text
    assert "login03" in caplog.text  # the node names itself, but is never the ssh target


def _launch_capturing(monkeypatch: pytest.MonkeyPatch, caplog, **browser: Any) -> list[str]:
    """Run ``launch`` with everything faked out; return the URLs the browser was given."""
    created: dict[str, Any] = {}
    opened: list[str] = []
    binds: list[tuple[str, int]] = []
    monkeypatch.setattr(serve_mod, "create_server", lambda app, *, sockets: _FakeServer(created))
    monkeypatch.setattr(serve_mod, "socket", _fake_socket_module(binds, effective=21244))
    monkeypatch.setattr(serve_mod, "webbrowser", _fake_browser(opened, **browser))
    with caplog.at_level(logging.INFO, logger="chemrefine.gui.serve"):
        serve_mod.launch(None, port=0, open_browser=True)
    return opened


def test_a_display_with_only_a_console_browser_still_gets_the_recipe(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    """``ssh -X`` onto a browserless node: a display is set, and lynx is not a browser.

    The regression this pins. ``DISPLAY`` was the old test, and ``ssh -X`` sets it, so the
    launch called ``webbrowser`` — which registers the console browsers whenever ``TERM``
    is set and returns one when nothing graphical is installed. lynx then took the
    terminal, and because ``GenericBrowser.open`` reports a clean exit as success, the
    recipe was suppressed too: the user got a text browser they did not ask for *instead
    of* the two lines telling them how to reach the GUI from their own machine.

    The old spelling of this test asked for the same scenario and could not fail on it —
    it forced the fake's return value to ``False``, while every real controller here
    returns ``True``.
    """
    opened = _launch_capturing(monkeypatch, caplog, windowed=False)
    assert opened == []  # never invoked: a blocking console browser is not a route
    assert "ssh -L 21244:127.0.0.1:21244" in caplog.text


def test_a_launcher_that_reports_success_does_not_silence_the_recipe(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    """A remote session gets the recipe even when something did open.

    ``BackgroundBrowser.open`` returns ``poll() is None`` — true the instant ``Popen``
    succeeds — so ``xdg-open`` reports success on a node where it went on to find no
    browser at all. Its "yes" is therefore not evidence, and over SSH the tunnel beats
    whatever the forwarded display is doing regardless.
    """
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.5 50000 192.0.2.7 22")
    opened = _launch_capturing(monkeypatch, caplog)
    assert len(opened) == 1  # it did try the local route
    assert "ssh -L 21244:127.0.0.1:21244 192.0.2.7" in caplog.text


def test_a_local_desktop_that_opened_a_browser_stays_quiet(monkeypatch: pytest.MonkeyPatch, caplog):
    """The other half of the rule: at your own keyboard, the recipe is noise."""
    opened = _launch_capturing(monkeypatch, caplog)  # display set, no SSH_* (autouse fixture)
    assert len(opened) == 1
    assert "ssh -L" not in caplog.text


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_desktop_platforms_always_open_a_window(monkeypatch: pytest.MonkeyPatch, platform: str):
    """macOS and Windows open browsers without DISPLAY — the POSIX questions do not apply."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(serve_mod.sys, "platform", platform)
    assert serve_mod._opens_a_window() is True


def test_a_session_with_no_browser_registered_at_all_opens_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    """``webbrowser.get()`` raises ``Error`` when nothing is registered — that is a no."""

    def raise_error(*_a: Any) -> Any:
        raise webbrowser.Error("no browser")

    monkeypatch.setattr(
        serve_mod, "webbrowser", types.SimpleNamespace(get=raise_error, Error=webbrowser.Error)
    )
    assert serve_mod._opens_a_window() is False
