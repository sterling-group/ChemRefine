"""The GUI backend: token-gated JSON endpoints, one YAML implementation, no surprises.

Driven through Flask's ``test_client`` against the real app factory — the same route as
the browser, minus the socket. What must hold: every ``/api/*`` endpoint refuses a
missing/wrong token (the static page stays open — it holds no secrets), the YAML
emit/parse pair round-trips a real config (server-side PyYAML is the *single* YAML
implementation, so the form pane and text pane cannot drift), library errors surface as
the documented ``{error, exit_code}`` shape, and ``serve.launch`` binds loopback with a
fresh token and the effective port in the URL it opens.
"""

from __future__ import annotations

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


def test_cli_gui_reports_a_taken_port(monkeypatch: pytest.MonkeyPatch, caplog):
    """A busy --port exits 1 with the fix named — not a waitress traceback."""
    from typer.testing import CliRunner

    from chemrefine.cli import app as cli_app
    from chemrefine.gui import serve

    def taken(config: Any, *, port: int, open_browser: bool) -> None:
        raise OSError(98, "Address already in use")

    monkeypatch.setattr(serve, "launch", taken)
    result = CliRunner().invoke(cli_app, ["gui", "--port", "8123"])
    assert result.exit_code == 1
    assert "--port 0" in caplog.text


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


def test_launch_binds_loopback_with_a_fresh_token(monkeypatch: pytest.MonkeyPatch):
    created: dict[str, Any] = {}
    opened: list[str] = []

    class _Server:
        effective_port = 43210

        def run(self) -> None:
            created["ran"] = True

    def fake_create_server(app: Any, host: str, port: int) -> _Server:
        created["host"], created["port"] = host, port
        return _Server()

    # Patched on `serve`, not on `waitress.server`: the name is bound at import time
    # (module scope, so the CLI's ImportError guard can fire), so patching the origin
    # after the fact leaves `launch` holding the real one — which binds a real socket and
    # runs a real server, and the suite then hangs on waitress's handler threads.
    monkeypatch.setattr(serve_mod, "create_server", fake_create_server)
    fake_browser = type("W", (), {"open": staticmethod(opened.append)})
    monkeypatch.setattr(serve_mod, "webbrowser", fake_browser)
    serve_mod.launch(None, port=0, open_browser=True)

    assert (created["host"], created["port"], created["ran"]) == ("127.0.0.1", 0, True)
    [url] = opened
    assert url.startswith("http://127.0.0.1:43210/?token=")


def test_launch_can_keep_the_browser_closed(monkeypatch: pytest.MonkeyPatch):
    opened: list[str] = []

    class _Server:
        effective_port = 1

        def run(self) -> None:
            return None

    monkeypatch.setattr(serve_mod, "create_server", lambda *a, **k: _Server())
    fake_browser = type("W", (), {"open": staticmethod(opened.append)})
    monkeypatch.setattr(serve_mod, "webbrowser", fake_browser)
    serve_mod.launch(None, open_browser=False)
    assert opened == []
