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
    assert _get(client, "/api/bootstrap").status_code == 200


def test_a_tokenless_app_is_the_unit_test_affordance(tmp_path: Path):
    open_app = create_app(token=None).test_client()
    assert open_app.get("/api/bootstrap").status_code == 200


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_carries_the_schema_and_no_initial_without_a_config(client: Any):
    data = _get(client, "/api/bootstrap").get_json()
    assert "config" in data["schema"]
    assert "engines" in data["schema"]
    assert data["initial"] is None


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


def test_save_writes_the_artifact(client: Any, tmp_path: Path):
    destination = tmp_path / "proj" / "input.yaml"
    saved = _post(
        client, "/api/save", {"path": str(destination), "yaml_text": "steps: []\n"}
    ).get_json()
    assert saved["path"] == str(destination)
    assert destination.read_text(encoding="utf-8") == "steps: []\n"


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


def test_cli_gui_names_the_missing_extra(monkeypatch: pytest.MonkeyPatch):
    import builtins

    from typer.testing import CliRunner

    from chemrefine.cli import app as cli_app

    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("chemrefine.gui"):
            raise ImportError("No module named 'flask'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    result = CliRunner().invoke(cli_app, ["gui"])
    assert result.exit_code == 1


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

    import waitress.server

    monkeypatch.setattr(waitress.server, "create_server", fake_create_server)
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

    import waitress.server

    monkeypatch.setattr(waitress.server, "create_server", lambda *a, **k: _Server())
    fake_browser = type("W", (), {"open": staticmethod(opened.append)})
    monkeypatch.setattr(serve_mod, "webbrowser", fake_browser)
    serve_mod.launch(None, open_browser=False)
    assert opened == []
