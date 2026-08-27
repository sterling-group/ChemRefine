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
from chemrefine.agent import providers
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
    assert (ready["installed"], ready["configured"], ready["detail"]) == (True, True, "some-model")

    monkeypatch.delenv("CHEMREFINE_LLM_MODEL")
    unconfigured = _get(client, "/api/agent/availability").get_json()
    assert unconfigured["installed"] is True
    assert unconfigured["configured"] is False

    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    missing = _get(client, "/api/agent/availability").get_json()
    assert missing["installed"] is False
    assert "chemrefine[agent]" in missing["detail"]
    # The field shapes ride along in all three states: the panel renders its inputs from
    # them before any check has run, and they are true whether or not the extra is there.
    for state in (ready, unconfigured, missing):
        assert set(state["presets"]) == set(providers._PRESETS)


def test_the_served_preset_shapes_drive_the_panels_fields(client: Any):
    """One preset table, on the server. A JS copy would outrank the environment.

    ``resolve`` lets an explicit base URL beat ``CHEMREFINE_LLM_BASE_URL``, so a frontend
    that prefilled a hardcoded ``localhost:11434`` would silently redirect a user whose
    environment points at their own box. The page gets these as *placeholders* and never
    sends them back.

    ``default_url`` is the effective default, not the raw preset row — which is what lets
    the panel decide field visibility with no special case: ``custom`` is the only entry
    with nowhere to go, so it is the only one that must be told where.
    """
    shapes = _get(client, "/api/agent/availability").get_json()["presets"]
    assert shapes["custom"]["default_url"] is None  # the only one that needs a URL typed
    assert shapes["openai"]["default_url"] == "https://api.openai.com/v1"
    assert shapes["ollama"]["default_url"] == providers._PRESETS["ollama"][0]
    # A preset that ships its own dummy key is one the user must not be asked for a key.
    assert shapes["ollama"]["needs_key"] is False
    assert shapes["vllm"]["needs_key"] is False
    assert shapes["custom"]["needs_key"] is True
    assert shapes["openai"]["needs_key"] is True


def _saving_model(path: Path, yaml_text: str) -> Any:
    """A model that asks to save ``yaml_text`` to ``path`` once, then reports done."""
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart("save_config", {"path": str(path), "yaml_text": yaml_text})]
            )
        return ModelResponse(parts=[TextPart("saved it")])

    return FunctionModel(script)


def test_load_reads_a_config_back_and_refuses_what_it_cannot(client: Any, tmp_path: Path):
    """The missing half of ``/api/save``, and the three ways it can be asked for nothing.

    ``UnicodeDecodeError`` is the one worth naming: it is a ``ValueError``, not an
    ``OSError``, so a binary file picked by mistake would sail past an ``OSError`` handler
    into ``surface`` and come back as a 500 with a traceback, where every other bad input
    to this app is a plain 400.
    """
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump({"steps": [{"step": 1, "engine": "orca"}]}), "utf-8")

    loaded = _get(client, f"/api/load?path={config}")
    assert loaded.status_code == 200
    assert loaded.get_json()["path"] == str(config)
    assert "steps" in loaded.get_json()["yaml_text"]

    assert _get(client, f"/api/load?path={tmp_path / 'nope.yaml'}").status_code == 400
    assert _get(client, f"/api/load?path={tmp_path}").status_code == 400  # a directory

    binary = tmp_path / "binary.yaml"
    binary.write_bytes(b"\xff\xfe\x00\x01")
    undecodable = _get(client, f"/api/load?path={binary}")
    assert undecodable.status_code == 400
    assert "cannot read" in undecodable.get_json()["error"]


def test_a_turn_that_writes_the_config_says_so_on_both_branches(
    client: Any, chat_env: pytest.MonkeyPatch, tmp_path: Path
):
    """The signal the editor reloads on, read off the turn's own messages.

    Reported on the suspended branch as well as the text one: an approved write and a
    fresh batch of approval cards arrive in the *same* turn, so a signal attached only to
    the reply would be dropped exactly while the agent is working steadily.
    """
    config = tmp_path / "input.yaml"
    runnable = yaml.safe_dump({"steps": [{"step": 1, "engine": "orca", "operation": "sp"}]})
    _inject_model(chat_env, _saving_model(config, runnable))

    suspended = _post(client, "/api/agent/chat", {"message": "save it"}).get_json()
    assert suspended["wrote_config"] is None  # nothing has run yet — it is still asking
    assert [card["tool"] for card in suspended["pending"]] == ["save_config"]
    assert not config.exists()

    approvals = {card["id"]: True for card in suspended["pending"]}
    resumed = _post(client, "/api/agent/chat", {"approvals": approvals}).get_json()
    assert resumed["wrote_config"] == str(config)
    assert config.is_file()


def test_a_refused_or_unwritten_save_is_not_a_write(
    client: Any, chat_env: pytest.MonkeyPatch, tmp_path: Path
):
    """Two ways ``save_config`` runs and writes nothing, both of which must stay silent.

    Denied, it never executes. Handed an unrunnable draft it *returns normally* with
    ``written: False`` rather than raising — which is precisely the case that would have
    the editor adopt a file that was never saved.
    """
    config = tmp_path / "input.yaml"
    _inject_model(chat_env, _saving_model(config, yaml.safe_dump({"steps": [{"engine": "nope"}]})))

    suspended = _post(client, "/api/agent/chat", {"message": "save it"}).get_json()
    denied = _post(
        client, "/api/agent/chat", {"approvals": {c["id"]: False for c in suspended["pending"]}}
    ).get_json()
    assert denied["wrote_config"] is None
    assert not config.exists()

    _post(client, "/api/agent/chat", {"reset": True})
    again = _post(client, "/api/agent/chat", {"message": "save it"}).get_json()
    allowed = _post(
        client, "/api/agent/chat", {"approvals": {c["id"]: True for c in again["pending"]}}
    ).get_json()
    assert allowed["wrote_config"] is None  # it ran, validation refused, nothing written
    assert not config.exists()


def test_the_agent_is_told_about_the_file_the_builder_has_open(
    client: Any, chat_env: pytest.MonkeyPatch, tmp_path: Path
):
    """The GUI's current file outranks the launch argument.

    ``create_app`` captured a ``config_path`` once; a user who saved somewhere else since
    was still introducing the agent to the file the editor had left behind.
    """
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    launched = tmp_path / "launched.yaml"
    launched.write_text("steps: []\n", encoding="utf-8")
    seen: list[str] = []

    def spy(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.instructions or "")
        return ModelResponse(parts=[TextPart("ok")])

    app = create_app(token=TOKEN, config_path=launched).test_client()
    _inject_model(chat_env, FunctionModel(spy))

    _post(app, "/api/agent/chat", {"message": "hi"})
    assert str(launched) in seen[-1]  # with nothing else open, the launch path stands

    _post(app, "/api/agent/chat", {"message": "hi", "config_path": str(tmp_path / "other.yaml")})
    assert str(tmp_path / "other.yaml") in seen[-1]
    assert str(launched) not in seen[-1]

    # A page with nothing saved yet sends nothing, and must not end up with a config_path
    # of "" — `or`, not `is None`, is what makes the fallback survive an empty string.
    _post(app, "/api/agent/chat", {"message": "hi", "config_path": ""})
    assert str(launched) in seen[-1]


def test_structure_serves_extended_xyz_for_the_viewer(client: Any, tmp_path: Path):
    """The Structure pane's data source, as query arguments rather than a path segment.

    ``test_every_route_is_behind_the_gate`` refuses a parameterized rule, because one
    cannot be probed for the token gate by enumeration — so ``/api/structure/<step>``
    would fail the suite outright.
    """
    from ase import Atoms

    from chemrefine import cache as cache_mod
    from chemrefine.config import load_config
    from chemrefine.state import StepResults, Structure

    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump({"steps": [{"step": 1, "engine": "fake"}]}), "utf-8")
    loaded = load_config(config)
    step_cfg = loaded.steps[0]
    cache_mod.save(
        step_cfg=step_cfg,
        key=cache_mod.StepKey(parent_ids=(), fingerprint="f"),
        results=StepResults(
            structures=(
                Structure(
                    id="0",
                    atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
                    energy_hartree=-1.0,
                ),
            )
        ),
        step_dir=loaded.step_dir(step_cfg),
        chemrefine_version="test",
    )

    served = _get(client, f"/api/structure?config_path={config}&step=1")
    assert served.status_code == 200
    body = served.get_json()
    assert body["format"] == "extxyz"
    assert body["structure_id"] == "0"
    assert body["text"].splitlines()[0] == "2"

    # A library refusal keeps the documented shape, like every other endpoint here.
    missing = _get(client, f"/api/structure?config_path={config}&step=9")
    assert missing.status_code == 400
    assert missing.get_json()["exit_code"] == 2


def test_structure_list_offers_what_the_step_holds(client: Any, tmp_path: Path):
    """What fills the pane's two combo boxes, over the wire.

    Same query-argument shape and the same seeds sentinel as ``/api/structure``: no
    ``step`` at all asks for the input seeds, because a step may be *named* anything and
    ``step=input`` would be read as a name.
    """
    from ase import Atoms

    from chemrefine import cache as cache_mod
    from chemrefine.config import load_config
    from chemrefine.state import StepResults, Structure

    (tmp_path / "seeds.xyz").write_text("1\na\nN 0 0 0\n", encoding="utf-8")
    config = tmp_path / "input.yaml"
    config.write_text(
        yaml.safe_dump({"input": "seeds.xyz", "steps": [{"step": 1, "engine": "fake"}]}), "utf-8"
    )
    loaded = load_config(config)
    step_cfg = loaded.steps[0]
    cache_mod.save(
        step_cfg=step_cfg,
        key=cache_mod.StepKey(parent_ids=(), fingerprint="f"),
        results=StepResults(
            structures=(
                Structure(
                    id="0",
                    atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
                    energy_hartree=-1.0,
                    imaginary_freqs={6: -512.4},
                    frequencies={6: -512.4, 7: 1103.7},
                ),
            )
        ),
        step_dir=loaded.step_dir(step_cfg),
        chemrefine_version="test",
    )

    served = _get(client, f"/api/structure-list?config_path={config}&step=1")
    assert served.status_code == 200
    [row] = served.get_json()["structures"]
    assert row["id"] == "0"
    assert row["modes"] == {"6": -512.4, "7": 1103.7}
    assert row["imaginary"] == [6]

    seeds = _get(client, f"/api/structure-list?config_path={config}")
    assert seeds.status_code == 200
    assert seeds.get_json() == {
        "step": None,
        "structures": [{"id": "0", "modes": {}, "imaginary": []}],
    }

    # And a library refusal keeps the documented shape, like every other endpoint here.
    missing = _get(client, f"/api/structure-list?config_path={config}&step=9")
    assert missing.status_code == 400
    assert missing.get_json()["exit_code"] == 2


def test_a_structure_file_opens_on_its_own_with_no_workflow_at_all(client: Any, tmp_path: Path):
    """The second door into the Structure pane, and it must stay a separate one.

    ``/api/load`` opens a *workflow* and brings its tree; this answers "what is in this
    file". Two ways in, because a file reaches the page two ways: a path on the machine the
    server runs on, and the contents of a file dropped from the machine the browser runs on
    — over a forwarded port those are different computers, and a browser hands over a
    basename and bytes, never a path.
    """
    from ase.build import bulk

    bulk("Si", "diamond", a=5.43).write(tmp_path / "POSCAR", format="vasp")
    poscar = (tmp_path / "POSCAR").read_text(encoding="utf-8")

    by_path = _post(client, "/api/structure-file", {"path": str(tmp_path / "POSCAR")})
    assert by_path.status_code == 200
    assert by_path.get_json()["formula"] == "Si2"
    # A periodic file is the first thing in ChemRefine that carries a cell at all, and the
    # viewer draws its box from exactly this.
    assert by_path.get_json()["periodic"] is True
    assert 'Lattice="' in by_path.get_json()["text"]

    dropped = _post(client, "/api/structure-file", {"name": "POSCAR", "text": poscar})
    assert dropped.status_code == 200
    assert dropped.get_json()["formula"] == "Si2"
    # Named as the user knows it, not as the temporary file it was staged at.
    assert dropped.get_json()["path"] == "POSCAR"

    # Neither is a 400 that says what to send, not a traceback.
    assert _post(client, "/api/structure-file", {}).status_code == 400
    assert "path" in _post(client, "/api/structure-file", {}).get_json()["error"]
    # And a library refusal keeps the documented shape, like every other endpoint here.
    gone = _post(client, "/api/structure-file", {"path": str(tmp_path / "nope.xyz")})
    assert gone.status_code == 400
    assert gone.get_json()["exit_code"] == 2


@pytest.mark.parametrize("mode", ["7a", "two", "1.5", " "])
def test_a_mode_number_that_is_not_one_is_a_400(client: Any, tmp_path: Path, mode: str):
    """The mode box is free text, and a bare ``int()`` on it is a 500.

    The same class ``_step_key`` exists to close, on the sibling argument of the same
    call: every unusable input to this app is the documented 400, not a traceback.
    """
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump({"steps": [{"step": 1, "engine": "fake"}]}), "utf-8")
    response = _get(client, f"/api/structure?config_path={config}&step=1&mode_index={mode}")
    assert response.status_code == 400
    assert "not a whole number" in response.get_json()["error"]
    # A negative one is a number, and gets the library's own range refusal instead.
    negative = _get(client, f"/api/structure?config_path={config}&step=1&mode_index=-1")
    assert negative.status_code == 400


def test_load_refuses_a_home_it_cannot_resolve(client: Any):
    """``~nosuchuser/x`` raises ``RuntimeError`` — neither OSError nor UnicodeDecodeError.

    It is raised by ``expanduser()`` before the read guard, so it escaped both and became
    a 500.
    """
    response = _get(client, "/api/load?path=~nosuchuser1234/x.yaml")
    assert response.status_code == 400
    assert "cannot resolve" in response.get_json()["error"]


def test_a_model_with_nowhere_to_go_does_not_pass_the_preflight(
    client: Any, monkeypatch: pytest.MonkeyPatch
):
    """The gate promised a typo costs a click, not a turn. This case broke that promise.

    ``provider: openai`` with a bare model name and no key resolves to no endpoint at
    all — ``build_model`` hands the bare string to PydanticAI, which raises ``UserError``
    at construction. Reporting it "provider-native; not probed" armed Send for a
    configuration that could never build a model, and the user paid a turn to find out.
    A ``provider:model`` spelling *is* provider-native and still passes.
    """
    monkeypatch.delenv("CHEMREFINE_LLM_API_KEY", raising=False)

    stranded = _post(client, "/api/agent/check", {"provider": "openai", "model": "gpt-5-mini"})
    assert stranded.get_json()["ok"] is False
    assert "no endpoint to reach it" in stranded.get_json()["findings"][0]

    native = _post(client, "/api/agent/check", {"provider": "openai", "model": "openai:gpt-5-mini"})
    assert native.get_json()["ok"] is True
    assert "not probed" in native.get_json()["findings"][0]


def test_check_answers_a_verdict_never_an_error(client: Any, monkeypatch: pytest.MonkeyPatch):
    """An unreachable endpoint is this endpoint's *answer*, not its failure.

    Like ``/api/agent/availability`` beside it, the preflight always answers 200 — a
    ``ChemRefineError`` from resolution is caught here rather than becoming ``surface``'s
    400, because "you have not named a model yet" is a finding to render in the panel,
    not a failed request.
    """
    monkeypatch.delenv("CHEMREFINE_LLM_MODEL", raising=False)

    unconfigured = _post(client, "/api/agent/check", {})
    assert unconfigured.status_code == 200
    assert unconfigured.get_json()["ok"] is False
    assert "no model configured" in unconfigured.get_json()["findings"][0]

    # Nothing is listening on the ollama port in a test environment, which is exactly the
    # case the panel needs rendered: a verdict plus the one-command fix.
    unreachable = _post(client, "/api/agent/check", {"provider": "ollama", "model": "qwen3"})
    assert unreachable.status_code == 200
    body = unreachable.get_json()
    assert body["ok"] is False
    assert "unreachable" in body["findings"][0]
    assert any("ollama serve" in f for f in body["findings"])

    # A provider-native string has nothing to aim at and says so, still usable.
    native = _post(client, "/api/agent/check", {"provider": "openai", "model": "openai:gpt-5-mini"})
    assert native.get_json()["ok"] is True
    assert "not probed" in native.get_json()["findings"][0]


@pytest.mark.parametrize(
    "key", ["sk-good\r\nX-Injected: 1", "sk-good\nX: 1", "sk\x00", "", "k" * 1025, 17]
)
def test_a_key_that_cannot_be_a_header_is_refused_before_it_becomes_one(client: Any, key: Any):
    """The key leaves as ``Authorization: Bearer …``; a newline in it is header injection.

    Refused up front rather than left to fail inside the probe: ``urllib`` raises
    ``ValueError`` on a CR/LF header, and ``providers.check`` catches ``ValueError`` to
    mean "the reply was not a model listing" — so a fault in the box the user just typed
    into would have been reported as a fault of the endpoint being probed.
    """
    response = _post(
        client,
        "/api/agent/check",
        {
            "provider": "custom",
            "model": "m",
            "base_url": "https://example.invalid/v1",
            "api_key": key,
        },
    )
    assert response.status_code == 200
    assert response.get_json() == {
        "ok": False,
        "findings": ["the API key contains invalid characters"],
    }


def test_the_panels_key_reaches_resolution_and_never_comes_back(
    client: Any, chat_env: pytest.MonkeyPatch
):
    """The key is an input to the request and appears in no response body.

    Asserted at ``resolve``, not at the model: ``_inject_model`` replaces ``build_model``,
    so the model never sees it — what matters is that the panel's key outranks the
    server's environment, and that nothing echoes it back to the page.
    """
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from chemrefine.agent.providers import ProviderConfig

    chat_env.setenv("CHEMREFINE_LLM_API_KEY", "from-the-environment")
    seen: list[str | None] = []
    real = ProviderConfig.resolve

    def spy(provider: str = "custom", **kwargs: Any) -> ProviderConfig:
        resolved = real(provider, **kwargs)
        seen.append(resolved.api_key)
        return resolved

    chat_env.setattr(ProviderConfig, "resolve", spy)
    _inject_model(chat_env, FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("ok")])))

    typed = _post(client, "/api/agent/chat", {"message": "hi", "api_key": "typed-in-the-panel"})
    assert seen == ["typed-in-the-panel"]  # the panel outranks the environment
    assert "typed-in-the-panel" not in typed.get_data(as_text=True)

    _post(client, "/api/agent/chat", {"message": "hi", "reset": True})
    _post(client, "/api/agent/chat", {"message": "hi"})
    assert seen[-1] == "from-the-environment"  # a blank box falls through, as documented


def test_an_unplaceable_model_is_a_502_not_a_traceback(client: Any, chat_env: pytest.MonkeyPatch):
    """Building the agent is an outside-world failure like running it.

    ``Agent("not-a-real-model")`` raises ``UserError`` at *construction*, and that is
    neither an ``HTTPException`` nor a ``ChemRefineError`` — so while the call sat above
    the try, a typo in the model box produced a 500 and a logged traceback instead of the
    panel's flash.
    """
    failed = _post(client, "/api/agent/chat", {"model": "not-a-real-model", "message": "hi"})
    assert failed.status_code == 502
    assert "Unknown model" in failed.get_json()["error"]


def test_chat_turns_thread_history(client: Any, chat_env: pytest.MonkeyPatch):
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    seen: list[int] = []

    def script(messages: Any, info: Any) -> ModelResponse:
        seen.append(len(messages))
        return ModelResponse(parts=[TextPart("reply " + str(len(seen)))])

    _inject_model(chat_env, FunctionModel(script))
    first = _post(client, "/api/agent/chat", {"message": "hello"}).get_json()
    # A subset, not an exact dict: the response grows keys (a write signal, and whatever
    # comes next), and an equality assertion here fails for every one of them while
    # testing nothing about the history threading this case is named for.
    assert first["reply"] == "reply 1"
    assert first["pending"] is None
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


def test_overlapping_chat_turns_contend_on_a_lock_not_on_the_state(
    client: Any, chat_env: pytest.MonkeyPatch
):
    """A second Send during a turn hears "busy" — it must not run from the same history.

    The chat state is one conversation for one user, but waitress serves on four threads
    and a turn is a read-modify-write around a model call that takes seconds to a minute.
    Unsynchronised, two overlapping POSTs both read the same history, both run turns from
    it, and the last writer wins: a turn's messages silently vanish, or a ``reset``
    lands mid-turn and the finishing turn resurrects the conversation it was told to
    forget. Non-blocking with a 409 rather than queueing, so a Send during a wedged turn
    answers now instead of hanging behind it — and the lock's release is proven by the
    follow-up request going through normally.
    """
    import threading

    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    entered = threading.Event()
    release = threading.Event()

    def slow(messages: Any, info: Any) -> ModelResponse:
        entered.set()
        assert release.wait(timeout=10), "the test never released the model"
        return ModelResponse(parts=[TextPart("slow reply")])

    _inject_model(chat_env, FunctionModel(slow))
    first: dict[str, Any] = {}

    def send_first() -> None:
        first["response"] = _post(client, "/api/agent/chat", {"message": "one"})

    turn = threading.Thread(target=send_first)
    turn.start()
    try:
        assert entered.wait(timeout=10), "the first turn never reached the model"
        contended = _post(client.application.test_client(), "/api/agent/chat", {"message": "two"})
        assert contended.status_code == 409
        assert "already running" in contended.get_json()["error"]
    finally:
        release.set()
        turn.join(timeout=10)
    assert first["response"].get_json()["reply"] == "slow reply"
    assert _post(client, "/api/agent/chat", {"message": "three"}).status_code == 200


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
