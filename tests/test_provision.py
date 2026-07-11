"""Tests for the managed backend-env provisioner (``engines/_provision.py``) + launch seam.

Everything is faked — no real ``uv`` / ``conda`` subprocess runs, no real backend imports.
The requirement DTOs come from the engines' own ``backend_requirement`` (Protocol capability),
so these tests also pin the end-to-end wiring: managed env → server command / script command.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines import _provision as provision
from chemrefine.engines import preflight_backends
from chemrefine.engines.api import BackendRequirement, get_engine
from chemrefine.errors import ConfigError
from chemrefine.state import PipelineState, StepContext, Structure

_REQ = BackendRequirement(extra="mlip-mace", import_name="mace")


def _provisioned(tmp_path: Path, extra: str) -> Path:
    """Create a fake managed env for ``extra`` under ``tmp_path`` and return its python."""
    py = tmp_path / "backends" / extra / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("", encoding="utf-8")
    return py


def _ctx(tmp_path: Path, *, engine: str, options: dict | None) -> StepContext:
    """A minimal real ``StepContext`` for launch-seam tests."""
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    return StepContext(
        step_cfg=StepConfig(step=1, engine=engine, operation="opt_sp", options=options or {}),
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=tmp_path / "templates",
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(structures=(seed,)),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        executables={},
    )


# ---------------------------------------------------------------------------
# chemrefine_home / backend_env_path
# ---------------------------------------------------------------------------


def test_chemrefine_home_explicit_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path / "crh"))
    assert provision.chemrefine_home() == tmp_path / "crh"


def test_chemrefine_home_alongside_writable_prefix(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("CHEMREFINE_HOME", raising=False)
    monkeypatch.setattr(provision.sys, "prefix", str(tmp_path))
    assert provision.chemrefine_home() == tmp_path / "share" / "chemrefine"


def test_chemrefine_home_falls_back_to_user_home(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("CHEMREFINE_HOME", raising=False)
    monkeypatch.setattr(provision.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(provision.os, "access", lambda _p, _m: False)
    monkeypatch.setattr(provision.Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    assert provision.chemrefine_home() == tmp_path / "home" / ".chemrefine"


def test_backend_env_path(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    assert provision.backend_env_path("pyscf") == tmp_path / "backends" / "pyscf"


# ---------------------------------------------------------------------------
# resolve_launcher / require_backend
# ---------------------------------------------------------------------------


def test_resolve_launcher_override_wins(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    _provisioned(tmp_path, _REQ.extra)  # even with a managed env present
    assert provision.resolve_launcher(_REQ, "/custom/python") == "/custom/python"


def test_resolve_launcher_managed_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    py = _provisioned(tmp_path, _REQ.extra)
    assert provision.resolve_launcher(_REQ) == str(py)


def test_resolve_launcher_falls_back_to_python(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    assert provision.resolve_launcher(_REQ) == "python"


def test_require_backend_override_ok(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    provision.require_backend(_REQ, "/custom/python")  # no raise


def test_require_backend_managed_ok(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    _provisioned(tmp_path, _REQ.extra)
    provision.require_backend(_REQ)  # no raise


def test_require_backend_importable_ok(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", lambda _n: object())
    provision.require_backend(_REQ)  # no raise


def test_require_backend_missing_raises_actionable(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", lambda _n: None)
    with pytest.raises(ConfigError, match="chemrefine backends install mlip-mace"):
        provision.require_backend(_REQ)


# ---------------------------------------------------------------------------
# preflight_backends — fail fast before any job submits
# ---------------------------------------------------------------------------


def test_preflight_skips_non_provisionable_engines(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", lambda _n: None)
    steps = [StepConfig(step=1, engine="fake", operation="opt_sp")]
    preflight_backends(steps)  # fake isn't provisionable → nothing to check


def test_preflight_raises_for_unavailable_backend(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", lambda _n: None)
    steps = [
        StepConfig(step=1, engine="fake", operation="opt_sp"),
        StepConfig(step=2, engine="mlip", operation="opt_sp", options={"task_name": "mace_off"}),
    ]
    with pytest.raises(ConfigError, match="mlip-mace"):
        preflight_backends(steps)


def test_preflight_passes_with_managed_envs(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision.importlib.util, "find_spec", lambda _n: None)
    _provisioned(tmp_path, "mlip-mace")
    _provisioned(tmp_path, "mlip-fairchem")
    steps = [
        StepConfig(step=1, engine="mlip", operation="opt_sp", options={"task": "mace_off"}),
        StepConfig(step=2, engine="mlip", operation="opt_sp", options={"task_name": "omol"}),
    ]
    preflight_backends(steps)  # both resolve → no raise


# ---------------------------------------------------------------------------
# detect_env_tool
# ---------------------------------------------------------------------------


def test_detect_env_tool_conda(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(provision.sys, "prefix", str(tmp_path))
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    assert provision.detect_env_tool() == "conda"


def test_detect_env_tool_uv(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(provision.sys, "prefix", str(tmp_path))
    (tmp_path / "pyvenv.cfg").write_text("home = /x\nuv = 0.5.0\n", encoding="utf-8")
    monkeypatch.setattr(provision.shutil, "which", lambda _n: "/usr/bin/uv")
    assert provision.detect_env_tool() == "uv"


def test_detect_env_tool_venv_without_uv_binary(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(provision.sys, "prefix", str(tmp_path))
    (tmp_path / "pyvenv.cfg").write_text("uv = 0.5.0\n", encoding="utf-8")
    monkeypatch.setattr(provision.shutil, "which", lambda _n: None)
    assert provision.detect_env_tool() == "venv"


def test_detect_env_tool_venv_without_uv_marker(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(provision.sys, "prefix", str(tmp_path))
    (tmp_path / "pyvenv.cfg").write_text("home = /x\n", encoding="utf-8")
    monkeypatch.setattr(provision.shutil, "which", lambda _n: "/usr/bin/uv")
    assert provision.detect_env_tool() == "venv"


# ---------------------------------------------------------------------------
# build_backend_env — mirrors the detected tool; idempotent
# ---------------------------------------------------------------------------


def test_build_commands_per_tool(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(provision, "_direct_url", lambda: None)  # index install
    env = tmp_path / "e"
    uv = provision._build_commands("uv", env, "mlip-mace")
    assert uv[0] == ["uv", "venv", str(env)]
    assert uv[1][:4] == ["uv", "pip", "install", "--python"]
    conda = provision._build_commands("conda", env, "pyscf")
    assert conda[0][:4] == ["conda", "create", "-y", "-p"]
    assert conda[0][-1].startswith("python=")
    venv = provision._build_commands("venv", env, "mlip-orb")
    assert venv[0][1:3] == ["-m", "venv"]
    # Every tool installs the extra pinned to the orchestrator's version.
    for cmds in (uv, conda, venv):
        assert cmds[1][-1].startswith("chemrefine[") and "==" in cmds[1][-1]


def test_build_backend_env_is_idempotent(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    py = _provisioned(tmp_path, "mlip-mace")
    calls: list[list[str]] = []
    monkeypatch.setattr(provision.subprocess, "run", lambda argv, **_k: calls.append(argv))
    assert provision.build_backend_env("mlip-mace") == py
    assert calls == []  # env already present → no subprocess


def test_build_backend_env_runs_the_detected_tool(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision, "_direct_url", lambda: None)
    monkeypatch.setattr(provision, "detect_env_tool", lambda: "venv")
    calls: list[list[str]] = []
    monkeypatch.setattr(provision.subprocess, "run", lambda argv, **_k: calls.append(argv))
    python = provision.build_backend_env("mlip-mace")
    assert python == tmp_path / "backends" / "mlip-mace" / "bin" / "python"
    assert len(calls) == 2 and calls[0][1:3] == ["-m", "venv"]


def test_build_backend_env_explicit_tool(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(provision, "_direct_url", lambda: None)
    calls: list[list[str]] = []
    monkeypatch.setattr(provision.subprocess, "run", lambda argv, **_k: calls.append(argv))
    provision.build_backend_env("pyscf", tool="uv")
    assert calls[0][0] == "uv"


# ---------------------------------------------------------------------------
# _install_target — PEP 610 source-matching
# ---------------------------------------------------------------------------


def test_install_target_index_install_pins_version(monkeypatch):
    monkeypatch.setattr(provision, "_direct_url", lambda: None)
    assert provision._install_target("pyscf") == f"chemrefine[pyscf]=={provision.__version__}"


def test_install_target_editable_local_dir(monkeypatch, tmp_path: Path):
    url = tmp_path.as_uri()
    direct = {"url": url, "dir_info": {"editable": True}}
    monkeypatch.setattr(provision, "_direct_url", lambda: direct)
    assert provision._install_target("mlip-mace") == f"chemrefine[mlip-mace] @ {url}"


def test_install_target_missing_source_dir_raises(monkeypatch, tmp_path: Path):
    url = (tmp_path / "gone").as_uri()
    direct = {"url": url, "dir_info": {"editable": True}}
    monkeypatch.setattr(provision, "_direct_url", lambda: direct)
    with pytest.raises(ConfigError, match="no longer exists"):
        provision._install_target("mlip-mace")


def test_install_target_git_install_pins_commit(monkeypatch):
    monkeypatch.setattr(
        provision,
        "_direct_url",
        lambda: {
            "url": "https://github.com/sterling-group/ChemRefine.git",
            "vcs_info": {"vcs": "git", "commit_id": "abc123"},
            "subdirectory": "pkg",
        },
    )
    assert provision._install_target("pyscf") == (
        "chemrefine[pyscf] @ git+https://github.com/sterling-group/ChemRefine.git"
        "@abc123#subdirectory=pkg"
    )


def test_install_target_git_install_without_ref(monkeypatch):
    direct = {"url": "https://example.com/repo.git", "vcs_info": {"vcs": "git"}}
    monkeypatch.setattr(provision, "_direct_url", lambda: direct)
    assert (
        provision._install_target("pyscf") == "chemrefine[pyscf] @ git+https://example.com/repo.git"
    )


def test_install_target_remote_archive_passes_url_through(monkeypatch):
    direct = {"url": "https://example.com/chemrefine.tar.gz", "archive_info": {}}
    monkeypatch.setattr(provision, "_direct_url", lambda: direct)
    assert provision._install_target("pyscf") == (
        "chemrefine[pyscf] @ https://example.com/chemrefine.tar.gz"
    )


def test_install_target_metadata_without_url_pins_version(monkeypatch):
    monkeypatch.setattr(provision, "_direct_url", lambda: {"dir_info": {}})
    assert provision._install_target("pyscf") == f"chemrefine[pyscf]=={provision.__version__}"


def test_direct_url_none_when_dist_missing(monkeypatch):
    def _raise(_name):
        raise provision.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(provision.importlib.metadata, "distribution", _raise)
    assert provision._direct_url() is None


def test_direct_url_none_without_metadata_file(monkeypatch):
    class _Dist:
        def read_text(self, _name):
            return None  # direct_url.json absent → index install

    monkeypatch.setattr(provision.importlib.metadata, "distribution", lambda _n: _Dist())
    assert provision._direct_url() is None


def _dist_with(monkeypatch, raw: str) -> None:
    class _Dist:
        def read_text(self, _name):
            return raw

    monkeypatch.setattr(provision.importlib.metadata, "distribution", lambda _n: _Dist())


def test_direct_url_parses_the_metadata(monkeypatch):
    _dist_with(monkeypatch, '{"url": "file:///src", "dir_info": {"editable": true}}')
    assert provision._direct_url() == {"url": "file:///src", "dir_info": {"editable": True}}


def test_direct_url_none_on_corrupt_metadata(monkeypatch):
    _dist_with(monkeypatch, "{not json")
    assert provision._direct_url() is None


def test_direct_url_none_on_non_dict_metadata(monkeypatch):
    _dist_with(monkeypatch, "[1, 2]")
    assert provision._direct_url() is None


# ---------------------------------------------------------------------------
# Launch seam — the resolved interpreter reaches the server + script commands
# ---------------------------------------------------------------------------


def test_extopt_server_cmd_uses_managed_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    py = _provisioned(tmp_path, "mlip-mace")
    ctx = _ctx(tmp_path, engine="mlip-extopt", options={"task_name": "mace_off"})
    cmd = get_engine("mlip-extopt")._server_cmd(ctx)
    assert cmd.startswith(f"{py} -m chemrefine.engines._backend_server.server")


def test_extopt_server_cmd_defaults_to_python(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    ctx = _ctx(tmp_path, engine="mlip-extopt", options={"task_name": "mace_off"})
    cmd = get_engine("mlip-extopt")._server_cmd(ctx)
    assert cmd.startswith("python -m chemrefine.engines._backend_server.server")


def test_script_run_block_uses_managed_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    py = _provisioned(tmp_path, "mlip-fairchem")
    ctx = _ctx(tmp_path, engine="mlip", options={"task_name": "omol"})
    block = get_engine("mlip").run_block(ctx, Path("step1_0.py"), Path("step1_0.json"))
    assert block.splitlines()[-1] == f"{py} step1_0.py"


def test_script_run_block_defaults_to_python(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    ctx = _ctx(tmp_path, engine="mlip", options=None)
    block = get_engine("mlip").run_block(ctx, Path("step1_0.py"), Path("step1_0.json"))
    assert block.splitlines()[-1] == "python step1_0.py"


def test_script_run_block_honours_backend_python_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    ctx = _ctx(tmp_path, engine="mlip", options={"backend_python": "/envs/x/bin/python"})
    block = get_engine("mlip").run_block(ctx, Path("step1_0.py"), Path("step1_0.json"))
    assert block.splitlines()[-1] == "/envs/x/bin/python step1_0.py"


def test_non_provisionable_script_engine_keeps_plain_python(monkeypatch, tmp_path: Path):
    """A third-party ScriptEngine without ``backend_requirement`` runs today's ``python``."""
    from chemrefine.engines._script import ScriptEngine

    class _PlainScript(ScriptEngine):
        name = "plain-script-test"
        label = "Plain"

    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    ctx = _ctx(tmp_path, engine="mlip", options={})
    block = _PlainScript().run_block(ctx, Path("step1_0.py"), Path("step1_0.json"))
    assert block.splitlines()[-1] == "python step1_0.py"


def test_non_provisionable_extopt_engine_keeps_plain_python(monkeypatch, tmp_path: Path):
    """A third-party ExtOpt engine without ``backend_requirement`` serves from ``python``."""
    from chemrefine.engines._options import EngineOptions
    from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator
    from chemrefine.engines.orca.extopt.engine import ExtOptOrcaEngine

    class _PlainExtOpt(ExtOptOrcaEngine):
        name = "plain-extopt-test"
        backend = "mlip"
        wrapper_filename = "plain.sh"
        options_cls = EngineOptions
        calculator_cls = MlipExtOptCalculator

    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    ctx = _ctx(tmp_path, engine="mlip-extopt", options={})
    cmd = _PlainExtOpt()._server_cmd(ctx)
    assert cmd.startswith("python -m chemrefine.engines._backend_server.server")


# ---------------------------------------------------------------------------
# known_backend_extras + the `chemrefine backends` CLI group
# ---------------------------------------------------------------------------


def test_known_backend_extras_is_registration_driven():
    """The union of every provisionable engine's declared extras — no hardcoded list."""
    from chemrefine.engines import known_backend_extras

    extras = known_backend_extras()
    assert {"mlip-fairchem", "mlip-mace", "mlip-sevenn", "mlip-orb", "mlip-chgnet", "pyscf"} <= (
        extras
    )


def test_backends_cli_list_and_path(monkeypatch, tmp_path: Path):
    from typer.testing import CliRunner

    from chemrefine.cli import app

    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    py = _provisioned(tmp_path, "pyscf")
    runner = CliRunner()

    result = runner.invoke(app, ["backends", "list"])
    assert result.exit_code == 0
    assert str(py) in result.output  # provisioned → shows the env python
    assert "not provisioned" in result.output  # the others aren't

    ok = runner.invoke(app, ["backends", "path", "pyscf"])
    assert ok.exit_code == 0 and ok.output.strip() == str(py)
    missing = runner.invoke(app, ["backends", "path", "mlip-mace"])
    assert missing.exit_code == 1


def test_backends_cli_install_validates_and_builds(monkeypatch, tmp_path: Path):
    from typer.testing import CliRunner

    import chemrefine.engines as engines_pkg
    from chemrefine.cli import app

    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    runner = CliRunner()

    bad = runner.invoke(app, ["backends", "install", "not-a-backend"])
    assert bad.exit_code != 0

    built: list[str] = []

    def _fake_build(extra: str):
        built.append(extra)
        return _provisioned(tmp_path, extra)

    monkeypatch.setattr(engines_pkg, "build_backend_env", _fake_build)
    ok = runner.invoke(app, ["backends", "install", "mlip-mace", "pyscf"])
    assert ok.exit_code == 0
    assert built == ["mlip-mace", "pyscf"]


def test_backends_cli_install_surfaces_chemrefine_errors(monkeypatch, tmp_path: Path):
    """A ConfigError from provisioning (e.g. vanished source tree) exits cleanly, no traceback."""
    from typer.testing import CliRunner

    import chemrefine.engines as engines_pkg
    from chemrefine.cli import app

    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))

    def _fail(extra: str):
        raise ConfigError("ChemRefine was installed from /gone, which no longer exists")

    monkeypatch.setattr(engines_pkg, "build_backend_env", _fail)
    result = CliRunner().invoke(app, ["backends", "install", "pyscf"])
    assert result.exit_code == ConfigError.exit_code
    assert "no longer exists" in result.output
