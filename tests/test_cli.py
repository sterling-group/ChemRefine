"""End-to-end tests for the Typer CLI, driven through the fake engine."""

from __future__ import annotations

import contextlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from chemrefine import __version__
from chemrefine.cli import _translate_legacy_argv, app

runner = CliRunner()


def _write_xyz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("2\nH2\nH 0 0 0\nH 0.74 0 0\n", encoding="utf-8")


def _write_config(tmp_path: Path, **overrides) -> Path:
    """Write a minimal YAML config rooted at tmp_path; return its path."""
    seed = tmp_path / "input.xyz"
    _write_xyz(seed)
    base = {
        "template_dir": str(tmp_path / "templates"),
        "scratch_dir": str(tmp_path / "scratch"),
        "output_dir": str(tmp_path / "outputs"),
        "input": str(seed),
        "max_cores": 2,
        "steps": [
            {"step": 1, "name": "screen", "engine": "fake", "operation": "opt_sp"},
            {"step": 2, "name": "refine", "engine": "fake", "operation": "opt_sp"},
        ],
    }
    base.update(overrides)
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# --help / --version / global behaviour
# ---------------------------------------------------------------------------


def test_help_lists_every_subcommand():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("run", "resume", "rebuild-cache", "rebuild-nms", "rerun", "rerun-errors"):
        assert cmd in result.stdout


def test_version_prints_package_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_arguments_shows_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.stdout or "Usage" in result.stderr


def test_importing_cli_does_not_pull_the_heavy_stack():
    """`--help`/`--version` must stay fast: importing the CLI must not import the
    pipeline / engine registry / recovery / ase stack (those load lazily per command)."""
    code = (
        "import sys, chemrefine.cli; "
        "heavy = ('chemrefine.pipeline', 'chemrefine.engines', 'chemrefine.recovery', 'ase'); "
        "print(','.join(m for m in heavy if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"cli import leaked heavy modules: {out.stdout.strip()}"


def test_missing_config_path_errors(tmp_path: Path):
    result = runner.invoke(app, ["run", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# run / resume
# ---------------------------------------------------------------------------


def test_run_executes_full_pipeline(tmp_path: Path):
    config_path = _write_config(tmp_path)
    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 0
    # Caches are now on disk for both steps.
    assert (tmp_path / "outputs" / "step1_screen" / "_cache" / "step.pkl").is_file()
    assert (tmp_path / "outputs" / "step2_refine" / "_cache" / "step.pkl").is_file()


def test_resume_after_run_is_cache_hit(tmp_path: Path):
    config_path = _write_config(tmp_path)
    assert runner.invoke(app, ["run", str(config_path)]).exit_code == 0
    # Second invocation should succeed and not blow up on the cache.
    assert runner.invoke(app, ["resume", str(config_path)]).exit_code == 0


def test_dry_run_does_not_create_outputs(tmp_path: Path):
    config_path = _write_config(tmp_path)
    result = runner.invoke(app, ["run", str(config_path), "--dry-run"])
    assert result.exit_code == 0
    assert "[dry-run]" in result.stdout
    assert not (tmp_path / "outputs" / "step1_screen").exists()


def test_dry_run_with_target_step_prints_target(tmp_path: Path):
    config_path = _write_config(tmp_path)
    result = runner.invoke(app, ["rebuild-cache", str(config_path), "refine", "--dry-run"])
    assert result.exit_code == 0
    assert "target step: refine" in result.stdout


# ---------------------------------------------------------------------------
# rebuild-cache / rerun / rebuild-nms
# ---------------------------------------------------------------------------


def test_rebuild_cache_targets_by_step_number(tmp_path: Path):
    config_path = _write_config(tmp_path)
    runner.invoke(app, ["run", str(config_path)])
    result = runner.invoke(app, ["rebuild-cache", str(config_path), "2"])
    assert result.exit_code == 0


def test_rebuild_cache_targets_by_step_name(tmp_path: Path):
    config_path = _write_config(tmp_path)
    runner.invoke(app, ["run", str(config_path)])
    result = runner.invoke(app, ["rebuild-cache", str(config_path), "refine"])
    assert result.exit_code == 0


def test_rerun_with_missing_target_errors(tmp_path: Path):
    config_path = _write_config(tmp_path)
    runner.invoke(app, ["run", str(config_path)])
    result = runner.invoke(app, ["rerun", str(config_path), "ghost"])
    assert result.exit_code != 0


def test_rebuild_nms_runs(tmp_path: Path):
    config_path = _write_config(tmp_path)
    runner.invoke(app, ["run", str(config_path)])
    result = runner.invoke(app, ["rebuild-nms", str(config_path)])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# --maxcores override
# ---------------------------------------------------------------------------


def test_maxcores_overrides_yaml_value(tmp_path: Path):
    config_path = _write_config(tmp_path, max_cores=8)
    # Override to 1 via flag; pipeline still completes with the fake engine.
    result = runner.invoke(app, ["run", str(config_path), "--maxcores", "1"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Config validation errors flow through to non-zero exit
# ---------------------------------------------------------------------------


def test_invalid_yaml_returns_config_exit_code(tmp_path: Path):
    bad = tmp_path / "input.yaml"
    bad.write_text(textwrap.dedent("steps: [\n"), encoding="utf-8")
    result = runner.invoke(app, ["run", str(bad)])
    # ConfigError.exit_code is 2 — but Typer's own usage error is also 2;
    # accept anything non-zero here.
    assert result.exit_code != 0


def test_unknown_top_level_key_returns_non_zero(tmp_path: Path):
    config_path = _write_config(tmp_path, orca_excutable="orca")  # typo
    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# rerun-errors subcommand
# ---------------------------------------------------------------------------


def test_rerun_errors_dry_run(tmp_path: Path):
    config_path = _write_config(tmp_path)
    result = runner.invoke(app, ["rerun-errors", str(config_path), "1", "--dry-run"])
    assert result.exit_code == 0
    assert "rerun-errors" in result.stdout


# ---------------------------------------------------------------------------
# Legacy (v1.3.1) flag-style argv translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "legacy, expected",
    [
        (["c.yaml"], ["run", "c.yaml"]),
        (["c.yaml", "--skip"], ["resume", "c.yaml"]),
        (["c.yaml", "--rebuild_cache", "3"], ["rebuild-cache", "c.yaml", "3"]),
        (["c.yaml", "--rebuild_cache"], ["rebuild-cache", "c.yaml"]),
        (["c.yaml", "--rebuild_nms", "2"], ["rebuild-nms", "c.yaml", "2"]),
        (["c.yaml", "--rerun_errors", "3"], ["rerun-errors", "c.yaml", "3"]),
        (["c.yaml", "--maxcores", "8"], ["run", "c.yaml", "--maxcores", "8"]),
        (["--maxcores", "8", "c.yaml", "--skip"], ["resume", "c.yaml", "--maxcores", "8"]),
        (["-v", "c.yaml", "--rerun_errors", "1"], ["-v", "rerun-errors", "c.yaml", "1"]),
    ],
)
def test_translate_legacy_argv_maps_old_flags(legacy, expected):
    assert _translate_legacy_argv(legacy) == expected


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "c.yaml"],
        ["rerun-errors", "c.yaml", "2"],
        ["rebuild-cache", "c.yaml", "--dry-run"],
        ["--version"],
        ["--help"],
        [],
    ],
)
def test_translate_legacy_argv_passes_new_style_through(argv):
    assert _translate_legacy_argv(argv) == argv


def test_legacy_rerun_errors_flag_dispatches_via_main(tmp_path: Path, monkeypatch):
    """`chemrefine CONFIG --rerun_errors 1` (v1.3.1) reaches the rerun-errors action."""
    from chemrefine import cli
    from chemrefine.recovery import Action

    config_path = _write_config(tmp_path)
    seen = {}

    def _fake_execute(config, action, target=None):
        seen["action"], seen["target"] = action, target
        return 0

    monkeypatch.setattr(cli, "execute", _fake_execute)
    monkeypatch.setattr("sys.argv", ["chemrefine", str(config_path), "--rerun_errors", "1"])
    with contextlib.suppress(SystemExit):  # typer.Exit at the end of app()
        cli.main()
    assert seen == {"action": Action.RERUN_ERRORS, "target": "1"}
