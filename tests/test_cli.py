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
from chemrefine.cli import app
from chemrefine.cli_legacy import translate_argv as _translate_legacy_argv
from chemrefine.errors import ConfigError

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
    assert (tmp_path / "outputs" / "step1_screen" / "_cache" / "step.json").is_file()
    assert (tmp_path / "outputs" / "step2_refine" / "_cache" / "step.json").is_file()


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


def _assert_rebuilds_step_2(tmp_path: Path, config_path: Path, target: str) -> None:
    """``rebuild-cache TARGET`` re-parses step 2, so its cache document is rewritten."""
    document = tmp_path / "outputs" / "step2_refine" / "_cache" / "step.json"
    before = document.stat().st_mtime_ns
    result = runner.invoke(app, ["rebuild-cache", str(config_path), target])
    assert result.exit_code == 0, result.output
    assert document.stat().st_mtime_ns != before, f"step 2's cache was not rewritten for {target!r}"


def test_rebuild_cache_targets_by_step_number(tmp_path: Path):
    config_path = _write_config(tmp_path)
    runner.invoke(app, ["run", str(config_path)])
    _assert_rebuilds_step_2(tmp_path, config_path, "2")


def test_rebuild_cache_targets_by_step_name(tmp_path: Path):
    """A name reaches the same step its number does."""
    config_path = _write_config(tmp_path)
    runner.invoke(app, ["run", str(config_path)])
    _assert_rebuilds_step_2(tmp_path, config_path, "refine")


def test_rerun_with_missing_target_errors(tmp_path: Path):
    config_path = _write_config(tmp_path)
    runner.invoke(app, ["run", str(config_path)])
    result = runner.invoke(app, ["rerun", str(config_path), "ghost"])
    assert result.exit_code != 0


def test_rebuild_nms_is_rerun_under_another_name(tmp_path: Path):
    """What ``rebuild-nms`` is, asserted on what it does.

    It carries no NMS-specific path and does not look for an NMS step: it redoes the step it
    is given, the last one when it is given none. Asserting only its exit code let it pass on
    a config with no NMS step at all — which is this one — and would go on passing if the
    command became a no-op. Redoing a step archives the previous attempt, so that is the
    evidence it ran.
    """
    config_path = _write_config(tmp_path)
    runner.invoke(app, ["run", str(config_path)])
    last_step = tmp_path / "outputs" / "step2_refine"
    assert not list(last_step.glob("*/attempt*")), "nothing archived by the first run"

    result = runner.invoke(app, ["rebuild-nms", str(config_path)])

    assert result.exit_code == 0, result.output
    assert list(last_step.glob("*/attempt1")), (
        "the last step was redone — with no target, that is the step rebuild-nms takes"
    )
    assert not list((tmp_path / "outputs" / "step1_screen").glob("*/attempt*")), (
        "and only that step; the earlier one resumed from its cache"
    )


# ---------------------------------------------------------------------------
# --maxcores override
# ---------------------------------------------------------------------------


def test_maxcores_overrides_yaml_value(tmp_path: Path):
    config_path = _write_config(tmp_path, max_cores=8)
    # Override to 1 via flag; pipeline still completes with the fake engine.
    result = runner.invoke(app, ["run", str(config_path), "--maxcores", "1"])
    assert result.exit_code == 0


def test_maxgpus_overrides_yaml_value(tmp_path: Path):
    """--maxgpus beats the YAML; dry-run echoes the resolved value."""
    config_path = _write_config(tmp_path)
    result = runner.invoke(app, ["run", str(config_path), "--maxgpus", "3", "--dry-run"])
    assert result.exit_code == 0
    assert "max_gpus=3" in result.stdout


def test_maxgpus_omitted_shows_auto(tmp_path: Path):
    """Without --maxgpus the YAML default (None) resolves to 'auto' in the dry-run."""
    config_path = _write_config(tmp_path)
    result = runner.invoke(app, ["run", str(config_path), "--dry-run"])
    assert result.exit_code == 0
    assert "max_gpus=auto" in result.stdout


def test_maxcores_zero_is_rejected_before_anything_runs(tmp_path: Path):
    """The override is applied via ``model_copy`` (no re-validation), so the
    flag itself must enforce the ``>= 1`` floor — otherwise an invalid budget
    surfaces only mid-run as a raw Throttler ValueError, after inputs exist."""
    config_path = _write_config(tmp_path)
    result = runner.invoke(app, ["run", str(config_path), "--maxcores", "0"])
    assert result.exit_code != 0
    assert not (tmp_path / "outputs").exists()


# ---------------------------------------------------------------------------
# Config validation errors flow through to non-zero exit
# ---------------------------------------------------------------------------


def test_invalid_yaml_returns_config_exit_code(tmp_path: Path):
    bad = tmp_path / "input.yaml"
    bad.write_text(textwrap.dedent("steps: [\n"), encoding="utf-8")
    result = runner.invoke(app, ["run", str(bad)])
    # The exit-code contract in chemrefine.errors: ConfigError → 2.
    assert result.exit_code == 2


def test_invalid_config_does_not_traceback(tmp_path: Path):
    """A malformed config must exit cleanly (logged error), never a traceback."""
    bad = tmp_path / "input.yaml"
    bad.write_text("steps: []\n", encoding="utf-8")
    result = runner.invoke(app, ["resume", str(bad)])
    assert result.exit_code == 2
    assert "Traceback" not in (result.stdout + str(result.stderr_bytes or b""))


def test_unknown_top_level_key_returns_config_exit_code(tmp_path: Path):
    config_path = _write_config(tmp_path, orca_excutable="orca")  # typo
    result = runner.invoke(app, ["run", str(config_path)])
    assert result.exit_code == 2


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


# ---------------------------------------------------------------------------
# The exit-code contract
#
# `chemrefine.errors` promises every exception carries an `exit_code` the CLI
# maps to a deterministic process exit status. That only holds if the failure
# actually raises a ChemRefineError: a missing template would otherwise raise a bare
# FileNotFoundError, so the likeliest first-run error greeted the user with a
# traceback and exit 1 instead of the documented code.
# ---------------------------------------------------------------------------


def _minimal_project(tmp_path: Path, *, engine: str = "orca") -> Path:
    """A config that validates but whose templates directory is empty."""
    (tmp_path / "templates").mkdir()
    (tmp_path / "seed.xyz").write_text("1\n\nH 0.0 0.0 0.0\n", encoding="utf-8")
    config = tmp_path / "input.yaml"
    config.write_text(
        "input: ./seed.xyz\n"
        "output_dir: ./out\n"
        "template_dir: ./templates\n"
        "dispatch: local\n"
        "steps:\n"
        f"  - step: 1\n    engine: {engine}\n    operation: opt_sp\n",
        encoding="utf-8",
    )
    return config


def test_missing_template_exits_with_the_config_error_code(tmp_path: Path):
    """A missing `step1.inp` is a config error (exit 2), not an uncaught traceback."""
    result = CliRunner().invoke(app, ["run", str(_minimal_project(tmp_path))])

    assert result.exit_code == ConfigError.exit_code
    assert not isinstance(result.exception, FileNotFoundError)


def test_missing_slurm_header_exits_with_the_config_error_code(tmp_path: Path):
    """A missing SLURM header is a config error too -- the other first-run stumble."""
    config = _minimal_project(tmp_path)
    (tmp_path / "templates" / "step1.inp").write_text("! SP\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["run", str(config)])

    assert result.exit_code == ConfigError.exit_code


def test_malformed_config_exits_with_the_config_error_code(tmp_path: Path):
    """The already-working case, pinned so the contract is asserted end to end."""
    config = tmp_path / "bad.yaml"
    config.write_text("steps: []\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["run", str(config)])

    assert result.exit_code == ConfigError.exit_code


# --- cli: malformed legacy argv passes through ------------------------------


def test_translate_legacy_argv_passes_through_on_argparse_error():
    from chemrefine.cli_legacy import translate_argv as _translate_legacy_argv

    argv = ["c.yaml", "--maxcores", "not-an-int"]  # argparse SystemExit
    assert _translate_legacy_argv(argv) == argv
