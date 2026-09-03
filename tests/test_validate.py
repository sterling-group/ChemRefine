"""The validation report: every failure class becomes a structured row, nothing raises.

The report is the GUI's and the agent's view of a config, so each test pins one failure
class to its row shape — the ``loc`` a GUI walks to highlight a field, the ``kind`` an
agent branches on — and the warning tests pin the silent no-ops a run would otherwise
swallow (undeclared option keys, ``nms: true`` on an engine that cannot NMS, templates
that do not exist yet). Where a check re-reads what the run reads (an options model, the
header dispatch would pick), the test drives the same input through both spellings.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from chemrefine.validate import ValidationReport, validate_config_file, validate_config_text


def _config_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "steps": [{"step": 1, "engine": "fake"}],
    }
    base.update(overrides)
    return base


def _validate(tmp_path: Path, **overrides: object) -> ValidationReport:
    """Validate a config dict as text, rooted at tmp_path (paths resolve against it)."""
    return validate_config_text(yaml.safe_dump(_config_dict(**overrides)), base_dir=tmp_path)


def _kinds(report: ValidationReport) -> list[str]:
    return [issue.kind for issue in report.issues]


# ---------------------------------------------------------------------------
# Failure classes → issue rows
# ---------------------------------------------------------------------------


def test_a_clean_config_is_ok_and_carries_the_model():
    report = validate_config_text(yaml.safe_dump(_config_dict()))
    assert report.ok
    assert report.issues == ()
    assert report.config is not None
    assert report.config.steps[0].engine == "fake"


def test_malformed_yaml_is_one_yaml_issue():
    report = validate_config_text("steps: [unclosed")
    assert not report.ok
    assert _kinds(report) == ["yaml"]
    assert report.config is None


def test_a_non_mapping_document_is_one_yaml_issue():
    report = validate_config_text("- just\n- a list\n")
    assert not report.ok
    assert _kinds(report) == ["yaml"]


def test_a_non_string_key_is_a_report_row_not_a_typeerror():
    """YAML 1.1 reads an unquoted ``on:`` key as a boolean — still a report, never a raise.

    ``Config(**raw)`` imposed a str-keys rule pydantic never saw, so this exact text
    escaped as ``TypeError: keywords must be strings`` — a 500 in the GUI and an
    unstructured error over MCP, from the one function whose contract is "never raises".
    """
    report = validate_config_text("on: true\nsteps: []\n")
    assert not report.ok
    assert any("string" in issue.message.lower() for issue in report.issues)


def test_model_errors_keep_pydantic_locs():
    """A GUI highlights the offending field by walking ``loc`` — it must survive."""
    report = validate_config_text(yaml.safe_dump(_config_dict(charge="not-an-int")))
    assert not report.ok
    assert any(issue.loc[0] == "charge" for issue in report.issues)
    assert report.config is None


def test_an_orca_step_under_a_whitespace_path_warns_and_does_not_block(tmp_path: Path):
    """ORCA cannot run from there, but the config is well-formed — so this warns, not fails.

    Held as a *warning* deliberately. Enforcing it as an issue made validity depend on
    where a project sits on disk: a relative `output_dir` inherits the config file's own
    directory, so an mlip-only workflow under `~/My Drive` was refused although it runs
    perfectly, and this repository's own example suite went red from any checkout path
    containing a space. The engine refuses at `prepare`; this is only the early word.
    """
    project = tmp_path / "my project"
    (project / "templates").mkdir(parents=True)
    (project / "templates" / "step1.inp").write_text("! HF\n", encoding="utf-8")
    (project / "templates" / "cpu.slurm.header").write_text("#!/bin/bash\n", encoding="utf-8")
    report = validate_config_text(
        yaml.safe_dump({"steps": [{"step": 1, "engine": "orca", "operation": "sp"}]}),
        base_dir=project,
    )
    assert report.ok, "a whitespace path must not make the config invalid"
    kinds = [w.kind for w in report.warnings]
    assert "whitespace-path" in kinds
    warning = next(w for w in report.warnings if w.kind == "whitespace-path")
    assert warning.loc == ("steps", 0, "engine")
    assert "whitespace-delimited" in warning.message


def test_an_engine_that_writes_no_orca_input_is_not_warned_about(tmp_path: Path):
    """The warning follows the capability, not the path — mlip under the same tree is fine.

    This is the half that matters: everything except ORCA's input file is whitespace-safe,
    so warning about all of them would be the same over-reach in a quieter form.
    """
    project = tmp_path / "my project"
    project.mkdir(parents=True)
    report = validate_config_text(
        yaml.safe_dump({"steps": [{"step": 1, "engine": "mlip"}]}), base_dir=project
    )
    assert [w.kind for w in report.warnings].count("whitespace-path") == 0


def test_a_metacharacter_from_the_config_directory_is_an_issue_not_a_warning(tmp_path: Path):
    """Engine-independent and dangerous, so it blocks — unlike the ORCA whitespace warning.

    Every job script exports template_dir/output_dir/scratch_dir, so a `$(...)` in any of
    them is a command substitution whatever engine runs. The report has to say so before a
    script is written, and `save_config` has to refuse to write such a config.
    """
    hostile = tmp_path / "$(echo pwned)"
    hostile.mkdir()
    report = _validate(hostile)
    assert not report.ok
    assert report.issues[0].kind == "shell-unsafe"
    assert report.config is None


def test_a_refused_legacy_spelling_is_one_legacy_issue():
    """The legacy normalizer raises before pydantic runs; the report catches that too."""
    report = validate_config_text(
        yaml.safe_dump(_config_dict(steps=[{"step": 1, "calculation_type": "OPT"}]))
    )
    assert not report.ok
    assert _kinds(report) == ["legacy"]
    assert "calculation_type" in report.issues[0].message


def test_an_unknown_engine_is_an_error_anchored_to_the_step(tmp_path: Path):
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "no-such-engine"}])
    assert not report.ok
    assert report.issues[0].kind == "engine"
    assert report.issues[0].loc == ("steps", 0, "engine")
    assert report.config is None


def test_a_bad_declared_option_value_is_an_error(tmp_path: Path):
    """Read through the engine's own model (lenient), so run and report agree."""
    report = _validate(
        tmp_path,
        steps=[{"step": 1, "engine": "qchem", "options": {"nprocs": "not-an-int"}}],
    )
    assert not report.ok
    assert report.issues[0].kind == "options"
    assert report.issues[0].loc == ("steps", 0, "options")


def test_an_engines_own_preflight_refusal_is_an_issue(tmp_path: Path):
    """The report makes the same refusals the run's t=0 walk makes — one hook, two readers.

    A training step without a device passed every pre-run pass before the
    ``PreflightChecking`` hook existed: the lenient options read fills the default,
    and the refusal lived only in ``prepare`` — reached after the steps computing the
    labels had run for days. ``kind: preflight`` is the row an agent branches on.
    """
    report = _validate(
        tmp_path,
        steps=[{"step": 1, "engine": "mlip-train", "options": {"task_name": "mace_off"}}],
    )
    assert not report.ok
    assert any(
        issue.kind == "preflight" and "must name a device" in issue.message
        for issue in report.issues
    )


def test_a_lenient_options_failure_is_not_double_reported(tmp_path: Path):
    """A bad declared value fails the lenient read; the preflight hook must not repeat it."""
    report = _validate(
        tmp_path,
        steps=[
            {
                "step": 1,
                "engine": "mlip-train",
                "options": {"task_name": "mace_off", "device": "not-a-device"},
            }
        ],
    )
    assert not report.ok
    assert [issue.kind for issue in report.issues] == ["options"]


def test_an_unknown_orca_operation_is_an_issue(tmp_path: Path):
    """A typo'd ``operation:`` must fail here, not after the step's jobs have run.

    Without the ORCA-family ``check_step`` this passed validation and the run's preflight,
    submitted and paid for every job, and was then refused per output by the parser — with
    the paid outputs unadoptable afterwards, since the operation is part of every row key.
    """
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "orca", "operation": "opt-sp"}])
    assert not report.ok
    assert any(issue.kind == "preflight" and "opt-sp" in issue.message for issue in report.issues)


def test_an_unknown_mlip_task_name_is_an_issue(tmp_path: Path):
    """The registry lookup the run's ``preflight_backends`` makes, made here too.

    Without it the report said ``ok: true`` for a config ``chemrefine run`` refuses one
    second in — the GUI's Validate button, the agent's ``validate_config`` and every MCP
    client all read this report and nothing else. Only the *registry* half is asked: the
    lookup imports no backend, so a config authored on a laptop for a cluster still
    validates without the cluster's environments.
    """
    report = _validate(
        tmp_path,
        steps=[{"step": 1, "engine": "mlip", "options": {"task_name": "mace_of"}}],
    )
    assert not report.ok
    assert report.issues[0].kind == "backend"
    assert report.issues[0].loc == ("steps", 0, "options")
    assert "mace_off" in report.issues[0].message, "the known names are offered"


def test_a_known_task_name_passes_the_backend_check(tmp_path: Path):
    report = _validate(
        tmp_path,
        steps=[{"step": 1, "engine": "mlip", "options": {"task_name": "mace_off"}}],
    )
    assert report.ok


def test_a_lenient_options_failure_skips_the_backend_check(tmp_path: Path):
    """A bad declared value already failed the options read; the lookup must not pile on.

    ``backend_requirement`` reads the selection through the same lenient model, so run
    against options that cannot be read it would raise the *options* error a second time
    — one mistake, two rows, and neither naming the other.
    """
    report = _validate(
        tmp_path,
        steps=[{"step": 1, "engine": "mlip", "options": {"task_name": "mace_of", "cores": 0}}],
    )
    assert _kinds(report) == ["options"]


def test_bad_nms_knobs_are_an_error(tmp_path: Path):
    report = _validate(
        tmp_path,
        steps=[{"step": 1, "engine": "fake", "nms": True, "options": {"target": "bogus"}}],
    )
    assert "nms" in _kinds(report)


# ---------------------------------------------------------------------------
# Silent no-ops → warnings (the config still runs)
# ---------------------------------------------------------------------------


def test_undeclared_option_keys_warn_but_do_not_block(tmp_path: Path):
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "fake", "options": {"typoed": 1}}])
    assert report.ok
    [warning] = report.warnings
    assert warning.kind == "options"
    assert "typoed" in warning.message


def test_undeclared_keys_are_judged_against_every_declared_reader():
    """One rule for the report and the run: a key the engine's model or NMS declares is read.

    Everything else changes nothing — there is no placeholder path that could read it, so the
    sentence no longer hedges on one — and the readers are named so a misspelt NMS knob on an
    ORCA step (whose only options reader *is* NMS) is described as exactly that.
    """
    from chemrefine.config import StepConfig
    from chemrefine.validate import undeclared_options

    assert undeclared_options(StepConfig(step=1, engine="mlip", options={"device": "cpu"})) is None
    assert (
        undeclared_options(StepConfig(step=1, engine="orca", nms=True, options={"target": "ts"}))
        is None
    )
    assert undeclared_options(StepConfig(step=1, engine="orca")) is None

    nms_typo = undeclared_options(
        StepConfig(step=1, engine="orca", nms=True, options={"targt": "ts"})
    )
    assert nms_typo is not None
    assert "['targt']" in nms_typo and "read by NMS" in nms_typo and "typo" in nms_typo
    assert "placeholder" not in nms_typo

    unread = undeclared_options(StepConfig(step=1, engine="orca", options={"maxiter": 3}))
    assert unread is not None and "declares no options" in unread


def test_nms_on_an_incapable_engine_warns(tmp_path: Path):
    """The run silently skips NMS for an incapable engine — the report must not."""
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "fake", "nms": True}])
    assert report.ok
    assert any(w.kind == "nms" and "ignored" in w.message for w in report.warnings)


def test_nms_on_a_capable_engine_does_not_warn(tmp_path: Path):
    """ORCA can NMS; only the missing-template warnings may appear for this step."""
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "orca", "nms": True}])
    assert report.ok
    assert not any(w.kind == "nms" for w in report.warnings)


def test_missing_template_and_header_warn_with_the_scaffold_hint(tmp_path: Path):
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "orca"}])
    assert report.ok
    kinds = {w.kind for w in report.warnings}
    assert kinds == {"template", "slurm-header"}
    assert all("scaffold" in w.message for w in report.warnings)


def test_existing_template_and_header_do_not_warn(tmp_path: Path):
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "step1.inp").write_text("! Opt\n", encoding="utf-8")
    (templates / "cpu.slurm.header").write_text("#SBATCH -N1\n", encoding="utf-8")
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "orca"}])
    assert report.ok
    assert report.warnings == ()


def test_a_gpu_step_is_held_to_the_cuda_header(tmp_path: Path):
    """Header choice mirrors dispatch: GPU demand read through the engine's own model."""
    report = _validate(
        tmp_path,
        steps=[{"step": 1, "engine": "mlip", "options": {"device": "cuda"}}],
    )
    assert any(
        w.kind == "slurm-header" and "cuda.slurm.header" in w.message for w in report.warnings
    )


def test_an_explicit_step_header_wins_over_the_gpu_default(tmp_path: Path):
    report = _validate(
        tmp_path,
        steps=[
            {
                "step": 1,
                "engine": "mlip",
                "slurm_template": "special.header",
                "options": {"device": "cuda"},
            }
        ],
    )
    assert any(w.kind == "slurm-header" and "special.header" in w.message for w in report.warnings)


def test_options_errors_suppress_the_gpu_header_probe(tmp_path: Path):
    """A step whose options already failed reports that failure, not a second crash."""
    report = _validate(
        tmp_path,
        steps=[{"step": 1, "engine": "mlip", "options": {"device": "warp-drive"}}],
    )
    assert not report.ok
    assert _kinds(report) == ["options"]
    # The header probe fell back to the global header rather than re-reading bad options.
    assert any(
        w.kind == "slurm-header" and "cpu.slurm.header" in w.message for w in report.warnings
    )


# ---------------------------------------------------------------------------
# File entry point + serialization
# ---------------------------------------------------------------------------


def test_validate_config_file_reads_and_resolves_like_load_config(tmp_path: Path):
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "step1.inp").write_text("! Opt\n", encoding="utf-8")
    (templates / "cpu.slurm.header").write_text("#SBATCH -N1\n", encoding="utf-8")
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(_config_dict(steps=[{"step": 1, "engine": "orca"}])))
    report = validate_config_file(path)
    assert report.ok
    assert report.warnings == ()  # relative template_dir resolved against the file's dir
    assert report.config is not None
    assert report.config.template_dir == templates


def test_an_unreadable_file_is_one_io_issue(tmp_path: Path):
    report = validate_config_file(tmp_path / "absent.yaml")
    assert not report.ok
    assert _kinds(report) == ["io"]


def test_a_deprecated_spelling_is_a_warning_not_silence(tmp_path: Path):
    """A legacy key still runs, but the report has to say so.

    It used to reach a logger and nowhere else, so ``ok`` came back true with an empty
    ``warnings`` list — and the agent, whose guide tells it to validate "until ok with no
    surprising warnings", was handed a clean bill for a spelling due for removal in 3.0.
    The GUI's Validate button rendered the same file as "valid — no findings".
    """
    report = validate_config_text(
        yaml.safe_dump(
            {"orca_executable": "/opt/orca/orca", "steps": [{"step": 1, "engine": "fake"}]}
        ),
        base_dir=tmp_path,
    )
    assert report.ok  # deprecated is not broken
    deprecations = [w for w in report.warnings if w.kind == "deprecated"]
    assert [w.loc for w in deprecations] == [("orca_executable",)]
    assert "executables" in deprecations[0].message


def test_deprecations_survive_a_config_that_also_fails_validation(tmp_path: Path):
    """A file can be both legacy *and* wrong; the vocabulary finding is not lost.

    The model-error path returns early, so without carrying them the deprecation would be
    dropped exactly when the user is already editing the file.
    """
    report = validate_config_text(
        yaml.safe_dump({"orca_executable": "/opt/orca/orca", "steps": "not-a-list"}),
        base_dir=tmp_path,
    )
    assert not report.ok
    assert [w.kind for w in report.warnings] == ["deprecated"]


def test_a_current_config_still_reports_no_deprecations(tmp_path: Path):
    report = _validate(tmp_path, executables={"orca": "/opt/orca/orca"})
    assert [w for w in report.warnings if w.kind == "deprecated"] == []


def test_the_report_serializes_whole(tmp_path: Path):
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "no-such-engine"}])
    wire = json.loads(json.dumps(report.to_json()))
    assert wire["ok"] is False
    assert wire["issues"][0]["kind"] == "engine"
    assert wire["issues"][0]["loc"] == ["steps", 0, "engine"]


def test_an_absent_executable_path_is_a_warning_row_not_only_a_log_line(tmp_path: Path):
    """`chemrefine validate`, the GUI, and MCP clients read the report, not stderr.

    The load-time check was logger-only, so every report reader was told `ok` with no
    warnings while a typo'd `/opt/orca/orca` path waited to fail at submit time — the
    same finding-only-for-stderr-watchers class the deprecation sink closed for legacy
    spellings. A bare command name stays off the report: it resolves on the executing
    host at submit time, exactly as the load-time warning's contract says.
    """
    report = _validate(tmp_path, executables={"orca": str(tmp_path / "no" / "orca")})
    [row] = [w for w in report.warnings if w.kind == "executable"]
    assert row.loc == ("executables", "orca")
    assert "does not exist on this host" in row.message

    bare = _validate(tmp_path, executables={"orca": "orca"})
    assert [w for w in bare.warnings if w.kind == "executable"] == []
