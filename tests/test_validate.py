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
import os
from pathlib import Path

import pytest
import yaml

from chemrefine.validate import (
    ValidationIssue,
    ValidationReport,
    validate_config_file,
    validate_config_text,
)


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
    containing a space. The run refuses it before any step runs; this is the same refusal,
    early.
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
    assert warning.message.startswith("step 1's directory contains whitespace")
    assert "whitespace-delimited" in warning.message


def test_only_the_directory_a_step_writes_under_is_held_to_the_whitespace_rule(tmp_path: Path):
    """A spaced template or scratch directory is no part of the rule; the step directory is.

    An ORCA input names the paths ChemRefine composes under the step's directory, and reads
    a template's own quoted ``%``-block paths fine — so the run and ``validate`` ask the rule
    of that directory alone.
    """
    spaced, clean = tmp_path / "my tree", tmp_path / "tree"
    (spaced / "templates").mkdir(parents=True)
    (spaced / "templates" / "step1.inp").write_text("! HF\n", encoding="utf-8")
    (spaced / "templates" / "cpu.slurm.header").write_text("#!/bin/bash\n", encoding="utf-8")
    step = {"step": 1, "engine": "orca", "operation": "sp"}

    def warned(output_dir: Path) -> bool:
        config = {
            "template_dir": str(spaced / "templates"),
            "scratch_dir": str(spaced / "scratch"),
            "output_dir": str(output_dir),
            "steps": [step],
        }
        report = validate_config_text(yaml.safe_dump(config), base_dir=tmp_path)
        return any(w.kind == "whitespace-path" for w in report.warnings)

    assert not warned(clean / "outputs")
    assert warned(spaced / "outputs")


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


def test_an_unknown_operation_is_an_issue_on_its_own_field(tmp_path: Path):
    """A typo'd ``operation:`` must fail here, not after the step's jobs have run.

    Unchecked, it passed validation and the run's preflight, submitted and paid for every
    job, and was then refused per output by the parser — with the paid outputs unadoptable
    afterwards, since the operation is part of every row key.
    """
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "orca", "operation": "opt-sp"}])
    assert report.issues == (
        ValidationIssue(
            loc=("steps", 0, "operation"),
            kind="preflight",
            message=(
                "step 1: unknown operation 'opt-sp' for engine 'orca', which reads docker, "
                "freq, goat, opt_sp, pes, solvator, sp. Correct it, or omit `operation:`."
            ),
        ),
    )


def test_an_operation_in_any_spelling_validates_clean(tmp_path: Path):
    """``OPT+SP`` is ``opt_sp``, permanently: no issue, no deprecation row."""
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "orca", "operation": "OPT+SP"}])
    assert report.ok
    assert not [w for w in report.warnings if w.kind == "deprecated"]
    assert report.config is not None and report.config.steps[0].operation == "opt_sp"


def test_v1s_one_checkpoint_step_validates_with_its_path_moved(tmp_path: Path):
    """Step 4 of v1.3.1's MLIPTraining tutorial named its checkpoint as ``model_name``; it
    is judged as the ``model_path`` it becomes — a file an upstream step has not produced
    yet — and not as a release the family does not have."""
    checkpoint = "../step3/checkpoints_dir/goat_model_run-123_stagetwo.model"
    block = {"model_name": checkpoint, "task_name": "mace_off", "device": "cuda"}
    step = {"step": 1, "operation": "OPT+SP", "engine": "MLFF", "mlff": block}
    report = _validate(tmp_path, steps=[step])
    assert report.ok, report.issues
    assert any(
        w.loc == ("steps", 0, "options", "model_path") and "does not exist yet" in w.message
        for w in report.warnings
    )


def test_an_engine_name_in_any_case_validates_clean(tmp_path: Path):
    """``ORCA`` is ``orca``: no issue, no deprecation row — the config's rule, not v1's."""
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "ORCA", "operation": "opt_sp"}])
    assert report.ok
    assert not [w for w in report.warnings if w.kind == "deprecated"]
    assert report.config is not None and report.config.steps[0].engine == "orca"


def test_the_operation_check_does_not_wait_on_the_options_read(tmp_path: Path):
    """It reads no option, so a bad options read beside it hides nothing."""
    step = {"step": 1, "engine": "qchem", "operation": "goat", "options": {"nprocs": "x"}}
    report = _validate(tmp_path, steps=[step])
    assert _kinds(report) == ["preflight", "options"]


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


@pytest.mark.parametrize("engine", ["mlip", "mlip-extopt", "mlip-train"])
def test_a_model_name_its_library_does_not_release_is_one_backend_row(tmp_path: Path, engine: str):
    """The run refuses it at t=0, so validate does — on the backend row, once."""
    report = _validate(
        tmp_path,
        steps=[
            {
                "step": 1,
                "engine": engine,
                "options": {"task_name": "mace_omol", "model_name": "medium", "device": "cpu"},
            }
        ],
    )
    assert _kinds(report) == ["backend"]
    assert "its releases: extra_large" in report.issues[0].message


def test_a_label_of_the_templates_own_belongs_in_extra_not_model_name(tmp_path: Path):
    """``model_name`` names the library's weights; a knob of the step's own goes in ``extra``."""
    labelled = {"task_name": "mace_off", "model_name": "my-label"}
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "mlip", "options": labelled}])
    assert _kinds(report) == ["backend"]
    in_extra = {"task_name": "mace_off", "extra": {"label": "my-label"}}
    assert _validate(tmp_path, steps=[{"step": 1, "engine": "mlip", "options": in_extra}]).ok


def test_an_unknown_training_task_is_one_row(tmp_path: Path):
    """The trainer is the library's, so an unknown ``task_name`` is the backend row alone."""
    options = {"task_name": "mace_of", "device": "cpu"}
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "mlip-train", "options": options}])
    assert _kinds(report) == ["backend"]


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
        steps=[{"step": 1, "engine": "orca", "nms": True, "options": {"target": "bogus"}}],
    )
    [issue] = [i for i in report.issues if i.kind == "nms"]
    assert issue.loc == ("steps", 0, "options") and "step 1: invalid NMS options" in issue.message


def test_nms_knobs_an_engine_cannot_use_are_left_to_the_ignored_warning(tmp_path: Path):
    """The run never reads them for an engine that cannot sample, so they are no error;
    the one thing worth saying is that ``nms: true`` does nothing."""
    report = _validate(
        tmp_path,
        steps=[{"step": 1, "engine": "fake", "nms": True, "options": {"target": "bogus"}}],
    )
    assert "nms" not in _kinds(report)
    assert any(w.kind == "nms" and "ignored" in w.message for w in report.warnings)


def test_nms_knobs_on_an_extopt_step_are_not_a_preflight_issue(tmp_path: Path):
    """``nms: true`` makes the NMS knobs a second reader's, so the strict engine read leaves them.

    Without it they are strangers to the server model, and the preflight issue stands.
    """
    step = {
        "step": 1,
        "engine": "pyscf-extopt",
        "options": {"basis": "def2-svp", "xc": "pbe", "target": "ts"},
    }
    assert _kinds(_validate(tmp_path, steps=[{**step, "nms": True}])) == []
    assert _kinds(_validate(tmp_path, steps=[step])) == ["preflight"]


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


def test_a_sampling_step_whose_input_computes_no_frequencies_is_an_issue_on_its_nms(
    tmp_path: Path,
):
    """The run refuses it when it starts, so validate reports it — on the field that asked
    for sampling, quoting how the engine asks for frequencies."""
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "step1.inp").write_text("! B3LYP def2-SVP Opt\n", encoding="utf-8")
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "orca", "nms": True}])
    [issue] = [i for i in report.issues if i.kind == "nms"]
    assert issue.loc == ("steps", 0, "nms") and "`NumFreq true`" in issue.message


def test_missing_template_and_header_warn_with_the_scaffold_hint(tmp_path: Path):
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "orca"}])
    assert report.ok
    kinds = {w.kind for w in report.warnings}
    assert kinds == {"template", "slurm-header"}
    assert all("scaffold" in w.message for w in report.warnings)


def test_a_path_valued_option_that_does_not_exist_yet_warns(tmp_path: Path):
    """``model_path`` gets the "does not exist yet" its three siblings have.

    Keyed to ``STEP_OPTION_PATHS`` — the config's own declaration of which option values
    are paths — rather than to the knob by name. A warning, not a refusal: the tutorials
    point ``model_path`` at a product an earlier step trains. A typo used to pass every
    pre-run check and fail inside the job's loader, after the upstream steps had run.
    """
    steps = [{"step": 1, "engine": "mlip", "options": {"model_path": "./models/typo.model"}}]
    report = _validate(tmp_path, steps=steps)
    assert report.ok
    missing = [w for w in report.warnings if w.loc == ("steps", 0, "options", "model_path")]
    assert len(missing) == 1
    assert "does not exist yet" in missing[0].message
    assert str(tmp_path / "models" / "typo.model") in missing[0].message  # the resolved path

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "typo.model").write_bytes(b"")
    report = _validate(tmp_path, steps=steps)
    assert not [w for w in report.warnings if w.loc == ("steps", 0, "options", "model_path")]


def test_existing_template_and_header_do_not_warn(tmp_path: Path):
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "step1.inp").write_text("! Opt\n", encoding="utf-8")
    (templates / "cpu.slurm.header").write_text("#SBATCH -N1\n", encoding="utf-8")
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "orca"}])
    assert report.ok
    assert report.warnings == ()


@pytest.mark.parametrize(
    ("name", "kind"), [("step1.inp", "template"), ("cpu.slurm.header", "slurm-header")]
)
def test_a_template_or_header_that_is_not_utf8_is_an_issue(tmp_path: Path, name: str, kind: str):
    """What the run refuses before it starts, validate refuses too, rather than saying ok.

    A latin-1 ``Å`` in a comment passed here as OK; the run then bootstrapped its seeds and
    refused the template at ``prepare``, or the header at the first submission.
    """
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "step1.inp").write_text("! Opt\n", encoding="utf-8")
    (templates / "cpu.slurm.header").write_text("#SBATCH -N1\n", encoding="utf-8")
    (templates / name).write_bytes("# Ångström\n! Opt\n".encode("latin-1"))
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "orca"}])
    assert not report.ok
    assert _kinds(report) == [kind]
    assert "is not UTF-8 text" in report.issues[0].message


@pytest.mark.parametrize(
    ("name", "kind", "what"),
    [
        ("step1.inp", "template", "ORCA template"),
        ("cpu.slurm.header", "slurm-header", "SLURM header template"),
    ],
)
def test_a_template_or_header_that_cannot_be_read_is_an_issue(
    tmp_path: Path, name: str, kind: str, what: str
):
    """Another account's file is a row naming it: validate never raises.

    The read raised ``PermissionError`` past that contract, which the GUI answered with a
    500 and the MCP server with a generic tool failure.
    """
    if os.geteuid() == 0:
        pytest.skip("root reads everything; the permission wall cannot be built")
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "step1.inp").write_text("! Opt\n", encoding="utf-8")
    (templates / "cpu.slurm.header").write_text("#SBATCH -N1\n", encoding="utf-8")
    (templates / name).chmod(0)
    try:
        report = _validate(tmp_path, steps=[{"step": 1, "engine": "orca"}])
    finally:
        (templates / name).chmod(0o644)
    assert _kinds(report) == [kind]
    assert f"cannot read {what} {templates / name}" in report.issues[0].message


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


def test_a_template_shared_across_engines_is_the_warning_the_planner_refuses_on(
    tmp_path: Path,
):
    """``validate`` and ``scaffold`` read one rule, so OK here is never a refusal there.

    Steps of different engines naming one template file run — each renders the same file — but
    the scaffold planner, and every template tool through it, refuse the config, and the
    validator said nothing: a green Validate button beside a red Scaffold button. One
    function answers both now; the validator reports it as a warning because the run
    itself is not broken. Sharing within one engine is ordinary and says nothing.
    """
    shared = [
        {"step": 1, "engine": "orca", "template": "shared.inp"},
        {"step": 2, "engine": "qchem", "template": "shared.inp"},
    ]
    report = _validate(tmp_path, steps=shared)
    assert report.ok
    clash = [w for w in report.warnings if w.kind == "template" and "both name" in w.message]
    assert len(clash) == 1
    assert clash[0].loc == ("steps",)
    assert "shared.inp" in clash[0].message

    within_one_engine = [dict(step, engine="orca") for step in shared]
    report = _validate(tmp_path, steps=within_one_engine)
    assert not [w for w in report.warnings if "both name" in w.message]


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


def test_a_file_that_is_not_utf8_is_one_io_issue(tmp_path: Path):
    """The decode half of "cannot be read", held to the same never-raises contract."""
    path = tmp_path / "input.yaml"
    path.write_bytes("# Ångström\nsteps: []\n".encode("latin-1"))
    report = validate_config_file(path)
    assert not report.ok
    assert _kinds(report) == ["io"]
    assert "not UTF-8" in report.issues[0].message


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


def test_a_seed_that_does_not_exist_yet_is_a_warning(tmp_path: Path):
    """``input:`` pointing at nothing was ``ok`` with no row here, and a traceback at the run.

    A warning like the template's, not an issue: the file may be written later or exist
    only on the cluster the config is for, and ``save_config`` must still write the config.
    """
    report = _validate(tmp_path, input="./seeds.xyz")
    assert report.ok
    [warning] = [w for w in report.warnings if w.kind == "input"]
    assert warning.loc == ("input",)
    assert "seeds.xyz" in warning.message

    (tmp_path / "seeds.xyz").write_text("1\n\nH 0 0 0\n", encoding="utf-8")
    present = _validate(tmp_path, input="./seeds.xyz")
    assert not any(w.kind == "input" for w in present.warnings)


# ---------------------------------------------------------------------------
# The scheduler's budget, by the run's own check
# ---------------------------------------------------------------------------


def _budget(report: ValidationReport) -> list[str]:
    return [w.message for w in report.warnings if w.kind == "budget"]


def test_a_cuda_step_against_a_stated_zero_gpu_budget_is_warned_about(tmp_path: Path):
    """The finding's own case: ``validate`` said OK, and the run refused at step 2's turn.

    A warning rather than an issue: ``--maxgpus`` (or ``start_run(max_gpus=…)``) can raise
    the budget for the run, and a config meant to run under one is valid as written.
    """
    report = _validate(
        tmp_path,
        max_gpus=0,
        steps=[
            {"step": 1, "engine": "fake"},
            {"step": 2, "engine": "mlip", "options": {"device": "cuda"}},
        ],
    )
    assert report.ok
    [warned] = _budget(report)
    assert "step 2 needs 1 GPU(s) but the budget is 0" in warned


def test_a_layout_over_max_cores_is_warned_about(tmp_path: Path):
    """An MPI x threads layout is refused rather than reshaped — so it is decidable here."""
    report = _validate(
        tmp_path, steps=[{"step": 1, "engine": "qchem", "options": {"nprocs": 4, "cores": 4}}]
    )
    assert report.ok
    [warned] = _budget(report)
    assert "16 cores, more than max_cores=4" in warned


def test_several_gpus_under_an_explicit_local_dispatch_are_warned_about(tmp_path: Path):
    """``dispatch: local`` pins one device per job, and says so without asking the host."""
    report = _validate(
        tmp_path,
        dispatch="local",
        steps=[
            {
                "step": 1,
                "engine": "mlip-train",
                "options": {"task_name": "mace_off", "device": "cuda", "gpus": 2},
            }
        ],
    )
    assert any("local dispatch pins one device" in message for message in _budget(report))


def test_what_only_the_host_can_answer_bounds_nothing(tmp_path: Path):
    """An unset ``max_gpus`` is the host's to resolve; a laptop must not judge the cluster."""
    report = _validate(
        tmp_path, steps=[{"step": 1, "engine": "mlip", "options": {"device": "cuda"}}]
    )
    assert _budget(report) == []


def test_a_preflight_refusal_does_not_hide_the_budget_row(tmp_path: Path):
    """Every finding at once: a step the preflight refuses is still held to the budget.

    The two rows are two consequences, each with its own fix; hiding the second behind the
    first means fixing one row, re-validating, and only then being told the next.
    """
    report = _validate(
        tmp_path,
        dispatch="local",
        steps=[
            {
                "step": 1,
                "engine": "mlip-train",
                "on_failure": "skip",
                "options": {"task_name": "mace_off", "device": "cuda", "gpus": 2},
            }
        ],
    )
    assert "preflight" in [issue.kind for issue in report.issues]
    assert any("local dispatch pins one device" in message for message in _budget(report))


_TRAINING_FACTS = "name: $RUN_NAME\nwork_dir: $RUN_DIR\ndevice: $DEVICE\nseed: $SEED\n"
"""A MACE training template carrying the plan facts and not the dataset."""


def _training(
    tmp_path: Path, template: str, options: dict[str, object], **config: object
) -> ValidationReport:
    """Validate one ``mlip-train`` step with ``options``, whose template is ``template``."""
    (tmp_path / "templates").mkdir(exist_ok=True)
    (tmp_path / "templates" / "step1.yaml").write_text(template, encoding="utf-8")
    return _validate(
        tmp_path, steps=[{"step": 1, "engine": "mlip-train", "options": options}], **config
    )


def test_a_training_template_its_trainer_refuses_is_an_issue_on_its_row(tmp_path: Path):
    """The run's t=0 walk refuses it, so validate reports it, on the template it is about."""
    report = _training(tmp_path, _TRAINING_FACTS, {"task_name": "mace_off", "device": "cpu"})
    [issue] = report.issues
    assert (issue.loc, issue.kind) == (("steps", 0, "template"), "template")
    assert "never references $TRAIN_SET" in issue.message


def test_an_inferred_operation_without_a_reader_is_a_warning_on_the_operation(tmp_path: Path):
    """The run reads such a step's outputs as their final structure, and says so when it
    starts; validate says the same, on the field that would change it."""
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "step1.inp").write_text("! B3LYP def2-SVP IRC\n", encoding="utf-8")
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "orca"}])
    [notice] = [w for w in report.warnings if w.kind == "operation"]
    assert notice.loc == ("steps", 0, "operation") and "'irc'" in notice.message


def test_a_template_that_cannot_be_read_does_not_hide_the_budget_row(tmp_path: Path):
    """The template's refusal is one row; the layout reads the options, not the template, so
    it is judged too."""
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "step1.yaml").write_bytes("# \u00c5\n".encode("latin-1"))
    options = {"task_name": "mace_off", "device": "cuda", "gpus": 2}
    report = _validate(
        tmp_path,
        dispatch="local",
        steps=[{"step": 1, "engine": "mlip-train", "options": options}],
    )
    assert _kinds(report) == ["template"] and "is not UTF-8 text" in report.issues[0].message
    assert any("local dispatch pins one device" in message for message in _budget(report))


def test_every_kind_validate_files_is_named_where_the_kinds_are_listed():
    """``ValidationIssue`` lists the kinds a GUI or an agent can switch on; the list is held
    to the kinds the module files, read off its source."""
    import ast

    from chemrefine import validate

    kinds: set[str] = set()
    for node in ast.walk(ast.parse(Path(validate.__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.keyword) and node.arg == "kind":
            value = node.value
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_failed":
            value = node.args[0]
        else:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            kinds.add(value.value)
    assert {"template", "operation", "budget"} <= kinds
    assert sorted(k for k in kinds if f"``{k}``" not in (ValidationIssue.__doc__ or "")) == []


def test_a_complete_training_template_validates_clean(tmp_path: Path):
    template = f"{_TRAINING_FACTS}train_file: $TRAIN_SET\nfoundation_model: $FOUNDATION_MODEL\n"
    options = {"task_name": "mace_off", "model_name": "small", "device": "cpu"}
    assert _training(tmp_path, template, options).issues == ()


@pytest.mark.parametrize(
    ("options", "kinds"),
    [
        ({"task_name": "mace_off"}, ["preflight"]),
        ({"task_name": "mace_omol", "model_name": "medium", "device": "cpu"}, ["backend"]),
    ],
    ids=["preflight", "backend"],
)
def test_a_step_whose_plan_another_row_refuses_gets_no_template_row(
    tmp_path: Path, options: dict[str, object], kinds: list[str]
):
    """The template rules read the step's plan, so a plan refused is that row's, once."""
    assert _kinds(_training(tmp_path, _TRAINING_FACTS, options)) == kinds


def test_a_template_refusal_does_not_hide_the_budget_row(tmp_path: Path):
    """Every finding at once: the rules leave the layout computable, so it is judged too."""
    options = {"task_name": "mace_off", "device": "cuda", "gpus": 2}
    report = _training(tmp_path, _TRAINING_FACTS, options, dispatch="local")
    assert _kinds(report) == ["template"]
    assert any("local dispatch pins one device" in message for message in _budget(report))


def test_a_layout_its_options_cannot_give_is_left_to_their_row(tmp_path: Path):
    """A typoed knob is one finding, on the preflight row, and never again as a budget row.

    A training step's layout reads its options strictly, so the knob the preflight row
    refuses refuses the layout too. That refusal is the row's to report; the budget is
    judged on a layout the step can compute, here the same over-asking ``gpus: 2`` the
    previous test is warned about, once the knob is fixed.
    """
    report = _validate(
        tmp_path,
        dispatch="local",
        steps=[
            {
                "step": 1,
                "engine": "mlip-train",
                "options": {"task_name": "mace_off", "device": "cuda", "gpus": 2, "max_epocs": 3},
            }
        ],
    )
    assert _kinds(report) == ["preflight"]
    assert "max_epocs" in report.issues[0].message
    assert _budget(report) == []
