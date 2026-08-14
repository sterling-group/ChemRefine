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


def test_model_errors_keep_pydantic_locs():
    """A GUI highlights the offending field by walking ``loc`` — it must survive."""
    report = validate_config_text(yaml.safe_dump(_config_dict(charge="not-an-int")))
    assert not report.ok
    assert any(issue.loc[0] == "charge" for issue in report.issues)
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
    assert any(
        w.kind == "slurm-header" and "special.header" in w.message for w in report.warnings
    )


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


def test_the_report_serializes_whole(tmp_path: Path):
    report = _validate(tmp_path, steps=[{"step": 1, "engine": "no-such-engine"}])
    wire = json.loads(json.dumps(report.to_json()))
    assert wire["ok"] is False
    assert wire["issues"][0]["kind"] == "engine"
    assert wire["issues"][0]["loc"] == ["steps", 0, "engine"]
