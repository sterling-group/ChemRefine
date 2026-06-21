"""Tests for the template-driven direct ``PyscfEngine``.

The new direct engine doesn't import PySCF at all — it renders the
user's ``step{N}.py`` template, runs it as a subprocess (via
``slurm.submit`` with its local-bash fallback), and reads the JSON
the script writes. The tests install a tiny template that emits a
known energy + gradient and check the prepare → submit → parse
lifecycle end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines.base import get_engine
from chemrefine.errors import OutputParseError
from chemrefine.state import JobBatch, PipelineState, StepContext, StepResults, Structure

# ---------------------------------------------------------------------------
# Sample template that bypasses PySCF and just emits a deterministic result.
# ---------------------------------------------------------------------------

_FAKE_TEMPLATE = """\
'''Fake PySCF template used by the direct-engine tests.

The user template only declares the result variables; the appended
ChemRefine output footer writes the JSON. No PySCF imports — these
tests don't need a real backend.
'''
charge = $CHARGE
mult = $MULTIPLICITY
energy_hartree = -1.234 + 0.01 * charge
gradient_hartree_per_bohr = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]
"""


def _write_templates(tmp_path: Path) -> Path:
    """Populate ``tmp_path`` with a PySCF template + a minimal SLURM header."""
    (tmp_path / "step1.py").write_text(_FAKE_TEMPLATE, encoding="utf-8")
    (tmp_path / "cpu.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --partition=normal\n",
        encoding="utf-8",
    )
    return tmp_path


def _ctx(
    tmp_path: Path,
    structures: tuple[Structure, ...],
    **overrides,
) -> StepContext:
    _write_templates(tmp_path)
    step_cfg = StepConfig(
        step=1,
        name="screen",
        engine="pyscf",
        operation="opt_sp",
        options=overrides.pop("options", {}),
    )
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1_screen",
        template_dir=tmp_path,
        scratch_dir=None,
        prev_state=PipelineState(structures=structures),
        charge=overrides.pop("charge", 0),
        multiplicity=overrides.pop("multiplicity", 1),
        max_cores=overrides.pop("max_cores", 1),
        slurm_template="cpu.slurm.header",
        executables={},
    )


def _seed(sid: str = "0") -> Structure:
    return Structure(
        id=sid,
        atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_pyscf_direct_engine_is_registered():
    engine = get_engine("pyscf")
    assert engine.name == "pyscf"
    assert engine.supports_nms is False


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def test_prepare_renders_one_py_and_xyz_per_structure(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed("0"), _seed("1")))
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    assert len(inputs.files) == 2
    for script_path, output_json, sid in inputs.files:
        assert script_path.parent.name == sid  # per-structure directory
        assert script_path.name == f"step1_{sid}.py"
        assert output_json.name == f"step1_{sid}.json"
        assert (script_path.parent / f"{script_path.stem}_inp.xyz").is_file()
        rendered = script_path.read_text()
        # Geometry placeholders should all be substituted.
        assert "$XYZ_PATH" not in rendered
        assert "$CHARGE" not in rendered
        assert "$MULTIPLICITY" not in rendered
        # The appended footer writes to the BASENAME (relative to cwd =
        # scratch); SLURM's *.json glob then copies it back to step_dir.
        assert f"with open('{output_json.name}', \"w\")" in rendered


def test_prepare_missing_template_raises(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed(),))
    (ctx.template_dir / "step1.py").unlink()
    engine = get_engine("pyscf")
    with pytest.raises(FileNotFoundError, match="PySCF template not found"):
        engine.prepare(ctx)


def test_prepare_uses_step_specific_template_when_given(tmp_path: Path):
    step_cfg = StepConfig(
        step=2,
        engine="pyscf",
        operation="opt_sp",
        template="custom.py",
    )
    _write_templates(tmp_path)
    (tmp_path / "custom.py").write_text(_FAKE_TEMPLATE, encoding="utf-8")
    ctx = StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step2",
        template_dir=tmp_path,
        scratch_dir=None,
        prev_state=PipelineState(structures=(_seed(),)),
        charge=0,
        multiplicity=1,
        max_cores=1,
        slurm_template="cpu.slurm.header",
        executables={},
    )
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    assert "$" not in inputs.files[0][0].read_text()  # placeholders gone


# ---------------------------------------------------------------------------
# submit (full template execution via the local-bash fallback)
# ---------------------------------------------------------------------------


def test_submit_runs_template_locally_when_no_sbatch(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    # No sbatch available → slurm.submit takes the local-bash path,
    # which runs the rendered template synchronously.
    with patch("chemrefine.slurm.shutil.which", return_value=None):
        batch = engine.submit(inputs, ctx)
    assert isinstance(batch, JobBatch)
    assert all(jid.startswith("local-") for jid in batch.jobs.values())
    # The template ran and wrote the JSON output.
    output_json = inputs.files[0][1]
    assert output_json.is_file()
    data = json.loads(output_json.read_text())
    assert "energy_hartree" in data


def test_submit_template_failure_is_deferred_to_parsing(tmp_path: Path):
    """A template that doesn't assign energy_hartree exits non-zero, but the background
    local run no longer raises at submit — the missing JSON output surfaces when the
    step parses results (the same path SLURM failures take → the on_failure ledger)."""
    ctx = _ctx(tmp_path, structures=(_seed(),))
    # Build the context first (which writes the good fake template), then
    # overwrite step1.py with a broken one before prepare() reads it.
    (ctx.template_dir / "step1.py").write_text(
        "x = 1  # forgot to assign energy_hartree\n",
        encoding="utf-8",
    )
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    with patch("chemrefine.slurm.shutil.which", return_value=None):
        batch = engine.submit(inputs, ctx)  # does not raise
    assert all(jid.startswith("local-") for jid in batch.jobs.values())
    # The failed run wrote no valid JSON output → detected downstream at parse time.
    assert not inputs.files[0][1].is_file()


def test_template_run_block_caps_threads_to_cores(tmp_path: Path):
    """pyscf/mlip direct runs are OpenMP/MKL-threaded → the run block pins them to cores."""
    ctx = _ctx(tmp_path, structures=(_seed(),), options={"cores": 4})
    engine = get_engine("pyscf")
    run_block = engine._run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.py",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "export OMP_NUM_THREADS=4" in run_block
    assert "export MKL_NUM_THREADS=4" in run_block
    assert "python step1_structure_0.py" in run_block


def test_submit_missing_slurm_header_raises(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed(),))
    (ctx.template_dir / "cpu.slurm.header").unlink()
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    with pytest.raises(FileNotFoundError, match="SLURM header"):
        engine.submit(inputs, ctx)


def test_submit_respects_cores_option(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed(),), options={"cores": 2}, max_cores=4)
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    with patch("chemrefine.slurm.shutil.which", return_value=None):
        engine.submit(inputs, ctx)
    # The generated SLURM script should request the configured cores.
    script_text = inputs.files[0][0].with_suffix(".slurm").read_text()
    assert "#SBATCH --ntasks=2" in script_text


def test_submit_uses_template_engine_output_globs(tmp_path: Path):
    """The direct engine's ``output_globs`` ClassVar flows through SlurmBatchEngine."""
    ctx = _ctx(tmp_path, structures=(_seed(),))
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    with patch("chemrefine.slurm.shutil.which", return_value=None):
        engine.submit(inputs, ctx)
    script_text = inputs.files[0][0].with_suffix(".slurm").read_text()
    assert "*.json" in script_text
    # ORCA-only globs must not leak into a direct-engine script.
    assert "*.gbw" not in script_text


# ---------------------------------------------------------------------------
# wait
# ---------------------------------------------------------------------------


def test_wait_is_noop():
    engine = get_engine("pyscf")
    engine.wait(JobBatch(jobs={}))


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


def test_parse_returns_structures_with_energy_and_forces(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    with patch("chemrefine.slurm.shutil.which", return_value=None):
        engine.submit(inputs, ctx)
    results = engine.parse(inputs, ctx)
    assert len(results.structures) == 1
    s = results.structures[0]
    assert s.id == "0"
    assert s.energy_hartree == pytest.approx(-1.234)
    assert s.forces_ev_per_a is not None
    assert s.forces_ev_per_a.shape == (2, 3)


def test_parse_uses_positions_from_output_when_present(tmp_path: Path):
    """If the script wrote ``positions_angstrom``, the parser updates the geometry."""
    ctx = _ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    out_json = inputs.files[0][1]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "energy_hartree": -2.0,
                "positions_angstrom": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
            }
        ),
        encoding="utf-8",
    )
    results = engine.parse(inputs, ctx)
    np.testing.assert_allclose(
        results.structures[0].atoms.get_positions(),
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]],
    )


def test_parse_raises_when_output_missing(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    # Don't submit — output JSON doesn't exist.
    with pytest.raises(OutputParseError, match="output not found"):
        engine.parse(inputs, ctx)


def test_parse_raises_when_output_not_json(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    output_json = inputs.files[0][1]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text("not json", encoding="utf-8")
    with pytest.raises(OutputParseError, match="not valid JSON"):
        engine.parse(inputs, ctx)


def test_parse_raises_when_energy_missing(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    output_json = inputs.files[0][1]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"oops": 0.0}), encoding="utf-8")
    with pytest.raises(OutputParseError, match="energy_hartree"):
        engine.parse(inputs, ctx)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------


def test_normal_mode_sample_not_supported(tmp_path: Path):
    engine = get_engine("pyscf")
    with pytest.raises(NotImplementedError, match="does not support normal-mode"):
        engine.normal_mode_sample(StepResults(structures=()), _ctx(tmp_path, ()))
