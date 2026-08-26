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
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines.api import get_engine
from chemrefine.errors import ConfigError, OutputParseError
from chemrefine.state import JobBatch, PipelineState, StepContext, Structure

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
        template=tmp_path / "step1.py",
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
        assert f'with open(\'{output_json.name}\', "w", encoding="utf-8")' in rendered


def test_prepare_missing_template_raises(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed(),))
    (ctx.template_dir / "step1.py").unlink()
    engine = get_engine("pyscf")
    with pytest.raises(ConfigError, match="PySCF template not found"):
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
        template=tmp_path / "custom.py",
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
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value=None):
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
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value=None):
        batch = engine.submit(inputs, ctx)  # does not raise
    assert all(jid.startswith("local-") for jid in batch.jobs.values())
    # The failed run wrote no valid JSON output → detected downstream at parse time.
    assert not inputs.files[0][1].is_file()


def test_template_run_block_caps_threads_to_cores(tmp_path: Path):
    """pyscf/mlip direct runs are OpenMP/MKL-threaded → the run block pins them to cores."""
    ctx = _ctx(tmp_path, structures=(_seed(),), options={"cores": 4}, max_cores=4)
    engine = get_engine("pyscf")
    run_block = engine.run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.py",
        out_path=ctx.step_dir / "step1_structure_0.out",
    ).body
    assert "export OMP_NUM_THREADS=4" in run_block
    assert "export MKL_NUM_THREADS=4" in run_block
    # The launcher is the *resolved* interpreter, whose basename depends on how
    # Python was invoked (``bin/python`` directly vs ``bin/python3.13`` via a
    # console-script shebang) — assert the real path, not a "python" substring.
    assert f"{shlex.quote(sys.executable)} step1_structure_0.py" in run_block


def test_thread_exports_say_the_grant_not_the_ask(tmp_path: Path):
    """A ``cores:`` above ``max_cores`` exports the clamped budget, not the request.

    The SLURM directives and the throttler charge come from the ``slurm_layout`` product —
    ``min(cores, max_cores)`` — and the thread exports must say the same number. The raw
    ``pal()`` here let a job charged 4 cores thread 8: invisible under SLURM's cgroups, an
    oversubscription on every local run, and the exact inversion of the invariant this
    export exists for. Q-Chem's run block always read the layout; this pins the script
    engines to the same rule.
    """
    ctx = _ctx(tmp_path, structures=(_seed(),), options={"cores": 8}, max_cores=4)
    run_block = (
        get_engine("pyscf")
        .run_block(
            ctx,
            inp_path=ctx.step_dir / "step1_structure_0.py",
            out_path=ctx.step_dir / "step1_structure_0.out",
        )
        .body
    )
    assert "export OMP_NUM_THREADS=4" in run_block
    assert "export MKL_NUM_THREADS=4" in run_block
    assert "export OPENBLAS_NUM_THREADS=4" in run_block
    assert "=8" not in run_block


def test_submit_missing_slurm_header_raises(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed(),))
    (ctx.template_dir / "cpu.slurm.header").unlink()
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    with pytest.raises(ConfigError, match="SLURM header"):
        engine.submit(inputs, ctx)


def test_submit_respects_cores_option(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed(),), options={"cores": 2}, max_cores=4)
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value=None):
        engine.submit(inputs, ctx)
    # The generated SLURM script should request the configured cores.
    script_text = inputs.files[0][0].with_suffix(".slurm").read_text()
    assert "#SBATCH --ntasks=2" in script_text


def test_submit_uses_template_engine_output_globs(tmp_path: Path):
    """The direct engine's ``output_globs`` ClassVar flows through the JobEngine base."""
    ctx = _ctx(tmp_path, structures=(_seed(),))
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value=None):
        engine.submit(inputs, ctx)
    script_text = inputs.files[0][0].with_suffix(".slurm").read_text()
    assert "*.json" in script_text
    # ORCA-only globs must not leak into a direct-engine script.
    assert "*.gbw" not in script_text


# ---------------------------------------------------------------------------
# wait
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


def test_parse_returns_structures_with_energy_and_forces(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("pyscf")
    inputs = engine.prepare(ctx)
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value=None):
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


def test_pyscf_direct_is_not_nms_capable():
    from chemrefine.engines.api import NmsCapableEngine

    engine = get_engine("pyscf")
    assert not isinstance(engine, NmsCapableEngine)  # provides neither NMS hook


# ---------------------------------------------------------------------------
# Provisioning capability
# ---------------------------------------------------------------------------


def test_pyscf_engines_are_provisionable_with_the_pyscf_env():
    """A CPU step requires the plain ``pyscf`` stack."""
    from chemrefine.engines.api import BackendRequirement, ProvisionableEngine

    for name in ("pyscf", "pyscf-extopt"):
        engine = get_engine(name)
        assert isinstance(engine, ProvisionableEngine)
        assert engine.backend_requirement({"basis": "def2-svp"}) == BackendRequirement(
            extra="pyscf", import_name="pyscf"
        )


@pytest.mark.parametrize("options", [{"gpu": True}, {"device": "cuda"}])
def test_a_gpu_step_requires_the_gpu_stack_by_name(options: dict):
    """A step that asks for a GPU must demand ``gpu4pyscf``, not merely ``pyscf``.

    The requirement used to be a constant, so a GPU step passed preflight against a CPU-only
    env, and `_build_scf` then caught the missing import and fell back to CPU — recording
    why in the ExtOpt *server* log, which nobody reads. The run reported success on the
    wrong hardware. Demanding the import by name is what turns that into a refusal before
    any job is submitted.

    ``device: cuda`` is included because `gpu` is *derived* from it when it is not given: a
    reader that only looked for an explicit `gpu` key would let that spelling through.
    """
    from chemrefine.engines.api import BackendRequirement

    requirement = get_engine("pyscf").backend_requirement({"basis": "def2-svp", **options})

    assert requirement == BackendRequirement(extra="pyscf-gpu", import_name="gpu4pyscf")


def test_both_pyscf_stacks_share_one_managed_env():
    """`[pyscf-gpu]` is `[pyscf]` plus gpu4pyscf — a superset, so one env holds both.

    Two envs would duplicate a large PySCF/libcint/libxc tree for no reason. The
    one-env-per-extra rule elsewhere exists because the MLIP stacks genuinely conflict, and
    that reason does not apply here.
    """
    from chemrefine.engines._provision import backend_env_path

    assert backend_env_path("pyscf-gpu") == backend_env_path("pyscf")
    assert backend_env_path("mlip-mace") != backend_env_path("mlip-fairchem")


def test_the_direct_engine_declares_only_knobs_it_can_honour():
    """An engine's options model is the set of knobs it reads, not its backend's.

    ``pyscf`` reaches its options through template placeholders and ``pyscf-extopt`` builds a
    gradient server from them, so the two share a backend and not a set of knobs. Sharing one
    model lets the direct engine accept knobs it has no channel for — and because
    ``accepted_names()`` reports them as declared, ``chemrefine validate`` cannot warn either,
    so naming one is silence in both directions.

    ``strict_scf`` is the one that matters: the ExtOpt path refuses a non-converged SCF, and
    the direct path has no way to, because a script reports what its output contract declares.
    """
    from chemrefine.engines.api import get_engine
    from chemrefine.engines.pyscf.options import PyscfExtOptOptions, PyscfOptions

    server_only = set(PyscfExtOptOptions.model_fields) - set(PyscfOptions.model_fields)
    assert server_only == {"df", "strict_scf", "save_tensors", "localized", "tensor_folder"}

    direct = get_engine("pyscf")
    assert direct.options_cls is PyscfOptions
    assert not server_only & direct.options_cls.accepted_names(), (
        "the direct engine declares a knob only the gradient server reads"
    )
    assert get_engine("pyscf-extopt").options_cls is PyscfExtOptOptions


def test_a_server_only_knob_is_refused_on_a_direct_step():
    """Refused by name, rather than accepted and ignored."""
    from chemrefine.engines.pyscf.options import PyscfOptions
    from chemrefine.errors import ConfigError

    with pytest.raises(ConfigError, match="strict_scf"):
        PyscfOptions.from_raw({"basis": "def2-svp", "xc": "pbe", "strict_scf": False})
