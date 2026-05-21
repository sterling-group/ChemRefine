"""Tests for the PySCF engine package.

The PySCF-side surface is :class:`PyscfEngine` (which uses the shared
``_extopt.server``) and :class:`PyscfExtOptCalculator` (which wraps
:mod:`chemrefine.engines.pyscf._runtime` for the SCF + gradient call).
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines.base import ENGINES, get_engine
from chemrefine.state import PipelineState, StepContext, Structure

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_pyscf_engines_are_registered():
    assert "pyscf" in ENGINES
    assert "pyscf-direct" in ENGINES


def test_pyscf_engine_supports_nms_is_false():
    assert get_engine("pyscf").supports_nms is False


# ---------------------------------------------------------------------------
# PyscfEngine — ORCA-driven mode
# ---------------------------------------------------------------------------


def _pyscf_ctx(tmp_path: Path, **option_overrides) -> StepContext:
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "step1.inp").write_text("! HF def2-SVP\n", encoding="utf-8")
    (template_dir / "cpu.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --partition=normal\n", encoding="utf-8"
    )
    options = {"method": "dft", "xc": "pbe", "basis": "def2-svp"}
    options.update(option_overrides)
    step_cfg = StepConfig(
        step=1,
        engine="pyscf",
        operation="opt_sp",
        options=options,
    )
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=template_dir,
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(structures=(seed,)),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        orca_executable="orca",
    )


def test_pyscf_extra_blocks_carries_method_settings(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path, basis="cc-pvdz", xc="b3lyp")
    extra = engine._extra_blocks(ctx)
    assert "%method" in extra
    assert "ProgExt" in extra
    assert "pyscf_extopt.sh" in extra
    assert "--basis cc-pvdz" in extra
    assert "--xc b3lyp" in extra


def test_pyscf_extra_blocks_includes_gpu_and_df_flags(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path, df=True, gpu=True)
    extra = engine._extra_blocks(ctx)
    assert "--df" in extra
    assert "--gpu" in extra


def test_pyscf_run_block_starts_shared_extopt_server(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path)
    run_block = engine._run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "python -m chemrefine.engines._extopt.server" in run_block
    assert "--backend pyscf" in run_block
    assert "--method dft" in run_block
    assert "--xc pbe" in run_block
    assert "--basis def2-svp" in run_block


def test_pyscf_run_block_emits_gpu_and_df_when_set(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path, gpu=True, df=True)
    run_block = engine._run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "--df" in run_block
    assert "--gpu" in run_block


def test_pyscf_run_block_omits_optional_flags_when_unset(tmp_path: Path):
    """Each optional flag is gated on its option being set."""
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path, method="", xc=None, basis=None, df=False, gpu=False)
    run_block = engine._run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "--method " not in run_block
    assert "--xc " not in run_block
    assert "--basis " not in run_block
    # Bool flags absent
    assert " --df" not in run_block
    assert " --gpu" not in run_block


def test_pyscf_run_block_includes_readiness_loop(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path)
    run_block = engine._run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "/healthz" in run_block
    assert "trap _on_extopt_exit EXIT INT TERM" in run_block


def test_pyscf_prepare_writes_inp_with_method_block(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path)
    inputs = engine.prepare(ctx)
    inp_text = inputs.files[0][0].read_text()
    assert "%method" in inp_text
    assert "ProgExt" in inp_text
    assert "pyscf_extopt.sh" in inp_text


# ---------------------------------------------------------------------------
# PyscfExtOptCalculator — full coverage lives in tests/test_engines_pyscf_extopt_calc.py
# ---------------------------------------------------------------------------


def test_pyscf_extopt_calculator_from_args_round_trips():
    from chemrefine.engines._extopt.server import parse_args
    from chemrefine.engines.pyscf.extopt_calc import PyscfExtOptCalculator

    args = parse_args([
        "--backend", "pyscf", "--method", "hf",
        "--xc", "b3lyp", "--basis", "cc-pvdz", "--df", "--gpu",
    ])
    calc = PyscfExtOptCalculator.from_args(args)
    assert calc.method == "hf"
    assert calc.xc == "b3lyp"
    assert calc.basis == "cc-pvdz"
    assert calc.df is True
    assert calc.gpu is True


# Direct engine body coverage lives in ``tests/test_engines_pyscf_direct.py``.
