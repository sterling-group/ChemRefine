"""Tests for the PySCF engine package (ported from the unmerged ``origin/pyscf`` PR)."""

from __future__ import annotations

from pathlib import Path

import pytest
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
# PyscfEngine — ORCA-driven mode (verified, no real PySCF needed)
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


def test_pyscf_extra_blocks_default_bind(tmp_path: Path):
    """Default bind for PySCF should be 127.0.0.1:8889 (distinct from MLFF)."""
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path)
    assert "127.0.0.1:8889" in engine._extra_blocks(ctx)


def test_pyscf_extra_blocks_includes_gpu_and_df_flags(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path, df=True, gpu=True)
    extra = engine._extra_blocks(ctx)
    assert "--df" in extra
    assert "--gpu" in extra


def test_pyscf_run_block_starts_pyscf_server(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path)
    run_block = engine._run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "python -m chemrefine.engines.pyscf.server" in run_block
    assert "--default-method dft" in run_block
    assert "--default-xc pbe" in run_block
    assert "--default-basis def2-svp" in run_block
    assert "SERVER_PID=$!" in run_block
    assert "kill $SERVER_PID" in run_block


def test_pyscf_run_block_emits_gpu_flag_when_set(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path, gpu=True, df=True)
    run_block = engine._run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "--default-df" in run_block
    assert "--default-gpu" in run_block


def test_pyscf_prepare_writes_inp_with_method_block(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path)
    inputs = engine.prepare(ctx)
    inp_text = inputs.files[0][0].read_text()
    assert "%method" in inp_text
    assert "ProgExt" in inp_text
    assert "pyscf_extopt.sh" in inp_text


# ---------------------------------------------------------------------------
# Placeholders raise NotImplementedError with TODO hint
# ---------------------------------------------------------------------------


def test_pyscf_direct_prepare_raises_with_todo():
    engine = get_engine("pyscf-direct")
    with pytest.raises(NotImplementedError):
        engine.prepare(ctx=None)  # type: ignore[arg-type]


def test_pyscf_server_main_placeholder_raises():
    from chemrefine.engines.pyscf.server import main as server_main

    with pytest.raises(NotImplementedError):
        server_main()


def test_pyscf_client_submit_placeholder_raises():
    from chemrefine.engines.pyscf.client import submit_calculation

    with pytest.raises(NotImplementedError):
        submit_calculation(
            server_url="x",
            atom_types=[],
            coordinates=[],
            charge=0,
            mult=1,
            dograd=False,
            nthreads=1,
            settings={},
        )
