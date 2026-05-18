"""Tests for the MLFF engine package.

The expensive bits (real MACE/UMA inference) are out of scope here.
What we verify:

* Engines register correctly.
* ``MlffCalculator`` rejects unknown backends and reads the right
  setup branch for each backend's task_name pattern (without actually
  loading a model — that path is exercised in the ``mlff`` extra's
  integration tests).
* ``MlffEngine`` produces the right ``%method`` block and SLURM
  ``run_block`` content.
* Placeholder modules raise ``NotImplementedError`` with a TODO hint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines.base import ENGINES, get_engine
from chemrefine.engines.mlff.calculator import MlffCalculator
from chemrefine.state import PipelineState, StepContext, Structure

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_mlff_engines_are_registered():
    assert "mlff" in ENGINES
    assert "mlff-direct" in ENGINES


def test_mlff_engine_supports_nms_is_false():
    assert get_engine("mlff").supports_nms is False


def test_mlff_direct_engine_supports_nms_is_false():
    assert get_engine("mlff-direct").supports_nms is False


# ---------------------------------------------------------------------------
# MlffCalculator backend dispatch
# ---------------------------------------------------------------------------


def test_mlff_calculator_unknown_backend_raises():
    with pytest.raises(ValueError):
        MlffCalculator(model_name="unknown_model", task_name="not_a_real_task")


def test_mlff_calculator_dispatches_mace_off():
    with patch.object(MlffCalculator, "_build_mace", return_value="MACE_OFF_CALC") as mock:
        calc = MlffCalculator(model_name="medium", task_name="mace_off")
    mock.assert_called_once()
    assert calc.calculator == "MACE_OFF_CALC"


def test_mlff_calculator_dispatches_mace_mp():
    with patch.object(MlffCalculator, "_build_mace", return_value="MACE_MP_CALC") as mock:
        MlffCalculator(model_name="medium", task_name="mace_mp")
    mock.assert_called_once()


def test_mlff_calculator_dispatches_fairchem_for_omol():
    with patch.object(
        MlffCalculator, "_build_fairchem", return_value="FAIRCHEM_CALC"
    ) as mock:
        MlffCalculator(model_name="uma-s-1", task_name="omol")
    mock.assert_called_once()


def test_mlff_calculator_dispatches_sevenn_by_model_name():
    with patch.object(MlffCalculator, "_build_sevenn", return_value="SEVENN_CALC") as mock:
        MlffCalculator(model_name="sevenn-tiny", task_name="custom")
    mock.assert_called_once()


def test_mlff_calculator_dispatches_orb_by_model_name():
    with patch.object(MlffCalculator, "_build_orb", return_value="ORB_CALC") as mock:
        MlffCalculator(model_name="orb-d3", task_name="custom")
    mock.assert_called_once()


def test_mlff_calculator_custom_model_path_picks_custom_mace(tmp_path: Path):
    """When ``model_path`` is given, the custom-MACE branch wins."""
    model_file = tmp_path / "fake.model"
    model_file.touch()
    with patch.object(
        MlffCalculator, "_build_custom_mace", return_value="CUSTOM_MACE_CALC"
    ) as mock:
        MlffCalculator(
            model_name="ignored", task_name="ignored", model_path=str(model_file)
        )
    mock.assert_called_once()


def test_mlff_calculator_custom_model_path_missing_raises(tmp_path: Path):
    """The custom-MACE branch should validate that the model file exists."""
    missing = tmp_path / "does_not_exist.model"
    with pytest.raises(FileNotFoundError):
        MlffCalculator(
            model_name="ignored", task_name="mace_off", model_path=str(missing)
        )


# ---------------------------------------------------------------------------
# MlffEngine — ORCA-driven mode
# ---------------------------------------------------------------------------


def _mlff_ctx(tmp_path: Path, **option_overrides) -> StepContext:
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "step1.inp").write_text("! B3LYP def2-SVP\n", encoding="utf-8")
    (template_dir / "cpu.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --partition=normal\n",
        encoding="utf-8",
    )
    options = {"model_name": "uma-s-1", "task_name": "omol", "device": "cuda"}
    options.update(option_overrides)
    step_cfg = StepConfig(
        step=1,
        engine="mlff",
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


def test_mlff_extra_blocks_contains_progext_pointing_to_wrapper(tmp_path: Path):
    engine = get_engine("mlff")
    ctx = _mlff_ctx(tmp_path)
    extra = engine._extra_blocks(ctx)
    assert "%method" in extra
    assert "ProgExt" in extra
    assert "mlff_extopt.sh" in extra
    assert "Ext_Params" in extra
    assert "127.0.0.1:8888" in extra


def test_mlff_custom_bind_address_round_trips(tmp_path: Path):
    engine = get_engine("mlff")
    ctx = _mlff_ctx(tmp_path, bind="10.0.0.5:5000")
    assert "10.0.0.5:5000" in engine._extra_blocks(ctx)


def test_mlff_run_block_starts_and_stops_server(tmp_path: Path):
    engine = get_engine("mlff")
    ctx = _mlff_ctx(tmp_path)
    run_block = engine._run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "python -m chemrefine.engines.mlff.server" in run_block
    assert "--model" in run_block
    assert "--bind" in run_block
    assert "SERVER_PID=$!" in run_block
    assert "kill $SERVER_PID" in run_block
    assert ctx.orca_executable in run_block


def test_mlff_prepare_writes_inp_with_method_block(tmp_path: Path):
    engine = get_engine("mlff")
    ctx = _mlff_ctx(tmp_path)
    inputs = engine.prepare(ctx)
    inp_text = inputs.files[0][0].read_text()
    assert "%method" in inp_text
    assert "ProgExt" in inp_text


# ---------------------------------------------------------------------------
# MlffDirectEngine — in-process scoring
# ---------------------------------------------------------------------------


def test_mlff_direct_round_trip_with_mocked_calculator(tmp_path: Path):
    engine = get_engine("mlff-direct")
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    step_cfg = StepConfig(
        step=1,
        engine="mlff-direct",
        operation="opt_sp",
        options={"model_name": "uma-s-1", "task_name": "omol"},
    )
    seeds = (
        Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])),
        Structure(id="1", atoms=Atoms("H2", positions=[[0, 0, 0], [0.80, 0, 0]])),
    )
    ctx = StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=template_dir,
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(structures=seeds),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        orca_executable="orca",
    )
    with (
        patch.object(MlffCalculator, "_build", return_value=None),
        patch.object(MlffCalculator, "single_point", side_effect=[(-1.5, []), (-2.0, [])]),
    ):
        inputs = engine.prepare(ctx)
        engine.submit(inputs, ctx)
        results = engine.parse(inputs, ctx)
    assert [s.id for s in results.structures] == ["0", "1"]
    # Energies should be sorted in eV → Hartree (negative)
    assert all(s.energy_hartree is not None and s.energy_hartree < 0 for s in results.structures)


def test_mlff_direct_does_not_support_nms(tmp_path: Path):
    engine = get_engine("mlff-direct")
    from chemrefine.state import StepResults

    with pytest.raises(NotImplementedError):
        engine.normal_mode_sample(StepResults(structures=()), ctx=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Placeholders — trainer / server / client
# ---------------------------------------------------------------------------


def test_trainer_placeholder_raises_with_todo():
    from chemrefine.engines.mlff.trainer import run_training
    from chemrefine.state import StepResults

    with pytest.raises(NotImplementedError):
        run_training(StepResults(structures=()), ctx=None)  # type: ignore[arg-type]


def test_server_placeholder_raises():
    from chemrefine.engines.mlff.server import main as server_main

    with pytest.raises(NotImplementedError):
        server_main()


def test_client_placeholder_raises():
    from chemrefine.engines.mlff.client import submit_calculation

    with pytest.raises(NotImplementedError):
        submit_calculation(
            server_url="x",
            atom_types=[],
            coordinates=[],
            charge=0,
            mult=1,
            dograd=False,
            nthreads=1,
        )
