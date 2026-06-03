"""Tests for the MLFF engine package.

Coverage:

* Engines register correctly (``mlff`` direct + ``mlff-extopt``).
* ``build_calculator`` / ``MlffCalculator`` dispatch through the
  registry pattern (no god-class with ``_build_*`` methods anymore).
* ``MlffExtOptEngine`` produces the right ``%method`` block and SLURM
  ``run_block`` content (dynamic port + readiness loop + trap).
* ``MlffExtOptCalculator`` adapts the ASE calculator to the shared
  ExtOpt server's ``ComputeBackend`` contract.
* ``MlffEngine`` (direct, template-driven) renders one ``.py`` per
  structure, runs the local-bash fallback, and parses the JSON the
  appended footer wrote.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines.base import ENGINES, get_engine
from chemrefine.engines.mlff import calculator as mlff_calculator
from chemrefine.engines.mlff.calculator import MlffCalculator, build_calculator
from chemrefine.errors import OutputParseError
from chemrefine.state import PipelineState, StepContext, StepResults, Structure

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_mlff_engines_are_registered():
    assert "mlff" in ENGINES
    assert "mlff-extopt" in ENGINES


def test_mlff_engine_supports_nms_is_false():
    assert get_engine("mlff").supports_nms is False


def test_mlff_extopt_engine_supports_nms_is_false():
    assert get_engine("mlff-extopt").supports_nms is False


# ---------------------------------------------------------------------------
# Backend registry dispatch (replaces the old _build_* god-class branches)
# ---------------------------------------------------------------------------


def test_build_calculator_unknown_backend_raises():
    with pytest.raises(ValueError, match="unsupported MLFF backend"):
        build_calculator(task_name="not_a_real_task", model_name="unknown_model")


def test_build_calculator_dispatches_mace_off():
    with patch.dict(
        mlff_calculator._BACKEND_BUILDERS,
        {"mace_off": lambda **_kw: "MACE_OFF_CALC"},
        clear=False,
    ):
        result = build_calculator(task_name="mace_off", model_name="medium")
    assert result == "MACE_OFF_CALC"


def test_build_calculator_dispatches_mace_mp():
    with patch.dict(
        mlff_calculator._BACKEND_BUILDERS,
        {"mace_mp": lambda **_kw: "MACE_MP_CALC"},
        clear=False,
    ):
        result = build_calculator(task_name="mace_mp", model_name="medium")
    assert result == "MACE_MP_CALC"


def test_build_calculator_routes_fairchem_family_to_omol():
    """omol / omat / odac / uma / fairchem all dispatch to the omol builder."""
    captured = []
    with patch.dict(
        mlff_calculator._BACKEND_BUILDERS,
        {"omol": lambda **kw: captured.append(kw) or "FAIRCHEM_CALC"},
        clear=False,
    ):
        for task in ("omol", "omat", "odac", "uma-foo", "fairchem-bar"):
            assert build_calculator(task_name=task, model_name="uma-s-1") == "FAIRCHEM_CALC"
    assert len(captured) == 5


def test_build_calculator_routes_sevenn_by_model_name():
    with patch.dict(
        mlff_calculator._BACKEND_BUILDERS,
        {"sevenn": lambda **_kw: "SEVENN_CALC"},
        clear=False,
    ):
        result = build_calculator(task_name="custom", model_name="sevenn-tiny")
    assert result == "SEVENN_CALC"


def test_build_calculator_routes_chgnet_for_chgnet_task():
    with patch.dict(
        mlff_calculator._BACKEND_BUILDERS,
        {"chgnet": lambda **_kw: "CHGNET_CALC"},
        clear=False,
    ):
        result = build_calculator(task_name="chgnet", model_name="ignored")
    assert result == "CHGNET_CALC"


def test_build_calculator_routes_orb_by_model_name():
    with patch.dict(
        mlff_calculator._BACKEND_BUILDERS,
        {"orb": lambda **_kw: "ORB_CALC"},
        clear=False,
    ):
        result = build_calculator(task_name="custom", model_name="orb-d3")
    assert result == "ORB_CALC"


def test_build_calculator_routes_custom_mace_when_model_path_given(tmp_path: Path):
    """If the caller supplies a ``model_path``, the custom_mace builder wins."""
    model_file = tmp_path / "fake.model"
    model_file.touch()
    seen = []
    with patch.dict(
        mlff_calculator._BACKEND_BUILDERS,
        {"custom_mace": lambda **kw: seen.append(kw) or "CUSTOM_MACE_CALC"},
        clear=False,
    ):
        result = build_calculator(
            task_name="mace_off",  # would normally pick mace_off
            model_name="ignored",
            model_path=str(model_file),
        )
    assert result == "CUSTOM_MACE_CALC"
    assert seen[0]["model_path"] == str(model_file)


def test_build_calculator_raises_when_custom_mace_unregistered(tmp_path: Path):
    """Defensive branch: if someone pops ``custom_mace`` and then passes a
    ``model_path``, we surface a clear ValueError instead of a KeyError."""
    model_file = tmp_path / "fake.model"
    model_file.touch()
    saved = mlff_calculator._BACKEND_BUILDERS.pop("custom_mace", None)
    try:
        with pytest.raises(ValueError, match="custom_mace backend not registered"):
            build_calculator(
                task_name="ignored", model_name="x", model_path=str(model_file)
            )
    finally:
        if saved is not None:
            mlff_calculator._BACKEND_BUILDERS["custom_mace"] = saved


def test_register_backend_appends_to_registry():
    """A new ``@register_backend`` adds the function to the registry."""
    mlff_calculator.register_backend("test_new_backend")(lambda **_kw: "NEW")
    try:
        assert "test_new_backend" in mlff_calculator._BACKEND_BUILDERS
        assert build_calculator(task_name="test_new_backend", model_name="x") == "NEW"
    finally:
        mlff_calculator._BACKEND_BUILDERS.pop("test_new_backend", None)


def test_mlff_calculator_wrapper_routes_through_build_calculator():
    """``MlffCalculator`` is a thin alias — exercises the registry indirectly."""
    with patch.dict(
        mlff_calculator._BACKEND_BUILDERS,
        {"mace_off": lambda **_kw: "WRAP_CALC"},
        clear=False,
    ):
        calc = MlffCalculator(model_name="medium", task_name="mace_off")
    assert calc.calculator == "WRAP_CALC"


# ---------------------------------------------------------------------------
# MlffExtOptEngine — ORCA-driven mode (ExtOpt server lifecycle)
# ---------------------------------------------------------------------------


def _mlff_extopt_ctx(tmp_path: Path, **option_overrides) -> StepContext:
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
        engine="mlff-extopt",
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
        executables={},
    )


def test_mlff_extopt_extra_blocks_contains_progext_pointing_to_wrapper(tmp_path: Path):
    engine = get_engine("mlff-extopt")
    ctx = _mlff_extopt_ctx(tmp_path)
    extra = engine._extra_blocks(ctx)
    assert "%method" in extra
    assert "ProgExt" in extra
    assert "mlff_extopt.sh" in extra


def test_mlff_extopt_run_block_starts_shared_extopt_server(tmp_path: Path):
    engine = get_engine("mlff-extopt")
    ctx = _mlff_extopt_ctx(tmp_path)
    run_block = engine._run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "python -m chemrefine.engines._backend_server.server" in run_block
    assert "--backend mlff" in run_block
    assert "--bind 127.0.0.1:0" in run_block
    assert "--model" in run_block
    assert ctx.executables.get("orca", "orca") in run_block


def test_mlff_extopt_run_block_includes_readiness_loop_and_trap(tmp_path: Path):
    engine = get_engine("mlff-extopt")
    ctx = _mlff_extopt_ctx(tmp_path)
    run_block = engine._run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "sleep 10" not in run_block
    assert "/healthz" in run_block
    assert "ps -p" in run_block
    assert "trap _on_extopt_exit EXIT INT TERM" in run_block
    assert "kill -TERM" in run_block


def test_mlff_extopt_prepare_writes_inp_with_method_block(tmp_path: Path):
    engine = get_engine("mlff-extopt")
    ctx = _mlff_extopt_ctx(tmp_path)
    inputs = engine.prepare(ctx)
    inp_text = inputs.files[0][0].read_text()
    assert "%method" in inp_text
    assert "ProgExt" in inp_text


def test_mlff_extopt_prepare_materializes_executable_wrapper(tmp_path: Path):
    """The ``ProgExt`` wrapper the ``.inp`` points at must actually be written."""
    import os

    engine = get_engine("mlff-extopt")
    ctx = _mlff_extopt_ctx(tmp_path)
    engine.prepare(ctx)

    wrapper = engine._wrapper_path(ctx)
    assert wrapper.is_file()
    assert os.access(wrapper, os.X_OK)

    text = wrapper.read_text()
    assert "chemrefine.engines.orca.extopt.bridge" in text
    assert "--backend mlff" in text


# ---------------------------------------------------------------------------
# MlffExtOptCalculator — ExtOpt-side adapter
# ---------------------------------------------------------------------------


def test_mlff_extopt_calculator_from_args_builds_instance():
    """``from_args`` should consume the shared server CLI namespace."""
    from chemrefine.engines._backend_server.server import parse_args
    from chemrefine.engines.mlff.extopt_calc import MlffExtOptCalculator

    args = parse_args(["--backend", "mlff", "--model", "medium", "--task-name", "mace_off"])
    with patch.dict(
        mlff_calculator._BACKEND_BUILDERS,
        {"mace_off": lambda **_kw: "MACE_CALC"},
        clear=False,
    ):
        calc = MlffExtOptCalculator.from_args(args)
    assert calc.name == "mlff"


def test_mlff_add_cli_args_registers_mlff_flags_with_pydantic_defaults():
    """Defaults must mirror :class:`MlffOptions` (single source of truth)."""
    import argparse

    from chemrefine.engines.mlff.extopt_calc import MlffExtOptCalculator
    from chemrefine.engines.mlff.options import MlffOptions

    parser = argparse.ArgumentParser()
    MlffExtOptCalculator.add_cli_args(parser)
    args = parser.parse_args([])
    defaults = MlffOptions()
    assert args.model is None
    assert args.task_name == defaults.task_name
    assert args.device == defaults.device
    assert args.model_path is None


def test_mlff_settings_from_args_returns_empty_dict():
    """MLFF has no per-call client knobs to forward today."""
    import argparse

    from chemrefine.engines.mlff.extopt_calc import MlffExtOptCalculator

    parser = argparse.ArgumentParser()
    MlffExtOptCalculator.add_cli_args(parser)
    args = parser.parse_args(["--model", "medium"])
    assert MlffExtOptCalculator.settings_from_args(args) == {}


def test_mlff_server_cli_from_options_emits_all_set_flags():
    from chemrefine.engines.mlff.extopt_calc import MlffExtOptCalculator

    tokens = MlffExtOptCalculator.server_cli_from_options(
        {
            "model_name": "medium",
            "task_name": "mace_off",
            "device": "cpu",
            "model_path": "/tmp/ckpt.model",
        }
    )
    assert tokens == [
        "--model", "medium",
        "--task-name", "mace_off",
        "--device", "cpu",
        "--model-path", "/tmp/ckpt.model",
    ]


def test_mlff_server_cli_from_options_omits_falsy_values():
    from chemrefine.engines.mlff.extopt_calc import MlffExtOptCalculator

    tokens = MlffExtOptCalculator.server_cli_from_options(
        {"model_name": "", "task_name": None, "device": None, "model_path": None}
    )
    assert tokens == []


def test_mlff_extopt_calculator_calc_converts_units():
    """``calc`` should return Hartree / Hartree-per-Bohr regardless of ASE eV units."""
    import numpy as np

    from chemrefine.engines._backend_server.base import CalculationData
    from chemrefine.engines.mlff.extopt_calc import MlffExtOptCalculator
    from chemrefine.quantities import BOHR_TO_ANGSTROM, HARTREE_TO_EV

    with (
        patch.dict(
            mlff_calculator._BACKEND_BUILDERS,
            {"mace_off": lambda **_kw: object()},  # sentinel calculator
            clear=False,
        ),
        # 1 eV/Å on x (the gradient already in eV/Å)
        patch.object(
            MlffCalculator,
            "single_point",
            return_value=(HARTREE_TO_EV, [[1.0, 0.0, 0.0]]),
        ),
    ):
        calc = MlffExtOptCalculator(
            model_name="medium", task_name="mace_off", device="cpu"
        )
        data = CalculationData(
            symbols=("H",),
            positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
            charge=0,
            multiplicity=1,
            nthreads=1,
            dograd=True,
            settings={},
        )
        energy_h, gradient = calc.calc(data)
    assert energy_h == pytest.approx(1.0, rel=1e-12)
    assert gradient[0][0] == pytest.approx(BOHR_TO_ANGSTROM / HARTREE_TO_EV, rel=1e-12)


# ---------------------------------------------------------------------------
# MlffEngine — template-driven direct mode
# ---------------------------------------------------------------------------

_FAKE_MLFF_TEMPLATE = """\
'''Fake MLFF template used by the direct-engine tests.

Reads no MLFF library; just declares the result variables ChemRefine's
appended footer harvests. Lets the lifecycle (prepare → submit → parse)
be exercised end-to-end without a real MLFF install.
'''
energy_hartree = -2.0 + 0.01 * $CHARGE
gradient_hartree_per_bohr = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]]
"""


def _write_mlff_templates(tmp_path: Path) -> Path:
    (tmp_path / "step1.py").write_text(_FAKE_MLFF_TEMPLATE, encoding="utf-8")
    (tmp_path / "cpu.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --partition=normal\n",
        encoding="utf-8",
    )
    return tmp_path


def _mlff_direct_ctx(
    tmp_path: Path, structures: tuple[Structure, ...], **overrides
) -> StepContext:
    _write_mlff_templates(tmp_path)
    step_cfg = StepConfig(
        step=1,
        name="screen",
        engine="mlff",
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


def test_mlff_direct_prepare_renders_one_py_and_xyz_per_structure(tmp_path: Path):
    ctx = _mlff_direct_ctx(tmp_path, structures=(_seed("0"), _seed("1")))
    engine = get_engine("mlff")
    inputs = engine.prepare(ctx)
    assert len(inputs.files) == 2
    for script_path, output_json, sid in inputs.files:
        assert script_path.name == f"step1_structure_{sid}.py"
        assert output_json.name == f"step1_structure_{sid}.json"
        assert script_path.with_suffix(".xyz").is_file()
        rendered = script_path.read_text()
        assert "$XYZ_PATH" not in rendered
        assert "$CHARGE" not in rendered
        assert f"with open('{output_json.name}', \"w\")" in rendered


def test_mlff_direct_prepare_missing_template_raises(tmp_path: Path):
    ctx = _mlff_direct_ctx(tmp_path, structures=(_seed(),))
    (ctx.template_dir / "step1.py").unlink()
    engine = get_engine("mlff")
    with pytest.raises(FileNotFoundError, match="MLFF template not found"):
        engine.prepare(ctx)


def test_mlff_direct_submit_runs_template_locally_when_no_sbatch(tmp_path: Path):
    """No sbatch → slurm.submit falls back to local bash; the template runs synchronously."""
    ctx = _mlff_direct_ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("mlff")
    inputs = engine.prepare(ctx)
    with patch("chemrefine.slurm.shutil.which", return_value=None):
        from chemrefine.state import JobBatch
        batch = engine.submit(inputs, ctx)
    assert isinstance(batch, JobBatch)
    assert all(jid.startswith("local-") for jid in batch.jobs.values())
    output_json = inputs.files[0][1]
    assert output_json.is_file()
    data = json.loads(output_json.read_text())
    assert data["energy_hartree"] == pytest.approx(-2.0)


def test_mlff_direct_parse_returns_structure_with_energy_and_forces(tmp_path: Path):
    ctx = _mlff_direct_ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("mlff")
    inputs = engine.prepare(ctx)
    with patch("chemrefine.slurm.shutil.which", return_value=None):
        engine.submit(inputs, ctx)
    results = engine.parse(inputs, ctx)
    assert len(results.structures) == 1
    s = results.structures[0]
    assert s.id == "0"
    assert s.energy_hartree == pytest.approx(-2.0)
    assert s.forces_ev_per_a is not None
    assert s.forces_ev_per_a.shape == (2, 3)


def test_mlff_direct_parse_raises_when_output_missing(tmp_path: Path):
    ctx = _mlff_direct_ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("mlff")
    inputs = engine.prepare(ctx)
    with pytest.raises(OutputParseError, match="output not found"):
        engine.parse(inputs, ctx)


def test_mlff_direct_parse_raises_when_output_not_json(tmp_path: Path):
    ctx = _mlff_direct_ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("mlff")
    inputs = engine.prepare(ctx)
    output_json = inputs.files[0][1]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text("not json", encoding="utf-8")
    with pytest.raises(OutputParseError, match="not valid JSON"):
        engine.parse(inputs, ctx)


def test_mlff_direct_parse_raises_when_energy_missing(tmp_path: Path):
    ctx = _mlff_direct_ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("mlff")
    inputs = engine.prepare(ctx)
    output_json = inputs.files[0][1]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"oops": 0.0}), encoding="utf-8")
    with pytest.raises(OutputParseError, match="energy_hartree"):
        engine.parse(inputs, ctx)


def test_mlff_direct_parse_uses_positions_when_present(tmp_path: Path):
    ctx = _mlff_direct_ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("mlff")
    inputs = engine.prepare(ctx)
    out_json = inputs.files[0][1]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "energy_hartree": -3.0,
                "positions_angstrom": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]],
            }
        ),
        encoding="utf-8",
    )
    import numpy as np

    results = engine.parse(inputs, ctx)
    np.testing.assert_allclose(
        results.structures[0].atoms.get_positions(),
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]],
    )


def test_mlff_direct_submit_missing_slurm_header_raises(tmp_path: Path):
    ctx = _mlff_direct_ctx(tmp_path, structures=(_seed(),))
    (ctx.template_dir / "cpu.slurm.header").unlink()
    engine = get_engine("mlff")
    inputs = engine.prepare(ctx)
    with pytest.raises(FileNotFoundError, match="SLURM header"):
        engine.submit(inputs, ctx)


def test_mlff_direct_submit_respects_cores_option(tmp_path: Path):
    ctx = _mlff_direct_ctx(tmp_path, structures=(_seed(),), options={"cores": 2}, max_cores=4)
    engine = get_engine("mlff")
    inputs = engine.prepare(ctx)
    with patch("chemrefine.slurm.shutil.which", return_value=None):
        engine.submit(inputs, ctx)
    script_text = inputs.files[0][0].with_suffix(".slurm").read_text()
    assert "#SBATCH --ntasks=2" in script_text


def test_mlff_direct_does_not_support_nms(tmp_path: Path):
    engine = get_engine("mlff")
    with pytest.raises(NotImplementedError, match="does not support normal-mode"):
        engine.normal_mode_sample(
            StepResults(structures=()), _mlff_direct_ctx(tmp_path, ())
        )


def test_mlff_direct_wait_is_noop():
    from chemrefine.state import JobBatch

    engine = get_engine("mlff")
    engine.wait(JobBatch(jobs={}))  # must not raise
