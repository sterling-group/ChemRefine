"""Tests for the MLIP engine package.

Coverage:

* Engines register correctly (``mlip`` direct + ``mlip-extopt``).
* ``build_calculator`` / ``MlipCalculator`` dispatch through the
  registry pattern (no god-class with ``_build_*`` methods anymore).
* ``MlipExtOptEngine`` produces the right ``%method`` block and SLURM
  ``run_block`` content (dynamic port + readiness loop + trap).
* ``MlipExtOptCalculator`` adapts the ASE calculator to the shared
  ExtOpt server's ``ComputeBackend`` contract.
* ``MlipEngine`` (direct, template-driven) renders one ``.py`` per
  structure, runs the local-bash fallback, and parses the JSON the
  appended footer wrote.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines import _execution as submit
from chemrefine.engines.api import ENGINES, NmsCapableEngine, get_engine
from chemrefine.engines.mlip import calculator as mlip_calculator
from chemrefine.engines.mlip.calculator import BackendSpec, MlipCalculator, build_calculator
from chemrefine.errors import ConfigError, OutputParseError
from chemrefine.state import PipelineState, StepContext, Structure


def _spec(fn) -> BackendSpec:
    """Wrap a fake builder in a ``BackendSpec`` with placeholder packaging metadata."""
    return BackendSpec(fn, extra="mlip-test", package="test-pkg", import_name="test_mod")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _recording_builder(seen, result):
    """A backend builder that records the kwargs it was called with."""

    def build(**kwargs):
        seen.append(kwargs)
        return result

    return build


def test_mlip_engines_are_registered():
    assert "mlip" in ENGINES
    assert "mlip-extopt" in ENGINES


def test_mlff_engine_keys_normalize_to_mlip_at_config_layer():
    """Legacy ``mlff*`` engine keys are rewritten to ``mlip*`` by the config
    normalizer (the single place), so the registry only ever sees canonical names."""
    from chemrefine.config import Config

    cfg = Config(
        template_dir="./t",
        steps=[
            {"step": 1, "engine": "mlff", "operation": "opt_sp"},
            {"step": 2, "engine": "mlff-extopt", "operation": "opt_sp"},
            {"step": 3, "engine": "mlff-train", "operation": "mlip_train"},
        ],
    )
    assert [s.engine for s in cfg.steps] == ["mlip", "mlip-extopt", "mlip-train"]
    assert get_engine(cfg.steps[1].engine).name == "mlip-extopt"


def test_mlip_direct_engine_is_not_nms_capable():
    assert not isinstance(get_engine("mlip"), NmsCapableEngine)


def test_mlip_extopt_engine_is_nms_capable():
    """ExtOpt inherits ORCA's NMS hooks — ORCA computes the Hessian over MLIP gradients."""
    assert isinstance(get_engine("mlip-extopt"), NmsCapableEngine)


# ---------------------------------------------------------------------------
# Provisioning capability — backend_requirement derives the env from the options
# ---------------------------------------------------------------------------


def test_mlip_engines_are_provisionable():
    """Both MLIP engines expose ``backend_requirement`` (capability by Protocol)."""
    from chemrefine.engines.api import ProvisionableEngine

    for name in ("mlip", "mlip-extopt"):
        engine = get_engine(name)
        assert isinstance(engine, ProvisionableEngine)
        req = engine.backend_requirement({"task_name": "mace_off"})
        assert (req.extra, req.import_name) == ("mlip-mace", "mace")


def test_orca_and_fake_are_not_provisionable():
    """Engines without a Python backend don't satisfy the Protocol."""
    from chemrefine.engines.api import ProvisionableEngine

    assert not isinstance(get_engine("orca"), ProvisionableEngine)
    assert not isinstance(get_engine("fake"), ProvisionableEngine)


def test_requirement_from_options_maps_the_task_family():
    """task/task_name aliases resolve; the default (omol) is the FAIRChem env."""
    from chemrefine.engines.mlip.calculator import requirement_from_options

    assert requirement_from_options({"task_name": "sevenn"}).extra == "mlip-sevenn"
    assert requirement_from_options({"task": "chgnet"}).extra == "mlip-chgnet"
    assert requirement_from_options(None).extra == "mlip-fairchem"


def test_requirement_from_options_model_path_routes_to_mace():
    """A ``model_path`` selects custom_mace — same rule as ``build_calculator``."""
    from chemrefine.engines.mlip.calculator import requirement_from_options

    req = requirement_from_options({"task_name": "omol", "model_path": "/some/ckpt.model"})
    assert (req.extra, req.import_name) == ("mlip-mace", "mace")


# ---------------------------------------------------------------------------
# Backend registry dispatch — task_name keys the registry, model_name = weights
# (the per-builder behaviour is covered in test_engines_mlip_calculator.py)
# ---------------------------------------------------------------------------


def test_build_calculator_unknown_backend_raises():
    with pytest.raises(ValueError, match="unsupported MLIP backend"):
        build_calculator(task_name="not_a_real_task", model_name="x")


def test_build_calculator_dispatches_by_task_name():
    """``task_name`` keys the registry and forwards task_name/model_name."""
    seen: list[dict] = []
    with patch.dict(
        mlip_calculator._BACKENDS,
        {"mace_off": _spec(_recording_builder(seen, "MACE_OFF_CALC"))},
        clear=False,
    ):
        result = build_calculator(task_name="mace_off", model_name="medium")
    assert result == "MACE_OFF_CALC"
    assert seen[0]["task_name"] == "mace_off"
    assert seen[0]["model_name"] == "medium"


def test_build_calculator_routes_custom_mace_when_model_path_given(tmp_path: Path):
    """A ``model_path`` selects the custom_mace builder regardless of task_name."""
    model_file = tmp_path / "fake.model"
    model_file.touch()
    seen: list[dict] = []
    with patch.dict(
        mlip_calculator._BACKENDS,
        {"custom_mace": _spec(_recording_builder(seen, "CUSTOM_MACE_CALC"))},
        clear=False,
    ):
        result = build_calculator(
            task_name="mace_off", model_name="ignored", model_path=str(model_file)
        )
    assert result == "CUSTOM_MACE_CALC"
    assert seen[0]["model_path"] == str(model_file)


def test_register_backend_appends_to_registry():
    """A new ``@register_backend`` is reachable by its ``task_name`` key with its metadata."""
    mlip_calculator.register_backend(
        "test_new_backend", extra="mlip-test", package="test-pkg", import_name="test_mod"
    )(lambda **_kw: "NEW")
    try:
        spec = mlip_calculator.backend_spec("test_new_backend")
        assert (spec.extra, spec.package, spec.import_name) == (
            "mlip-test",
            "test-pkg",
            "test_mod",
        )
        assert build_calculator(task_name="test_new_backend", model_name="x") == "NEW"
    finally:
        mlip_calculator._BACKENDS.pop("test_new_backend", None)


def test_mlip_calculator_wrapper_routes_through_build_calculator():
    """``MlipCalculator`` is a thin alias — exercises the registry indirectly."""
    with patch.dict(
        mlip_calculator._BACKENDS,
        {"mace_off": _spec(lambda **_kw: "WRAP_CALC")},
        clear=False,
    ):
        calc = MlipCalculator(task_name="mace_off", model_name="small")
    assert calc.calculator == "WRAP_CALC"


def test_mlip_options_aliases_resolve_to_canonical():
    """`model`/`size`→`model_name`, `task`→`task_name`; dump emits canonical keys."""
    from chemrefine.engines.mlip.options import MlipOptions

    a = MlipOptions(**{"model": "medium", "task": "mace_off"})
    assert (a.model_name, a.task_name) == ("medium", "mace_off")
    b = MlipOptions(**{"size": "small", "task_name": "mace_off"})
    assert (b.model_name, b.task_name) == ("small", "mace_off")
    dumped = a.model_dump()
    assert dumped["model_name"] == "medium" and dumped["task_name"] == "mace_off"


# ---------------------------------------------------------------------------
# MlipExtOptEngine — ORCA-driven mode (ExtOpt server lifecycle)
# ---------------------------------------------------------------------------


def _mlip_extopt_ctx(tmp_path: Path, **option_overrides) -> StepContext:
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "step1.inp").write_text("! B3LYP def2-SVP\n", encoding="utf-8")
    (template_dir / "cpu.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --partition=normal\n",
        encoding="utf-8",
    )
    options = {"model_name": "uma-s-1p2", "task_name": "omol", "device": "cuda"}
    options.update(option_overrides)
    step_cfg = StepConfig(
        step=1,
        engine="mlip-extopt",
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


def test_mlip_extopt_cuda_step_selects_cuda_header(tmp_path: Path):
    """device: cuda → the whole ORCA-driven job lands on a GPU node (cuda header)."""
    engine = get_engine("mlip-extopt")
    ctx = _mlip_extopt_ctx(tmp_path)  # device: cuda by default
    assert engine.gpus(ctx) == 1
    assert submit._header_name(engine, ctx) == "cuda.slurm.header"


def test_mlip_extopt_cpu_step_keeps_global_header(tmp_path: Path):
    engine = get_engine("mlip-extopt")
    ctx = _mlip_extopt_ctx(tmp_path, device="cpu")
    assert engine.gpus(ctx) == 0
    assert submit._header_name(engine, ctx) == ctx.slurm_template


def test_mlip_direct_cuda_step_selects_cuda_header(tmp_path: Path):
    engine = get_engine("mlip")
    ctx = _mlip_direct_ctx(tmp_path, structures=(_seed(),), options={"device": "cuda"})
    assert engine.gpus(ctx) == 1
    assert submit._header_name(engine, ctx) == "cuda.slurm.header"


def test_per_step_slurm_template_overrides_device_pick(tmp_path: Path):
    """An explicit per-step slurm_template wins over the device-driven cuda pick."""
    engine = get_engine("mlip-extopt")
    ctx0 = _mlip_extopt_ctx(tmp_path)  # device: cuda
    step_cfg = ctx0.step_cfg.model_copy(update={"slurm_template": "special.header"})
    ctx = replace(ctx0, step_cfg=step_cfg)
    assert engine.gpus(ctx) == 1  # still a GPU job
    assert submit._header_name(engine, ctx) == "special.header"


def test_local_gpu_jobs_get_distinct_cuda_visible_devices(tmp_path: Path, monkeypatch):
    """Two concurrent local CUDA jobs are pinned to distinct GPUs via CUDA_VISIBLE_DEVICES."""
    ctx = _mlip_direct_ctx(
        tmp_path,
        structures=(_seed("0"), _seed("1")),
        options={"device": "cuda", "cores": 1},
        max_cores=4,
    )
    ctx = replace(ctx, max_gpus=2)
    (ctx.template_dir / "cuda.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --gres=gpu:1\n", encoding="utf-8"
    )
    engine = get_engine("mlip")
    inputs = engine.prepare(ctx)

    captured: list[dict | None] = []

    def fake_submit(script_path, *, env=None, dispatch="auto"):
        captured.append(env)
        return f"local-{len(captured)}"

    state = {"polls": 0}

    def fake_finished(job_ids):
        state["polls"] += 1
        # One poll per wait_for_room; both jobs stay active through the submit
        # loop so the second must land on a different device than the first.
        return set(job_ids) if state["polls"] > 2 else set()

    monkeypatch.setattr("chemrefine.slurm.sbatch_available", lambda **k: False)
    monkeypatch.setattr("chemrefine.slurm.submit", fake_submit)
    monkeypatch.setattr("chemrefine.slurm.finished_jobs", fake_finished)

    engine.submit(inputs, ctx)
    devices = sorted(env["CUDA_VISIBLE_DEVICES"] for env in captured if env)
    assert devices == ["0", "1"]


def test_gpu_step_with_zero_gpu_budget_raises_config_error(tmp_path: Path, monkeypatch):
    """`max_gpus: 0` + a CUDA step is a config mistake — surface it as a
    ConfigError with its documented exit code, not the throttler's internal
    ValueError traceback."""
    from chemrefine.errors import ConfigError

    ctx = _mlip_direct_ctx(
        tmp_path,
        structures=(_seed("0"),),
        options={"device": "cuda", "cores": 1},
    )
    ctx = replace(ctx, max_gpus=0)
    (ctx.template_dir / "cuda.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --gres=gpu:1\n", encoding="utf-8"
    )
    engine = get_engine("mlip")
    inputs = engine.prepare(ctx)
    monkeypatch.setattr("chemrefine.slurm.sbatch_available", lambda **k: False)
    with pytest.raises(ConfigError, match="max_gpus"):
        engine.submit(inputs, ctx)


def test_mlip_extopt_extra_blocks_contains_progext_pointing_to_wrapper(tmp_path: Path):
    engine = get_engine("mlip-extopt")
    ctx = _mlip_extopt_ctx(tmp_path)
    extra = engine._extra_blocks(ctx)
    assert "%method" in extra
    assert "ProgExt" in extra
    assert "mlip_extopt.sh" in extra


def test_mlip_extopt_run_block_starts_shared_extopt_server(tmp_path: Path):
    engine = get_engine("mlip-extopt")
    ctx = _mlip_extopt_ctx(tmp_path)
    run_block = engine.run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "-m chemrefine.engines._backend_server.server" in run_block
    assert "--backend mlip" in run_block
    assert "--bind 127.0.0.1:0" in run_block
    assert "--model" in run_block
    assert ctx.executables.get("orca", "orca") in run_block


def test_mlip_extopt_run_block_has_a_readiness_loop_and_a_cleanup_hook(tmp_path: Path):
    engine = get_engine("mlip-extopt")
    ctx = _mlip_extopt_ctx(tmp_path)
    run_block = engine.run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "sleep 10" not in run_block
    assert "/healthz" in run_block
    assert "ps -p" in run_block
    # Teardown is a hook the surrounding script's EXIT trap calls, never a trap of our own:
    # bash keeps one handler per signal, so trapping EXIT here replaced the script's and took
    # the copy-back, the runlog footer and the scratch teardown with it.
    assert "_chemrefine_engine_cleanup()" in run_block
    assert "trap " not in run_block
    assert "kill -TERM" in run_block


def test_mlip_extopt_prepare_writes_inp_with_method_block(tmp_path: Path):
    engine = get_engine("mlip-extopt")
    ctx = _mlip_extopt_ctx(tmp_path)
    inputs = engine.prepare(ctx)
    inp_text = inputs.files[0][0].read_text()
    assert "%method" in inp_text
    assert "ProgExt" in inp_text


def test_mlip_extopt_prepare_materializes_executable_wrapper(tmp_path: Path):
    """The ``ProgExt`` wrapper the ``.inp`` points at must actually be written."""
    import os

    engine = get_engine("mlip-extopt")
    ctx = _mlip_extopt_ctx(tmp_path)
    engine.prepare(ctx)

    wrapper = engine._wrapper_path(ctx)
    assert wrapper.is_file()
    assert os.access(wrapper, os.X_OK)

    text = wrapper.read_text()
    assert "chemrefine.engines.orca.extopt.bridge" in text
    assert "--backend mlip" in text


# ---------------------------------------------------------------------------
# MlipExtOptCalculator — ExtOpt-side adapter
# ---------------------------------------------------------------------------


def test_mlip_extopt_calculator_from_args_builds_instance():
    """``from_args`` should consume the shared server CLI namespace."""
    from chemrefine.engines._backend_server.server import parse_args
    from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator

    args = parse_args(["--backend", "mlip", "--model", "small", "--task-name", "mace_off"])
    with patch.dict(
        mlip_calculator._BACKENDS,
        {"mace_off": _spec(lambda **_kw: "MACE_CALC")},
        clear=False,
    ):
        calc = MlipExtOptCalculator.from_args(args)
    assert calc.name == "mlip"


def test_mlip_add_cli_args_registers_mlip_flags_with_pydantic_defaults():
    """Defaults must mirror :class:`MlipOptions` (single source of truth)."""
    import argparse

    from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator
    from chemrefine.engines.mlip.options import MlipOptions

    parser = argparse.ArgumentParser()
    MlipExtOptCalculator.add_cli_args(parser)
    args = parser.parse_args([])
    defaults = MlipOptions()
    assert args.model == defaults.model_name
    assert args.task_name == defaults.task_name
    assert args.device == defaults.device
    assert args.model_path is None


def test_mlip_settings_from_args_returns_empty_dict():
    """MLIP has no per-call client knobs to forward today."""
    import argparse

    from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator

    parser = argparse.ArgumentParser()
    MlipExtOptCalculator.add_cli_args(parser)
    args = parser.parse_args(["--model", "medium"])
    assert MlipExtOptCalculator.settings_from_args(args) == {}


def test_mlip_server_cli_from_options_emits_all_set_flags():
    from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator

    tokens = MlipExtOptCalculator.server_cli_from_options(
        {
            "model_name": "small",
            "task_name": "mace_off",
            "device": "cpu",
            "model_path": "/tmp/ckpt.model",
        }
    )
    assert tokens == [
        "--model",
        "small",
        "--task-name",
        "mace_off",
        "--device",
        "cpu",
        "--model-path",
        "/tmp/ckpt.model",
    ]


def test_mlip_server_cli_from_options_omits_falsy_values():
    from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator

    tokens = MlipExtOptCalculator.server_cli_from_options(
        {"model_name": "", "task_name": None, "device": None, "model_path": None}
    )
    assert tokens == []


def _calc_data(charge: int = 0, multiplicity: int = 1):
    import numpy as np

    from chemrefine.engines._backend_server.base import CalculationData

    return CalculationData(
        symbols=("H",),
        positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
        charge=charge,
        multiplicity=multiplicity,
        nthreads=1,
        dograd=True,
        settings={},
    )


def test_mlip_extopt_calculator_calc_converts_units():
    """``calc`` should return Hartree / Hartree-per-Bohr regardless of ASE eV units."""
    from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator
    from chemrefine.quantities import BOHR_TO_ANGSTROM, HARTREE_TO_EV

    with (
        patch.dict(
            mlip_calculator._BACKENDS,
            {"mace_off": _spec(lambda **_kw: object())},  # sentinel calculator
            clear=False,
        ),
        # 1 eV/Å on x (the gradient already in eV/Å)
        patch.object(
            MlipCalculator,
            "single_point",
            return_value=(HARTREE_TO_EV, [[1.0, 0.0, 0.0]]),
        ),
    ):
        calc = MlipExtOptCalculator(model_name="small", task_name="mace_off", device="cpu")
        energy_h, gradient = calc.calc(_calc_data())
    assert energy_h == pytest.approx(1.0, rel=1e-12)
    assert gradient[0][0] == pytest.approx(BOHR_TO_ANGSTROM / HARTREE_TO_EV, rel=1e-12)


def test_mlip_extopt_calculator_calc_stamps_charge_and_spin():
    """``calc`` writes charge + spin into atoms.info (the FAIRChem omol head needs them)."""
    from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator

    seen: dict = {}

    def _capture(self, atoms):
        seen["charge"] = atoms.info.get("charge")
        seen["spin"] = atoms.info.get("spin")
        return 0.0, [[0.0, 0.0, 0.0]]

    with (
        patch.dict(
            mlip_calculator._BACKENDS,
            {"omol": _spec(lambda **_kw: object())},
            clear=False,
        ),
        patch.object(MlipCalculator, "single_point", _capture),
    ):
        calc = MlipExtOptCalculator(model_name="uma-s-1p2", task_name="omol", device="cpu")
        calc.calc(_calc_data(charge=-1, multiplicity=2))
    assert seen == {"charge": -1, "spin": 2}


# ---------------------------------------------------------------------------
# MlipEngine — template-driven direct mode
# ---------------------------------------------------------------------------

_FAKE_MLIP_TEMPLATE = """\
'''Fake MLIP template used by the direct-engine tests.

Reads no MLIP library; just declares the result variables ChemRefine's
appended footer harvests. Lets the lifecycle (prepare → submit → parse)
be exercised end-to-end without a real MLIP install.
'''
energy_hartree = -2.0 + 0.01 * $CHARGE
gradient_hartree_per_bohr = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]]
"""


def _write_mlip_templates(tmp_path: Path) -> Path:
    (tmp_path / "step1.py").write_text(_FAKE_MLIP_TEMPLATE, encoding="utf-8")
    (tmp_path / "cpu.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --partition=normal\n",
        encoding="utf-8",
    )
    return tmp_path


def _mlip_direct_ctx(tmp_path: Path, structures: tuple[Structure, ...], **overrides) -> StepContext:
    _write_mlip_templates(tmp_path)
    step_cfg = StepConfig(
        step=1,
        name="screen",
        engine="mlip",
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


def test_mlip_direct_prepare_renders_one_py_and_xyz_per_structure(tmp_path: Path):
    ctx = _mlip_direct_ctx(tmp_path, structures=(_seed("0"), _seed("1")))
    engine = get_engine("mlip")
    inputs = engine.prepare(ctx)
    assert len(inputs.files) == 2
    for script_path, output_json, sid in inputs.files:
        assert script_path.parent.name == sid  # per-structure directory
        assert script_path.name == f"step1_{sid}.py"
        assert output_json.name == f"step1_{sid}.json"
        assert (script_path.parent / f"{script_path.stem}_inp.xyz").is_file()
        rendered = script_path.read_text()
        assert "$XYZ_PATH" not in rendered
        assert "$CHARGE" not in rendered
        assert f"with open('{output_json.name}', \"w\")" in rendered


def test_mlip_direct_substitutes_option_placeholders(tmp_path: Path):
    """The YAML ``step.options`` drive the rendered script via $MODEL_NAME/$TASK_NAME/$DEVICE."""
    template = (
        "model = '$MODEL_NAME'\ntask = '$TASK_NAME'\ndevice = '$DEVICE'\nenergy_hartree = -1.0\n"
    )
    (tmp_path / "step1.py").write_text(template, encoding="utf-8")
    (tmp_path / "cpu.slurm.header").write_text("#!/bin/bash\n", encoding="utf-8")
    step_cfg = StepConfig(
        step=1,
        name="screen",
        engine="mlip",
        operation="opt_sp",
        options={"model_name": "medium", "task_name": "mace_off", "device": "cpu"},
    )
    ctx = StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "out" / "step1_screen",
        template_dir=tmp_path,
        scratch_dir=None,
        prev_state=PipelineState(structures=(_seed("0"),)),
        charge=0,
        multiplicity=1,
        max_cores=1,
        slurm_template="cpu.slurm.header",
        executables={},
    )
    rendered = get_engine("mlip").prepare(ctx).files[0][0].read_text()
    assert "model = 'medium'" in rendered
    assert "task = 'mace_off'" in rendered
    assert "device = 'cpu'" in rendered


def test_mlip_direct_substitutes_option_aliases(tmp_path: Path):
    """`size`/`task` aliases resolve before substitution into the template."""
    (tmp_path / "step1.py").write_text(
        "model = '$MODEL_NAME'\ntask = '$TASK_NAME'\nenergy_hartree = -1.0\n",
        encoding="utf-8",
    )
    (tmp_path / "cpu.slurm.header").write_text("#!/bin/bash\n", encoding="utf-8")
    step_cfg = StepConfig(
        step=1,
        name="screen",
        engine="mlip",
        operation="opt_sp",
        options={"size": "large", "task": "mace_mp"},
    )
    ctx = StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "out" / "step1_screen",
        template_dir=tmp_path,
        scratch_dir=None,
        prev_state=PipelineState(structures=(_seed("0"),)),
        charge=0,
        multiplicity=1,
        max_cores=1,
        slurm_template="cpu.slurm.header",
        executables={},
    )
    rendered = get_engine("mlip").prepare(ctx).files[0][0].read_text()
    assert "model = 'large'" in rendered
    assert "task = 'mace_mp'" in rendered


def test_mlip_direct_prepare_missing_template_raises(tmp_path: Path):
    ctx = _mlip_direct_ctx(tmp_path, structures=(_seed(),))
    (ctx.template_dir / "step1.py").unlink()
    engine = get_engine("mlip")
    with pytest.raises(ConfigError, match="MLIP template not found"):
        engine.prepare(ctx)


def test_mlip_direct_submit_runs_template_locally_when_no_sbatch(tmp_path: Path):
    """No sbatch → slurm.submit falls back to local bash; the template runs synchronously."""
    ctx = _mlip_direct_ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("mlip")
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


def test_mlip_direct_parse_returns_structure_with_energy_and_forces(tmp_path: Path):
    ctx = _mlip_direct_ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("mlip")
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


def test_mlip_direct_parse_raises_when_output_missing(tmp_path: Path):
    ctx = _mlip_direct_ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("mlip")
    inputs = engine.prepare(ctx)
    with pytest.raises(OutputParseError, match="output not found"):
        engine.parse(inputs, ctx)


def test_mlip_direct_parse_raises_when_output_not_json(tmp_path: Path):
    ctx = _mlip_direct_ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("mlip")
    inputs = engine.prepare(ctx)
    output_json = inputs.files[0][1]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text("not json", encoding="utf-8")
    with pytest.raises(OutputParseError, match="not valid JSON"):
        engine.parse(inputs, ctx)


def test_mlip_direct_parse_raises_when_energy_missing(tmp_path: Path):
    ctx = _mlip_direct_ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("mlip")
    inputs = engine.prepare(ctx)
    output_json = inputs.files[0][1]
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"oops": 0.0}), encoding="utf-8")
    with pytest.raises(OutputParseError, match="energy_hartree"):
        engine.parse(inputs, ctx)


def test_mlip_direct_parse_uses_positions_when_present(tmp_path: Path):
    ctx = _mlip_direct_ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("mlip")
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


def test_mlip_direct_submit_missing_slurm_header_raises(tmp_path: Path):
    ctx = _mlip_direct_ctx(tmp_path, structures=(_seed(),))
    (ctx.template_dir / "cpu.slurm.header").unlink()
    engine = get_engine("mlip")
    inputs = engine.prepare(ctx)
    with pytest.raises(ConfigError, match="SLURM header"):
        engine.submit(inputs, ctx)


def test_mlip_direct_submit_respects_cores_option(tmp_path: Path):
    ctx = _mlip_direct_ctx(tmp_path, structures=(_seed(),), options={"cores": 2}, max_cores=4)
    engine = get_engine("mlip")
    inputs = engine.prepare(ctx)
    with patch("chemrefine.slurm.shutil.which", return_value=None):
        engine.submit(inputs, ctx)
    script_text = inputs.files[0][0].with_suffix(".slurm").read_text()
    assert "#SBATCH --ntasks=2" in script_text


# --- orb backend success path (mock the optional library) -------------------


def test_build_orb_success_path(monkeypatch):
    pretrained = types.SimpleNamespace(orb_v2=MagicMock(return_value="ORBFF"))
    forcefield = types.ModuleType("orb_models.forcefield")
    forcefield.pretrained = pretrained
    calc_mod = types.ModuleType("orb_models.forcefield.inference.calculator")
    calc_mod.ORBCalculator = MagicMock(return_value="ORB_CALC")
    for name, mod in {
        "orb_models": types.ModuleType("orb_models"),
        "orb_models.forcefield": forcefield,
        "orb_models.forcefield.inference": types.ModuleType("orb_models.forcefield.inference"),
        "orb_models.forcefield.inference.calculator": calc_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    from chemrefine.engines.mlip.backends.orb import _build_orb

    assert _build_orb(model_name="orb_v2", device="cpu") == "ORB_CALC"


# --- orb older-layout fallback ----------------------------------------------


@pytest.mark.parametrize(
    "task, lib, package, extra",
    [
        ("mace_off", "mace", "mace-torch", "mlip-mace"),
        ("omol", "fairchem", "fairchem-core", "mlip-fairchem"),
        ("sevenn", "sevenn", "sevenn", "mlip-sevenn"),
        ("chgnet", "chgnet", "chgnet", "mlip-chgnet"),
        ("orb", "orb_models", "orb-models", "mlip-orb"),
    ],
)
def test_backend_missing_dependency_names_the_extra(task, lib, package, extra, monkeypatch):
    """A missing backend lib → a helpful ImportError naming the package + extra."""
    monkeypatch.setitem(sys.modules, lib, None)  # force the lazy import to fail
    from chemrefine.engines.mlip.calculator import build_calculator

    with pytest.raises(ImportError, match=f"{package}.*{extra}"):
        build_calculator(task_name=task, model_name="x")


def test_build_orb_older_layout(monkeypatch):
    """When the v3 ``inference.calculator`` import fails, fall back to the older path."""
    pretrained = types.SimpleNamespace(orb_v2=MagicMock(return_value="ORBFF"))
    forcefield = types.ModuleType("orb_models.forcefield")
    forcefield.pretrained = pretrained
    older_calc = types.ModuleType("orb_models.forcefield.calculator")
    older_calc.ORBCalculator = MagicMock(return_value="OLD_CALC")
    for name, mod in {
        "orb_models": types.ModuleType("orb_models"),
        "orb_models.forcefield": forcefield,
        # v3 layout absent (None ⇒ ImportError) → exercises the older-layout branch
        "orb_models.forcefield.inference.calculator": None,
        "orb_models.forcefield.calculator": older_calc,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    from chemrefine.engines.mlip.backends.orb import _build_orb

    assert _build_orb(model_name="orb_v2", device="cpu") == "OLD_CALC"
