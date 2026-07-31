"""Tests for the ORCA-driven PySCF engine (``pyscf-extopt``).

The PySCF-side surface is :class:`PyscfExtOptEngine` (which uses the
shared ``_extopt.server``) and :class:`PyscfExtOptCalculator` (which
wraps :mod:`chemrefine.engines.pyscf._runtime` for the SCF + gradient
call). The template-driven direct engine has its own test file
(``test_engines_pyscf.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines.api import ENGINES, NmsCapableEngine, get_engine
from chemrefine.engines.pyscf.options import PyscfOptions
from chemrefine.errors import ConfigError
from chemrefine.state import PipelineState, StepContext, Structure

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_pyscf_engines_are_registered():
    assert "pyscf" in ENGINES
    assert "pyscf-extopt" in ENGINES


def test_pyscf_extopt_engine_is_nms_capable():
    """ExtOpt inherits ORCA's NMS hooks — ORCA computes the Hessian over PySCF gradients."""
    assert isinstance(get_engine("pyscf-extopt"), NmsCapableEngine)


# ---------------------------------------------------------------------------
# PyscfExtOptEngine — ORCA-driven mode
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
        engine="pyscf-extopt",
        operation="opt_sp",
        options=options,
    )
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=template_dir,
        template=template_dir / "step1.inp",
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(structures=(seed,)),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        executables={},
    )


def test_pyscf_extra_blocks_point_progext_at_wrapper_without_ext_params(tmp_path: Path):
    """Settings live on the server (single channel); the ``.inp`` carries no ``Ext_Params``."""
    engine = get_engine("pyscf-extopt")
    ctx = _pyscf_ctx(tmp_path, basis="cc-pvdz", xc="b3lyp")
    extra = engine._extra_blocks(ctx)
    assert "%method" in extra
    assert "ProgExt" in extra
    assert "pyscf_extopt.sh" in extra
    # The knobs live on the server launch, not duplicated into the ORCA input.
    assert "Ext_Params" not in extra
    assert "--basis" not in extra


def test_pyscf_wrapper_carries_no_per_call_flags(tmp_path: Path):
    """Single channel: the wrapper is the plain single-arg form; df/gpu live on the server."""
    import os

    engine = get_engine("pyscf-extopt")
    ctx = _pyscf_ctx(tmp_path, df=True, gpu=True)
    engine.prepare(ctx)
    wrapper = engine._wrapper_path(ctx)
    assert os.access(wrapper, os.X_OK)
    text = wrapper.read_text()
    assert "--df" not in text
    assert "--gpu" not in text
    # ... they reach the backend via the server construction instead.
    run_block = engine.run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    ).body
    assert "--df" in run_block
    assert "--gpu" in run_block


def test_pyscf_run_block_starts_shared_extopt_server(tmp_path: Path):
    engine = get_engine("pyscf-extopt")
    ctx = _pyscf_ctx(tmp_path)
    run_block = engine.run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    ).body
    assert "-m chemrefine.engines._backend_server.server" in run_block
    assert "--backend pyscf" in run_block
    assert "--method dft" in run_block
    assert "--xc pbe" in run_block
    assert "--basis def2-svp" in run_block


def test_pyscf_run_block_emits_gpu_and_df_when_set(tmp_path: Path):
    engine = get_engine("pyscf-extopt")
    ctx = _pyscf_ctx(tmp_path, gpu=True, df=True)
    run_block = engine.run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    ).body
    assert "--df" in run_block
    assert "--gpu" in run_block


def test_pyscf_save_tensors_reaches_server_cmd(tmp_path: Path):
    """``options: {save_tensors: true, ...}`` reaches the server launch (single channel);
    the wrapper stays the plain single-arg form."""
    import os

    engine = get_engine("pyscf-extopt")
    # A relative tensor_folder is fine now — it's copied back into the structure
    # dir on exit (see _output_dirs); no absolute path required.
    ctx = _pyscf_ctx(tmp_path, save_tensors=True, localized=True, tensor_folder="td")
    run_block = engine.run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    ).body
    assert "--save_tensors" in run_block
    assert "--localized" in run_block
    assert "--tensor_folder td" in run_block

    engine.prepare(ctx)
    wrapper_text = engine._wrapper_path(ctx).read_text()
    assert os.access(engine._wrapper_path(ctx), os.X_OK)
    assert "--save_tensors" not in wrapper_text
    assert "--tensor_folder" not in wrapper_text


def test_pyscf_extopt_output_dirs_copies_relative_tensor_folder(tmp_path: Path):
    """A relative tensor_folder is copied back wholesale; absolute / off → nothing."""
    engine = get_engine("pyscf-extopt")
    assert engine.output_dirs(_pyscf_ctx(tmp_path, save_tensors=True, tensor_folder="tensors")) == (
        "tensors",
    )
    assert (
        engine.output_dirs(_pyscf_ctx(tmp_path, save_tensors=True, tensor_folder="/abs/keep")) == ()
    )
    assert engine.output_dirs(_pyscf_ctx(tmp_path, save_tensors=False)) == ()


def test_pyscf_run_block_omits_bool_flags_when_unset(tmp_path: Path):
    """Bool flags stay gated on their option; key-value knobs carry validated values."""
    engine = get_engine("pyscf-extopt")
    ctx = _pyscf_ctx(tmp_path, df=False, gpu=False)
    run_block = engine.run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    ).body
    assert "--method dft" in run_block
    assert " --df" not in run_block
    assert " --gpu" not in run_block
    assert " --save_tensors" not in run_block
    assert " --localized" not in run_block


def test_pyscf_unknown_option_fails_fast(tmp_path: Path):
    """A typoed knob (``basis_set:`` for ``basis:``) raises instead of silently
    running the calculation with the default basis.

    The raw options dict must not bypass :class:`PyscfOptions`, or
    unknown keys were dropped and the run proceeded with wrong settings.

    ConfigError rather than pydantic's ValidationError — a bad knob is a config error
    and must exit with the documented code instead of escaping the CLI as a traceback.
    """
    engine = get_engine("pyscf-extopt")
    ctx = _pyscf_ctx(tmp_path, basis_set="def2-tzvp")
    with pytest.raises(ConfigError, match="basis_set"):
        engine.run_block(
            ctx,
            inp_path=ctx.step_dir / "step1_structure_0.inp",
            out_path=ctx.step_dir / "step1_structure_0.out",
        )


def test_pyscf_run_block_includes_readiness_loop(tmp_path: Path):
    engine = get_engine("pyscf-extopt")
    ctx = _pyscf_ctx(tmp_path)
    block = engine.run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "/healthz" in block.body
    # Teardown is returned as data the script places inside its own EXIT handler, never bash
    # the engine traps: bash keeps one handler per signal, so a `trap ... EXIT` here replaced
    # the script's and took the tensor copy-back, the runlog footer and the scratch teardown.
    assert "kill -TERM" in block.cleanup
    assert "trap " not in block.body
    assert "trap " not in block.cleanup


def test_pyscf_prepare_writes_inp_with_method_block(tmp_path: Path):
    engine = get_engine("pyscf-extopt")
    ctx = _pyscf_ctx(tmp_path)
    inputs = engine.prepare(ctx)
    inp_text = inputs.files[0][0].read_text()
    assert "%method" in inp_text
    assert "ProgExt" in inp_text
    assert "pyscf_extopt.sh" in inp_text


def test_pyscf_prepare_materializes_executable_wrapper(tmp_path: Path):
    """The ``ProgExt`` wrapper the ``.inp`` points at must actually be written."""
    import os

    engine = get_engine("pyscf-extopt")
    ctx = _pyscf_ctx(tmp_path, basis="cc-pvdz", xc="b3lyp")
    engine.prepare(ctx)

    wrapper = engine._wrapper_path(ctx)
    assert wrapper.is_file()
    assert os.access(wrapper, os.X_OK)

    text = wrapper.read_text()
    assert "chemrefine.engines.orca.extopt.bridge" in text
    assert "--backend pyscf" in text
    # Single channel: the SCF knobs live on the server, never baked into the wrapper.
    assert "--basis" not in text
    assert "--xc" not in text


# ---------------------------------------------------------------------------
# PyscfExtOptCalculator — full coverage lives in tests/test_engines_pyscf_extopt_calc.py
# ---------------------------------------------------------------------------


def test_pyscf_extopt_calculator_from_args_round_trips():
    from chemrefine.engines._backend_server.server import parse_args
    from chemrefine.engines.pyscf.extopt_calc import PyscfExtOptCalculator

    args = parse_args(
        [
            "--backend",
            "pyscf",
            "--method",
            "hf",
            "--xc",
            "b3lyp",
            "--basis",
            "cc-pvdz",
            "--df",
            "--gpu",
        ]
    )
    calc = PyscfExtOptCalculator.from_args(args)
    assert calc.method == "hf"
    assert calc.xc == "b3lyp"
    assert calc.basis == "cc-pvdz"
    assert calc.df is True
    assert calc.gpu is True


# Direct engine body coverage lives in ``tests/test_engines_pyscf.py``.


@pytest.mark.parametrize(
    "bad", ["tensors$(id -un)", "tensors`whoami`", 'tensors"x', "tensors\\x", "tensors\nx"]
)
def test_tensor_folder_with_shell_metacharacters_rejected(bad: str):
    """`tensor_folder` reaches generated bash, so it is held to the same rule as the paths.

    It is the engine option that lands in the on-exit copy-back as
    `cp -r "<tensor_folder>" "$OUTPUT_DIR/"` — and bash performs command substitution
    *inside* double quotes, so quoting there is not protection. Verified before the fix:
    `tensor_folder: 'tensors$(id -un > /abs/path)'` wrote that file when the job ran.

    Found by sweeping every value that reaches the generated script rather than patching
    the one field a report named -- the same omission that left `operation` unguarded.
    """
    with pytest.raises(ConfigError):
        PyscfOptions.from_raw({"basis": "def2-svp", "xc": "pbe", "tensor_folder": bad})


def test_tensor_folder_allows_ordinary_names():
    """Only metacharacters are refused; a plain or nested folder name stays legal."""
    opts = PyscfOptions.from_raw(
        {"basis": "def2-svp", "xc": "pbe", "tensor_folder": "run1/tensors"}
    )
    assert opts.tensor_folder == "run1/tensors"
