"""Cross-engine invariants: properties every registered engine must satisfy at once.

The per-engine test files each cover their own engine, and the contract tests cover
each parser against a golden. What neither covers is a property that only breaks when
*two* correct components disagree -- which is where the defects in this file's history
actually lived:

* the `device` knob had two readers (the engine's options model and the scheduler's
  raw-dict heuristic) with different defaults, so a step that named no device rendered
  a CUDA script and was scheduled as a CPU job;
* the ORCA executable was shell-quoted in `OrcaEngine.run_block` but not in the ExtOpt
  subclass that overrides it, so a path with a space broke one path and not the other.

Both are properties over *every* registered engine, so they are asserted that way here
rather than per engine, where the next engine would simply not be covered.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from chemrefine.config import StepConfig
from chemrefine.engines._options import EngineOptions
from chemrefine.engines.api import ENGINES, JobExecutable, get_engine
from chemrefine.engines.mlip.calculator import requirement_from_options
from chemrefine.errors import ConfigError
from chemrefine.state import PipelineState, StepContext

# Options each engine needs before it will validate at all (no defaults on purpose).
_REQUIRED_OPTIONS: dict[str, dict[str, object]] = {
    "pyscf": {"basis": "def2-svp", "xc": "pbe"},
    "pyscf-extopt": {"basis": "def2-svp", "xc": "pbe"},
}


def _ctx(tmp_path: Path, engine_name: str, options: dict[str, object]) -> StepContext:
    """A minimal context for asking an engine about one step's options."""
    step_cfg = StepConfig(
        step=1,
        engine=engine_name,
        operation="opt_sp",
        options={**_REQUIRED_OPTIONS.get(engine_name, {}), **options},
    )
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path,
        template_dir=tmp_path,
        scratch_dir=None,
        prev_state=PipelineState(),
        charge=0,
        multiplicity=1,
        max_cores=8,
        slurm_template="cpu.slurm.header",
        executables={"orca": "orca"},
    )


def _job_executables() -> list[str]:
    """Every engine that generates a SLURM run block (so every engine that emits bash)."""
    return [name for name in sorted(ENGINES) if isinstance(get_engine(name), JobExecutable)]


def _gpu_capable() -> list[str]:
    """Engines that can ask for a GPU: they declare an options model, so they read `device`."""
    return [
        name
        for name in _job_executables()
        if getattr(get_engine(name), "options_cls", None) is not None
    ]


@pytest.mark.parametrize("engine_name", _gpu_capable())
@pytest.mark.parametrize("device", [None, "cpu", "cuda"])
def test_gpu_demand_matches_the_engines_own_device_option(
    engine_name: str, device: str | None, tmp_path: Path
):
    """`gpus()` must agree with the engine's validated `device`, including when unset.

    The two must be read through the same model. Reading the raw dict made "unset"
    mean `cpu` to the scheduler and `cuda` to the options model, which scheduled a
    CPU job whose script then asked for a device it had not been allocated -- and
    bypassed both the GPU budget and `Throttler.assign_device`, so concurrent local
    steps collided on device 0.
    """
    engine = get_engine(engine_name)
    ctx = _ctx(tmp_path, engine_name, {} if device is None else {"device": device})

    options_cls = getattr(engine, "options_cls", EngineOptions)
    resolved = options_cls.from_raw_lenient(ctx.step_cfg.options).device

    assert engine.gpus(ctx) == (1 if resolved == "cuda" else 0)


@pytest.mark.parametrize("engine_name", _gpu_capable())
def test_unset_device_never_silently_requests_a_gpu(engine_name: str, tmp_path: Path):
    """CPU is the floor: a step that names no device must schedule as a CPU job."""
    engine = get_engine(engine_name)
    assert engine.gpus(_ctx(tmp_path, engine_name, {})) == 0


@pytest.mark.parametrize("engine_name", _job_executables())
def test_run_block_survives_paths_with_spaces(engine_name: str, tmp_path: Path):
    """Every engine's generated bash must parse, with a space in the executable path.

    A config-supplied path is the one value that reaches generated bash, and an
    unquoted one with a space in it silently becomes two words. Asserted over every
    engine so a new `run_block` -- or an override of an existing one -- cannot
    reintroduce it for its own path only.
    """
    engine = get_engine(engine_name)
    ctx = _ctx(tmp_path, engine_name, {})
    object.__setattr__(ctx, "executables", {"orca": "/opt/my orca/orca"})

    block = engine.run_block(ctx, Path("step1_0.inp"), Path("step1_0.out"))

    # `bash -n` parses the block without running it: unbalanced quoting fails here.
    subprocess.run(["bash", "-n"], input=block, text=True, check=True, capture_output=True)
    for line in block.splitlines():
        if "my orca" in line:
            command = shlex.split(line.split(">")[0])
            assert "/opt/my orca/orca" in command, f"executable was split by the shell: {command}"


# ---------------------------------------------------------------------------
# Alias handling is the model's job, and every reader must get the same answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine_name", _gpu_capable())
def test_every_options_model_refuses_two_spellings_of_one_knob(engine_name: str):
    """A step naming one field twice fails the same way for every reader.

    Aliases let each backend read naturally (`task` for `task_name`, `model`/`size` for
    `model_name`), which means a step *can* set one knob twice. Pydantic rejected that as
    `extra="forbid"` on whichever spelling it did not pick -- a message naming the wrong
    problem -- and only on the strict path. So `requirement_from_options`, which
    `preflight_backends` calls, accepted a config that the direct engine's template render
    then refused: it passed the fail-fast check and died in `prepare`.
    """
    options_cls = getattr(get_engine(engine_name), "options_cls", EngineOptions)
    aliased = {
        field: sorted(names)
        for field, names in options_cls._spellings_by_field().items()
        if len(names) > 1
    }
    if not aliased:
        pytest.skip(f"{engine_name} declares no aliases")

    for field, spellings in aliased.items():
        raw = dict.fromkeys(spellings, "x")
        for read in (options_cls.from_raw, options_cls.from_raw_lenient):
            with pytest.raises(ConfigError, match=field):
                read(raw)


def test_preflight_and_template_render_agree_on_an_ambiguous_mlip_step(tmp_path: Path):
    """The concrete case: both mlip readers raise, and raise a ConfigError.

    ConfigError specifically -- it carries the documented exit code, where the raw
    pydantic ValidationError escaped the CLI's contract as a traceback.
    """
    both = {"task": "mace_off", "task_name": "omol", "model_name": "small"}
    engine = get_engine("mlip")

    with pytest.raises(ConfigError):
        requirement_from_options(both)
    with pytest.raises(ConfigError):
        engine._template_vars(_ctx(tmp_path, "mlip", both))
