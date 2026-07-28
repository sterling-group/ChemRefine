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

import ast
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from chemrefine import slurm
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


def _assemble(
    engine_name: str, tmp_path: Path, *, array: bool = False
) -> tuple[Path, dict[str, str]]:
    """Build a real SLURM script for ``engine_name``; return ``(script, output_dir)``.

    The whole point is to exercise the *composed* script -- header + job_log + scratch trap +
    the engine's own run block -- rather than the run block on its own.
    """
    engine = get_engine(engine_name)
    ctx = _ctx(tmp_path, engine_name, {})
    # Point ORCA at a binary that cannot exist, so the run block dies on its first real
    # command. `orca` on PATH is not safe to leave resolvable: desktop Linux ships
    # /usr/bin/orca, the GNOME screen reader (the same trap scripts/release-check.sh guards).
    object.__setattr__(ctx, "executables", {"orca": str(tmp_path / "no-such-orca")})
    header = tmp_path / "cpu.slurm.header"
    header.write_text("#!/bin/bash\n#SBATCH --partition=test\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    inp = out_dir / "step1_0.inp"
    inp.write_text("! SP\n", encoding="utf-8")

    common = {
        "pal": 1,
        "template_path": header,
        "output_dir": out_dir,
        "scratch_dir": None,
        "engine": engine_name,
        "operation": "opt_sp",
        "step": 1,
        "output_globs": engine.output_globs,
        "output_dirs": engine.output_dirs(ctx),
        "extra_header_fields": engine.extra_header_fields(ctx),
    }
    if array:
        script = slurm.build_array_script(
            step_label="step1",
            script_path=tmp_path / "array.slurm",
            run_block=engine.run_block(ctx, Path("$INP_NAME"), Path("$OUT_NAME")),
            **common,
        )
        # The array task resolves its own row from $CR_MANIFEST, which sbatch normally
        # exports via --export=ALL,CR_MANIFEST=...; supply it here.
        manifest = slurm.write_array_manifests(
            [(inp, out_dir / "step1_0.out", "0")], out_dir, step_label="step1"
        )[0][0]
        return script, {"CR_MANIFEST": str(manifest)}

    script = slurm.build_script(
        job_name="step1_0",
        script_path=tmp_path / "job.slurm",
        input_path=inp,
        structure_id="0",
        step_label="step1",
        run_block=engine.run_block(ctx, inp, out_dir / "step1_0.out"),
        **common,
    )
    return script, {}


def _run_script(script: Path, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run an assembled script for real; it is expected to fail, and to clean up anyway.

    The shell and coreutils stay on PATH (the script needs them); what is missing is the
    engine's own compute -- ORCA points at a nonexistent binary, the script engines find no
    rendered `.py`, and the ExtOpt server dies importing a backend that is not installed.
    Every one of those unwinds the script, which is precisely the path the EXIT trap must
    survive.
    """
    env = {**os.environ, "SLURM_ARRAY_TASK_ID": "0", **extra_env}
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, env=env, timeout=180
    )


@pytest.mark.parametrize("engine_name", _job_executables())
def test_the_assembled_script_still_runs_its_exit_handler(engine_name: str, tmp_path: Path):
    """An engine's run block must not seize the script's EXIT trap.

    The copy-back, the scratch cleanup and the runlog footer all live in the outer `_on_exit`.
    An engine that installs its own `trap ... EXIT` *replaces* it -- silently, because the ORCA
    path redirects its `.out` to `$OUTPUT_DIR` directly, so parsing still succeeds and every
    other gate stays green. The visible damage is elsewhere: `pyscf-extopt`'s `save_tensors`
    directory never arrives, `.gbw`/`.hess` are abandoned in scratch, the runlog has no footer,
    and `$WORK_DIR` is never removed -- a scratch leak on every ExtOpt job.

    `bash -n` on the run block alone cannot see this; only the composed script can.
    """
    script, extra_env = _assemble(engine_name, tmp_path)

    result = _run_script(script, extra_env)

    assert "files_copied=" in result.stdout, (
        f"{engine_name}: the outer _on_exit never fired -- the run block replaced the EXIT trap.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert not list(tmp_path.glob("out/_work_*")), f"{engine_name}: scratch dir was not cleaned up"


@pytest.mark.parametrize("engine_name", _job_executables())
def test_the_assembled_array_script_still_runs_its_exit_handler(
    engine_name: str, tmp_path: Path
) -> None:
    """The array path shares ``_run_body_lines``, so it has exactly the same exposure.

    Asserted separately because a fix applied to the per-job path only would leave every
    ``slurm_array: true`` step still leaking -- and the ExtOpt engines are `JobExecutable`,
    so they reach `_run_array` too.

    The array task ``exec``-redirects itself to the canonical per-structure runlog, so unlike
    the per-job case the footer lands in that file rather than on stdout.
    """
    script, extra_env = _assemble(engine_name, tmp_path, array=True)

    _run_script(script, extra_env)

    runlog = tmp_path / "out" / "step1_0.runlog"
    assert runlog.is_file(), f"{engine_name}: the array task wrote no runlog"
    assert "files_copied=" in runlog.read_text(encoding="utf-8"), (
        f"{engine_name}: the array script's _on_exit never fired.\n{runlog.read_text()}"
    )


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

    block = engine.run_block(ctx, Path("step1_0.inp"), Path("step1_0.out")).body

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


# ---------------------------------------------------------------------------
# No second reader of a declared knob
# ---------------------------------------------------------------------------

# Names a raw `step.options` dict conventionally goes by in this codebase.
_RAW_DICT_NAMES = {"options", "raw", "opts"}

# Reads that are deliberately raw, with the reason they are allowed to be.
_ALLOWED_RAW_READS = {
    # Operates on `validated.model_dump()` (see extopt/engine.py), so this *is* the
    # model's output -- it just arrives as a dict because the server CLI is generic.
    "engines/_backend_server/base.py",
}


def _raw_option_reads(source: str) -> list[str]:
    """Every `<raw dict>.get("key")` literal key in ``source``.

    Parsed rather than grepped. A regex over the text also matches prose: the docstring
    on `gpus_from_options` quotes the very call it exists to have replaced, and a
    text scan flagged the module that fixed the bug.
    """
    keys: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "get" or not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id not in _RAW_DICT_NAMES or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            keys.append(first.value)
    return keys


def test_no_module_reads_a_declared_option_key_off_the_raw_dict():
    """A knob a model declares must be read through that model, everywhere.

    Every options divergence this project has had was one shape: a raw `options.get("x")`
    beside a model that already declares `x`, the two carrying different defaults or
    alias rules, and nothing noticing until the mismatch produced a wrong job -- a CUDA
    script scheduled on a CPU node, a training step picking the wrong SLURM header, a
    backend requirement that preflight accepted and `prepare` refused.

    So the rule is checked rather than remembered. If a genuinely raw read is needed, add
    the file to `_ALLOWED_RAW_READS` with the reason, which makes it a decision someone
    made rather than one that crept in.
    """
    declared: set[str] = set()
    for name in sorted(ENGINES):
        options_cls = getattr(get_engine(name), "options_cls", None)
        if options_cls is not None:
            declared |= options_cls._accepted_names()
    assert declared, "no engine declares an options model — has options_cls been dropped?"

    src = Path(__file__).resolve().parent.parent / "src" / "chemrefine"
    offenders = [
        f"{rel}: reads declared knob {key!r} off the raw dict"
        for path in sorted(src.rglob("*.py"))
        if (rel := path.relative_to(src).as_posix()) not in _ALLOWED_RAW_READS
        for key in _raw_option_reads(path.read_text(encoding="utf-8"))
        if key in declared
    ]

    assert offenders == [], "\n".join(offenders)


@pytest.mark.parametrize("engine_name", _job_executables())
def test_no_engine_emits_a_trap_of_its_own(engine_name: str, tmp_path: Path):
    """Teardown is returned as data, never trapped by the engine.

    `RunBlock` splits an engine's bash into `body` and `cleanup` so teardown has somewhere to
    go that is not a `trap`: the script installs exactly one `EXIT` handler and interpolates
    `cleanup` inside it. That removes the *reason* an engine would trap -- but the handler is
    armed before the body runs, so an engine that wrote `trap ... EXIT` into `body` anyway
    would still displace it, exactly as the ExtOpt engines did for ten weeks.

    So the type carries the intent and this carries the rule, over every registered engine
    rather than the two that happened to have the bug. Its companion,
    `test_the_assembled_script_still_runs_its_exit_handler`, catches the consequence by
    running the composed script; this one names the cause.
    """
    engine = get_engine(engine_name)
    block = engine.run_block(_ctx(tmp_path, engine_name, {}), Path("in.inp"), Path("out.out"))

    assert "trap " not in block.body, f"{engine_name}: run block installs its own trap"
    assert "trap " not in block.cleanup, f"{engine_name}: cleanup installs its own trap"
