"""Cross-engine invariants: properties every registered engine must satisfy at once.

The per-engine test files each cover their own engine, and the contract tests cover each
parser against a golden. Neither covers a property that breaks only when *two* correct
components disagree — a knob whose options model and whose scheduler heuristic default
differently, an executable quoted where it is run but not where it is logged. Each part
passes its own tests; the pair produces a wrong job.

Asserted over *every* registered engine rather than per engine, because per engine the next
one added is simply not covered.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from ase import Atoms

from chemrefine import slurm
from chemrefine.config import StepConfig, reject_shell_unsafe
from chemrefine.engines._options import EngineOptions
from chemrefine.engines.api import (
    ENGINES,
    FrequencyOutputParsing,
    JobExecutable,
    NmsCapableEngine,
    OptionsDeclaring,
    PreflightChecking,
    StructureArtifacts,
    TemplateDriven,
    get_engine,
)
from chemrefine.engines.mlip.registry import requirement_from_options
from chemrefine.errors import ConfigError
from chemrefine.slurm import script
from chemrefine.state import PipelineState, StepContext, Structure

# Options each engine needs before it will validate at all (no defaults on purpose).
_REQUIRED_OPTIONS: dict[str, dict[str, object]] = {
    "pyscf": {"basis": "def2-svp", "xc": "pbe"},
    "pyscf-extopt": {"basis": "def2-svp", "xc": "pbe"},
    # A training step must say what it trains and where. `task_name` has an inference
    # default (UMA) that would be the wrong thing to *train* by accident, and `device`
    # cannot default either way without silently costing a GPU or a week of CPU.
    "mlip-train": {"task_name": "mace_off", "device": "cpu"},
}


def _ctx(tmp_path: Path, engine_name: str, options: dict[str, object]) -> StepContext:
    """A minimal context for asking an engine about one step's options.

    The template file really exists, with a ``%pal`` line for the ORCA-family engines:
    the ExtOpt engines read their pal from it inside ``run_block`` (the server's thread
    budget), so a context whose template is only a path would fail exactly the invariants
    this module exists to hold.
    """
    step_cfg = StepConfig(
        step=1,
        engine=engine_name,
        operation="opt_sp",
        options={**_REQUIRED_OPTIONS.get(engine_name, {}), **options},
    )
    suffix = get_engine(engine_name).template_suffix
    template = tmp_path / f"step1.{suffix}"
    if not template.exists():
        template.write_text(
            "! Opt\n%pal nprocs 2 end\n" if suffix == "inp" else "", encoding="utf-8"
        )
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path,
        template_dir=tmp_path,
        template=template,
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


def _alias_capable() -> list[str]:
    """Engines whose options model spells at least one field more than one way.

    Only these can be given one knob twice, so only these have the ambiguity the test below
    asserts is refused. Selecting them here rather than skipping inside the test keeps a skip
    from standing permanently in the run, where it says nothing and hides a real one.
    """
    return [
        name
        for name in _gpu_capable()
        if any(
            len(spellings) > 1
            for spellings in getattr(get_engine(name), "options_cls", EngineOptions)
            ._spellings_by_field()
            .values()
        )
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
        "ntasks": 1,
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


@pytest.mark.parametrize("engine_name", _job_executables())
def test_an_array_run_block_leaves_its_sentinels_expandable(
    engine_name: str, tmp_path: Path
) -> None:
    """``$INP_NAME`` must reach the script bare, so the array task's assignment applies.

    The array path renders one run block against sentinel paths whose *names* are the bash
    variables `_run_array` documents (`_execution.py`), and each task assigns them from its own
    manifest row. Any engine that runs its input through ``shlex.quote`` produces
    ``'$INP_NAME'`` — single quotes suppress expansion, so every task of the array operates on
    a file *literally* called ``$INP_NAME`` and the step fails identically for all of them.

    The sibling test above cannot catch it: the exit trap fires whether or not the command it
    wrapped found its input, so a step can copy its results back and have computed nothing.
    Asserted for every `JobExecutable` because the trap is what the whole array path shares —
    the engines that had this bug were the most recently written ones.
    """
    engine = get_engine(engine_name)
    ctx = _ctx(tmp_path, engine_name, {})

    body = engine.run_block(ctx, Path("$INP_NAME"), Path("$OUT_NAME")).body

    for sentinel in ("$INP_NAME", "$OUT_NAME"):
        assert f"'{sentinel}'" not in body, (
            f"{engine_name}: {sentinel} is single-quoted in the run block, so the array task's "
            f"assignment cannot expand it:\n{body}"
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


def test_options_capability_matches_the_getattr_consumers():
    """The Protocol and the ``getattr`` readers must partition the registry identically.

    ``_provision._backend_python`` and ``gpus_from_options`` read ``options_cls`` via
    ``getattr`` with a fallback; :class:`OptionsDeclaring` formalizes the same seam for
    ``isinstance`` consumers (schema introspection). Two detection idioms for one
    capability may not disagree about a single engine — or the fallback reader and the
    introspector would describe different knobs for the same step.
    """
    by_protocol = {n for n in ENGINES if isinstance(get_engine(n), OptionsDeclaring)}
    by_getattr = {n for n in ENGINES if getattr(get_engine(n), "options_cls", None) is not None}
    assert by_protocol == by_getattr
    for name in sorted(by_protocol):
        engine = get_engine(name)
        assert isinstance(engine, OptionsDeclaring)
        assert issubclass(engine.options_cls, EngineOptions)


def test_the_preflight_capability_stays_a_claim_not_boilerplate():
    """Exactly the engines that own a fail-fast refusal declare ``check_step``.

    ``mlip-train`` (its refusals — no device, no task, a policy with nothing to act on
    — are decidable from the config, and the step usually sits after days of label
    computation), the ExtOpt family (their options configure a server, so the
    strict read the run block makes is made up front too), and ``orca`` and ``qchem``
    (an explicit ``operation`` outside a family's parser dispatch would otherwise fail
    only after the step's jobs had run). The direct script engines
    stay out by documented design — lenient reads over templates that may carry knobs
    of their own — and the fake engine is the minimal third-party shape. A new engine
    that takes the hook extends this pin; one that grows a prepare-time refusal
    without the hook is the Thursday failure coming back.
    """
    checking = {n for n in ENGINES if isinstance(get_engine(n), PreflightChecking)}
    assert sorted(checking) == ["mlip-extopt", "mlip-train", "orca", "pyscf-extopt", "qchem"]


def test_every_orca_family_engine_refuses_an_unknown_operation_up_front():
    """The parser's operation vocabulary is enforced at ``check_step``, family-wide.

    An explicit ``operation`` picks the parser and nothing else, so a value the dispatch
    does not know cannot fail until the outputs are read — after every job in the step
    has run at full cost, with the paid outputs then unadoptable because the operation
    is part of every row key. The refusal lives on :class:`OrcaEngine.check_step`; this
    holds the *inheritance*: an ExtOpt subclass that overrides ``check_step`` for its own
    options must still call up the chain, or its steps quietly lose the guard. Derived
    from the registry, so a fourth ORCA-driven engine is covered by existing.
    """
    from chemrefine.engines.orca.engine import OrcaEngine

    family = sorted(n for n in ENGINES if isinstance(get_engine(n), OrcaEngine))
    assert family == ["mlip-extopt", "orca", "pyscf-extopt"], "the inheriting set moved"
    for name in family:
        engine = get_engine(name)
        assert isinstance(engine, PreflightChecking)
        step_cfg = StepConfig(
            step=1, engine=name, operation="opt-sp", options=_REQUIRED_OPTIONS.get(name, {})
        )
        with pytest.raises(ConfigError, match="opt-sp"):
            engine.check_step(step_cfg, charge=0, multiplicity=1)


def test_declaring_an_options_model_stays_a_claim_not_boilerplate():
    """Exactly the engines that read their options declare a model — no more, no fewer.

    ORCA is the deliberate holdout: its knobs live in the step template, ``options:`` on
    an ORCA step is read only by NMS, and a declared model would advertise ``device`` to
    the GPU invariants above while its ``gpus()`` stays 0. The fake engine is the minimal
    third-party shape — three methods and a name — and must keep registering without the
    capability. A new engine belongs in the declaring set the moment it reads one knob;
    extend this pin rather than defaulting ``options_cls`` onto a base class.
    """
    declaring = {n for n in ENGINES if isinstance(get_engine(n), OptionsDeclaring)}
    assert sorted(set(ENGINES) - declaring) == ["fake", "orca"]


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


@pytest.mark.parametrize("engine_name", _alias_capable())
def test_every_options_model_refuses_two_spellings_of_one_knob(engine_name: str):
    """A step naming one field twice fails the same way for every reader.

    Aliases let each backend read naturally (`task` for `task_name`, `model`/`size` for
    `model_name`), which means a step *can* set one knob twice. Left to pydantic that is an
    `extra="forbid"` on whichever spelling it did not pick -- a message naming the wrong
    problem -- and only on the strict path, so `requirement_from_options`, which
    `preflight_backends` calls, accepts a config that the direct engine's template render
    then refuses: it passes the fail-fast check and dies in `prepare`.
    """
    options_cls = getattr(get_engine(engine_name), "options_cls", EngineOptions)
    aliased = {
        field: sorted(names)
        for field, names in options_cls._spellings_by_field().items()
        if len(names) > 1
    }
    assert aliased, f"{engine_name} was parametrized as alias-capable but declares none"

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
    # `_resolve_step_option_paths` rewrites the path-valued knobs (`_STEP_OPTION_PATHS`)
    # against the config file's directory -- before any engine model exists to read
    # through, because resolution is the loader's job and the model cannot know the
    # file's directory. The read is variable-keyed over that declared list, which is
    # exactly the shape the matcher's dynamic-key marker exists to put on this record.
    "config.py",
}

_DYNAMIC_KEY = "<dynamic key>"
"""What :func:`_raw_option_reads` reports for a raw read whose key is not a literal.

A variable-keyed ``options.get(key)`` can read any declared knob, so it cannot be
cleared against the declared set -- it is an offender unless its module is on the
allow-list with a reason. Without this marker the config loader's own such read sat
outside the register the test's docstring promises, invisible."""


def _raw_option_reads(source: str) -> list[str]:
    """Every raw-dict ``.get(...)`` in ``source``: literal keys, else :data:`_DYNAMIC_KEY`.

    Parsed rather than grepped. A regex over the text also matches prose: the docstring on
    `gpus_from_options` quotes the very call it exists to replace, so a text scan flags the
    module that avoids it.

    The receiver may be a bare name (``options.get``) or an attribute chain ending in one
    (``ctx.step_cfg.options.get``) -- the chained spelling reads the same raw dict and
    was invisible to the bare-``Name`` matcher this extends.
    """

    def _is_raw_dict(node: ast.expr) -> bool:
        return (isinstance(node, ast.Name) and node.id in _RAW_DICT_NAMES) or (
            isinstance(node, ast.Attribute) and node.attr in _RAW_DICT_NAMES
        )

    keys: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "get" or not _is_raw_dict(node.func.value):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            keys.append(first.value)
        else:
            keys.append(_DYNAMIC_KEY)
    return keys


def test_no_module_reads_a_declared_option_key_off_the_raw_dict():
    """A knob a model declares must be read through that model, everywhere.

    An options divergence takes one shape: a raw `options.get("x")` beside a model that
        already declares `x`, the two carrying different defaults or alias rules. Nothing
        notices until the mismatch produces a wrong job — a CUDA script scheduled on a CPU node,
        a training step picking the wrong SLURM header, a backend requirement that preflight
        accepts and `prepare` refuses.

        A genuinely raw read belongs in `_ALLOWED_RAW_READS` with its reason, so it is a
        decision on the record rather than one that crept in.
    """
    declared: set[str] = set()
    for name in sorted(ENGINES):
        options_cls = getattr(get_engine(name), "options_cls", None)
        if options_cls is not None:
            declared |= options_cls.accepted_names()
    assert declared, "no engine declares an options model — has options_cls been dropped?"

    src = Path(__file__).resolve().parent.parent / "src" / "chemrefine"
    offenders = [
        f"{rel}: reads declared knob {key!r} off the raw dict"
        for path in sorted(src.rglob("*.py"))
        if (rel := path.relative_to(src).as_posix()) not in _ALLOWED_RAW_READS
        for key in _raw_option_reads(path.read_text(encoding="utf-8"))
        # A dynamic key can read any declared knob, so it offends unless allow-listed.
        if key in declared or key == _DYNAMIC_KEY
    ]

    assert offenders == [], "\n".join(offenders)


@pytest.mark.parametrize("engine_name", _job_executables())
def test_no_engine_emits_a_trap_of_its_own(engine_name: str, tmp_path: Path):
    """Teardown is returned as data, never trapped by the engine.

    `RunBlock` splits an engine's bash into `body` and `cleanup` so teardown has somewhere to
    go that is not a `trap`: the script installs exactly one `EXIT` handler and interpolates
    `cleanup` inside it. That removes the *reason* an engine would trap — but the handler is
    armed before the body runs, so an engine that wrote `trap ... EXIT` into `body` anyway
    would still displace it, and the loss is quiet.

    The type carries the intent; this carries the rule, over every registered engine. Its
    companion,
    `test_the_assembled_script_still_runs_its_exit_handler`, catches the consequence by
    running the composed script; this one names the cause.
    """
    engine = get_engine(engine_name)
    block = engine.run_block(_ctx(tmp_path, engine_name, {}), Path("in.inp"), Path("out.out"))

    assert "trap " not in block.body, f"{engine_name}: run block installs its own trap"
    assert "trap " not in block.cleanup, f"{engine_name}: cleanup installs its own trap"


# Options that switch an engine's teardown *on*, so the cleanup rule below sweeps the bash
# the engine actually emits rather than an empty string. Separate from _REQUIRED_OPTIONS,
# which is about validating at all.
_CLEANUP_OPTIONS: dict[str, dict[str, object]] = {
    "qchem": {"save": True},
}

_BRACED_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)([^}]*)\}")
_UNBRACED_VAR = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")

#: Variables the generated preamble exports before the trap is armed — the only names a
#: cleanup may expand bare. Everything else is set (if at all) by the body, which the
#: cleanup cannot assume ran: the handler is armed first, deliberately.
_PREAMBLE_EXPORTS = frozenset({"WORK_DIR", "OUTPUT_DIR"})


@pytest.mark.parametrize("engine_name", _job_executables())
def test_every_cleanup_survives_the_armed_trap_window(engine_name: str, tmp_path: Path):
    """A cleanup expansion is `${VAR:-…}` unless the preamble exported the name.

    The other half of the single-trap doctrine. The EXIT handler is armed *before* the
    body runs — it has to be, or a failure inside the body would clean up nothing — and
    the script runs under `set -u`. So a cleanup that expands a variable the *body* sets
    is a nounset abort whenever the job dies in the armed-trap window: the handler exits
    mid-line and takes the copy-back, the output_dirs copy and the runlog footer with it
    (`set +e` does not suppress nounset). Q-Chem's `save: true` copy shipped exactly
    that, referencing the body-set `$QCSAVE`.

    The rule is mechanical so the sweep is: every `${VAR …}` carries a default
    (`:-` spelling), and every bare `$VAR` names a preamble export. A `-n` guard around
    the action is welcome but not sufficient — the guard keeps the action from firing on
    nothing, the `:-` keeps the expansion itself from aborting the handler.
    """
    engine = get_engine(engine_name)
    ctx = _ctx(tmp_path, engine_name, _CLEANUP_OPTIONS.get(engine_name, {}))
    cleanup = engine.run_block(ctx, Path("in.inp"), Path("out.out")).cleanup

    for name, spec in _BRACED_VAR.findall(cleanup):
        assert spec.startswith(":-") or name in _PREAMBLE_EXPORTS, (
            f"{engine_name}: cleanup expands ${{{name}{spec}}} without a `:-` default; "
            f"in the armed-trap window that is a nounset abort inside the exit handler"
        )
    for name in _UNBRACED_VAR.findall(_BRACED_VAR.sub(" ", cleanup)):
        assert name in _PREAMBLE_EXPORTS, (
            f"{engine_name}: cleanup expands bare ${name}, which only the body sets; "
            f"spell it ${{{name}:-}} so a death in the armed-trap window cannot abort "
            f"the exit handler"
        )


# ---------------------------------------------------------------------------
# Shell safety: every value reaching generated bash is classified
# ---------------------------------------------------------------------------

#: How each parameter of the bash-emitting functions is kept safe. The keys are the union of
#: their signatures; the values are the reason, mirroring the table in
#: `config.reject_shell_unsafe`. `VALIDATED` means the value is user-supplied text that
#: reaches bash and is run through that rule; everything else says why it cannot carry a
#: hostile character in the first place.
VALIDATED = "reject_shell_unsafe"

_BASH_PARAM_SAFETY: dict[str, str] = {
    # --- user-supplied text, validated at config load -----------------------------------
    "operation": VALIDATED,
    "output_dir": VALIDATED,  # config.output_dir
    "work_dir_expr": VALIDATED,  # built from output_dir + scratch_dir, both validated
    "output_dirs": VALIDATED,  # engine-declared; pyscf's tensor_folder goes through the rule
    # --- safe by construction ------------------------------------------------------------
    "engine": "a registry key",
    "step": "an integer",
    "cores": "an integer",
    "structure_id": "minted by chemrefine.ids",
    "step_label": "StepConfig.dir_name(): 'step{int}' plus a name matched against _NAME_RE",
    "step_dir": "output_dir (validated) joined with step_label",
    "input_path": "minted by chemrefine.ids under output_dir",
    "globs_expr": "a join of output_globs — engine-declared constants, never config",
    "extra_fields": "engine-supplied runlog rows, not interpolated as code",
    # --- the script builders' own surface ------------------------------------------------
    "scratch_dir": VALIDATED,  # config.scratch_dir
    "job_name": "the input path's stem (ids-minted under the validated output_dir), or "
    "the _NAME_RE-checked step label plus a literal '_array' suffix",
    "ntasks": "an integer",
    "cpus_per_task": "an integer",
    "memory_mb": "an integer or None",
    "template_path": "a path that is read, never interpolated — its lines become the "
    "cluster header the user already owns",
    "script_path": "the write destination; never part of the script's text",
    "output_globs": "engine-declared constants (a ClassVar, or a property over trainer "
    "declarations); never config",
    "extra_header_fields": "engine-supplied runlog rows, not interpolated as code",
    # --- bash this project wrote ---------------------------------------------------------
    "header": "the output of job_log.bash_header, itself covered by this table",
    "footer": "the output of job_log.bash_footer, itself covered by this table",
    "cleanup": "bash authored in this repo's source",
    "run_block": "a RunBlock of bash authored in this repo's source",
}


def _bash_emitting_params() -> set[str]:
    """Every parameter name of the functions that turn values into generated bash.

    ``build_script`` and ``build_array_script`` are enumerated alongside the three
    primitives they compose: they take values of their own (``job_name``, the header
    template, the write destination) that never pass through ``_run_body_lines``, and
    leaving them out is how a stale ``job_name`` row survived in
    ``config.reject_shell_unsafe``'s table for a trainer field that no longer exists —
    the enumeration answered for three of the five functions and claimed the whole.
    """
    from chemrefine import job_log

    functions = (
        job_log.bash_header,
        job_log.bash_footer,
        script._run_body_lines,
        script.build_script,
        script.build_array_script,
    )
    return {
        name for func in functions for name in inspect.signature(func).parameters if name != "self"
    }


def test_every_value_reaching_generated_bash_is_classified():
    """Adding a value to the generated script must be a decision someone records.

    `reject_shell_unsafe` is a single rule with a docstring table of what it validates
    against what is safe by construction. A table is only as good as the last person to
    read it, and bash reaches further than the path fields it was first written for:
    `operation` lands in the runlog heredoc, `tensor_folder` in a `cp -r "..."` where bash
    substitutes inside the quotes.

    So the enumeration is taken from the signatures rather than from the table. A new parameter
    on any of the three bash-emitting functions fails here until it is classified -- which is
    the point: the failure asks for a decision, at the moment the value is added, instead of
    after it turns up in a shell.
    """
    unclassified = _bash_emitting_params() - _BASH_PARAM_SAFETY.keys()
    assert unclassified == set(), (
        f"these values reach generated bash with no recorded reason they are safe: "
        f"{sorted(unclassified)} — validate each with config.reject_shell_unsafe, or add it "
        f"to _BASH_PARAM_SAFETY saying why it cannot carry a hostile character"
    )

    stale = _BASH_PARAM_SAFETY.keys() - _bash_emitting_params()
    assert stale == set(), f"_BASH_PARAM_SAFETY classifies values that no longer exist: {stale}"


@pytest.mark.parametrize("hostile", ['"', "$", "`", "\\", "\n", "\t"])
def test_the_rule_rejects_every_character_it_claims_to(hostile: str):
    """The classification above is only worth anything if `VALIDATED` actually bites.

    Each of these ends a quoted string, starts a substitution, or breaks the line -- the
    ways a value interpolated into the generated script stops being a value. The tab is
    the odd one out: it breaks no quoting, but the array manifest is tab-delimited, so a
    tab in a path shifts every field after it.
    """
    with pytest.raises(ValueError, match="cannot be safely embedded"):
        reject_shell_unsafe(f"/tmp/x{hostile}y", what="path", fix="rename it")


def test_every_engine_with_the_nms_hook_satisfies_the_nms_protocol():
    """Declaring the NMS hook is not enough — the whole protocol has to hold.

    `step.run_step` gates on `isinstance(engine, NmsCapableEngine)` and, when it fails,
    runs a plain step: `nms: true` becomes a no-op with nothing said. So an engine that
    grows the frequency hook but misses another member of the protocol would quietly stop
    doing normal-mode sampling. Assert the two never come apart.
    """
    declared = [n for n in sorted(ENGINES) if hasattr(get_engine(n), "nms_input_info")]
    assert declared, "no engine offers NMS — has the hook been renamed?"
    not_capable = [n for n in declared if not isinstance(get_engine(n), NmsCapableEngine)]
    assert not_capable == [], (
        f"{not_capable} declare nms_input_info but fail isinstance(NmsCapableEngine), "
        f"so `nms: true` would silently run a plain step"
    )


def test_every_nms_capable_engine_can_reparse_its_frequency_output():
    """NMS capability implies the viewer capability — the two must never come apart.

    `agent_tools._parse_output_frames` re-reads a finished step's output through
    `FrequencyOutputParsing` when `analyze_mode` or `get_structure` asks about a mode
    (the tensor is deliberately not cached, so the file is the source). An engine whose
    `parse_one` populates the tensor but that skips the ctx-free hook would compute
    normal modes the viewer tools then refuse to show.
    """
    capable = [n for n in sorted(ENGINES) if isinstance(get_engine(n), NmsCapableEngine)]
    assert capable, "no engine is NMS-capable — has the protocol been renamed?"
    blind = [n for n in capable if not isinstance(get_engine(n), FrequencyOutputParsing)]
    assert blind == [], (
        f"{blind} are NMS-capable but not FrequencyOutputParsing, so mode analysis "
        f"would refuse outputs that really carry a normal-mode section"
    )


def test_artifact_paths_resolves_to_the_input_that_was_written(tmp_path: Path):
    """A structure's input really is at the path the engine reports.

    Weaker than it looks on purpose: `JobEngine.prepare` derives its paths *from*
    `artifact_paths`, so the base class cannot disagree with itself — that guarantee is
    structural, not tested. What this catches is an engine that overrides one of the two
    without the other, which is the only way they can still come apart.
    """
    checked = []
    for name in sorted(ENGINES):
        engine = get_engine(name)
        if not isinstance(engine, StructureArtifacts):
            continue
        root = tmp_path / name
        root.mkdir(parents=True)
        ctx = replace(
            _ctx(root, name, {}),
            prev_state=PipelineState(structures=(Structure(id="0", atoms=Atoms("H")),)),
        )
        (root / f"step1.{engine.template_suffix}").write_text("", encoding="utf-8")
        engine.prepare(ctx)

        claimed_input, claimed_output = engine.artifact_paths(ctx, "0")
        assert claimed_input.is_file(), f"{name}: no input at the path artifact_paths claims"
        assert claimed_output.parent == claimed_input.parent
        checked.append(name)

    assert checked, "no engine declares the capability — has StructureArtifacts been dropped?"


# ---------------------------------------------------------------------------
# Layering — the engine subsystem knows nothing about caching
# ---------------------------------------------------------------------------


def _chemrefine_imports(module: Path) -> set[str]:
    """Every ``chemrefine.*`` module a source file imports, at any nesting depth."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("chemrefine")
        ):
            tail = node.module[len("chemrefine.") :]
            found.update([tail] if node.module != "chemrefine" else [a.name for a in node.names])
        elif isinstance(node, ast.Import):
            found.update(
                a.name[len("chemrefine.") :] for a in node.names if a.name.startswith("chemrefine.")
            )
    return found


def test_no_engine_module_imports_the_cache() -> None:
    """An engine turns a specification into a calculation. Caching is not its concern.

    `docs/internals/architecture.md` draws the layering with `engines/*` depending on
    `slurm, throttle, io, ids, job_log, quantities` — and not on `cache`. That was never
    true: the subsystem imported `cache` from the day it was written, first for
    `save_manifest` and latterly so every engine could implement an `input_digest` the
    cache alone consumed. The template is on the `StepContext` now and the cache digests it
    there, so the edge is gone and this is what keeps it gone.
    """
    engines_root = Path(inspect.getfile(get_engine)).parent
    offenders = {
        str(p.relative_to(engines_root)): sorted(
            m for m in _chemrefine_imports(p) if m == "cache" or m.startswith("cache.")
        )
        for p in sorted(engines_root.rglob("*.py"))
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"engines/ must not import cache: {offenders}"


def test_every_template_driven_engine_declares_its_suffix() -> None:
    """`TemplateDriven` is what `build_context` reads to resolve the step's template.

    An engine that renders a template but does not declare the capability gets
    ``ctx.template is None`` and fails at render time, which pushes the suffix into the
    engine's own body instead.
    """
    for name in ENGINES:
        engine = get_engine(name)
        if not isinstance(engine, TemplateDriven):
            continue
        assert engine.template_suffix, f"{name}: TemplateDriven with an empty template_suffix"
        assert engine.label, f"{name}: TemplateDriven with an empty label"
