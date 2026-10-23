"""Tests for ``XtbEngine``: preflight, template rule, run block, NMS hook, submission proof."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine import pipeline
from chemrefine.config import Config, StepConfig
from chemrefine.engines.api import NmsCapableEngine, get_engine, preflight_steps
from chemrefine.engines.xtb.engine import XtbEngine
from chemrefine.errors import ConfigError, OutputParseError
from chemrefine.state import PipelineState, StepContext, StepInputs, Structure

DATA = Path(__file__).resolve().parent / "data" / "engines" / "xtb"

_STARTER = XtbEngine.template_starter


def _ctx(
    tmp_path: Path,
    *,
    operation: str | None = "sp",
    options: dict[str, object] | None = None,
    executables: dict[str, str] | None = None,
    max_cores: int = 8,
    template: str = _STARTER,
    charge: int = 0,
    multiplicity: int = 1,
) -> StepContext:
    """A StepContext with a real xcontrol template on disk; the operation is the step's."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "step1.xcontrol").write_text(template, encoding="utf-8")
    step_cfg = StepConfig(step=1, engine="xtb", operation=operation, options=options or {})
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=template_dir,
        template=template_dir / "step1.xcontrol",
        operation=operation,
        scratch_dir=None,
        prev_state=PipelineState(
            structures=(Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])),)
        ),
        charge=charge,
        multiplicity=multiplicity,
        max_cores=max_cores,
        slurm_template="cpu.slurm.header",
        executables=executables or {},
    )


def _body(tmp_path: Path, **kwargs: object) -> str:
    """The run block body for a step shaped by ``kwargs`` (passed to ``_ctx``)."""
    ctx = _ctx(tmp_path, **kwargs)
    return XtbEngine().run_block(ctx, Path("step1_0.xcontrol"), Path("step1_0.out")).body


def _xtb_line(body: str) -> list[str]:
    """The xtb invocation's words, the redirect left off."""
    (line,) = [ln for ln in body.splitlines() if "--input" in ln]
    return shlex.split(line.partition(">")[0])


# ---------------------------------------------------------------------------
# The run block
# ---------------------------------------------------------------------------


def test_the_default_run_block_runs_a_gradient_on_one_thread(tmp_path: Path):
    """No options: bare ``xtb``, the structure's own geometry, GFN2 in gas, ``-P 1``."""
    body = _body(tmp_path)
    assert "export OMP_NUM_THREADS=1,1" in body
    assert "export MKL_NUM_THREADS=1" in body
    assert 'export OMP_STACKSIZE="${OMP_STACKSIZE:-4G}"' in body
    assert "ulimit -s unlimited 2>/dev/null || true" in body
    assert 'XTB_STEM="step1_0.xcontrol"; XTB_STEM="${XTB_STEM%.*}"' in body
    assert "XTBPATH" not in body, "a bare `xtb` names no install to find parameters in"
    words = _xtb_line(body)
    assert words == [
        "xtb",
        "$OUTPUT_DIR/${XTB_STEM}_inp.xyz",
        "--input",
        "step1_0.xcontrol",
        "--namespace",
        "$XTB_STEM",
        "--grad",
        "--gfn",
        "2",
        "--chrg",
        "0",
        "--uhf",
        "0",
        "-P",
        "1",
    ]
    assert body.rstrip().endswith('> "$OUTPUT_DIR/step1_0.out"')


def test_the_run_block_passes_the_steps_operation_options_charge_and_spin(tmp_path: Path):
    """What the step computes reaches xtb's command line, in xtb's own spellings."""
    words = _xtb_line(
        _body(
            tmp_path,
            operation="opt_freq",
            options={
                "method": "gfn1",
                "solvation": "alpb",
                "solvent": "water",
                "solvation_reference": "bar1M",
                "opt_level": "vtight",
                "cores": 4,
            },
            charge=-1,
            multiplicity=3,
        )
    )
    assert words[4:] == [
        "--namespace",
        "$XTB_STEM",
        "--ohess",
        "vtight",
        "--grad",
        "--gfn",
        "1",
        "--alpb",
        "water",
        "bar1M",
        "--chrg",
        "-1",
        "--uhf",
        "2",
        "-P",
        "4",
    ]


def test_threads_clamp_to_the_budget_in_every_spelling(tmp_path: Path):
    """``cores`` over ``max_cores`` is one task of ``max_cores`` CPUs, and xtb is told so."""
    ctx = _ctx(tmp_path, options={"cores": 16}, max_cores=4)
    engine = get_engine("xtb")
    assert engine.pal(ctx) == 16
    assert engine.slurm_layout(ctx) == (1, 4)
    body = engine.run_block(ctx, Path("step1_0.xcontrol"), Path("step1_0.out")).body
    assert "export OMP_NUM_THREADS=4,1" in body and "export MKL_NUM_THREADS=4" in body
    assert _xtb_line(body)[-2:] == ["-P", "4"]


def test_an_installed_binary_points_xtbpath_at_its_parameters_unless_set(tmp_path: Path):
    """``<prefix>/bin/xtb`` → ``<prefix>/share/xtb``, where GFN0's parameter file lives."""
    body = _body(tmp_path, executables={"xtb": "/opt/xtb 6.7/bin/xtb"})
    assert "if [ -z \"${XTBPATH:-}\" ]; then export XTBPATH='/opt/xtb 6.7/share/xtb'; fi" in body
    assert _xtb_line(body)[0] == "/opt/xtb 6.7/bin/xtb"


@pytest.mark.parametrize("executable", [None, "xtb", "relative/bin/xtb", "/opt/xtb/xtb"])
def test_a_binary_outside_an_install_layout_leaves_xtbpath_alone(executable: str | None):
    """Only an absolute ``…/bin/xtb`` says where its parameters are."""
    assert XtbEngine.parameter_dir(executable) is None


def test_an_environment_that_set_xtbpath_keeps_it(tmp_path: Path):
    """The derived default yields to a module or header that exported its own."""
    body = _body(tmp_path, executables={"xtb": "/opt/xtb/bin/xtb"})
    script = f'XTBPATH=/from/module\n{body.splitlines()[4]}\necho "$XTBPATH"\n'
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "/from/module"
    unset = subprocess.run(
        ["bash", "-c", f'unset XTBPATH\n{body.splitlines()[4]}\necho "$XTBPATH"\n'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert unset.stdout.strip() == "/opt/xtb/share/xtb"


def test_the_array_sentinel_reaches_the_geometry_and_input_unexpanded(tmp_path: Path):
    """In a job array the structure's names are bash variables, expanded per task."""
    ctx = _ctx(tmp_path)
    body = get_engine("xtb").run_block(ctx, Path("$INP_NAME"), Path("$OUT_NAME")).body
    assert 'XTB_STEM="$INP_NAME"; XTB_STEM="${XTB_STEM%.*}"' in body
    assert '--input "$INP_NAME"' in body and '> "$OUTPUT_DIR/$OUT_NAME"' in body
    stem = subprocess.run(
        ["bash", "-c", f'INP_NAME=step2_7-1.xcontrol\n{body.splitlines()[4]}\necho "$XTB_STEM"'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert stem.stdout.strip() == "step2_7-1"


def test_a_run_block_for_a_step_with_no_operation_is_refused(tmp_path: Path):
    """The run type cannot be guessed, even by a caller that skipped the preflight."""
    with pytest.raises(ConfigError, match="names its `operation:`"):
        _body(tmp_path, operation=None)


def test_the_header_records_the_binary(tmp_path: Path):
    """The runlog names the xtb that ran, as configured."""
    engine = get_engine("xtb")
    assert engine.extra_header_fields(_ctx(tmp_path)) == (("xtb_executable", "xtb"),)
    ctx = _ctx(tmp_path, executables={"xtb": "/opt/xtb/bin/xtb"})
    assert engine.extra_header_fields(ctx) == (("xtb_executable", "/opt/xtb/bin/xtb"),)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _preflight(operation: str | None, **options: object) -> None:
    """The run's t=0 walk over one xtb step."""
    step = StepConfig(step=1, engine="xtb", operation=operation, options=options)
    preflight_steps([step], charge=0, multiplicity=1)


@pytest.mark.parametrize("operation", ["sp", "opt_sp", "freq", "opt_freq"])
def test_every_operation_passes_preflight(operation: str):
    """The four run types, with and without implicit solvent."""
    _preflight(operation)
    _preflight(operation, solvation="gbsa", solvent="thf", solvation_reference="reference")


def test_a_step_without_an_operation_is_refused_up_front():
    """An xcontrol file cannot say what to run, so the step has to."""
    with pytest.raises(ConfigError, match=r"step 1: an xtb step names its `operation:`"):
        _preflight(None)


def test_an_unknown_operation_is_refused_up_front():
    """The vocabulary is the run types the command line knows."""
    with pytest.raises(ConfigError, match="unknown operation 'goat' for engine 'xtb'"):
        _preflight("goat")


@pytest.mark.parametrize("operation", ["opt_sp", "freq", "opt_freq"])
def test_cpcmx_runs_only_as_a_single_point(operation: str):
    """xtb's CPCM-X gives no gradient, so a run that needs one is refused before it runs."""
    _preflight("sp", solvation="cpcmx", solvent="1,2-dichloroethane")
    with pytest.raises(ConfigError, match="CPCM-X gives no gradient"):
        _preflight(operation, solvation="cpcmx", solvent="water")


@pytest.mark.parametrize("operation", ["sp", "freq"])
def test_an_opt_level_on_a_run_that_does_not_optimise_is_refused(operation: str):
    """A knob nothing reads is said, not dropped."""
    _preflight("opt_sp", opt_level="tight")
    with pytest.raises(ConfigError, match="does not optimise"):
        _preflight(operation, opt_level="tight")


@pytest.mark.parametrize(
    ("knob", "value", "match"),
    [("device", "cuda", "xtb runs on the CPU"), ("backend_python", "/x/python", "not a Python")],
)
def test_an_inherited_knob_the_engine_never_reads_is_refused_up_front(
    knob: str, value: str, match: str
):
    """The shared model accepts both; nothing here reads either."""
    with pytest.raises(ConfigError, match=match):
        _preflight("sp", **{knob: value})


def test_a_misspelled_knob_is_refused_up_front():
    """Strict options: a typo fails the run's t=0 walk rather than running gas phase."""
    with pytest.raises(ConfigError, match="solvant"):
        _preflight("sp", solvant="water")


# ---------------------------------------------------------------------------
# The template rule and prepare
# ---------------------------------------------------------------------------


def test_a_template_that_sets_the_charge_is_refused_when_the_run_starts(tmp_path: Path):
    """``check_template`` is asked with the template the run read."""
    from chemrefine.ids import require_template

    ctx = _ctx(tmp_path, template="$chrg 1\n$end\n")
    engine = get_engine("xtb")
    assert isinstance(engine, XtbEngine)
    with pytest.raises(ConfigError, match=r"sets \$chrg"):
        engine.check_template(ctx, require_template(ctx.template, label="xtb"))
    with pytest.raises(ConfigError, match=r"sets \$chrg"):
        engine.prepare(ctx)
    assert not ctx.step_dir.exists(), "a refused template leaves nothing on disk"


def test_prepare_writes_the_template_and_the_geometry_the_run_block_names(tmp_path: Path):
    """One xcontrol input per structure, the template's text, beside the ``_inp.xyz``."""
    ctx = _ctx(tmp_path, template="$opt\n   maxcycle=5\n$end")
    inputs = get_engine("xtb").prepare(ctx)
    ((inp, out, sid),) = inputs.files
    assert (inp.name, out.name, sid) == ("step1_0.xcontrol", "step1_0.out", "0")
    assert inp.read_text(encoding="utf-8") == "$opt\n   maxcycle=5\n$end\n"
    assert (inp.parent / "step1_0_inp.xyz").is_file(), "the geometry the run block reads"


def test_the_starter_template_is_accepted():
    """``chemrefine scaffold`` writes a template the engine runs."""
    from chemrefine.engines.xtb import input as xtb_input
    from chemrefine.ids import StepTemplate

    xtb_input.require_fillable(StepTemplate(path=Path("step1.xcontrol"), text=_STARTER))


# ---------------------------------------------------------------------------
# Parse and the NMS hook
# ---------------------------------------------------------------------------


def _staged(tmp_path: Path, case: str) -> Path:
    """The contract case's files copied into a structure directory; its output's path."""
    out_dir = tmp_path / "outputs" / "step1" / "0"
    shutil.copytree(DATA / case, out_dir)
    return out_dir / "step1_0.out"


def test_parse_one_reads_the_output_and_the_files_beside_it(tmp_path: Path):
    """The structure dir's own ``_inp.xyz`` is the geometry of a run that moves no atoms."""
    out = _staged(tmp_path, "freq")
    ctx = _ctx(tmp_path, operation="freq")
    (parsed,) = get_engine("xtb").parse_one(out, "0", ctx)
    assert parsed.symbols == ("N", "H", "H", "H")
    assert parsed.imaginary_freqs is not None and len(parsed.imaginary_freqs) == 1
    assert parsed.forces_ev_per_a is not None


def test_a_failure_reason_names_its_output_relative_to_its_step(tmp_path: Path):
    """A reason still points at the file once the tree is copied or moved."""
    out = _staged(tmp_path, "opt")
    (out.parent / "step1_0.xtbopt.xyz").unlink()
    with pytest.raises(
        OutputParseError, match=r"no optimised geometry step1_0\.xtbopt\.xyz .* 0/step1_0\.out"
    ):
        get_engine("xtb").parse(StepInputs(files=((out, out, "0"),)), _ctx(tmp_path))


def test_an_nms_winner_promoted_to_its_parents_name_reads_its_own_files(tmp_path: Path):
    """The namespace is what makes xtb's files follow their output through a promotion.

    :func:`chemrefine.attempts.promote` copies the files that start with the child's stem
    to the parent's. Named without the namespace (``xtbopt.xyz``, ``g98.out``) they stayed
    in the attempt, and the promoted output said it had written files that were not beside
    it. Its stdout still names them as the child's run printed them, so the parent's
    copies are found by its own stem.
    """
    from chemrefine import attempts

    attempt = tmp_path / "outputs" / "step1" / "0" / "attempt1"
    child = attempt / "0_m6_neg"
    child.mkdir(parents=True)
    for path in (DATA / "opt_freq").glob("step1_0*"):
        shutil.copy(path, child / path.name.replace("step1_0", "step1_0_m6_neg", 1))
    attempts.promote(attempt, step=1, source_id="0_m6_neg", target_id="0")
    ctx = _ctx(tmp_path, operation="opt_freq")
    (promoted,) = get_engine("xtb").parse_one(attempt.parent / "step1_0.out", "0", ctx)
    (original,) = get_engine("xtb").parse_one(child / "step1_0_m6_neg.out", "0_m6_neg", ctx)
    assert promoted.frequencies == original.frequencies is not None
    assert np.array_equal(promoted.positions, original.positions)
    assert promoted.forces_ev_per_a is not None


def test_the_engine_is_nms_capable_for_its_frequency_operations(tmp_path: Path):
    """``freq`` and ``opt_freq`` compute frequencies; xtb searches no transition state."""
    engine = get_engine("xtb")
    assert isinstance(engine, NmsCapableEngine)
    for operation, computes in (
        ("sp", False),
        ("opt_sp", False),
        ("freq", True),
        ("opt_freq", True),
    ):
        info = engine.nms_input_info(_ctx(tmp_path, operation=operation))
        assert info.computes_frequencies is computes
        assert info.is_transition_state is False


def test_the_viewer_reparse_reads_what_the_step_parse_read(tmp_path: Path):
    """``parse_frequency_output`` needs no context and finds the same files."""
    out = _staged(tmp_path, "opt_freq")
    engine = get_engine("xtb")
    assert isinstance(engine, XtbEngine)
    (viewer,) = engine.parse_frequency_output(out)
    (step,) = engine.parse_one(out, "0", _ctx(tmp_path, operation="opt_freq"))
    assert viewer.frequencies == step.frequencies and viewer.energy_hartree == step.energy_hartree


# ---------------------------------------------------------------------------
# The submission proof: the whole pipeline, a stub binary
# ---------------------------------------------------------------------------


def test_an_xtb_step_runs_end_to_end_through_local_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """prepare → generated script → local run → copy-back → parse → cache, argv asserted.

    The stub stands where the real ``xtb`` would: it checks the argv the run block promises
    (the structure's geometry, its xcontrol input, the run type, the threads), writes the
    contract fixture's stdout and leaves its ``gradient`` and ``xtbopt.xyz`` in the working
    directory, where xtb writes them. Everything else in the chain is the real thing,
    including the copy-back the parse depends on.
    """
    monkeypatch.delenv("XTBPATH", raising=False)
    case = DATA / "opt"
    stub = tmp_path / "bin" / "xtb"
    stub.parent.mkdir()
    stub.write_text(
        "#!/bin/bash\n"
        '[ -f "$1" ] || { echo "geometry $1 missing" >&2; exit 64; }\n'
        '[ "$2" = "--input" ] && [ -f "$3" ] || { echo "no xcontrol input: $*" >&2; exit 65; }\n'
        '[ "$6 $7 $8" = "--opt tight --grad" ] || { echo "run type: $*" >&2; exit 66; }\n'
        'case " $* " in *" -P 2 "*) ;; *) echo "threads: $*" >&2; exit 67;; esac\n'
        '[ "$XTBPATH" = "' + str(tmp_path / "share" / "xtb") + '" ] || exit 68\n'
        '[ "$4 $5" = "--namespace step1_0" ] || { echo "namespace: $*" >&2; exit 69; }\n'
        f'cp "{case / "step1_0.gradient"}" "{case / "step1_0.xtbopt.xyz"}" .\n'
        f'cat "{case / "step1_0.out"}"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "step1.xcontrol").write_text(_STARTER, encoding="utf-8")
    (template_dir / "cpu.slurm.header").write_text("#!/bin/bash\n", encoding="utf-8")
    seed = tmp_path / "input.xyz"
    shutil.copy(case / "step1_0_inp.xyz", seed)
    config = Config(
        template_dir=template_dir,
        output_dir=tmp_path / "outputs",
        input=seed,
        max_cores=2,
        dispatch="local",
        executables={"xtb": str(stub)},
        steps=[
            StepConfig(
                step=1, engine="xtb", operation="opt_sp", options={"cores": 2, "opt_level": "tight"}
            )
        ],
    )

    outcomes = pipeline.run(config)

    assert len(outcomes) == 1 and not outcomes[0].cache_hit
    (survivor,) = outcomes[0].state.structures
    assert survivor.converged is True and survivor.forces_ev_per_a is not None
    structure_dir = config.output_dir / "step1" / "0"
    assert (structure_dir / "step1_0.xtbopt.xyz").is_file()
    assert (structure_dir / "step1_0.gradient").is_file()
    script = (structure_dir / "step1_0.slurm").read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=1" in script and "#SBATCH --cpus-per-task=2" in script
    assert pipeline.run(config)[0].cache_hit, "a second run is a pure cache hit"


def test_the_seed_atoms_reach_xtb_through_the_geometry_file(tmp_path: Path):
    """``prepare`` writes the ``_inp.xyz`` the run block names, from the structure's atoms."""
    ctx = _ctx(tmp_path)
    get_engine("xtb").prepare(ctx)
    written = (ctx.step_dir / "0" / "step1_0_inp.xyz").read_text(encoding="utf-8").splitlines()
    assert written[0].strip() == "2"
    assert [float(v) for v in written[3].split()[1:]] == [0.74, 0.0, 0.0]
