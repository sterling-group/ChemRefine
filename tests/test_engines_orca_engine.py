"""Tests for ``OrcaEngine`` — prepare/parse paths; submission via SLURM mocks."""

from __future__ import annotations

import re
import shutil
from collections.abc import Collection
from pathlib import Path
from unittest.mock import patch

import pytest
import synthetic
from ase import Atoms

from chemrefine import cache, lifecycle, slurm
from chemrefine.config import StepConfig
from chemrefine.engines import _execution as submit
from chemrefine.engines.api import (
    NmsCapableEngine,
    get_engine,
    operation_notice,
    preflight_steps,
    resolve_operation,
)
from chemrefine.errors import ConfigError
from chemrefine.ids import require_template, step_template_path
from chemrefine.slurm import dispatch
from chemrefine.state import (
    FailureKind,
    JobBatch,
    PipelineState,
    StepContext,
    StepInputs,
    Structure,
)

FIXTURE = Path(__file__).parent / "data" / "engines" / "orca" / "dft" / "step1_0.out"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(
    tmp_path: Path,
    structures: tuple[Structure, ...],
    step_cfg: StepConfig | None = None,
    template: str = "! B3LYP def2-SVP\n%pal\n  nprocs 2\nend\n",
) -> StepContext:
    """Build a StepContext with a usable template + SLURM header on disk.

    The operation is resolved as the run resolves it, from the step and this template.
    """
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "step1.inp").write_text(template, encoding="utf-8")
    (template_dir / "cpu.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --partition=normal\n#SBATCH --time=24:00:00\nmodule load orca/6.0\n",
        encoding="utf-8",
    )
    step_cfg = step_cfg or StepConfig(step=1, engine="orca", operation="opt_sp")
    step_dir = tmp_path / "outputs" / step_cfg.dir_name()
    template_path = step_template_path(
        template_dir, step_cfg.step, template=step_cfg.template, suffix="inp"
    )
    return StepContext(
        step_cfg=step_cfg,
        step_dir=step_dir,
        template_dir=template_dir,
        template=template_path,
        operation=resolve_operation(get_engine("orca"), step_cfg, template_path),
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(structures=structures),
        charge=0,
        multiplicity=1,
        max_cores=4,
        slurm_template="cpu.slurm.header",
        executables={},
    )


def _seed_structure(sid: str = "0") -> Structure:
    return Structure(id=sid, atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))


def _drained(job_ids: Collection[str]) -> slurm.QueueState:
    """A poll in which everything asked about has left the queue."""
    return slurm.QueueState(frozenset(job_ids), frozenset())


def _still_queued(job_ids: Collection[str]) -> slurm.QueueState:
    """A poll in which everything asked about is still running."""
    return slurm.QueueState(frozenset(), frozenset(job_ids))


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def test_prepare_writes_inp_and_xyz_per_structure(tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure("0"), _seed_structure("1")))
    inputs = engine.prepare(ctx)
    assert len(inputs.files) == 2
    for inp_path, _out_path, sid in inputs.files:
        assert inp_path.exists()
        assert inp_path.suffix == ".inp"
        # Each structure has its own directory; the input geometry is _inp.xyz.
        assert inp_path.parent.name == sid
        assert (inp_path.parent / f"{inp_path.stem}_inp.xyz").exists()


def test_prepare_input_contains_charge_and_multiplicity(tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    # Override charge / multiplicity via context attributes
    ctx = StepContext(
        step_cfg=ctx.step_cfg,
        step_dir=ctx.step_dir,
        template_dir=ctx.template_dir,
        template=ctx.template,
        operation=ctx.operation,
        scratch_dir=ctx.scratch_dir,
        prev_state=ctx.prev_state,
        charge=-2,
        multiplicity=3,
        max_cores=ctx.max_cores,
        slurm_template=ctx.slurm_template,
        executables=ctx.executables,
    )
    inputs = engine.prepare(ctx)
    text = inputs.files[0][0].read_text(encoding="utf-8")
    assert "* xyzfile -2 3" in text


def test_prepare_clamps_template_pal_to_max_cores(tmp_path: Path):
    """A template PAL above ``max_cores`` is clamped in every generated ``.inp``."""
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    (ctx.template_dir / "step1.inp").write_text(
        "! B3LYP def2-SVP\n%pal\n  nprocs 16\nend\n", encoding="utf-8"
    )
    inputs = engine.prepare(ctx)
    text = inputs.files[0][0].read_text(encoding="utf-8")
    assert "nprocs 4" in text  # ctx.max_cores
    assert "nprocs 16" not in text


def test_run_block_keeps_orca_single_threaded_per_mpi_rank(tmp_path: Path):
    """ORCA parallelises via MPI ranks (``%pal nprocs``), so each rank stays OMP=1 —
    never OMP=pal (that would oversubscribe pal x pal threads)."""
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))  # template declares nprocs 2
    run_block = engine.run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    ).body
    assert "export OMP_NUM_THREADS=1" in run_block
    assert "OMP_NUM_THREADS=2" not in run_block


def test_orca_step_requests_no_gpu_and_keeps_global_header(tmp_path: Path):
    """ORCA is CPU/MPI: no GPU demand, and it keeps the global slurm_template."""
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    assert engine.gpus(ctx) == 0
    assert submit._header_name(engine, ctx) == ctx.slurm_template


def test_prepare_missing_template_raises(tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    # Delete the template after _ctx wrote it.
    (ctx.template_dir / "step1.inp").unlink()
    with pytest.raises(ConfigError, match="ORCA template not found"):
        engine.prepare(ctx)


def test_a_template_that_is_not_utf8_is_refused_by_every_reader(tmp_path: Path):
    """The inputs and the PAL count read the template through one reader, on one rule."""
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    (ctx.template_dir / "step1.inp").write_bytes("! B3LYP\n# Grüße\n".encode("latin-1"))
    with pytest.raises(ConfigError, match=r"ORCA template .* is not UTF-8 text"):
        engine.prepare(ctx)
    with pytest.raises(ConfigError, match=r"ORCA template .* is not UTF-8 text"):
        engine.pal(ctx)


def test_prepare_uses_step_specific_template_when_given(tmp_path: Path):
    engine = get_engine("orca")
    step_cfg = StepConfig(step=2, engine="orca", operation="opt_sp", template="custom.inp")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),), step_cfg=step_cfg)
    (ctx.template_dir / "custom.inp").write_text("! HF\n", encoding="utf-8")
    inputs = engine.prepare(ctx)
    text = inputs.files[0][0].read_text(encoding="utf-8")
    assert "! HF" in text


# ---------------------------------------------------------------------------
# submit / wait
# ---------------------------------------------------------------------------


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit")
def test_submit_creates_script_per_input_and_returns_batch(
    submit_mock, _finished_jobs, tmp_path: Path
):
    submit_mock.side_effect = ["1001", "1002"]
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure("0"), _seed_structure("1")))
    inputs = engine.prepare(ctx)
    batch = engine.submit(inputs, ctx)
    assert isinstance(batch, JobBatch)
    assert set(batch.jobs.values()) == {"1001", "1002"}
    # Each input has an accompanying .slurm script on disk.
    for inp, _out, _sid in inputs.files:
        assert inp.with_suffix(".slurm").exists()


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit", return_value="9001")
def test_submit_script_contains_orca_executable_invocation(_submit, _finished_jobs, tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    inputs = engine.prepare(ctx)
    engine.submit(inputs, ctx)
    script_text = inputs.files[0][0].with_suffix(".slurm").read_text(encoding="utf-8")
    assert "orca step1_0.inp" in script_text
    assert "$OUTPUT_DIR/step1_0.out" in script_text
    # ORCA's output_globs ClassVar flows through chemrefine.engines._execution.run_batch.
    assert "*.gbw" in script_text
    assert "*.hess" in script_text


def test_the_restart_and_readable_property_files_are_copied_back():
    """``.opt`` and ``.property.txt`` must survive the scratch dir.

    A live ORCA 6.1.1 ``! HF-3c Opt`` writes ``.bibtex``, ``.densities``, ``.densitiesinfo``,
    ``.engrad``, ``.gbw``, ``.opt``, ``.out``, ``.property.txt`` and the ``.xyz`` pair. Only
    the last few are result files. Undeclared, the optimisation restart file — the one thing
    that lets a stalled optimisation resume rather than start over — stays behind in
    ``$WORK_DIR`` and is deleted with it.
    """
    globs = get_engine("orca").output_globs
    assert "*.opt" in globs
    assert "*.property.txt" in globs
    assert "*.txt" not in globs, "a bare *.txt would sweep up whatever a template writes"


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit", return_value="9001")
def test_submit_missing_slurm_header_raises(_submit, _finished_jobs, tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    (ctx.template_dir / "cpu.slurm.header").unlink()
    inputs = engine.prepare(ctx)
    with pytest.raises(ConfigError):
        engine.submit(inputs, ctx)


@patch.object(dispatch, "sbatch_available", return_value=True)
@patch.object(slurm, "poll_jobs", side_effect=lambda ids: _drained(ids))
@patch.object(slurm, "submit_array", return_value="777")
@patch.object(slurm, "submit")
def test_submit_uses_one_array_when_slurm_array_set(
    submit_mock, submit_array_mock, _poll_jobs, _sbatch, tmp_path: Path
):
    """`slurm_array: true` on a SLURM host → one array submission, zero per-job
    sbatch calls, every input mapped to the parent id, one script + manifest."""
    from dataclasses import replace

    engine = get_engine("orca")
    ctx = replace(
        _ctx(tmp_path, structures=(_seed_structure("0"), _seed_structure("1"))),
        slurm_array=True,
    )
    inputs = engine.prepare(ctx)
    batch = engine.submit(inputs, ctx)
    submit_mock.assert_not_called()
    submit_array_mock.assert_called_once()
    kwargs = submit_array_mock.call_args.kwargs
    assert kwargs["n_tasks"] == 2
    assert kwargs["max_concurrent"] == 2  # max_cores=4 // pal=2
    assert set(batch.jobs.values()) == {"777"}
    script = ctx.step_dir / "step1_array.slurm"
    assert script.is_file()
    assert "$INP_NAME" in script.read_text(encoding="utf-8")
    manifest = kwargs["manifest"]
    assert manifest.read_text(encoding="utf-8").count("\n") == 2


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit_array")
@patch.object(slurm, "submit", return_value="9001")
def test_submit_ignores_slurm_array_locally(
    _submit, submit_array_mock, _finished_jobs, tmp_path: Path
):
    """Without sbatch on PATH the knob is inert — the local per-job path runs."""
    from dataclasses import replace

    engine = get_engine("orca")
    ctx = replace(_ctx(tmp_path, structures=(_seed_structure(),)), slurm_array=True)
    inputs = engine.prepare(ctx)
    with patch.object(dispatch, "sbatch_available", return_value=False):
        engine.submit(inputs, ctx)
    submit_array_mock.assert_not_called()


@patch.object(dispatch, "sbatch_available", return_value=True)
def test_submit_array_empty_batch_short_circuits(_sbatch, tmp_path: Path):
    """An empty batch returns an empty JobBatch without touching sbatch."""
    from dataclasses import replace

    from chemrefine.state import StepInputs

    engine = get_engine("orca")
    ctx = replace(_ctx(tmp_path, structures=()), slurm_array=True)
    assert engine.submit(StepInputs(files=()), ctx).jobs == {}


@patch.object(dispatch, "sbatch_available", return_value=True)
@patch.object(slurm, "submit_array", return_value="777")
def test_submit_array_polls_until_the_array_drains(_submit_array, _sbatch, tmp_path: Path):
    """The wait loop re-polls (with the SLURM cadence) while tasks remain."""
    from dataclasses import replace

    engine = get_engine("orca")
    ctx = replace(_ctx(tmp_path, structures=(_seed_structure(),)), slurm_array=True)
    inputs = engine.prepare(ctx)
    polls = [_still_queued({"777"}), _drained({"777"})]
    with (
        patch.object(slurm, "poll_jobs", side_effect=lambda ids: polls.pop(0)) as poll_mock,
        patch("chemrefine.slurm.dispatch.time.sleep") as sleep_mock,
    ):
        engine.submit(inputs, ctx)
    assert poll_mock.call_count == 2
    sleep_mock.assert_called_once()


@patch.object(dispatch, "sbatch_available", return_value=True)
def test_submit_array_missing_header_raises(_sbatch, tmp_path: Path):
    from dataclasses import replace

    engine = get_engine("orca")
    ctx = replace(_ctx(tmp_path, structures=(_seed_structure(),)), slurm_array=True)
    inputs = engine.prepare(ctx)
    (ctx.template_dir / "cpu.slurm.header").unlink()
    with pytest.raises(ConfigError):
        engine.submit(inputs, ctx)


# ---------------------------------------------------------------------------
# parse — uses the real fixture for the DFT happy path
# ---------------------------------------------------------------------------


def test_parse_uses_orca_fixture_as_output(tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure("0"),))
    inputs = engine.prepare(ctx)
    _inp, out, _sid = inputs.files[0]
    # Stage the fixture as the engine's expected output file.
    shutil.copy(FIXTURE, out)
    results = engine.parse(inputs, ctx)
    assert len(results.structures) == 1
    parsed = results.structures[0]
    assert parsed.id == "0"
    assert parsed.energy_hartree is not None
    assert abs(parsed.energy_hartree + 6044.555) < 1e-2
    assert len(parsed.atoms) > 50  # fixture molecule is large


def test_parse_unknown_operation_raises(tmp_path: Path):
    engine = get_engine("orca")
    step_cfg = StepConfig(step=1, engine="orca", operation="weird_op")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),), step_cfg=step_cfg)
    inputs = engine.prepare(ctx)
    shutil.copy(FIXTURE, inputs.files[0][1])
    from chemrefine.errors import OutputParseError

    with pytest.raises(OutputParseError):
        engine.parse(inputs, ctx)


def test_an_inferred_operation_without_a_reader_still_runs_and_reads_the_final_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A reaction path runs today, read as its final structure, and the run says so once
    when it starts; it fans out once the dispatch has a reader for ``irc`` — with no edit
    to the engine.

    A run kind is never refused for want of a reader: the template names a feature the
    program has, and the step has to run. The word is inferred, the notice the run's t=0
    walk logs says the final structure is all that will be read, and the same step fans out
    the day the reader joins the dispatch. An explicit ``operation:`` is still held to the
    readers' vocabulary.
    """
    from chemrefine.engines.orca.output import coordinator

    engine = get_engine("orca")
    inferred = StepConfig(step=1, engine="orca")
    ctx = _ctx(
        tmp_path,
        structures=(_seed_structure(),),
        step_cfg=inferred,
        template="! B3LYP def2-SVP IRC\n%pal\n  nprocs 2\nend\n",
    )
    assert ctx.operation == "irc" and ctx.template is not None
    inputs = engine.prepare(ctx)  # not refused
    out = inputs.files[0][1]
    shutil.copy(FIXTURE, out)

    parsed = engine.parse_one(out, "0", ctx)
    assert len(parsed) == 1 and parsed[0].positions.shape == (77, 3)
    template = require_template(ctx.template, label="ORCA")
    notice = operation_notice(engine, inferred, template)
    assert notice is not None and "'irc'" in notice and "final structure" in notice

    explicit = StepConfig(step=1, engine="orca", operation="irc")
    with pytest.raises(ConfigError, match=re.escape("unknown operation 'irc' for engine 'orca'")):
        preflight_steps([explicit], charge=0, multiplicity=1)  # explicit: held to the vocabulary

    monkeypatch.setattr(coordinator, "_DFT_OPERATIONS", coordinator._DFT_OPERATIONS | {"irc"})
    monkeypatch.setattr(type(engine), "operations", (*engine.operations, "irc"))
    assert len(engine.parse_one(out, "0", ctx)) == 1
    assert operation_notice(engine, inferred, template) is None


# ---------------------------------------------------------------------------
# The t=0 walk holds an explicit operation to what ORCA reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operation",
    ["docker", "freq", "goat", "opt_sp", "pes", "solvator", "sp", None, "OPT+SP", "GOAT"],
)
def test_the_preflight_walk_passes_every_operation_orca_reads(operation: str | None):
    """The vocabulary in any spelling the config folds, and an omitted one to infer."""
    preflight_steps(
        [StepConfig(step=1, engine="orca", operation=operation)], charge=0, multiplicity=1
    )


@pytest.mark.parametrize("operation", ["opt-sp", "dft"])
def test_the_preflight_walk_refuses_an_operation_orca_does_not_read(operation: str):
    """A typo'd ``operation:`` is refused from the config alone, naming the vocabulary.

    Unchecked, it is first refused by the *parser* — after every job in the step has run
    at full cost — and because the operation is part of every row key, correcting the typo
    re-keys the rows so no recovery command adopts the paid outputs. ``dft`` was never in
    ORCA's vocabulary, and is refused like any other unknown name.
    """
    step_cfg = StepConfig(step=2, engine="orca", operation=operation)
    with pytest.raises(ConfigError) as excinfo:
        preflight_steps([step_cfg], charge=0, multiplicity=1)
    assert str(excinfo.value) == (
        f"step 2: unknown operation {operation!r} for engine 'orca', which reads docker, "
        f"freq, goat, opt_sp, pes, solvator, sp. Correct it, or omit `operation:`."
    )


# ---------------------------------------------------------------------------
# operation is optional: explicit value wins, else the template is inspected
# ---------------------------------------------------------------------------


def test_parse_without_operation_infers_parser_from_template(tmp_path: Path):
    """A step with no ``operation`` parses via the template-inspected run type."""
    engine = get_engine("orca")
    ctx = _ctx(
        tmp_path,
        structures=(_seed_structure("0"),),
        step_cfg=StepConfig(step=1, engine="orca"),  # operation omitted
    )
    inputs = engine.prepare(ctx)
    shutil.copy(FIXTURE, inputs.files[0][1])
    # The default template (`! B3LYP def2-SVP`, no Opt) inspects to sp → parse_dft.
    assert ctx.operation == "sp"
    results = engine.parse(inputs, ctx)
    assert results.structures[0].energy_hartree == pytest.approx(-6044.555726221861)


def test_outputs_are_read_as_the_operation_the_step_resolved(tmp_path: Path):
    """A template edited while its step runs does not change how the step's outputs are read:
    each is read as the operation its row was keyed by."""
    engine = get_engine("orca")
    ctx = _ctx(
        tmp_path, structures=(_seed_structure(),), step_cfg=StepConfig(step=1, engine="orca")
    )
    assert ctx.operation == "sp" and ctx.template is not None
    inputs = engine.prepare(ctx)
    shutil.copy(FIXTURE, inputs.files[0][1])
    ctx.template.write_text("! GOAT XTB\n", encoding="utf-8")
    [parsed] = engine.parse_one(inputs.files[0][1], "0", ctx)
    assert parsed.energy_hartree == pytest.approx(-6044.555726221861)


def test_template_digest_tracks_template_contents(tmp_path: Path):
    """The template-content digest changes on edit and is empty when missing.

    The engine is not involved: the template is on the context, and the cache reads it
    from there. That is the point of the field.
    """
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    first = cache.template_digest(ctx.template)
    assert first  # non-empty for an existing template
    (ctx.template_dir / "step1.inp").write_text("! PBE def2-TZVP\n", encoding="utf-8")
    assert cache.template_digest(ctx.template) != first  # an edit re-runs the step
    (ctx.template_dir / "step1.inp").unlink()
    assert cache.template_digest(ctx.template) == ""  # missing template → empty digest


# ---------------------------------------------------------------------------
# Engine registration
# ---------------------------------------------------------------------------


def test_orca_engine_registered():
    from chemrefine.engines.api import ENGINES

    assert "orca" in ENGINES


def test_orca_engine_is_nms_capable():
    assert isinstance(get_engine("orca"), NmsCapableEngine)


# ---------------------------------------------------------------------------
# A crashed ensemble job reaches the failure ledger
# ---------------------------------------------------------------------------


def _goat_job(tmp_path: Path, *, terminated_normally: bool) -> tuple[StepContext, StepInputs]:
    """A GOAT step whose one structure has an ensemble sidecar and a `.out` on disk."""
    step_cfg = StepConfig(step=1, engine="orca", operation="goat")
    ctx = _ctx(tmp_path, (_seed_structure("0"),), step_cfg)
    job_dir = ctx.step_dir / "0"
    job_dir.mkdir(parents=True, exist_ok=True)
    body = "GOAT                             ...       75.347 sec\n"
    if terminated_normally:
        body += "                             ****ORCA TERMINATED NORMALLY****\n"
    out = job_dir / "step1_0.out"
    out.write_text(body, encoding="utf-8")
    (job_dir / "step1_0.finalensemble.xyz").write_text(
        "1\n  -40.123456   converged=true\n  C   0.000000   0.000000   0.000000\n",
        encoding="utf-8",
    )
    return ctx, StepInputs(files=((job_dir / "step1_0.inp", out, "0"),))


def test_completed_goat_job_is_a_success(tmp_path: Path):
    ctx, inputs = _goat_job(tmp_path, terminated_normally=True)
    successes, failures = lifecycle.parse_with_failures(get_engine("orca"), inputs, ctx)
    assert [s.id for s in successes] == ["0"]
    assert failures == []


def test_crashed_goat_job_is_ledgered_not_silently_accepted(tmp_path: Path):
    """The end-to-end shape of B2: no termination banner → a real, visible failure.

    A sidecar frame carrying ``terminated_normally=None`` must not be accepted by
    ``succeeded()``: a killed GOAT run would otherwise produce a clean success with an empty
    ledger, the partial ensemble flowing downstream as if complete.
    """
    ctx, inputs = _goat_job(tmp_path, terminated_normally=False)
    successes, failures = lifecycle.parse_with_failures(get_engine("orca"), inputs, ctx)
    assert successes == []
    assert [(f.sid, f.reason) for f in failures] == [("0", "did not terminate normally")]


_DIED = synthetic.ORCA_ERROR_TERMINATION_OUT
_FINISHED = "                             ****ORCA TERMINATED NORMALLY****\n"


@pytest.mark.parametrize(
    ("operation", "out_text", "sidecar", "kind", "named"),
    [
        # A single-structure output that died: named, and its `.err` quoted.
        ("opt_sp", _DIED, None, FailureKind.NOT_TERMINATED_NORMALLY, "0/step1_0.out"),
        # A scan that died before a point converged reads the same way as any other run
        # that died — it is read through the same route as every declared operation.
        ("pes", _DIED, None, FailureKind.NOT_TERMINATED_NORMALLY, "0/step1_0.out"),
        # A sidecar is named by its own file, beside the output, under the same step.
        (
            "goat",
            _FINISHED,
            "step1_0.finalensemble.xyz",
            FailureKind.UNPARSEABLE,
            "0/step1_0.finalensemble.xyz",
        ),
    ],
)
def test_a_failure_reason_names_its_output_relative_to_its_step(
    tmp_path: Path,
    operation: str,
    out_text: str,
    sidecar: str | None,
    kind: FailureKind,
    named: str,
):
    """The ledgered reason must still point at the file after the tree is copied or moved.

    Spelled absolute, it named the directory the step ran in — a cluster's scratch, another
    machine's home — so a tree copied off the cluster reported locations that are not there.
    The `.err` is still read from where it lies, and quoted.
    """
    ctx = _ctx(
        tmp_path,
        (_seed_structure("0"),),
        StepConfig(step=1, engine="orca", operation=operation),
    )
    job_dir = ctx.step_dir / "0"
    job_dir.mkdir(parents=True)
    out = job_dir / "step1_0.out"
    out.write_text(out_text, encoding="utf-8")
    out.with_suffix(".err").write_text(synthetic.ORCA_ERROR_TERMINATION_ERR, encoding="utf-8")
    if sidecar is not None:
        (job_dir / sidecar).write_text("no frames here\n", encoding="utf-8")

    _, failures = lifecycle.parse_with_failures(
        get_engine("orca"), StepInputs(files=((job_dir / "step1_0.inp", out, "0"),)), ctx
    )

    [failure] = failures
    assert failure.kind is kind
    assert f" {named}" in failure.detail
    assert str(tmp_path) not in failure.detail
    if kind is FailureKind.NOT_TERMINATED_NORMALLY:
        assert "step1_0.err ends: sh: 1: /opt/orca/orca_startup: not found" in failure.detail


# ---------------------------------------------------------------------------
# The configured executable reaches bash as one word
# ---------------------------------------------------------------------------


def test_run_block_quotes_an_executable_path_with_spaces(tmp_path: Path):
    """An unquoted path with a space silently becomes two words to bash."""
    from dataclasses import replace

    engine = get_engine("orca")
    ctx = replace(
        _ctx(tmp_path, structures=(_seed_structure(),)),
        executables={"orca": "/opt/ORCA 6.1.1/orca"},
    )
    run_block = engine.run_block(ctx, tmp_path / "step1_0.inp", tmp_path / "step1_0.out").body
    assert "'/opt/ORCA 6.1.1/orca' step1_0.inp" in run_block


def test_run_block_neutralises_shell_metacharacters_in_the_executable(tmp_path: Path):
    """The YAML is the user's own, but a stray metacharacter must not become a command."""
    from dataclasses import replace

    engine = get_engine("orca")
    ctx = replace(
        _ctx(tmp_path, structures=(_seed_structure(),)),
        executables={"orca": "/opt/orca; rm -rf /tmp/x"},
    )
    run_block = engine.run_block(ctx, tmp_path / "step1_0.inp", tmp_path / "step1_0.out").body
    assert "; rm -rf" not in run_block.replace("'/opt/orca; rm -rf /tmp/x'", "")
    assert "'/opt/orca; rm -rf /tmp/x'" in run_block


def test_plain_executable_name_is_not_needlessly_quoted(tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    assert (
        "orca step1_0.inp"
        in engine.run_block(ctx, tmp_path / "step1_0.inp", tmp_path / "step1_0.out").body
    )


# ---------------------------------------------------------------------------
# memory_mb — the SLURM request %maxcore implies
# ---------------------------------------------------------------------------


def test_memory_mb_inverts_the_75_percent_rule(tmp_path: Path):
    """The request sizes the allocation so the declared maxcore is 75% of it.

    %maxcore is a promise ORCA routinely overshoots per process, which is why qorca holds
    it to at most 75% of the granted memory; ceil(3000 x 2 / 0.75) = 8000.
    """
    ctx = _ctx(tmp_path, (_seed_structure("0"),))
    ctx.template.write_text(
        "! B3LYP def2-SVP\n%maxcore 3000\n%pal\n  nprocs 2\nend\n", encoding="utf-8"
    )
    assert get_engine("orca").memory_mb(ctx) == 8000


def test_memory_mb_uses_the_granted_core_count(tmp_path: Path):
    """The multiplier is the clamped PAL — the cores the job actually gets, not the ask."""
    ctx = _ctx(tmp_path, (_seed_structure("0"),))  # max_cores=4
    ctx.template.write_text(
        "! B3LYP def2-SVP\n%maxcore 1000\n%pal\n  nprocs 16\nend\n", encoding="utf-8"
    )
    assert get_engine("orca").memory_mb(ctx) == -(-(1000 * 4 * 4) // 3)  # ceil(1000*4/0.75)


def test_memory_mb_is_none_without_a_maxcore(tmp_path: Path):
    """No %maxcore, no request: the header's memory policy stands, as it always has."""
    assert get_engine("orca").memory_mb(_ctx(tmp_path, (_seed_structure("0"),))) is None
