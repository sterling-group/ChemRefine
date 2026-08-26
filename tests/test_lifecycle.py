"""Tests for the failure vocabulary and the shared attempt primitive.

:mod:`chemrefine.lifecycle` owns what counts as a failure (:func:`succeeded`,
:func:`failure_kind`), how one is recorded (:class:`Failure`, :class:`FailureRecord`,
the ``failed_jobs.json`` ledger), the ``on_failure`` policy, and the numbered-attempt
primitive both the convergence retry and NMS build on. It had no test file: these tests
were split between ``test_coverage_gaps.py`` and ``test_step.py``, so someone editing the
module found neither.

What stays elsewhere is deliberate. ``test_step.py``, ``test_nms.py`` and
``test_recovery.py`` drive ``run_step`` / ``run_nms`` / a recovery action and then assert
on the ledger — those are lifecycle tests that read this module's output, not tests of
this module. Only the direct callers moved here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine import filtering, lifecycle
from chemrefine.config import MinSample, StepConfig
from chemrefine.engines._job import build_structures
from chemrefine.engines.api import ParsedResult
from chemrefine.errors import ChemRefineError, OutputParseError
from chemrefine.state import (
    Failure,
    FailureKind,
    JobBatch,
    PipelineState,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)


def _struct(**kw) -> Structure:
    return Structure(id="0", atoms=Atoms("H", positions=[[0, 0, 0]]), **kw)


def _ctx(tmp_path: Path) -> StepContext:
    """A minimal context: these tests need somewhere to put a ledger, nothing more."""
    return StepContext(
        step_cfg=StepConfig(step=1, engine="fake", operation="opt_sp"),
        step_dir=tmp_path,
        template_dir=tmp_path,
        template=None,
        scratch_dir=None,
        prev_state=PipelineState(structures=()),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_failure_kind_branches():
    from chemrefine.lifecycle import failure_kind
    from chemrefine.state import FailureKind

    assert failure_kind(_struct(terminated_normally=False)) is FailureKind.NOT_TERMINATED_NORMALLY
    assert failure_kind(_struct(converged=False)) is FailureKind.NOT_CONVERGED
    assert failure_kind(_struct()) is FailureKind.FAILED


def test_failure_kind_values_are_the_human_wording():
    """The enum values *are* the messages, so the ledger stays readable and the
    recovery paths still branch on a name rather than on that wording."""
    from chemrefine.state import Failure, FailureKind

    assert FailureKind.NOT_CONVERGED.value == "did not converge"
    assert Failure("0", FailureKind.MISSING_OUTPUT, None).reason == "output missing"
    # A kind that carries detail appends it rather than replacing the name.
    detailed = Failure("0", FailureKind.UNPARSEABLE, None, detail="bad token at line 3")
    assert detailed.reason == "unparseable: bad token at line 3"


def test_failure_record_round_trips_through_the_ledger():
    from chemrefine.state import Failure, FailureKind, FailureRecord

    record = FailureRecord.of(Failure("7", FailureKind.NOT_CONVERGED, None))
    assert FailureRecord.from_json(record.to_json()) == record
    assert record.to_json()["kind"] == "did not converge"


def test_a_job_that_died_is_ledgered_as_not_terminated(
    tmp_path: Path, orca_error_termination: Path
):
    """The ledger must name the job, not the parser, when the job is what failed.

    Driven through the real ORCA engine over a captured ORCA 6.1.1 abort, because the
    thing under test is which kind a real failure ends up filed under.
    """
    from chemrefine.engines.api import get_engine

    out = orca_error_termination
    ctx = replace(_ctx(tmp_path), step_cfg=StepConfig(step=2, engine="orca", operation="opt_sp"))
    inputs = StepInputs(files=((tmp_path / f"{out.stem}.inp", out, "5-54"),))

    successes, failures = lifecycle.parse_with_failures(get_engine("orca"), inputs, ctx)

    assert successes == []
    assert [f.kind for f in failures] == [FailureKind.NOT_TERMINATED_NORMALLY]
    assert "error termination in Startup" in failures[0].reason
    assert "orca_startup: not found" in failures[0].reason


def test_parse_with_failures_records_unparseable(tmp_path: Path):
    out = tmp_path / "s.out"
    out.write_text("garbage", encoding="utf-8")

    class _Engine:
        def parse(self, inputs, ctx):
            raise OutputParseError("boom")

    inputs = StepInputs(files=((tmp_path / "s.inp", out, "0"),))
    successes, failures = lifecycle.parse_with_failures(_Engine(), inputs, _ctx(tmp_path))
    assert successes == []
    assert failures[0].reason.startswith("unparseable")


def test_a_fanout_failure_carries_its_lowest_energy_bad_frame(tmp_path: Path):
    """Among several bad frames, the *lowest-energy* one is the failure's geometry.

    That frame is what a convergence retry restarts from, what ``on_failure: best``
    backfills, and what the failure is classified by — so a job with two unconverged
    frames must be described by the better of them, not by whichever the parse
    happened to yield last.
    """
    out = tmp_path / "s.out"
    out.write_text("fine", encoding="utf-8")

    def _frame(sid: str, energy: float) -> Structure:
        return Structure(id=sid, atoms=Atoms("H"), energy_hartree=energy, converged=False)

    class _Engine:
        def parse(self, inputs, ctx):
            return StepResults(
                structures=(
                    Structure(id="0-0", atoms=Atoms("H"), energy_hartree=-2.0, converged=True),
                    _frame("0-1", -0.5),
                    _frame("0-2", -1.0),  # lower energy: the geometry the failure must carry
                )
            )

    inputs = StepInputs(files=((tmp_path / "s.inp", out, "0"),))
    successes, failures = lifecycle.parse_with_failures(_Engine(), inputs, _ctx(tmp_path))
    assert [s.id for s in successes] == ["0-0"]
    assert [f.sid for f in failures] == ["0"]
    assert failures[0].kind is FailureKind.NOT_CONVERGED
    assert failures[0].best is not None
    assert failures[0].best.id == "0-2"


def test_parse_with_failures_writes_nothing_and_parse_and_record_does(tmp_path: Path):
    """The one thing separating the two, pinned where it is decided.

    Both are the same ledger over the same jobs; only `_ResultLedger.read_only` withholds the
    result record. `test_rebuilding_does_not_rewrite_the_children_it_reads` covers the whole
    tree three layers up, where the consequence bites — this says which line owns it, so a
    flipped default fails next to the code that flipped it.

    Both halves matter: a parse that *fails* produces no structures and so writes nothing
    either way, which would make the read-only half pass for the wrong reason.
    """
    out = tmp_path / "s.out"
    out.write_text("fine", encoding="utf-8")

    class _Engine:
        def parse(self, inputs, ctx):
            return StepResults(structures=(Structure(id="0", atoms=Atoms("H")),))

    inputs = StepInputs(files=((tmp_path / "s.inp", out, "0"),))
    ctx = _ctx(tmp_path)
    before = {p for p in tmp_path.rglob("*") if p.is_file()}

    successes, _ = lifecycle.parse_with_failures(_Engine(), inputs, ctx)
    assert [s.id for s in successes] == ["0"], "it really did parse a structure to record"
    assert {p for p in tmp_path.rglob("*") if p.is_file()} == before, "read-only wrote a record"

    lifecycle.parse_and_record(_Engine(), inputs, ctx)
    assert {p for p in tmp_path.rglob("*") if p.is_file()} > before, "no record was written"


# ---------------------------------------------------------------------------
# The convergence retry — a re-run must still be the same structure
# ---------------------------------------------------------------------------


class _FlakyEngine:
    """Reports NOT_CONVERGED once, then succeeds — the shape ``retry_unconverged`` handles.

    Assembles its results through :func:`~chemrefine.engines._job.build_structures`, like
    every real job engine, because that is where a re-run's lineage is resolved: it reads
    the parent back off ``ctx.prev_state``, so whatever the retry submits decides what the
    re-parsed structure descends from.

    ``parses`` counts **per structure**, not per call: ``lifecycle._parse_job`` invokes
    ``parse`` once per job, so a single counter would flip to converged partway through the
    first batch and only the first structure would ever fail.

    ``submissions`` records the structure ids of each :meth:`submit` call, which is what
    tells one batched retry from several serial ones — the two are indistinguishable in the
    structures that come back.
    """

    name = "flaky"

    def __init__(self) -> None:
        self.parses: dict[str, int] = {}
        self.submissions: list[list[str]] = []

    def prepare(self, ctx: StepContext) -> StepInputs:
        files = []
        for s in ctx.prev_state.structures:
            inp = ctx.step_dir / s.id / f"step1_{s.id}.inp"
            inp.parent.mkdir(parents=True, exist_ok=True)
            inp.write_text("input", encoding="utf-8")
            out = inp.with_suffix(".out")
            out.write_text("output", encoding="utf-8")
            files.append((inp, out, s.id))
        return StepInputs(files=tuple(files))

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        self.submissions.append([sid for _inp, _out, sid in inputs.files])
        return JobBatch(jobs={})

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        parsed = []
        for _inp, _out, sid in inputs.files:
            self.parses[sid] = self.parses.get(sid, 0) + 1
            parsed.append(
                (
                    sid,
                    [
                        ParsedResult(
                            symbols=("H",),
                            positions=np.zeros((1, 3)),
                            energy_hartree=-1.0,
                            forces_ev_per_a=None,
                            converged=self.parses[sid] > 1,
                            terminated_normally=True,
                        )
                    ],
                )
            )
        return build_structures(parsed, ctx.prev_state)


def test_a_retried_structure_keeps_its_lineage(tmp_path: Path):
    """A NOT_CONVERGED retry must come back as the same structure, parent and all.

    The retry was the one place a ``Structure`` was rebuilt from parts, and it rebuilt it
    without ``parent_id``. ``build_structures`` then read the parent back off the
    (rebuilt) prev_state and found none, so a structure that merely needed a second
    attempt came out of the step an orphan — permanently, into the cache.
    """
    child = Structure(id="7", atoms=Atoms("H", positions=[[0, 0, 0]]), parent_id="PARENT-42")
    ctx = replace(_ctx(tmp_path), prev_state=PipelineState(structures=(child,)))
    engine = _FlakyEngine()

    successes, failures = lifecycle.submit_and_parse(engine, ctx, [child])
    assert [f.kind for f in failures] == [FailureKind.NOT_CONVERGED]
    assert failures[0].best is not None and failures[0].best.parent_id == "PARENT-42"

    successes, failures = lifecycle.retry_unconverged(engine, ctx, successes, failures)

    assert failures == []
    assert [s.id for s in successes] == ["7"]
    assert successes[0].parent_id == "PARENT-42"


def test_unconverged_retries_go_out_as_one_batch(tmp_path: Path):
    """Two unconverged structures, one retry submission — so ``max_cores`` can apply.

    The core budget lives in a :class:`~chemrefine.throttle.Throttler` built per
    :func:`~chemrefine.engines._execution.run_batch` call, which blocks until *its* jobs
    finish. Retrying structure by structure therefore hands it one job at a time: the second
    is not submitted until the first has finished, and a budget sized for eight concurrent
    jobs runs one. Nothing in the returned structures distinguishes the two orderings — both
    give back the same successes — so what has to be asserted is that a single submission
    carried both.
    """
    atoms = Atoms("H", positions=[[0, 0, 0]])
    children = (
        Structure(id="7", atoms=atoms, parent_id="P"),
        Structure(id="9", atoms=atoms, parent_id="P"),
    )
    ctx = replace(_ctx(tmp_path), prev_state=PipelineState(structures=children))
    engine = _FlakyEngine()

    successes, failures = lifecycle.submit_and_parse(engine, ctx, list(children))
    assert [f.kind for f in failures] == [FailureKind.NOT_CONVERGED] * 2

    successes, failures = lifecycle.retry_unconverged(engine, ctx, successes, failures)

    assert failures == []
    assert sorted(s.id for s in successes) == ["7", "9"]
    assert engine.submissions[1:] == [["7", "9"]], "one retry submission carrying both"
    assert all((tmp_path / sid / "attempt1").is_dir() for sid in ("7", "9"))


def test_an_orphaned_retry_would_change_which_structures_survive():
    """Why the lineage matters: ``by_parent`` groups on ``parent_id or id``.

    A structure whose parent is missing is indistinguishable from a seed, so it forms its own
    singleton group, cannot be outranked by its siblings, and survives a filter that should
    have discarded it.
    """
    sample = MinSample(method="min", count=1, by_parent=True)

    def survivors(*structures: Structure) -> list[str]:
        kept = filtering.apply(StepResults(structures=structures), sample)
        return sorted(s.id for s in kept.structures)

    atoms = Atoms("H", positions=[[0, 0, 0]])
    good = Structure(id="a", atoms=atoms, parent_id="P", energy_hartree=-1.0)
    sibling = Structure(id="b", atoms=atoms, parent_id="P", energy_hartree=-0.5)

    assert survivors(good, sibling) == ["a"]
    assert survivors(good, replace(sibling, parent_id=None)) == ["a", "b"]


# ---------------------------------------------------------------------------
# The numbered-attempt primitive
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unsupported failure policy
# ---------------------------------------------------------------------------


def test_unknown_on_failure_policy_resolves_nothing(tmp_path: Path):
    """A policy with no branch must not quietly behave like one that has one.

    ``apply_failure_policy`` matches the closed ``on_failure`` literal with no catch-all, so
    adding a fourth value without a branch is a mypy error at the ``match`` — the guard that
    actually protects this, since the wrong outcome is silent: a policy meant to keep
    structures would drop them exactly as ``skip`` does, and every downstream step would run
    against the smaller set without anything to show for it.

    What is left to assert at runtime is that the unmatched case resolves *nothing* rather
    than falling into a neighbour's branch, so the mistake surfaces where it happens. The
    value has to be forced past validation, which is the point: the config layer is what
    makes this unreachable in a real run.
    """
    ctx = _ctx(tmp_path)
    bogus = ctx.step_cfg.model_copy(update={"on_failure": "retry-all"})
    failure = Failure("0", FailureKind.MISSING_OUTPUT, None)

    resolved = lifecycle.apply_failure_policy([_struct()], [failure], ctx, bogus)

    assert resolved is None, "no branch ran, so no survivor set was chosen"


# ---------------------------------------------------------------------------
# Streaming — one sink, two schedulers
# ---------------------------------------------------------------------------


class _Recorder:
    """A job engine reduced to a canned verdict per structure.

    ``fanout`` makes a structure parse into several frames, which is what separates the
    interesting cases from the 1:1 ones: a fan-out job's *best* geometry is a child, so its
    re-run carries an id the step's parents do not contain.
    """

    name = "recorder"

    def __init__(self, unconverged: set[str], fanout: dict[str, int] | None = None) -> None:
        self.unconverged = set(unconverged)
        self.fanout = dict(fanout or {})
        self.parses: dict[str, int] = {}
        self.submitted: list[list[str]] = []

    def prepare(self, ctx: StepContext) -> StepInputs:
        files = []
        for s in ctx.prev_state.structures:
            inp = ctx.step_dir / s.id / f"step1_{s.id}.inp"
            inp.parent.mkdir(parents=True, exist_ok=True)
            inp.write_text("in", encoding="utf-8")
            out = inp.with_suffix(".out")
            out.write_text("out", encoding="utf-8")
            files.append((inp, out, s.id))
        return StepInputs(files=tuple(files))

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        self.submitted.append([sid for _i, _o, sid in inputs.files])
        return JobBatch(jobs={})

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        parsed = []
        for _inp, _out, sid in inputs.files:
            self.parses[sid] = self.parses.get(sid, 0) + 1
            frames = self.fanout.get(sid, 1)
            # The last frame of a fan-out is the one that fails, so `best` is a child id.
            parsed.append(
                (
                    sid,
                    [
                        ParsedResult(
                            symbols=("H",),
                            positions=np.zeros((1, 3)),
                            energy_hartree=-1.0 - i,
                            forces_ev_per_a=None,
                            converged=not (
                                sid in self.unconverged
                                and self.parses[sid] == 1
                                and i == frames - 1
                            ),
                            terminated_normally=True,
                        )
                        for i in range(frames)
                    ],
                )
            )
        return build_structures(parsed, ctx.prev_state)


class _StreamingRecorder(_Recorder):
    """The same engine, reporting its own completions — **in reverse**.

    Reverse on purpose: it satisfies `StreamingSubmit`, and it makes completion order differ
    from manifest order so anything that leaked the former into results shows up.
    """

    def submit_streaming(self, inputs: StepInputs, ctx: StepContext, sink) -> JobBatch:
        pending = list(inputs.files)
        while pending:
            self.submitted.append([sid for _i, _o, sid in pending])
            nxt: list = []
            for job in reversed(pending):
                nxt.extend(sink.on_complete(job))
            pending = nxt
        return JobBatch(jobs={})


def _run(engine, tmp_path: Path, ids: tuple[str, ...]):
    seeds = tuple(
        Structure(id=i, atoms=Atoms("H", positions=[[0, 0, 0]]), parent_id="P") for i in ids
    )
    ctx = replace(_ctx(tmp_path), prev_state=PipelineState(structures=seeds))
    return lifecycle.run_with_retries(engine, ctx, engine.prepare(ctx))


@pytest.mark.parametrize("cls", [_Recorder, _StreamingRecorder], ids=["batched", "streaming"])
def test_both_schedulers_agree(cls, tmp_path: Path):
    """The streaming and batched routes are one algorithm, not two that must be kept in step.

    They share the sink and the ledger and differ only in *when* a completion is reported, so
    this pins the thing that would break if that ever stopped being true.
    """
    engine = cls(unconverged={"1"})
    successes, failures = _run(engine, tmp_path, ("0", "1", "2"))

    assert [s.id for s in successes] == ["0", "1", "2"]
    assert failures == []
    assert (tmp_path / "1" / "attempt1").is_dir()
    assert engine.parses["1"] == 2  # ran once, retried once


def test_results_come_back_in_manifest_order_not_completion_order(tmp_path: Path):
    """Completion order is nondeterministic; the step fingerprint is not.

    `StepKey.of` composes it from the row keys *in order*, so the same structures in a
    different order are a different step. Emitting results as they land would give the
    *next* step a new fingerprint on every run and invalidate its cache for nothing.
    `_StreamingRecorder` completes in reverse, so a ledger that appended as it went would
    show it.
    """
    successes, _failures = _run(_StreamingRecorder(unconverged=set()), tmp_path, ("0", "1", "2"))

    assert [s.id for s in successes] == ["0", "1", "2"]


def test_a_fanout_retry_keeps_its_lineage(tmp_path: Path):
    """The retry of a fan-out job must be parsed against its own structure, not the step's.

    Its id is the best geometry's — a *child* the step's parents do not contain — so parsing
    it with the step context would find no parent and silently emit an orphan. A 1:1 retry
    would survive that mistake by coincidence, which is why this case is the one that guards
    it.
    """
    engine = _StreamingRecorder(unconverged={"0"}, fanout={"0": 3})
    successes, failures = _run(engine, tmp_path, ("0",))

    assert failures == []
    retried = [s for s in successes if s.id == "0-2"]
    assert retried and retried[0].parent_id == "0", "the retried child was orphaned"


def test_a_fanout_retry_does_not_discard_its_siblings(tmp_path: Path):
    """One bad frame must not cost the good ones.

    A GOAT job with usable conformers and one stubborn frame contributes both successes and a
    failure; the retry has to *add* to the survivors, not replace them.
    """
    engine = _StreamingRecorder(unconverged={"0"}, fanout={"0": 3})
    successes, _failures = _run(engine, tmp_path, ("0",))

    assert sorted(s.id for s in successes) == ["0-0", "0-1", "0-2"]


def test_a_structure_is_retried_at_most_once(tmp_path: Path):
    """A queue that feeds itself needs the budget stated; the old single pass got it free."""
    engine = _StreamingRecorder(unconverged=set())
    engine.parse = lambda inputs, ctx: build_structures(  # type: ignore[method-assign]
        [
            (
                sid,
                [
                    ParsedResult(
                        symbols=("H",),
                        positions=np.zeros((1, 3)),
                        energy_hartree=-1.0,
                        forces_ev_per_a=None,
                        converged=False,
                        terminated_normally=True,
                    )
                ],
            )
            for _i, _o, sid in inputs.files
        ],
        ctx.prev_state,
    )
    successes, failures = _run(engine, tmp_path, ("0",))

    assert successes == []
    assert [f.kind for f in failures] == [FailureKind.NOT_CONVERGED]
    assert (tmp_path / "0" / "attempt1").is_dir()
    assert not (tmp_path / "0" / "attempt2").exists()


def test_the_retry_budget_belongs_to_a_scope_not_to_a_queue(tmp_path: Path):
    """Two scopes, one structure id: each gets its own attempt.

    One queue can carry jobs answering to several verdicts — a step's, and each NMS parent's
    round-2 children. Held on the sink instead, the budget would be shared: a child named
    after its parent, or two parents' children colliding, would silently spend each other's
    only retry. The counterpart matters as much — *within* a scope it is at most once, which
    is what stops a queue that feeds itself from feeding forever.
    """
    ctx = _ctx(tmp_path)
    one = lifecycle._JobScope(ctx, lifecycle._ResultLedger(StepInputs(files=())))
    other = lifecycle._JobScope(ctx, lifecycle._ResultLedger(StepInputs(files=())))

    assert one.claim_retry("0") is True
    assert one.claim_retry("0") is False, "a scope grants one attempt per origin"
    assert other.claim_retry("0") is True, "a different scope's budget is its own"


def test_a_missing_output_is_classified_at_once(tmp_path: Path):
    """A job out of the queue has copied back or never will — so there is nothing to wait for.

    It is also not retryable: there is no geometry to restart from, so it goes to the failure
    policy rather than round the queue again.
    """
    engine = _StreamingRecorder(unconverged=set())
    seeds = (Structure(id="0", atoms=Atoms("H", positions=[[0, 0, 0]])),)
    ctx = replace(_ctx(tmp_path), prev_state=PipelineState(structures=seeds))
    inputs = engine.prepare(ctx)
    inputs.files[0][1].unlink()

    successes, failures = lifecycle.run_with_retries(engine, ctx, inputs)

    assert successes == []
    assert [f.kind for f in failures] == [FailureKind.MISSING_OUTPUT]
    assert engine.submitted == [["0"]], "a missing output must not be re-queued"


def test_an_engine_with_no_prepared_jobs_still_gets_submitted(tmp_path: Path):
    """`mlip-train` prepares nothing and does its whole job inside `submit`.

    Gating submission on a non-empty batch would skip the training silently — no jobs, no
    error, an empty model. It contributes no structures either way, exactly as today.
    """

    class _TrainerShaped:
        name = "trainer-shaped"

        def __init__(self) -> None:
            self.submits = 0

        def prepare(self, ctx: StepContext) -> StepInputs:
            return StepInputs(files=())

        def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
            self.submits += 1
            return JobBatch(jobs={})

        def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
            raise AssertionError("never reached: there are no jobs to parse")

    engine = _TrainerShaped()
    ctx = _ctx(tmp_path)

    assert lifecycle.run_with_retries(engine, ctx, engine.prepare(ctx)) == ([], [])
    assert engine.submits == 1


# ---------------------------------------------------------------------------
# Child rounds — a finished structure earning work of its own
# ---------------------------------------------------------------------------


class _FakeRound:
    """A `ChildRound` that gives every named parent two children in a directory of its own."""

    def __init__(self, *, wanted: set[str], unconverged: set[str] | None = None) -> None:
        self.wanted = set(wanted)
        self.unconverged = set(unconverged or ())
        self.asked: list[str] = []
        self.settled_at: dict[str, list[str]] = {}
        self.order: list[str] = []

    def children_for(self, structure):
        self.asked.append(structure.id)
        if structure.id not in self.wanted:
            return None
        directory = structure_dir_of(structure)
        kids = tuple(
            Structure(
                id=f"{structure.id}_c{i}",
                atoms=Atoms("H", positions=[[0, 0, 0]]),
                parent_id=structure.id,
            )
            for i in (0, 1)
        )
        return lifecycle.ChildRun(directory, kids)

    def settled(self, origin_sid, successes, failures):
        self.order.append(f"settled:{origin_sid}")
        self.settled_at[origin_sid] = [s.id for s in successes]


def structure_dir_of(structure):
    """Where `_FakeRound` puts a parent's children — the real shape, one level down."""
    return _ROUND_ROOT / structure.id / "attempt1"


_ROUND_ROOT = Path()


def _run_with_round(cls, tmp_path: Path, ids, round_):
    global _ROUND_ROOT
    _ROUND_ROOT = tmp_path
    engine = cls(unconverged=round_.unconverged)
    seeds = tuple(
        Structure(id=i, atoms=Atoms("H", positions=[[0, 0, 0]]), parent_id="P") for i in ids
    )
    ctx = replace(_ctx(tmp_path), prev_state=PipelineState(structures=seeds))
    out = lifecycle.run_with_retries(engine, ctx, engine.prepare(ctx), children=round_)
    return engine, out


@pytest.mark.parametrize("cls", [_Recorder, _StreamingRecorder], ids=["batched", "streaming"])
def test_a_child_round_lands_in_its_own_ledger(cls, tmp_path: Path):
    """Children are the parent's results, not the step's.

    A step whose ledger held them would emit a survivor per child and fingerprint the next
    step on geometries it discarded.
    """
    round_ = _FakeRound(wanted={"1"})
    _engine, (successes, failures) = _run_with_round(cls, tmp_path, ("0", "1"), round_)

    assert [s.id for s in successes] == ["0", "1"], "no child reached the step's results"
    assert failures == []
    assert round_.settled_at == {"1": ["1_c0", "1_c1"]}


@pytest.mark.parametrize("cls", [_Recorder, _StreamingRecorder], ids=["batched", "streaming"])
def test_a_child_is_retried_once_in_its_own_directory(cls, tmp_path: Path):
    """A child's re-run nests under the round's directory, not the step's.

    `ctx.step_dir` is what `archive_previous` and `prepare` nest by, so the scope's context is
    what puts a child's second attempt beside its first instead of beside the step's.
    """
    round_ = _FakeRound(wanted={"0"}, unconverged={"0_c0"})
    _engine, (successes, _f) = _run_with_round(cls, tmp_path, ("0",), round_)

    assert [s.id for s in successes] == ["0"]
    assert (tmp_path / "0" / "attempt1" / "0_c0" / "attempt1").is_dir()
    assert not (tmp_path / "0" / "attempt2").exists(), "the parent's own budget was untouched"
    assert round_.settled_at["0"] == ["0_c0", "0_c1"]


@pytest.mark.parametrize("cls", [_Recorder, _StreamingRecorder], ids=["batched", "streaming"])
def test_a_child_never_earns_a_round_of_its_own(cls, tmp_path: Path):
    """Rounds are second rounds; a child spawning one would recurse with nothing to stop it."""
    round_ = _FakeRound(wanted={"0", "0_c0", "0_c1"})
    _engine, _out = _run_with_round(cls, tmp_path, ("0",), round_)

    assert round_.asked == ["0"], "only root-scope structures are asked"
    assert set(round_.settled_at) == {"0"}


@pytest.mark.parametrize("cls", [_Recorder, _StreamingRecorder], ids=["batched", "streaming"])
def test_only_a_structure_that_succeeded_earns_a_round(cls, tmp_path: Path):
    """An unconverged frame is about to be replaced — its round starts on the retry.

    Fanning out first would spend a whole round on a geometry the retry supersedes, and would
    put the children in the very `attempt1/` the retry is about to archive round 1 into.
    """
    round_ = _FakeRound(wanted={"0"}, unconverged={"0"})
    _engine, (successes, _f) = _run_with_round(cls, tmp_path, ("0",), round_)

    assert [s.id for s in successes] == ["0"]
    # Asked once — after the retry landed, not for the unconverged first parse.
    assert round_.asked == ["0"]
    # The retry archived round 1 into attempt1, so the children went to attempt2.
    assert (tmp_path / "0" / "attempt1").is_dir()
    assert round_.settled_at["0"] == ["0_c0", "0_c1"]


@pytest.mark.parametrize("cls", [_Recorder, _StreamingRecorder], ids=["batched", "streaming"])
def test_a_round_settles_only_after_its_queue_drains(cls, tmp_path: Path):
    """A round settled early would resolve a parent from children still in flight.

    Stated as completeness rather than as ordering: *which* round settles first follows
    completion order and is nobody's business (`_resolve_all` reads the results back in
    manifest order), but a round arriving with one of its two children would be the bug.
    """
    round_ = _FakeRound(wanted={"0", "1"})
    engine, _out = _run_with_round(cls, tmp_path, ("0", "1"), round_)

    submitted_children = [b for b in engine.submitted if any("_c" in s for s in b)]
    assert submitted_children, "children were submitted"
    assert round_.settled_at == {"0": ["0_c0", "0_c1"], "1": ["1_c0", "1_c1"]}
    assert len(round_.order) == 2, "each round settles exactly once"


def test_run_child_rounds_submits_nothing_when_no_parent_earns_one(tmp_path: Path):
    """The entry point for a caller whose round 1 came off disk, with nothing to do."""
    round_ = _FakeRound(wanted=set())
    engine = _Recorder(unconverged=set())
    ctx = _ctx(tmp_path)
    parents = (Structure(id="0", atoms=Atoms("H", positions=[[0, 0, 0]])),)

    lifecycle.run_child_rounds(engine, ctx, parents, round_)

    assert engine.submitted == [], "an empty round must not submit an empty batch"
    assert round_.settled_at == {}


def test_retry_unconverged_leaves_a_failure_it_cannot_restart(tmp_path: Path):
    """Convergence-only: a crashed or geometry-less failure passes through untouched.

    It is handed to the `on_failure` policy instead. Re-running the identical input would
    fail the same way, and a failure with no geometry has nothing to restart from — so the
    early return is the whole of "this pass has nothing to do", and it must not submit.
    """
    engine = _Recorder(unconverged=set())
    seed = Structure(id="0", atoms=Atoms("H", positions=[[0, 0, 0]]))
    ctx = replace(_ctx(tmp_path), prev_state=PipelineState(structures=(seed,)))
    crashed = Failure("0", FailureKind.MISSING_OUTPUT, None)

    successes, failures = lifecycle.retry_unconverged(engine, ctx, [seed], [crashed])

    assert [s.id for s in successes] == ["0"]
    assert failures == [crashed]
    assert engine.submitted == [], "nothing to retry means nothing submitted"


def test_emit_raises_for_a_structure_that_was_never_reported(tmp_path: Path):
    """Dropping it silently would shrink the survivor set and move every later fingerprint.

    Unreachable through either scheduler — neither returns while work is outstanding — which
    is exactly why it is worth an error rather than a `.get(sid, ...)`: the day it happens,
    the alternative is a step that quietly produces one structure fewer.
    """
    manifest = StepInputs(files=((tmp_path / "a.inp", tmp_path / "a.out", "0"),))
    ledger = lifecycle._ResultLedger(manifest)

    with pytest.raises(ChemRefineError, match="submitted but never reported"):
        ledger.emit()


def test_a_retry_of_an_engine_that_prepares_the_wrong_number_of_jobs_says_so(tmp_path: Path):
    """One structure in, one job out — named, so a breach is not an `IndexError` three frames on.

    Every streaming engine is a `JobEngine`, whose `prepare` emits exactly one job per
    structure in `prev_state`. Indexing `[0]` blindly would turn a future engine that stopped
    doing that into a crash somewhere else entirely.
    """

    class _PreparesTwo(_Recorder):
        def prepare(self, ctx):
            files = super().prepare(ctx).files
            return StepInputs(files=(*files, *files))

    best = Structure(id="0", atoms=Atoms("H", positions=[[0, 0, 0]]))
    with pytest.raises(ChemRefineError, match="expected one prepared job, got 2"):
        lifecycle._retry_input(_PreparesTwo(unconverged=set()), _ctx(tmp_path), best)
