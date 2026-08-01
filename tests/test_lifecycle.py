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
from ase import Atoms

from chemrefine import filtering, lifecycle
from chemrefine.config import MinSample, StepConfig
from chemrefine.engines._job import build_structures
from chemrefine.engines.api import ParsedResult
from chemrefine.errors import OutputParseError
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


def test_a_job_that_died_is_ledgered_as_not_terminated(tmp_path: Path):
    """The ledger must name the job, not the parser, when the job is what failed.

    Driven through the real ORCA engine over a captured ORCA 6.1.1 abort, because the
    thing under test is which kind a real failure ends up filed under.
    """
    from chemrefine.engines.api import get_engine

    fixture = Path(__file__).parent / "data" / "orca_failures" / "startup"
    out = tmp_path / "step2_5-54.out"
    out.write_text((fixture / "step2_5-54.out").read_text(), encoding="utf-8")
    out.with_suffix(".err").write_text((fixture / "step2_5-54.err").read_text(), encoding="utf-8")

    ctx = replace(_ctx(tmp_path), step_cfg=StepConfig(step=2, engine="orca", operation="opt_sp"))
    inputs = StepInputs(files=((tmp_path / "step2_5-54.inp", out, "5-54"),))

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


# ---------------------------------------------------------------------------
# The convergence retry — a re-run must still be the same structure
# ---------------------------------------------------------------------------


class _FlakyEngine:
    """Reports NOT_CONVERGED once, then succeeds — the shape ``retry_unconverged`` handles.

    Assembles its results through :func:`~chemrefine.engines._job.build_structures`, like
    every real job engine, because that is where a re-run's lineage is resolved: it reads
    the parent back off ``ctx.prev_state``, so whatever the retry submits decides what the
    re-parsed structure descends from.
    """

    name = "flaky"

    def __init__(self) -> None:
        self.parses = 0

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
        return JobBatch(jobs={})

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        self.parses += 1
        converged = self.parses > 1
        parsed = [
            (
                sid,
                [
                    ParsedResult(
                        symbols=("H",),
                        positions=np.zeros((1, 3)),
                        energy_hartree=-1.0,
                        forces_ev_per_a=None,
                        converged=converged,
                        terminated_normally=True,
                    )
                ],
            )
            for _inp, _out, sid in inputs.files
        ]
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
