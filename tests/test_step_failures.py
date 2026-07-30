"""Tests for the failure vocabulary and the shared attempt primitive.

:mod:`chemrefine.step_failures` owns what counts as a failure (:func:`succeeded`,
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

from pathlib import Path

from ase import Atoms

from chemrefine import step_failures
from chemrefine.config import StepConfig
from chemrefine.errors import OutputParseError
from chemrefine.state import PipelineState, StepContext, StepInputs, Structure


def _struct(**kw) -> Structure:
    return Structure(id="0", atoms=Atoms("H", positions=[[0, 0, 0]]), **kw)


def _ctx(tmp_path: Path) -> StepContext:
    """A minimal context: these tests need somewhere to put a ledger, nothing more."""
    return StepContext(
        step_cfg=StepConfig(step=1, engine="fake", operation="opt_sp"),
        step_dir=tmp_path,
        template_dir=tmp_path,
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
    from chemrefine.state import FailureKind
    from chemrefine.step_failures import failure_kind

    assert failure_kind(_struct(terminated_normally=False)) is FailureKind.NOT_TERMINATED
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


def test_parse_with_failures_records_unparseable(tmp_path: Path):
    out = tmp_path / "s.out"
    out.write_text("garbage", encoding="utf-8")

    class _Engine:
        def parse(self, inputs, ctx):
            raise OutputParseError("boom")

    inputs = StepInputs(files=((tmp_path / "s.inp", out, "0"),))
    successes, failures = step_failures.parse_with_failures(_Engine(), inputs, _ctx(tmp_path))
    assert successes == []
    assert failures[0].reason.startswith("unparseable")


# ---------------------------------------------------------------------------
# The numbered-attempt primitive
# ---------------------------------------------------------------------------
