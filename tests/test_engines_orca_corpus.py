"""Run the ORCA readers over every recorded real output — the whole corpus, not a sample.

The per-section tests use small synthetic snippets, which is right for edge cases but is
also how two defects survived: a fabricated fixture agreeing with a fabricated regex looks
exactly like a passing test. This file asserts the readers against every real ORCA output
already in the repo, streamed straight out of the ``tests/data/e2e/recordings``
archives (nothing is extracted to disk).

Every recorded run passed its live test, and the ones that failed failed on purpose — and
ORCA says so in its own words, which no reader here uses: an abort names the module and
source line it died in (:data:`ORCA_ABORT_RE`), and an optimisation cut short says it
"did not converge but reached the maximum number of optimization cycles"
(:data:`CYCLE_LIMIT_RE`). So the corpus-wide expectation stays simple, strong and
independent of the parser: an output reads as terminated exactly when ORCA printed no abort,
a terminated one reads as unconverged exactly when ORCA says it ran out of cycles, and every
frequency job yields a well-shaped normal-mode tensor. A reader that disagrees with ORCA
here is wrong about real ORCA, whatever the unit tests say.

Marked ``slow``, not ``integration``: it reads every ``.out`` out of the compressed archives,
which is a second or so, but it invokes no ORCA, no SLURM and no external service. That
distinction decides whether the file ever runs. ``integration`` is deselected by the default
``addopts``, which would exclude this — the strongest reader-versus-real-output assertion in
the suite — from every CI run, leaving a change to the recordings unnoticed until a release
gate. ``slow`` still runs by default.
"""

from __future__ import annotations

import re
import tarfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from chemrefine.engines.orca.output import status
from chemrefine.engines.orca.output.forces import parse_forces_from_text
from chemrefine.engines.orca.output.frequencies import (
    _NORMAL_MODES_MARKER,
    parse_normal_modes_tensor_from_text,
)
from chemrefine.engines.orca.output.geometry import parse_coordinates_from_text

pytestmark = pytest.mark.slow

RECORDINGS = Path(__file__).resolve().parent / "data" / "e2e" / "recordings"


_ORCA_BANNER = "O   R   C   A"
"""The banner every ORCA output opens with, an aborted one included."""


def _recorded_outputs() -> Iterator[tuple[str, str]]:
    """Yield ``(label, text)`` once per distinct ORCA ``.out`` in the recording archives.

    ORCA's only: another engine's ``.out`` (xtb's stdout, its ``g98.out``) is not this
    corpus's to judge, and is told apart by ORCA's banner, which it does not carry.

    Deduplicated by member name on purpose. ``tar`` permits the same path to appear several
    times in one archive, so a plain member walk can count one recorded output several times
    and size the "has the corpus shrunk?" guards below against an inflated number rather than
    against the distinct ORCA runs. A guard calibrated on duplicates is not a guard.
    """
    for archive in sorted(RECORDINGS.glob("*.tar.xz")):
        with tarfile.open(archive) as tf:
            seen: set[str] = set()
            for member in tf.getmembers():
                if not member.name.endswith(".out") or member.name in seen:
                    continue
                seen.add(member.name)
                handle = tf.extractfile(member)
                if handle is None:  # a directory member has no payload to read
                    continue
                text = handle.read().decode("utf-8", "replace")
                if _ORCA_BANNER in text:
                    yield f"{archive.stem}:{member.name}", text


def test_the_corpus_is_actually_there():
    """Guards against this file silently passing because it found nothing to check.

    Sized against the *distinct* runs in the archives, not the tar member count. Each floor
    here and below sits a few under the current corpus: losing any of the recordings that
    carry most of it trips the floor, and an output more or less in a re-recorded case does
    not.
    """
    assert sum(1 for _ in _recorded_outputs()) >= 39


ORCA_ABORT_RE = re.compile(
    r"\[file orca_[a-z_]+/[^\]]*, line \d+\]|Error \(ORCA|ORCA finished by error termination"
)
"""ORCA's own words for a run it aborted — the source line it died at, its error banner.

Read by no reader here. A run that died silently, killed with no word, prints neither this
nor the normal-termination banner; such an output must never be recorded, since nothing in
it says what happened."""

CYCLE_LIMIT_RE = re.compile(
    r"did\s+not\s+converge\s+but\s+reached\s+the\s+maximum\s+number\s+of\s+"
    r"optimization\s+cycles"
)
"""ORCA's own notice for an optimisation that hit its cycle limit — read by no reader here.

Whitespace-tolerant because ORCA wraps it differently by run type: an ``Opt Freq`` breaks the
line before "number", an ExtOpt run after it."""


def test_every_recorded_run_terminated_exactly_when_orca_printed_no_abort():
    """A reader's termination verdict agrees with ORCA's own words, both ways.

    A run the recording aborted on purpose reads as not terminated, and nothing else does —
    the half that catches an over-broad negative reading a finished run as dead.
    """
    wrong = [
        label
        for label, text in _recorded_outputs()
        if status.parse_terminated_normally(text) is bool(ORCA_ABORT_RE.search(text))
    ]
    assert wrong == []
    assert any(ORCA_ABORT_RE.search(text) for _label, text in _recorded_outputs()), (
        "no aborted run in the corpus — has the recording set changed?"
    )


def test_every_terminated_run_converged_exactly_when_orca_did_not_run_out_of_cycles():
    """No false failures across the corpus, and the cycle-limited runs read as failures.

    This is the assertion that catches a bare ``NOT CONVERGED`` matching ORCA's
    ``LOCALIZATION HAS NOT CONVERGED``, or any other over-broad negative — and, over the
    optimisations recorded running out of cycles, a negative that stopped matching.
    """
    terminated = [
        (label, text)
        for label, text in _recorded_outputs()
        if status.parse_terminated_normally(text)
    ]
    wrong = [
        label
        for label, text in terminated
        if status.parse_converged(text) is bool(CYCLE_LIMIT_RE.search(text))
    ]
    assert wrong == []
    assert any(CYCLE_LIMIT_RE.search(text) for _label, text in terminated), (
        "no cycle-exhausted optimisation in the corpus — has the recording set changed?"
    )


def test_pending_optimisation_lines_are_never_the_last_verdict_in_a_finished_run():
    """The premise "last verdict wins" rests on.

    ORCA prints "has not yet converged" after every non-final geometry cycle, so most
    optimisation outputs contain it. Treating it as a failure is only safe because a run
    that *did* finish always prints its success banner afterwards. If that ever stops
    holding, every optimisation in the corpus starts failing — so assert it directly
    rather than relying on the previous test to notice.
    """
    pending = [
        label
        for label, text in _recorded_outputs()
        if "The optimization has not yet converged" in text
    ]
    assert pending, "no optimisation outputs in the corpus — has the recording set changed?"
    finished = [
        text
        for _label, text in _recorded_outputs()
        if status.parse_terminated_normally(text) and not CYCLE_LIMIT_RE.search(text)
    ]
    assert all(status.parse_converged(text) for text in finished)


def test_every_frequency_output_yields_a_well_shaped_normal_mode_tensor():
    """The NMS displacement maths gets a real tensor, of the right shape, every time."""
    checked = 0
    for label, text in _recorded_outputs():
        if _NORMAL_MODES_MARKER not in text:
            continue
        coords = parse_coordinates_from_text(text)
        assert coords is not None, label
        n_atoms = len(coords[0])
        tensor = parse_normal_modes_tensor_from_text(text, num_atoms=n_atoms)
        assert tensor.shape[:2] == (n_atoms, 3), label
        assert tensor.shape[2] >= 1, label
        checked += 1
    assert checked >= 19, f"only {checked} frequency outputs found — corpus shrank?"


def test_every_recorded_gradient_reads_one_finite_row_per_atom():
    """The forces reader, held to the corpus like every other reader in this file.

    A numeric reader nothing checks against real output diverges from its siblings unnoticed,
    which is what this closes for the forces. Both guards are asserted from the outside:
    ``parse_forces_from_text`` refuses a non-finite component and refuses a row count that
    disagrees with the geometry, so a clean pass over every recorded gradient is
    what proves those guards do not fire on real ORCA — in particular that the summary lines
    ORCA closes each block with are still skipped rather than counted.
    """
    checked = 0
    for label, text in _recorded_outputs():
        coords = parse_coordinates_from_text(text)
        if coords is None:
            continue
        n_atoms = len(coords[0])
        forces = parse_forces_from_text(text, n_atoms=n_atoms)
        if forces is None:  # a plain single point writes no gradient block
            continue
        assert forces.shape == (n_atoms, 3), label
        assert np.isfinite(forces).all(), label
        checked += 1
    assert checked >= 30, f"only {checked} gradient outputs found — corpus shrank?"
