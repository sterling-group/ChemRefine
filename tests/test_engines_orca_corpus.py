"""Run the ORCA readers over every recorded real output — the whole corpus, not a sample.

The per-section tests use small synthetic snippets, which is right for edge cases but is
also how two defects survived: a fabricated fixture agreeing with a fabricated regex looks
exactly like a passing test. This file asserts the readers against every real ORCA output
already in the repo, streamed straight out of the ``tests/data/e2e/recordings``
archives (nothing is extracted to disk).

Every recorded run is a *successful* one — they were captured from passing live runs — so
the corpus-wide expectation is simple and strong: every output must read as terminated and
converged, and every frequency job must yield a well-shaped normal-mode tensor. A reader
that starts finding failures here is wrong about real ORCA, whatever the unit tests say.

Marked ``slow``, not ``integration``: it reads every ``.out`` out of six compressed archives,
which is a second or so, but it invokes no ORCA, no SLURM and no external service. That
distinction decides whether the file ever runs. ``integration`` is deselected by the default
``addopts``, which would exclude this — the strongest reader-versus-real-output assertion in
the suite — from every CI run, leaving a change to the recordings unnoticed until a release
gate. ``slow`` still runs by default.
"""

from __future__ import annotations

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


def _recorded_outputs() -> Iterator[tuple[str, str]]:
    """Yield ``(label, text)`` once per distinct ``.out`` in the recording archives.

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
                yield f"{archive.stem}:{member.name}", handle.read().decode("utf-8", "replace")


def test_the_corpus_is_actually_there():
    """Guards against this file silently passing because it found nothing to check.

    Sized against the *distinct* runs in the archives, not the tar member count.
    The corpus holds 18 today, down from 26 — the conformers case stopped running a
    DFT opt+freq on eleven GOAT conformers to prove that the next filter picks two.
    """
    assert sum(1 for _ in _recorded_outputs()) >= 17


def test_every_recorded_run_reads_as_successful():
    """No false failures across the corpus.

    This is the assertion that catches a bare ``NOT CONVERGED`` matching ORCA's
    ``LOCALIZATION HAS NOT CONVERGED``, or any other over-broad negative.
    """
    bad = [
        label
        for label, text in _recorded_outputs()
        if not (status.parse_terminated_normally(text) and status.parse_converged(text))
    ]
    assert bad == []


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
    assert all(status.parse_converged(text) for label, text in _recorded_outputs())


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
    assert checked >= 7, f"only {checked} frequency outputs found — corpus shrank?"


def test_every_recorded_gradient_reads_one_finite_row_per_atom():
    """The forces reader, held to the corpus like every other reader in this file.

    It was the one numeric reader here that nothing checked against real output, which is
    how it came to have neither of the two guards its siblings carry. Both are asserted from
    the outside: ``parse_forces_from_text`` refuses a non-finite component and refuses a row
    count that disagrees with the geometry, so a clean pass over every recorded gradient is
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
    assert checked >= 11, f"only {checked} gradient outputs found — corpus shrank?"
