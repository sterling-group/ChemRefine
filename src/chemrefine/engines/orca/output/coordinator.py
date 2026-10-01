"""Coordinator for ORCA output parsing: read the ``.out`` once, assemble, dispatch.

The per-section extractors live in sibling modules — :mod:`geometry`, :mod:`energy`,
:mod:`forces`, :mod:`frequencies`, :mod:`status` — and the multi-structure parsers in
:mod:`ensembles`. This module is the thin coordinator: for a single-structure ``.out`` it reads
the file **once** and runs every extractor over the shared text to assemble one
:class:`~chemrefine.engines.api.ParsedResult` (geometry + energy + forces + thermochemistry +
status + frequencies, all in that single pass); it then dispatches by ``operation`` to the
single-structure assembler vs the :mod:`ensembles` parsers.

ExtOpt ``.extinp.tmp`` / ``.engrad`` round-trip helpers live in
:mod:`chemrefine.engines.orca.extopt.protocol`.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from chemrefine.engines.api import ParsedResult
from chemrefine.engines.orca.output import (
    energy,
    ensembles,
    forces,
    frequencies,
    geometry,
    status,
)
from chemrefine.errors import OutputParseError, OutputTerminationError

# Operations whose structure comes from the ``.out`` itself (not a sidecar). ``freq`` is an
# opt+freq run — its geometry/energy parse exactly like opt_sp.
_DFT_OPERATIONS = frozenset({"opt_sp", "sp", "freq"})
# A relaxed scan: also read from the ``.out``, but as every point of the path.
_PES_OPERATION = "pes"


def parse_dft(path: str | Path, *, src: str | None = None) -> list[ParsedResult]:
    """Parse a single-structure DFT ``.out`` (``opt_sp`` / ``sp`` / ``freq``).

    ``src`` is what error messages call the output — the path itself unless the caller
    names it otherwise (an engine names it relative to its step). The job's ``.err`` is
    read from beside ``path`` either way.
    """
    path = Path(path)
    return parse_dft_from_text(
        path.read_text(encoding="utf-8", errors="replace"),
        src=src or str(path),
        stderr=path.with_suffix(".err"),
    )


def parse_dft_from_text(
    text: str, *, src: str = "<text>", stderr: Path | None = None
) -> list[ParsedResult]:
    """Assemble one :class:`ParsedResult` from a single read of an ORCA ``.out``.

    Runs every section extractor over the shared ``text`` — energy, geometry, forces,
    thermochemistry, status, and (when a ``VIBRATIONAL FREQUENCIES`` block is present) the
    imaginary modes + normal-mode tensor — so each value is parsed exactly once. The
    structure is the last evaluation ORCA printed an energy for (:func:`_last_energy`):
    its geometry, energy, gradient, thermochemistry and frequencies are read from that
    evaluation alone (:func:`_evaluation_of`), its convergence from every verdict printed
    by its end — the optimiser judges a geometry before re-printing it for a final energy —
    and termination from the whole output. ``src`` only labels error messages, and
    ``stderr`` is the job's ``.err``, quoted when the run died. Raises
    :class:`OutputParseError` if the energy or coordinates are missing / malformed, or
    :class:`OutputTerminationError` when the reason they are missing is that ORCA died — see
    :func:`_unreadable`.
    """
    found = _last_energy(text)
    if found is None:
        raise _unreadable("no FINAL SINGLE POINT ENERGY", text, src, stderr)
    electronic, energy_at = found
    start, end = _evaluation_of(text, energy_at)
    evaluation = text[max(start, 0) : end]
    try:
        coords = geometry.parse_coordinates_from_text(evaluation)
    except ValueError as e:
        # Corrupt token (e.g. a ``*****`` overflow) — a per-file parse failure, not a crash.
        raise OutputParseError(f"malformed coordinate row in {src}: {e}") from e
    if coords is None:
        raise _unreadable("no CARTESIAN COORDINATES block", text, src, stderr)
    symbols, positions = coords
    if not symbols:
        raise _unreadable("CARTESIAN COORDINATES block has no atoms", text, src, stderr)
    try:
        # The atom count comes from the coordinates block just read, which is what lets the
        # gradient reader tell a skipped summary line from a lost atom row.
        force_vectors = forces.parse_forces_from_text(evaluation, n_atoms=len(symbols))
    except ValueError as e:
        # Held to the same rule as the coordinates above. A bare ValueError escaping here
        # would pass straight through `lifecycle._parse_job`, which contains only
        # `OutputParseError` — so one malformed row would end the whole run in a traceback
        # rather than becoming that structure's ledgered failure.
        raise OutputParseError(f"malformed gradient row in {src}: {e}") from e

    thermo = energy.parse_thermochemistry_from_text(evaluation, electronic_hartree=electronic)
    imaginary, table, modes = _parse_frequency_block(evaluation, n_atoms=len(symbols))
    return [
        ParsedResult(
            symbols=symbols,
            positions=positions,
            energy_hartree=electronic,
            forces_ev_per_a=force_vectors,
            terminated_normally=status.parse_terminated_normally(text),
            converged=status.parse_converged(text[:end]),
            gibbs_hartree=thermo.gibbs_hartree if thermo else None,
            enthalpy_hartree=thermo.enthalpy_hartree if thermo else None,
            energy_zpe_hartree=thermo.energy_zpe_hartree if thermo else None,
            imaginary_freqs=imaginary,
            frequencies=table,
            normal_modes=modes,
        )
    ]


class _SidecarReader(Protocol):
    """An ensemble reader: the sidecar to read, and what its messages call it."""

    def __call__(self, path: str | Path, *, src: str | None = None) -> list[ParsedResult]:
        """Read every frame of the sidecar at ``path``."""
        ...


class _Ensemble(NamedTuple):
    """How one multi-frame operation finds and reads its sidecar."""

    parse: _SidecarReader
    names: tuple[str, ...]
    """Sidecar suffixes to try in order; a filename changed across ORCA releases has more
    than one, and the first that exists wins."""


_ENSEMBLE_OPERATIONS: dict[str, _Ensemble] = {
    "goat": _Ensemble(ensembles.parse_goat_ensemble, (ensembles.GOAT_SUFFIX,)),
    "docker": _Ensemble(
        ensembles.parse_docker, (ensembles.DOCKER_SUFFIX, ensembles.DOCKER_SUFFIX_611)
    ),
    "solvator": _Ensemble(
        ensembles.parse_solvator, (ensembles.SOLVATOR_SUFFIX, ensembles.SOLVATOR_SUFFIX_611)
    ),
}
"""The sidecar-reading operations, as data — so the dispatch below is one branch that
applies the same termination rule to all of them rather than three that could diverge."""


def known_operations() -> frozenset[str]:
    """The ``operation:`` vocabulary: the dispatch's own sets, read when asked.

    Derived rather than restated, so a new operation is in the answer the moment its
    parser exists — this is what the GUI's operation dropdown, the schema document and the
    run's t=0 check serve.
    """
    return _DFT_OPERATIONS | {_PES_OPERATION} | frozenset(_ENSEMBLE_OPERATIONS)


def multi_structure_operations() -> frozenset[str]:
    """The operations whose output is read as more than one structure.

    Every operation the dispatch knows but the single-structure reads, which share one
    reader (:func:`parse_dft`) with each other and with a run kind that has no reader of
    its own yet. Derived like :func:`known_operations`, so a reader that joins the dispatch
    is in the answer the moment it exists.
    """
    return known_operations() - _DFT_OPERATIONS


_ERROR_TERMINATION_RE = re.compile(r"^ORCA finished by error termination in .*$", re.MULTILINE)
"""ORCA's own verdict when it aborts, naming the module it died in."""

_ERR_TAIL_LINES = 5
"""How much of the job's stderr to quote — enough for the message that caused the abort."""


def _unreadable(what: str, text: str, src: str, stderr: Path | None) -> OutputParseError:
    """Return the error for a section that is missing, saying *why* it is missing.

    A section can be absent because the parser cannot read it, or because the program
    never got far enough to write it. Those send a reader to different places, so when the
    output shows an abnormal termination this reports the run rather than the section, and
    :func:`chemrefine.lifecycle._parse_job` files it as ``NOT_TERMINATED_NORMALLY``.

    ORCA's abort banner names the module it died in but not the cause, which it writes to
    stderr — so the job's ``.err`` (``stderr``, which shares the output's stem) is quoted
    alongside it.
    Without that the message is "error termination in Startup" and the actual reason (a
    missing basis file, a helper binary that is not on ``PATH``) is in a file nothing points
    at.
    """
    if status.parse_terminated_normally(text):
        return OutputParseError(f"{what} in {src}")
    return OutputTerminationError(
        f"{_termination_detail(text)} — {what} in {src}{_stderr_tail(stderr)}"
    )


def _termination_detail(text: str) -> str:
    """ORCA's own abort line, or a plain statement when it printed none."""
    banner = _ERROR_TERMINATION_RE.search(text)
    return banner.group(0) if banner else "the run did not terminate normally"


def _stderr_tail(err_path: Path | None) -> str:
    """The last few non-empty lines of the job's ``.err``, or ``""`` if there is none."""
    if err_path is None:
        return ""
    try:
        text = err_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = [ln.strip() for ln in text.splitlines()]
    tail = [ln for ln in lines if ln][-_ERR_TAIL_LINES:]
    return f"; {err_path.name} ends: " + " | ".join(tail) if tail else ""


_COORDINATES_MARKER = "CARTESIAN COORDINATES (ANGSTROEM)"


def _last_energy(text: str) -> tuple[float, int] | None:
    """The energy of the last evaluation ORCA printed one for, and the offset it is at.

    An evaluation's energy is its ``FINAL SINGLE POINT ENERGY``, except in an ExtOpt
    optimisation after its first cycle, which prints its energy only as the optimiser's
    ``Current Energy`` (:func:`~chemrefine.engines.orca.output.energy.find_relaxation_energy`).
    Every cycle closes with that line, after the full-precision energy where there is one, so
    it is taken only when it belongs to a later geometry than the last
    ``FINAL SINGLE POINT ENERGY``. Read from that line alone, an ExtOpt run that stopped
    before converging — at its cycle limit, or with its external program failing — would be
    read at its first cycle, the geometry it was submitted with.
    """
    final = energy.find_final_energy(text)
    relaxed = energy.find_relaxation_energy(text)
    if relaxed is None or (
        final is not None and text.find(_COORDINATES_MARKER, final[1], relaxed[1]) < 0
    ):
        return final
    return relaxed


def _evaluation_of(text: str, energy_at: int) -> tuple[int, int]:
    """Where the evaluation of the energy printed at ``energy_at`` starts and ends.

    ORCA prints each geometry it evaluates before anything computed there — the SCF and its
    verdict, the energy, the gradient, a Hessian's frequencies and thermochemistry — and the
    optimiser's verdict on it before the next geometry. So an evaluation runs from the last
    geometry printed before its energy to the next geometry printed after it, or to the end
    of the output, and a value printed outside it describes another geometry.

    Taking "the last of each" instead pairs values from different evaluations whenever the
    output does not end on a complete one: a run killed mid-cycle has printed its next
    geometry but no energy for it, and a TS search's initial Hessian is not its result's. A
    converged native ``Opt`` re-prints its geometry for a final energy with no gradient, so
    its last gradient belongs to the cycle before and the result carries no forces — label
    such a step with a single point, which is what the training tutorial does. A gradient
    inside the evaluation is its own: an ``EnGrad`` single point, an ExtOpt evaluation that
    prints one (its first and its final), the last cycle of an ``Opt`` that stopped at its
    cycle limit or was killed.
    """
    start = text.rfind(_COORDINATES_MARKER, 0, energy_at)
    end = text.find(_COORDINATES_MARKER, energy_at)
    return start, len(text) if end < 0 else end


def _parse_frequency_block(
    text: str, *, n_atoms: int
) -> tuple[dict[int, float] | None, dict[int, float] | None, NDArray[np.float64] | None]:
    """Imaginary modes, the whole table, and the normal-mode tensor from the shared text.

    ``None`` for all three when the output has no ``VIBRATIONAL FREQUENCIES`` block at all,
    or has the banner and no mode under it (distinct from ``{}`` = a freq calc with zero
    imaginary modes); the tensor is ``None`` when it's absent / unparseable.
    """
    if "VIBRATIONAL FREQUENCIES" not in text:
        return None, None, None
    imaginary = frequencies.parse_imaginary_frequencies_from_text(text)
    table = frequencies.parse_mode_table_from_text(text)
    if not table:
        # The banner is there and no mode parsed under it: no data, not a verified
        # minimum. ``{}`` for the imaginary set is the verdict "counted, and none" —
        # `nms._already_at_target` passes such a structure through as resolved and
        # `get_frequencies` reports it as a minimum — and a table that parsed nothing has
        # counted nothing.
        return None, None, None
    try:
        modes: NDArray[np.float64] | None = frequencies.parse_normal_modes_tensor_from_text(
            text, num_atoms=n_atoms
        )
    except ValueError:
        modes = None
    return imaginary, table, modes


def _stamp_run_status(frames: list[ParsedResult], text: str) -> list[ParsedResult]:
    """Copy the ``.out``'s termination verdict onto every sidecar frame.

    The ensemble sidecar carries geometries and energies but no run status, so frames
    parsed straight out of it default to ``terminated_normally=None`` — and
    :func:`chemrefine.lifecycle.succeeded` reads ``None`` as "not a failure signal". Without
    this, a GOAT job killed mid-run after writing a partial ensemble is an unconditional
    success: no ledger entry, no ``on_failure``, no signal to the user.

    Only ``terminated_normally`` is stamped. A whole-run ``NOT CONVERGED`` says nothing reliable
    about an individual pose — an ensemble is a *set* of stationary points, and one
    stubborn conformer must not condemn the rest — so ``converged`` stays ``None``
    (not reported per frame). :func:`ensembles.parse_pes_from_text` already does the
    same for scan points.
    """
    normally = status.parse_terminated_normally(text)
    return [replace(frame, terminated_normally=normally) for frame in frames]


def parse_output(path: str | Path, operation: str, *, src: str | None = None) -> list[ParsedResult]:
    """Pick the right parser based on the YAML ``operation`` string.

    ``opt_sp`` / ``sp`` / ``freq`` / ``pes`` read the ``.out`` directly; the multi-frame
    ensemble operations read their ``<base>.<suffix>`` sidecar (via :mod:`ensembles`) and
    then take their run status from the ``.out`` beside it (:func:`_stamp_run_status`).
    ``src`` is what error messages call the output, as for :func:`parse_dft`; a sidecar is
    called by its own name beside it.
    """
    if operation in _DFT_OPERATIONS:
        return parse_dft(path, src=src)
    if operation not in _ENSEMBLE_OPERATIONS and operation != _PES_OPERATION:
        # A name no engine offers is a config mistake, not a job that died — it must not
        # be reported as one, so it is raised before the termination check below.
        raise OutputParseError(f"unknown ORCA operation: {operation!r}")
    out_path = Path(path)
    label = Path(src) if src else out_path
    text = out_path.read_text(encoding="utf-8", errors="replace")
    try:
        if operation == _PES_OPERATION:
            return ensembles.parse_pes_from_text(text, src=str(label))
        suffixes = _ENSEMBLE_OPERATIONS[operation]
        sidecar = ensembles.ensemble_sidecar(out_path, *suffixes.names)
        frames = suffixes.parse(sidecar, src=str(label.with_name(sidecar.name)))
    except OutputParseError as e:
        # A sidecar that is absent, empty or frameless usually means the job never wrote
        # one, and the `.out` beside it says whether that is what happened. Reporting the
        # run rather than the missing file is what sends a reader to the right place.
        if status.parse_terminated_normally(text):
            raise
        raise OutputTerminationError(
            f"{_termination_detail(text)} — {e}{_stderr_tail(out_path.with_suffix('.err'))}"
        ) from e
    return _stamp_run_status(frames, text)
