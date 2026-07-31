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
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

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
_DFT_OPERATIONS = frozenset({"opt_sp", "dft", "sp", "freq"})
TEXT_BASED_OPERATIONS = _DFT_OPERATIONS | {"pes"}


def parse_dft(path: str | Path) -> list[ParsedResult]:
    """Parse a single-structure DFT ``.out`` (``opt_sp`` / ``sp`` / ``freq``)."""
    return parse_dft_from_text(
        Path(path).read_text(encoding="utf-8", errors="replace"), src=str(path)
    )


def parse_dft_from_text(text: str, *, src: str = "<text>") -> list[ParsedResult]:
    """Assemble one :class:`ParsedResult` from a single read of an ORCA ``.out``.

    Runs every section extractor over the shared ``text`` — energy, geometry, forces,
    thermochemistry, status, and (when a ``VIBRATIONAL FREQUENCIES`` block is present) the
    imaginary modes + normal-mode tensor — so each value is parsed exactly once. ``src`` only
    labels error messages. Raises :class:`OutputParseError` if the energy or coordinates are
    missing / malformed, or :class:`OutputTerminationError` when the reason they are missing
    is that ORCA died — see :func:`_unreadable`.
    """
    electronic = energy.parse_final_energy_from_text(text)
    if electronic is None:
        raise _unreadable("no FINAL SINGLE POINT ENERGY", text, src)
    try:
        coords = geometry.parse_coordinates_from_text(text)
    except ValueError as e:
        # Corrupt token (e.g. a ``*****`` overflow) — a per-file parse failure, not a crash.
        raise OutputParseError(f"malformed coordinate row in {src}: {e}") from e
    if coords is None:
        raise _unreadable("no CARTESIAN COORDINATES block", text, src)
    symbols, positions = coords
    if not symbols:
        raise _unreadable("CARTESIAN COORDINATES block has no atoms", text, src)
    try:
        force_vectors = forces.parse_forces_from_text(text)
    except ValueError as e:
        # Held to the same rule as the coordinates above. A bare ValueError escaping here
        # would pass straight through `lifecycle._parse_job`, which contains only
        # `OutputParseError` — so one malformed row would end the whole run in a traceback
        # rather than becoming that structure's ledgered failure.
        raise OutputParseError(f"malformed gradient row in {src}: {e}") from e

    thermo = energy.parse_thermochemistry_from_text(text, electronic_hartree=electronic)
    imaginary, modes = _parse_frequency_block(text, n_atoms=len(symbols))
    return [
        ParsedResult(
            symbols=symbols,
            positions=positions,
            energy_hartree=electronic,
            forces_ev_per_a=force_vectors,
            terminated_normally=status.parse_terminated_normally(text),
            converged=status.parse_converged(text),
            gibbs_hartree=thermo.gibbs_hartree if thermo else None,
            enthalpy_hartree=thermo.enthalpy_hartree if thermo else None,
            energy_zpe_hartree=thermo.energy_zpe_hartree if thermo else None,
            imaginary_freqs=imaginary,
            normal_modes=modes,
        )
    ]


class _Ensemble(NamedTuple):
    """How one multi-frame operation finds and reads its sidecar."""

    parse: Callable[[Path], list[ParsedResult]]
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

_ERROR_TERMINATION_RE = re.compile(r"^ORCA finished by error termination in .*$", re.MULTILINE)
"""ORCA's own verdict when it aborts, naming the module it died in."""

_ERR_TAIL_LINES = 5
"""How much of the job's stderr to quote — enough for the message that caused the abort."""


def _unreadable(what: str, text: str, src: str) -> OutputParseError:
    """Return the error for a section that is missing, saying *why* it is missing.

    A section can be absent because the parser cannot read it, or because the program
    never got far enough to write it. Those send a reader to different places, so when the
    output shows an abnormal termination this reports the run rather than the section, and
    :func:`chemrefine.lifecycle._parse_job` files it as ``NOT_TERMINATED_NORMALLY``.

    ORCA's abort banner names the module it died in but not the cause, which it writes to
    stderr — so the job's ``.err``, which shares the output's stem, is quoted alongside it.
    Without that the message is "error termination in Startup" and the actual reason (a
    missing basis file, a helper binary that is not on ``PATH``) is in a file nothing points
    at.
    """
    if status.parse_terminated_normally(text):
        return OutputParseError(f"{what} in {src}")
    return OutputTerminationError(
        f"{_termination_detail(text)} — {what} in {src}{_stderr_tail(src)}"
    )


def _termination_detail(text: str) -> str:
    """ORCA's own abort line, or a plain statement when it printed none."""
    banner = _ERROR_TERMINATION_RE.search(text)
    return banner.group(0) if banner else "the run did not terminate normally"


def _stderr_tail(src: str) -> str:
    """The last few non-empty lines of the job's ``.err``, or ``""`` if there is none."""
    err_path = Path(src).with_suffix(".err")
    try:
        lines = [ln.strip() for ln in err_path.read_text(errors="replace").splitlines()]
    except OSError:
        return ""
    tail = [ln for ln in lines if ln][-_ERR_TAIL_LINES:]
    return f"; {err_path.name} ends: " + " | ".join(tail) if tail else ""


def _parse_frequency_block(
    text: str, *, n_atoms: int
) -> tuple[dict[int, float] | None, NDArray[np.float64] | None]:
    """Imaginary modes + normal-mode tensor from the shared text, or ``(None, None)``.

    ``None`` for both when the output has no ``VIBRATIONAL FREQUENCIES`` block at all (distinct
    from ``{}`` = a freq calc with zero imaginary modes); the tensor is ``None`` when it's
    absent / unparseable.
    """
    if "VIBRATIONAL FREQUENCIES" not in text:
        return None, None
    imaginary = frequencies.parse_imaginary_frequencies_from_text(text)
    try:
        modes: NDArray[np.float64] | None = frequencies.parse_normal_modes_tensor_from_text(
            text, num_atoms=n_atoms
        )
    except ValueError:
        modes = None
    return imaginary, modes


def parse_text(text: str, operation: str, *, src: str = "<text>") -> list[ParsedResult]:
    """Parse a ``.out``-based operation from already-read text (parse-once).

    Only the :data:`TEXT_BASED_OPERATIONS` are handled here; the sidecar ensemble operations
    use :func:`parse_output`.
    """
    op = operation.lower().replace("+", "_")
    if op in _DFT_OPERATIONS:
        return parse_dft_from_text(text, src=src)
    if op == "pes":
        return ensembles.parse_pes_from_text(text, src=src)
    raise OutputParseError(f"{operation!r} is not a text-based ORCA operation")


def _stamp_run_status(frames: list[ParsedResult], text: str) -> list[ParsedResult]:
    """Copy the ``.out``'s termination verdict onto every sidecar frame.

    The ensemble sidecar carries geometries and energies but no run status, so frames
    parsed straight out of it default to ``terminated_normally=None`` — and
    :func:`chemrefine.lifecycle.succeeded` reads ``None`` as "not a failure signal".
    A GOAT job killed mid-run after writing a partial ensemble was therefore an
    unconditional success: no ledger entry, no ``on_failure``, no signal to the user.

    Only ``terminated_normally`` is stamped. A whole-run ``NOT CONVERGED`` says nothing reliable
    about an individual pose — an ensemble is a *set* of stationary points, and one
    stubborn conformer must not condemn the rest — so ``converged`` stays ``None``
    (not reported per frame). :func:`ensembles.parse_pes_from_text` already does the
    same for scan points.
    """
    normally = status.parse_terminated_normally(text)
    return [replace(frame, terminated_normally=normally) for frame in frames]


def parse_output(path: str | Path, operation: str) -> list[ParsedResult]:
    """Pick the right parser based on the YAML ``operation`` string.

    ``opt_sp`` / ``sp`` / ``freq`` / ``pes`` read the ``.out`` directly; the multi-frame
    ensemble operations read their ``<base>.<suffix>`` sidecar (via :mod:`ensembles`) and
    then take their run status from the ``.out`` beside it (:func:`_stamp_run_status`).
    """
    op = operation.lower().replace("+", "_")
    if op in _DFT_OPERATIONS:
        return parse_dft(path)
    if op not in _ENSEMBLE_OPERATIONS and op != "pes":
        # A name no engine offers is a config mistake, not a job that died — it must not
        # be reported as one, so it is raised before the termination check below.
        raise OutputParseError(f"unknown ORCA operation: {operation!r}")
    out_path = Path(path)
    text = out_path.read_text(encoding="utf-8", errors="replace")
    try:
        if op == "pes":
            return ensembles.parse_pes_from_text(text, src=str(out_path))
        suffixes = _ENSEMBLE_OPERATIONS[op]
        frames = suffixes.parse(ensembles.ensemble_sidecar(out_path, *suffixes.names))
    except OutputParseError as e:
        # A sidecar that is absent, empty or frameless usually means the job never wrote
        # one, and the `.out` beside it says whether that is what happened. Reporting the
        # run rather than the missing file is what sends a reader to the right place.
        if status.parse_terminated_normally(text):
            raise
        raise OutputTerminationError(
            f"{_termination_detail(text)} — {e}{_stderr_tail(str(out_path))}"
        ) from e
    return _stamp_run_status(frames, text)
