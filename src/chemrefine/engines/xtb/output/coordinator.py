"""Coordinator for xtb output parsing: read stdout once, then each file it says xtb wrote.

xtb's results are spread over files: the energy and the verdicts on stdout (the ``.out``),
the gradient in ``gradient``, the optimised geometry in ``xtbopt.xyz``, the modes in
``g98.out``. The per-file grammars live in sibling modules — :mod:`.stdout`,
:mod:`.gradient`, :mod:`.geometry`, :mod:`.frequencies` — and this module is the thin
assembly over them into one :class:`~chemrefine.engines.api.ParsedResult`.

**The files are named in the run's namespace.** xtb is run with ``--namespace`` set to its
stdout's stem, so it writes ``<stem>.gradient``, ``<stem>.xtbopt.xyz``, ``<stem>.g98.out``,
while its stdout still names them as ``xtbopt.xyz`` and ``g98.out``. Each file is looked
for as ``<namespace>.<name>``, the namespace being the stdout's own stem — so an output
copied to another stem with its files (an NMS winner promoted to its parent's name) reads
its own copies, not the names its stdout was written with.

**A file is read on the output's evidence.** The optimised geometry and the modes are read
only when stdout says xtb wrote them, and the gradient only when its own energy and
coordinates are the structure's. A file left in the directory by an earlier calculation is
then never attached to this one: a stale ``xtbopt.xyz`` is not read for a single point, and
a gradient of another geometry or another method is refused rather than handed on as
forces.

A run that moves no atoms (a single point, a Hessian) reports no geometry: xtb computes on
the one it was given, read from the structure's input geometry (``input_geometry``).

One assembler serves every operation, because what a run computed is what its stdout
says it wrote; :mod:`chemrefine.engines.xtb.command` is where the operation decides what
runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chemrefine.engines.api import ParsedResult
from chemrefine.engines.xtb.output import frequencies, geometry, gradient, stdout
from chemrefine.errors import OutputParseError, OutputTerminationError
from chemrefine.quantities import BOHR_TO_ANGSTROM, HARTREE_PER_BOHR_TO_EV_PER_A

_POSITION_TOLERANCE_A = 1e-5
"""How closely the gradient file's coordinates must match the structure's (Å). The input
geometry is written to six decimals, and xtb's own bohr differs from CODATA's in the eighth
significant figure, so a gradient of this geometry agrees to about 1e-6 Å."""

_ENERGY_TOLERANCE_HARTREE = 1e-8
"""How closely the gradient file's energy must match stdout's (it prints eleven decimals)."""


def _unreadable(what: str, text: str, src: str) -> OutputParseError:
    """The error for something missing, saying whether xtb died before writing it.

    An output whose run ended in xtb's fatal-error block, or was cut off before xtb's
    closing line, is the run's failure rather than the parser's:
    :class:`~chemrefine.errors.OutputTerminationError`, filed as
    ``NOT_TERMINATED_NORMALLY``, carrying xtb's own reasons when it printed them.
    """
    if stdout.parse_terminated_normally(text):
        return OutputParseError(f"{what} in {src}")
    reasons = stdout.fatal_reasons(text)
    said = f"xtb stopped: {'; '.join(reasons)}" if reasons else "xtb did not finish"
    return OutputTerminationError(f"{said} — {what} in {src}")


def _read_sibling(path: Path, *, what: str, text: str, src: str) -> str:
    """The text of a file the output depends on, or the error for one that is not there.

    Only a missing file is the output's problem; any other refusal of the read is the
    filesystem's, and leaves as the ``OSError`` it is
    (:meth:`~chemrefine.engines._job.JobEngine.parse_one`).
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise _unreadable(f"no {what} {path.name} beside the output", text, src) from None


def parse_xtb(
    path: str | Path, *, input_geometry: Path, src: str | None = None
) -> list[ParsedResult]:
    """Parse one xtb run from its stdout at ``path`` into a single-element ``[ParsedResult]``.

    The run's files are read from beside ``path``, in the namespace of its stem (a run
    whose stdout is ``step1_0.out`` was run with ``--namespace step1_0``);
    ``input_geometry`` is the XYZ file the run was given. ``src`` is what error messages
    call the output — the path itself unless the caller names it otherwise (an engine
    names it relative to its step).
    """
    out = Path(path)
    return parse_xtb_text(
        out.read_text(encoding="utf-8", errors="replace"),
        directory=out.parent,
        namespace=out.stem,
        input_geometry=input_geometry,
        src=src or str(out),
    )


def parse_xtb_text(
    text: str,
    *,
    directory: Path,
    namespace: str,
    input_geometry: Path,
    src: str = "<text>",
) -> list[ParsedResult]:
    """Assemble one :class:`ParsedResult` from xtb's stdout ``text`` and the files beside it.

    Raises :class:`OutputParseError` when the energy or the geometry is missing or does not
    read, when a file stdout names is missing or does not read, or when the gradient file
    is another calculation's; upgraded to :class:`OutputTerminationError` by
    :func:`_unreadable` when xtb did not finish.
    """
    energy = stdout.parse_total_energy(text)
    if energy is None:
        raise _unreadable("no TOTAL ENERGY", text, src)
    written = stdout.written_geometry(text)
    geometry_path = directory / f"{namespace}.{written}" if written else input_geometry
    what = "optimised geometry" if written else "input geometry"
    try:
        symbols, positions = geometry.parse_xyz_text(
            _read_sibling(geometry_path, what=what, text=text, src=src)
        )
    except ValueError as e:
        raise OutputParseError(f"malformed {what} {geometry_path.name} for {src}: {e}") from e
    forces = _forces(
        directory / f"{namespace}.{gradient.GRADIENT_FILE}",
        symbols=symbols,
        positions=positions,
        energy=energy,
        src=src,
    )
    thermo = stdout.parse_thermochemistry(text, electronic_hartree=energy)
    imaginary, table, modes = _frequencies(
        directory, namespace, text, n_atoms=len(symbols), src=src
    )
    return [
        ParsedResult(
            symbols=symbols,
            positions=positions,
            energy_hartree=energy,
            forces_ev_per_a=forces,
            converged=stdout.parse_converged(text),
            terminated_normally=stdout.parse_terminated_normally(text),
            gibbs_hartree=thermo.gibbs_hartree if thermo else None,
            enthalpy_hartree=thermo.enthalpy_hartree if thermo else None,
            energy_zpe_hartree=thermo.energy_zpe_hartree if thermo else None,
            imaginary_freqs=imaginary,
            frequencies=table,
            normal_modes=modes,
        )
    ]


def _forces(
    path: Path,
    *,
    symbols: tuple[str, ...],
    positions: NDArray[np.float64],
    energy: float,
    src: str,
) -> NDArray[np.float64] | None:
    """ASE-native forces (eV/Å) from the ``gradient`` file at ``path``, or ``None`` without one.

    ``F = -∂E/∂x``, converted from Eh/bohr. ``None`` means the run wrote no gradient (a
    CPCM-X single point), which is not an error. The file must be this structure's
    gradient: the same atoms, at its coordinates, at stdout's energy.
    """
    try:
        file_text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    try:
        cycle = gradient.parse_gradient_text(file_text)
    except ValueError as e:
        raise OutputParseError(f"malformed {path.name} beside {src}: {e}") from e
    same_atoms = tuple(s.capitalize() for s in symbols) == cycle.symbols
    if not same_atoms or not np.allclose(
        cycle.positions_bohr * BOHR_TO_ANGSTROM, positions, rtol=0.0, atol=_POSITION_TOLERANCE_A
    ):
        raise OutputParseError(
            f"{path.name} beside {src} is the gradient of another geometry, not of the "
            f"structure the output reports"
        )
    if abs(cycle.energy_hartree - energy) > _ENERGY_TOLERANCE_HARTREE:
        raise OutputParseError(
            f"{path.name} beside {src} is the gradient of another calculation: its energy "
            f"{cycle.energy_hartree!r} Eh is not the output's {energy!r} Eh"
        )
    return -cycle.gradient_hartree_per_bohr * HARTREE_PER_BOHR_TO_EV_PER_A


def _frequencies(
    directory: Path, namespace: str, text: str, *, n_atoms: int, src: str
) -> tuple[dict[int, float] | None, dict[int, float] | None, NDArray[np.float64] | None]:
    """The modes from the ``g98.out`` stdout says xtb wrote; all ``None`` when it wrote none."""
    name = stdout.written_frequencies(text)
    if name is None:
        return None, None, None
    path = directory / f"{namespace}.{name}"
    try:
        return frequencies.parse_g98_text(
            _read_sibling(path, what="frequency file", text=text, src=src), n_atoms=n_atoms
        )
    except ValueError as e:
        raise OutputParseError(f"malformed {path.name} beside {src}: {e}") from e
