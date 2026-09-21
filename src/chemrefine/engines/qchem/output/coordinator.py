"""Coordinator for Q-Chem output parsing: read the ``.out`` once, assemble, dispatch.

The per-section extractors live in sibling modules — :mod:`.geometry`, :mod:`.energy`,
:mod:`.forces`, :mod:`.frequencies`, :mod:`.status` — each the one place its section's
grammar is interpreted. This module is the thin assembly over them: it reads the file
**once**, runs every extractor over the shared text, builds one
:class:`~chemrefine.engines.api.ParsedResult`, and dispatches by ``operation``.

The ``operation:`` vocabulary is deliberately small and uniform: ``sp`` / ``opt_sp`` /
``freq`` all route to the one single-pass assembler, because the sections it scans are
the union of what those runs print and an absent section reads as "not computed". A
future operation whose output needs a parser of its own (an ``@@@`` chain split per job,
a PES-style scan) joins :data:`_QCHEM_OPERATIONS` with its parser — the dispatch is a
table, not a chain.
"""

from __future__ import annotations

from pathlib import Path

from chemrefine.engines.api import ParsedResult
from chemrefine.engines.qchem.output import energy, forces, frequencies, geometry, status
from chemrefine.errors import OutputParseError, OutputTerminationError

_QCHEM_OPERATIONS = frozenset({"sp", "opt_sp", "freq"})
"""Every ``operation:`` this family interprets — all served by the one assembler today."""


def known_operations() -> frozenset[str]:
    """The canonical ``operation:`` vocabulary — what a config may name.

    Derived from the dispatch set rather than restated, so an operation is in the
    answer the moment its parser exists. This is what the engine declares through
    :class:`~chemrefine.engines.api.OperationsDeclaring`, the GUI's dropdown offers, and
    the preflight refusal checks against.
    """
    return _QCHEM_OPERATIONS


def _unreadable(what: str, text: str, src: str) -> OutputParseError:
    """The error for a section that is missing, saying *why* it is missing.

    A section can be absent because the parser cannot read it, or because the program
    never got far enough to write it — and those send a reader to different places:
    :class:`~chemrefine.errors.OutputTerminationError` is filed as
    ``NOT_TERMINATED_NORMALLY``, the ledger entry that points at the job rather than
    the parser. Only an explicit ``False`` verdict makes that upgrade — ``None`` (no
    verdict reported) keeps the plain :class:`OutputParseError`, because no signal is
    not a death signal.
    """
    if status.parse_terminated_normally(text) is False:
        return OutputTerminationError(f"the run did not terminate normally — {what} in {src}")
    return OutputParseError(f"{what} in {src}")


def parse_qchem(path: str | Path) -> list[ParsedResult]:
    """Parse one Q-Chem ``.out`` into a single-element ``[ParsedResult]``."""
    return parse_qchem_text(Path(path).read_text(encoding="utf-8", errors="replace"), src=str(path))


def parse_qchem_text(text: str, *, src: str = "<text>") -> list[ParsedResult]:
    """Assemble one :class:`ParsedResult` from a single read of a Q-Chem ``.out``.

    Runs every section extractor over the shared ``text``, so each value is parsed
    exactly once. Raises :class:`OutputParseError` when the energy or the geometry is
    missing or malformed, upgraded to :class:`OutputTerminationError` by
    :func:`_unreadable` when the status verdict can say the run died. ``src`` only
    labels error messages.
    """
    final = energy.parse_final_energy_from_text(text)
    if final is None:
        raise _unreadable("no 'Total energy in the final basis set'", text, src)
    try:
        coords = geometry.parse_orientation_from_text(text)
    except ValueError as e:
        # A bare ValueError escaping here would pass straight through
        # `lifecycle._parse_job`, which contains only OutputParseError — so one
        # malformed row would end the whole run in a traceback rather than becoming
        # that structure's ledgered failure.
        raise OutputParseError(f"malformed coordinate row in {src}: {e}") from e
    if coords is None:
        raise _unreadable(f"no '{geometry.ORIENTATION_MARKER}' block", text, src)
    symbols, positions = coords
    if not symbols:
        raise _unreadable(f"'{geometry.ORIENTATION_MARKER}' block has no atoms", text, src)
    try:
        # The atom count comes from the block just read, which is what lets the gradient
        # reader tell a summary line from a lost atom row.
        force_vectors = forces.parse_forces_from_text(text, n_atoms=len(symbols))
    except ValueError as e:
        # Held to the same rule as the coordinates above, with the label that points at
        # the right section.
        raise OutputParseError(f"malformed gradient row in {src}: {e}") from e
    thermo = energy.parse_thermochemistry_from_text(text, electronic_hartree=final)
    imaginary, table, modes = frequencies.parse_frequency_block(text, n_atoms=len(symbols))
    return [
        ParsedResult(
            symbols=symbols,
            positions=positions,
            energy_hartree=final,
            forces_ev_per_a=force_vectors,
            converged=status.parse_converged(text),
            terminated_normally=status.parse_terminated_normally(text),
            gibbs_hartree=thermo.gibbs_hartree if thermo else None,
            enthalpy_hartree=thermo.enthalpy_hartree if thermo else None,
            energy_zpe_hartree=thermo.energy_zpe_hartree if thermo else None,
            imaginary_freqs=imaginary,
            frequencies=table,
            normal_modes=modes,
        )
    ]


def parse_output(path: str | Path, operation: str) -> list[ParsedResult]:
    """Pick the parser for the YAML ``operation`` string — one table.

    Every known operation routes to the single-pass assembler today (see the module
    docstring for why that is uniform rather than lazy). A name outside the vocabulary
    is a config mistake and is raised as such before any file is read — the preflight
    (:meth:`~chemrefine.engines.qchem.engine.QchemEngine.check_step`) refuses it at t=0,
    and this is the same refusal for a caller that skipped the preflight.
    """
    op = operation.lower().replace("+", "_")
    if op not in _QCHEM_OPERATIONS:
        raise OutputParseError(f"unknown Q-Chem operation: {operation!r}")
    return parse_qchem(path)
