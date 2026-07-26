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

from dataclasses import replace
from pathlib import Path

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
from chemrefine.errors import OutputParseError

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
    missing / malformed.
    """
    electronic = energy.parse_final_energy_from_text(text)
    if electronic is None:
        raise OutputParseError(f"no FINAL SINGLE POINT ENERGY in {src}")
    try:
        coords = geometry.parse_coordinates_from_text(text)
    except ValueError as e:
        # Corrupt token (e.g. a ``*****`` overflow) — a per-file parse failure, not a crash.
        raise OutputParseError(f"malformed coordinate row in {src}: {e}") from e
    if coords is None:
        raise OutputParseError(f"no CARTESIAN COORDINATES block in {src}")
    symbols, positions = coords
    if not symbols:
        raise OutputParseError(f"CARTESIAN COORDINATES block has no atoms in {src}")

    thermo = energy.parse_thermochemistry_from_text(text, electronic_hartree=electronic)
    imaginary, modes = _parse_frequency_block(text, n_atoms=len(symbols))
    return [
        ParsedResult(
            symbols=symbols,
            positions=positions,
            energy_hartree=electronic,
            forces_ev_per_a=forces.parse_forces_from_text(text),
            terminated=status.parse_terminated(text),
            converged=status.parse_converged(text),
            gibbs_hartree=thermo.gibbs_hartree if thermo else None,
            enthalpy_hartree=thermo.enthalpy_hartree if thermo else None,
            energy_zpe_hartree=thermo.energy_zpe_hartree if thermo else None,
            imaginary_freqs=imaginary,
            normal_modes=modes,
        )
    ]


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


def _stamp_run_status(frames: list[ParsedResult], out_path: Path) -> list[ParsedResult]:
    """Copy the ``.out``'s termination verdict onto every sidecar frame (B2).

    The ensemble sidecar carries geometries and energies but no run status, so frames
    parsed straight out of it default to ``terminated=None`` — and
    :func:`chemrefine.step_failures.succeeded` reads ``None`` as "not a failure signal".
    A GOAT job killed mid-run after writing a partial ensemble was therefore an
    unconditional success: no ledger entry, no ``on_failure``, no signal to the user.

    Only ``terminated`` is stamped. A whole-run ``NOT CONVERGED`` says nothing reliable
    about an individual pose — an ensemble is a *set* of stationary points, and one
    stubborn conformer must not condemn the rest — so ``converged`` stays ``None``
    (not reported per frame). :func:`ensembles.parse_pes_from_text` already does the
    same for scan points.
    """
    terminated = status.parse_terminated(out_path.read_text(encoding="utf-8", errors="replace"))
    return [replace(frame, terminated=terminated) for frame in frames]


def parse_output(path: str | Path, operation: str) -> list[ParsedResult]:
    """Pick the right parser based on the YAML ``operation`` string.

    ``opt_sp`` / ``sp`` / ``freq`` / ``pes`` read the ``.out`` directly; the multi-frame
    ensemble operations read their ``<base>.<suffix>`` sidecar (via :mod:`ensembles`) and
    then take their run status from the ``.out`` beside it (:func:`_stamp_run_status`).
    """
    op = operation.lower().replace("+", "_")
    if op in _DFT_OPERATIONS:
        return parse_dft(path)
    if op == "pes":
        return ensembles.parse_pes(path)
    out_path = Path(path)
    if op == "goat":
        frames = ensembles.parse_goat_ensemble(
            ensembles.ensemble_sidecar(out_path, ensembles.GOAT_SUFFIX)
        )
    elif op == "docker":
        frames = ensembles.parse_docker(
            ensembles.ensemble_sidecar(
                out_path, ensembles.DOCKER_SUFFIX, ensembles.DOCKER_SUFFIX_611
            )
        )
    elif op == "solvator":
        frames = ensembles.parse_solvator(
            ensembles.ensemble_sidecar(
                out_path, ensembles.SOLVATOR_SUFFIX, ensembles.SOLVATOR_SUFFIX_611
            )
        )
    else:
        raise OutputParseError(f"unknown ORCA operation: {operation!r}")
    return _stamp_run_status(frames, out_path)
