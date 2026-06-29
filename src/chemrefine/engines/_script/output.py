"""Parse a script engine's output JSON into a :class:`ParsedResult`.

The user's ``step{N}.py`` writes a JSON document (``energy_hartree`` required; optional
``positions_angstrom`` for an optimised geometry and ``gradient_hartree_per_bohr`` for
forces); this reads it back into the shared ``ParsedResult`` the assembler turns into a
:class:`~chemrefine.state.Structure`. Sibling to :mod:`chemrefine.engines._script.render`
(the input writer).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from chemrefine.engines.api import ParsedResult
from chemrefine.errors import OutputParseError
from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A


def parse_output(output_path: Path, *, label: str, fallback: Atoms | None) -> list[ParsedResult]:
    """Read one template output JSON into a single-element ``[ParsedResult]``.

    ``label`` (the human backend name) tags error messages; ``fallback`` is the seed
    geometry, used when the script didn't write its own ``positions_angstrom``.
    """
    data = _load_output_json(output_path, label=label)
    atoms = _atoms_from_output(data, fallback=fallback)
    forces = _forces_from_gradient(data.get("gradient_hartree_per_bohr"))
    return [
        ParsedResult(
            symbols=tuple(atoms.get_chemical_symbols()),
            positions=atoms.get_positions(),
            energy_hartree=float(data["energy_hartree"]),
            forces_ev_per_a=forces,
        )
    ]


def _load_output_json(out_path: Path, *, label: str) -> dict[str, Any]:
    """Read the user's script output JSON; raise :class:`OutputParseError` if malformed."""
    if not out_path.is_file():
        raise OutputParseError(f"{label} output not found: {out_path}")
    try:
        data: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise OutputParseError(f"{label} output {out_path} is not valid JSON: {e}") from e
    if "energy_hartree" not in data:
        raise OutputParseError(f"{label} output {out_path} missing required 'energy_hartree' field")
    return data


def _atoms_from_output(data: dict[str, Any], *, fallback: Atoms | None) -> Atoms:
    """Return ASE ``Atoms`` from the output JSON, falling back to the seed geometry.

    If the script wrote a ``positions_angstrom`` block (an optimised geometry), the
    returned ``Atoms`` carries those positions on the seed's symbols; otherwise the seed
    atoms come back unchanged.
    """
    positions = data.get("positions_angstrom")
    if positions is None or fallback is None:
        if fallback is None:
            raise OutputParseError(
                "output lacks positions_angstrom and no seed atoms are available"
            )
        return cast(Atoms, fallback.copy())
    updated: Atoms = fallback.copy()
    updated.set_positions(np.asarray(positions, dtype=float))
    return updated


def _forces_from_gradient(gradient: list[list[float]] | None) -> NDArray[np.float64] | None:
    """Convert a template gradient (Hartree/Bohr) to ASE forces (eV/Å)."""
    if not gradient:
        return None
    return np.asarray(gradient, dtype=float) * (-HARTREE_PER_BOHR_TO_EV_PER_A)
