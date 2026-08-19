"""Parse a script engine's output JSON into a :class:`ParsedResult`.

The user's ``step{N}.py`` writes a JSON document (``energy_hartree`` required; optional
``positions_angstrom`` for an optimised geometry and ``gradient_hartree_per_bohr`` for
forces); this reads it back into the shared ``ParsedResult`` the assembler turns into a
:class:`~chemrefine.state.Structure`. Sibling to :mod:`chemrefine.engines._script.render`
(the input writer).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
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
    _require_finite(data["energy_hartree"], what="energy_hartree", label=label, path=out_path)
    for what in ("positions_angstrom", "gradient_hartree_per_bohr"):
        for component in _components(data.get(what) or ()):
            _require_finite(component, what=what, label=label, path=out_path)
    return data


def _components(block: Any) -> Iterator[Any]:
    """Yield every scalar of a coordinate or gradient block, whatever shape it arrived in.

    Shape is :func:`_atoms_from_output`'s business and is judged *after* this, so the
    finiteness check must not presume one. A flat ``3N`` list is the natural mistake — a
    backend handing back ``coords.ravel()`` produces one — and iterating it as though it
    held rows raises a bare ``TypeError``, which is the very class of escape the finiteness
    guard exists to close. Yielding scalars either way means a malformed *and* diverged
    block still fails as the shape error it is, with the message that names it.
    """
    for item in block:
        if isinstance(item, (list, tuple)):
            yield from item
        else:
            yield item


def _require_finite(value: Any, *, what: str, label: str, path: Path) -> float:
    """Return ``value`` as a finite float, or raise :class:`OutputParseError`.

    A diverged calculation reports ``nan`` / ``inf`` rather than failing, and nothing
    downstream treats that as a failure: :func:`chemrefine.lifecycle.succeeded` only reads
    an explicit ``False`` flag, and :func:`chemrefine.filtering.apply` drops an energy that
    is ``None``, not one that is ``NaN``. So the structure ranks as a real result — and
    because every ``NaN`` comparison is false, it sorts by position rather than by energy
    and displaces a genuine survivor. Refused here, at the boundary, where it becomes an
    ordinary :attr:`~chemrefine.state.FailureKind.UNPARSEABLE` ledger entry and the step's
    ``on_failure`` policy decides what happens next.

    The gradient is held to the same rule: it is the other half of the same diverged
    calculation, and a non-finite force is what the MLIP trainer would go on to train on.

    So is the geometry, and it is the one with two ways to go wrong. A NaN coordinate
    reaches :func:`chemrefine.cache.structure_record`, whose positions are inline in the
    ``.result.json`` — so ``write_json``'s ``allow_nan=False`` raises a bare ``ValueError``
    that :func:`chemrefine.lifecycle._parse_job` does not catch, ending the whole run over
    one structure. And on the paths that never write a record, the coordinates go to the
    ``arrays.npz`` sidecar instead, which has no such check: there the NaN is simply stored
    and served to every later step. Refused here, both become this one ledger entry.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise OutputParseError(
            f"{label} output {path} has a non-numeric {what!r} ({value!r})"
        ) from e
    if not np.isfinite(number):
        raise OutputParseError(
            f"{label} output {path} reports a non-finite {what!r} ({value!r}); "
            f"the calculation diverged"
        )
    return number


def _atoms_from_output(data: dict[str, Any], *, fallback: Atoms | None) -> Atoms:
    """Return ASE ``Atoms`` from the output JSON, falling back to the seed geometry.

    If the script wrote a ``positions_angstrom`` block (an optimised geometry), the
    returned ``Atoms`` carries those positions on the seed's symbols; otherwise the seed
    atoms come back unchanged.

    A block of the wrong *shape* is refused here rather than left to ASE. ``set_positions``
    raises a bare :class:`ValueError`, which is outside this package's hierarchy: it would
    pass straight through :func:`chemrefine.lifecycle._parse_job` (which contains only
    :class:`~chemrefine.errors.OutputParseError`) and out of ``cli._dispatch``, ending the
    whole run in a traceback over one structure — and discarding the successes of the same
    step, which are about to be cached. The flat ``3N`` list is the natural mistake, since a
    backend that hands back ``coords.ravel()`` produces one.
    """
    positions = data.get("positions_angstrom")
    if positions is None or fallback is None:
        if fallback is None:
            raise OutputParseError(
                "output lacks positions_angstrom and no seed atoms are available"
            )
        return cast(Atoms, fallback.copy())
    updated: Atoms = fallback.copy()
    try:
        updated.set_positions(np.asarray(positions, dtype=float))
    except ValueError as e:
        raise OutputParseError(
            f"malformed 'positions_angstrom' for a {len(updated)}-atom structure: {e}"
        ) from e
    return updated


def _forces_from_gradient(gradient: list[list[float]] | None) -> NDArray[np.float64] | None:
    """Convert a template gradient (Hartree/Bohr) to ASE forces (eV/Å).

    Held to the same rule as the coordinates above: a ragged gradient makes ``np.asarray``
    raise a bare :class:`ValueError`, which would leave the exit-code contract the same way.
    """
    if not gradient:
        return None
    try:
        rows = np.asarray(gradient, dtype=float)
    except ValueError as e:
        raise OutputParseError(f"malformed 'gradient_hartree_per_bohr': {e}") from e
    return rows * (-HARTREE_PER_BOHR_TO_EV_PER_A)
