"""What a rendered ``step{N}.py`` may report back — declared once, read by four consumers.

A script engine's output contract used to be a list of names spelled at four independent
sites: the harvest loop inside the generated footer (:mod:`chemrefine.engines._script.render`),
the finiteness sweep and the :class:`~chemrefine.engines.api.ParsedResult` mapping
(:mod:`chemrefine.engines._script.output`), and the starter comment
:mod:`chemrefine.scaffold` writes. Four copies of one contract, agreeing only by discipline —
and the failure mode was silent in the direction that matters: a quantity added to the footer
but not to the finiteness sweep is written by the script, read onto the structure, and skips
the guard that stops a diverged calculation being cached as a result.

Declared here instead, so every consumer derives from it. This is the shape
:data:`chemrefine.engines.mlip.options.CALCULATOR_KNOBS` already gives the MLIP knob list, for
the reason it gives: spelled once, beside the thing that defines it, after a sweep found that
tuple hand-enumerated at ten sites.

The second thing this buys is what the old shape could not offer at all. Because the contract
is a *value*, an engine can extend it:
:attr:`chemrefine.engines._script.engine.ScriptEngine.output_fields` is a ClassVar a subclass
overrides — the mirror of ``_vars_from`` on the input side. A script engine that needs to
report something beyond the shared three declares it in its own module, and the footer, the
finiteness guard, the JSON mapping and the scaffold comment all follow. No building block is
edited, which is what ``docs/developer/adding-an-engine.md`` promises and what this makes true
for the script kind.

The conversions live here rather than in the reader because they *are* the contract: what
``gradient_hartree_per_bohr`` means is "Hartree/Bohr, one row per atom, and forces are its
negative in eV/Å", and that sentence belongs beside the name it defines.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from chemrefine.errors import OutputParseError
from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A


@dataclass(frozen=True)
class OutputField:
    """One quantity a ``step{N}.py`` may assign, and how it becomes a ``ParsedResult`` field."""

    name: str
    """The template-local name the footer harvests, and the JSON key it writes."""

    field: str
    """The :class:`~chemrefine.engines.api.ParsedResult` field it lands in."""

    required: bool = False
    """Whether a script that never assigns it has failed.

    A required name is emitted into the footer's result dict directly, so the generated script
    raises ``NameError`` at the point of omission rather than writing a document the driver
    then has to reject. Optional names go through the harvest loop, which skips whatever the
    template never defined."""

    finite: bool = True
    """Whether every scalar in the value is held to the finiteness rule.

    ``True`` for anything numeric, which is nearly everything: a diverged calculation reports
    ``nan``/``inf`` rather than failing, and nothing downstream reads that as a failure — see
    :func:`chemrefine.engines._script.output._require_finite` for the whole argument. ``False``
    is for a value the question does not apply to, such as a flag."""

    convert: Callable[[Any, Atoms], Any] | None = None
    """``(value, seed) -> ParsedResult value``; ``None`` passes the JSON value through.

    Takes the seed geometry because the two conversions that are not pass-throughs both need
    it — the geometry for its shape, the gradient for its atom count."""


def _as_float(value: Any, _seed: Atoms) -> float:
    """The energy: one number, already vetted finite by the sweep."""
    return float(value)


def positions_from(value: Any, seed: Atoms) -> NDArray[np.float64]:
    """The optimised geometry, on the seed's atom count.

    A block of the wrong *shape* is refused here rather than left to ASE. ``set_positions``
    raises a bare :class:`ValueError`, which is outside this package's hierarchy: it would
    pass straight through :func:`chemrefine.lifecycle._parse_job` (which contains only
    :class:`~chemrefine.errors.OutputParseError`) and out of ``cli._dispatch``, ending the
    whole run in a traceback over one structure — and discarding the successes of the same
    step, which are about to be cached. The flat ``3N`` list is the natural mistake, since a
    backend that hands back ``coords.ravel()`` produces one.

    ASE stays the shape oracle: a scratch copy of the seed is what judges the array, so the
    rule enforced here is the one ASE would have enforced anyway — only as an
    ``OutputParseError``, at the boundary, instead of as a bare ``ValueError`` three frames on.
    """
    scratch: Atoms = seed.copy()
    try:
        scratch.set_positions(np.asarray(value, dtype=float))
    except ValueError as e:
        raise OutputParseError(
            f"malformed 'positions_angstrom' for a {len(scratch)}-atom structure: {e}"
        ) from e
    # `asarray` is a no-copy pass-through on the float64 array ASE actually returns; it is
    # here to reach the typed surface rather than to convert anything.
    return np.asarray(scratch.get_positions(), dtype=np.float64)


def forces_from_gradient(value: Any, seed: Atoms) -> NDArray[np.float64] | None:
    """Convert a template gradient (Hartree/Bohr) to ASE forces (eV/Å).

    Held to the same rule as the coordinates above, and by the same two checks: a ragged
    gradient makes ``np.asarray`` raise a bare :class:`ValueError`, which would leave the
    exit-code contract the same way — and a well-formed array of the wrong *shape* would leave
    it silently. The positions path has ``set_positions`` as its shape oracle; a gradient has
    none, so the flat ``3N`` list the positions guard names as "the natural mistake"
    (``grad.ravel()``) parsed here as a perfectly valid ``(3N,)`` array, became
    :attr:`~chemrefine.state.Structure.forces_ev_per_a` — which declares no shape — and
    round-tripped the cache into any downstream ``mlip-train`` dataset. Nothing between this
    line and the trainer re-checks, so this is the one place the contract can be held.
    """
    if not value:
        return None
    n_atoms = len(seed)
    try:
        rows = np.asarray(value, dtype=float)
    except ValueError as e:
        raise OutputParseError(f"malformed 'gradient_hartree_per_bohr': {e}") from e
    if rows.shape != (n_atoms, 3):
        raise OutputParseError(
            f"malformed 'gradient_hartree_per_bohr' for a {n_atoms}-atom structure: "
            f"expected shape ({n_atoms}, 3), got {rows.shape} — a flat 3N list is the "
            f"usual cause (return rows, not gradient.ravel())"
        )
    return rows * (-HARTREE_PER_BOHR_TO_EV_PER_A)


SCRIPT_OUTPUT: tuple[OutputField, ...] = (
    OutputField("energy_hartree", "energy_hartree", required=True, convert=_as_float),
    OutputField("positions_angstrom", "positions", convert=positions_from),
    OutputField("gradient_hartree_per_bohr", "forces_ev_per_a", convert=forces_from_gradient),
)
"""The three quantities every script engine shares — the contract as it has always been."""
