"""Parse a script engine's output JSON into a :class:`ParsedResult`.

The user's ``step{N}.py`` writes a JSON document; this reads it back into the shared
``ParsedResult`` the assembler turns into a :class:`~chemrefine.state.Structure`. Sibling to
:mod:`chemrefine.engines._script.render` (the input writer).

**What may be in that document is not decided here.** It is
:data:`chemrefine.engines._script.contract.SCRIPT_OUTPUT`, which the engine may extend
(:attr:`~chemrefine.engines._script.engine.ScriptEngine.output_fields`) — this module only
applies it: check the required names, hold the numeric ones to the finiteness rule, convert
each present value, and hand the result to ``ParsedResult``. Adding a quantity is a line in
the declaration, not an edit here.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from chemrefine.engines._script.contract import SCRIPT_OUTPUT, OutputField
from chemrefine.engines.api import ParsedResult
from chemrefine.errors import OutputParseError


def parse_output(
    output_path: Path,
    *,
    label: str,
    fallback: Atoms | None,
    fields: Sequence[OutputField] = SCRIPT_OUTPUT,
) -> list[ParsedResult]:
    """Read one template output JSON into a single-element ``[ParsedResult]``.

    ``label`` (the human backend name) tags error messages; ``fallback`` is the seed geometry,
    which supplies the symbols always and the positions when the script wrote none; ``fields``
    is the engine's output contract, defaulting to the shared one.

    The seed is required outright rather than only when ``positions_angstrom`` is absent: a
    parsed structure needs symbols, and the output document carries none. That has always been
    true — the message is the one this has always raised.
    """
    data = _load_output_json(output_path, label=label, fields=fields)
    if fallback is None:
        raise OutputParseError("output lacks positions_angstrom and no seed atoms are available")
    seed = fallback.copy()

    # `forces_ev_per_a` is the one ParsedResult field with no default of its own, so the
    # mapping supplies it; everything else the contract does not mention keeps ParsedResult's.
    values: dict[str, Any] = {"forces_ev_per_a": None}
    for spec in fields:
        raw = data.get(spec.name)
        if raw is None:
            continue
        values[spec.field] = spec.convert(raw, seed) if spec.convert else raw
    positions = values.pop("positions", None)
    if positions is None:
        positions = seed.get_positions()

    return [
        ParsedResult(
            symbols=tuple(seed.get_chemical_symbols()),
            positions=positions,
            **values,
        )
    ]


def _load_output_json(
    out_path: Path, *, label: str, fields: Sequence[OutputField] = SCRIPT_OUTPUT
) -> dict[str, Any]:
    """Read the user's script output JSON; raise :class:`OutputParseError` if malformed.

    Both checks are driven by ``fields`` rather than by a list of names spelled here — which
    is the point of the declaration. A quantity that is declared ``finite`` is swept whether
    or not anyone remembered to add it to a tuple in this module.
    """
    if not out_path.is_file():
        raise OutputParseError(f"{label} output not found: {out_path}")
    try:
        data: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise OutputParseError(f"{label} output {out_path} is not valid JSON: {e}") from e
    for spec in fields:
        if spec.required and spec.name not in data:
            raise OutputParseError(
                f"{label} output {out_path} missing required {spec.name!r} field"
            )
        if not spec.finite:
            continue
        for component in _components(data.get(spec.name)):
            _require_finite(component, what=spec.name, label=label, path=out_path)
    return data


def _components(block: Any) -> Iterator[Any]:
    """Yield every scalar of a value, whatever shape it arrived in.

    Shape is the field's own converter's business and is judged *after* this, so the
    finiteness check must not presume one. A flat ``3N`` list is the natural mistake — a
    backend handing back ``coords.ravel()`` produces one — and iterating it as though it held
    rows raises a bare ``TypeError``, which is the very class of escape the finiteness guard
    exists to close. Yielding scalars either way means a malformed *and* diverged block still
    fails as the shape error it is, with the message that names it.

    A scalar (the energy) yields itself, and an absent value yields nothing, so the caller
    needs no special case for either.
    """
    if block is None:
        return
    if not isinstance(block, (list, tuple)):
        yield block
        return
    for item in block:
        if isinstance(item, (list, tuple)):
            yield from item
        else:
            yield item


def _require_finite(value: Any, *, what: str, label: str, path: Path) -> float:
    """Return ``value`` as a finite float, or raise :class:`OutputParseError`.

    A diverged calculation reports ``nan`` / ``inf`` rather than failing, and nothing
    downstream treats that as a failure: :func:`chemrefine.lifecycle.succeeded` only reads an
    explicit ``False`` flag, and :func:`chemrefine.filtering.apply` drops an energy that is
    ``None``, not one that is ``NaN``. So the structure ranks as a real result — and because
    every ``NaN`` comparison is false, it sorts by position rather than by energy and displaces
    a genuine survivor. Refused here, at the boundary, where it becomes an ordinary
    :attr:`~chemrefine.state.FailureKind.UNPARSEABLE` ledger entry and the step's
    ``on_failure`` policy decides what happens next.

    The gradient is held to the same rule: it is the other half of the same diverged
    calculation, and a non-finite force is what the MLIP trainer would go on to train on.

    So is the geometry, and it is the one with two ways to go wrong. A NaN coordinate reaches
    :func:`chemrefine.cache.structure_record`, whose positions are inline in the
    ``.result.json`` — so ``write_json``'s ``allow_nan=False`` raises a bare ``ValueError``
    that :func:`chemrefine.lifecycle._parse_job` does not catch, ending the whole run over one
    structure. And on the paths that never write a record, the coordinates go to the
    ``arrays.npz`` sidecar instead, which has no such check: there the NaN is simply stored and
    served to every later step. Refused here, both become this one ledger entry.
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
