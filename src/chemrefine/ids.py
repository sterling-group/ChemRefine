"""Hierarchical structure-ID allocation and persistence.

ChemRefine tracks every conformer through the pipeline with a string ID
that records its lineage. The seed structures of step 1 get plain integer
IDs (``"0"``, ``"1"``, ...). Whenever a step expands one parent into
multiple children (e.g. a GOAT ensemble), the children inherit a
hyphen-suffixed ID: ``"0-1"`` means "child 1 of parent 0",
``"0-1-2"`` means "child 2 of that branch", and so on.

The functions here own three concerns:

1. Allocate IDs for new children (:func:`allocate_child_ids`).
2. Resolve IDs after a step that may either preserve a 1:1 mapping
   (``OPT+SP``) or fan out into a larger set (``GOAT``, ``PES``),
   inferring the fan-out from the parsed structure count
   (:func:`resolve_persistent_ids`).
3. Extract a structure ID from an engine input/output filename
   (:func:`extract_structure_id`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

_ID_PATTERN = re.compile(
    r"step(?P<step>\d+)_structure_(?P<id>[0-9\-]+)\.(?:inp|out|xyz)",
    re.IGNORECASE,
)
"""Matches the canonical ``step{N}_structure_{ID}.{ext}`` filename shape."""

_ID_ANYWHERE_RE = re.compile(
    r"step(?P<step>\d+)_structure_(?P<id>[0-9\-]+)(?:_|\.|$)",
    re.IGNORECASE,
)
"""Matches the ID even when the filename has a trailing suffix (``_trj``, ``_atom46`` ...)."""


def extract_structure_id(filename: str | PathLike) -> str | None:
    """Return the structure ID encoded in a ``step{N}_structure_{ID}.ext`` filename, or ``None``."""
    m = _ID_PATTERN.match(Path(filename).name)
    return m.group("id") if m else None


def extract_structure_id_any(filename: str | PathLike) -> str | None:
    """Return the structure ID from a filename that may carry trailing suffixes."""
    m = _ID_ANYWHERE_RE.search(Path(filename).stem)
    return m.group("id") if m else None


def validate_structure_ids(structure_ids: Sequence[object], step_id: int | str) -> list[str]:
    """Normalize an arbitrary sequence of IDs to a list of clean strings.

    Accepts ``int`` (must be non-negative) and ``str`` (non-empty, not
    ``"-1"``). Raises :class:`ValueError`/:class:`TypeError` on invalid
    input so callers can surface the problem to the user.
    """
    if structure_ids is None:
        raise ValueError(f"[step {step_id}] structure_ids is None")
    if isinstance(structure_ids, (str, bytes)) or not isinstance(structure_ids, Sequence):
        raise TypeError(f"[step {step_id}] structure_ids must be a sequence of IDs")
    if len(structure_ids) == 0:
        raise ValueError(f"[step {step_id}] structure_ids is empty")

    out: list[str] = []
    for idx, raw in enumerate(structure_ids):
        if isinstance(raw, int):
            if raw < 0:
                raise ValueError(f"[step {step_id}] structure_ids[{idx}] is negative: {raw}")
            out.append(str(raw))
        elif isinstance(raw, str):
            stripped = raw.strip()
            if not stripped or stripped == "-1":
                raise ValueError(f"[step {step_id}] structure_ids[{idx}] is invalid: {raw!r}")
            out.append(stripped)
        else:
            raise TypeError(
                f"[step {step_id}] structure_ids[{idx}] has unsupported type "
                f"{type(raw).__name__}; only int or str are allowed"
            )
    return out


def allocate_child_ids(parents: Sequence[str], fanouts: Sequence[int]) -> list[str]:
    """Allocate persistent IDs for one step's children given parent IDs and per-parent fanout.

    Rules:

    * ``fanout == 0`` — the parent contributes nothing.
    * ``fanout == 1`` — the child inherits the parent ID unchanged.
    * ``fanout >= 2`` — each child gets ``"{parent}-{i}"`` for ``i`` in ``0..fanout-1``.
    """
    if len(parents) != len(fanouts):
        raise ValueError(
            f"parents and fanouts must have the same length: got {len(parents)} vs {len(fanouts)}"
        )
    children: list[str] = []
    for parent, fanout in zip(parents, fanouts, strict=True):
        if fanout < 0:
            raise ValueError(f"fanout must be >= 0; got {fanout} for parent {parent!r}")
        if fanout == 0:
            continue
        if fanout == 1:
            children.append(str(parent))
        else:
            for i in range(fanout):
                children.append(f"{parent}-{i}")
    return children


def resolve_persistent_ids(
    *,
    step_number: int,
    parent_ids: Sequence[str] | None,
    child_count: int,
) -> list[str]:
    """Infer child IDs for a step from the parent IDs and how many children appeared.

    Cases:

    * **Step 1** — bootstrap. Children are ``"0", "1", ..., child_count - 1``.
    * **No parents** — same as step 1 (defensive fallback).
    * **1:1 preservation** — ``child_count == len(parent_ids)``. Children
      inherit parent IDs unchanged.
    * **Single parent fan-out** — ``len(parent_ids) == 1``. Children
      become ``"{parent}-0", ... "{parent}-{N-1}"``.
    * **Even fan-out** — ``child_count % len(parent_ids) == 0``. Each
      parent fans out evenly.
    * **Uneven fan-out** — falls back to giving the extras to parent 0
      and 1:1 to the rest.
    """
    if step_number <= 1 or not parent_ids:
        return [str(i) for i in range(child_count)]

    if child_count == len(parent_ids):
        return list(parent_ids)  # defensive copy on return

    p = len(parent_ids)
    if p == 1:
        fanouts: list[int] = [child_count]
    elif child_count % p == 0:
        fanouts = [child_count // p] * p
    elif child_count <= p:
        fanouts = [1] * child_count + [0] * (p - child_count)
    else:
        # Uneven overflow: give every non-primary parent exactly 1 child;
        # the leftover goes to parent 0 (the lowest-energy branch).
        primary_extra = child_count - (p - 1)
        fanouts = [primary_extra] + [1] * (p - 1)
    return allocate_child_ids(parent_ids, fanouts)


def parent_of(structure_id: str) -> str:
    """Return the parent ID (everything before the last ``"-"``), or the ID itself."""
    return structure_id.rsplit("-", 1)[0] if "-" in structure_id else structure_id
