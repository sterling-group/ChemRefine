"""Hierarchical structure-ID allocation and canonical filenames.

ChemRefine tracks every conformer through the pipeline with a string ID
that records its lineage. The seed structures of step 1 get plain integer
IDs (``"0"``, ``"1"``, ...). Whenever a step expands one parent into
multiple children (e.g. a GOAT ensemble), the children inherit a
hyphen-suffixed ID: ``"0-1"`` means "child 1 of parent 0",
``"0-1-2"`` means "child 2 of that branch", and so on. The lineage
itself is carried by :attr:`chemrefine.state.Structure.parent_id`; the
hyphenated form here is just the display convention used for filenames
and grep-friendliness.

The functions here own three concerns:

1. Allocate IDs for new children (:func:`allocate_child_ids`) —
   engines compute their per-parent fan-out and mint child IDs here.
2. Build the canonical per-structure artifact paths
   (:func:`structure_artifact_path`, :func:`input_geometry_path`) — each
   structure gets its own ``step_dir/{ID}/`` directory holding
   ``step{N}_{ID}.{ext}`` (and ``step{N}_{ID}_inp.xyz`` for the input
   geometry). (IDs travel in the step manifest, never re-parsed out of
   filenames — note that NMS child IDs like ``0_m5_pos`` contain letters,
   so a filename is not a reliable place to recover an ID from.)
3. Resolve a step's input *template* path (:func:`resolve_step_template`,
   :func:`default_template_name`) — the ``step{N}.{ext}`` template-naming
   convention, kept here beside the artifact-path convention so every
   canonical ChemRefine filename lives in one module.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

_ATTEMPT_DIR_RE = re.compile(r"attempt(\d+)$")


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


def structure_artifact_path(step_dir: Path, step: int, structure_id: str, ext: str) -> Path:
    """Canonical per-structure artifact path.

    Each structure gets its **own directory** under the step dir, so a
    calculation's files (input, output, optimized geometry, ORCA scratch
    copied back, pyscf tensors) sit together and never collide across
    structures: ``step_dir/{structure_id}/step{step}_{structure_id}.{ext}``.
    Keeping the naming convention here means a layout change touches one file.
    """
    return step_dir / structure_id / f"step{step}_{structure_id}.{ext}"


def input_geometry_path(step_dir: Path, step: int, structure_id: str) -> Path:
    """Path of a structure's **input** geometry, distinct from any output xyz.

    Named ``step{step}_{structure_id}_inp.xyz`` (an ``_inp`` stem) so it never
    shares a name with an engine's *output* geometry (e.g. ORCA writes the
    optimized geometry to ``step{step}_{structure_id}.xyz``); both survive the
    copy-back into the structure's directory.
    """
    return step_dir / structure_id / f"step{step}_{structure_id}_inp.xyz"


def next_attempt_dir(structure_dir: Path) -> Path:
    """Return the next free ``attemptK/`` sub-directory under a structure's dir.

    ``K`` is one past the highest existing ``attempt<n>`` (``attempt1`` if none), so
    a re-run — or a manually-added ``attempt2/`` — never collides with, or is blocked
    by, an existing one. This is the shared "attempt" primitive of the unified
    resolution model: the convergence retry moves a failed attempt here, and NMS
    archives a structure's exploration here. The directory is **not** created here
    (path only); a non-matching ``attempt*`` entry (e.g. a stray file) is ignored.
    """
    existing = [
        int(m.group(1))
        for d in structure_dir.glob("attempt*")
        if d.is_dir() and (m := _ATTEMPT_DIR_RE.fullmatch(d.name))
    ]
    return structure_dir / f"attempt{(max(existing) + 1) if existing else 1}"


def latest_attempt_dir(structure_dir: Path) -> Path | None:
    """Return the highest-numbered existing ``attemptK/`` under a structure's dir.

    The read-side counterpart of :func:`next_attempt_dir`: ``rebuild-cache`` reads a
    structure's most recent attempt (e.g. an NMS exploration) from here. ``None`` when
    no ``attempt<n>/`` exists.
    """
    attempts = [
        (int(m.group(1)), d)
        for d in structure_dir.glob("attempt*")
        if d.is_dir() and (m := _ATTEMPT_DIR_RE.fullmatch(d.name))
    ]
    return max(attempts)[1] if attempts else None


def default_template_name(step: int, suffix: str) -> str:
    """Return the default per-step template basename ``step{step}.{suffix}``.

    The single source of truth for the ``step{N}.<ext>`` template-naming
    convention; every engine resolves its template through here so a future
    layout change touches one place.
    """
    return f"step{step}.{suffix}"


def resolve_step_template(
    template_dir: Path,
    step: int,
    *,
    template: str | None,
    suffix: str,
    label: str,
) -> Path:
    """Resolve a step's input template under ``template_dir``.

    Uses the step's explicit ``template`` override when set, otherwise
    :func:`default_template_name`. Raises :class:`FileNotFoundError` naming
    ``label`` (the human backend name) when the file is missing.
    """
    name = template or default_template_name(step, suffix)
    path = template_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"{label} template not found: {path}")
    return path
