"""Attempt directories: sealing a structure's canonical state away, and installing a new one.

A structure lives at ``stepN/<id>/``, and resolving it may take more than one try. Each try is
an *attempt*, archived under ``stepN/<id>/attemptK/``. Two callers use this, and they compose
the same three operations differently:

.. code-block:: text

    convergence retry   archive(dir)                            → re-run at canonical
    NMS resolution      begin(dir) → children run inside → seal → promote the winner

The retry has nothing to run inside the attempt, so it seals immediately and installs a new
canonical state by re-running. NMS runs its displaced children inside the attempt first, then
seals round 1 alongside them and installs by promoting the winner. Keeping ``begin`` and
``seal`` separable is what lets both spell their own order instead of passing a destination
into a shared one.

The invariant either way: **the attempt holds what was at canonical, and canonical holds one
calculation.** Only ``attempt*/`` is ever left behind by a seal — folding one attempt into
another would lose a run's history.

Naming lives in :mod:`chemrefine.ids`; this module only moves and copies.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from chemrefine import ids


def begin(structure_dir: Path) -> Path:
    """Create and return the next free ``attemptK/`` under a structure's directory."""
    attempt = ids.next_attempt_dir(structure_dir)
    attempt.mkdir(parents=True, exist_ok=True)
    return attempt


def seal(structure_dir: Path, attempt_dir: Path) -> None:
    """Move a structure's canonical artifacts into ``attempt_dir``.

    Everything the attempt produced moves — loose files and any engine-written sub-directory
    such as ``pyscf-extopt``'s ``tensors/`` (an ``output_dirs`` entry). A ``tensors/`` left at
    the canonical path is the same disagreement this module exists to prevent: whatever runs
    there next writes its own files beside the stale ones, and the directory describes two
    calculations at once.

    ``attempt*/`` sub-directories stay where they are.
    """
    attempt_dir.mkdir(parents=True, exist_ok=True)
    for item in structure_dir.iterdir():
        if item.is_dir() and ids.is_attempt_dir(item):
            continue
        shutil.move(str(item), str(attempt_dir / item.name))


def archive(structure_dir: Path) -> Path:
    """Seal a structure's canonical state into a fresh ``attemptK/``; return it.

    :func:`begin` and :func:`seal` fused, for the caller that has nothing to run inside the
    attempt first.
    """
    attempt = begin(structure_dir)
    seal(structure_dir, attempt)
    return attempt


def archive_previous(step_dir: Path, structure_ids: Iterable[str]) -> list[Path]:
    """Archive any prior artifacts of ``structure_ids`` before they are re-executed.

    The invariant this protects: **a parsed output must have been produced by this run's
    submission of that job.** Success is decided by ``out.is_file()``, which cannot tell this
    run's output from a leftover — so without this, a re-executed step whose job dies before
    writing anything re-reads the *previous* run's result and reports it as current. The
    exposed paths are those that do not truncate their output in place: the script engines
    (whose JSON is written in ``$WORK_DIR`` and only copied back on success) and every ORCA
    ensemble operation (which reads a ``*.finalensemble.xyz``-style sidecar, not the ``.out``).

    Structures with no loose files are skipped, so a first run is a no-op.
    """
    archived: list[Path] = []
    for sid in structure_ids:
        struct_dir = step_dir / sid
        if struct_dir.is_dir() and any(p.is_file() for p in struct_dir.iterdir()):
            archived.append(archive(struct_dir))
    return archived


def promote(attempt_dir: Path, *, step: int, source_id: str, target_id: str) -> None:
    """Copy one attempt child's artifacts up to the parent's canonical basenames.

    Every loose file of ``attempt_dir/<source_id>/`` lands beside the parent with
    ``step{N}_{source_id}`` in its basename rewritten to ``step{N}_{target_id}`` — a prefix
    swap, because engine artifact names are not all single-extension (``_trj.xyz``,
    ``.property.json``). So the output, orbitals, Hessian, restart file and the input that
    produced them all arrive together, and the canonical location describes one calculation.

    The copies are verbatim: an engine's own log still refers to the directory it ran in, which
    is where that job really happened. :attr:`chemrefine.state.Structure.resolved_from` records
    which child it was.

    The source's own ``attempt*/`` directories stay behind — they are the runs *it* discarded,
    and ``attemptK`` resolves against the parent, so copying one up would merge a discarded run
    into the attempt this is reading from. Any other sub-directory is engine-written and named
    by the engine, so it is promoted under its own name.
    """
    stem = ids.structure_stem(step, source_id)
    for item in sorted((attempt_dir / source_id).iterdir()):
        if ids.is_attempt_dir(item):
            continue
        if item.is_dir():
            shutil.copytree(item, attempt_dir.parent / item.name, dirs_exist_ok=True)
            continue
        if not item.name.startswith(stem):
            continue
        renamed = ids.structure_stem(step, target_id) + item.name[len(stem) :]
        shutil.copy2(item, attempt_dir.parent / renamed)
