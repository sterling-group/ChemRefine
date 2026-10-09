"""Filesystem I/O — the two write nets, atomic replacement, XYZ, SMILES → 3D, CSV reports.

Keeps every concrete file format in one place so individual engine
modules don't grow their own XYZ-handling code. :func:`write_single_xyz` accepts
either an :class:`ase.Atoms` or ``(symbol, x, y, z)`` rows, which is what
:func:`embed_smiles` returns.

An engine may keep an XYZ reader of its own only for a stated price, stated where the
reader lives; the rule for a new engine is "use :func:`read_xyz_frames` /
:func:`write_single_xyz` unless you can name your price too".

A write the filesystem refuses is one failure with one sentence, ``cannot write <file>:
<reason>``, and one exit code, and it is worded here. Two nets say whose file it was. Every
write the driver makes under a run's ``output_dir`` goes through :func:`tree_write` —
:func:`write_tree_file` and :func:`remove_tree_file` for a whole file — so a refusal is
:class:`~chemrefine.errors.OutputTreeError`, whichever file it was. A file a tool writes for
a person outside that tree — the config, the template or the seed at the path they named —
goes through :func:`user_write` — :func:`write_user_file` and :func:`remove_user_file` for
a whole file — and refuses as :class:`~chemrefine.errors.WriteRefusedError`, the class the
tree's specialises. A file a job writes for itself is outside both: its refusal comes back
as the job's failure.
``test_the_run_tree_is_written_only_through_io`` holds the rest of the package to this:
outside this module a call that writes a path stands inside one of the two nets, or its
function is listed there with what it writes instead.

What a run reads back from its own tree is asked here too. :func:`absent` is the one
question whether something is there, and :func:`tree_entries` the one listing of a
directory: each answers the same on every supported Python, and neither reads what the
filesystem refuses as nothing being there.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from chemrefine.errors import CacheError, ConfigError, OutputTreeError, WriteRefusedError
from chemrefine.quantities import HARTREE_TO_KCALMOL, boltzmann_weights

if TYPE_CHECKING:
    # Annotation-only: the ensemble writer reads `Structure` fields via getattr, so the
    # runtime import graph stays as it is — this module keeps importing no pydantic, and
    # the config, which imports this module, is not imported back.
    from chemrefine.config import EnergyType
    from chemrefine.state import Structure

_CSV_PRECISION = 8
_NATURAL_PART = re.compile(r"(\d+)")

STEPS_CSV_COLUMNS = (
    "Step",
    "Conformer",
    "Energy (Hartree)",
    "Energy (kcal/mol)",
    "dE (kcal/mol)",
    "Boltzmann Weight",
    "% Total",
    "% Cumulative",
    "Energy type",
)
"""The columns of ``steps.csv``, in order — the report's schema, addressable.

Public and load-bearing rather than a line of prose in a docstring, because three things
outside this function depend on these exact strings: :func:`chemrefine.agent_tools.
get_results` hands whole rows to agents, the GUI's results table names them in
``index.html``, and whatever a user reads the file with names them too.

:func:`save_step_csv` selects by this tuple before writing, so the frame it assembled and
the schema it promises cannot drift apart; ``tests/test_gui_assets.py`` compares the
page's hardcoded names against it, so the page cannot drift from either. A column is
renamed here, where every reader is looking."""

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def natural_key(name: str | Path) -> list[object]:
    """Return a list suitable for ``sorted(..., key=natural_key)`` natural ordering.

    ``"step10.out"`` sorts after ``"step2.out"`` instead of before it.

    ``isdecimal``, not ``isdigit``, for the reason ``StepConfig.matches`` uses it: the
    two disagree on the Unicode ``No`` category — ``"²".isdigit()`` is ``True`` and
    ``int("²")`` raises — and this predicate exists only to guard that ``int``. A seed
    directory holding a file like ``v2²3.xyz`` therefore ended the run in a bare
    ``ValueError`` from the sort, outside the exit-code contract. Ordering is
    unchanged: every chunk the ``\\d+`` capture isolates is decimal already.
    """
    return [int(p) if p.isdecimal() else p.lower() for p in _NATURAL_PART.split(str(name))]


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


_PROC_STATUS = Path("/proc/self/status")
"""Where Linux publishes this process's umask — see :func:`_umask`."""


def _umask() -> int:
    """This process's umask, read without writing it.

    ``os.umask`` is the only POSIX way to *read* the umask and it reads by setting it,
    which is process-global: between the clear and the restore, anything another thread
    creates is made with no mask at all, so a ``mkdir`` lands 0777. The GUI is served by
    waitress on four threads whose handlers both write through :func:`atomic_write` and
    call ``Path.mkdir``, so that window is reachable.

    ``umask(2)`` names the remedy itself: since Linux 4.7 the value is published in
    ``/proc/self/status``, and reading it changes nothing (CPython declined a thread-safe
    wrapper, bpo-35275, since the POSIX API has none either). The probe stays as the
    fallback for a kernel that does not publish the field; there the window is back, which
    is the narrower of the two evils.
    """
    try:
        for line in _PROC_STATUS.read_text(encoding="utf-8").splitlines():
            if line.startswith("Umask:"):
                return int(line.split()[1], 8)
    except (OSError, ValueError, IndexError):
        pass
    mask = os.umask(0)
    os.umask(mask)
    return mask


def atomic_write(path: Path, data: bytes, *, mode: int | None = None) -> None:
    """Replace ``path`` with ``data`` whole — a temp file, its mode, fsync, then one rename.

    The one writer for every file this package replaces: the caches, the configs and
    templates it saves, the SLURM lease ledger, the ExtOpt sidecars and wrapper, the record
    beside a trained model. A process that opens or executes ``path`` at any moment gets the
    old file or the new one, never a truncated one, and the ``fsync`` orders the rename after
    the data, so a machine crash (not only a killed process) cannot leave it empty either.

    ``mode`` is the file's permission bits, set on the temp file before the rename so the
    file never appears under ``path`` with other ones: ``None`` gives what a plain
    ``open()`` would — ``0o666`` under the umask, readable by the colleague a tree is
    shared with — ``0o600`` keeps a secret owner-only, ``0o755`` makes a script executable.
    The mode is set through the open file, inside the ``with``: it is the one step here
    that shared filesystems and FUSE/CIFS mounts refuse, and there a failure still closes
    the descriptor and removes the temp file rather than stranding either.

    Raw by design: a refusal leaves as the :class:`OSError` the filesystem raised, for the
    net its caller stands in to name — :func:`write_tree_file` for a run's own tree,
    :func:`write_user_file` for a person's file. It names ``path``, never the temp file the
    data was staged in, which no caller ever saw. The temp file is removed under
    :func:`cleanup`, so failing to remove it never replaces the error that left it behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                os.fchmod(fh.fileno(), 0o666 & ~_umask() if mode is None else mode)
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            Path(tmp).replace(path)
        finally:
            with cleanup():
                Path(tmp).unlink(missing_ok=True)
    except OSError as e:
        if e.filename is None:
            raise
        # The only file the OS can name in here is the staged one: the name `mkstemp` could
        # not create, the temp file the rename could not move or the cleanup could not remove.
        raise OSError(e.errno, e.strerror, os.fspath(path)) from e


def read_utf8(path: Path, *, what: str) -> str:
    """Read a user-authored text file as UTF-8, or refuse it as a :class:`ConfigError`.

    The one reader for the files a person writes and ChemRefine parses — the config, the
    step templates, the SLURM header — and the one refusal for each that cannot be used:
    missing, unreadable (a directory, another account's mode, an I/O error), or not UTF-8
    text. A :class:`ConfigError` carries the exit code the CLI maps and is a row in
    ``chemrefine validate`` and a 400 in the GUI, where a bare ``OSError`` or
    ``UnicodeDecodeError`` (a ``ValueError``) would be a traceback. ``what`` names the kind
    of file so the refusal reads as the mistake it is, and no caller words it again.
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"{what} not found: {path}") from None
    except UnicodeDecodeError as e:
        raise ConfigError(f"{what} {path} is not UTF-8 text ({e})") from e
    except OSError as e:
        raise ConfigError(f"cannot read {what} {path}: {e.strerror or e}") from e


def expand_home(raw: str) -> str:
    """``raw`` as a shell reads a path's start: a leading ``~`` is a home, all else as typed.

    The one reading of a typed path's ``~``, decided on the text itself — ``~/runs`` is the
    user's home, ``./~/runs`` a directory named ``~`` — and taken by every entry point: the
    config model for each key that names a path (:data:`chemrefine.config.USER_PATH`), the
    agent tools and the GUI through :func:`expand_user_path`. A ``~name`` the host knows no
    home for, or cannot even look up (an embedded NUL), is the :class:`ConfigError` the CLI
    maps, ``chemrefine validate`` reports, the GUI answers as a 400 and an MCP client
    receives as a typed refusal.
    """
    if not raw.startswith("~"):
        return raw
    try:
        return str(Path(raw).expanduser())
    except (RuntimeError, ValueError) as e:
        raise ConfigError(f"cannot resolve {raw!r}: {e}") from e


def expand_user_path(raw: str) -> Path:
    """The absolute path a typed path argument names: :func:`expand_home`, then the cwd.

    Absolute, so the text a tool hands back means the same place to whatever reads it
    next: a relative ``./~/seeds`` kept relative prints as ``~/seeds``, which every reader
    of that text takes for the user's home.
    """
    return Path(expand_home(raw)).absolute()


# ---------------------------------------------------------------------------
# Refused writes — the run's output tree, and a file written for a person — and what
# the run asks of its tree before it reads
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _net(path: Path, refusal: type[WriteRefusedError]) -> Generator[None]:
    """What the filesystem refuses inside the block leaves as ``refusal``, naming ``path``.

    The one body under :func:`tree_write` and :func:`user_write`, so the sentence — ``cannot
    write {path}: {reason}`` — and what counts as a refusal are decided once for both.
    """
    try:
        yield
    except _refusal_types() as e:
        reason = _refused(path, e)
        if reason is None:
            raise
        raise refusal(f"cannot write {path}: {reason}") from e


def tree_write(path: Path) -> contextlib.AbstractContextManager[None]:
    """Name a write the run's output tree refuses: :class:`~chemrefine.errors.OutputTreeError`.

    The one net for every mutation the driver makes under ``output_dir`` — writing,
    appending, creating a directory, linking, moving, copying, removing. A full disk or a
    quota, a read-only mount or another account's directory refuses whichever of them comes
    next, so the refusal reads the same at each: one class, with the exit code the CLI maps
    and the typed refusal every other surface shows, where a bare ``OSError`` would leave as
    a traceback with exit 1, the code a halted step exits with. Removing is in the net
    because it is a write: a read-only mount refuses even the removal of a file that is not
    there.

    The message is ``cannot write {path}: {reason}``, with the file the filesystem named
    before the reason when that is another — a directory a ``mkdir`` could not create, the
    source a copy could not read. The ``OSError`` stays on ``__cause__``. An ``OSError`` a
    caller expects as normal flow — a lost race, a file already there — is caught inside
    the block, before it reaches the net.

    A library that keeps its data in a sqlite file — ``ase.db`` — reports a refused write
    through sqlite's own exception, never an ``OSError``. The net names that one too when its
    code says the file could not be opened or written — a full disk or quota, a read-only
    mount or directory, a write cut short — with sqlite's sentence as the reason and the
    sqlite error on ``__cause__``. Any other sqlite error, a SQL mistake or a busy lock,
    leaves as itself.
    """
    return _net(path, OutputTreeError)


def user_write(path: Path) -> contextlib.AbstractContextManager[None]:
    """Name a refused write of a person's file: :class:`~chemrefine.errors.WriteRefusedError`.

    The net for what a tool writes on a person's behalf outside a run's output tree: the
    config an agent or the builder saves, a step template, the starters ``scaffold`` fills
    in, a seed set, each at the path the person named — and the scratch copy of a file
    they dropped on the builder. The sentence and the exit code are :func:`tree_write`'s;
    the class is the one the tree's refusal specialises, because the path is no part of a
    run's output tree and a refusal named for one would send its reader to ``resume``. What
    they do instead is free space, fix the permissions or name another place, and ask again.

    Never :class:`~chemrefine.errors.ConfigError`: the refusal says nothing about the text,
    and a class that says "fix the config" about a full disk has an agent rewrite a YAML
    that was right.
    """
    return _net(path, WriteRefusedError)


def _refusal_types() -> tuple[type[Exception], ...]:
    """What the nets catch: :class:`OSError`, and sqlite's error once loaded.

    ``ase.db`` keeps a dataset in a sqlite file, and sqlite refuses a write through
    :class:`sqlite3.OperationalError`, which is no ``OSError``. The class is read off the
    loaded module rather than imported: until a writer has imported :mod:`sqlite3` no sqlite
    error can exist, and importing it here would load sqlite into every process in the
    package — the ExtOpt wrapper's, once per optimizer step — for the few that write a
    database. An ``except`` clause's expression is evaluated only when an exception reaches
    it, so this is asked then and not before.
    """
    sqlite_error = getattr(sys.modules.get("sqlite3"), "OperationalError", None)
    return (OSError,) if sqlite_error is None else (OSError, sqlite_error)


_SQLITE_REFUSALS = frozenset({3, 8, 10, 13, 14})
"""The sqlite result codes that say the filesystem refused a write, by primary code.

``SQLITE_PERM`` (3) and ``SQLITE_READONLY`` (8) for a database sqlite may not create or
change, ``SQLITE_IOERR`` (10) for a write the filesystem cut short, ``SQLITE_FULL`` (13) for
a full disk or quota, ``SQLITE_CANTOPEN`` (14) for a directory this account cannot write or
a read-only mount. Numbers, since :mod:`sqlite3` and its constants are not imported here
(:func:`_refusal_types`); the primary code, since an error carries its extended one, whose
low byte is its family's — ``SQLITE_READONLY_DIRECTORY`` is 1544, 8 in its low byte — so an
extended code this Python's :mod:`sqlite3` has no name for counts all the same. The
exception class these arrive in also carries a SQL mistake and a busy lock, which refuse no
write.
"""


def _refused(path: Path, error: Exception) -> str | None:
    """Why the filesystem refused a write of ``path``, after the file it named if another.

    An error with no ``strerror``, a bare message, is given as its own text. The aggregate
    :func:`shutil.copytree` raises once it has tried every file is given as the first
    refusal it gathered, which names its file, and how many more there were. A sqlite error
    whose code is of a family :data:`_SQLITE_REFUSALS` holds is given as sqlite's own
    sentence; any other is ``None``, no refusal of the filesystem's at all.
    """
    rows = error.args[0] if isinstance(error, shutil.Error) and error.args else None
    if isinstance(rows, list) and rows:
        first = str(rows[0][2])
        return first if len(rows) == 1 else f"{first} (and {len(rows) - 1} more)"
    if not isinstance(error, OSError):
        code = getattr(error, "sqlite_errorcode", None)
        refused = isinstance(code, int) and (code & 0xFF) in _SQLITE_REFUSALS
        return str(error) if refused else None
    reason = error.strerror or str(error)
    named = error.filename
    if not isinstance(named, (str, bytes, os.PathLike)) or Path(os.fsdecode(named)) == path:
        return reason
    return f"{os.fsdecode(named)}: {reason}"


def write_tree_file(
    path: Path, data: str | bytes, *, atomic: bool = False, mode: int | None = None
) -> None:
    """Write ``data`` (text as UTF-8) to ``path`` in a run's output tree, parents first.

    ``atomic`` replaces the file whole (:func:`atomic_write`): for a file another process
    may read while it is rewritten, or a later run trusts — the cache's documents, its
    ledgers, the records beside each output. Plain, the default, is for what a run writes
    afresh whenever it reaches it — the per-structure inputs, the step's reports — where an
    ``fsync`` per file is a metadata round trip per structure on a shared filesystem, and a
    file a refusal cuts short is written afresh by the next run that reaches it.

    ``mode`` is the permission bits the file has from its first appearance — a script a job
    executes the moment it is there. Only a file replaced whole has them from its first
    byte, so a file given a ``mode`` is written as ``atomic`` writes it.

    Either way a refusal is :func:`tree_write`'s.
    """
    payload = data.encode("utf-8") if isinstance(data, str) else data
    with tree_write(path):
        if atomic or mode is not None:
            atomic_write(path, payload, mode=mode)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)


def remove_tree_file(path: Path) -> None:
    """Remove ``path`` from a run's output tree if it is there, refusing as :func:`tree_write`."""
    with tree_write(path):
        path.unlink(missing_ok=True)


def write_user_file(path: Path, data: str | bytes) -> None:
    """Replace a person's file at ``path`` with ``data`` (text as UTF-8), parents first.

    Whole or not at all (:func:`atomic_write`), always: it is a file a person edits or
    points a run at, and a write a kill or a full disk cuts short would leave a stub where
    their config or template had been. A refusal is :func:`user_write`'s, and leaves what
    was at ``path`` as it was.
    """
    payload = data.encode("utf-8") if isinstance(data, str) else data
    with user_write(path):
        atomic_write(path, payload)


def remove_user_file(path: Path) -> None:
    """Remove a person's file at ``path`` if it is there, refusing as :func:`user_write`."""
    with user_write(path):
        path.unlink(missing_ok=True)


@contextlib.contextmanager
def cleanup() -> Generator[None]:
    """Run cleanup that must never replace the exception it runs during.

    Cleanup — a ``finally``, or an ``except`` that ends in a ``raise``, removing a temp file
    or an empty log, or releasing what a run held, its lock or its job leases — runs while
    an exception may be on its way out: a refused write, a ``KeyboardInterrupt``, the
    ``SystemExit(143)`` a SIGTERM is turned into. An exception raised there takes that one's
    place, and the run exits with the cleanup's code and says the cleanup's sentence. So
    while an exception is in flight, a refusal in here is logged and dropped, and the one in
    flight goes on; with none in flight, it is raised like any other.

    In flight means :func:`sys.exception`: the exception an ``except`` or ``finally`` block
    is running for, in this frame or in any caller's. Cleanup reached from inside a
    caller's handler therefore counts the handled exception as in flight, which is why only
    cleanup belongs in here, never a write the run needs. A refusal is an :class:`OSError`,
    or what this package turns one into — the
    :class:`~chemrefine.errors.WriteRefusedError` of a refused write, the
    :class:`~chemrefine.errors.CacheError` of a record the cleanup reads back before it
    writes; anything else — a bug, a second ``KeyboardInterrupt`` — surfaces.
    """
    in_flight = sys.exception()
    try:
        yield
    except (OSError, WriteRefusedError, CacheError) as e:
        if in_flight is None:
            raise
        logger.warning("could not clean up during %s: %s", type(in_flight).__name__, e)


def absent(path: Path) -> bool:
    """Whether the filesystem says nothing is at ``path``.

    The run's question whether a job left a file at all, or a directory was ever made.
    Only "no such file or directory" proves that. Under a directory this account cannot
    search, ``Path.exists`` and ``Path.is_file`` raise ``PermissionError`` on Python 3.11
    to 3.13 and answer ``False`` on 3.14, so asked of them one tree is a traceback on one
    interpreter and an output that was never written on the next. Here any other refusal
    — another account's mode, a path through a regular file, an I/O error — leaves
    something that may be there, and the read that follows names it.
    """
    try:
        path.stat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def tree_entries(directory: Path) -> list[tuple[Path, bool]]:
    """What a directory of a run's output tree holds: each entry, and whether it is a directory.

    The one listing of the run's own tree, sorted by path; ``[]`` when ``directory`` is
    not there. A directory the filesystem will not list — another account's mode, a
    regular file in its place — is :class:`~chemrefine.errors.CacheError`, where
    ``Path.glob`` answers an empty list and a run would take a structure that has
    attempts for one that has none. Whether an entry is a directory is read with the
    listing, so a directory that can be listed and not searched answers too.
    """
    try:
        with os.scandir(directory) as scan:
            return sorted((Path(entry.path), entry.is_dir()) for entry in scan)
    except FileNotFoundError:
        return []
    except OSError as e:
        raise CacheError(f"cannot read {directory}: {e.strerror or e}") from e


# ---------------------------------------------------------------------------
# XYZ
# ---------------------------------------------------------------------------


CoordList = Sequence[tuple[str, float, float, float]]


def _xyz_frame_lines(atoms: Atoms, comment: str) -> list[str]:
    """One plain-XYZ frame: the count line, the comment line, one row per atom.

    The single spelling of the frame format, shared by the one-geometry writer and the
    multi-frame ensemble writer — two writers with their own row formatting are
    byte-identical only until one of them is edited.
    """
    lines = [str(len(atoms)), comment]
    for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions(), strict=True):
        lines.append(f"{symbol:2s} {x:.6f} {y:.6f} {z:.6f}")
    return lines


def extended_xyz_text(atoms: Atoms, *, displacements: NDArray[np.float64] | None = None) -> str:
    """One structure as extended-XYZ text, optionally carrying a per-atom displacement.

    Extended XYZ rather than the plain frame above, because it is the one format that
    carries all three things a viewer needs and does it in columns anything can read: the
    geometry, the cell as ``Lattice="…"`` when the structure has one, and — with
    ``displacements`` — three more columns per atom. ASE writes it; the GUI's viewer reads
    the same file for a molecule, for a periodic cell, and for an animated normal mode,
    which is why this is a text format and not a JSON payload of our own design.

    ``displacements`` must be one 3-vector per atom, in the same order — a normal mode's
    column out of the ``(n_atoms, 3, n_modes)`` tensor, typically.
    """
    import io as _io

    from ase.io import write

    payload = atoms
    columns = ["symbols", "positions"]
    if displacements is not None:
        vectors = np.asarray(displacements, dtype=float)
        if vectors.shape != (len(atoms), 3):
            raise ValueError(
                f"displacements must be one 3-vector per atom: got {vectors.shape}, "
                f"expected {(len(atoms), 3)}"
            )
        # A copy, so attaching the array for a render cannot mutate a caller's structure —
        # these come out of the step cache, which other readers share.
        payload = atoms.copy()
        payload.new_array("displacement", vectors)
        columns = [*columns, "displacement"]
    buffer = _io.StringIO()
    write(buffer, payload, format="extxyz", columns=columns)
    return buffer.getvalue()


def plain_xyz_text(geometry: Atoms | CoordList, *, comment: str = "") -> str:
    """One geometry as plain-XYZ text: the count line, ``comment``, one row per atom.

    ``geometry`` may be an :class:`ase.Atoms` or a list of ``(symbol, x, y, z)`` tuples
    (coerced on the fly). Text rather than a file, so the two writers of a single geometry
    share its bytes and differ only in whose file it is: :func:`write_single_xyz` in a
    run's output tree, :func:`chemrefine.agent_tools.build_structures` in the seed
    directory a person named.
    """
    if isinstance(geometry, Atoms):
        atoms = geometry
    else:
        atoms = Atoms(
            symbols=[row[0] for row in geometry],
            positions=np.array([row[1:] for row in geometry], dtype=float),
        )
    return "\n".join(_xyz_frame_lines(atoms, comment)) + "\n"


def write_single_xyz(geometry: Atoms | CoordList, path: str | Path, *, comment: str = "") -> Path:
    """Write one geometry to ``path`` in a run's output tree as plain XYZ; return ``path``.

    :func:`plain_xyz_text`, written through :func:`write_tree_file`, so a refusal is
    :class:`~chemrefine.errors.OutputTreeError`. The parent directory is created if
    needed — this is how a structure's per-id directory comes into being (engines call this
    with :func:`chemrefine.ids.input_geometry_path`).
    """
    target = Path(path)
    write_tree_file(target, plain_xyz_text(geometry, comment=comment))
    return target


def write_ensemble_xyz(
    structures: Sequence[Structure],
    path: str | Path,
    *,
    step: int,
    energy_attr: str,
    energy_label: str,
) -> Path:
    """Write ``structures`` as one multi-frame XYZ at ``path``; return ``path``.

    The human-facing ensemble a step leaves behind — every frame is a final geometry, so a
    user opens one file in Avogadro/VMD instead of walking per-structure directories.
    Written whole, never appended: completion order is nondeterministic, and a file built
    as jobs land would differ between two runs of the same step, where this one is
    byte-identical on a re-run, a resume, and a rebuild.

    Frames are sorted ascending by ``energy_attr`` — the step's own ranking energy
    (:func:`chemrefine.filtering.ranking_energy`), so the file leads with the conformer
    the step's filter would keep first. The sort is stable and sends energy-less
    structures (``on_failure: best`` backfills, missing thermochemistry) to the end in
    their incoming results order, which is the cache's stable manifest order.

    Each comment line carries ``step{N} id={id} {label}={value:.8f} Eh`` (``n/a`` when the
    energy is absent): the id is what ties a frame back to its structure directory and its
    ``steps.csv`` row. ase's extxyz reader parses these frames, so
    :func:`read_xyz_frames` round-trips the file.

    An **empty** ``structures`` removes any stale file rather than writing a zero-frame
    one: a step whose filter kept nothing must not leave the previous run's survivors
    lying about — the same reasoning as :func:`save_step_csv`'s step-1 truncation.
    """
    target = Path(path)
    if not structures:
        remove_tree_file(target)
        return target

    def _key(s: Structure) -> tuple[bool, float]:
        value = getattr(s, energy_attr)
        return (value is None, 0.0 if value is None else float(value))

    lines: list[str] = []
    for s in sorted(structures, key=_key):
        value = getattr(s, energy_attr)
        caption = f"{energy_label}=n/a" if value is None else f"{energy_label}={value:.8f} Eh"
        lines.extend(_xyz_frame_lines(s.atoms, f"step{step} id={s.id} {caption}"))
    write_tree_file(target, "\n".join(lines) + "\n")
    return target


def gather_output_files(directory: str | Path, pattern: str) -> list[Path]:
    """Return files in ``directory`` matching ``pattern``, sorted naturally."""
    d = Path(directory)
    if not d.exists():
        return []
    matches = sorted(d.glob(pattern), key=natural_key)
    return list(matches)


def read_xyz_frames(path: str | Path) -> list[Atoms]:
    """Read every frame of an XYZ file into ASE ``Atoms`` — the one XYZ reader.

    ``format="extxyz"`` selects ASE's robust reader: the naive ``"xyz"`` parser
    loops ``int(lines.pop(0))`` over every line and dies (``invalid literal for
    int(): '\\n'``) on the trailing blank lines that editors and programs routinely
    leave on real files. ``index=":"`` returns all frames; a single-frame file
    yields a one-element list, so callers never special-case frame count.

    ``ase.io`` is imported here, on the first read, rather than with the module: it costs
    about half a second, and importing this module is how every process in the package
    starts — the CLI answering ``--help``, and the ExtOpt wrapper spawned once per
    optimizer step, which never reads a frame through here.

    A file that is not XYZ raises what ASE's reader raises about its text — a
    :class:`ValueError`, an :class:`IndexError`, a :class:`KeyError`, or its own error,
    which is an :class:`OSError` — and those are what the readers of a person's file hold.
    A file that ends on an atom count, with no frame after it, is one of them, a
    :class:`ValueError` raised here: ASE takes a frame's comment line with a bare ``next``
    inside a generator, so the end of the file there reaches a caller as the
    :class:`RuntimeError` Python makes of a ``StopIteration`` in a generator.
    """
    from ase.io import read as ase_read

    try:
        return cast("list[Atoms]", ase_read(str(path), index=":", format="extxyz"))
    except RuntimeError as e:
        if not isinstance(e.__cause__, StopIteration):
            raise
        raise ValueError("the file ends on an atom count, with no frame after it") from e


# ---------------------------------------------------------------------------
# SMILES → 3D
# ---------------------------------------------------------------------------


def _conformer_rows(mol: Any) -> CoordList:
    """An embedded RDKit ``mol``'s conformer as ``(symbol, x, y, z)`` rows.

    Rows rather than rendered text, so the writing goes through
    :func:`write_single_xyz` like every other geometry this module emits. Its
    predecessor rendered the lines itself — a second XYZ writer one function away from
    the canonical one, byte-identical only for as long as nobody edited either.
    """
    conf = mol.GetConformer()
    rows: list[tuple[str, float, float, float]] = []
    for atom_idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        rows.append((mol.GetAtomWithIdx(atom_idx).GetSymbol(), pos.x, pos.y, pos.z))
    return rows


def embed_smiles(smiles: str, *, max_attempts: int = 10, random_seed: int = 42) -> CoordList:
    """One SMILES → embedded, UFF-relaxed 3D coordinate rows; raises on failure.

    The single-molecule seam under :func:`smiles_to_xyz`'s CSV loop, public because the
    two callers want opposite failure behaviour: a CSV sweep logs a bad row and keeps
    going, while a caller embedding one *explicit* SMILES (the agent's structure-building
    tool) wants the :class:`ValueError` raised at the molecule it names.

    ``random_seed`` seeds RDKit's conformer embedding (which is otherwise
    non-deterministic), so repeated calls regenerate identical 3D geometries.
    """
    from rdkit import Chem

    # Imported from the modules that define them rather than from ``rdkit.Chem.AllChem``,
    # which collects them with ``import *`` — a star import re-exports nothing, so reading
    # them off ``AllChem`` means reaching for names its stubs do not carry.
    from rdkit.Chem.rdDistGeom import EmbedMolecule
    from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if EmbedMolecule(mol, maxAttempts=max_attempts, randomSeed=random_seed) != 0:
        raise ValueError(f"failed 3D embedding for SMILES: {smiles}")
    UFFOptimizeMolecule(mol)
    return _conformer_rows(mol)


def smiles_to_xyz(
    csv_file: str | Path,
    output_dir: str | Path,
    *,
    smiles_column: str = "smiles",
    max_attempts: int = 10,
    random_seed: int = 42,
) -> list[Path]:
    """Convert a CSV column of SMILES into individual 3D XYZ files.

    Each successful conversion writes ``output_dir/structure_{row}.xyz``.
    Invalid SMILES are logged and skipped — they do not abort the run.
    Embedding itself is :func:`embed_smiles`, per row.
    """
    import pandas as pd

    out = Path(output_dir)
    df = pd.read_csv(csv_file)
    if smiles_column not in df.columns:
        raise ValueError(f"column {smiles_column!r} not found in {csv_file}")

    written: list[Path] = []
    for idx, raw in enumerate(df[smiles_column]):
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            rows = embed_smiles(raw, max_attempts=max_attempts, random_seed=random_seed)
        except ValueError as e:
            logger.warning("row %d: %s", idx, e)
            continue
        written.append(
            write_single_xyz(rows, out / f"structure_{idx}.xyz", comment=f"SMILES: {raw}")
        )
    return written


# ---------------------------------------------------------------------------
# Per-step CSV report
# ---------------------------------------------------------------------------


def _boltzmann_columns(
    energy_kcal: NDArray[np.float64], temperature_k: float
) -> dict[str, NDArray[np.float64]]:
    """Return the four Boltzmann-derived report columns for a sorted energy array."""
    dE = energy_kcal - energy_kcal.min()
    weights = boltzmann_weights(dE, temperature_k)
    pct_total = weights * 100.0
    return {
        "dE (kcal/mol)": dE,
        "Boltzmann Weight": weights,
        "% Total": pct_total,
        "% Cumulative": np.cumsum(pct_total),
    }


def save_step_csv(
    energies_hartree: Iterable[float | None],
    structure_ids: Iterable[str],
    step_number: int,
    path: Path,
    *,
    temperature_k: float,
    energy_type: EnergyType,
) -> Path:
    """Append a per-structure summary row for ``step_number`` to the cumulative CSV at ``path``.

    Columns are :data:`STEPS_CSV_COLUMNS`, in that order — selected by it just before the
    write, so this function cannot quietly emit a shape the constant does not describe.
    Sorted by energy ascending. Step 1 writes the header; later steps
    append without a header.

    ``energy_type`` names which energy the caller passed (a
    :data:`~chemrefine.config.EnergyType`) and is recorded verbatim in the last
    column. It is a column rather than a rename of the energy headers because
    ``steps.csv`` is cumulative across steps that may filter on different energies —
    one header has to serve them all.

    Structures whose energy is ``None`` are dropped from the report (nothing to
    summarise), which is how a backfilled ``on_failure: best`` structure with no
    computed energy passes through without breaking the Boltzmann columns.
    """
    import pandas as pd

    if step_number == 1:
        # Step 1 starts a fresh report each run; later steps append to it.
        # Truncate up front so a step 1 with no finite energies (the early
        # return below) can't leave a prior run's stale rows for step 2 to
        # append onto.
        remove_tree_file(path)

    df = pd.DataFrame(
        {
            "Step": step_number,
            "Conformer": list(structure_ids),
            "Energy (Hartree)": list(energies_hartree),
        }
    )
    df["Energy (kcal/mol)"] = pd.to_numeric(
        df["Energy (Hartree)"] * HARTREE_TO_KCALMOL, errors="coerce"
    )
    df = df.dropna(subset=["Energy (kcal/mol)"])
    if df.empty:
        # Nothing finite to summarise (all energies None/NaN) — skip the row
        # rather than crash computing Boltzmann columns on an empty frame.
        logger.warning("step %d: no finite energies to summarise; skipping CSV", step_number)
        return path
    df = df.sort_values("Energy (kcal/mol)").reset_index(drop=True)

    for column, values in _boltzmann_columns(
        df["Energy (kcal/mol)"].to_numpy(dtype=float), temperature_k
    ).items():
        df[column] = values
    df = df.round(
        {
            "Energy (kcal/mol)": _CSV_PRECISION,
            "dE (kcal/mol)": _CSV_PRECISION,
            "Boltzmann Weight": _CSV_PRECISION,
            "% Total": _CSV_PRECISION,
            "% Cumulative": _CSV_PRECISION,
        }
    )
    # Last column, so the leading header columns stay stable for existing tooling.
    df["Energy type"] = energy_type

    # Order and completeness in one line: a column this function forgot to build raises
    # KeyError here rather than shipping a report missing it, and the write order is the
    # documented one by construction instead of by the order the frame happened to grow.
    df = df[list(STEPS_CSV_COLUMNS)]

    # The header follows the *file*, not the step number. Keyed off `step_number == 1`, a
    # step 1 that summarises nothing (every energy None — which `on_failure: best` produces
    # on step 1, where the backfills are seeds with no energy yet) leaves step 2 appending
    # header-less to a file that does not exist, so the report opens with a data row and
    # every `read_csv` takes it for the column names.
    with tree_write(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, mode="a", index=False, header=not path.exists())
    logger.info("saved step %d summary to %s", step_number, path)
    return path
