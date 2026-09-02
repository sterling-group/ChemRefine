"""Per-step result cache for skip-on-resume.

Each step writes its parsed results to ``{step_dir}/_cache/`` as **two
files**: ``step.json`` holds the step metadata and every structure's
scalar fields (id, lineage, energies, status flags, symbols), and
``arrays.npz`` holds the bulk — the coordinates and forces.

Neither can execute code on load, which is why this is not pickle: JSON
cannot by construction, and ``.npz`` cannot because :func:`_read_arrays`
passes ``allow_pickle=False`` and numpy *raises* rather than running an
object array's reduce. Coordinates round-trip byte-identically either
way, so :func:`structure_digest` is stable across save → load.

The split is what makes the cache scale. Coordinates dominate a record,
and as decimal text each float64 costs 18 bytes on disk, a ``strtod``
call to parse and 32 bytes live; in a ``.npy`` member it costs 8 bytes,
a memcpy and 8 bytes. ``tests/test_perf_cache.py`` measures the
difference and ``docs/running/caching.md`` reports it. Structures in a
step need not share an atom count, so the arrays are concatenated with
an offsets index rather than stacked — see :func:`_split_arrays`.

``step.json`` is written without indentation (see :func:`write_json`) —
read it with ``jq`` or :func:`json.load`, not by eye. The per-structure
``.result.json`` records beside each output stay indented and keep their
coordinates inline; those are the ones a person opens, and they are a
few KB each.
The cache is keyed by a SHA-1 *fingerprint* covering the step's config
(engine, operation, options, charge, multiplicity, template, NMS flag)
plus the parent structures that fed into the step — their IDs **and**
their content (:func:`structure_digest`: symbols, coordinates, energy).
If the YAML changes, or the seed file / any upstream result changes,
the fingerprint changes and the next run re-executes the step. The
``sample:`` filter is deliberately **excluded**: the cache stores the
*pre-filter* results and filtering re-runs on every load, so tuning a
filter must refilter the cached results, not redo the calculations
(downstream steps still invalidate through the changed survivor set).

Writes are atomic — each file is written to a ``.tmp_*`` file inside the
cache directory and renamed into place, so an interrupted write never
produces a half-baked file. The *pair* needs one more thing: the two are
separate writes, so an interrupted save can leave a new ``arrays.npz``
beside the previous ``step.json``. The document records the digest of the
arrays it was written with (:func:`_require_paired`), so a mismatched pair
is a rebuild rather than a structure wearing another's coordinates.

This module also owns the per-step **manifest** (``{step_dir}/_cache/
manifest.json``): the input→output→structure-ID file layout that
produced the results. The cache document records the *parsed results*;
the manifest records the *file layout*, so ``rerun`` / recovery can
rehydrate which input produced which output after a restart. Both are
the same concern — per-step state under ``_cache/`` — so they live in
one module and share the atomic writer.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from chemrefine import ids
from chemrefine.config import StepConfig
from chemrefine.errors import CacheError
from chemrefine.state import (
    FailureRecord,
    StepInputs,
    StepResults,
    Structure,
)

CACHE_FORMAT_VERSION = "v2.0"
"""On-disk cache schema version, tracking the 2.0.0 release line.

Bump whenever the ``step.json`` document layout changes in a way that
would silently misread an older cache, **or** when the fingerprint
payload changes (older stored fingerprints would never match again,
which looks like a silent mass invalidation); :func:`load` rejects any
cache whose ``cache_format`` differs, forcing a clean rebuild rather
than a wrong read."""

RESULT_FORMAT_VERSION = "v1.0"
"""Schema version of the per-calculation ``*.result.json`` records.

The record body is :func:`structure_record` — the same schema the cache
document's ``structures`` entries use — wrapped in a ``result_format``
envelope. Bump when a key is renamed/removed or its semantics change;
purely additive keys need no bump (:func:`structure_from_record` reads
the optional ones with ``.get``)."""

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepCache:
    """In-memory snapshot of one step's parsed results.

    :func:`save` serializes it to the ``step.json`` document (structures
    via :func:`structure_record`); :func:`load` rebuilds it.
    """

    cache_format: str
    chemrefine_version: str
    fingerprint: str
    step: int
    name: str | None
    engine: str
    operation: str | None
    parent_ids: tuple[str, ...]
    results: StepResults
    on_failure: str = ""
    """The ``on_failure`` policy the stored results were finalized under.

    The policy shapes what :func:`save` persists — ``stop`` and ``skip`` store the
    successes alone, ``best`` stores the backfilled failures too — so a cache can only
    *serve* a config whose policy wants that same shape.
    :func:`chemrefine.step._policy_conflict` compares this against the current config; a
    step whose policy moved across that line re-attempts its ledgered failures and
    re-finalizes rather than serving the previous policy's survivor set as if it were its
    own. ``""`` is a document written before this key existed and is treated as serving
    any policy — an additive key read with ``.get``, so older caches and the recorded e2e
    archives are not stranded (which is also why there is no format bump)."""


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def _fingerprint_sha1() -> hashlib._Hash:
    """A SHA-1 marked as a content fingerprint — the constructor form for a streamed hash.

    Every other hash in this module says ``usedforsecurity=False`` inline, which is what
    lets a host whose crypto policy forbids SHA-1 *as a digest* still compute a fingerprint
    with it. :func:`hashlib.file_digest` cannot say that: handed the name ``"sha1"`` it
    builds the object through ``hashlib.new(...)`` with the flag left at its default, and
    the only way to reach the flag is to hand it a constructor instead. This is that
    constructor.

    The gap it closes was invisible to review for a mechanical reason worth recording:
    ruff's ``S324`` matches ``hashlib.sha1(...)`` by name and does not match
    ``file_digest(handle, "sha1")``, so the lint-visible sites acquired the flag and the two
    streamed ones did not. Nothing chose that asymmetry.
    """
    return hashlib.sha1(usedforsecurity=False)


def structure_digest(s: Structure) -> str:
    """Return a 16-char SHA-1 over one structure's *content* — its identity as an input.

    Covers the ID, chemical symbols, Cartesian coordinates (exact float64 bytes —
    identical inputs parse to identical floats), and energy. This is the geometry half
    of a :func:`row_key`: a job is keyed to the structure it computes on, so editing a
    seed or any upstream change to one parent's geometry/energy re-keys exactly the
    rows that consumed it — and no others.
    """
    h = hashlib.sha1(usedforsecurity=False)  # a content fingerprint, not a digest
    h.update(s.id.encode())
    h.update("".join(s.atoms.get_chemical_symbols()).encode())
    h.update(np.asarray(s.atoms.get_positions(), dtype=np.float64).tobytes())
    h.update(repr(s.energy_hartree).encode())
    return h.hexdigest()[:16]


def template_digest(path: Path | None) -> str:
    """Return a 16-char SHA-1 over a step template's bytes; ``""`` when there is none.

    The ``template_digest`` half of :func:`fingerprint`, in the module that owns cache keys —
    the same reason :func:`reuse_fingerprint` lives here. One reader for
    :attr:`~chemrefine.state.StepContext.template`, so the format cannot drift between
    engines: two of them hashing the same file to different keys is not a crash, it is a step
    that silently re-runs or silently does not.

    **The one home for what an absent template means to a key**, and both absences mean the
    same thing here. ``None`` is an engine that reads no template at all; a path that is not a
    file is a template that was named and is missing. Neither can contribute bytes, so neither
    contributes a digest — and for the second the ``""`` is load-bearing: it *changes* the
    fingerprint, so the step misses its cache and re-runs, and the actionable error arrives
    from :func:`chemrefine.ids.require_template` at the moment of rendering rather than from
    the cache.
    """
    if path is None or not path.is_file():
        return ""
    return hashlib.sha1(path.read_bytes(), usedforsecurity=False).hexdigest()[:16]


def option_file_digests(options: Mapping[str, Any] | None) -> dict[str, str]:
    """Digest every ``step.options`` value that names a file on disk, keyed by option.

    The counterpart to :func:`template_digest` for the *other* file a step can be pinned to.
    A step that names a model — ``model_path``, or a ``model_name`` that is a path — depends
    on that file's contents exactly as it depends on its template, and the raw ``options``
    dict in :func:`fingerprint` records only the *string*. Without these digests, retraining
    a model in place would leave every consuming step's key unchanged and ``resume`` would
    serve results computed with the previous weights — and nothing else moves that key,
    because a training step passes its structures through untouched.

    Generic rather than a list of known knobs, and that is the point: it needs no
    engine vocabulary, so a backend that invents a checkpoint knob tomorrow is covered by
    existing rather than by remembering to edit this. The cost of the generality is bounded
    — only values that resolve to a real file are read, and a step naming none pays nothing.

    ``model_path`` — the shipped option this exists for — arrives here already absolute: the
    config loader resolves it against the config file's directory
    (:func:`chemrefine.config._resolve_step_option_paths`), exactly as it resolves the
    config's own paths, so the digest and the engine that later loads the file read the same
    one. Any *other* value that happens to name a file resolves against the working
    directory, nothing having declared a better anchor for it. A value that is not an
    existing file contributes **no entry at all** rather than an empty one: "not a path" and
    "a path that is missing" are different claims, and only the latter should later change
    the key when the file appears.

    Each file is **streamed**, not read whole. This runs on every :meth:`StepKey.of` — for
    every step, on every run, in the driver process — and the files it is here for are model
    checkpoints: a UMA one is 1-2 GB, and on a cluster the driver is a login node.
    """
    digests: dict[str, str] = {}
    for name, value in sorted((options or {}).items()):
        if not isinstance(value, str) or not value:
            continue
        try:
            path = Path(value)
            if not path.is_file():
                continue
            with path.open("rb") as handle:
                digests[name] = hashlib.file_digest(handle, _fingerprint_sha1).hexdigest()[:16]
        except OSError:
            # A value that merely looks like a path — too long for the filesystem, a
            # permission wall, a dangling mount. Not a file we can pin to, and not a reason
            # to fail a run that never asked for one.
            continue
    return digests


def aux_file_digests(references: Mapping[str, Path]) -> dict[str, str]:
    """Digest the files a step's template references, keyed by the *written* reference.

    The third member of the file-pinning family, beside :func:`template_digest` and
    :func:`option_file_digests`, with the latter's rationale applying verbatim: a
    template that quotes a docking guest or point-charge file makes every job's result
    depend on that file's bytes, while the template digest covers only the *path
    string* — so editing the referenced file in place left every fingerprint standing
    and ``resume`` served results computed from the old file. Which strings reference
    which files is the engine's reading, not this module's
    (:class:`chemrefine.engines.api.AuxFileConsuming`); this digests the enumeration it
    is handed, streamed like :func:`option_file_digests` and with its same escape — a
    file that cannot be read contributes no entry, and the key moves when it can be.

    The key is the reference as the template writes it, **never** the resolved path:
    resolved paths are absolute, and an absolute string inside a row key breaks the
    guarantee a relocated tree depends on — ``rebuild-cache`` re-deriving the same keys
    from the same bytes wherever the project now sits. The written spelling travels
    with the template, and it also distinguishes two references whose files happen to
    carry identical bytes today.
    """
    digests: dict[str, str] = {}
    for written, path in sorted(references.items()):
        try:
            with path.open("rb") as handle:
                digests[written] = hashlib.file_digest(handle, _fingerprint_sha1).hexdigest()[:16]
        except OSError:
            continue
    return digests


def _hash_payload(payload: dict[str, Any]) -> str:
    """A 16-char SHA-1 over a compact, key-sorted JSON encoding of ``payload``.

    The one encoder behind every key in this module, so two keys can never disagree
    about how a value is spelled into bytes.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded, usedforsecurity=False).hexdigest()[:16]


def row_key(
    *,
    engine: str,
    operation: str | None,
    template_digest: str,
    charge: int,
    multiplicity: int,
    engine_options: Mapping[str, Any],
    option_digests: Mapping[str, str],
    aux_digests: Mapping[str, str],
    parent_digest: str,
) -> str:
    """One structure's job identity — everything that determines *this row's* result.

    The fields are exactly the config surface that can reach a job plus its geometry:
    the engine and operation select the code, the template digest is the input text,
    ``charge``/``multiplicity`` are the **effective** (inheritance-resolved) physics
    inputs, ``engine_options`` is the options mapping *as the engine's own declared
    model reads it* (``{}`` for an engine that declares none — an undeclared key can
    reach no job), ``option_digests`` pins the bytes of any file an option names,
    ``aux_digests`` (:func:`aux_file_digests`) the bytes of any file the template
    references, and ``parent_digest`` (:func:`structure_digest`) is the geometry the
    job computes on. The two digest families join the payload only when non-empty, so
    a step naming no files keys exactly as it always has.

    Deliberately absent: the NMS family (post-round-1 resolution — the resolution key's
    business), the parent *set* (aggregation — the step fingerprint's business),
    ``sample``/``on_failure`` (filter and policy re-run on every load), and every
    location/scheduler knob. A row key that matches is proof the job on disk is the job
    this configuration would submit for this parent.
    """
    payload: dict[str, Any] = {
        "format": CACHE_FORMAT_VERSION,
        "engine": engine,
        "operation": operation,
        "template_digest": template_digest,
        "charge": charge,
        "multiplicity": multiplicity,
        "engine_options": dict(engine_options),
        "parent_digest": parent_digest,
    }
    if option_digests:
        payload["option_digests"] = dict(option_digests)
    if aux_digests:
        payload["aux_digests"] = dict(aux_digests)
    return _hash_payload(payload)


@dataclass(frozen=True)
class ResolutionSpec:
    """What an NMS step's resolution reads, split the way the machinery splits it.

    ``criterion`` (``target`` / ``ts_mode_index``) decides what counts as resolved;
    ``search`` (``displacement_value`` / ``num_random_displacements`` / ``seed``) tunes
    how children are generated. Both mappings are the *validated* ``NmsOptions``
    reading, passed in by the caller that owns that model — this module keys it without
    knowing what it means. The split is load-bearing: an ``attemptK/`` resolution stays
    honourable across a search retune (same criterion) but never across a criterion
    change.
    """

    criterion: Mapping[str, Any]
    search: Mapping[str, Any]


def resolution_keys(resolution: ResolutionSpec | None) -> tuple[str, str, str]:
    """``(resolution_key, criterion_key, search_key)`` — all ``""`` when nothing resolves.

    The ``nms`` flag is expressed as this key's presence; it appears in no row key,
    because it is read only by the resolution machinery and can never change a job.

    The two halves are keyed separately because they age differently on disk. A
    *criterion* retune (``target`` / ``ts_mode_index``) changes which children are
    selected and what counts as resolved, but never a child's geometry — so an
    ``attemptK/`` on disk stays re-readable across it, which is ``rebuild-nms``'s
    whole offer. A *search* retune (``displacement_value`` / ``num_random_displacements``
    / ``seed``) changes the geometries themselves while the child ids stay the same,
    so the same attempt answers a question this configuration never asked — the one
    adoption ``rebuild-cache`` must refuse.
    """
    if resolution is None:
        return "", "", ""
    criterion = _hash_payload({"criterion": dict(resolution.criterion)})
    search = _hash_payload({"search": dict(resolution.search)})
    full = _hash_payload(
        {"criterion": dict(resolution.criterion), "search": dict(resolution.search)}
    )
    return full, criterion, search


# ---------------------------------------------------------------------------
# Paths + I/O
# ---------------------------------------------------------------------------


def _cache_path(step_dir: Path) -> Path:
    """Return the ``step.json`` cache document path inside ``step_dir/_cache/``."""
    return step_dir / "_cache" / "step.json"


def _arrays_path(step_dir: Path) -> Path:
    """Return the ``arrays.npz`` coordinate sidecar path inside ``step_dir/_cache/``."""
    return step_dir / "_cache" / "arrays.npz"


#: Record keys held in the ``arrays.npz`` sidecar rather than inline in ``step.json``.
_ARRAY_KEYS = ("positions", "forces_ev_per_a")


class _SidecarArrays(TypedDict):
    """Every member ``arrays.npz`` carries, named once.

    A shape rather than a ``dict[str, NDArray]`` because the members are not
    interchangeable — two are ``int64`` indices into the other two, and one is a mask —
    and because ``np.savez`` takes each as its own keyword beside ``allow_pickle``. Spelled
    out, the payload can be handed to it whole.
    """

    positions: NDArray[np.float64]
    position_offsets: NDArray[np.int64]
    forces: NDArray[np.float64]
    forces_offsets: NDArray[np.int64]
    forces_present: NDArray[np.bool_]


def _split_arrays(records: list[dict[str, Any]]) -> _SidecarArrays:
    """Move every record's coordinate arrays into one flat ``.npz`` payload.

    ``records`` is **mutated**: the two array keys are removed, leaving the metadata
    document. Structures in one step need not share an atom count — ``_seed_from_directory``
    and ``_seed_from_smiles_csv`` both seed different molecules into a single step — so the
    arrays are concatenated into one ``(Σn_atoms, 3)`` block plus an ``offsets`` index rather
    than stacked. A stack would simply raise on that input.

    ``forces_ev_per_a`` is ``None`` per structure whenever the engine reported none, so it
    carries its own offsets and a boolean present-mask; only the structures that have forces
    contribute rows.
    """
    positions = [np.asarray(r.pop("positions"), dtype=np.float64) for r in records]
    forces_raw = [r.pop("forces_ev_per_a") for r in records]
    present = np.array([f is not None for f in forces_raw], dtype=np.bool_)
    forces = [np.asarray(f, dtype=np.float64) for f in forces_raw if f is not None]
    return {
        "positions": _concat(positions),
        "position_offsets": _offsets(positions),
        "forces": _concat(forces),
        "forces_offsets": _offsets(forces),
        "forces_present": present,
    }


def _concat(blocks: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Concatenate per-structure ``(n, 3)`` blocks; an empty list gives a ``(0, 3)`` array.

    ``np.concatenate`` raises on an empty sequence, and a step where no structure reported
    forces is ordinary, not exceptional.
    """
    return np.concatenate(blocks) if blocks else np.zeros((0, 3), dtype=np.float64)


def _offsets(blocks: list[NDArray[np.float64]]) -> NDArray[np.int64]:
    """Row index where each block starts, plus a final total — ``len(blocks) + 1`` entries."""
    return np.concatenate([[0], np.cumsum([len(b) for b in blocks])]).astype(np.int64)


def _arrays_digest(arrays: Any) -> str:
    """Return a 16-char SHA-1 over the sidecar's array *contents*.

    The pairing key between the two files of one save. It covers the values rather than the
    ``.npz`` bytes because a ZIP member carries a timestamp, so the same coordinates written
    twice are not the same file — and a key that changed every save would make a parse-only
    rebuild differ from the recording it rebuilt.

    Two saves whose arrays hash alike hold the same coordinates, which is exactly when
    pairing either document with either sidecar is harmless.
    """
    h = hashlib.sha1(usedforsecurity=False)  # a content fingerprint, not a digest
    for name in sorted(arrays):
        h.update(name.encode())
        h.update(np.ascontiguousarray(arrays[name]).tobytes())
    return h.hexdigest()[:16]


def _require_paired(arrays: Any, document: dict[str, Any], path: Path) -> None:
    """Raise :class:`CacheError` unless the sidecar was written by the save that wrote ``document``.

    The two files are separate atomic writes, so a save interrupted between them leaves a new
    sidecar beside the previous document. Nothing about that pair is malformed and no check
    upstream of this one can see it: the records parse, and the fingerprint still matches the
    configuration, because it covers the step's *inputs* rather than what is on disk. What
    reaches the caller is a structure keeping its old energy and adopting another structure's
    geometry — the value then feeds :func:`structure_digest`, so the wrong coordinates propagate
    into every step computed from them.

    Checked before :func:`_join_arrays` rather than after, because the same mismatch that
    misreads coordinates also indexes past the end of a shorter sidecar.
    """
    found = _arrays_digest(arrays)
    if found != document["arrays_digest"]:
        raise CacheError(
            f"step.json and arrays.npz at {path.parent} are from different saves "
            f"(document expects {document['arrays_digest']}, sidecar holds {found}); "
            f"rebuild the step"
        )


def _join_arrays(records: list[dict[str, Any]], arrays: Any) -> None:
    """Re-attach the sidecar's arrays to their records (inverse of :func:`_split_arrays`).

    Slices are copied out of the loaded blocks: a view would keep the whole concatenated
    array alive for as long as any one structure survives filtering.
    """
    pos, pos_off = arrays["positions"], arrays["position_offsets"]
    frc, frc_off, present = arrays["forces"], arrays["forces_offsets"], arrays["forces_present"]
    seen = 0
    for i, record in enumerate(records):
        record["positions"] = pos[pos_off[i] : pos_off[i + 1]].copy()
        if present[i]:
            record["forces_ev_per_a"] = frc[frc_off[seen] : frc_off[seen + 1]].copy()
            seen += 1
        else:
            record["forces_ev_per_a"] = None


def structure_record(s: Structure) -> dict[str, Any]:
    """Serialize one :class:`Structure` to the canonical parsed-result record.

    **This is the one schema for a parsed calculation result.** It feeds the
    ``structures`` entries of the ``_cache/step.json`` document, the per-job
    ``step{N}_{id}.result.json`` artifacts (:func:`save_result_records`), and
    the per-engine contract-test goldens (``tests/data/engines/``) — every
    engine's parse must land in this shape.

    Keys (null when the calculation didn't report the value):

    - ``id`` / ``parent_id`` — lineage
    - ``energy_hartree``, ``gibbs_hartree``, ``enthalpy_hartree``,
      ``energy_zpe_hartree`` — energies [Hartree]
    - ``converged`` / ``terminated_normally`` — run status (``None`` = not reported)
    - ``symbols``, ``positions`` — the geometry [Å]
    - ``forces_ev_per_a`` — forces [eV/Å]
    - ``imaginary_freqs`` — mode index (JSON string) → frequency [cm⁻¹]
    - ``frequencies`` — the whole mode table, same shape, same index space
    - ``resolved_from`` — the NMS child this structure's artifacts came from

    ``resolved_from`` and ``frequencies`` are additive: :func:`structure_from_record` reads
    them with ``.get``, so a record written before they existed loads as ``None`` and needs
    no :data:`RESULT_FORMAT_VERSION` bump. A tree cached before ``frequencies`` existed
    gains it by re-parsing its outputs — that is what ``rebuild-cache`` is for.

    Only symbols + positions of the ``Atoms`` are stored — that is all the
    pipeline ever reads back (and all that :func:`structure_digest` hashes).
    ``normal_modes`` is deliberately excluded: a transient displacement
    tensor used only during an active NMS run, which always re-parses the
    native output.
    """
    return {
        "id": s.id,
        "parent_id": s.parent_id,
        "energy_hartree": s.energy_hartree,
        "gibbs_hartree": s.gibbs_hartree,
        "enthalpy_hartree": s.enthalpy_hartree,
        "energy_zpe_hartree": s.energy_zpe_hartree,
        "converged": s.converged,
        "terminated_normally": s.terminated_normally,
        "symbols": list(s.atoms.get_chemical_symbols()),
        "positions": np.asarray(s.atoms.get_positions(), dtype=np.float64).tolist(),
        "forces_ev_per_a": (
            None
            if s.forces_ev_per_a is None
            else np.asarray(s.forces_ev_per_a, dtype=np.float64).tolist()
        ),
        # The mode tables round-trip (small, useful metadata); JSON keys must be strings.
        # ``normal_modes`` is deliberately NOT persisted — it's a transient displacement tensor
        # only used during an active NMS run (which always re-parses), so a cache-reloaded
        # structure carries ``None`` (never read).
        "imaginary_freqs": _freq_record(s.imaginary_freqs),
        "frequencies": _freq_record(s.frequencies),
        "resolved_from": s.resolved_from,
    }


def _freq_record(table: dict[int, float] | None) -> dict[str, float] | None:
    """A mode table on its way to JSON, whose object keys can only be strings.

    ``None`` is carried through rather than flattened to ``{}``: no frequency table at all
    and a table with nothing in it are different answers, and NMS's resolution rule
    (``nms._is_resolved``) branches on exactly that difference.
    """
    return None if table is None else {str(mode): cm1 for mode, cm1 in table.items()}


def _freq_from_record(raw: Any) -> dict[int, float] | None:
    """The inverse of :func:`_freq_record` — mode indices back to the ints they are."""
    return None if raw is None else {int(mode): cm1 for mode, cm1 in raw.items()}


def structure_from_record(d: dict[str, Any]) -> Structure:
    """Rebuild a :class:`Structure` from its canonical record (inverse of the above)."""
    forces = d["forces_ev_per_a"]
    return Structure(
        id=d["id"],
        atoms=Atoms(symbols=d["symbols"], positions=d["positions"]),
        parent_id=d["parent_id"],
        energy_hartree=d["energy_hartree"],
        forces_ev_per_a=None if forces is None else np.asarray(forces, dtype=np.float64),
        converged=d["converged"],
        terminated_normally=d["terminated_normally"],
        # Thermochemistry + the mode tables are additive — older caches lack these keys.
        gibbs_hartree=d.get("gibbs_hartree"),
        enthalpy_hartree=d.get("enthalpy_hartree"),
        energy_zpe_hartree=d.get("energy_zpe_hartree"),
        imaginary_freqs=_freq_from_record(d.get("imaginary_freqs")),
        frequencies=_freq_from_record(d.get("frequencies")),
        resolved_from=d.get("resolved_from"),
    )


def save_result_records(structures: Sequence[Structure], job_dir: Path, step: int) -> None:
    """Write each structure's canonical ``*.result.json`` record into ``job_dir``.

    The engine-independent parsed-result artifact: whatever native output a
    backend produced (ORCA ``.out``, a script footer JSON, …), the pipeline
    drops the normalized :func:`structure_record` next to it after parsing.
    A derived artifact — inspection, tooling, and contract goldens read it;
    the pipeline itself never does (rebuilds re-parse the native output).
    """
    for s in structures:
        record = {"result_format": RESULT_FORMAT_VERSION, **structure_record(s)}
        write_json(ids.result_record_path(job_dir, step, s.id), record)


_PROC_STATUS = Path("/proc/self/status")
"""Where Linux publishes this process's umask — see :func:`_umask`."""


def _umask() -> int:
    """This process's umask, read without writing it.

    ``os.umask`` is the only POSIX way to *read* the umask and it reads by setting it,
    which is process-global: between the clear and the restore, anything another thread
    creates is made with no mask at all, so a ``mkdir`` lands 0777. That window is
    reachable rather than theoretical — the GUI is served by waitress on four threads
    whose handlers both write through here (``agent_tools.save_config``) and call
    ``Path.mkdir`` (``/api/save``, ``/api/scaffold``, ``/api/template``, ``/api/run``).

    ``umask(2)`` names the remedy itself: since Linux 4.7 the value is published in
    ``/proc/self/status``, and reading it changes nothing. CPython was asked for a
    thread-safe wrapper and declined (bpo-35275, wontfix — the POSIX API has none
    either), so this is the documented answer rather than a workaround. The probe stays
    as the fallback for a kernel that does not publish the field; there the window is
    back, which is the narrower of the two evils.
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


def atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file + fsync + rename.

    The ``fsync`` is what makes the atomicity survive more than a process death: a
    rename is ordered against the data only once the data is on the device, so without
    it a machine crash (not a kill) could leave the renamed file truncated or empty —
    exactly the half-baked cache the temp-file dance exists to prevent.

    Public, not underscored, because :func:`chemrefine.agent_tools.save_config` writes the
    user's config through it: a write that must not be torn has to be able to reach the
    writer that guarantees it, wherever the caller lives.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            # ``mkstemp`` creates 0600 and the rename preserves it — right for a private temp
            # file, wrong for the cache document it becomes. Left alone, every ``_cache/``
            # file on a shared tree was owner-only: a colleague handed the outputs could read
            # the ``.out`` files but not the cache, the manifest or the failure ledger beside
            # them. Re-moded to what a plain ``open()`` would have given — 0666 honouring the
            # umask. (The server *token* sidecar keeps mkstemp's 0600; there the restriction
            # is the point.)
            #
            # Done through the open file rather than the bare descriptor, and inside the
            # ``with`` rather than before it, because this is the one step here that can
            # genuinely fail — shared filesystems and FUSE/CIFS mounts do refuse ``fchmod``.
            # Ahead of the ``fdopen`` its failure stranded both halves: nothing owned the
            # descriptor yet, so it leaked, and the ``finally`` had not been entered, so the
            # ``.tmp_*.part`` stayed behind in the user's directory. One failed write in a
            # long-lived driver is a nuisance; the GUI writes the user's config through here
            # on four waitress threads, where it would have been one of each per attempt.
            os.fchmod(fh.fileno(), 0o666 & ~_umask())
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(path)
    finally:
        Path(tmp).unlink(missing_ok=True)


#: ``json.dumps`` separators with no padding: ``{"a":1,"b":2}`` rather than ``{"a": 1, "b": 2}``.
#: Only worth using where a document is machine-read and large — see :func:`write_json`.
_COMPACT_SEPARATORS = (",", ":")


def write_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
    """Serialize ``data`` to JSON and write it atomically to ``path``.

    Indented by default: the manifest, the failed-jobs ledger and the per-structure
    ``.result.json`` records are all things a person opens, and all small enough that
    readability is the only property that matters.

    ``indent=None`` switches to compact separators, for the step document alone. At a
    realistic step size the indentation is roughly half the file — a coordinate record is
    thousands of short numeric values, and each carries its own newline and run of spaces.
    Nothing reads that document by eye, and nothing reads it by line either: the loader
    parses it whole, so the layout reaches no consumer.

    ``allow_nan=False`` because Python's default emits bare ``NaN`` / ``Infinity``, which
    are not JSON: ``jq`` reads ``NaN`` back as ``null``, so a diverged calculation would
    reach any external consumer of a ``.result.json`` as "no energy computed" instead. The
    parse boundary (:func:`chemrefine.engines._script.output._require_finite`) already
    refuses those values, so this is the assertion that it did — a ``ValueError`` here
    means something upstream let one through.
    """
    separators = None if indent is not None else _COMPACT_SEPARATORS
    atomic_write(
        path, json.dumps(data, indent=indent, separators=separators, allow_nan=False).encode()
    )


def _npz_bytes(arrays: _SidecarArrays) -> bytes:
    """Serialize ``arrays`` to an uncompressed ``.npz`` in memory.

    Uncompressed on purpose. A ``.npz`` is an ordinary ZIP of ``.npy`` members, and a ``.npy``
    is a short ASCII header plus the array's raw buffer — byte-identical to ``tobytes()``,
    which is what keeps coordinates exact through save → load and :func:`structure_digest`
    stable. Deflating float64 coordinates buys about 5% and costs roughly twenty times the
    encode; the point of this format is that writing it is a memcpy.

    Going through bytes rather than writing the file directly is what lets it reuse
    :func:`atomic_write`, so a killed run never leaves a half-written sidecar.
    """
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _read_arrays(path: Path) -> Any:
    """Load the coordinate sidecar, or raise :class:`CacheError` if it is absent or corrupt.

    ``allow_pickle=False`` is spelled out because it is the whole reason this is not a
    pickle: numpy *enforces* it, raising rather than executing when a file smuggles in an
    object array. The cache's promise that loading it can never run code from the file
    therefore holds for the ``.npz`` sidecar too, as a check rather than a convention.
    """
    if not path.is_file():
        raise CacheError(f"step cache at {path.parent} has no {path.name}; rebuild the step")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}
    except (OSError, ValueError) as e:
        raise CacheError(f"corrupt coordinate sidecar at {path}: {e}") from e


def read_json(path: Path, default: Any, *, label: str) -> Any:
    """Return the JSON parsed from ``path``, or ``default`` if it doesn't exist.

    Raises :class:`CacheError` (naming ``label``) if the file is present but
    holds malformed JSON, so callers treat a corrupt sidecar as fatal rather
    than silently continuing from an empty state.
    """
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CacheError(f"corrupt {label} at {path}: {e}") from e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _require_finite_arrays(structure: Structure, step_dir: Path) -> None:
    """Raise :class:`CacheError` unless ``structure``'s coordinates and forces are finite.

    **The half of the no-NaN promise the JSON writer cannot make.** :func:`write_json` passes
    ``allow_nan=False``, so a non-finite value in the *document* is refused — but the
    coordinates and forces are the one part of a record that never reaches it: they are moved
    out by :func:`_split_arrays` and written to the ``arrays.npz`` sidecar, which is a raw
    buffer with no such check. So the very fields the document protects most carefully were
    the only ones that could be stored unexamined.

    What that cost is not a crash but a silence. A NaN geometry round-trips ``save`` → ``load``
    intact, and :func:`structure_digest` hashes it to a perfectly stable key, so the step
    validates, the fingerprint matches, and every later step is computed from coordinates
    that are not numbers — with nothing anywhere reporting a problem.

    A backstop, and meant to stay one: the parse boundaries and
    :func:`chemrefine.pipeline._state_from_frames` refuse these values where a person can act
    on them, naming the output file or the seed row. Reaching *here* means one of those was
    bypassed, which is why this raises rather than repairs.

    Forces are held to the same rule for the reason the parse boundary gives: a non-finite
    force is what an ``mlip-train`` step would go on to fit.
    """
    for field, array in (
        ("coordinates", structure.atoms.get_positions()),
        ("forces", structure.forces_ev_per_a),
    ):
        if array is not None and not np.isfinite(np.asarray(array, dtype=np.float64)).all():
            raise CacheError(
                f"structure {structure.id} has non-finite {field} (nan/inf) and will not be "
                f"cached to {step_dir / '_cache'}: the sidecar would store them and every "
                f"later step would be computed from them. This should have been refused at "
                f"the parse boundary — please report it."
            )


def save(
    *,
    step_cfg: StepConfig,
    key: StepKey,
    results: StepResults,
    step_dir: Path,
    chemrefine_version: str,
) -> None:
    """Persist ``results`` for ``step_cfg`` to ``step_dir/_cache/`` under ``key``.

    The key is a value the caller computed once (:meth:`StepKey.of`), not a recipe this
    function re-follows. Deriving it here instead would mean deriving it from whichever
    ``StepContext`` a caller happened to pass — and ``prev_state`` is rebound to a subset of
    the parents on the retry paths, which would key the step to a fingerprint nothing can
    match again.

    Refuses a structure whose coordinates or forces are not finite — see
    :func:`_require_finite_arrays` for why this is the one check the JSON half does not
    already make.
    """
    for structure in results.structures:
        _require_finite_arrays(structure, step_dir)
    records = [structure_record(s) for s in results.structures]
    # Moves the coordinates out of `records`, leaving the metadata document behind.
    arrays = _split_arrays(records)
    document = {
        "cache_format": CACHE_FORMAT_VERSION,
        "chemrefine_version": chemrefine_version,
        "fingerprint": key.fingerprint,
        "step": step_cfg.step,
        "name": step_cfg.name,
        "engine": step_cfg.engine,
        "operation": step_cfg.operation,
        # The policy these results were finalized under — see StepCache.on_failure.
        "on_failure": step_cfg.on_failure,
        "parent_ids": list(key.parent_ids),
        # Names the sidecar this document belongs to — see `_require_paired`.
        "arrays_digest": _arrays_digest(arrays),
        "structures": records,
    }
    # Sidecar first: a crash between the two writes then leaves an orphan `.npz` and no
    # `step.json`, which reads as a plain cache miss. The other order would leave a document
    # whose arrays are missing, and `load` fails closed on that — a miss is cheaper than an
    # error. The order alone is not enough, though: it makes the *first* save into an empty
    # `_cache/` atomic, and a step is re-saved whenever `resume` repairs one, which leaves the
    # previous document beside the new sidecar. `arrays_digest` is what makes the pair
    # provable rather than merely likely.
    atomic_write(_arrays_path(step_dir), _npz_bytes(arrays))
    write_json(_cache_path(step_dir), document, indent=None)
    logger.info("saved step %d cache (fingerprint %s)", step_cfg.step, key.fingerprint)


class ManifestStamp(TypedDict):
    """What a manifest records of the :class:`StepKey` it was written under.

    A shape rather than four loose keywords because :func:`save_manifest` defaults every
    stamp field to ``""`` — the *unprovable, adoptable* value — so a field forgotten at one
    of its call sites is not an error but a manifest that reads as "never proven wrong"
    and silently disarms the refusals built on it. Handed to ``save_manifest`` whole
    (``**key.manifest_stamp()``), the only way to forget is to forget the whole stamp,
    which is a row-less manifest and fails loud at the next resume.
    """

    fingerprint: str
    criterion_key: str
    search_key: str
    rows: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class StepKey:
    """A step's cache identity — computed once per step, then passed around as a value.

    Every route through :mod:`chemrefine.step` needs the same answer to "is the work on
    disk the work this configuration would produce". Computed once and passed as a value,
    that answer cannot vary between the routes; re-derived per route it is that many
    chances to disagree. A cache written under an inconsistent key does not crash: it
    silently re-runs work that was done, or reuses work it should not have, much later.

    The identity is layered the way the domain is layered. ``row_keys`` (one per parent,
    aligned with ``parent_ids``/``parent_digests``) are the per-structure job identities
    (:func:`row_key`); ``criterion_key``/``search_key`` are the NMS resolution's
    identity, kept as two halves because they age differently on disk
    (:func:`resolution_keys`, both ``""`` for a step that resolves nothing); and
    ``fingerprint`` composes them with the step number — an exact hit means "the whole
    step, bit for bit", while every finer question is asked of the rows.

    :meth:`of` is the one place the derivation happens. The effective charge and
    multiplicity, the engine's own reading of the options, and the resolution spec are
    *passed in* by the caller that holds the context and the engine
    (:func:`chemrefine.step.derive_step_key`) — this module keys values it does not
    interpret, which is what keeps it importable by everything.
    """

    parent_ids: tuple[str, ...] = ()
    parent_digests: tuple[str, ...] = ()
    row_keys: tuple[str, ...] = ()
    criterion_key: str = ""
    search_key: str = ""
    fingerprint: str = ""

    def manifest_rows(self) -> dict[str, tuple[str, str]]:
        """``id -> (row_key, parent_digest)`` — the per-row provenance a manifest stores."""
        return {
            sid: (key, digest)
            for sid, key, digest in zip(
                self.parent_ids, self.row_keys, self.parent_digests, strict=True
            )
        }

    def manifest_stamp(self) -> ManifestStamp:
        """The key's projection onto a manifest: ``save_manifest(..., **key.manifest_stamp())``.

        The step stamp (``fingerprint``), the two resolution halves and the row
        provenance, in one shape, so the four routes that write a manifest cannot each
        copy the quintet by hand and drift — :func:`save` already takes the key whole,
        and this is the manifest's equivalent.
        """
        return ManifestStamp(
            fingerprint=self.fingerprint,
            criterion_key=self.criterion_key,
            search_key=self.search_key,
            rows=self.manifest_rows(),
        )

    @classmethod
    def of(
        cls,
        step_cfg: StepConfig,
        parents: Sequence[Structure],
        template: Path | None,
        *,
        charge: int = 0,
        multiplicity: int = 1,
        engine_options: Mapping[str, Any] | None = None,
        resolution: ResolutionSpec | None = None,
        aux_files: Mapping[str, Path] | None = None,
    ) -> StepKey:
        """Derive the key for ``step_cfg`` run over ``parents`` with ``template``.

        ``charge``/``multiplicity`` are the **effective** values (the ones jobs render),
        not the per-step overrides — hashing the override let a workflow-level edit
        change every job's physics while every fingerprint stood still. Their defaults
        are the workflow defaults, so a bare call keys a default config exactly as a run
        would. ``engine_options`` is the engine's declared model's resolved dump
        (``None``/``{}`` for a non-declaring engine — an undeclared key can reach no
        job); ``resolution`` is the validated NMS reading for a step that resolves,
        else ``None``; ``aux_files`` is the engine's enumeration of the files the
        template references, written reference → file
        (:class:`chemrefine.engines.api.AuxFileConsuming` — empty for an engine whose
        templates name none), digested here beside the option files.
        """
        template_dig = template_digest(template)
        option_digs = option_file_digests(step_cfg.options)
        aux_digs = aux_file_digests(aux_files or {})
        parent_ids = tuple(s.id for s in parents)
        parent_digs = tuple(structure_digest(s) for s in parents)
        rows = tuple(
            row_key(
                engine=step_cfg.engine,
                operation=step_cfg.operation,
                template_digest=template_dig,
                charge=charge,
                multiplicity=multiplicity,
                engine_options=dict(engine_options or {}),
                option_digests=option_digs,
                aux_digests=aux_digs,
                parent_digest=digest,
            )
            for digest in parent_digs
        )
        res_key, crit_key, search_key = resolution_keys(resolution)
        step_fingerprint = _hash_payload(
            {
                "format": CACHE_FORMAT_VERSION,
                "step": step_cfg.step,
                "rows": list(rows),
                "resolution": res_key,
            }
        )
        return cls(
            parent_ids=parent_ids,
            parent_digests=parent_digs,
            row_keys=rows,
            criterion_key=crit_key,
            search_key=search_key,
            fingerprint=step_fingerprint,
        )


def load(step_dir: Path) -> StepCache | None:
    """Return the cached :class:`StepCache` under ``step_dir``, or ``None``.

    Raises :class:`CacheError` when the document exists but is malformed —
    bad JSON, missing fields, or a ``cache_format`` this version doesn't
    read. A summary-only ``step.json`` sitting beside a ``step.pkl`` lands here too
    (it has no ``structures``), forcing a clean rebuild rather than a wrong read.
    """
    path = _cache_path(step_dir)
    data = read_json(path, None, label="step cache")
    if data is None:
        return None
    try:
        if data["cache_format"] != CACHE_FORMAT_VERSION:
            raise CacheError(
                f"cache at {path} has format {data['cache_format']}; "
                f"expected {CACHE_FORMAT_VERSION}"
            )
        records = data["structures"]
        # Arrays come from the sidecar or not at all. A document written before the split
        # still carries them inline, and quietly reading those would be the one way to load
        # a cache half in each format — so the absent sidecar is an error, not a fallback.
        arrays = _read_arrays(_arrays_path(step_dir))
        _require_paired(arrays, data, path)
        _join_arrays(records, arrays)
        return StepCache(
            cache_format=data["cache_format"],
            chemrefine_version=data["chemrefine_version"],
            fingerprint=data["fingerprint"],
            step=data["step"],
            name=data["name"],
            engine=data["engine"],
            operation=data["operation"],
            parent_ids=tuple(data["parent_ids"]),
            results=StepResults(structures=tuple(structure_from_record(d) for d in records)),
            on_failure=data.get("on_failure", ""),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise CacheError(f"stale or corrupt cache at {path}: {e!r}") from e


def load_if_valid(*, key: StepKey, step_dir: Path) -> StepCache | None:
    """Return the cached :class:`StepCache` iff it was written under ``key``; else ``None``.

    A single ``load`` + fingerprint compare, so a caller that needs the cached results on a
    hit (e.g. :func:`chemrefine.step._cached_outcome`) reads ``step.json`` **once** instead
    of validating and then re-loading — and there is no window in which the cache could
    vanish between the two reads. A corrupt or absent cache returns ``None`` (treated as
    "re-run"), never raises.

    Takes the key rather than the ingredients to derive one: the whole point of
    :class:`StepKey` is that the comparison and the write cannot use different recipes.
    """
    try:
        cached = load(step_dir)
    except CacheError:
        return None
    if cached is None or cached.fingerprint != key.fingerprint:
        return None
    return cached


def invalidate(step_dir: Path) -> None:
    """Delete the cached *results* for ``step_dir``. No-op if nothing is cached.

    Deliberately narrow: the manifest survives, so ``rebuild-cache`` can still re-parse the
    outputs on disk after the results document is gone. Callers that mean "redo this step
    from scratch" want :func:`discard_step` instead.

    Also removes any ``step.pkl`` in the cache directory, so re-running over an old output
    tree leaves no stale binary around.
    """
    _cache_path(step_dir).unlink(missing_ok=True)
    (step_dir / "_cache" / "step.pkl").unlink(missing_ok=True)


def discard_step(step_dir: Path) -> None:
    """Forget that ``step_dir`` ever ran — results **and** manifest.

    The difference from :func:`invalidate` is the manifest, and it is load-bearing. Together
    the two files say "this step already ran with this configuration"; the manifest alone
    says "these outputs on disk belong to this configuration", which is what
    :func:`chemrefine.step._incremental_step_outcome` reads to continue an interrupted step.

    So leaving the manifest behind would make a step the user deliberately invalidated
    indistinguishable from one the driver was killed in the middle of — and ``rerun`` would
    re-parse the very outputs it was asked to discard instead of resubmitting them.
    """
    invalidate(step_dir)
    manifest_path(step_dir).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Manifest — the input→output→structure-ID file layout for one step
# ---------------------------------------------------------------------------


def manifest_path(step_dir: Path) -> Path:
    """Return the manifest file path for ``step_dir``."""
    return step_dir / "_cache" / "manifest.json"


def save_manifest(
    inputs: StepInputs,
    step_dir: Path,
    *,
    operation: str | None,
    engine: str,
    fingerprint: str = "",
    criterion_key: str = "",
    search_key: str = "",
    rows: Mapping[str, tuple[str, str]] | None = None,
) -> Path:
    """Persist ``inputs`` plus step metadata to ``manifest.json``; return the path.

    The file layout (which input produced which output for which structure ID)
    is what ``rerun`` / recovery rehydrates via :func:`load_manifest` after a
    restart. Written atomically, like the cache document.

    The stamp — ``fingerprint``, the two resolution halves and ``rows`` — arrives whole
    as :meth:`StepKey.manifest_stamp` from every route that has a key. ``fingerprint``
    is the step key :func:`save` would store, written **before** the jobs go out;
    ``rows`` (``id -> (row_key, parent_digest)``) is the same provenance at row grain;
    ``criterion_key``/``search_key`` are the two halves of the NMS resolution's identity
    (:func:`resolution_keys`), stamped separately because ``resume`` and
    ``rebuild-cache`` each ask a different half — the composed key would answer
    neither. Together they are what lets a later ``resume`` prove, structure by
    structure, that an output on disk is the one this configuration would compute —
    the manifest with row provenance *is* the current-format marker
    (:func:`load_manifest_provenance`); one without is adoptable only by the explicit
    ``rebuild-cache``, never silently. The ``""`` defaults exist for exactly that
    manifest: the hand-written v1 adoption record, which has no key to stamp.
    """
    path = manifest_path(step_dir)
    provenance = rows or {}
    data = {
        "operation": operation,
        "engine": engine,
        "fingerprint": fingerprint,
        "criterion_key": criterion_key,
        "search_key": search_key,
        "files": [
            {
                "input": str(inp),
                "output": str(out),
                "id": sid,
                **(
                    {"row_key": provenance[sid][0], "parent_digest": provenance[sid][1]}
                    if sid in provenance
                    else {}
                ),
            }
            for inp, out, sid in inputs.files
        ],
    }
    write_json(path, data)
    return path


@dataclass(frozen=True)
class ManifestProvenance:
    """The provable half of a manifest: the step stamp and each row's job identity.

    ``rows`` is ``id -> (row_key, parent_digest)`` for exactly the rows that carry
    provenance; empty means the manifest predates the current rules (a pre-release tree,
    or a hand-written v1 adoption manifest) and is *unprovable* — never proven wrong,
    never proven right. Absence of proof routes to the explicit ``rebuild-cache``.
    """

    fingerprint: str
    criterion_key: str
    """The criterion half of the resolution the rows were resolved under — what decides
    whether a passthrough's ``resolved_from`` label may be worn on *resume*, which fans
    out fresh children (an attempt on disk predates its submission)."""
    search_key: str
    """The search half — what decides whether the ``attemptK/`` children themselves may
    be re-read at all: a search retune changes the displaced geometries under unchanged
    child ids, so an attempt from another search key answers a different question.
    ``""`` for a manifest written before this key existed — unprovable, adoptable only
    by the explicit ``rebuild-cache``, the row doctrine."""
    rows: dict[str, tuple[str, str]]


def load_manifest_provenance(step_dir: Path) -> ManifestProvenance:
    """The provenance recorded alongside a step's manifest; all-empty when there is none.

    Read separately from :func:`load_manifest` so callers that want only the file layout
    are untouched, exactly as :func:`load_manifest_fingerprint` is.
    """
    data = read_json(manifest_path(step_dir), None, label="manifest")
    if not isinstance(data, dict):
        return ManifestProvenance(fingerprint="", criterion_key="", search_key="", rows={})
    rows: dict[str, tuple[str, str]] = {}
    for rec in data.get("files") or []:
        if isinstance(rec, dict) and "row_key" in rec:
            rows[str(rec["id"])] = (str(rec["row_key"]), str(rec.get("parent_digest", "")))
    return ManifestProvenance(
        fingerprint=str(data.get("fingerprint", "")),
        criterion_key=str(data.get("criterion_key", "")),
        search_key=str(data.get("search_key", "")),
        rows=rows,
    )


def load_manifest_fingerprint(step_dir: Path) -> str:
    """The fingerprint recorded alongside a step's manifest, or ``""``.

    ``""`` for a manifest written before this key existed, and for a missing manifest —
    either way it can never equal a real fingerprint, so the caller falls back to the
    full re-run. Read separately from :func:`load_manifest` so every existing caller,
    which wants only the file layout, is untouched.
    """
    data = read_json(manifest_path(step_dir), None, label="manifest")
    return str(data.get("fingerprint", "")) if isinstance(data, dict) else ""


def load_manifest(step_dir: Path) -> StepInputs | None:
    """Rehydrate :class:`StepInputs` from the persisted manifest, or ``None``.

    Raises :class:`CacheError` if the JSON is malformed or is missing the
    expected ``files`` field — callers should treat a corrupt manifest as fatal
    rather than silently re-parsing an empty batch.
    """
    path = manifest_path(step_dir)
    data = read_json(path, None, label="manifest")
    if data is None:
        return None
    try:
        files = tuple((Path(rec["input"]), Path(rec["output"]), rec["id"]) for rec in data["files"])
    except (KeyError, TypeError) as e:
        raise CacheError(f"corrupt manifest at {path}: {e}") from e
    return StepInputs(files=files)


# ---------------------------------------------------------------------------
# Failed-job ledger — structures whose job produced no output (for `rerun`)
# ---------------------------------------------------------------------------


def failed_jobs_path(step_dir: Path) -> Path:
    """Return the failed-jobs ledger path for ``step_dir``."""
    return step_dir / "_cache" / "failed_jobs.json"


def save_failure_records(step_dir: Path, failed: Sequence[FailureRecord]) -> None:
    """Persist a step's failure ledger."""
    write_json(failed_jobs_path(step_dir), [record.to_json() for record in failed])


def load_failure_records(step_dir: Path) -> list[FailureRecord]:
    """Return a step's failure ledger as typed records (``[]`` if there is none).

    The ledger is a ``_cache/`` file, so this module owns it end to end — bytes through to
    :class:`~chemrefine.state.FailureRecord`. The recovery paths read these back to decide
    what to re-attempt.

    Valid JSON of the wrong *shape* is held to the same rule as malformed JSON:
    :func:`read_json` guarantees only that the file parsed, not that it is the list of
    records :func:`save_failure_records` writes, and an object or a list of strings must
    reach the user as the :class:`CacheError` every other ``_cache/`` reader raises — not
    as a ``TypeError`` from a subscript three frames down. ``ValueError`` is in the net
    for the same reason: a ``kind`` this version's :class:`~chemrefine.state.FailureKind`
    does not know is the same fact as a corrupt file.
    """
    path = failed_jobs_path(step_dir)
    raw = read_json(path, [], label="failed-jobs ledger")
    try:
        return [FailureRecord.from_json(rec) for rec in raw]
    except (KeyError, TypeError, ValueError) as e:
        raise CacheError(f"corrupt failed-jobs ledger at {path}: {e!r}") from e


def clear_failed_jobs(step_dir: Path) -> None:
    """Delete the failed-jobs ledger. No-op if absent."""
    failed_jobs_path(step_dir).unlink(missing_ok=True)
