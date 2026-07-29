"""Per-step result cache for skip-on-resume.

Each step writes its parsed results to ``{step_dir}/_cache/`` as **two
files**: ``step.json`` holds the step metadata and every structure's
scalar fields (id, lineage, energies, status flags, symbols), and
``arrays.npz`` holds the bulk — the coordinates and forces.

Neither can execute code on load, which is why this is not pickle: JSON
cannot by construction, and ``.npz`` cannot because :func:`_read_arrays`
passes ``allow_pickle=False`` and numpy *raises* rather than running an
object array's reduce. Coordinates round-trip byte-identically either
way, so :func:`parents_digest` is stable across save → load.

The split is what makes the cache scale. Coordinates are 93% of a
record, and as decimal text each float64 costs 18 bytes on disk, a
``strtod`` call to parse, and 32 bytes live; in a ``.npy`` member it
costs 8 bytes, a memcpy, and 8 bytes. Measured over 10,000 structures of
10-120 atoms: **69.1 MB / 1.73 s load / 268 MB peak** as one JSON
document against **31.3 MB / 0.36 s / 75 MB** split. Structures in a
step need not share an atom count, so the arrays are concatenated with
an offsets index rather than stacked — see :func:`_split_arrays`.

``step.json`` is written without indentation (see :func:`_write_json`) —
read it with ``jq`` or :func:`json.load`, not by eye. The per-structure
``.result.json`` records beside each output stay indented and keep their
coordinates inline; those are the ones a person opens, and they are a
few KB each.
The cache is keyed by a SHA-1 *fingerprint* covering the step's config
(engine, operation, options, charge, multiplicity, template, NMS flag)
plus the parent structures that fed into the step — their IDs **and**
their content (:func:`parents_digest`: symbols, coordinates, energy).
If the YAML changes, or the seed file / any upstream result changes,
the fingerprint changes and the next run re-executes the step. The
``sample:`` filter is deliberately **excluded**: the cache stores the
*pre-filter* results and filtering re-runs on every load, so tuning a
filter must refilter the cached results, not redo the calculations
(downstream steps still invalidate through the changed survivor set).

Writes are atomic — the JSON is written to a ``.tmp_*`` file inside the
cache directory and renamed into place, so an interrupted write never
produces a half-baked cache.

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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from chemrefine import ids
from chemrefine.config import StepConfig
from chemrefine.errors import CacheError
from chemrefine.state import StepContext, StepInputs, StepResults, Structure

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
    reuse_fingerprint: str = ""
    """Coarser fingerprint (NMS steps only) that's stable across search-param
    tuning but not across the resolution criterion — lets ``resume`` re-attempt
    only the unresolved parents and reuse the round-1 freq. ``""`` for steps
    that don't use it. See :func:`reuse_fingerprint`."""


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def parents_digest(structures: Sequence[Structure]) -> str:
    """Return a 16-char SHA-1 over the parent structures' *content*.

    Covers each parent's ID, chemical symbols, Cartesian coordinates (exact
    float64 bytes — identical inputs parse to identical floats), and energy.
    Folding this into :func:`fingerprint` is what makes the cache sensitive to
    the structures themselves, not just their positional IDs: editing the seed
    ``input.xyz`` (same path, same count → same IDs) or any upstream change to
    a parent's geometry/energy must invalidate the step, or ``resume`` would
    silently reuse results computed from the old geometry.
    """
    h = hashlib.sha1(usedforsecurity=False)  # a content fingerprint, not a digest
    for s in structures:
        h.update(s.id.encode())
        h.update("".join(s.atoms.get_chemical_symbols()).encode())
        h.update(np.asarray(s.atoms.get_positions(), dtype=np.float64).tobytes())
        h.update(repr(s.energy_hartree).encode())
    return h.hexdigest()[:16]


def fingerprint(
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    *,
    parents_digest: str = "",
    template_digest: str = "",
) -> str:
    """Return a 16-char SHA-1 over the inputs that determine a step's output.

    Two runs whose YAML produces identical fingerprints are eligible for
    cache reuse. ``parents_digest`` (see :func:`parents_digest`) ties the
    fingerprint to the parent structures' content so a changed seed file or
    changed upstream result invalidates the step even when the IDs match.
    ``template_digest`` (see :meth:`chemrefine.engines.api.CalculationEngine.input_digest`)
    ties it to the *contents* of the resolved template — editing the template
    in place (which now also drives ORCA's run-type detection when ``operation``
    is omitted) re-runs the step, where the template basename alone could not.
    ``sample:`` is excluded on purpose — the cached results are pre-filter
    and filtering re-runs on every load, so a filter-only edit is a cache
    hit, not a re-run.
    """
    payload = {
        "format": CACHE_FORMAT_VERSION,
        "step": step_cfg.step,
        "engine": step_cfg.engine,
        "operation": step_cfg.operation,
        "options": step_cfg.options,
        "charge": step_cfg.charge,
        "multiplicity": step_cfg.multiplicity,
        "template": step_cfg.template,
        "template_digest": template_digest,
        "nms": step_cfg.nms,
        "parent_ids": list(parent_ids),
        "parents_digest": parents_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded, usedforsecurity=False).hexdigest()[:16]


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


def _split_arrays(records: list[dict[str, Any]]) -> dict[str, NDArray[Any]]:
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

    Only symbols + positions of the ``Atoms`` are stored — that is all the
    pipeline ever reads back (and all that :func:`parents_digest` hashes).
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
        # Imaginary modes round-trip (small, useful metadata); JSON keys must be strings.
        # ``normal_modes`` is deliberately NOT persisted — it's a transient displacement tensor
        # only used during an active NMS run (which always re-parses), so a cache-reloaded
        # structure carries ``None`` (never read).
        "imaginary_freqs": (
            None if s.imaginary_freqs is None else {str(k): v for k, v in s.imaginary_freqs.items()}
        ),
    }


def structure_from_record(d: dict[str, Any]) -> Structure:
    """Rebuild a :class:`Structure` from its canonical record (inverse of the above)."""
    forces = d["forces_ev_per_a"]
    imaginary = d.get("imaginary_freqs")
    return Structure(
        id=d["id"],
        atoms=Atoms(symbols=d["symbols"], positions=d["positions"]),
        parent_id=d["parent_id"],
        energy_hartree=d["energy_hartree"],
        forces_ev_per_a=None if forces is None else np.asarray(forces, dtype=np.float64),
        converged=d["converged"],
        terminated_normally=d["terminated_normally"],
        # Thermochemistry + imaginary modes are additive — older caches lack these keys.
        gibbs_hartree=d.get("gibbs_hartree"),
        enthalpy_hartree=d.get("enthalpy_hartree"),
        energy_zpe_hartree=d.get("energy_zpe_hartree"),
        imaginary_freqs=None if imaginary is None else {int(k): v for k, v in imaginary.items()},
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
        _write_json(ids.result_record_path(job_dir, step, s.id), record)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file + fsync + rename.

    The ``fsync`` is what makes the atomicity survive more than a process death: a
    rename is ordered against the data only once the data is on the device, so without
    it a machine crash (not a kill) could leave the renamed file truncated or empty —
    exactly the half-baked cache the temp-file dance exists to prevent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(path)
    finally:
        Path(tmp).unlink(missing_ok=True)


#: ``json.dumps`` separators with no padding: ``{"a":1,"b":2}`` rather than ``{"a": 1, "b": 2}``.
#: Only worth using where a document is machine-read and large — see :func:`_write_json`.
_COMPACT_SEPARATORS = (",", ":")


def _write_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
    """Serialize ``data`` to JSON and write it atomically to ``path``.

    Indented by default: the manifest, the failed-jobs ledger and the per-structure
    ``.result.json`` records are all things a person opens, and all small enough that
    readability is the only property that matters.

    ``indent=None`` switches to compact separators, for the step document alone. At the
    10,000 structures of 30 atoms that :mod:`tests.test_perf_cache` measures, that file is
    **73.3 MB indented and 38.7 MB compact** — the indentation is not a rounding error, it
    is 47% of the file, because a coordinate record is thousands of short numeric values and
    each one carries its own newline and run of spaces. Nothing reads it by eye at that size,
    and nothing reads it by line either: the loader parses the whole document, so the layout
    is invisible to every consumer.
    """
    separators = None if indent is not None else _COMPACT_SEPARATORS
    _atomic_write(path, json.dumps(data, indent=indent, separators=separators).encode())


def _npz_bytes(arrays: dict[str, NDArray[Any]]) -> bytes:
    """Serialize ``arrays`` to an uncompressed ``.npz`` in memory.

    Uncompressed on purpose. A ``.npz`` is an ordinary ZIP of ``.npy`` members, and a ``.npy``
    is a short ASCII header plus the array's raw buffer — byte-identical to ``tobytes()``,
    which is what keeps coordinates exact through save → load and :func:`parents_digest`
    stable. Deflating float64 coordinates buys about 5% and costs roughly twenty times the
    encode; the point of this format is that writing it is a memcpy.

    Going through bytes rather than writing the file directly is what lets it reuse
    :func:`_atomic_write`, so a killed run never leaves a half-written sidecar.
    """
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _read_arrays(path: Path) -> Any:
    """Load the coordinate sidecar, or raise :class:`CacheError` if it is absent or corrupt.

    ``allow_pickle=False`` is spelled out because it is the whole reason this is not a
    pickle: numpy *enforces* it, raising rather than executing when a file smuggles in an
    object array. The cache's promise that loading it can never run code from the file
    therefore survives the move off pure JSON, as a check rather than a convention.
    """
    if not path.is_file():
        raise CacheError(f"step cache at {path.parent} has no {path.name}; rebuild the step")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}
    except (OSError, ValueError) as e:
        raise CacheError(f"corrupt coordinate sidecar at {path}: {e}") from e


def _read_json(path: Path, default: Any, *, label: str) -> Any:
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


def save(
    *,
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    results: StepResults,
    step_dir: Path,
    chemrefine_version: str,
    reuse_fingerprint: str = "",
    parents_digest: str = "",
    template_digest: str = "",
) -> None:
    """Persist ``results`` for ``step_cfg`` to ``step_dir/_cache/``.

    ``parents_digest`` (the parent structures' content digest) and
    ``template_digest`` (the resolved template's content digest) are folded
    into the stored fingerprint; pass the same values to :func:`load_if_valid`.
    """
    fp = fingerprint(
        step_cfg, parent_ids, parents_digest=parents_digest, template_digest=template_digest
    )
    records = [structure_record(s) for s in results.structures]
    document = {
        "cache_format": CACHE_FORMAT_VERSION,
        "chemrefine_version": chemrefine_version,
        "fingerprint": fp,
        "reuse_fingerprint": reuse_fingerprint,
        "step": step_cfg.step,
        "name": step_cfg.name,
        "engine": step_cfg.engine,
        "operation": step_cfg.operation,
        "parent_ids": list(parent_ids),
        "structures": records,
    }
    # Sidecar first: a crash between the two writes then leaves an orphan `.npz` and no
    # `step.json`, which reads as a plain cache miss. The other order would leave a document
    # whose arrays are missing — `load` fails closed on that, but a miss is cheaper than an
    # error, and this way the pair is effectively atomic without a second mechanism.
    _atomic_write(_arrays_path(step_dir), _npz_bytes(_split_arrays(records)))
    _write_json(_cache_path(step_dir), document, indent=None)
    logger.info("saved step %d cache (fingerprint %s)", step_cfg.step, fp)


#: ``step.options`` keys that tune an NMS *search* without changing what counts as
#: resolved. Stripped from :func:`reuse_fingerprint` so raising ``displacement_value``
#: reuses round-1 instead of re-running it; the resolution criterion (``target`` /
#: ``ts_mode_index``) stays in, because changing that changes the answer.
_NMS_SEARCH_KEYS = frozenset({"displacement_value", "num_random_displacements", "seed"})


def reuse_fingerprint(
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    *,
    parents_digest: str = "",
    template_digest: str = "",
) -> str:
    """A coarser :func:`fingerprint` that survives NMS search-param tuning.

    Identical to :func:`fingerprint` — same content keys — but with the NMS search
    parameters stripped from ``options``, so bumping one leaves it unchanged and
    ``resume`` can reuse the round-1 frequencies plus the already-resolved children and
    re-attempt only the unresolved parents. Returns ``""`` for non-NMS steps, which
    never take that path.

    Lives here rather than in :mod:`chemrefine.nms` because it is a cache-validity key,
    and this module owns those.
    """
    if not step_cfg.nms:
        return ""
    trimmed = {k: v for k, v in (step_cfg.options or {}).items() if k not in _NMS_SEARCH_KEYS}
    return fingerprint(
        step_cfg.model_copy(update={"options": trimmed}),
        parent_ids,
        parents_digest=parents_digest,
        template_digest=template_digest,
    )


def save_step_results(
    *,
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    results: StepResults,
    ctx: StepContext,
    template_digest: str,
    chemrefine_version: str,
) -> None:
    """Persist a step's results, deriving every digest and fingerprint from ``ctx``.

    The one place a completed step is written. Five call sites used to spell out the
    same tail — hash the parents, take the template digest, compute the plain and the
    reuse fingerprint, call :func:`save` — and one of them had already drifted, omitting
    the reuse fingerprint. That is the failure mode this exists to prevent: a cache
    written with an inconsistent key is not a crash, it is a silent re-run or a silent
    reuse much later.
    """
    digest = parents_digest(ctx.prev_state.structures)
    save(
        step_cfg=step_cfg,
        parent_ids=parent_ids,
        results=results,
        step_dir=ctx.step_dir,
        chemrefine_version=chemrefine_version,
        reuse_fingerprint=reuse_fingerprint(
            step_cfg, parent_ids, parents_digest=digest, template_digest=template_digest
        ),
        parents_digest=digest,
        template_digest=template_digest,
    )


def load(step_dir: Path) -> StepCache | None:
    """Return the cached :class:`StepCache` under ``step_dir``, or ``None``.

    Raises :class:`CacheError` when the document exists but is malformed —
    bad JSON, missing fields, or a ``cache_format`` this version doesn't
    read. The summary-only ``step.json`` sidecar that pickle-era versions
    wrote next to ``step.pkl`` lands here too (it has no ``structures``),
    forcing a clean rebuild rather than a wrong read.
    """
    path = _cache_path(step_dir)
    data = _read_json(path, None, label="step cache")
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
        _join_arrays(records, _read_arrays(_arrays_path(step_dir)))
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
            reuse_fingerprint=data.get("reuse_fingerprint", ""),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise CacheError(f"stale or corrupt cache at {path}: {e!r}") from e


def load_if_valid(
    *,
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    step_dir: Path,
    parents_digest: str = "",
    template_digest: str = "",
) -> StepCache | None:
    """Return the cached :class:`StepCache` iff its fingerprint matches; else ``None``.

    A single ``load`` + fingerprint compare, so a caller that needs the cached
    results on a hit (e.g. :func:`chemrefine.step._cached_outcome`) reads
    ``step.json`` **once** instead of validating and then re-loading — and there
    is no window in which the cache could vanish between the two reads. A corrupt
    or absent cache returns ``None`` (treated as "re-run"), never raises.
    """
    try:
        cached = load(step_dir)
    except CacheError:
        return None
    if cached is None:
        return None
    current = fingerprint(
        step_cfg, parent_ids, parents_digest=parents_digest, template_digest=template_digest
    )
    if cached.fingerprint != current:
        return None
    return cached


def invalidate(step_dir: Path) -> None:
    """Delete the cached *results* for ``step_dir``. No-op if nothing is cached.

    Deliberately narrow: the manifest survives, so ``rebuild-cache`` can still re-parse the
    outputs on disk after the results document is gone. Callers that mean "redo this step
    from scratch" want :func:`discard_step` instead.

    Also removes the ``step.pkl`` a pre-JSON version may have left behind, so re-running
    over an old output tree leaves no stale binary around.
    """
    _cache_path(step_dir).unlink(missing_ok=True)
    (step_dir / "_cache" / "step.pkl").unlink(missing_ok=True)


def discard_step(step_dir: Path) -> None:
    """Forget that ``step_dir`` ever ran — results **and** manifest.

    The difference from :func:`invalidate` is the manifest, and it is load-bearing. Together
    the two files say "this step already ran with this configuration"; the manifest alone
    says "these outputs on disk belong to this configuration", which is what
    :func:`chemrefine.step._partial_step_outcome` reads to continue an interrupted step.

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
) -> Path:
    """Persist ``inputs`` plus step metadata to ``manifest.json``; return the path.

    The file layout (which input produced which output for which structure ID)
    is what ``rerun`` / recovery rehydrates via :func:`load_manifest` after a
    restart. Written atomically, like the cache document.

    ``fingerprint`` is the same key :func:`save` would store, written **before** the
    jobs go out. It is what lets a ``resume`` after an interrupted step prove that the
    outputs sitting on disk were produced for *this* step config and *these* parents —
    without it there is no way to tell them from a stale leftover, so the whole step had
    to be re-run. See :func:`chemrefine.step._partial_step_outcome`.
    """
    path = manifest_path(step_dir)
    data = {
        "operation": operation,
        "engine": engine,
        "fingerprint": fingerprint,
        "files": [
            {"input": str(inp), "output": str(out), "id": sid} for inp, out, sid in inputs.files
        ],
    }
    _write_json(path, data)
    return path


def load_manifest_fingerprint(step_dir: Path) -> str:
    """The fingerprint recorded alongside a step's manifest, or ``""``.

    ``""`` for a manifest written before this key existed, and for a missing manifest —
    either way it can never equal a real fingerprint, so the caller falls back to the
    full re-run. Read separately from :func:`load_manifest` so every existing caller,
    which wants only the file layout, is untouched.
    """
    data = _read_json(manifest_path(step_dir), None, label="manifest")
    return str(data.get("fingerprint", "")) if isinstance(data, dict) else ""


def load_manifest(step_dir: Path) -> StepInputs | None:
    """Rehydrate :class:`StepInputs` from the persisted manifest, or ``None``.

    Raises :class:`CacheError` if the JSON is malformed or is missing the
    expected ``files`` field — callers should treat a corrupt manifest as fatal
    rather than silently re-parsing an empty batch.
    """
    path = manifest_path(step_dir)
    data = _read_json(path, None, label="manifest")
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


def save_failed_jobs(step_dir: Path, failed: list[dict[str, str]]) -> None:
    """Persist the failed-job ledger (serialized
    :class:`chemrefine.step_failures.FailureRecord` entries)."""
    _write_json(failed_jobs_path(step_dir), failed)


def load_failed_jobs(step_dir: Path) -> list[dict[str, str]]:
    """Return the raw failed-job ledger entries for ``step_dir`` (``[]`` if none).

    Deliberately untyped at this layer: this module owns bytes-to-JSON, and the
    domain meaning of an entry belongs to
    :func:`chemrefine.step_failures.load_failure_records`, which reads it back into
    :class:`~chemrefine.step_failures.FailureRecord`. Typing it here would mean
    importing the policy module that already imports this one.
    """
    return cast(
        list[dict[str, str]],
        _read_json(failed_jobs_path(step_dir), [], label="failed-jobs ledger"),
    )


def clear_failed_jobs(step_dir: Path) -> None:
    """Delete the failed-jobs ledger. No-op if absent."""
    failed_jobs_path(step_dir).unlink(missing_ok=True)
