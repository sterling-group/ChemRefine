"""Per-step result cache for skip-on-resume.

Each step writes its parsed results to ``{step_dir}/_cache/step.json``
— one human-inspectable JSON document holding the step metadata and
every structure (symbols, coordinates, energy, forces, status flags).
Plain JSON rather than pickle on purpose: loading it can never execute
code from the file, and Python's float round-tripping keeps coordinates
byte-identical so :func:`parents_digest` is stable across save → load.
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

from chemrefine import ids
from chemrefine.config import StepConfig
from chemrefine.errors import CacheError
from chemrefine.state import StepInputs, StepResults, Structure

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
    that don't use it. See :func:`chemrefine.nms.nms_reuse_fingerprint`."""


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
    h = hashlib.sha1()
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
    return hashlib.sha1(encoded).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Paths + I/O
# ---------------------------------------------------------------------------


def _cache_path(step_dir: Path) -> Path:
    """Return the ``step.json`` cache document path inside ``step_dir/_cache/``."""
    return step_dir / "_cache" / "step.json"


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
    - ``converged`` / ``terminated`` — run status (``None`` = not reported)
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
        "terminated": s.terminated,
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
        terminated=d["terminated"],
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
    """Write ``data`` to ``path`` via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        Path(tmp).replace(path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _write_json(path: Path, data: Any) -> None:
    """Serialize ``data`` to indented JSON and write it atomically to ``path``."""
    _atomic_write(path, json.dumps(data, indent=2).encode())


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
    into the stored fingerprint; pass the same values to :func:`is_valid`.
    """
    fp = fingerprint(
        step_cfg, parent_ids, parents_digest=parents_digest, template_digest=template_digest
    )
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
        "structures": [structure_record(s) for s in results.structures],
    }
    _write_json(_cache_path(step_dir), document)
    logger.info("saved step %d cache (fingerprint %s)", step_cfg.step, fp)


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
        return StepCache(
            cache_format=data["cache_format"],
            chemrefine_version=data["chemrefine_version"],
            fingerprint=data["fingerprint"],
            step=data["step"],
            name=data["name"],
            engine=data["engine"],
            operation=data["operation"],
            parent_ids=tuple(data["parent_ids"]),
            results=StepResults(
                structures=tuple(structure_from_record(d) for d in data["structures"])
            ),
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


def is_valid(
    *,
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    step_dir: Path,
    parents_digest: str = "",
    template_digest: str = "",
) -> bool:
    """True iff a cache exists and its fingerprint matches the current step config."""
    return (
        load_if_valid(
            step_cfg=step_cfg,
            parent_ids=parent_ids,
            step_dir=step_dir,
            parents_digest=parents_digest,
            template_digest=template_digest,
        )
        is not None
    )


def invalidate(step_dir: Path) -> None:
    """Delete the cache for ``step_dir``. No-op if nothing is cached.

    Also removes the ``step.pkl`` a pre-JSON version may have left behind,
    so re-running over an old output tree leaves no stale binary around.
    """
    _cache_path(step_dir).unlink(missing_ok=True)
    (step_dir / "_cache" / "step.pkl").unlink(missing_ok=True)


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
) -> Path:
    """Persist ``inputs`` plus step metadata to ``manifest.json``; return the path.

    The file layout (which input produced which output for which structure ID)
    is what ``rerun`` / recovery rehydrates via :func:`load_manifest` after a
    restart. Written atomically, like the cache document.
    """
    path = manifest_path(step_dir)
    data = {
        "operation": operation,
        "engine": engine,
        "files": [
            {"input": str(inp), "output": str(out), "id": sid} for inp, out, sid in inputs.files
        ],
    }
    _write_json(path, data)
    return path


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


def save_failed_jobs(step_dir: Path, failed: list[dict[str, Any]]) -> None:
    """Persist the list of failed-job records (``{"structure_id", "reason"}``)."""
    _write_json(failed_jobs_path(step_dir), failed)


def load_failed_jobs(step_dir: Path) -> list[dict[str, Any]]:
    """Return the failed-job records for ``step_dir`` (``[]`` if none)."""
    return cast(
        list[dict[str, Any]],
        _read_json(failed_jobs_path(step_dir), [], label="failed-jobs ledger"),
    )


def clear_failed_jobs(step_dir: Path) -> None:
    """Delete the failed-jobs ledger. No-op if absent."""
    failed_jobs_path(step_dir).unlink(missing_ok=True)
