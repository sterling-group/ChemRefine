"""Per-step result cache for skip-on-resume.

Each step writes its parsed results to ``{step_dir}/_cache/step.pkl``
plus a JSON sidecar (``step.json``) for human inspection. The cache is
keyed by a SHA-1 *fingerprint* covering the step's config (engine,
operation, options, charge, multiplicity, template, NMS flag, sample
config) plus the parent structure IDs that fed into the step. If the
YAML changes the fingerprint changes, so the next run re-executes the
step.

Writes are atomic — the pickle and JSON are written to ``.tmp_*`` files
inside the cache directory and renamed into place, so an interrupted
write never produces a half-baked cache.

This module also owns the per-step **manifest** (``{step_dir}/_cache/
manifest.json``): the input→output→structure-ID file layout that
produced the results. The cache pickle records the *parsed results*;
the manifest records the *file layout*, so ``rerun`` / recovery can
rehydrate which input produced which output after a restart. Both are
the same concern — per-step state under ``_cache/`` — so they live in
one module and share the atomic writer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

from chemrefine.config import StepConfig
from chemrefine.errors import CacheError
from chemrefine.state import StepInputs, StepResults

CACHE_FORMAT_VERSION = "v2.0"
"""On-disk cache schema version, tracking the 2.0.0 release line.

Bump whenever the pickled :class:`StepCache` /
:class:`~chemrefine.state.Structure` / :class:`~chemrefine.state.StepResults`
layout changes in a way that would silently misread an older pickle;
:func:`load` rejects any cache whose ``cache_format`` differs, forcing a clean
rebuild rather than a wrong read."""

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepCache:
    """Persistable snapshot of one step's parsed results.

    The dataclass is the on-disk format: pickle serializes the whole
    object including its :class:`~chemrefine.state.StepResults` payload.
    """

    cache_format: str
    chemrefine_version: str
    fingerprint: str
    step: int
    name: str | None
    engine: str
    operation: str
    parent_ids: tuple[str, ...]
    results: StepResults
    reuse_fingerprint: str = ""
    """Coarser fingerprint (NMS steps only) that's stable across search-param
    tuning but not across the resolution criterion — lets ``resume`` re-attempt
    only the unresolved parents and reuse the round-1 freq. ``""`` for steps
    that don't use it. See :func:`chemrefine.step_nms._nms_reuse_fingerprint`."""


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def fingerprint(step_cfg: StepConfig, parent_ids: tuple[str, ...]) -> str:
    """Return a 16-char SHA-1 over the inputs that determine a step's output.

    Two runs whose YAML produces identical fingerprints are eligible for
    cache reuse.
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
        "nms": step_cfg.nms,
        "sample": (
            step_cfg.sample.model_dump(mode="json") if step_cfg.sample else None
        ),
        "parent_ids": list(parent_ids),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Paths + I/O
# ---------------------------------------------------------------------------


def _paths(step_dir: Path) -> tuple[Path, Path]:
    """Return ``(pickle_path, json_path)`` inside ``step_dir/_cache/``."""
    cache_dir = step_dir / "_cache"
    return cache_dir / "step.pkl", cache_dir / "step.json"


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".part")
    try:
        with open(fd, "wb") as fh:
            fh.write(data)
        Path(tmp).replace(path)
    finally:
        Path(tmp).unlink(missing_ok=True)


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
) -> None:
    """Persist ``results`` for ``step_cfg`` to ``step_dir/_cache/``."""
    fp = fingerprint(step_cfg, parent_ids)
    cache = StepCache(
        cache_format=CACHE_FORMAT_VERSION,
        chemrefine_version=chemrefine_version,
        fingerprint=fp,
        step=step_cfg.step,
        name=step_cfg.name,
        engine=step_cfg.engine,
        operation=step_cfg.operation,
        parent_ids=parent_ids,
        results=results,
        reuse_fingerprint=reuse_fingerprint,
    )
    pkl_path, json_path = _paths(step_dir)
    _atomic_write(pkl_path, pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))
    sidecar = {
        "cache_format": CACHE_FORMAT_VERSION,
        "chemrefine_version": chemrefine_version,
        "fingerprint": fp,
        "reuse_fingerprint": reuse_fingerprint,
        "step": step_cfg.step,
        "name": step_cfg.name,
        "engine": step_cfg.engine,
        "operation": step_cfg.operation,
        "structure_ids": [s.id for s in results.structures],
        "parent_ids": [s.parent_id for s in results.structures],
        "energies_hartree": [s.energy_hartree for s in results.structures],
    }
    _atomic_write(json_path, json.dumps(sidecar, indent=2).encode())
    logger.info("saved step %d cache (fingerprint %s)", step_cfg.step, fp)


def load(step_dir: Path) -> StepCache | None:
    """Return the cached :class:`StepCache` under ``step_dir``, or ``None``."""
    pkl_path, _ = _paths(step_dir)
    if not pkl_path.is_file():
        return None
    try:
        with pkl_path.open("rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, StepCache):
            # Schema probe: a pickle written before a Structure field was added
            # would deserialize "successfully" but blow up later when the
            # pipeline touches the missing attribute. Touching every required
            # field here makes that AttributeError surface inside this try
            # block, so the orchestrator transparently rebuilds the cache.
            for s in obj.results.structures:
                _ = (s.id, s.parent_id, s.atoms, s.energy_hartree, s.forces_ev_per_a)
    except (pickle.UnpicklingError, EOFError, AttributeError) as e:
        raise CacheError(f"stale or corrupt cache at {pkl_path}: {e}") from e
    if not isinstance(obj, StepCache):
        raise CacheError(f"cache at {pkl_path} is not a StepCache (got {type(obj).__name__})")
    if obj.cache_format != CACHE_FORMAT_VERSION:
        raise CacheError(
            f"cache at {pkl_path} has format {obj.cache_format}; "
            f"expected {CACHE_FORMAT_VERSION}"
        )
    return obj


def is_valid(
    *,
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    step_dir: Path,
) -> bool:
    """True iff a cache exists and its fingerprint matches the current step config."""
    try:
        cached = load(step_dir)
    except CacheError:
        return False
    if cached is None:
        return False
    return cached.fingerprint == fingerprint(step_cfg, parent_ids)


def invalidate(step_dir: Path) -> None:
    """Delete the cache for ``step_dir``. No-op if nothing is cached."""
    pkl_path, json_path = _paths(step_dir)
    pkl_path.unlink(missing_ok=True)
    json_path.unlink(missing_ok=True)


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
    operation: str,
    engine: str,
) -> Path:
    """Persist ``inputs`` plus step metadata to ``manifest.json``; return the path.

    The file layout (which input produced which output for which structure ID)
    is what ``rerun`` / recovery rehydrates via :func:`load_manifest` after a
    restart. Written atomically, like the cache pickle.
    """
    path = manifest_path(step_dir)
    data = {
        "operation": operation,
        "engine": engine,
        "files": [
            {"input": str(inp), "output": str(out), "id": sid}
            for inp, out, sid in inputs.files
        ],
    }
    _atomic_write(path, json.dumps(data, indent=2).encode())
    return path


def load_manifest(step_dir: Path) -> StepInputs | None:
    """Rehydrate :class:`StepInputs` from the persisted manifest, or ``None``.

    Raises :class:`CacheError` if the JSON is malformed or is missing the
    expected ``files`` field — callers should treat a corrupt manifest as fatal
    rather than silently re-parsing an empty batch.
    """
    path = manifest_path(step_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        files = tuple(
            (Path(rec["input"]), Path(rec["output"]), rec["id"])
            for rec in data["files"]
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise CacheError(f"corrupt manifest at {path}: {e}") from e
    return StepInputs(files=files)


# ---------------------------------------------------------------------------
# Failed-job ledger — structures whose job produced no output (for `rerun`)
# ---------------------------------------------------------------------------


def failed_jobs_path(step_dir: Path) -> Path:
    """Return the failed-jobs ledger path for ``step_dir``."""
    return step_dir / "_cache" / "failed_jobs.json"


def save_failed_jobs(step_dir: Path, failed: list[dict]) -> None:
    """Persist the list of failed-job records (``{"structure_id", "reason"}``)."""
    _atomic_write(failed_jobs_path(step_dir), json.dumps(failed, indent=2).encode())


def load_failed_jobs(step_dir: Path) -> list[dict]:
    """Return the failed-job records for ``step_dir`` (``[]`` if none)."""
    path = failed_jobs_path(step_dir)
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CacheError(f"corrupt failed-jobs ledger at {path}: {e}") from e


def clear_failed_jobs(step_dir: Path) -> None:
    """Delete the failed-jobs ledger. No-op if absent."""
    failed_jobs_path(step_dir).unlink(missing_ok=True)
