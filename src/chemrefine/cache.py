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
from chemrefine.state import StepResults

CACHE_FORMAT_VERSION = "v4.0"

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
) -> StepCache:
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
    )
    pkl_path, json_path = _paths(step_dir)
    _atomic_write(pkl_path, pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))
    sidecar = {
        "cache_format": CACHE_FORMAT_VERSION,
        "chemrefine_version": chemrefine_version,
        "fingerprint": fp,
        "step": step_cfg.step,
        "name": step_cfg.name,
        "engine": step_cfg.engine,
        "operation": step_cfg.operation,
        "structure_ids": [s.id for s in results.structures],
        "energies_hartree": [s.energy_hartree for s in results.structures],
    }
    _atomic_write(json_path, json.dumps(sidecar, indent=2).encode())
    logger.info("saved step %d cache (fingerprint %s)", step_cfg.step, fp)
    return cache


def load(step_dir: Path) -> StepCache | None:
    """Return the cached :class:`StepCache` under ``step_dir``, or ``None``."""
    pkl_path, _ = _paths(step_dir)
    if not pkl_path.is_file():
        return None
    try:
        with pkl_path.open("rb") as fh:
            obj = pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, AttributeError) as e:
        raise CacheError(f"corrupt cache at {pkl_path}: {e}") from e
    if not isinstance(obj, StepCache):
        raise CacheError(f"cache at {pkl_path} is not a StepCache (got {type(obj).__name__})")
    if obj.cache_format != CACHE_FORMAT_VERSION:
        raise CacheError(f"cache at {pkl_path} has format {obj.cache_format}; expected {CACHE_FORMAT_VERSION}")
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
