# chemrefine/cache_utils.py
from __future__ import annotations

import os
import json
import pickle
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

CACHE_VERSION = "1.0"


@dataclass
class StepCache:
    version: str
    step: int
    operation: str
    engine: str
    fingerprint: str
    parent_ids: Optional[list]
    ids: list
    n_outputs: int
    by_parent: Optional[dict]
    coords: Any
    energies: Optional[Any]
    forces: Optional[Any]
    extras: Optional[dict] = None


def _cache_dir(step_dir: str) -> str:
    d = os.path.join(step_dir, "_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_paths(step_dir: str) -> Tuple[str, str]:
    step_name = Path(step_dir).name  # "stepN"
    base = os.path.join(_cache_dir(step_dir), step_name)
    return base + ".pkl", base + ".json"


def compute_fingerprint(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def atomic_write_bytes(path: str, data: bytes) -> None:
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def save_step_cache(step_dir: str, cache: StepCache) -> None:
    pkl_path, json_path = _cache_paths(step_dir)
    data = pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_write_bytes(pkl_path, data)
    sidecar = {
        "version": cache.version,
        "step": cache.step,
        "operation": cache.operation,
        "engine": cache.engine,
        "fingerprint": cache.fingerprint,
        "n_outputs": cache.n_outputs,
        "has_by_parent": bool(cache.by_parent),
        "ids_head": list(cache.ids) if cache.ids else [],
    }
    atomic_write_bytes(json_path, json.dumps(sidecar, indent=2).encode())


def load_step_cache(step_dir: str) -> Optional[StepCache]:
    pkl_path, _ = _cache_paths(step_dir)
    if not os.path.isfile(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return StepCache(
            version=obj.get("version", CACHE_VERSION),
            step=obj.get("step"),
            operation=obj.get("operation"),
            engine=obj.get("engine"),
            fingerprint=obj.get("fingerprint"),
            parent_ids=obj.get("parent_ids"),
            ids=obj.get("ids", []),
            n_outputs=obj.get("n_outputs", len(obj.get("ids", []) or [])),
            by_parent=obj.get("by_parent"),
            coords=obj.get("coords"),
            energies=obj.get("energies"),
            forces=obj.get("forces"),
            extras=obj.get("extras"),
        )
    return obj


def build_step_fingerprint(
    step_cfg: dict, parent_ids: list | None, sample_cfg: dict | None, step_number: int
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "step": step_number,
        "operation": step_cfg.get("operation"),
        "engine": step_cfg.get("engine"),
        "charge": step_cfg.get("charge"),
        "multiplicity": step_cfg.get("multiplicity"),
        "mlff": step_cfg.get("mlff"),
        "mlip": step_cfg.get("mlip"),
        "parent_ids": list(parent_ids) if parent_ids is not None else None,
        "sample": sample_cfg or None,
    }
    return compute_fingerprint(payload)
