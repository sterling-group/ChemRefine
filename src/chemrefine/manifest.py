"""Per-step manifest persistence.

The manifest is a small JSON file (``{step_dir}/_cache/manifest.json``)
recording which input file produced which output file for which
structure ID. It mirrors the in-memory
:class:`~chemrefine.state.StepInputs` so that ``rebuild-cache`` and
``rerun`` actions can rehydrate that mapping after a process restart.

The cache pickle (:mod:`chemrefine.cache`) records the *parsed results*;
the manifest records the *file layout* that produced them.
"""

from __future__ import annotations

import json
from pathlib import Path

from chemrefine.state import StepInputs


def manifest_path(step_dir: Path) -> Path:
    """Return the manifest file path for ``step_dir``."""
    return step_dir / "_cache" / "manifest.json"


def save(
    inputs: StepInputs,
    step_dir: Path,
    *,
    operation: str,
    engine: str,
) -> Path:
    """Persist ``inputs`` plus step metadata to JSON; return the written path."""
    path = manifest_path(step_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "operation": operation,
        "engine": engine,
        "files": [
            {"input": str(inp), "output": str(out), "id": sid}
            for inp, out, sid in inputs.files
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load(step_dir: Path) -> StepInputs | None:
    """Rehydrate :class:`StepInputs` from the persisted manifest, or ``None``."""
    path = manifest_path(step_dir)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    files = tuple(
        (Path(rec["input"]), Path(rec["output"]), rec["id"]) for rec in data["files"]
    )
    return StepInputs(files=files)
