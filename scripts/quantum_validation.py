"""Capture a validation command and reproducible environment evidence without a shell."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def snapshot() -> dict:
    """Record source, installed distributions, hardware and numerical thread settings."""
    commands = {
        "revision": ["git", "rev-parse", "HEAD"],
        "worktree": ["git", "status", "--short"],
        "cpu": ["lscpu", "--json"],
        "gpu": [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv",
        ],
    }
    result = {
        "schema_version": 1,
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
        "python": sys.version,
        "interpreter": sys.executable,
        "platform": platform.platform(),
        "packages": dict(
            sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
        ),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
                "CHEMREFINE_REQUIRE_NODE",
                "PYTHONPATH",
                "QISKIT_NUM_PROCS",
            )
        },
    }
    root = Path(__file__).resolve().parents[1]
    specification = importlib.util.find_spec("chemrefine")
    package = (
        Path(specification.origin).parent
        if specification and specification.origin
        else (root / "src/chemrefine")
    )
    result["source_package"] = str(package)
    result["source_sha256"] = {
        str(prefix / path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder, prefix in (
            (package / "engines/qiskit", Path("src/chemrefine/engines/qiskit")),
            (root / "scripts", Path("scripts")),
        )
        for path in sorted(folder.rglob("*.py"))
    }
    conda_metadata = Path(sys.prefix) / "conda-meta"
    if conda_metadata.exists():
        result["conda_packages"] = [
            {key: record.get(key) for key in ("name", "version", "build", "channel", "sha256")}
            for path in sorted(conda_metadata.glob("*.json"))
            if (record := json.loads(path.read_text()))
        ]
    for key, command in commands.items():
        try:
            response = subprocess.run(  # noqa: S603 - fixed inspection commands
                command,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            result[key] = response.stdout.strip() or response.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[key] = str(exc)
    return result


def main() -> int:
    """Execute the explicitly supplied command and retain its status and full output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.output / args.label
    record = snapshot()
    record["command"] = command
    prefix.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    started = time.perf_counter()
    with prefix.with_suffix(".txt").open("w") as log:
        process = subprocess.run(  # noqa: S603 - user-supplied argv, no shell
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record.update(returncode=process.returncode, wall_seconds=time.perf_counter() - started)
    prefix.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "label": args.label,
                "returncode": process.returncode,
                "wall_seconds": record["wall_seconds"],
                "log": str(prefix.with_suffix(".txt")),
            }
        )
    )
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
