"""Capture a finished e2e run as a compressed replay fixture.

Usage::

    python capture.py <case-dir> [--repo <repo-root>]

Trims the case's ``outputs/`` to what the parsers actually read (``.out``,
ensemble sidecars, script-engine ``.json``, the ``_cache`` documents),
rewrites the absolute output prefix in the cache documents to the
``@OUTPUT_DIR@`` token, and packs ``input.yaml`` + ``templates/`` + the seed
+ the trimmed outputs into ``tests/data/e2e/<case>.tar.xz``.

The replay tests (``tests/replay.py``) reverse the token rewrite at
extraction time, so the fixture is relocatable to any tmp path.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml

MAX_ARCHIVE_BYTES = 1_000_000
OUTPUT_DIR_TOKEN = "@OUTPUT_DIR@"

KEEP_PATTERNS = (
    "*.out",
    "*.finalensemble.xyz",
    "*.docker.struc1.allopt.xyz",
    "*.docker.struc1.all.optimized.xyz",  # ORCA 6.1.1's docker sidecar name
    "*.solventbuild.xyz",  # also matches 6.1.1's *.solvator.solventbuild.xyz
    "*.json",
)


def _kept(path: Path) -> bool:
    if path.parent.name == "_cache":
        return path.suffix == ".json"
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in KEEP_PATTERNS)


def capture(case_dir: Path, repo: Path) -> Path:
    """Build ``tests/data/e2e/<case>.tar.xz`` from a finished run in ``case_dir``."""
    case_dir = case_dir.resolve()
    config = yaml.safe_load((case_dir / "input.yaml").read_text())
    outputs = (case_dir / config.get("output_dir", "outputs")).resolve()
    if not outputs.is_dir():
        raise SystemExit(f"no outputs to capture in {case_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        shutil.copy2(case_dir / "input.yaml", staging / "input.yaml")
        shutil.copytree(case_dir / "templates", staging / "templates")
        seed = config.get("input")
        if seed is not None and not Path(seed).is_absolute():
            seed_path = (case_dir / seed).resolve()
            if seed_path.is_relative_to(case_dir) and not seed_path.is_relative_to(
                case_dir / "templates"
            ):
                shutil.copy2(seed_path, staging / seed_path.name)

        captured = staging / "captured_outputs"
        kept = 0
        for path in sorted(outputs.rglob("*")):
            if not path.is_file() or not _kept(path):
                continue
            dest = captured / path.relative_to(outputs)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if path.parent.name == "_cache":
                # Cache documents carry absolute paths (the manifest's
                # input/output entries); tokenize them for relocation.
                dest.write_text(path.read_text().replace(str(outputs), OUTPUT_DIR_TOKEN))
            else:
                shutil.copy2(path, dest)
            kept += 1
        if not kept:
            raise SystemExit(f"nothing matched the keep patterns under {outputs}")

        archive_dir = repo / "tests" / "data" / "e2e"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / f"{case_dir.name}.tar.xz"
        with tarfile.open(archive, "w:xz") as tar:
            for path in sorted(staging.rglob("*")):
                tar.add(path, arcname=str(path.relative_to(staging)))

    size = archive.stat().st_size
    print(f"{archive}: {kept} files, {size / 1024:.0f} KiB")
    if size > MAX_ARCHIVE_BYTES:
        archive.unlink()
        raise SystemExit(f"archive would exceed {MAX_ARCHIVE_BYTES} bytes — trim the case")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent.parent)
    args = parser.parse_args()
    capture(args.case_dir, args.repo)


if __name__ == "__main__":
    sys.exit(main())
