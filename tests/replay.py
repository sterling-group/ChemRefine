"""Replay captured real runs through the live pipeline.

The fixtures under ``tests/data/e2e/`` are trimmed, compressed outputs of
real ORCA / MLIP runs (built by ``scripts/e2e_capture/capture.py``). Tests
extract one into a tmp dir and drive the real pipeline against it in two
modes:

* **fresh replay** — monkeypatch :func:`chemrefine.engines._execution.run_batch`
  with :func:`replay_run_batch`, so the shipped engines run their real
  ``prepare`` / ``parse`` / NMS / filter / cache code and only the compute
  step is satisfied from the archive;
* **relocation** — copy ``captured_outputs`` to the config's output dir and
  run with :func:`forbid_run_batch`, asserting the cache fingerprints (and
  the ``rebuild-cache`` path) survive a move to a different machine/path.

``@OUTPUT_DIR@`` tokens in the archived ``_cache`` documents are rewritten
to the extraction-specific output dir, mirroring what ``capture.py`` did.
"""

from __future__ import annotations

import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from chemrefine.state import JobBatch, StepContext, StepInputs

DATA_DIR = Path(__file__).resolve().parent / "data" / "e2e"
OUTPUT_DIR_TOKEN = "@OUTPUT_DIR@"


@dataclass(frozen=True)
class ReplayCase:
    """An extracted capture fixture."""

    root: Path
    """Extraction dir: holds ``input.yaml``, ``templates/``, the seed."""

    captured: Path
    """The archived output tree (``captured_outputs/``)."""

    @property
    def config_path(self) -> Path:
        """The case's pipeline config."""
        return self.root / "input.yaml"

    @property
    def output_dir(self) -> Path:
        """Where the replayed pipeline writes (the config's ``./outputs``)."""
        return self.root / "outputs"


def extract_case(name: str, dest: Path) -> ReplayCase:
    """Extract ``tests/data/e2e/<name>.tar.xz`` into ``dest``."""
    with tarfile.open(DATA_DIR / f"{name}.tar.xz") as tar:
        tar.extractall(dest, filter="data")
    case = ReplayCase(root=dest, captured=dest / "captured_outputs")
    for doc in case.captured.rglob("_cache/*.json"):
        doc.write_text(doc.read_text().replace(OUTPUT_DIR_TOKEN, str(case.output_dir)))
    return case


@dataclass
class ReplaySubmitter:
    """A ``run_batch``-shaped callable that satisfies jobs from the archive.

    For every prepared ``(input, output, id)`` triple it copies the archived
    files of that structure dir into place — the ``.out``, ensemble sidecars,
    script-engine ``.json``, whatever the capture kept. A structure missing
    from the archive fails loudly with the available paths, so nondeterminism
    drift (changed child IDs, renamed dirs) is diagnosed, not swallowed —
    unless the test listed it in ``allow_missing`` to simulate a failed job.
    """

    case: ReplayCase
    allow_missing: frozenset[str] = frozenset()
    calls: list[StepInputs] = field(default_factory=list)

    def __call__(self, engine: object, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        self.calls.append(inputs)
        for _inp, out, sid in inputs.files:
            rel = out.parent.relative_to(self.case.output_dir)
            source = self.case.captured / rel
            if not source.is_dir():
                if sid in self.allow_missing:
                    continue
                available = sorted(
                    str(p.relative_to(self.case.captured))
                    for p in self.case.captured.rglob("*")
                    if p.is_dir()
                )
                raise AssertionError(
                    f"no archived outputs for structure {sid!r} at {rel} — archive has: {available}"
                )
            for archived in source.iterdir():
                if archived.is_file():
                    shutil.copy2(archived, out.parent / archived.name)
        return JobBatch(jobs={})

    def submissions(self) -> list[str]:
        """Every structure id submitted, in submission order."""
        return [sid for inputs in self.calls for _inp, _out, sid in inputs.files]


def replay_run_batch(case: ReplayCase, allow_missing: set[str] | None = None) -> ReplaySubmitter:
    """A fresh archive-backed ``run_batch`` stand-in for ``case``."""
    return ReplaySubmitter(case, allow_missing=frozenset(allow_missing or ()))


def forbid_run_batch(engine: object, inputs: StepInputs, ctx: StepContext) -> JobBatch:
    """A ``run_batch`` stand-in that fails the test on any submission."""
    ids = [sid for _inp, _out, sid in inputs.files]
    raise AssertionError(f"unexpected submission of {ids} — this run must be cache-only")


def relocate(case: ReplayCase) -> None:
    """Stage the captured output tree as the config's live output dir."""
    shutil.copytree(case.captured, case.output_dir)
