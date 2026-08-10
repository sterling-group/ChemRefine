"""Replay captured real runs through the live pipeline.

The recordings under ``tests/data/e2e/recordings/`` are trimmed, compressed
outputs of real runs of the case definitions in ``tests/data/e2e/cases/``
(re-packed by :func:`pack_case` via ``pytest -m integration --record``).
Tests extract one into a tmp dir and drive the real pipeline against it in
two modes:

* **fresh replay** — monkeypatch :func:`chemrefine.engines._execution.run_batch`
  with :func:`replay_run_batch`, so the shipped engines run their real
  ``prepare`` / ``parse`` / NMS / filter / cache code and only the compute
  step is satisfied from the archive;
* **relocation** — copy ``captured_outputs`` to the config's output dir and
  run with :func:`forbid_run_batch`, asserting the cache fingerprints (and
  the ``rebuild-cache`` path) survive a move to a different machine/path.

``@OUTPUT_DIR@`` tokens in the archived ``_cache`` documents are rewritten
to the extraction-specific output dir, mirroring what :func:`pack_case` did.
"""

from __future__ import annotations

import fnmatch
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from chemrefine.engines import api
from chemrefine.engines.api import CompletionSink
from chemrefine.state import JobBatch, StepContext, StepInputs

DATA_DIR = Path(__file__).resolve().parent / "data" / "e2e" / "recordings"
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
    """Extract ``tests/data/e2e/recordings/<name>.tar.xz`` into ``dest``."""
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

    def __call__(
        self,
        engine: object,
        inputs: StepInputs,
        ctx: StepContext,
        *,
        sink: CompletionSink | None = None,
    ) -> JobBatch:
        """Satisfy a batch from the archive, then drive ``sink`` over it.

        Standing in for ``run_batch`` means standing in for *all* of it. A replacement that
        accepted ``sink`` and ignored it would leave every structure unparsed and the step
        failing for a reason nothing points at — the same trap the real job-array path has,
        answered the same way: sweep the drained batch, then run whatever it asks for.
        """
        batch = inputs
        while True:
            self.calls.append(batch)
            self._satisfy(batch)
            if sink is None or not (follow := api.sweep(batch, sink)):
                return JobBatch(jobs={})
            batch = StepInputs(follow)

    def _satisfy(self, inputs: StepInputs) -> None:
        """Copy each job's archived outputs into the place the engine will parse them from."""
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


# ---------------------------------------------------------------------------
# Recording (tier 3 → tier 2): pack a finished live run into an archive
# ---------------------------------------------------------------------------

# An archive holds what the *jobs* produced, keyed by the directory each job ran in —
# `ReplaySubmitter` serves a submission by copying that directory's files into place. Anything
# the pipeline does to a tree *after* its jobs finish is replayed, not recorded.
#
# NMS promotion is the case where that distinction bites. Once a parent is resolved, its
# canonical `.out` is a copy of the winning child's, and round 1 has moved into `attempt1/`.
# Packing that tree as-is would yield an archive whose round-1 submission returns a structure
# already at the target, so a fresh replay would resolve it in one round and never run round 2
# — the behaviour the recording exists to exercise. `pack_case` therefore restores the
# pre-promotion view while staging: wherever a structure directory holds an `attempt1/`, the
# attempt's loose files (round 1's originals) overwrite their canonical namesakes, and the
# children stay inside the attempt, where round 2's submissions are served from.
#
# The post-promotion tree is covered where it belongs: `tests/test_nms.py` pins the passthrough
# reading `attemptK/resolution.json`, and `pytest -m integration` runs it against real ORCA.

MAX_ARCHIVE_BYTES = 1_000_000

KEEP_PATTERNS = (
    "*.out",
    "*.finalensemble.xyz",
    "*.docker.struc1.allopt.xyz",
    "*.docker.struc1.all.optimized.xyz",  # ORCA 6.1.1's docker sidecar name
    "*.solventbuild.xyz",  # also matches 6.1.1's *.solvator.solventbuild.xyz
    "*.json",
)

# Derived / native-convenience artifacts a fresh replay must re-create itself —
# the generic *.json keep-pattern must not swallow them into the archive.
DROP_SUFFIXES = (".result.json", ".property.json")


def _kept(path: Path) -> bool:
    if path.name.endswith(DROP_SUFFIXES):
        return False
    if path.parent.name == "_cache":
        # `.npz` is the step cache's coordinate sidecar. Dropping it would archive a
        # `step.json` whose arrays are gone, and since `cache.load` fails closed on a missing
        # sidecar, every replay would resubmit instead of hitting the cache.
        return path.suffix in (".json", ".npz")
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in KEEP_PATTERNS)


def pack_case(run_dir: Path, name: str, dest_dir: Path = DATA_DIR) -> Path:
    """Pack a finished live run into ``dest_dir/<name>.tar.xz`` (the recording).

    Trims ``run_dir/outputs`` to what the parsers read (see ``KEEP_PATTERNS``),
    tokenizes the absolute output prefix in the ``_cache`` documents, and
    stages ``input.yaml`` + ``templates/`` + the seed alongside the trimmed
    ``captured_outputs/`` tree. Fails when the archive would exceed
    ``MAX_ARCHIVE_BYTES`` — recordings stay light by construction.
    """
    import tempfile

    import yaml

    run_dir = run_dir.resolve()
    config = yaml.safe_load((run_dir / "input.yaml").read_text())
    outputs = (run_dir / config.get("output_dir", "outputs")).resolve()
    if not outputs.is_dir():
        raise AssertionError(f"no outputs to pack in {run_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        shutil.copy2(run_dir / "input.yaml", staging / "input.yaml")
        shutil.copytree(run_dir / "templates", staging / "templates")
        seed = config.get("input")
        if seed is not None and not Path(seed).is_absolute():
            seed_path = (run_dir / seed).resolve()
            if seed_path.is_relative_to(run_dir) and not seed_path.is_relative_to(
                run_dir / "templates"
            ):
                shutil.copy2(seed_path, staging / seed_path.name)

        captured = staging / "captured_outputs"
        kept = 0
        for path in sorted(outputs.rglob("*")):
            if not path.is_file() or not _kept(path):
                continue
            target = captured / path.relative_to(outputs)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.parent.name == "_cache" and path.suffix == ".json":
                target.write_text(path.read_text().replace(str(outputs), OUTPUT_DIR_TOKEN))
            else:
                # The `.npz` sidecar holds no paths to tokenize, and is binary — running the
                # text substitution over it would fail to decode.
                shutil.copy2(path, target)
            kept += 1
        assert kept, f"nothing matched the keep patterns under {outputs}"

        # Restore the pre-promotion view (see the module comment above): round 1's own
        # outputs back at the canonical paths, so a replayed round-1 submission returns the
        # structure that *triggers* the resolution rather than its winner.
        for attempt in sorted(captured.rglob("attempt1")):
            for item in attempt.iterdir():
                if item.is_file() and (attempt.parent / item.name).exists():
                    shutil.copy2(item, attempt.parent / item.name)

        dest_dir.mkdir(parents=True, exist_ok=True)
        archive = dest_dir / f"{name}.tar.xz"
        with tarfile.open(archive, "w:xz") as tar:
            for path in sorted(staging.rglob("*")):
                tar.add(path, arcname=str(path.relative_to(staging)))

    size = archive.stat().st_size
    assert size <= MAX_ARCHIVE_BYTES, f"{archive} exceeds {MAX_ARCHIVE_BYTES} bytes — trim the case"
    return archive
