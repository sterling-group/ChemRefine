"""Replay captured real runs through the live pipeline.

The recordings under ``tests/data/e2e/recordings/`` are trimmed, compressed
outputs of real runs of the case definitions in ``tests/data/e2e/cases/``
(re-packed by :func:`pack_case` via ``pytest -m integration --record``).
Tests extract one into a tmp dir and drive the real pipeline against it in
two modes:

* **fresh replay** — monkeypatch :func:`chemrefine.engines._execution.run_batch`
  with a :class:`ReplaySubmitter`, so the shipped engines run their real
  ``prepare`` / ``parse`` / NMS / filter / cache code and only the compute
  step is satisfied from the archive;
* **relocation** — copy ``captured_outputs`` to the config's output dir and
  run with :func:`forbid_run_batch`, asserting the cache fingerprints (and
  the ``rebuild-cache`` path) survive a move to a different machine/path.

A recording is two things. ``captured_outputs/`` is the tree the run left, as a user's
tree would be — what relocation stages and a rebuild re-parses. ``submissions.json`` is
what each job produced, logged by :class:`SubmissionRecorder` the moment the job finished —
what a fresh replay serves. The tree alone cannot say that: the pipeline goes on working in
it after a job finishes (a convergence retry seals the first attempt into ``attemptK/`` and
re-runs in the same directory; NMS promotes a child's copy to its parent's), so one
directory holds several submissions, or none, and nothing on disk tells which.

A case may also carry ``case.yaml``: the scenario a run of it takes (:class:`CaseSpec`),
driven the same way live and in replay by :func:`drive`. Without one, a case is one ``run``.

An archive extracts and runs wherever it lands: nothing under ``_cache/``
names an absolute path any more, manifests included (they spell their files
relative to the step directory). Recordings packed before that spelling
carried an ``@OUTPUT_DIR@`` token that :func:`extract_case` re-anchored on the
way out; every shipped archive has since been re-packed, so the fixtures
relocate by the same rule a user's tree does rather than by a rewrite only
this harness knows about.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import tarfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from chemrefine import cache, recovery
from chemrefine.config import Config, load_config
from chemrefine.engines import api
from chemrefine.engines.api import (
    ArtifactEngine,
    CompletionSink,
    JobExecutable,
    ProvisionableEngine,
    get_engine,
)
from chemrefine.engines.mlip.train.engine import SIDECAR_NAME
from chemrefine.engines.orca.output import status as orca_status
from chemrefine.errors import ChemRefineError
from chemrefine.state import JobBatch, JobTriple, PipelineState, StepContext, StepInputs
from chemrefine.step import build_context

CASES_DIR = Path(__file__).resolve().parent / "data" / "e2e" / "cases"
DATA_DIR = Path(__file__).resolve().parent / "data" / "e2e" / "recordings"

CASE_SPEC = "case.yaml"
"""A case's scenario: beside its ``input.yaml`` in the case dir, and at an archive's root."""

SUBMISSIONS = "submissions.json"
"""The submission log's name, in a live run's directory and at an archive's root."""

Submissions = dict[str, list[list[dict[str, str]]]]
"""``{job dir relative to the output dir: [[{name, sha256}, …] per submission, in order]}``."""


def recorded_cases() -> list[str]:
    """Every recording on disk, by case name — what the drift detector has to cover."""
    return sorted(path.name.removesuffix(".tar.xz") for path in DATA_DIR.glob("*.tar.xz"))


def recording(name: str) -> Path:
    """The shipped archive of case ``name``."""
    return DATA_DIR / f"{name}.tar.xz"


# ---------------------------------------------------------------------------
# A case's scenario, driven the same way live and in replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Act:
    """One recovery action of a scenario, and the step it is declared to halt at (if any)."""

    action: recovery.Action
    target: str | int | None = None
    halts_at: int | None = None


@dataclass(frozen=True)
class CaseSpec:
    """What a case declares beside its config: the actions a run of it takes, and its cost.

    ``case.yaml`` holds ``scenario: [{action, target?, halts_at?}]`` (one ``run`` without
    it), ``budget_s`` — the wall time a live run of the scenario may take, so the tier
    cannot grow unnoticed — and ``heavy``, for a case whose load cannot be made small (it
    is marked, and a developer may deselect it). Unknown keys are refused at both levels:
    a misspelled ``halts_at`` would let a halt pass as a completion no check asserted.
    """

    scenario: tuple[Act, ...] = (Act(recovery.Action.RUN),)
    budget_s: float | None = None
    heavy: bool = False

    @property
    def halts_at(self) -> int | None:
        """The step the scenario leaves the tree halted at: its last action's declared halt."""
        return self.scenario[-1].halts_at

    @classmethod
    def read(cls, root: Path) -> CaseSpec:
        """The spec in ``root`` (a case dir, a run dir or an extracted archive)."""
        path = root / CASE_SPEC
        if not path.is_file():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _refuse_unknown(path, data, {"scenario", "budget_s", "heavy"})
        acts = []
        for entry in data.get("scenario") or [{"action": "run"}]:
            _refuse_unknown(path, entry, {"action", "target", "halts_at"})
            acts.append(
                Act(recovery.Action(entry["action"]), entry.get("target"), entry.get("halts_at"))
            )
        budget = data.get("budget_s")
        return cls(
            tuple(acts),
            budget_s=float(budget) if budget is not None else None,
            heavy=bool(data.get("heavy", False)),
        )

    def assert_within_budget(self, seconds: float) -> None:
        """A live run of the scenario took no longer than the case declares it may."""
        assert self.budget_s is not None, f"a live case declares budget_s in {CASE_SPEC}"
        assert seconds <= self.budget_s, (
            f"the scenario ran {seconds:.1f} s against a budget of {self.budget_s:g} s — "
            f"make the case lighter, or raise the budget with the reason (a busy machine "
            f"is not one)"
        )


def _refuse_unknown(path: Path, data: Mapping[str, Any], known: set[str]) -> None:
    """Fail on any key of ``data`` outside ``known``, naming the file."""
    unknown = sorted(set(data) - known)
    assert not unknown, f"{path}: unknown key(s) {unknown}; known: {sorted(known)}"


def drive(config: Config, spec: CaseSpec) -> None:
    """Take ``spec``'s actions on ``config`` through the CLI's own dispatcher.

    A halt is declared, never tolerated. An action with no declared halt must complete,
    leaving the last step cached; a halting one raises through. An action declared to halt
    at step k must raise the halt naming step k and leave the tree :func:`assert_halted`
    describes; one that completes instead fails.
    """
    for act in spec.scenario:
        if act.halts_at is None:
            recovery.execute(config, act.action, act.target)
            last = config.steps[-1]
            assert cache.load(config.step_dir(last).resolve()) is not None, (
                f"{act.action.value} completed without caching step {last.step}"
            )
            continue
        try:
            recovery.execute(config, act.action, act.target)
        except ChemRefineError as e:
            assert f"step {act.halts_at} halted" in str(e), (
                f"{act.action.value} was declared to halt at step {act.halts_at}: {e}"
            )
        else:
            raise AssertionError(
                f"{act.action.value} was declared to halt at step {act.halts_at} and completed"
            )
        assert_halted(config, act.halts_at)


def assert_halted(config: Config, step_number: int) -> None:
    """The tree a halt at ``step_number`` leaves: that step pending, nothing after it cached.

    The step is an ``on_failure: stop`` step with a non-empty ledger — what ``resume`` and
    ``rerun-errors`` read to know what to re-attempt — and no later step has a cache, since
    the halt is what stopped them running.
    """
    step = config.find_step(step_number)
    assert step.on_failure == "stop", f"step {step_number} halted but is {step.on_failure!r}"
    assert cache.load_failure_records(config.step_dir(step).resolve()), (
        f"step {step_number} halted with an empty ledger"
    )
    later = [
        s.step
        for s in config.steps
        if s.step > step_number and cache.load(config.step_dir(s).resolve()) is not None
    ]
    assert not later, f"steps {later} were cached after the halt at step {step_number}"


def with_backend_python(config: Config) -> Config:
    """Point every provisioned step at a dummy interpreter, so preflight passes without it.

    A replay submits nothing to a backend, and a machine replaying a recording (CI) has
    none of the stacks installed; the pipeline's own backend preflight would refuse it.
    """
    steps = [
        step.model_copy(update={"options": {**(step.options or {}), "backend_python": "python"}})
        if isinstance(get_engine(step.engine), ProvisionableEngine)
        else step
        for step in config.steps
    ]
    return config.model_copy(update={"steps": steps})


def reached(root: Path, config: Config) -> tuple[Config, int | None]:
    """``config`` cut to the steps a recording at ``root`` reached and a rebuild can verify.

    Those before an artifact step (:func:`verifiable_prefix`), and none past the step the
    scenario leaves halted (:attr:`CaseSpec.halts_at`), which is returned with it when the
    cut keeps it.
    """
    halt = CaseSpec.read(root).halts_at
    steps = [
        step
        for step in config.steps[: verifiable_prefix(config)]
        if halt is None or step.step <= halt
    ]
    kept = halt if halt is not None and any(step.step == halt for step in steps) else None
    return config.model_copy(update={"steps": steps}), kept


def verifiable_prefix(config: Config) -> int:
    """How many leading steps a parse-only rebuild can re-derive: those before an artifact step.

    An artifact step's product is not archived (:func:`_stub_artifact`), and every step
    after it keys that product's bytes, which no stub reproduces. So the drift detector
    rebuilds only the prefix, and :func:`pack_case` archives no cache document past it — one
    nothing could verify is a fossil from the day it was packed.
    """
    for index, step in enumerate(config.steps):
        if isinstance(get_engine(step.engine), ArtifactEngine):
            return index
    return len(config.steps)


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

    def submissions(self) -> Submissions:
        """What each job directory's submissions produced, in the order they ran."""
        loaded: Submissions = json.loads((self.root / SUBMISSIONS).read_text(encoding="utf-8"))
        return loaded

    def submitted_file(self, rel: str, index: int, name: str) -> Path:
        """The archived copy of ``name`` as submission ``index`` (from 0) of ``rel`` wrote it."""
        [entry] = [e for e in self.submissions()[rel][index] if e["name"] == name]
        found = _locate(self.captured, entry)
        assert found is not None, f"{rel} submission {index}: {name} is not in the archive"
        return found


def _sha256(path: Path) -> str:
    """A file's content digest — what the submission log names each output by."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locate(captured: Path, entry: dict[str, str]) -> Path | None:
    """The archived file with ``entry``'s name and digest, or ``None`` if there is none.

    Found by name, then digest: a sealed attempt moved it into ``attemptK/``, and a promoted
    winner is a copy under its parent's name, so the directory it was written in is not
    always where it lives now — but its bytes are.
    """
    for path in sorted(captured.rglob(entry["name"])):
        if path.is_file() and _sha256(path) == entry["sha256"]:
            return path
    return None


def extract_case(archive: Path, dest: Path) -> ReplayCase:
    """Extract the recording at ``archive`` into ``dest``."""
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    return ReplayCase(root=dest, captured=dest / "captured_outputs")


@dataclass
class ReplaySubmitter:
    """A ``run_batch``-shaped callable that satisfies jobs from the archive.

    Each prepared ``(input, output, id)`` triple is a submission of its job directory, and
    the *k*-th submission of a directory is served what the *k*-th recorded one produced —
    the ``.out``, ensemble sidecars, script-engine ``.json``, whatever the capture kept —
    each file found by its digest wherever the run left it. A submission the recording does
    not hold fails loudly, so nondeterminism drift (changed child IDs, renamed dirs, a
    retry the live run did not make) is diagnosed, not swallowed. An artifact step's
    product, which no recording holds, is stubbed (:func:`_stub_artifact`).
    """

    case: ReplayCase
    calls: list[StepInputs] = field(default_factory=list)
    served: Counter[str] = field(default_factory=Counter)

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
            _stub_artifact(engine, ctx)
            if sink is None or not (follow := api.sweep(batch, sink)):
                return JobBatch(jobs={})
            batch = StepInputs(follow)

    def _satisfy(self, inputs: StepInputs) -> None:
        """Copy each job's recorded outputs into the place the engine will parse them from."""
        recorded = self.case.submissions()
        for _inp, out, sid in inputs.files:
            rel = out.parent.relative_to(self.case.output_dir).as_posix()
            spent = self.served[rel]
            if spent >= len(recorded.get(rel, [])):
                raise AssertionError(
                    f"submission {spent + 1} of {rel} ({sid!r}) was never recorded — the "
                    f"recording holds {len(recorded.get(rel, []))}; recorded directories: "
                    f"{sorted(recorded)}"
                )
            entries = recorded[rel][spent]
            found = {entry["name"]: _locate(self.case.captured, entry) for entry in entries}
            missing = sorted(name for name, path in found.items() if path is None)
            if missing:
                raise AssertionError(f"{rel} submission {spent + 1}: {missing} not in the archive")
            self.served[rel] += 1
            for name, path in found.items():
                assert path is not None
                shutil.copy2(path, out.parent / name)

    def submissions(self) -> list[str]:
        """Every structure id submitted, in submission order."""
        return [sid for inputs in self.calls for _inp, _out, sid in inputs.files]

    def assert_replayed(self) -> None:
        """Every recorded submission was served, once, and nothing else was.

        What makes a passing replay a replay of *the recorded run*: a pipeline that skipped
        a retry or an NMS round the live run made would otherwise pass on whatever it did
        submit.
        """
        recorded = {rel: len(subs) for rel, subs in self.case.submissions().items()}
        assert dict(self.served) == recorded, (
            f"served {dict(self.served)} but the recording holds {recorded}"
        )


ARTIFACT_STUB = b"replayed-stub-model"


def _stub_artifact(engine: object, ctx: StepContext) -> None:
    """Stand in for an artifact step's product, which no recording holds.

    A trained model is megabytes of weights nothing offline parses: the pipeline checks it
    exists and digests its bytes. So it is not archived, and the replay writes stub bytes
    where the engine looks for it, when it is absent. The steps after it then key the stub
    consistently; a parse-only rebuild stops before it (:func:`verifiable_prefix`).
    """
    if not isinstance(engine, ArtifactEngine):
        return
    product = engine.artifact(ctx)
    if not product.exists():
        product.parent.mkdir(parents=True, exist_ok=True)
        product.write_bytes(ARTIFACT_STUB)


def forbid_run_batch(
    engine: object,
    inputs: StepInputs,
    ctx: StepContext,
    *,
    sink: CompletionSink | None = None,
) -> JobBatch:
    """A ``run_batch`` stand-in that fails the test on any submission.

    Its signature is ``run_batch``'s, ``sink`` included: a job engine submits with ``sink=``,
    and a stand-in without it failed as a ``TypeError`` rather than saying what happened.
    """
    ids = [sid for _inp, _out, sid in inputs.files]
    raise AssertionError(f"unexpected submission of {ids} — this run must be cache-only")


def relocate(case: ReplayCase) -> None:
    """Stage the captured output tree as the config's live output dir."""
    shutil.copytree(case.captured, case.output_dir)


def forget_provenance(case: ReplayCase) -> None:
    """Strip the cache-key provenance from every manifest in the staged output dir.

    The one legitimate use is regenerating a recording after a deliberate key move — an
    options default, a digest input — which is what ``--update-recordings`` is for. A
    rebuild refuses rows keyed under a different configuration, and an archive whose keys
    moved is exactly that; without its provenance the same manifest is *unprovable rather
    than wrong* (``rebuild_cache_step``'s own distinction), so the rebuild proceeds and
    writes the provenance back under the current key. The re-packed archive then carries
    the keys today's code derives, with every archived output untouched.
    """
    for step_dir in sorted(p for p in case.output_dir.iterdir() if p.is_dir()):
        manifest = cache.manifest_path(step_dir)
        if not manifest.is_file():
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for key in ("fingerprint", "criterion_key", "search_key"):
            data.pop(key, None)
        for record in data.get("files") or []:
            for key in ("row_key", "parent_digest"):
                record.pop(key, None)
        manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Recording (tier 3 → tier 2): pack a finished live run into an archive
# ---------------------------------------------------------------------------

# What a replay serves is a fact recorded as the run happened, not something read back off the
# tree it left. Every job passes `sink.on_complete` exactly once, after its outputs are in
# place and before anything parses them, so that is where `SubmissionRecorder` logs them.
# Everything the pipeline writes into a job directory afterwards — the canonical
# `.result.json`, an NMS `resolution.json`, a trained model's sidecar — is later than that
# moment, so it is not a job output, and a replay recreates it the way the run did.


@dataclass
class SubmissionRecorder:
    """Stands in for ``run_batch`` on a live run and logs what each submission produced.

    Wraps the sink each batch runs under, so every completion — streamed or swept, retried
    or a child of an NMS round — is recorded as the job's kept files (``_kept``), by name
    and digest, before the pipeline goes on. :meth:`write` leaves the log beside the run's
    ``input.yaml``, where :func:`pack_case` requires it.
    """

    run_batch: Callable[..., JobBatch]
    output_dir: Path
    log: Submissions = field(default_factory=dict)

    def __call__(
        self,
        engine: object,
        inputs: StepInputs,
        ctx: StepContext,
        *,
        sink: CompletionSink = api.NULL_SINK,
    ) -> JobBatch:
        """Run the batch for real, under a sink that records each job as it completes."""
        # The array path re-enters `run_batch` for its follow-ups with the sink it was
        # handed — this one, already recording.
        recording = sink if isinstance(sink, _Recording) else _Recording(self, sink)
        return self.run_batch(engine, inputs, ctx, sink=recording)

    def record(self, job: JobTriple) -> None:
        """Log the kept files ``job`` left in its directory as that directory's next submission."""
        _inp, out, _sid = job
        job_dir = out.parent.resolve()
        rel = job_dir.relative_to(self.output_dir.resolve())
        files = [
            {"name": path.name, "sha256": _sha256(path)}
            for path in sorted(job_dir.iterdir())
            if path.is_file() and _kept(path, rel / path.name, None)
        ]
        self.log.setdefault(rel.as_posix(), []).append(files)

    def write(self, run_dir: Path) -> None:
        """Leave the log as ``run_dir/submissions.json``."""
        text = json.dumps(self.log, indent=1, sort_keys=True) + "\n"
        (run_dir / SUBMISSIONS).write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class _Recording:
    """The sink :class:`SubmissionRecorder` runs a batch under: record, then pass it on."""

    recorder: SubmissionRecorder
    inner: CompletionSink

    def on_complete(self, job: JobTriple) -> tuple[JobTriple, ...]:
        """Record the finished job, then let the real sink decide what follows."""
        self.recorder.record(job)
        return self.inner.on_complete(job)


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
# the generic *.json keep-pattern must not swallow them into the archive. A trained model's
# record is written by the pipeline once the job is done, and nothing reads it back.
DROP_SUFFIXES = (".result.json", ".property.json", SIDECAR_NAME)

ABSOLUTE_PATH_RE = re.compile(r"(?<![\w.:/])/[\w.@+-]+/")
"""A ``/`` that opens a token and is followed by a segment and another ``/``: an absolute path.

What no archived ``*.json`` and no kept ``.err`` may hold — a recording that names the
machine it was made on cannot replay anywhere else. A URL, a ``host:port``, a ``./`` path,
a step-relative label (``3/step1_3.out``) and a source location ORCA prints
(``orca_main/x.cpp``) all pass."""


def _stderr_says_why(err: Path) -> bool:
    """Whether a job's ``.err`` is kept: non-empty, beside an ORCA output that died.

    ORCA puts the cause of an abort on stderr, and the failure reason quotes it, so a replay
    of that job needs it to reproduce the reason. A normal run's stderr (a library's
    warnings) and a job with no ``.out`` explain nothing a replay reads.
    """
    out = err.with_suffix(".out")
    return (
        err.stat().st_size > 0
        and out.is_file()
        and not orca_status.parse_terminated_normally(
            out.read_text(encoding="utf-8", errors="replace")
        )
    )


def _step_number(dir_name: str) -> int:
    """The step a ``stepN`` / ``stepN_name`` output directory belongs to."""
    return int(dir_name.removeprefix("step").split("_", 1)[0])


def _kept(path: Path, rel: Path, verifiable: int | None) -> bool:
    """Whether the file at ``path`` (``rel`` under the output dir) belongs in the archive."""
    if rel.name.endswith(DROP_SUFFIXES):
        return False
    if rel.suffix == ".err":
        return _stderr_says_why(path)
    if rel.parent.name == "_cache":
        if verifiable is not None and _step_number(rel.parts[0]) > verifiable:
            return False  # past what a rebuild can verify: a fossil by construction
        # `.npz` is the step cache's coordinate sidecar. Dropping it would archive a
        # `step.json` whose arrays are gone, and since `cache.load` fails closed on a missing
        # sidecar, every replay would resubmit instead of hitting the cache.
        return rel.suffix in (".json", ".npz")
    return any(fnmatch.fnmatch(rel.name, pattern) for pattern in KEEP_PATTERNS)


def pack_case(run_dir: Path, name: str, dest_dir: Path = DATA_DIR) -> Path:
    """Pack a finished live run into ``dest_dir/<name>.tar.xz`` (the recording).

    Trims ``run_dir/outputs`` to what the parsers read (see ``KEEP_PATTERNS``) and a
    parse-only rebuild can verify (:func:`verifiable_prefix`), refuses a ``_cache`` document
    that names the machine it was made on, and stages ``input.yaml`` + ``case.yaml`` +
    ``templates/`` + the seed + the run's ``submissions.json`` alongside the trimmed
    ``captured_outputs/``
    tree, which is packed as the run left it. Refuses a log that names a file the tree
    no longer holds, and fails when the archive would exceed ``MAX_ARCHIVE_BYTES`` —
    recordings stay light by construction.
    """
    import tempfile

    run_dir = run_dir.resolve()
    config = yaml.safe_load((run_dir / "input.yaml").read_text(encoding="utf-8"))
    outputs = (run_dir / config.get("output_dir", "outputs")).resolve()
    if not outputs.is_dir():
        raise AssertionError(f"no outputs to pack in {run_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        shutil.copy2(run_dir / "input.yaml", staging / "input.yaml")
        shutil.copytree(run_dir / "templates", staging / "templates")
        assert (run_dir / SUBMISSIONS).is_file(), (
            f"{run_dir} has no {SUBMISSIONS} — record the run with SubmissionRecorder"
        )
        shutil.copy2(run_dir / SUBMISSIONS, staging / SUBMISSIONS)
        if (run_dir / CASE_SPEC).is_file():
            shutil.copy2(run_dir / CASE_SPEC, staging / CASE_SPEC)
        seed = config.get("input")
        if seed is not None and not Path(seed).is_absolute():
            seed_path = (run_dir / seed).resolve()
            if seed_path.is_relative_to(run_dir) and not seed_path.is_relative_to(
                run_dir / "templates"
            ):
                shutil.copy2(seed_path, staging / seed_path.name)

        captured = staging / "captured_outputs"
        kept = 0
        verifiable = verifiable_prefix(load_config(run_dir / "input.yaml"))
        for path in sorted(outputs.rglob("*")):
            rel = path.relative_to(outputs)
            if not path.is_file() or not _kept(path, rel, verifiable):
                continue
            target = captured / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            kept += 1
        assert kept, f"nothing matched the keep patterns under {outputs}"

        # What the harness reads — every JSON document, and the stderr a failure reason
        # quotes — must name no machine: loudly, at record time, rather than as a relocated
        # replay reading paths that are not there. Captured engine output is exempt on
        # purpose: an ORCA log names the scratch dir it really ran in, and that is a record
        # of the run, not an address anything follows.
        machine_bound = sorted(
            str(doc.relative_to(captured))
            for doc in (*captured.rglob("*.json"), *captured.rglob("*.err"))
            if ABSOLUTE_PATH_RE.search(doc.read_text(encoding="utf-8", errors="replace"))
        )
        assert not machine_bound, (
            f"{machine_bound} name absolute paths — a recording that carries the machine it "
            f"was made on cannot replay anywhere else"
        )

        # Every logged output must still be in the tree. The pipeline only moves (a seal)
        # and copies (a promotion) what its jobs wrote, so one that is gone was rewritten in
        # place — a replay would have nothing to serve for that submission.
        logged = json.loads((staging / SUBMISSIONS).read_text(encoding="utf-8"))
        unfound = sorted(
            f"{rel} #{index + 1}: {entry['name']}"
            for rel, subs in logged.items()
            for index, files in enumerate(subs)
            for entry in files
            if _locate(captured, entry) is None
        )
        assert not unfound, f"logged outputs missing from the packed tree: {unfound}"

        dest_dir.mkdir(parents=True, exist_ok=True)
        archive = dest_dir / f"{name}.tar.xz"
        with tarfile.open(archive, "w:xz") as tar:
            for path in sorted(staging.rglob("*")):
                # `recursive=False` because `rglob` already yields every descendant:
                # left at its default, `add` walked each directory's whole subtree, so a
                # file was archived once per ancestor directory as well as for itself —
                # six copies of a round-2 output. Extraction hid it (each copy overwrote
                # the last with identical bytes) and `xz` squeezed the repeats to a couple
                # of percent, so what it really cost was the archive's honesty: its own
                # member list said a recording held three times what it holds.
                tar.add(path, arcname=str(path.relative_to(staging)), recursive=False)

    size = archive.stat().st_size
    assert size <= MAX_ARCHIVE_BYTES, f"{archive} exceeds {MAX_ARCHIVE_BYTES} bytes — trim the case"
    return archive


# ---------------------------------------------------------------------------
# What a live run must leave, beyond what its recording can carry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunlogJob:
    """One job as its runlog reports it: where it ran, when, and on how many cores."""

    host: str
    job_id: str
    start: str
    finish: str | None
    cores: int
    scratch: Path
    output: Path


def runlog_jobs(path: Path) -> list[RunlogJob]:
    """Every job a ``*.runlog`` reports, one per ``starting`` banner.

    A ``finished`` banner closes the job before it; a job without one never finished and
    is left open.
    """
    jobs: list[RunlogJob] = []
    fields: dict[str, str] = {}
    start: str | None = None

    def close(finish: str | None) -> None:
        if start is not None:
            jobs.append(
                RunlogJob(
                    fields["host"],
                    fields["job_id"],
                    start,
                    finish,
                    int(fields["cores"]),
                    Path(fields["scratch"]),
                    Path(fields["output"]),
                )
            )

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("[") and line.rstrip().endswith("starting"):
            close(None)
            start, fields = line[1 : line.index("]")], {}
        elif line.startswith("[") and line.rstrip().endswith("finished"):
            close(line[1 : line.index("]")])
            start = None
        elif start is not None and "=" in line:
            key, _, value = line.strip().partition("=")
            fields.setdefault(key, value)
    close(None)
    return jobs


def assert_throttled(jobs: list[RunlogJob], max_cores: int) -> None:
    """At no moment did the jobs running together hold more than ``max_cores``.

    Intervals are half-open, ``[start, finish)``, so a job that starts the second another
    finishes does not overlap it; a job that never finished runs to the end. The runlogs
    of a promoted NMS winner repeat its child's, so jobs are counted once by host, id and
    start. An upper bound only: whole-second timestamps give a sub-second job no length,
    so reaching the cap is never asserted.
    """
    unique = {(job.host, job.job_id, job.start): job for job in jobs}.values()
    events = sorted(
        [(job.start, 1, job.cores) for job in unique]
        + [(job.finish, 0, -job.cores) for job in unique if job.finish is not None]
    )
    running = 0
    for when, _order, delta in events:
        running += delta
        assert running <= max_cores, f"{running} cores ran at {when} under max_cores={max_cores}"


def assert_live_invariants(config: Config, run_dir: Path) -> None:
    """What every live run must leave, whatever its case.

    ORCA's own ``property.json`` beside every output that terminated normally; the scratch
    directories an engine copies home, present and non-empty wherever its jobs recorded a
    result; no working directory left in the tree, and none under ``scratch_dir``; each
    job's scratch where the config puts it; no process still naming the run; and the
    throttle held.
    """
    outputs = config.output_dir
    for out in outputs.rglob("*.out"):
        text = out.read_text(encoding="utf-8", errors="replace")
        if orca_status.parse_terminated_normally(text):
            assert out.with_suffix(".property.json").is_file(), f"{out}: no property.json"

    for step in config.steps:
        engine = get_engine(step.engine)
        if not isinstance(engine, JobExecutable):
            continue
        ctx = build_context(config, step, PipelineState(), engine)
        wanted = engine.output_dirs(ctx)
        if not wanted:
            continue
        homes = (
            {engine.artifact(ctx).parent}
            if isinstance(engine, ArtifactEngine)
            else {record.parent for record in ctx.step_dir.rglob("*.result.json")}
        )
        for home in homes:
            for name in wanted:
                assert any((home / name).iterdir()), f"{home / name}: empty or missing"

    assert not list(outputs.rglob("_work_*")), "a job's working directory was left behind"
    jobs = [job for runlog in outputs.rglob("*.runlog") for job in runlog_jobs(runlog)]
    for job in jobs:
        if config.scratch_dir is not None:
            assert job.scratch.is_relative_to(config.scratch_dir), job.scratch
        else:
            assert job.scratch.parent == job.output and job.scratch.name.startswith("_work_")
    if config.scratch_dir is not None and config.scratch_dir.is_dir():
        assert not list(config.scratch_dir.glob("ChemRefine_*")), "scratch left behind"

    lingering = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if str(run_dir) in argv:
            lingering.append(argv)
    assert not lingering, f"processes outlived the run: {lingering}"

    assert_throttled(jobs, config.max_cores)
