"""Full-pipeline replays of captured real runs (see ``tests/replay.py``).

Every recording is driven by one test. Its scenario (``case.yaml``, one ``run`` without it)
goes through the CLI's own dispatcher with the shipped engines doing their real prepare /
parse / NMS / filter / cache work; only the compute is served from the archive. The test
then holds what every case must satisfy, and :data:`CHECKS` holds what each case exists
for. Assertions are data-driven where they can be, recomputed from the archive with the
real parsers, so re-capturing a case does not stale them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from replay import (
    ARTIFACT_STUB,
    CASE_SPEC,
    CASES_DIR,
    SUBMISSIONS,
    Act,
    CaseSpec,
    ReplayCase,
    ReplaySubmitter,
    SubmissionRecorder,
    _stub_artifact,
    drive,
    extract_case,
    forbid_run_batch,
    pack_case,
    recorded_cases,
    recording,
    verifiable_prefix,
)

from chemrefine import cache, filtering, ids, pipeline
from chemrefine.config import Config, StepConfig, load_config
from chemrefine.engines.api import ArtifactEngine, ProvisionableEngine, get_engine
from chemrefine.engines.orca.output import parse_output
from chemrefine.errors import ChemRefineError
from chemrefine.io import read_xyz_frames
from chemrefine.recovery import Action
from chemrefine.state import JobBatch, JobTriple, StepContext, StepInputs
from chemrefine.step import RunPlan, StepMode, StepOutcome

RUN_BATCH = "chemrefine.engines._execution.run_batch"


@dataclass(frozen=True)
class Replayed:
    """A recording driven through its scenario, and each step it reached as a cache serves it."""

    case: ReplayCase
    config: Config
    spec: CaseSpec
    submitter: ReplaySubmitter
    reached: list[StepConfig]
    """Every step up to the scenario's final halt; all of them if it completes."""
    outcomes: list[StepOutcome]
    """What each reached step ends with, from a cache-only pass that submits nothing — what
    a user reading the finished tree gets."""


def _with_backend_python(config: Config) -> Config:
    """Point every provisioned step at a dummy interpreter, so preflight passes without it."""
    steps = [
        step.model_copy(update={"options": {**(step.options or {}), "backend_python": "python"}})
        if isinstance(get_engine(step.engine), ProvisionableEngine)
        else step
        for step in config.steps
    ]
    return config.model_copy(update={"steps": steps})


def replay(name: str, root: Path, monkeypatch: pytest.MonkeyPatch) -> Replayed:
    """Extract ``name`` under ``root``, drive its scenario, and read back what it reached."""
    case = extract_case(recording(name), root)
    config = _with_backend_python(load_config(case.config_path))
    spec = CaseSpec.read(case.root)
    submitter = ReplaySubmitter(case)
    monkeypatch.setattr(RUN_BATCH, submitter)
    drive(config, spec)
    submitter.assert_replayed()

    monkeypatch.setattr(RUN_BATCH, forbid_run_batch)
    halt = spec.halts_at
    reached = [s for s in config.steps if halt is None or s.step <= halt]
    outcomes = pipeline.run(
        config.model_copy(update={"steps": reached}), RunPlan(default=StepMode.CACHE_ONLY)
    )
    return Replayed(case, config, spec, submitter, reached, outcomes)


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


@pytest.mark.parametrize("name", recorded_cases())
def test_a_recording_replays_the_run_it_recorded(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every recorded submission is served once, the tree is whole, and the case holds.

    Whole means what every tree must be: ``steps.csv`` reports each step's survivors with a
    finite ranking energy, each step's survivors file holds one frame per survivor, every
    cached structure with an energy has the canonical record of the job it came from (the
    NMS child's, for a promoted winner), and a completed tree resumes without submitting
    anything, to the same survivors.
    """
    run = replay(name, tmp_path, monkeypatch)

    reported: dict[str, list[str]] = {}
    with (run.case.output_dir / "steps.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            reported.setdefault(row["Step"], []).append(row["Conformer"])
    for step, outcome in zip(run.reached, run.outcomes, strict=True):
        survivors = outcome.state.structures
        ranking = filtering.ranking_energy(step.sample).attr
        finite = sorted(s.id for s in survivors if _finite(getattr(s, ranking)))
        assert sorted(reported.get(str(step.step), [])) == finite, f"step {step.step}: steps.csv"

        step_dir = run.config.step_dir(step).resolve()
        frames = read_xyz_frames(ids.step_survivors_path(step_dir, step.step))
        assert len(frames) == len(survivors), f"step {step.step}: survivors file"

        if isinstance(get_engine(step.engine), ArtifactEngine):
            continue
        records = {
            json.loads(path.read_text(encoding="utf-8"))["id"]
            for path in step_dir.rglob("*.result.json")
        }
        cached = cache.load(step_dir)
        assert cached is not None
        wanted = {
            s.resolved_from or s.id
            for s in cached.results.structures
            if s.energy_hartree is not None
        }
        assert records >= wanted, f"step {step.step}: no record for {sorted(wanted - records)}"

    CHECKS[name](run)

    if run.spec.halts_at is None:
        again = pipeline.run(run.config)
        assert all(outcome.cache_hit for outcome in again), "a finished tree resumes from cache"
        assert [s.id for s in again[-1].state.structures] == [
            s.id for s in run.outcomes[-1].state.structures
        ]


@pytest.mark.parametrize("name", recorded_cases())
def test_a_recording_keys_every_step_the_same_at_two_roots(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tree copied elsewhere keys every step the same, so it resumes rather than recomputes.

    A path that reaches a key names the machine: ``model_path`` once did, resolved to an
    absolute path by the loader, so the same recording replayed at two roots derived two
    fingerprints for the step that loads a trained model. Every recording is replayed
    twice and compared.
    """

    def fingerprints(root: Path) -> dict[str, str]:
        run = replay(name, root, monkeypatch)
        return {
            doc.parent.parent.name: json.loads(doc.read_text(encoding="utf-8"))["fingerprint"]
            for doc in sorted(run.case.output_dir.glob("step*/_cache/step.json"))
        }

    first = fingerprints(tmp_path / "cluster")
    assert first, "the replay cached no step"
    assert first == fingerprints(tmp_path / "laptop")


def test_every_case_has_one_recording_and_one_check() -> None:
    """A case without a recording is never replayed, and a recording without a check proves
    only that it runs. Each is one directory, one archive and one entry here."""
    cases = sorted(p.name for p in CASES_DIR.iterdir() if (p / "input.yaml").is_file())
    assert cases == recorded_cases() == sorted(CHECKS)


# ---------------------------------------------------------------------------
# What each case exists for
# ---------------------------------------------------------------------------


def _conformers(run: Replayed) -> None:
    """GOAT ensemble → opt+freq ranked by Gibbs → single point: 3 → 2 → 1 survivors.

    The inputs carry the workflow's charge and the template's cores clamped to the budget.
    The survivors of step 2 are the lowest by the energy its filter names. These three
    conformers order identically by electronic and Gibbs energy (the degeneracy the case's
    input.yaml documents), so the Gibbs-vs-electronic distinction itself is pinned at the
    unit tier; this holds that the ranking energy reaches the filter end to end.
    """
    case, submitter = run.case, run.submitter
    ensemble = read_xyz_frames(next(case.captured.glob("step1/*/step1_*.finalensemble.xyz")))
    assert len(ensemble) >= 3, "the GOAT ensemble must feed the downstream count: 3"
    fan_out = len(submitter.calls[1].files)
    assert fan_out == min(3, len(ensemble)), "step 2 runs step 1's three survivors"
    assert [len(o.state.structures) for o in run.outcomes] == [fan_out, min(2, fan_out), 1]

    parsed: dict[str, tuple[float, float]] = {}
    for out in case.captured.glob("step2/*/step2_*.out"):
        result = parse_output(out, "opt_sp")[0]
        assert result.gibbs_hartree is not None and result.energy_hartree is not None
        parsed[out.parent.name] = (result.gibbs_hartree, result.energy_hartree)
    survivors = [s.id for s in run.outcomes[1].state.structures]
    assert survivors == sorted(parsed, key=lambda sid: parsed[sid][0])[: len(survivors)]
    assert all(s.gibbs_hartree is not None for s in run.outcomes[1].state.structures)

    goat_inp = (case.output_dir / "step1" / "0" / "step1_0.inp").read_text(encoding="utf-8")
    assert f"nprocs {run.config.max_cores}" in goat_inp, "the template's 16 cores are clamped"
    assert "nprocs 16" not in goat_inp
    assert "* xyzfile 0 1 " in goat_inp
    step2_inp = next(case.output_dir.glob("step2/*/step2_*.inp")).read_text(encoding="utf-8")
    assert "* xyzfile 0 1 " in step2_inp


def _nms_minimum(run: Replayed) -> None:
    """A saddle-point seed resolved to a true minimum by NMS round 2, under the parent's id."""
    case, submitter = run.case, run.submitter
    # Round 1 as it was submitted: the canonical `.out` of the recorded tree is the promoted
    # winner, as a user's tree has it.
    round_one = parse_output(case.submitted_file("step1/0", 0, "step1_0.out"), "opt_sp")[0]
    assert round_one.imaginary_freqs, "the captured seed must be a saddle"

    structure_dir = case.output_dir / "step1" / "0"
    attempt = structure_dir / "attempt1"
    children = sorted(p.name for p in attempt.iterdir() if p.is_dir())
    assert children and all(name.startswith("0_m") for name in children)
    assert set(submitter.submissions()) >= set(children)
    assert list(attempt.glob("*/*.result.json")), "round-2 children leave canonical records"
    # The attempt holds round 1 (loose files) and its children (subdirectories); the
    # canonical location holds exactly one calculation — the winner's.
    assert (attempt / "step1_0.out").is_file(), "round 1 is archived, not overwritten"
    canonical_out = (structure_dir / "step1_0.out").read_text(encoding="utf-8")
    child_outs = [p.read_text(encoding="utf-8") for p in sorted(attempt.glob("*/step1_*.out"))]
    assert canonical_out != (attempt / "step1_0.out").read_text(encoding="utf-8")
    assert child_outs.count(canonical_out) == 1, "canonical is one specific child's output"
    caption = (structure_dir / "step1_0.xyz").read_text(encoding="utf-8").splitlines()[1]
    assert caption == "NMS-resolved 0"

    (survivor,) = run.outcomes[0].state.structures
    assert survivor.id == "0", "the resolved child is written back under the parent id"
    assert survivor.converged
    assert survivor.imaginary_freqs == {}, "a verified minimum — a table, with nothing in it"
    assert len(run.outcomes[1].state.structures) == 1


def _ts_pes(run: Replayed) -> None:
    """A scan max-sampled to its highest point, then an OptTS that passes the inferred ts target."""
    scan_points = parse_output(run.case.captured / "step1" / "0" / "step1_0.out", "pes")
    assert len(scan_points) >= 3, "the scan must resolve several points"
    assert len(run.submitter.calls[1].files) == 1, "max count: 1 keeps one TS candidate"
    (ts_seed,) = run.outcomes[0].state.structures
    assert ts_seed.energy_hartree == max(
        p.energy_hartree for p in scan_points if p.energy_hartree is not None
    )
    (ts,) = run.outcomes[1].state.structures
    assert ts.imaginary_freqs is not None and len(ts.imaginary_freqs) == 1
    assert not list((run.case.output_dir / "step2" / ts.id).glob("attempt*")), (
        "a clean TS needs no NMS round 2"
    )


def _host_guest(run: Replayed) -> None:
    """Docker fan-out with the seed's lineage, a per-step charge, and a solvator parse."""
    case = run.case
    poses = parse_output(case.captured / "step1" / "0" / "step1_0.out", "docker")
    assert len(poses) >= 2, "docking must produce poses to filter"
    kept = len(run.outcomes[0].state.structures)
    assert kept == min(3, len(poses))
    assert {s.id for s in run.outcomes[0].state.structures} <= {
        f"0-{i}" for i in range(len(poses))
    }, "docker children carry the seed's lineage"
    step2_inp = next(case.output_dir.glob("step2/*/step2_*.inp")).read_text(encoding="utf-8")
    assert "* xyzfile -1 1 " in step2_inp, "the per-step charge override reaches the input"
    assert len(run.submitter.calls[1].files) == kept
    assert run.outcomes[1].state.structures, "solvator output parses into survivors"


def _mlip_screen(run: Replayed) -> None:
    """Two seeds through MACE, the knobs rendered, a Boltzmann cut keeping both."""
    assert len(read_xyz_frames(run.case.root / "glycol2.xyz")) == 2
    assert len(run.submitter.calls[0].files) == 2
    rendered = (run.case.output_dir / "step1" / "0" / "step1_0.py").read_text(encoding="utf-8")
    assert '"small"' in rendered and '"mace_off"' in rendered and '"cpu"' in rendered
    # At 99% cumulative weight over two close MACE energies, both survive.
    assert len(run.outcomes[0].state.structures) == 2, "boltzmann keeps both close conformers"


def _mlip_extopt(run: Replayed) -> None:
    """ORCA's optimiser over MACE gradients through the wrapper, NumFreq included."""
    rendered = (run.case.output_dir / "step1" / "0" / "step1_0.inp").read_text(encoding="utf-8")
    assert "ProgExt" in rendered, "ExtOpt inputs must point ORCA at the gradient wrapper"
    (survivor,) = run.outcomes[0].state.structures
    assert survivor.converged and survivor.energy_hartree is not None


def _extopt_cycle_limit(run: Replayed) -> None:
    """An ExtOpt run out of cycles is retried from its last cycle; ``best`` keeps the retry's.

    ExtOpt prints ``FINAL SINGLE POINT ENERGY`` for its first cycle only, so a run read at
    the last of those came back at the geometry it was submitted with: the retry resubmitted
    the seed, stopped at the same limit on the same path, and ``best`` carried an
    unoptimised input downstream.
    """
    case = run.case
    assert run.submitter.submissions() == ["0", "0"], "one convergence retry"
    stopped = parse_output(case.submitted_file("step1/0", 0, "step1_0.out"), "opt_sp")[0]
    retried = parse_output(case.submitted_file("step1/0", 1, "step1_0.out"), "opt_sp")[0]
    (retry_input,) = read_xyz_frames(case.output_dir / "step1" / "0" / "step1_0_inp.xyz")
    assert np.allclose(retry_input.positions, stopped.positions), "retried from its last cycle"
    (seed,) = read_xyz_frames(case.root / "water_distorted.xyz")
    assert not np.allclose(retry_input.positions, seed.positions, atol=1e-3)
    (survivor,) = run.outcomes[0].state.structures
    assert survivor.converged is False
    assert survivor.energy_hartree == retried.energy_hartree
    assert np.allclose(survivor.atoms.positions, retried.positions)


def _mlip_train(run: Replayed) -> None:
    """Label → fine-tune (one job over the ensemble) → run the trained model on every structure.

    The dataset is written from the labelled ensemble, the real MACE template renders
    against it, the one job flows through the scheduler under the training id, the
    artifact decides success, the sidecar cites the model, and the next step consumes it.
    """
    case = run.case
    train_calls = [
        call
        for call in run.submitter.calls
        if any(sid == ids.TRAINING_ID for *_x, sid in call.files)
    ]
    assert len(train_calls) == 1 and len(train_calls[0].files) == 1
    run_dir = case.output_dir / "step2" / "train"
    rendered = (run_dir / "step2_train.yaml").read_text(encoding="utf-8")
    assert str(run_dir / "train.xyz") in rendered and (run_dir / "train.xyz").is_file()
    assert str(run_dir / "valid.xyz") in rendered and (run_dir / "valid.xyz").is_file()
    sidecar = json.loads((run_dir / "trained_model.json").read_text(encoding="utf-8"))
    assert (sidecar["task_name"], sidecar["backend"]) == ("mace_off", "mlip-mace")
    assert sidecar["n_structures"] == 4

    labelled = [s.id for s in run.outcomes[0].state.structures]
    assert [s.id for s in run.outcomes[1].state.structures] == labelled
    rendered3 = (case.output_dir / "step3" / "0" / "step3_0.py").read_text(encoding="utf-8")
    assert str(run_dir / "train.model") in rendered3, "step 3 loads the model step 2 produced"
    assert len(run.outcomes[2].state.structures) == 4


def _pyscf_sp(run: Replayed) -> None:
    """Water, HF/STO-3G through the PySCF script engine: about -74.96 Eh."""
    (structure,) = run.outcomes[0].state.structures
    assert structure.energy_hartree is not None
    assert -75.5 < structure.energy_hartree < -74.5, structure.energy_hartree


def _pyscf_extopt(run: Replayed) -> None:
    """Water optimised over PySCF HF gradients, then NumFreq: a minimum with all-real modes."""
    _pyscf_sp(run)
    (structure,) = run.outcomes[0].state.structures
    assert structure.imaginary_freqs == {}, "HF water minimum has no imaginary modes"


def _fairchem_sp(run: Replayed) -> None:
    """UMA's omol head on water: about -76.4 Eh, a number a quantum chemist recognises."""
    (structure,) = run.outcomes[0].state.structures
    assert structure.energy_hartree is not None
    assert -77.0 < structure.energy_hartree < -76.0, structure.energy_hartree


CHECKS: dict[str, Callable[[Replayed], None]] = {
    "conformers": _conformers,
    "extopt_cycle_limit": _extopt_cycle_limit,
    "fairchem_sp": _fairchem_sp,
    "host_guest": _host_guest,
    "mlip_extopt": _mlip_extopt,
    "mlip_screen": _mlip_screen,
    "mlip_train": _mlip_train,
    "nms_minimum": _nms_minimum,
    "pyscf_extopt": _pyscf_extopt,
    "pyscf_sp": _pyscf_sp,
    "ts_pes": _ts_pes,
}
"""What each case exists for, beyond what every recording is held to."""


# ---------------------------------------------------------------------------
# The archives themselves
# ---------------------------------------------------------------------------


def test_every_recording_holds_each_file_once() -> None:
    """A recording is a set of files, and a tar can hold the same name many times.

    ``pack_case`` walks the staged tree with ``rglob`` and adds each entry, and
    ``TarFile.add`` recurses by default — so every file was archived once per ancestor
    directory as well as for itself, six copies deep under an ``attemptK/``. Nothing
    failed: extraction overwrote each copy with identical bytes and ``xz`` squeezed the
    repeats to a couple of percent. What it cost is the property asserted here — that
    the archive's own member list describes the tree it captured, which is what anyone
    reads it with, and what the next packer change would be checked against.
    """
    from tarfile import open as tar_open

    from replay import DATA_DIR

    archives = sorted(DATA_DIR.glob("*.tar.xz"))
    assert archives, "no recordings to check"
    for archive in archives:
        with tar_open(archive) as tar:
            repeated = [name for name, n in Counter(tar.getnames()).items() if n > 1]
        assert not repeated, f"{archive.name} archives {len(repeated)} name(s) more than once"


# ---------------------------------------------------------------------------
# The harness: what a recording says each submission produced
# ---------------------------------------------------------------------------


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _logged_case(root: Path, outputs: dict[str, list[str]]) -> ReplayCase:
    """A hand-made recording: ``{job dir: [what each submission wrote as step1_0.out]}``.

    Laid out as the pipeline leaves a retried structure — each submission but the last
    sealed into ``attemptK/``, the last at the directory itself.
    """
    captured = root / "captured_outputs"
    log: dict[str, list[list[dict[str, str]]]] = {}
    for rel, texts in outputs.items():
        for k, text in enumerate(texts, start=1):
            home = captured / rel if k == len(texts) else captured / rel / f"attempt{k}"
            home.mkdir(parents=True, exist_ok=True)
            (home / "step1_0.out").write_text(text, encoding="utf-8")
            log.setdefault(rel, []).append([{"name": "step1_0.out", "sha256": _digest(text)}])
    (root / SUBMISSIONS).write_text(json.dumps(log), encoding="utf-8")
    return ReplayCase(root=root, captured=captured)


def _submit(submitter: ReplaySubmitter, rel: str) -> str:
    """Submit ``rel``'s job once; return the output the submitter served for it."""
    out = submitter.case.output_dir / rel / "step1_0.out"
    out.parent.mkdir(parents=True, exist_ok=True)
    submitter._satisfy(StepInputs(files=((out.with_suffix(".inp"), out, "0"),)))
    return out.read_text(encoding="utf-8")


def test_a_directory_is_served_its_recorded_submissions_in_order(tmp_path: Path) -> None:
    """A retried structure's directory is submitted twice, and replays both runs in order.

    The directory alone cannot say that: the run sealed the first submission into
    ``attempt1/`` and ran the retry where it had been, while an NMS parent that stayed
    unresolved leaves an ``attempt1/`` holding only its children. The log recorded each
    submission as it finished, and a replay serves exactly that.
    """
    submitter = ReplaySubmitter(_logged_case(tmp_path, {"step1/0": ["first try", "the retry"]}))
    assert [_submit(submitter, "step1/0") for _ in range(2)] == ["first try", "the retry"]
    submitter.assert_replayed()


def test_a_submission_past_the_recording_is_refused(tmp_path: Path) -> None:
    """A replay that submits more than the live run did has diverged — it is not served."""
    submitter = ReplaySubmitter(_logged_case(tmp_path, {"step1/0": ["only try"]}))
    _submit(submitter, "step1/0")
    with pytest.raises(AssertionError, match=r"submission 2 of step1/0 .* never recorded"):
        _submit(submitter, "step1/0")


def test_replay_that_skips_a_recorded_submission_is_caught(tmp_path: Path) -> None:
    """A replay that never makes the retry the live run made is not a replay of that run."""
    submitter = ReplaySubmitter(_logged_case(tmp_path, {"step1/0": ["first try", "the retry"]}))
    _submit(submitter, "step1/0")
    with pytest.raises(AssertionError, match="but the recording holds"):
        submitter.assert_replayed()


def test_the_recorder_logs_each_completion_as_the_job_left_it(tmp_path: Path) -> None:
    """One entry per completion, holding the kept files present at that moment.

    A file the pipeline writes after a job completes (an NMS ``resolution.json``) is not
    the job's, and an input is not an output. The follow-up batch re-enters ``run_batch``
    with the sink it was handed, as the array path does — already recording, so nothing is
    logged twice.
    """
    outputs = tmp_path / "outputs"
    job = outputs / "step1" / "0"
    job.mkdir(parents=True)
    triple: JobTriple = (job / "step1_0.inp", job / "step1_0.out", "0")

    def run_batch(engine: object, inputs: StepInputs, ctx: StepContext, *, sink: Any) -> JobBatch:
        retry = (job / "step1_0.out").exists()
        (job / "step1_0.inp").write_text("an input", encoding="utf-8")
        (job / "step1_0.out").write_text("the retry" if retry else "first try", encoding="utf-8")
        sink.on_complete(triple)
        if not retry:
            recorder(engine, inputs, ctx, sink=sink)
            (job / "resolution.json").write_text("{}", encoding="utf-8")
        return JobBatch(jobs={})

    recorder = SubmissionRecorder(run_batch, outputs)
    recorder(object(), StepInputs(files=(triple,)), cast(StepContext, None))
    recorder.write(tmp_path)

    logged = json.loads((tmp_path / SUBMISSIONS).read_text(encoding="utf-8"))
    assert logged == {
        "step1/0": [
            [{"name": "step1_0.out", "sha256": _digest("first try")}],
            [{"name": "step1_0.out", "sha256": _digest("the retry")}],
        ]
    }


def test_a_log_naming_a_file_the_tree_lost_is_refused_at_pack_time(tmp_path: Path) -> None:
    """A recorded output rewritten in place would leave its submission nothing to serve."""
    run_dir = tmp_path / "run"
    _fake_case(run_dir)
    job = run_dir / "outputs" / "step1" / "0"
    job.mkdir(parents=True)
    (job / "step1_0.out").write_text("rewritten", encoding="utf-8")
    log = {"step1/0": [[{"name": "step1_0.out", "sha256": _digest("as the job wrote it")}]]}
    (run_dir / SUBMISSIONS).write_text(json.dumps(log), encoding="utf-8")
    with pytest.raises(AssertionError, match="logged outputs missing from the packed tree"):
        pack_case(run_dir, "lost", dest_dir=tmp_path / "recordings")


# ---------------------------------------------------------------------------
# The harness: a case's scenario, its halts, and what a recording cannot hold
# ---------------------------------------------------------------------------

_TWO_SEEDS = "2\nzero\nH 0 0 0\nH 0.74 0 0\n2\none\nH 0 0 0\nH 0.75 0 0\n"

_FAKE_INPUT = """\
template_dir: ./templates
output_dir: ./outputs
input: ./seeds.xyz
steps:
  - {step: 1, engine: fake, on_failure: stop}
  - {step: 2, engine: fake}
"""


def _fake_case(root: Path, spec: str | None = None) -> Path:
    """A two-step case on the in-memory engine; ``spec`` is its ``case.yaml``, if any."""
    (root / "templates").mkdir(parents=True)
    (root / "seeds.xyz").write_text(_TWO_SEEDS, encoding="utf-8")
    (root / "input.yaml").write_text(_FAKE_INPUT, encoding="utf-8")
    if spec is not None:
        (root / CASE_SPEC).write_text(spec, encoding="utf-8")
    return root


def _losing_structure_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the fake engine's jobs leave no output for structure ``1``: a real MISSING_OUTPUT."""
    from fake_engine import FakeEngine

    submit = FakeEngine.submit

    def losing(self: FakeEngine, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        return submit(self, StepInputs(tuple(t for t in inputs.files if t[2] != "1")), ctx)

    monkeypatch.setattr(FakeEngine, "submit", losing)


def test_a_case_without_a_spec_is_one_run(tmp_path: Path) -> None:
    assert CaseSpec.read(_fake_case(tmp_path)) == CaseSpec((Act(Action.RUN),))


@pytest.mark.parametrize(
    "spec",
    [
        "scenario: [{action: run}]\nbudget: 5\n",
        "scenario: [{action: run, halt_at: 1}]\n",
    ],
    ids=["top-level", "per-action"],
)
def test_a_case_spec_refuses_a_key_it_does_not_know(tmp_path: Path, spec: str) -> None:
    """A misspelled ``halts_at`` read as absent would let a halt pass as a completion."""
    with pytest.raises(AssertionError, match="unknown key"):
        CaseSpec.read(_fake_case(tmp_path, spec))


def test_a_declared_halt_is_held_to_the_tree_it_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 1 halts with its ledger, and step 2 never ran."""
    _losing_structure_one(monkeypatch)
    root = _fake_case(tmp_path, "scenario: [{action: run, halts_at: 1}]\n")
    config = load_config(root / "input.yaml")
    drive(config, CaseSpec.read(root))
    assert [
        r.structure_id for r in cache.load_failure_records(config.step_dir(config.steps[0]))
    ] == ["1"]
    assert not (config.step_dir(config.steps[1]) / "_cache" / "step.json").exists()


@pytest.mark.parametrize(
    ("spec", "loses_one", "error", "match"),
    [
        ("scenario: [{action: run}]\n", True, ChemRefineError, "step 1 halted"),
        ("scenario: [{action: run, halts_at: 1}]\n", False, AssertionError, "and completed"),
        (
            "scenario: [{action: run, halts_at: 2}]\n",
            True,
            AssertionError,
            "declared to halt at step 2",
        ),
    ],
    ids=["undeclared", "never-halts", "another-step"],
)
def test_a_halt_other_than_the_declared_one_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spec: str,
    loses_one: bool,
    error: type[Exception],
    match: str,
) -> None:
    """A halt is declared, never tolerated — and a declared one must happen, where declared."""
    if loses_one:
        _losing_structure_one(monkeypatch)
    root = _fake_case(tmp_path, spec)
    with pytest.raises(error, match=match):
        drive(load_config(root / "input.yaml"), CaseSpec.read(root))


def test_an_artifact_steps_product_is_stubbed_only_where_it_is_missing(tmp_path: Path) -> None:
    """The replay stands in for the model no recording holds, and for nothing else."""
    from ase import Atoms

    from chemrefine.config import StepConfig
    from chemrefine.engines.mlip.train.engine import MlipTrainEngine
    from chemrefine.state import PipelineState, Structure

    ctx = StepContext(
        step_cfg=StepConfig(
            step=2, engine="mlip-train", options={"task_name": "mace_off", "device": "cpu"}
        ),
        step_dir=tmp_path / "step2",
        template_dir=tmp_path,
        template=None,
        scratch_dir=None,
        prev_state=PipelineState(structures=(Structure(id="0", atoms=Atoms("H")),)),
        charge=0,
        multiplicity=1,
        max_cores=1,
        slurm_template="cpu.slurm.header",
    )
    engine = MlipTrainEngine()
    _stub_artifact(engine, ctx)
    assert engine.artifact(ctx).read_bytes() == ARTIFACT_STUB

    engine.artifact(ctx).write_bytes(b"real weights")
    _stub_artifact(engine, ctx)
    assert engine.artifact(ctx).read_bytes() == b"real weights"

    _stub_artifact(get_engine("fake"), ctx)  # not an artifact engine: nothing to stand in for
    assert sorted(p.name for p in engine.artifact(ctx).parent.iterdir()) == ["train.model"]


def test_a_parse_only_rebuild_can_verify_the_steps_before_an_artifact_step(tmp_path: Path) -> None:
    """Every step after a trained model keys its bytes, which no stub reproduces."""
    with_training = extract_case(recording("mlip_train"), tmp_path / "train")
    assert verifiable_prefix(load_config(with_training.config_path)) == 1
    without = extract_case(recording("conformers"), tmp_path / "conformers")
    assert verifiable_prefix(load_config(without.config_path)) == 3
