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
import inspect
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from ase.db import connect
from ase.io import iread
from replay import (
    ABSOLUTE_PATH_RE,
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
    with_backend_python,
)
from test_engines_orca_corpus import CYCLE_LIMIT_RE

from chemrefine import cache, filtering, ids, job_log, nms, pipeline
from chemrefine.config import BoltzmannSample, Config, StepConfig, load_config
from chemrefine.engines.api import NULL_SINK, ArtifactEngine, get_engine
from chemrefine.engines.mlip.train.engine import SIDECAR_NAME
from chemrefine.engines.orca.output import parse_output
from chemrefine.errors import ChemRefineError
from chemrefine.io import read_xyz_frames
from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A, HARTREE_TO_EV, HARTREE_TO_KCALMOL
from chemrefine.recovery import Action
from chemrefine.scaffold import scaffold_templates
from chemrefine.state import FailureKind, JobBatch, JobTriple, StepContext, StepInputs
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


def replay(archive: Path, root: Path, monkeypatch: pytest.MonkeyPatch) -> Replayed:
    """Extract ``archive`` under ``root``, drive its scenario, and read back what it reached."""
    case = extract_case(archive, root)
    config = with_backend_python(load_config(case.config_path))
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


def check_recording(archive: Path, root: Path, monkeypatch: pytest.MonkeyPatch) -> Replayed:
    """Replay ``archive`` under ``root`` and hold it to what every recording and its case must be.

    Every recorded submission is served once, and the tree is whole: ``steps.csv`` reports
    each step's survivors with a finite ranking energy, each step's survivors file holds one
    frame per survivor, every cached structure with an energy has the canonical record of
    the job it came from (the NMS child's, for a promoted winner), and a completed tree
    resumes without submitting anything, to the same survivors. Then the case's own
    :data:`CHECKS`. The live tier runs this on the archive it just packed, before it may
    replace a recording.
    """
    run = replay(archive, root, monkeypatch)

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

    CHECKS[archive.name.removesuffix(".tar.xz")](run)

    if run.spec.halts_at is None:
        again = pipeline.run(run.config)
        assert all(outcome.cache_hit for outcome in again), "a finished tree resumes from cache"
        assert [s.id for s in again[-1].state.structures] == [
            s.id for s in run.outcomes[-1].state.structures
        ]
    return run


@pytest.mark.parametrize("name", recorded_cases())
def test_a_recording_replays_the_run_it_recorded(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each shipped recording, held to :func:`check_recording`."""
    check_recording(recording(name), tmp_path, monkeypatch)


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
        run = replay(recording(name), root, monkeypatch)
        return {
            doc.parent.parent.name: json.loads(doc.read_text(encoding="utf-8"))["fingerprint"]
            for doc in sorted(run.case.output_dir.glob("step*/_cache/step.json"))
        }

    first = fingerprints(tmp_path / "cluster")
    assert first, "the replay cached no step"
    assert first == fingerprints(tmp_path / "laptop")


def test_a_resume_reads_a_resolved_structure_and_leaves_its_inputs_the_winners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ts_path``'s sampled step, its results document gone, resumes around the winner's inputs.

    The step promotes a displaced child under its parent's id, and the child's ``.inp`` and
    ``_inp.xyz`` arrive at the canonical path with its ``.out``. A resume that finds no
    results document adopts the row from the manifest and re-parses it. It prepares nothing
    for it, so the two inputs stay those of the job that wrote that output; prepared again
    they would be round 1's, the saddle the sampling moved away from.
    """
    run = replay(recording("ts_path"), tmp_path, monkeypatch)
    settle = run.config.steps[-1]
    step_dir = run.config.step_dir(settle).resolve()
    (survivor,) = run.outcomes[-1].state.structures
    winner = survivor.resolved_from
    assert winner is not None, "precondition: the step promoted a child"
    home = step_dir / survivor.id
    attempt = ids.latest_attempt_dir(home)
    assert attempt is not None
    stem = ids.structure_stem(settle.step, survivor.id)
    child_stem = ids.structure_stem(settle.step, winner)
    promoted = {
        f"{stem}{suffix}": (attempt / winner / f"{child_stem}{suffix}").read_bytes()
        for suffix in (".inp", "_inp.xyz")
    }
    assert {name: (home / name).read_bytes() for name in promoted} == promoted
    assert all((attempt / name).read_bytes() != data for name, data in promoted.items()), (
        "precondition: round 1's inputs, sealed in the attempt, are other files"
    )

    cache.invalidate(step_dir)
    outcomes = pipeline.run(run.config)  # the replay left every submission forbidden

    assert not outcomes[-1].cache_hit, "precondition: the step is read back row by row"
    assert [s.resolved_from for s in outcomes[-1].state.structures] == [winner]
    assert {name: (home / name).read_bytes() for name in promoted} == promoted


def test_every_case_has_one_recording_and_one_check() -> None:
    """A case without a recording is never replayed, and a recording without a check proves
    only that it runs. Each is one directory, one archive and one entry here; a live check
    names a case, or it never runs."""
    from test_e2e_live import LIVE_CHECKS

    cases = sorted(p.name for p in CASES_DIR.iterdir() if (p / "input.yaml").is_file())
    assert cases == recorded_cases() == sorted(CHECKS)
    assert set(LIVE_CHECKS) <= set(cases), "a live check for a case that does not exist"


NOT_LIVE: dict[str, str] = {
    "engine fake": "the suite's in-memory engine, not a shipped one",
    "engine qchem": "no Q-Chem binary on the machines that record",
    "trains mlip-fairchem": "a FAIRChem fit needs about 7.3 GB, past what a recording machine has",
}
"""What no live case covers, each with the reason — an engine, a library run directly, a
library fitted. Everything else registered must be covered, and an entry here that a case
now covers, or that names nothing registered, is stale."""


def _required() -> set[str]:
    """Every engine, every MLIP library run directly, and every trainer registered."""
    from chemrefine.engines.api import ENGINES
    from chemrefine.engines.mlip.registry import (
        backend_spec,
        registered_backends,
        registered_trainers,
    )

    return (
        {f"engine {name}" for name in ENGINES}
        | {f"runs {backend_spec(task).extra}" for task in registered_backends()}
        | {f"trains {backend_spec(task).extra}" for task in registered_trainers()}
    )


def _covered() -> set[str]:
    """What the case directories run: each step's engine, and the library it runs or fits."""
    from chemrefine.engines.mlip.engine import MlipEngine
    from chemrefine.engines.mlip.train.engine import MlipTrainEngine

    covered = set()
    for case in sorted(CASES_DIR.iterdir()):
        if not (case / "input.yaml").is_file():
            continue
        for step in load_config(case / "input.yaml").steps:
            engine = get_engine(step.engine)
            covered.add(f"engine {step.engine}")
            if isinstance(engine, MlipEngine | MlipTrainEngine):
                verb = "trains" if isinstance(engine, MlipTrainEngine) else "runs"
                covered.add(f"{verb} {engine.backend_requirement(step.options or {}).extra}")
    return covered


def gaps(
    required: set[str], covered: set[str], exclusions: dict[str, str]
) -> tuple[list[str], list[str]]:
    """``(uncovered, stale)``: what nothing covers or excuses, and excuses nothing needs."""
    uncovered = sorted(required - covered - set(exclusions))
    stale = sorted((set(exclusions) & covered) | (set(exclusions) - required))
    return uncovered, stale


def test_every_engine_library_and_trainer_runs_live_in_some_case() -> None:
    """A plugin joins the live tier with a case of its own, or with a reason it cannot.

    The tier is lean by covering shared behaviour once, on the cheapest engine, and each
    engine, MLIP library and trainer only at its own seam. That only holds while every one
    of them has that case: a new engine or library with none would ship untried against its
    real program.
    """
    uncovered, stale = gaps(_required(), _covered(), NOT_LIVE)
    assert not uncovered, f"no live case covers {uncovered}: add one, or excuse it in NOT_LIVE"
    assert not stale, f"NOT_LIVE excuses {stale}, which a case covers or nothing registers"


def test_the_submission_guard_takes_run_batchs_own_arguments() -> None:
    """A stray submission is reported as one, whichever way the engine submits it.

    Job engines call ``run_batch(..., sink=sink)``; a guard without the keyword reported a
    cache-only run's submission as a ``TypeError`` about itself.
    """
    from chemrefine.engines import _execution

    assert list(inspect.signature(forbid_run_batch).parameters) == list(
        inspect.signature(_execution.run_batch).parameters
    )
    inputs = StepInputs(files=((Path("0.inp"), Path("0.out"), "0"),))
    with pytest.raises(AssertionError, match=r"unexpected submission of \['0'\]"):
        forbid_run_batch(object(), inputs, cast(StepContext, None), sink=NULL_SINK)


def test_an_exclusion_nothing_needs_is_stale() -> None:
    required = {"engine a", "engine b", "runs x"}
    assert gaps(required, {"engine a"}, {"runs x": "why"}) == (["engine b"], [])
    assert gaps(required, {"engine a", "engine b", "runs x"}, {"runs x": "why"}) == (
        [],
        ["runs x"],
    )
    assert gaps(required, {"engine a", "engine b", "runs x"}, {"runs y": "why"}) == (
        [],
        ["runs y"],
    )


# ---------------------------------------------------------------------------
# What each case exists for
# ---------------------------------------------------------------------------


def _conformers(run: Replayed) -> None:
    """Formic acid: a lean GOAT, a Boltzmann cut that drops E, a seeded kick, a cation.

    GOAT runs one worker at one temperature for one global iteration and still finds both
    planar conformers, with E about 7.1 kcal/mol above Z. The 99% cut, recomputed here from
    the ensemble ORCA wrote, keeps Z alone. The template's ``nprocs 8`` reaches ORCA as
    ``max_cores``. The kick is ``random`` NMS on a minimum: seed 7 draws one of the nine
    vibrations the parse names, and the two children start ±0.3 along it from the round-1
    geometry and relax back to Z. Both are refined as a cation doublet: ORCA echoes UHF,
    the charge, the multiplicity and one electron fewer, and each energy sits a vertical
    ionisation above the neutral. The children end 3e-8 Eh apart at xTB and 7e-5 Eh apart
    at HF, so which one survives is a near-tie. The check never names it, only that it is
    the lower.
    """
    case, config, served = run.case, run.config, run.submitter.served
    goat, kick, ionise = config.steps

    # Step 1: the lean settings reached GOAT, and so did the clamped core count.
    goat_out = case.submitted_file(f"{goat.dir_name()}/0", 0, "step1_0.out")
    goat_text = goat_out.read_text(encoding="utf-8")
    # One worker: without NWORKERS and TEMPLIST, ORCA starts four base workers, one per
    # default temperature, and splits each in two: 8 workers and 40 s.
    assert re.findall(r"Final number of workers\s+\.{3}\s+(\d+)", goat_text) == ["1"]
    assert "Reached the maximum number of global iterations (1)!" in goat_text
    template = (config.template_dir / "step1.inp").read_text(encoding="utf-8")
    goat_dir = config.step_dir(goat).resolve()
    rendered = (goat_dir / "0" / "step1_0.inp").read_text(encoding="utf-8")
    [asked] = re.findall(r"nprocs\s+(\d+)", template)
    assert int(asked) > config.max_cores, "a template within the budget leaves nothing to clamp"
    assert re.findall(r"nprocs\s+(\d+)", rendered) == [str(config.max_cores)], "clamped"
    assert re.findall(r"Number of available CPUs\s+\.{3}\s+(\d+)", goat_text) == [
        str(config.max_cores)
    ], "GOAT ran on the clamped count"
    assert "* xyzfile 0 1 " in rendered, "the workflow itself is the neutral singlet"

    # The cut, recomputed apart from the filter it checks: the constants are written out.
    ensemble = parse_output(goat_out, "goat")
    assert len(ensemble) == 2, "formic acid's two planar conformers, Z and E"
    energies = sorted(frame.energy_hartree for frame in ensemble)
    above = [(e - energies[0]) * 627.5094740631 for e in energies]
    assert 6.99 < above[1] < 7.19, "E-formic acid lies 7.09 kcal/mol above Z at GFN2-xTB"
    sample = goat.sample
    assert isinstance(sample, BoltzmannSample)
    weights = [math.exp(-kcal / (1.987204259e-3 * sample.temperature_k)) for kcal in above]
    kept: list[float] = []
    cumulative = 0.0
    for energy, weight in zip(energies, weights, strict=True):
        kept.append(energy)
        cumulative += 100 * weight / sum(weights)
        if cumulative >= sample.percent_cumulative:
            break
    assert len(kept) < len(ensemble), "a cut that keeps every conformer proves nothing"
    survivors = run.outcomes[0].state.structures
    assert [s.energy_hartree for s in survivors] == kept, "the filter keeps what 99% keeps"
    cached = cache.load(goat_dir)
    assert cached is not None
    assert sorted(s.id for s in cached.results.structures) == [
        f"0-{i}" for i in range(len(ensemble))
    ], "one id per GOAT frame, under the seed's"
    (parent,) = survivors
    assert parent.id == "0-0", "the draw pinned below is a function of this id"
    # O=C-O-H over the seed's atom order (C, O, O, H on C, H on O): 0 deg is Z, 180 deg E.
    (dropped,) = [s for s in cached.results.structures if s.id != parent.id]
    assert math.cos(math.radians(parent.atoms.get_dihedral(1, 0, 2, 4))) > 0.99, "Z is kept"
    assert math.cos(math.radians(dropped.atoms.get_dihedral(1, 0, 2, 4))) < -0.99, "E is cut"

    # Step 2: seed 7 draws one of the vibrations the parse names, and the children start there.
    kick_rel = f"{kick.dir_name()}/{parent.id}"
    (round_one,) = parse_output(
        case.submitted_file(kick_rel, 0, f"step2_{parent.id}.out"), "opt_sp"
    )
    assert round_one.imaginary_freqs == {}, "Z is a minimum: the kick explores, it repairs nothing"
    vibrations, modes = round_one.frequencies, round_one.normal_modes
    assert vibrations is not None and modes is not None
    assert sorted(vibrations) == list(range(6, 3 * len(parent.atoms))), (
        "a non-linear molecule's 3N-6 vibrations, past its six translations and rotations"
    )
    # rng_for("0-0", 7) over modes 6-14 draws mode 9. The draw is a pure function of the seed
    # and the id, so a rerun of this config kicks along the same mode.
    assert kick.options["seed"] == 7, "the draw pinned below is seed 7's"
    k = 9
    pair = {f"{parent.id}_m{k}_pos": 1.0, f"{parent.id}_m{k}_neg": -1.0}
    children = run.outcomes[1].state.structures
    assert sorted(c.id for c in children) == sorted(pair)
    assert {rel: n for rel, n in served.items() if rel.startswith(f"{kick.dir_name()}/")} == {
        kick_rel: 1,
        **{f"{kick_rel}/attempt1/{cid}": 1 for cid in pair},
    }, "round 1 once, then the pair once each in attempt1/, nothing retried"
    amplitude = float(kick.options["displacement_value"])
    attempt = config.step_dir(kick).resolve() / parent.id / "attempt1"
    for child in children:
        (start,) = read_xyz_frames(attempt / child.id / f"step2_{child.id}_inp.xyz")
        displaced = round_one.positions + pair[child.id] * amplitude * modes[:, :, k]
        assert np.allclose(start.positions, displaced, atol=1e-5), (
            f"{child.id} must start {amplitude} along mode {k} from the round-1 geometry"
        )
        assert child.parent_id == parent.id
        assert child.converged and child.imaginary_freqs == {}, "each child relaxes to a minimum"
        assert child.resolved_from is None, "random fans out: nothing is promoted"
        assert child.energy_hartree is not None
        # The kicked starts sit ~0.02 Eh above Z, and E 0.011 Eh: both come back to Z.
        assert abs(child.energy_hartree - round_one.energy_hartree) < 1e-5, "back in Z's basin"

    # Step 3: the cation doublet reached ORCA, one ionisation above the neutral.
    charge, multiplicity = 1, 2
    assert (ionise.charge, ionise.multiplicity) == (charge, multiplicity)
    electrons = int(parent.atoms.numbers.sum()) - charge
    neutral = -186.2115  # RHF/STO-3G, the neutral singlet at the pos child's geometry
    refined: dict[str, float] = {}
    for cid in pair:
        rel = f"{ionise.dir_name()}/{cid}"
        assert served[rel] == 1, "both children are refined"
        inp = config.step_dir(ionise).resolve() / cid / f"step3_{cid}.inp"
        assert f"* xyzfile {charge} {multiplicity} " in inp.read_text(encoding="utf-8")
        out = case.submitted_file(rel, 0, f"step3_{cid}.out")
        text = out.read_text(encoding="utf-8")
        echoed = {
            key: re.findall(rf"\b{key}\s+\.{{4}}\s+(\S+)", text)
            for key in ("HFTyp", "Charge", "Mult", "NEL")
        }
        assert echoed == {
            "HFTyp": ["UHF"],
            "Charge": [str(charge)],
            "Mult": [str(multiplicity)],
            "NEL": [str(electrons)],
        }, "an open shell: ORCA picks UHF from the multiplicity the step gave it"
        (single_point,) = parse_output(out, "sp")
        assert 0.2 < single_point.energy_hartree - neutral < 0.6, "a charge ORCA never saw gives 0"
        refined[cid] = single_point.energy_hartree
    (lowest,) = run.outcomes[2].state.structures
    assert lowest.energy_hartree == min(refined.values()), "the lower; which child is a near-tie"


def _nms_recovery(run: Replayed) -> None:
    """Every way a ``ts`` resolution ends, and recovery re-attempting exactly what is pending.

    Planar NH3 is already a TS and passes through. Pyramidal NH3 has no imaginary mode to
    keep and is unresolved without a round 2. Linear H2O keeps one of its two bends and is
    displaced along the other; its children relax to bent water, never a TS, so it is
    unresolved after one. Which bend is displaced is decided by a tie, so it is read from
    the tree, never named. FrH makes ORCA abort: ledgered as a run that did not terminate
    normally, its reason naming the output relative to the step and quoting ORCA's stderr.
    The step halts on the three, and ``resume`` and ``rerun-errors`` each re-run the
    displaced pair, in an attempt of its own, and the abort, sealing the one before —
    nothing else — and halt again. Every job ran in a scratch directory of its own under
    ``scratch_dir``, as ORCA itself reports.
    """
    case, served = run.case, run.submitter.served
    assert [(act.action, act.target, act.halts_at) for act in run.spec.scenario] == [
        (Action.RUN, None, 1),
        (Action.RESUME, None, 1),
        (Action.RERUN_ERRORS, 1, 1),
    ], "a halted run, then both recovery actions over it, each declared to halt again"
    actions = len(run.spec.scenario)
    step_dir = run.config.step_dir(run.config.steps[0]).resolve()

    imaginary: dict[str, dict[int, float]] = {}
    for sid in ("0", "1", "2"):
        out = case.submitted_file(f"step1/{sid}", 0, f"step1_{sid}.out")
        freqs = parse_output(out, "opt_sp")[0].imaginary_freqs
        assert freqs is not None, f"frame {sid}: round 1 has no frequency table"
        imaginary[sid] = freqs
    assert [len(imaginary[sid]) for sid in ("0", "1", "2")] == [1, 0, 2], (
        "the seeds must be a TS, a minimum and a second-order saddle"
    )
    runs = [served[f"step1/{sid}"] for sid in ("0", "1", "2", "3")]
    assert runs == [1, 1, 1, actions], "round 1 is reused; the abort re-runs every time"
    for sid in ("0", "1"):
        assert not list((step_dir / sid).glob("attempt*")), f"frame {sid} has no round 2"

    rounds: dict[str, set[str]] = {}
    for rel, count in served.items():
        if rel.startswith("step1/2/attempt"):
            attempt, child = rel.removeprefix("step1/2/").split("/")
            assert count == 1, f"{rel} ran {count} times"
            rounds.setdefault(attempt, set()).add(child)
    assert sorted(rounds) == [f"attempt{k}" for k in range(1, actions + 1)], (
        "each action runs frame 2's round 2 in an attempt of its own"
    )
    pairs = {frozenset(children) for children in rounds.values()}
    assert len(pairs) == 1, f"every action displaces the same mode: {rounds}"
    (pair,) = pairs
    stems = {child.rpartition("_")[0] for child in pair}
    assert len(stems) == 1, pair
    (stem,) = stems
    assert pair == {f"{stem}_pos", f"{stem}_neg"}, pair
    mode = stem.removeprefix("2_m")
    assert mode.isdigit() and int(mode) in imaginary["2"], "one bend displaced, the other kept"
    for attempt, children in rounds.items():
        for child in children:
            out = case.submitted_file(f"step1/2/{attempt}/{child}", 0, f"step1_{child}.out")
            parsed = parse_output(out, "opt_sp")[0]
            assert parsed.terminated_normally and parsed.imaginary_freqs == {}, (
                f"{attempt}/{child} relaxes to bent water, a minimum"
            )

    ledger = {r.structure_id: r for r in cache.load_failure_records(step_dir)}
    assert {sid: r.kind for sid, r in ledger.items()} == {
        "1": FailureKind.UNRESOLVED_NMS,
        "2": FailureKind.UNRESOLVED_NMS,
        "3": FailureKind.NOT_TERMINATED_NORMALLY,
    }
    reason = ledger["3"].reason
    err = step_dir / "3" / "step1_3.err"
    assert err.is_file(), "the abort's stderr is recorded, so a replay can quote it"
    tail = [line.strip() for line in err.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert tail, "ORCA says on stderr why it aborted"
    assert " in 3/step1_3.out" in reason, f"named relative to its step: {reason}"
    assert f"{err.name} ends: " in reason and reason.endswith(tail[-1]), reason
    assert not ABSOLUTE_PATH_RE.search(reason), f"names the machine: {reason}"
    sealed = sorted(p.name for p in (step_dir / "3").glob("attempt*"))
    assert sealed == [f"attempt{k}" for k in range(1, actions)], sealed
    held = [step_dir / "3" / name / "step1_3.out" for name in sealed]
    held.append(step_dir / "3" / "step1_3.out")
    assert [p.read_bytes() if p.is_file() else None for p in held] == [
        case.submitted_file("step1/3", k, "step1_3.out").read_bytes() for k in range(actions)
    ], "each re-run seals the abort before it, in order, rather than overwriting it"

    (survivor,) = run.outcomes[0].state.structures
    assert survivor.id == "0" and survivor.resolved_from is None, "the TS passes through"
    assert survivor.imaginary_freqs == imaginary["0"]

    scratch = run.config.scratch_dir
    assert scratch is not None
    places: list[Path] = []
    for out in sorted(case.captured.rglob("*.out")):
        said = [ln for ln in out.read_text(encoding="utf-8").splitlines() if "Working dir.:" in ln]
        assert len(said) == 1, f"{out.name}: ORCA names the directory it ran in, once"
        places.append(Path(said[0].partition("Working dir.:")[2].strip()))
    assert len(places) == sum(served.values()), "every submission's output, the aborts too"
    strays = [
        p for p in places if p.parent.name != scratch.name or not p.name.startswith("ChemRefine_")
    ]
    assert not strays, f"ran outside scratch_dir: {strays}"
    assert len(set(places)) == len(places), "each job in a scratch directory of its own"


def _triage(run: Replayed) -> None:
    """What ``best`` and ``skip`` do with real failures, and a ranking by Gibbs energy.

    Step 1 (``best``, ``MaxIter 2``, no sample) meets both kinds of failure. The distorted
    water runs out of cycles, as ORCA itself says, far from converged. It is retried once
    from the last geometry it reached and runs out again, still far off, so ``best`` keeps
    the retry's last cycle, unconverged. FrH makes ORCA abort at its input check, since xTB
    stops at radon. An abort is not retried: its reason names the output relative to the
    step and quotes ORCA's stderr, and ``best`` carries the input it was submitted with
    forward, as a geometry alone. Step 2 (``skip``) runs FrH again and drops the abort. It
    ranks the three minima by Gibbs energy, and the one it keeps is not the electronically
    lowest. Which one that is gets recomputed from the archive, never named: the case holds
    the two orders apart, by margins well clear of a tie.
    """
    case, served = run.case, run.submitter.served
    budget, rank = run.reached
    step_dirs = [run.config.step_dir(step).resolve() for step in run.reached]
    seeds = read_xyz_frames(case.root / "triage.xyz")

    # Frame 2: out of cycles twice, and nowhere near converged either time.
    water = f"{budget.dir_name()}/2"
    assert served[water] == 2, "a run out of cycles is retried once"
    outs = [case.submitted_file(water, k, "step1_2.out") for k in (0, 1)]
    for out in outs:
        text = out.read_text(encoding="utf-8")
        assert CYCLE_LIMIT_RE.search(text), "ORCA's own words: out of cycles"
        rows = re.findall(r"^\s*RMS gradient\s+(\S+)\s+(\S+)\s+(?:YES|NO)\s*$", text, re.M)
        assert len(rows) == 2, f"{out.name}: MaxIter 2 is two cycles, each judged"
        rms, tolerance = map(float, rows[-1])
        assert rms >= 100 * tolerance, f"stopped {rms / tolerance:.0f}x its tolerance: not definite"
    (stopped,) = parse_output(outs[0], "opt_sp")
    (retried,) = parse_output(outs[1], "opt_sp")
    for attempt in (stopped, retried):
        assert attempt.terminated_normally is True and attempt.converged is False
    (retry_input,) = read_xyz_frames(case.output_dir / water / "step1_2_inp.xyz")
    assert np.allclose(retry_input.positions, stopped.positions), "retried from its last cycle"
    assert not np.allclose(retry_input.positions, seeds[2].positions, atol=1e-3)
    assert not np.allclose(retried.positions, retry_input.positions, atol=1e-3), "and moved on"

    # Step 1 under `best`: every structure kept, each failure as far as it got.
    kept = {s.id: s for s in run.outcomes[0].state.structures}
    assert list(kept) == ["0", "1", "2", "3"], "best keeps every structure"
    assert kept["2"].converged is False
    assert kept["2"].energy_hartree == retried.energy_hartree, "the retry's last cycle"
    assert np.allclose(kept["2"].atoms.positions, retried.positions)
    assert kept["3"].energy_hartree is None, "an abort has no energy to carry"
    assert kept["3"].terminated_normally is None, "the input as a geometry, not a parsed run"
    assert np.allclose(kept["3"].atoms.positions, seeds[3].positions), "its input, carried"
    for sid, carried in (("2", retried.positions), ("3", seeds[3].positions)):
        (onward,) = read_xyz_frames(step_dirs[1] / sid / f"step2_{sid}_inp.xyz")
        assert np.allclose(onward.positions, carried), f"step 2 runs what best kept of {sid}"

    # The abort, in each step: not retried, and its reason portable and ORCA's own.
    ledgers = [{r.structure_id: r for r in cache.load_failure_records(d)} for d in step_dirs]
    assert {sid: r.kind for sid, r in ledgers[0].items()} == {
        "2": FailureKind.NOT_CONVERGED,
        "3": FailureKind.NOT_TERMINATED_NORMALLY,
    }
    assert {sid: r.kind for sid, r in ledgers[1].items()} == {
        "3": FailureKind.NOT_TERMINATED_NORMALLY
    }
    for step, step_dir, ledger in zip(run.reached, step_dirs, ledgers, strict=True):
        home, stem = f"{step.dir_name()}/3", f"step{step.step}_3"
        assert served[home] == 1, "an abort would abort again: it is not retried"
        out_text = case.submitted_file(home, 0, f"{stem}.out").read_text(encoding="utf-8")
        assert "Z>86" in out_text, "ORCA refused the element: xTB stops at radon"
        err = step_dir / "3" / f"{stem}.err"
        assert err.is_file(), "the abort's stderr is recorded, so a replay can quote it"
        tail = [ln.strip() for ln in err.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert tail, "ORCA says on stderr why it aborted"
        reason = ledger["3"].reason
        assert f" in 3/{stem}.out" in reason, f"named relative to its step: {reason}"
        assert f"{err.name} ends: " in reason and reason.endswith(tail[-1]), reason
        assert not ABSOLUTE_PATH_RE.search(reason), f"names the machine: {reason}"

    # Step 2: skip drops the abort, the rest are minima, and Gibbs picks the survivor.
    cached = cache.load(step_dirs[1])
    assert cached is not None
    minima = cached.results.structures
    assert [s.id for s in minima] == ["0", "1", "2"], "skip drops the abort; best would keep it"
    for s in minima:
        assert s.converged and s.imaginary_freqs == {}, f"{s.id}: a minimum"
        assert all(_finite(v) for v in (s.gibbs_hartree, s.enthalpy_hartree, s.energy_zpe_hartree))
    assert not list(step_dirs[1].glob("*/attempt*")), "a minimum under a minimum target: no round 2"

    energy: dict[str, float] = {}
    gibbs: dict[str, float] = {}
    for s in minima:
        out = case.submitted_file(f"{rank.dir_name()}/{s.id}", 0, f"step2_{s.id}.out")
        (result,) = parse_output(out, "opt_sp")
        assert result.gibbs_hartree is not None
        energy[s.id], gibbs[s.id] = result.energy_hartree, result.gibbs_hartree
    by_energy = sorted(energy, key=energy.__getitem__)
    by_gibbs = sorted(gibbs, key=gibbs.__getitem__)
    assert by_gibbs[0] != by_energy[0], "the case exists for a Gibbs order that is not E's"
    for values, order in ((energy, by_energy), (gibbs, by_gibbs)):
        gap = values[order[1]] - values[order[0]]
        assert gap >= 1e-4, f"the lowest two are {gap:.1e} Eh apart: a tie, not an order"
    (survivor,) = run.outcomes[1].state.structures
    assert survivor.id == by_gibbs[0], "min by gibbs keeps the lowest free energy"
    with (case.output_dir / "steps.csv").open(encoding="utf-8", newline="") as fh:
        (row,) = [r for r in csv.DictReader(fh) if r["Step"] == str(rank.step)]
    assert row["Energy type"] == "gibbs"
    assert math.isclose(float(row["Energy (Hartree)"]), gibbs[survivor.id], abs_tol=1e-8)


def _ts_path(run: Replayed) -> None:
    """HCN to HNC over its barrier: a scan up to it, the saddle, and down both sides.

    No step names its operation or its NMS target; each template implies them. The relaxed
    scan of the H-C-N angle peaks inside its range, so ``max`` keeps the barrier rather than
    an end of the scan. ``OptTS`` implies the ``ts`` target, and the saddle it finds, with
    one imaginary mode, passes through with nothing to displace. ``Opt`` implies
    ``minimum``: started on that saddle the optimiser stays there, so NMS displaces along
    the imaginary mode both ways, and the two children fall to the two isomers. ``MaxIter
    30`` cuts a child short. ORCA says it ran out of cycles, the output reads as terminated
    but not converged, and the retry from its last geometry converges. Which child that is
    depends on the limit, so none is named. The lower child, HCN, about 20 kcal/mol under
    HNC, is promoted under the parent's id. ORCA recognizes it as linear, so five of its
    modes are translations and rotations, not six, and its table keeps both halves of the
    degenerate bend.
    """
    case, config, served = run.case, run.config, run.submitter.served
    assert all(s.operation is None and "target" not in s.options for s in config.steps), (
        "the templates imply every operation and target; a spelled-out one infers nothing"
    )
    scan, saddle, settle = config.steps
    s1, s2, s3 = scan.dir_name(), saddle.dir_name(), settle.dir_name()
    step3_dir = config.step_dir(settle).resolve()

    # Step 1: a relaxed scan whose highest point lies inside it, clear of the others.
    points = parse_output(case.submitted_file(f"{s1}/0", 0, "step1_0.out"), "pes")
    energies = [p.energy_hartree for p in points]
    top = int(np.argmax(energies))
    assert 0 < top < len(points) - 1, "a scan's end as its maximum makes `max` prove nothing"
    runner_up = max(e for i, e in enumerate(energies) if i != top)
    assert (energies[top] - runner_up) * HARTREE_TO_KCALMOL > 1.0, "no near-tie picks the top"
    scanned = cache.load(config.step_dir(scan).resolve())
    assert scanned is not None
    assert [s.id for s in scanned.results.structures] == [f"0-{i}" for i in range(len(points))]
    (barrier,) = run.outcomes[0].state.structures
    assert barrier.id == f"0-{top}" and barrier.energy_hartree == energies[top]

    # Step 2: OptTS implies `ts`, and a saddle with one imaginary mode already is one.
    (ts,) = run.outcomes[1].state.structures
    assert ts.id == barrier.id and ts.resolved_from is None
    assert ts.imaginary_freqs is not None and len(ts.imaginary_freqs) == 1
    assert {rel: n for rel, n in served.items() if rel.startswith(f"{s2}/")} == {
        f"{s2}/{ts.id}": 1
    }, "a TS under a ts target is run once, with no child to displace"

    # Step 3: Opt implies `minimum`; round 1 stays on the saddle, so NMS displaces.
    home = f"{s3}/{ts.id}"
    prefix = f"{home}/attempt1/"
    attempt = step3_dir / ts.id / "attempt1"
    assert served[home] == 1
    (round_one,) = parse_output(case.submitted_file(home, 0, f"step3_{ts.id}.out"), "opt_sp")
    assert round_one.imaginary_freqs is not None and len(round_one.imaginary_freqs) == 1
    (mode,) = round_one.imaginary_freqs
    children = sorted(rel.removeprefix(prefix) for rel in served if rel.startswith(prefix))
    assert children == [f"{ts.id}_m{mode}_neg", f"{ts.id}_m{mode}_pos"], (
        "a minimum target displaces both ways along the one imaginary mode"
    )

    # A child out of cycles, read as not converged, retried from where it stopped.
    retried = [child for child in children if served[prefix + child] == 2]
    assert retried, "MaxIter 30 must leave a child short of convergence"
    for child in retried:
        stopped_out = case.submitted_file(prefix + child, 0, f"step3_{child}.out")
        assert CYCLE_LIMIT_RE.search(stopped_out.read_text(encoding="utf-8")), (
            "ORCA's own notice that it ran out of cycles"
        )
        (stopped,) = parse_output(stopped_out, "opt_sp")
        assert stopped.terminated_normally is True and stopped.converged is False
        (retry_input,) = read_xyz_frames(attempt / child / f"step3_{child}_inp.xyz")
        (displaced,) = read_xyz_frames(attempt / child / "attempt1" / f"step3_{child}_inp.xyz")
        assert np.allclose(retry_input.positions, stopped.positions), "retried from its last cycle"
        assert not np.allclose(retry_input.positions, displaced.positions, atol=1e-3)

    # Every child, the retried one too, ends at a minimum: one isomer on each side. Its table
    # leaves out what ORCA counts as translations and rotations, five when ORCA says the
    # molecule is linear and six when it does not, whatever its geometry.
    linear = "The molecule is recognized as being linear"
    settled: dict[str, float] = {}
    isomers: dict[str, str] = {}
    for child in children:
        final_out = attempt / child / f"step3_{child}.out"
        (final,) = parse_output(final_out, "opt_sp")
        assert final.converged is True and final.imaginary_freqs == {}, f"{child}: a minimum"
        trivial = 5 if linear in final_out.read_text(encoding="utf-8") else 6
        assert final.frequencies is not None
        assert len(final.frequencies) == 3 * len(final.symbols) - trivial, (
            f"{child}: the vibrations are the modes ORCA does not call trivial"
        )
        settled[child] = final.energy_hartree
        at = dict(zip(final.symbols, final.positions, strict=True))
        on_carbon = np.linalg.norm(at["H"] - at["C"]) < np.linalg.norm(at["H"] - at["N"])
        isomers[child] = "HCN" if on_carbon else "HNC"
    assert sorted(isomers.values()) == ["HCN", "HNC"], "the saddle's two sides are the isomers"

    # The lower child is promoted under the parent's id, by a margin no near-tie decides.
    winner = min(settled, key=settled.__getitem__)
    gap = (max(settled.values()) - settled[winner]) * HARTREE_TO_KCALMOL
    assert gap > 10.0, "HNC lies about 20 kcal/mol above HCN"
    assert isomers[winner] == "HCN"
    (survivor,) = run.outcomes[2].state.structures
    assert survivor.id == ts.id and survivor.resolved_from == winner
    assert survivor.energy_hartree == settled[winner]
    assert survivor.converged and survivor.imaginary_freqs == {}
    canonical = (step3_dir / ts.id / f"step3_{ts.id}.out").read_text(encoding="utf-8")
    assert canonical == (attempt / winner / f"step3_{winner}.out").read_text(encoding="utf-8")
    assert (attempt / f"step3_{ts.id}.out").read_text(encoding="utf-8") != canonical, (
        "round 1 is sealed into the attempt, not overwritten"
    )
    # The attempt's record names the winner and the criterion that judged it: the target
    # `Opt` implies, which no step spells. The recording holds the same record, so a
    # relocated tree rebuilds to the winner the run promoted.
    resolution = json.loads((attempt / "resolution.json").read_text(encoding="utf-8"))
    assert resolution == {
        "resolved_from": winner,
        "criterion_key": nms.NmsOptions(target="minimum").criterion_key(),
    }
    recorded = case.captured / s3 / ts.id / "attempt1" / "resolution.json"
    assert json.loads(recorded.read_text(encoding="utf-8")) == resolution, (
        "the recording's record is not the one a run writes: regenerate the recording"
    )
    xyz = (step3_dir / ts.id / f"step3_{ts.id}.xyz").read_text(encoding="utf-8")
    assert xyz.splitlines()[1] == f"NMS-resolved {ts.id}"

    # ORCA says after the table that HCN is linear: indices 0-4 are its translations and
    # rotations, so index 5 is a vibration, the bend's second half, level with the first.
    assert linear in canonical
    frequencies = survivor.frequencies
    assert frequencies is not None
    assert len(frequencies) == 3 * len(survivor.atoms) - 5, "HCN keeps its bend"
    assert abs(frequencies[5] - frequencies[6]) < 1.0, "the two halves of a degenerate bend"


def _host_guest(run: Replayed) -> None:
    """Chloride docked onto water, then each pose microsolvated: an ensemble inside an ensemble.

    Neither step names its operation, and no ``!`` keyword names the run kind: the
    ``%DOCKER`` and ``%solvator`` blocks alone make ORCA dock and solvate, and each step
    infers its reader from its block. The docking template names its guest relative to
    itself; the job is handed the file's absolute path, and ORCA reads it there.
    ``DOCKLEVEL QUICK`` keeps exactly two poses, 0.0025 kcal/mol apart, so neither is named.
    The complex carries the guest's charge, and step 2 runs each pose at it, -1, as ORCA
    itself reports. SOLVATOR adds one water at a time around its pose, held fixed, and
    writes a frame per water with the energy ORCA tabulates for it. Those frames fan out
    under the pose that made them, so the ids nest, and ``by_parent`` keeps the lowest
    frame of each pose: two, where the same filter over the whole step keeps one.
    """
    case, config = run.case, run.config
    dock, solvate = config.steps
    step1_dir, step2_dir = (config.step_dir(step).resolve() for step in config.steps)

    # The blocks alone imply both run kinds.
    templates = {
        step.step: (config.template_dir / ids.default_template_name(step.step, "inp")).read_text(
            encoding="utf-8"
        )
        for step in config.steps
    }
    for step, block in ((dock, "%docker"), (solvate, "%solvator")):
        text = templates[step.step]
        keywords = {
            word.lower()
            for line in text.splitlines()
            if line.startswith("!")
            for word in line[1:].split()
        }
        assert step.operation is None, f"step {step.step}: a spelled-out operation infers nothing"
        assert block in text.lower(), f"step {step.step}: no {block} block"
        assert not keywords & {"docker", "solvator"}, (
            f"step {step.step}: the {block} block alone must imply the run kind"
        )

    # Step 1: the guest, named beside the template, is handed to ORCA where it is.
    assert 'GUEST "cl.xyz"' in templates[dock.step], "named relative to its template"
    guest = (config.template_dir / "cl.xyz").resolve()
    rendered = (step1_dir / "0" / "step1_0.inp").read_text(encoding="utf-8")
    assert f'GUEST "{guest}"' in rendered, "pinned to the file, since ORCA runs elsewhere"
    assert "* xyzfile 0 1 " in rendered, "the host alone is the workflow's neutral singlet"
    dock_out = case.submitted_file(f"{dock.dir_name()}/0", 0, "step1_0.out")
    dock_text = dock_out.read_text(encoding="utf-8")
    read_from = re.search(r"^Reading guests from file\s+(\S.*?)\s*$", dock_text, re.MULTILINE)
    assert read_from is not None, "ORCA read no guest file"
    read_path = Path(read_from[1])
    assert read_path.is_absolute(), f"ORCA was handed a relative guest: {read_path}"
    assert read_path.name == guest.name
    assert re.search(r"^Docking level\s+quick\s*$", dock_text, re.MULTILINE)
    final = re.search(
        r"^Final charge and multiplicity\s+\.+\s+\((-?\d+)\s*,\s*(\d+)\)", dock_text, re.MULTILINE
    )
    assert final is not None, "DOCKER reports the complex's charge"
    charge, multiplicity = int(final[1]), int(final[2])
    assert (charge, multiplicity) == (-1, 1), "a neutral water and a chloride"
    assert solvate.charge == charge, "step 2 runs the complex at the charge DOCKER gives it"

    # Two poses, both kept, each a child of the seed.
    poses = parse_output(dock_out, "docker")
    assert len(poses) == 2, "DOCKLEVEL QUICK optimises exactly two poses"
    pose_ids = [f"0-{i}" for i in range(len(poses))]
    docked = cache.load(step1_dir)
    assert docked is not None
    assert [(s.id, s.parent_id, s.energy_hartree) for s in docked.results.structures] == [
        (sid, "0", pose.energy_hartree) for sid, pose in zip(pose_ids, poses, strict=True)
    ], "a fan-out: one child per pose, in the order ORCA wrote them"
    assert sorted(s.id for s in run.outcomes[0].state.structures) == pose_ids
    assert sorted(run.submitter.submissions()) == ["0", *pose_ids], "one solvation per pose"

    # Step 2: each pose solvated at the complex's charge, a frame per water, ids nested.
    pose_by_id = {s.id: s for s in docked.results.structures}
    solvated = cache.load(step2_dir)
    assert solvated is not None
    children: dict[str, dict[str, float]] = {}
    for pose_id in pose_ids:
        stem = f"step2_{pose_id}"
        rendered = (step2_dir / pose_id / f"{stem}.inp").read_text(encoding="utf-8")
        assert f"* xyzfile {charge} {multiplicity} " in rendered, "the step's own charge"
        out = case.submitted_file(f"{solvate.dir_name()}/{pose_id}", 0, f"{stem}.out")
        text = out.read_text(encoding="utf-8")
        ran_at = set(re.findall(r"^\s*:\s*net charge\s+(-?\d+)\s*:", text, re.MULTILINE))
        assert ran_at == {str(charge)}, "ORCA ran the complex at that charge"
        asked = re.search(
            r"^Number of solvent molecules to be added:\s+\.+\s+(\d+) molecules", text, re.MULTILINE
        )
        assert asked is not None
        frames = parse_output(out, "solvator")
        assert len(frames) == int(asked[1]) >= 2, "a frame per water; one frame is no fan-out"
        solvator = text[text.index("* ORCA Solvator *") : text.index("ORCA-SOLVATOR TERMINATED")]
        tabulated = [
            float(energy)
            for energy in re.findall(
                r"^\s+\d+\s+(-?\d+\.\d+)\s+-?\d+\.\d+\s+-?\d+\.\d+\s+\d+\.\d+\s*$",
                solvator,
                re.MULTILINE,
            )
        ]
        assert len(tabulated) == len(frames), "SOLVATOR tabulates an energy per water it adds"
        assert np.allclose([f.energy_hartree for f in frames], tabulated, rtol=0, atol=1e-6), (
            "each frame carries the energy ORCA reports for it, not a placement score"
        )
        pose = pose_by_id[pose_id]
        for k, frame in enumerate(frames):
            waters = ("O", "H", "H") * (k + 1)
            assert list(frame.symbols) == [*pose.atoms.get_chemical_symbols(), *waters]
            assert np.allclose(
                frame.positions[: len(pose.atoms)], pose.atoms.positions, atol=1e-5
            ), f"{pose_id} frame {k}: solvated around its own pose"
        children[pose_id] = {
            f"{pose_id}-{k}": frame.energy_hartree for k, frame in enumerate(frames)
        }
    # By id: the poses reach step 2 in energy order, and that order is a near-tie.
    assert {s.id: (s.parent_id, s.energy_hartree) for s in solvated.results.structures} == {
        sid: (pose_id, energy)
        for pose_id, frames_of in children.items()
        for sid, energy in frames_of.items()
    }, "each job's frames fan out under the pose it solvated"

    # by_parent: the lowest frame of each pose, where the flat filter keeps one in all.
    lowest = sorted(min(frames_of, key=frames_of.__getitem__) for frames_of in children.values())
    assert sorted(s.id for s in run.outcomes[1].state.structures) == lowest, "one per pose"
    assert solvate.sample is not None and solvate.sample.by_parent
    flat = filtering.apply(solvated.results, solvate.sample.model_copy(update={"by_parent": False}))
    assert len(flat.structures) == 1 < len(lowest), "without by_parent, count 1 keeps one"


def _mlip(run: Replayed) -> None:
    """The direct MLIP engine, one library per step, each at its own seam.

    MACE-OFF relaxes a stretched water in four LBFGS steps, which stop short: the script
    reports ``converged: false``, and the job is retried from the geometry it stopped at and
    converges. A water with both hydrogens on one site writes positions the engine cannot
    read, which ``skip`` ledgers as unparseable, naming the output relative to its step, and
    never retries. The ``task``/``model`` aliases reach the script spelled out. SevenNet's
    and CHGNet's defaults and ORB's molecular loader each take a single point of the relaxed
    water, and each energy sits in a window around its library's own, which another library,
    another model or a lost geometry cannot reach. ORB's loader needs a charge and a spin, so
    its script carries the step's.
    """
    case, config = run.case, run.config
    mace, *single_points = config.steps
    step_dir = {step.step: config.step_dir(step).resolve() for step in config.steps}
    job = {step.step: f"{step.dir_name()}/0" for step in config.steps}
    assert dict(run.submitter.served) == {
        job[1]: 2,
        f"{mace.dir_name()}/1": 1,
        **{job[step.step]: 1 for step in single_points},
    }, "one retry, for the run cut short; the unparseable frame is not retried"

    # Step 1: the aliases spelled out, a relaxation cut short and retried from where it stopped.
    rendered = (step_dir[1] / "0" / "step1_0.py").read_text(encoding="utf-8")
    for line in ('model_name="small",', 'task_name="mace_off",', 'device="cpu",'):
        assert line in rendered, f"the aliases, spelled out: {line!r} is not rendered"
    stopped, retried = (
        json.loads(case.submitted_file(job[1], k, "step1_0.json").read_text(encoding="utf-8"))
        for k in (0, 1)
    )
    assert stopped["converged"] is False and retried["converged"] is True
    (retry_input,) = read_xyz_frames(step_dir[1] / "0" / "step1_0_inp.xyz")
    assert np.allclose(retry_input.positions, stopped["positions_angstrom"], atol=1e-6), (
        "retried from the geometry it stopped at"
    )
    (seed, _junk) = read_xyz_frames(case.root / "waters.xyz")
    assert not np.allclose(retry_input.positions, seed.positions, atol=1e-3)
    (relaxed,) = run.outcomes[0].state.structures
    assert relaxed.id == "0" and relaxed.converged is True
    assert relaxed.energy_hartree == retried["energy_hartree"]
    # MACE-OFF small's water: -76.47928349 Eh relaxed; attempt 1 stopped at -76.47887394.
    assert relaxed.energy_hartree is not None
    assert abs(relaxed.energy_hartree - -76.47928349) < 1e-5, relaxed.energy_hartree

    (failure,) = cache.load_failure_records(step_dir[1])
    assert (failure.structure_id, failure.kind) == ("1", FailureKind.UNPARSEABLE)
    assert "output 1/step1_1.json reports a non-finite 'positions_angstrom'" in failure.reason, (
        f"named relative to its step, with what it could not read: {failure.reason}"
    )
    assert not ABSOLUTE_PATH_RE.search(failure.reason), f"names the machine: {failure.reason}"

    # Steps 2-4: each library's own energy for the relaxed water, from its own script.
    (onward,) = read_xyz_frames(ids.step_survivors_path(step_dir[1], mace.step))
    assert np.allclose(onward.positions, retried["positions_angstrom"], atol=1e-6), (
        "the retry's geometry goes on"
    )
    # Each library's energy for that water, bitwise repeatable, in a window no wrong wiring
    # measured reaches: SevenNet's nearest is ORB's default (3.6e-5 away), CHGNet's SevenNet
    # (2.6e-2), and ORB-omol's MACE-OMOL (5.4e-5).
    expected = {
        2: ("sevenn", "", -0.51720720, 1e-5),
        3: ("chgnet", "", -0.54316987, 1e-5),
        4: ("orb", "orb_v3_direct_omol", -76.43313360, 1e-5),
    }
    for step in single_points:
        n = step.step
        task, model, energy, window = expected[n]
        rendered = (step_dir[n] / "0" / f"step{n}_0.py").read_text(encoding="utf-8")
        for line in (
            f'model_name="{model}",',
            f'task_name="{task}",',
            "charge=0,",
            "multiplicity=1,",
        ):
            assert line in rendered, f"step {n}: {line!r} is not rendered"
        (single_input,) = read_xyz_frames(step_dir[n] / "0" / f"step{n}_0_inp.xyz")
        assert np.allclose(single_input.positions, onward.positions, atol=1e-8), (
            f"step {n} evaluates step 1's relaxed water"
        )
        (survivor,) = run.outcomes[n - 1].state.structures
        assert survivor.id == "0"
        assert survivor.energy_hartree is not None
        assert abs(survivor.energy_hartree - energy) < window, (
            f"step {n}: {survivor.energy_hartree} is not {task}'s water"
        )


def _mlip_extopt(run: Replayed) -> None:
    """ORCA's optimiser over MACE-MP gradients: the adapter's units, judged by ORCA itself.

    One H2, one ExtOpt job. The rendered input sends ORCA to the step's wrapper, which relays
    each geometry to the MLIP server; MACE answers in eV and eV/Å, and the adapter hands ORCA
    hartree and hartree/bohr. H2 stretched to 0.85 Å relaxes to MACE-MP's bond and converges,
    read at the energy the server gave for the converged geometry, below the one it gave for
    the seed. That energy sits in a window around MACE-MP small's own H2, which holds the
    energy's conversion (left in eV it is 27.2 times lower) and the model the step names
    (MACE-OFF small's H2 is 0.93 Eh lower). The gradient's conversion is held by ORCA's own
    ratio of the energy change it observed after its first step to the one it predicted from
    the gradient: a gradient 51 times too large or too small (eV/Å against hartree/bohr)
    takes it far outside [0.1, 10], and one of the wrong sign makes it negative. That needs
    no reference number, only an energy and a gradient in the same units. The first ratio is
    about 0.3, not 1, because the step is cut to ORCA's trust radius and overshoots.
    """
    (step,) = run.config.steps
    step_dir = run.config.step_dir(step).resolve()
    job = f"{step_dir.name}/0"
    assert dict(run.submitter.served) == {job: 1}, "one job, converged on its first submission"

    rendered = (step_dir / "0" / "step1_0.inp").read_text(encoding="utf-8")
    (target,) = re.findall(r'^\s*ProgExt "([^"]+)"$', rendered, re.MULTILINE)
    wrapper = step_dir / "mlip_extopt.sh"
    assert Path(target).samefile(wrapper), "ORCA is sent to the step's own wrapper"
    assert " --backend mlip " in wrapper.read_text(encoding="utf-8"), "relaying to the MLIP server"

    out = run.case.submitted_file(job, 0, "step1_0.out")
    text = out.read_text(encoding="utf-8")
    parsed = parse_output(out, "opt_sp")[0]
    assert parsed.terminated_normally and parsed.converged, "ORCA converged the optimisation"
    (survivor,) = run.outcomes[0].state.structures
    energy = survivor.energy_hartree
    assert survivor.converged is True and energy is not None
    assert energy == parsed.energy_hartree
    external = [
        float(e)
        for e in re.findall(
            r"FINAL SINGLE POINT ENERGY \(From external program\)\s+(-?\d+\.\d+)", text
        )
    ]
    assert external and energy == external[-1], "the server's energy at the converged geometry"
    assert energy < external[0], "below the seed's: not read at the first cycle"
    # MACE-MP small's H2, converged: -0.24099058 Eh. Left in eV it is -6.558; MACE-OFF small's
    # is -1.17051 Eh. The seed's own energy (-0.23511) is inside: the line above excludes it.
    assert -0.25099058 < energy < -0.23099058, f"not MACE-MP small's H2 in hartree: {energy}"

    ratios = [
        float(r)
        for r in re.findall(r"Ratio of predicted to observed change\s+\.+\s+(-?\d+\.\d+)", text)
    ]
    assert ratios, "the seed must sit off the minimum, so ORCA judges a step it took"
    assert 0.1 <= ratios[0] <= 10, (
        f"ORCA observed {ratios[0]} times the energy change it predicted from the gradient: "
        "the gradient is not in hartree/bohr, or not the energy's slope"
    )

    # MACE-MP small's H2 bond: 0.7504 Å. The seed is off it, and the survivor is on it.
    (seed,) = read_xyz_frames(run.case.root / "h2.xyz")
    seed_bond = float(np.linalg.norm(seed.positions[1] - seed.positions[0]))
    assert abs(seed_bond - 0.7504) > 0.05, f"the seed must be stretched: {seed_bond:.4f} Å"
    bond = float(np.linalg.norm(survivor.atoms.positions[1] - survivor.atoms.positions[0]))
    assert abs(bond - 0.7504) < 0.05, f"not relaxed to MACE-MP's bond: {bond:.4f} Å"


def _mlip_train(run: Replayed) -> None:
    """ORCA's labels fitted by each training library, and each fitted model run back.

    Each dataset holds ORCA's own final energy and the negative of its Cartesian gradient,
    in eV and eV/Å, one water to train on and one held out, and CHGNet's in a cubic
    vacuum box, since it is periodic-only. Each template renders against the dataset the
    step wrote, each library trains in one job, and the sidecar cites the model and what it
    started from: the release the step names, as the step names it, or, unset, the
    trainer's declared default. The lower-energy water is then run through each fitted
    model, by the library that fitted it, and each energy sits in a window around that
    model's own, which the pretrained model a dropped ``model_path`` would load cannot reach.
    """
    case, config = run.case, run.config
    served = dict(run.submitter.served)
    labels_step = config.steps[0]
    fits = [step for step in config.steps if step.engine == "mlip-train"]
    round_trips = [step for step in config.steps if step.engine == "mlip"]

    # ORCA's own words for each label: its final energy and its Cartesian gradient.
    labels: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for sid in ("0", "1"):
        job = f"{labels_step.dir_name()}/{sid}"
        assert served[job] == 1
        text = case.submitted_file(job, 0, f"step1_{sid}.out").read_text(encoding="utf-8")
        energy = float(re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)[-1])
        rows = text[text.rindex("CARTESIAN GRADIENT") :].splitlines()[3:6]
        gradient = np.array([[float(x) for x in row.split()[3:6]] for row in rows])
        (structure,) = [s for s in run.outcomes[0].state.structures if s.id == sid]
        shape = structure.atoms.positions - structure.atoms.positions[0]
        labels[sid] = (energy * HARTREE_TO_EV, -gradient * HARTREE_PER_BOHR_TO_EV_PER_A, shape)

    def labelled_as(frame: Any) -> str:
        """The step-1 id a dataset frame is, by its shape (a boxed frame is shifted)."""
        shape = frame.positions - frame.positions[0]
        (sid,) = [
            sid for sid, (*_, known) in labels.items() if np.allclose(shape, known, atol=1e-6)
        ]
        return sid

    # What each fit starts from, as its sidecar cites it, and the file it writes.
    expected = {
        "mace_off": ("mlip-mace", "small", "train.model", "xyz"),
        "sevenn": ("mlip-sevenn", "7net-0", "checkpoint_best.pth", "xyz"),
        "chgnet": ("mlip-chgnet", "0.3.0", "train.pth.tar", "xyz"),
        "orb": ("mlip-orb", "orb_v3_conservative_inf_omat", "train.ckpt", "db"),
    }
    artifacts: dict[str, Path] = {}
    for fit in fits:
        task = fit.options["task_name"]
        backend, started_from, artifact, suffix = expected[task]
        run_dir = config.step_dir(fit).resolve() / ids.TRAINING_ID
        assert served[f"{fit.dir_name()}/{ids.TRAINING_ID}"] == 1, f"{task}: one job, once"
        rendered = (run_dir / f"step{fit.step}_train.yaml").read_text(encoding="utf-8")
        assert str(run_dir / f"train.{suffix}") in rendered, f"{task}: renders its dataset"
        if task == "mace_off":
            assert "foundation_model: small_off" in rendered, "MACE-OFF's small, as MACE names it"
        seen = []
        for split in ("train", "valid"):
            path = run_dir / f"{split}.{suffix}"
            if suffix == "db":
                with connect(str(path)) as db:
                    frames = [row.toatoms() for row in db.select()]
            else:
                frames = list(iread(str(path)))
            assert len(frames) == 1, f"{task}: one water to {split} on"
            for frame in frames:
                sid = labelled_as(frame)
                seen.append(sid)
                energy_ev, forces, _shape = labels[sid]
                if task == "mace_off":
                    got_e, got_f = frame.info["REF_energy"], frame.arrays["REF_forces"]
                else:
                    got_e, got_f = frame.get_potential_energy(), frame.get_forces()
                assert abs(got_e - energy_ev) < 1e-6, f"{task}/{sid}: ORCA's energy, in eV"
                assert np.allclose(got_f, forces, rtol=0, atol=1e-6), (
                    f"{task}/{sid}: minus ORCA's gradient, in eV/Å"
                )
                if task == "chgnet":
                    # 10 Å each side, as the docs promise: past CHGNet's 6 Å atom graph.
                    side = float(np.ptp(frame.positions, axis=0).max()) + 2 * 10.0
                    assert frame.pbc.all() and np.allclose(frame.cell.array, np.eye(3) * side), (
                        f"{sid}: fitted in a cubic vacuum box"
                    )
                else:
                    assert not frame.pbc.any(), f"{task}/{sid}: a molecule, fitted as one"
        assert sorted(seen) == ["0", "1"], f"{task}: both labels, one per split"
        sidecar = json.loads((run_dir / SIDECAR_NAME).read_text(encoding="utf-8"))
        assert sidecar["task_name"] == task and sidecar["backend"] == backend
        assert sidecar["model_path"] == artifact and sidecar["n_structures"] == 2
        assert sidecar["started_from"] == started_from, f"{task}: {sidecar['started_from']}"
        artifacts[task] = run_dir / artifact
        if fit.sample is None:
            kept = [s.id for s in run.outcomes[fit.step - 1].state.structures]
            assert kept == ["0", "1"], f"{task}: the ensemble passes through"

    # Each fitted model, run by its library on the lower-energy water. Each window is the
    # trained model's own energy; the pretrained model's is 1e-3 Eh or more away.
    trained = {
        "mace_off": -75.33151031,  # pretrained MACE-OFF small: -76.47933836
        "sevenn": -0.52229955,  # pretrained 7net-0: -0.51791364
        "chgnet": -0.54846524,  # pretrained 0.3.0: -0.54424447
        "orb": -0.52095277,  # pretrained default loader: -0.51794676
    }
    (onward,) = run.outcomes[fits[-1].step - 1].state.structures
    assert onward.id == "1", "the lower-energy water"
    for step in round_trips:
        task = step.options["task_name"]
        step_dir = config.step_dir(step).resolve()
        assert served[f"{step.dir_name()}/1"] == 1
        script = (step_dir / "1" / f"step{step.step}_1.py").read_text(encoding="utf-8")
        assert Path(step.options["model_path"]) == artifacts[task], f"{task}: its own fit"
        assert f'model_path="{artifacts[task]}",' in script, f"{task}: loads the fitted model"
        assert f'task_name="{task}",' in script, f"{task}: run by the library that fitted it"
        assert 'model_name="",' in script and "charge=0," in script and "multiplicity=1," in script
        (single_input,) = read_xyz_frames(step_dir / "1" / f"step{step.step}_1_inp.xyz")
        assert np.allclose(single_input.positions, onward.atoms.positions, atol=1e-6)
        (survivor,) = run.outcomes[step.step - 1].state.structures
        assert survivor.energy_hartree is not None
        assert abs(survivor.energy_hartree - trained[task]) < 1e-5, (
            f"{task}: {survivor.energy_hartree} is not its fitted model's energy"
        )


def _pyscf(run: Replayed) -> None:
    """The engine's own starter on H2+: the stamp reaches ``gto.M``, the footer comes back.

    The case's template is the starter ``chemrefine scaffold`` writes, so the live run tries
    the file a user starts from; a change to the starter fails here until the case takes it
    and is re-recorded. Rendered per frame, it carries the step's charge and multiplicity
    (overriding the workflow's) into ``gto.M`` and the step's options into the SCF. H2+, one
    electron, comes back converged at the energy its footer wrote, inside the B3LYP/STO-3G
    window: with only one of the charge and the multiplicity PySCF refuses H2, and with
    neither it is neutral H2, 0.52 Eh lower. HeH under the same stamp has two electrons as a
    doublet, so PySCF refuses it before any SCF; the job leaves no output, and ``skip``
    ledgers a missing output, never retried, while H2+ goes on alone.
    """
    (step,) = run.config.steps
    step_dir = run.config.step_dir(step).resolve()

    fresh = run.config.model_copy(update={"template_dir": run.case.root / "starters"})
    (starter,) = [path for path in scaffold_templates(fresh) if path.name == "step1.py"]
    assert (run.config.template_dir / starter.name).read_text(encoding="utf-8") == (
        starter.read_text(encoding="utf-8")
    ), "the case runs the starter `chemrefine scaffold` writes: copy it in and re-record"

    for sid in ("0", "1"):
        rendered = (step_dir / sid / f"step1_{sid}.py").read_text(encoding="utf-8")
        for line in (
            "charge=1,",
            "spin=2 - 1,",
            'basis="sto-3g",',
            'if "dft" == "hf":',
            'mf = dft.KS(mol, xc="b3lyp")',
            "if True:\n    mf = mf.density_fit()",
        ):
            assert line in rendered, f"frame {sid}: {line!r} is not in its rendered script"
    assert dict(run.submitter.served) == {f"{step_dir.name}/0": 1, f"{step_dir.name}/1": 1}, (
        "each frame runs once: a missing output is not retried"
    )

    footer = run.case.submitted_file(f"{step_dir.name}/0", 0, "step1_0.json")
    written = json.loads(footer.read_text(encoding="utf-8"))
    (survivor,) = run.outcomes[0].state.structures
    assert survivor.id == "0"
    assert survivor.converged is True, "the starter's `converged` comes back through the footer"
    energy = survivor.energy_hartree
    assert energy is not None and energy == written["energy_hartree"], energy
    assert -0.60497 < energy < -0.58497, f"not UKS-B3LYP/STO-3G H2+: {energy}"

    ledger = cache.load_failure_records(step_dir)
    assert [(r.structure_id, r.kind) for r in ledger] == [("1", FailureKind.MISSING_OUTPUT)]


def _pyscf_extopt(run: Replayed) -> None:
    """ORCA's optimiser over a PySCF gradient server: cut short, run to a minimum, given up on.

    Step 1 runs LiH+ as a UKS-B3LYP doublet, its charge and multiplicity carried to the
    server by the ``.extinp`` ORCA writes: the first energy ORCA gets back is PySCF's own
    LiH+ doublet at the seed. ``MaxIter 2`` stops each attempt far from converged, as ORCA
    says itself. ExtOpt prints ``FINAL SINGLE POINT ENERGY`` for the first cycle only, so
    each attempt is read at its last ``Current Energy``; the retry starts from the geometry
    the first attempt stopped at, its first energy that one again, and ``best`` keeps the
    retry's last cycle, unconverged, and hands it on to step 2. Step 2 optimises neutral
    LiH at HF and takes a numerical Hessian over server gradients. ORCA recognizes the
    molecule as linear, so its one vibration is the last of six modes and survives the
    read; with no imaginary mode, NMS passes it through without a second round. Step 3
    gives each gradient 1 ms: ORCA stops on the bridge's failure, and the ledgered reason
    names the output relative to its step and quotes the bridge's one-line refusal, which
    names ``gradient_timeout_seconds`` and carries no traceback. ``best`` carries step 2's
    minimum on, with no energy. The tensors, the density-fitted gradients and the servers'
    logs are not archived; the live check holds them.
    """
    case, served = run.case, run.submitter.served
    budget, _freq, no_answer = run.reached
    job = {step.step: f"{step.dir_name()}/0" for step in run.reached}
    assert dict(served) == {job[1]: 2, job[2]: 1, job[3]: 1}, (
        "one retry, for the cycle limit; the minimum and the abort run once"
    )

    # Step 1: both attempts cut short, far from converged, each read at its last cycle.
    fspe_re = re.compile(r"FINAL SINGLE POINT ENERGY.*?(-?\d+\.\d+)")
    current_re = re.compile(r"Current Energy\s+\.+\s+(-?\d+\.\d+)")
    rms_re = re.compile(r"^\s*RMS gradient\s+(\S+)\s+(\S+)\s+(?:YES|NO)\s*$", re.MULTILINE)
    attempts = []
    for k, home in enumerate((case.output_dir / job[1] / "attempt1", case.output_dir / job[1])):
        inp = (home / "step1_0.inp").read_text(encoding="utf-8")
        assert "* xyzfile 1 2 " in inp, "the step's charge and multiplicity reach ORCA"
        out = case.submitted_file(job[1], k, "step1_0.out")
        text = out.read_text(encoding="utf-8")
        assert CYCLE_LIMIT_RE.search(text), "ORCA's own notice that it ran out of cycles"
        rms, tolerance = map(float, rms_re.findall(text)[-1])
        assert rms >= 100 * tolerance, f"RMS gradient {rms / tolerance:.0f}x its tolerance"
        [first] = [float(e) for e in fspe_re.findall(text)]  # ExtOpt: the first cycle's alone
        last = float(current_re.findall(text)[-1])
        parsed = parse_output(out, "opt_sp")[0]
        assert parsed.terminated_normally and parsed.converged is False
        assert parsed.energy_hartree == last, "read at its last cycle"
        assert first - last > 1e-3, "which is not the first cycle's"
        attempts.append((first, last, parsed))
    (seed_energy, stopped_at, stopped), (retry_first, _, retried) = attempts
    # PySCF alone, UKS-B3LYP/STO-3G LiH+ doublet at the seed: -7.61361824 Eh. Neutral LiH at
    # RHF is 0.22 Eh lower there, and a half-lost stamp is an electron count PySCF refuses.
    assert abs(seed_energy - -7.61361824) < 0.01, seed_energy

    (seed,) = read_xyz_frames(case.root / "lih.xyz")
    (retry_input,) = read_xyz_frames(case.output_dir / job[1] / "step1_0_inp.xyz")
    assert np.allclose(retry_input.positions, stopped.positions), "retried from its last cycle"
    assert not np.allclose(retry_input.positions, seed.positions, atol=1e-3)
    assert abs(retry_first - stopped_at) < 1e-6, "its first energy is that one again"
    assert not np.allclose(retried.positions, retry_input.positions, atol=1e-3), "and moved on"
    (kept,) = run.outcomes[0].state.structures
    assert kept.converged is False
    assert kept.energy_hartree == retried.energy_hartree, "the retry's last cycle"
    assert np.allclose(kept.atoms.positions, retried.positions)
    (onward,) = read_xyz_frames(case.output_dir / job[2] / "step2_0_inp.xyz")
    assert np.allclose(onward.positions, retried.positions), "and step 2 starts from it"
    (failure,) = cache.load_failure_records(run.config.step_dir(budget).resolve())
    assert (failure.structure_id, failure.kind) == ("0", FailureKind.NOT_CONVERGED)

    # Step 2: a minimum, read with the one vibration a linear molecule has.
    inp = (case.output_dir / job[2] / "step2_0.inp").read_text(encoding="utf-8")
    assert "* xyzfile 0 1 " in inp, "the workflow's neutral singlet, not step 1's stamp"
    freq_out = case.submitted_file(job[2], 0, "step2_0.out")
    text = freq_out.read_text(encoding="utf-8")
    assert "The molecule is recognized as being linear" in text
    modes = [
        (int(index), float(cm1))
        for index, cm1 in re.findall(r"^\s*(\d+):\s+(-?\d+\.\d+) cm\*\*-1", text, re.MULTILINE)
    ]
    assert [index for index, _ in modes] == list(range(6)), "ORCA's table: 3N = 6 modes"
    assert all(cm1 == 0.0 for _, cm1 in modes[:5]) and modes[5][1] > 0, "five trivial, one real"
    minimum = parse_output(freq_out, "opt_sp")[0]
    assert minimum.converged and minimum.imaginary_freqs == {}
    assert minimum.frequencies == dict(modes[5:]), "a linear molecule keeps its one vibration"
    assert minimum.normal_modes is not None and minimum.normal_modes.shape == (2, 3, 6)
    # PySCF alone, RHF/STO-3G LiH at its minimum: -7.86346639 Eh.
    assert abs(minimum.energy_hartree - -7.86346639) < 0.01, minimum.energy_hartree
    (survivor,) = run.outcomes[1].state.structures
    assert survivor.energy_hartree == minimum.energy_hartree
    assert survivor.imaginary_freqs == {} and survivor.frequencies == minimum.frequencies
    assert survivor.resolved_from is None
    assert not list((case.output_dir / job[2]).glob("attempt*")), "a minimum: no NMS round 2"

    # Step 3: the bridge gives the first gradient up, and says why in one line.
    stopped_text = case.submitted_file(job[3], 0, "step3_0.out").read_text(encoding="utf-8")
    assert "Error (ORCA/EXT): external program failed" in stopped_text, "ORCA stopped on it"
    err = case.submitted_file(job[3], 0, "step3_0.err").read_text(encoding="utf-8")
    said = [line.strip() for line in err.splitlines() if line.strip()]
    assert len(said) == 1, f"the bridge's one-line refusal, alone on stderr: {said}"
    (refusal,) = said
    assert refusal.startswith("JobFailureError: "), refusal
    assert "gave no reply within 0.001s" in refusal, "a timeout, not an unreachable server"
    (failure,) = cache.load_failure_records(run.config.step_dir(no_answer).resolve())
    assert (failure.structure_id, failure.kind) == ("0", FailureKind.NOT_TERMINATED_NORMALLY)
    reason = failure.reason
    assert " in 0/step3_0.out;" in reason, f"named relative to its step: {reason}"
    assert not ABSOLUTE_PATH_RE.search(reason), f"names the machine: {reason}"
    assert reason.endswith(f"step3_0.err ends: {refusal}"), "the bridge's refusal, whole"
    assert "gradient_timeout_seconds" in reason, "naming the knob to raise"
    assert "Traceback" not in reason and 'File "' not in reason, f"a traceback: {reason}"
    (carried,) = run.outcomes[2].state.structures
    assert carried.energy_hartree is None, "an abort has no energy to carry"
    assert np.allclose(carried.atoms.positions, survivor.atoms.positions), "step 2's minimum, on"


def _fairchem_sp(run: Replayed) -> None:
    """UMA's omol head on water: about -76.4 Eh, a number a quantum chemist recognises.

    Held to the default backend's own energy, bitwise repeatable, in the window the direct
    MLIP engine's checks use — a window an Eh wide was one ORB's omol head and MACE-OFF reach
    too, so a default that dispatched to either passed it.
    """
    (structure,) = run.outcomes[0].state.structures
    assert structure.energy_hartree is not None
    assert abs(structure.energy_hartree - -76.43356970) < 1e-5, structure.energy_hartree


CHECKS: dict[str, Callable[[Replayed], None]] = {
    "conformers": _conformers,
    "fairchem_sp": _fairchem_sp,
    "host_guest": _host_guest,
    "mlip": _mlip,
    "mlip_extopt": _mlip_extopt,
    "mlip_train": _mlip_train,
    "nms_recovery": _nms_recovery,
    "pyscf": _pyscf,
    "pyscf_extopt": _pyscf_extopt,
    "triage": _triage,
    "ts_path": _ts_path,
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
        operation=None,
        scratch_dir=None,
        prev_state=PipelineState(structures=(Structure(id="0", atoms=Atoms("H")),)),
        charge=0,
        multiplicity=1,
        max_cores=1,
        slurm_template="cpu.slurm.header",
    )
    engine = MlipTrainEngine()
    job = (engine.run_dir(ctx) / "step2_train.yaml", engine.artifact(ctx), "train")
    inputs = StepInputs(files=(job,))
    _stub_artifact(engine, ctx, inputs)
    assert engine.artifact(ctx).read_bytes() == ARTIFACT_STUB
    # ...beside the runlog of a job that ran to its end, which the step asks for too
    runlog = job_log.runlog_path(job[1].parent, job[0].stem)
    assert job_log.read_footer(runlog) == job_log.Footer(exit_code=0, scratch_kept=False)

    engine.artifact(ctx).write_bytes(b"real weights")
    _stub_artifact(engine, ctx, inputs)
    assert engine.artifact(ctx).read_bytes() == b"real weights"

    runlog.unlink()
    _stub_artifact(get_engine("fake"), ctx, inputs)  # not an artifact engine: nothing to do
    assert sorted(p.name for p in engine.artifact(ctx).parent.iterdir()) == ["train.model"]


def test_a_parse_only_rebuild_can_verify_the_steps_before_an_artifact_step(tmp_path: Path) -> None:
    """Every step after a trained model keys its bytes, which no stub reproduces."""
    with_training = extract_case(recording("mlip_train"), tmp_path / "train")
    assert verifiable_prefix(load_config(with_training.config_path)) == 1
    without = extract_case(recording("conformers"), tmp_path / "conformers")
    assert verifiable_prefix(load_config(without.config_path)) == 3


# ---------------------------------------------------------------------------
# The harness: what a live run's archive may carry, and what the run must leave
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "absolute"),
    [
        ('"reason": "in /home/x/run/step1_0.out"', True),
        ("scratch at /tmp/a/b", True),
        ("server 127.0.0.1:47665", False),
        ("see http://h:1/x", False),
        ("./a/b and ../c/d", False),
        ("no FINAL SINGLE POINT ENERGY in 3/step1_3.out", False),
        ("[file orca_main/x.cpp, line 10711]: Error", False),
    ],
)
def test_the_machine_path_guard_knows_an_absolute_path_from_a_relative_one(
    text: str, absolute: bool
) -> None:
    from replay import ABSOLUTE_PATH_RE

    assert bool(ABSOLUTE_PATH_RE.search(text)) is absolute


@pytest.mark.parametrize(
    ("err", "out", "kept"),
    [
        ("sh: orca_startup: not found\n", "ORCA finished by error termination in Startup\n", True),
        ("a library's warning\n", "****ORCA TERMINATED NORMALLY****\n", False),
        ("", "ORCA finished by error termination in Startup\n", False),
        ("a script's traceback\n", None, False),
    ],
    ids=["orca-abort", "normal-run", "empty", "no-out"],
)
def test_a_jobs_stderr_is_kept_only_where_it_says_why_orca_died(
    tmp_path: Path, err: str, out: str | None, kept: bool
) -> None:
    """The failure reason quotes it, so a replay needs it; anywhere else it is noise."""
    from replay import _kept

    (tmp_path / "step1_0.err").write_text(err, encoding="utf-8")
    if out is not None:
        (tmp_path / "step1_0.out").write_text(out, encoding="utf-8")
    rel = Path("step1/0/step1_0.err")
    assert _kept(tmp_path / "step1_0.err", rel, None) is kept


def test_what_the_pipeline_derives_after_a_job_is_not_archived(tmp_path: Path) -> None:
    """A replay re-creates what the pipeline writes once a job is done; archived, it would
    only go stale — a trained model's record did, still citing an absolute path."""
    from replay import _kept

    for name in ("step1_0.result.json", "step1_0.property.json", "trained_model.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
        assert not _kept(tmp_path / name, Path("step2/train") / name, None), name
    (tmp_path / "step1_0.json").write_text("{}", encoding="utf-8")
    assert _kept(tmp_path / "step1_0.json", Path("step1/0/step1_0.json"), None)


def test_a_live_run_is_held_to_its_declared_budget() -> None:
    within = CaseSpec(budget_s=10.0)
    within.assert_within_budget(9.9)
    with pytest.raises(AssertionError, match=r"ran 10\.5 s against a budget of 10 s"):
        within.assert_within_budget(10.5)
    with pytest.raises(AssertionError, match="declares budget_s"):
        CaseSpec().assert_within_budget(1.0)


_RUNLOG = """\
[2026-09-28T00:46:25+02:00] ChemRefine orca step1 starting
  host=h
  job_id={job}
  scratch=/s/ChemRefine_{job}
  output=/o/step1/{job}
  cores={cores}
[{finish}] ChemRefine orca step1 finished
  exit_code=0
"""


def test_the_throttle_bound_reads_every_runlog_once_per_job(tmp_path: Path) -> None:
    """Two 2-core jobs side by side fit 4 cores and not 3; a repeated runlog counts once."""
    from replay import assert_throttled, runlog_jobs

    jobs = []
    for job in ("1", "2"):
        path = tmp_path / f"{job}.runlog"
        path.write_text(
            _RUNLOG.format(job=job, cores=2, finish="2026-09-28T00:46:30+02:00"), encoding="utf-8"
        )
        jobs += runlog_jobs(path)
    assert [(j.job_id, j.cores, j.scratch.name) for j in jobs] == [
        ("1", 2, "ChemRefine_1"),
        ("2", 2, "ChemRefine_2"),
    ]
    assert_throttled(jobs + jobs[:1], max_cores=4)  # a promoted winner repeats its child's
    with pytest.raises(AssertionError, match="4 cores ran"):
        assert_throttled(jobs, max_cores=3)


def test_a_job_that_finishes_as_another_starts_does_not_overlap_it(tmp_path: Path) -> None:
    """Intervals are half-open; and a job that never finished holds its cores to the end."""
    from replay import RunlogJob, assert_throttled

    def job(job_id: str, start: str, finish: str | None) -> RunlogJob:
        return RunlogJob("h", job_id, start, finish, 2, Path("/s"), Path("/o"))

    t1, t2, t3 = (f"2026-09-28T00:{minute}:00+02:00" for minute in ("00", "10", "20"))
    assert_throttled([job("1", t1, t2), job("2", t2, t3)], max_cores=2)
    with pytest.raises(AssertionError):
        assert_throttled([job("1", t1, None), job("2", t2, t3)], max_cores=2)


def test_runlog_times_are_ordered_as_instants_across_a_clock_change() -> None:
    """``date -Iseconds`` writes the local offset, so a time's text is no order.

    The October change sets the clock back an hour: a job from 02:50+02:00 to 02:30+01:00
    ran forty minutes, and one from 02:10+01:00 overlapped twenty of them. Ordered as text,
    the first job's finish sorts before its own start, and the overlap disappears.
    """
    from replay import RunlogJob, assert_throttled

    def job(job_id: str, start: str, finish: str) -> RunlogJob:
        return RunlogJob(
            "h", job_id, f"2026-10-25T{start}", f"2026-10-25T{finish}", 2, Path("/s"), Path("/o")
        )

    with pytest.raises(AssertionError, match="4 cores ran"):
        assert_throttled(
            [
                job("1", "02:50:00+02:00", "02:30:00+01:00"),
                job("2", "02:10:00+01:00", "02:40:00+01:00"),
            ],
            max_cores=2,
        )
