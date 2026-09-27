"""Full-pipeline replays of captured real runs (see ``tests/replay.py``).

Every test drives ``chemrefine.pipeline.run`` with the shipped engines doing
their real prepare / parse / NMS / filter / cache work; only the compute is
satisfied from a ``tests/data/e2e/recordings`` archive. Assertions are data-driven —
recomputed from the archive with the real parsers — so re-capturing the
fixtures never stales them.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from replay import (
    SUBMISSIONS,
    ReplayCase,
    ReplaySubmitter,
    SubmissionRecorder,
    extract_case,
    pack_case,
    replay_run_batch,
)

from chemrefine import ids, pipeline
from chemrefine.config import Config, load_config
from chemrefine.engines.orca.output import parse_output
from chemrefine.io import read_xyz_frames
from chemrefine.state import JobBatch, JobTriple, StepContext, StepInputs

RUN_BATCH = "chemrefine.engines._execution.run_batch"


def _with_step_update(config: Config, step_number: int, **updates: Any) -> Config:
    """A config copy with one step's fields replaced."""
    steps = [
        step.model_copy(update=updates) if step.step == step_number else step
        for step in config.steps
    ]
    return config.model_copy(update={"steps": steps})


def _with_backend_python(config: Config, step_number: int) -> Config:
    """Point a backend step at a dummy interpreter so preflight passes without torch."""
    step = config.find_step(step_number)
    options = dict(step.options or {})
    options["backend_python"] = "python"
    return _with_step_update(config, step_number, options=options)


def _replay(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ReplayCase, ReplaySubmitter, Config]:
    case = extract_case(name, tmp_path)
    submitter = replay_run_batch(case)
    monkeypatch.setattr(RUN_BATCH, submitter)
    return case, submitter, load_config(case.config_path)


# ---------------------------------------------------------------------------
# conformers: goat ensemble -> opt+freq with Gibbs sorting -> sp
# ---------------------------------------------------------------------------


def test_conformers_full_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, submitter, config = _replay("conformers", tmp_path, monkeypatch)
    ensemble = read_xyz_frames(next(case.captured.glob("step1/*/step1_*.finalensemble.xyz")))
    assert len(ensemble) >= 3, "the GOAT ensemble must feed the downstream count: 3"

    outcomes = pipeline.run(config)

    assert len(outcomes) == 3
    # Step 1 keeps the 3 lowest of the GOAT ensemble, which is what bounds the DFT
    # opt+freq below; step 2 then keeps 2 of those 3 by Gibbs energy.
    fan_out = len(submitter.calls[1].files)
    assert fan_out == min(3, len(ensemble)), "step 2 runs step 1's three survivors"
    assert len(outcomes[1].state.structures) == min(2, fan_out)
    assert len(outcomes[2].state.structures) == 1
    # Per step, not a total: the case is 3 -> 2 -> 1 survivors, so the report's exact
    # shape is knowable. The `count("\n") >= 4` this replaces was satisfied by the header
    # plus step 1's rows alone — dropping step 2's and step 3's CSV emission entirely
    # stayed green, the same class 944aade pinned with literals elsewhere.
    with (case.output_dir / "steps.csv").open(encoding="utf-8", newline="") as fh:
        rows_per_step = Counter(row["Step"] for row in csv.DictReader(fh))
    assert rows_per_step == {"1": 3, "2": 2, "3": 1}, "every step reports its survivors"
    submitter.assert_replayed()


def test_conformers_survivors_follow_the_steps_ranking_energy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 2's survivors are the lowest structures by the energy its filter names.

    Named for what this recording can actually test. The old name promised
    "sorts by Gibbs, *not* electronic" — a discrimination these three conformers cannot
    make: they order identically by both energies (the KNOWN DEGENERACY the case's own
    input.yaml documents), so inverting the `energy_type` handling stayed green here. The
    Gibbs-vs-electronic distinction is pinned at the unit tier with adversarial orderings
    (test_filtering); what the replay adds is the end-to-end fact that the ranking energy
    reaches the filter at all, and that Gibbs values survive to the survivors. A future
    re-record with split orderings can take the stronger name back.
    """
    case, _submitter, config = _replay("conformers", tmp_path, monkeypatch)
    outcomes = pipeline.run(config)

    parsed: dict[str, tuple[float, float]] = {}
    for out in case.captured.glob("step2/*/step2_*.out"):
        result = parse_output(out, "opt_sp")[0]
        assert result.gibbs_hartree is not None
        assert result.energy_hartree is not None
        parsed[out.parent.name] = (
            result.gibbs_hartree,
            result.energy_hartree,
        )
    ran = set(parsed)
    survivors = [s.id for s in outcomes[1].state.structures]
    expected = sorted(ran, key=lambda sid: parsed[sid][0])[: len(survivors)]
    assert survivors == expected, "step-2 survivors must be the lowest-Gibbs structures"
    for structure in outcomes[1].state.structures:
        assert structure.gibbs_hartree is not None


def test_conformers_inputs_carry_charge_and_clamped_pal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, _submitter, config = _replay("conformers", tmp_path, monkeypatch)
    pipeline.run(config)

    goat_inp = (case.output_dir / "step1" / "0" / "step1_0.inp").read_text(encoding="utf-8")
    assert f"nprocs {config.max_cores}" in goat_inp, "the template's 16 cores are clamped"
    assert "nprocs 16" not in goat_inp
    assert "* xyzfile 0 1 " in goat_inp
    step2_inp = next(case.output_dir.glob("step2/*/step2_*.inp")).read_text(encoding="utf-8")
    assert "* xyzfile 0 1 " in step2_inp


def test_conformers_second_run_hits_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _case, submitter, config = _replay("conformers", tmp_path, monkeypatch)
    first = pipeline.run(config)
    calls_after_first = len(submitter.calls)

    second = pipeline.run(config)

    assert all(outcome.cache_hit for outcome in second)
    assert len(submitter.calls) == calls_after_first, "a cache hit submits nothing"
    assert [s.id for s in second[-1].state.structures] == [s.id for s in first[-1].state.structures]


# ---------------------------------------------------------------------------
# nms_minimum: a saddle-point seed resolved to a true minimum by NMS round 2
# ---------------------------------------------------------------------------


def test_nms_resolves_saddle_via_round_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, submitter, config = _replay("nms_minimum", tmp_path, monkeypatch)
    # Round 1 as it was submitted: the canonical `.out` of the recorded tree is the promoted
    # winner, as a user's tree has it.
    round_one_out = case.submitted_file("step1/0", 0, "step1_0.out")
    round_one = parse_output(round_one_out, "opt_sp")[0]
    assert round_one.imaginary_freqs, "the captured seed must be a saddle"

    outcomes = pipeline.run(config)

    structure_dir = case.output_dir / "step1" / "0"
    attempt = structure_dir / "attempt1"
    children = sorted(p.name for p in attempt.iterdir() if p.is_dir())
    assert children and all(name.startswith("0_m") for name in children)
    assert {sid for c in submitter.calls for _i, _o, sid in c.files} >= set(children)
    child_records = list(attempt.glob("*/*.result.json"))
    assert child_records, "round-2 children leave canonical result records too"

    # The attempt holds both halves: the round-1 calculation that triggered the resolution
    # (loose files) and the children it spawned (subdirectories).
    assert (attempt / "step1_0.out").is_file(), "round 1 is archived, not overwritten"

    # ...and the canonical location holds exactly one calculation — the winner's. Before,
    # only the winning *geometry* was written back, leaving a .xyz from one calculation
    # beside the .out of another with nothing to show they disagreed.
    canonical_out = (structure_dir / "step1_0.out").read_text(encoding="utf-8")
    child_outs = [p.read_text(encoding="utf-8") for p in sorted(attempt.glob("*/step1_*.out"))]
    assert canonical_out != (attempt / "step1_0.out").read_text(encoding="utf-8"), (
        "not round 1's output"
    )
    assert child_outs.count(canonical_out) == 1, "canonical is one specific child's output"
    assert (structure_dir / "step1_0.xyz").read_text(encoding="utf-8").splitlines()[
        1
    ] == "NMS-resolved 0"

    (survivor,) = outcomes[0].state.structures
    assert survivor.id == "0", "the resolved child is written back under the parent id"
    assert survivor.converged
    assert survivor.imaginary_freqs == {}, "a verified minimum — a table, with nothing in it"
    assert len(outcomes[1].state.structures) == 1
    submitter.assert_replayed()


# ---------------------------------------------------------------------------
# ts_pes: max-sampling a scan, then an OptTS gated by the inferred ts target
# ---------------------------------------------------------------------------


def test_ts_pes_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, submitter, config = _replay("ts_pes", tmp_path, monkeypatch)
    scan_points = parse_output(case.captured / "step1" / "0" / "step1_0.out", "pes")
    assert len(scan_points) >= 3, "the scan must resolve several points"

    outcomes = pipeline.run(config)

    assert len(submitter.calls[1].files) == 1, "max count: 1 keeps one TS candidate"
    energies = [p.energy_hartree for p in scan_points]
    (ts_seed,) = outcomes[0].state.structures
    assert ts_seed.energy_hartree == max(e for e in energies if e is not None)
    (ts,) = outcomes[1].state.structures
    assert len(ts.imaginary_freqs) == 1, "a first-order saddle passes the ts target"
    ts_dir = case.output_dir / "step2" / ts.id
    assert not list(ts_dir.glob("attempt*")), "a clean TS needs no NMS round 2"
    submitter.assert_replayed()


# ---------------------------------------------------------------------------
# host_guest: docker fan-out, per-step charge override, solvator parse
# ---------------------------------------------------------------------------


def test_host_guest_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, submitter, config = _replay("host_guest", tmp_path, monkeypatch)
    poses = parse_output(case.captured / "step1" / "0" / "step1_0.out", "docker")
    assert len(poses) >= 2, "docking must produce poses to filter"

    outcomes = pipeline.run(config)

    kept = len(outcomes[0].state.structures)
    assert kept == min(3, len(poses))
    if kept > 1:
        assert {s.id for s in outcomes[0].state.structures} <= {
            f"0-{i}" for i in range(len(poses))
        }, "docker children carry the seed's lineage"
    step2_inp = next(case.output_dir.glob("step2/*/step2_*.inp")).read_text(encoding="utf-8")
    assert "* xyzfile -1 1 " in step2_inp, "the per-step charge override reaches the input"
    assert len(submitter.calls[1].files) == kept
    assert outcomes[1].state.structures, "solvator output parses into survivors"
    submitter.assert_replayed()


# ---------------------------------------------------------------------------
# mlip_screen / mlip_extopt: option carry-through into rendered inputs
# ---------------------------------------------------------------------------


def test_mlip_screen_renders_options_and_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, submitter, config = _replay("mlip_screen", tmp_path, monkeypatch)
    config = _with_backend_python(config, 1)
    seed_frames = read_xyz_frames(case.root / "glycol2.xyz")
    assert len(seed_frames) == 2, "the multi-frame seed drives two jobs"

    outcomes = pipeline.run(config)

    assert len(submitter.calls[0].files) == 2
    rendered = (case.output_dir / "step1" / "0" / "step1_0.py").read_text(encoding="utf-8")
    assert '"small"' in rendered, "the model alias reaches the script"
    assert '"mace_off"' in rendered, "the task alias reaches the script"
    assert '"cpu"' in rendered, "device: cpu reaches the script"
    # This case carries the live tier's only Boltzmann filter (it moved here from
    # conformers, where filtering 14 GOAT results cost twenty minutes of DFT). At 99%
    # cumulative weight over two close MACE energies, both survive.
    assert len(outcomes[0].state.structures) == 2, "boltzmann keeps both close conformers"
    assert all(s.energy_hartree is not None for s in outcomes[0].state.structures)
    submitter.assert_replayed()


def test_mlip_extopt_renders_server_block_and_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, submitter, config = _replay("mlip_extopt", tmp_path, monkeypatch)
    config = _with_backend_python(config, 1)

    outcomes = pipeline.run(config)

    rendered = (case.output_dir / "step1" / "0" / "step1_0.inp").read_text(encoding="utf-8")
    assert "ProgExt" in rendered, "ExtOpt inputs must point ORCA at the gradient wrapper"
    (survivor,) = outcomes[0].state.structures
    assert survivor.converged
    assert survivor.energy_hartree is not None
    submitter.assert_replayed()


def test_extopt_cycle_limit_retries_from_where_it_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ExtOpt run out of cycles is retried from its last cycle; ``best`` keeps the retry's.

    ExtOpt prints ``FINAL SINGLE POINT ENERGY`` for its first cycle only, so a run read at
    the last of those came back at the geometry it was submitted with: the retry resubmitted
    the seed, stopped at the same limit on the same path, and ``best`` carried an
    unoptimised input downstream.
    """
    case, submitter, config = _replay("extopt_cycle_limit", tmp_path, monkeypatch)
    config = _with_backend_python(config, 1)

    outcomes = pipeline.run(config)

    assert submitter.submissions() == ["0", "0"], "one convergence retry"
    stopped = parse_output(case.submitted_file("step1/0", 0, "step1_0.out"), "opt_sp")[0]
    retried = parse_output(case.submitted_file("step1/0", 1, "step1_0.out"), "opt_sp")[0]
    (retry_input,) = read_xyz_frames(case.output_dir / "step1" / "0" / "step1_0_inp.xyz")
    assert np.allclose(retry_input.positions, stopped.positions), "retried from its last cycle"
    (seed,) = read_xyz_frames(case.root / "water_distorted.xyz")
    assert not np.allclose(retry_input.positions, seed.positions, atol=1e-3)
    (survivor,) = outcomes[0].state.structures
    assert survivor.converged is False
    assert survivor.energy_hartree == retried.energy_hartree
    assert np.allclose(survivor.atoms.positions, retried.positions)
    submitter.assert_replayed()


# ---------------------------------------------------------------------------
# mlip_train: label -> fine-tune (artifact step) -> run the trained model
# ---------------------------------------------------------------------------


class _StubModelSubmitter(ReplaySubmitter):
    """`ReplaySubmitter` that materialises the one artifact too big to record.

    The archive keeps everything the parsers read. The trained model is 4.7 MB of torch
    weights that nothing offline ever *parses* — the pipeline checks it exists and digests
    its bytes — so recording it would dwarf every other archive combined for a file whose
    content no offline assertion can see. Stub bytes keep both facts true; the real model
    is the live tier's business, and the parse-only rebuild that genuinely needs its bytes
    (step 3's fingerprint) is why `replay.VERIFIABLE_STEPS` stops that case's drift check
    after step 1.
    """

    def _satisfy(self, inputs: StepInputs) -> None:
        super()._satisfy(inputs)
        for _inp, out, sid in inputs.files:
            if sid == ids.TRAINING_ID and not out.exists():
                out.write_bytes(b"replayed-stub-model")


def test_mlip_train_full_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Label -> train -> run-the-model, replayed with no MACE installed anywhere.

    This is the one offline test in which `MlipTrainEngine` runs its whole lifecycle
    against a real captured run: the dataset really gets written from the labelled
    ensemble, the real MACE template really renders (placeholder validation included),
    the one job flows through the scheduler seam under the training id, the artifact
    decides success, the sidecar cites the model, and the next step consumes it.
    """
    case = extract_case("mlip_train", tmp_path)
    submitter = _StubModelSubmitter(case)
    monkeypatch.setattr(RUN_BATCH, submitter)
    config = load_config(case.config_path)
    for step in (1, 2, 3):
        config = _with_backend_python(config, step)

    outcomes = pipeline.run(config)

    # One job over the whole ensemble, under the training id — not one per structure.
    train_calls = [
        call for call in submitter.calls if any(sid == ids.TRAINING_ID for *_x, sid in call.files)
    ]
    assert len(train_calls) == 1 and len(train_calls[0].files) == 1

    # The rendered trainer config names the dataset the step just wrote.
    run_dir = case.output_dir / "step2" / "train"
    rendered = (run_dir / "step2_train.yaml").read_text(encoding="utf-8")
    assert str(run_dir / "train.xyz") in rendered and (run_dir / "train.xyz").is_file()
    assert str(run_dir / "valid.xyz") in rendered and (run_dir / "valid.xyz").is_file()

    # The sidecar cites what was produced, and for which library.
    sidecar = json.loads((run_dir / "trained_model.json").read_text(encoding="utf-8"))
    assert sidecar["task_name"] == "mace_off"
    assert sidecar["backend"] == "mlip-mace"
    assert sidecar["n_structures"] == 4

    # The ensemble passes through training untouched; step 3 runs all of it on the model.
    labelled = [s.id for s in outcomes[0].state.structures]
    assert [s.id for s in outcomes[1].state.structures] == labelled
    rendered3 = (case.output_dir / "step3" / "0" / "step3_0.py").read_text(encoding="utf-8")
    assert str(run_dir / "train.model") in rendered3, "step 3 loads the model step 2 produced"
    assert len(outcomes[2].state.structures) == 4
    assert all(s.energy_hartree is not None for s in outcomes[2].state.structures)
    submitter.assert_replayed()


def _replay_mlip_train(root: Path, monkeypatch: pytest.MonkeyPatch) -> ReplayCase:
    """The label → train → run replay, rooted wherever the caller says."""
    case = extract_case("mlip_train", root)
    monkeypatch.setattr(RUN_BATCH, _StubModelSubmitter(case))
    config = load_config(case.config_path)
    for step in (1, 2, 3):
        config = _with_backend_python(config, step)
    pipeline.run(config)
    return case


def test_mlip_train_replays_at_two_roots_derive_the_same_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tree copied elsewhere keys every step — the model-consuming one included — the same.

    Step 3 names the model by ``model_path: ./outputs/step2/train/train.model``, which the
    loader resolves to an absolute path, and that string used to ride into the row key: the
    same recording replayed at two roots derived two fingerprints for step 3, so a copied
    tree recomputed it on ``resume`` and was refused by ``rebuild-cache``. The relocation
    tests cannot see this case (its real model is too large to record), which is exactly
    where it hid; this replays it twice with the stub model and compares.
    """
    first = _replay_mlip_train(tmp_path / "cluster", monkeypatch)
    second = _replay_mlip_train(tmp_path / "laptop", monkeypatch)

    def fingerprints(case: ReplayCase) -> dict[str, str]:
        return {
            doc.parent.parent.name: json.loads(doc.read_text(encoding="utf-8"))["fingerprint"]
            for doc in sorted(case.output_dir.glob("step*/_cache/step.json"))
        }

    assert set(fingerprints(first)) == {"step1", "step2", "step3"}
    assert fingerprints(first) == fingerprints(second)


def test_mlip_train_second_run_hits_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A re-run serves every step from cache — including the training step.

    The training step's key covers the config and the parents; the consuming step's key
    covers the model's bytes (`option_file_digests`). Both must hold across a resume, or a
    pipeline with a trained model in the middle would retrain on every invocation.
    """
    case = extract_case("mlip_train", tmp_path)
    submitter = _StubModelSubmitter(case)
    monkeypatch.setattr(RUN_BATCH, submitter)
    config = load_config(case.config_path)
    for step in (1, 2, 3):
        config = _with_backend_python(config, step)
    first = pipeline.run(config)
    calls_after_first = len(submitter.calls)

    second = pipeline.run(config)

    assert all(outcome.cache_hit for outcome in second)
    assert len(submitter.calls) == calls_after_first, "a cache hit submits nothing"
    assert [s.id for s in second[-1].state.structures] == [s.id for s in first[-1].state.structures]


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
    from collections import Counter
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
    (run_dir / "templates").mkdir(parents=True)
    (run_dir / "input.yaml").write_text("output_dir: ./outputs\n", encoding="utf-8")
    job = run_dir / "outputs" / "step1" / "0"
    job.mkdir(parents=True)
    (job / "step1_0.out").write_text("rewritten", encoding="utf-8")
    log = {"step1/0": [[{"name": "step1_0.out", "sha256": _digest("as the job wrote it")}]]}
    (run_dir / SUBMISSIONS).write_text(json.dumps(log), encoding="utf-8")
    with pytest.raises(AssertionError, match="logged outputs missing from the packed tree"):
        pack_case(run_dir, "lost", dest_dir=tmp_path / "recordings")
