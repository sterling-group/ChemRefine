"""Relocated-tree replays: cache fingerprints, rebuild-cache, failure policies.

These tests stage a captured output tree as the live output dir (a run moved
to a different machine/path) and assert that the cache layer recognises it
without resubmitting anything — plus the ``rebuild-cache`` parse-only path
and the ``on_failure`` policies driven from a genuinely missing output.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import replay
from replay import extract_case, forbid_run_batch, relocate, replay_run_batch

from chemrefine import pipeline
from chemrefine.config import Config, MinSample, load_config
from chemrefine.errors import ChemRefineError
from chemrefine.step import RunPlan, StepMode

RUN_BATCH = "chemrefine.engines._execution.run_batch"

ORCA_CASES = ["conformers", "nms_minimum", "ts_pes", "host_guest"]

ALL_CASES = [
    *ORCA_CASES,
    "mlip_screen",
    "mlip_extopt",
    "fairchem_sp",
    "pyscf_sp",
    "pyscf_extopt",
]
"""Every recorded case, for the checks that only re-parse.

The ORCA-only list above is for replays that submit through a stubbed `run_batch`; a
parse-only rebuild needs no backend at all, so the drift detector — the one thing standing
between a recording and becoming a fossil — covers every case that has an archive.

`mlip_train` has an archive but sits out of this list: the parse-only rebuild of its step 3
validates a fingerprint that covers the trained model's **bytes** (`option_file_digests`),
and the 4.7 MB model cannot live in a 1 MB recording — no stub can hash like the original.
Its offline coverage is `test_e2e_replay.test_mlip_train_full_pipeline`, a fresh replay
that computes its fingerprints from a stubbed model self-consistently."""


def _with_step_update(config: Config, step_number: int, **updates: object) -> Config:
    steps = [
        step.model_copy(update=updates) if step.step == step_number else step
        for step in config.steps
    ]
    return config.model_copy(update={"steps": steps})


@pytest.mark.parametrize("name", ORCA_CASES)
def test_relocated_tree_is_a_full_cache_hit(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A captured tree moved to a new path must revalidate without submissions."""
    case = extract_case(name, tmp_path)
    relocate(case)
    monkeypatch.setattr(RUN_BATCH, forbid_run_batch)

    outcomes = pipeline.run(load_config(case.config_path))

    assert all(outcome.cache_hit for outcome in outcomes)
    assert outcomes[-1].state.structures, "the cached survivors come back"


@pytest.mark.parametrize("name", ORCA_CASES)
def test_rebuild_cache_reparses_outputs_without_submitting(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`rebuild-cache N` re-parses each step from disk and lands on the same survivors."""
    baseline_case = extract_case(name, tmp_path / "baseline")
    relocate(baseline_case)
    monkeypatch.setattr(RUN_BATCH, forbid_run_batch)
    baseline = pipeline.run(load_config(baseline_case.config_path))

    for step_number in range(1, len(baseline) + 1):
        case = extract_case(name, tmp_path / f"rebuild{step_number}")
        relocate(case)
        outcomes = pipeline.run(
            load_config(case.config_path),
            RunPlan(default=StepMode.CACHE_ONLY, overrides={step_number: StepMode.REBUILD}),
        )
        assert [s.id for s in outcomes[-1].state.structures] == [
            s.id for s in baseline[-1].state.structures
        ], f"rebuilding step {step_number} changed the survivors"


@pytest.mark.parametrize("name", ALL_CASES)
def test_rebuilt_records_match_the_archived_ones_field_for_field(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Re-parsing the archived outputs must reproduce the archived cache exactly.

    The drift detector for the recordings. Comparing only survivor *ids* — which is
    all the tests above do — lets the archives fall a parser change behind with no signal:
    the recorded caches keep the old value and the suite stays green. An archive that does
    not match what the code produces is not a fixture, it is a fossil, and the whole value
    of record/replay rests on the difference.

    A failure here means the parse changed on purpose and the recordings need
    regenerating (a parse-only rebuild from these same archived outputs — no ORCA and
    no MLIP stack required), not that the assertion is too strict.
    """
    case = extract_case(name, tmp_path)
    relocate(case)
    monkeypatch.setattr(RUN_BATCH, forbid_run_batch)

    archived = {
        doc.relative_to(case.output_dir): json.loads(doc.read_text())
        for doc in sorted(case.output_dir.rglob("_cache/step.json"))
    }
    assert archived, "the recording carries no cache documents to compare against"

    pipeline.run(load_config(case.config_path), RunPlan(default=StepMode.REBUILD))

    if request.config.getoption("--update-recordings"):
        # The rebuild above rewrote every cache in `case.output_dir` from the archived native
        # outputs, with `forbid_run_batch` proving nothing was submitted. That tree is exactly
        # what `pack_case` trims and re-archives, so regenerating is a re-pack — no ORCA, no
        # MLIP stack, which is what this test's own message has always promised.
        assert replay.pack_case(case.root, name).is_file()
        return

    for rel, before in archived.items():
        after = json.loads((case.output_dir / rel).read_text())
        assert after["structures"] == before["structures"], (
            f"{rel}: rebuilt records differ from the archive — regenerate the recording"
        )
        # The geometry half. Coordinates and forces are not in these records — the
        # sidecar split moved them to arrays.npz before the document is written — so a
        # record comparison alone left a coordinate or forces parse free to drift with
        # no signal (an earlier version even filtered the two keys out of records that
        # no longer carry them). The document's `arrays_digest` is a content hash over
        # the sidecar's values, so one equality pins every coordinate and force byte
        # without unpacking anything.
        assert after["arrays_digest"] == before["arrays_digest"], (
            f"{rel}: the rebuilt coordinate sidecar differs from the archive — a geometry "
            f"or forces parse changed; regenerate the recording"
        )
        # The key half: a fingerprint that moves means the cache-key payload changed
        # (an options default, a digest input), which strands every user's tree the
        # same way it strands this archive — a fact to surface here, not discover in
        # a user's resume.
        assert after["fingerprint"] == before["fingerprint"], (
            f"{rel}: the rebuilt fingerprint differs from the archive — the cache-key "
            f"payload changed; note the migration in the changelog and regenerate"
        )


# ---------------------------------------------------------------------------
# on_failure policies, driven by deleting one captured step-2 output
# ---------------------------------------------------------------------------


def _fail_one_step2_structure(tmp_path: Path) -> tuple[object, Config, str]:
    """Extract conformers, delete one step-2 archived output, return (case, config, sid)."""
    case = extract_case("conformers", tmp_path)
    victim_dir = sorted(case.captured.glob("step2/*"))[0]
    if victim_dir.name == "_cache":
        victim_dir = sorted(case.captured.glob("step2/*"))[1]
    sid = victim_dir.name
    shutil.rmtree(victim_dir)
    return case, load_config(case.config_path), sid


def test_on_failure_stop_halts_and_records_the_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, config, sid = _fail_one_step2_structure(tmp_path)
    monkeypatch.setattr(RUN_BATCH, replay_run_batch(case, allow_missing={sid}))

    with pytest.raises(ChemRefineError, match="halted"):
        pipeline.run(config)

    ledger = case.output_dir / "step2" / "_cache" / "failed_jobs.json"
    assert ledger.is_file()
    assert sid in ledger.read_text()


def test_on_failure_skip_drops_the_failed_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, config, sid = _fail_one_step2_structure(tmp_path)
    config = _with_step_update(config, 2, on_failure="skip")
    # The policy under test lives in step 2; step 3's archive only holds the
    # original survivors, so the replay stops after step 2.
    config = config.model_copy(update={"steps": config.steps[:2]})
    submitter = replay_run_batch(case, allow_missing={sid})
    monkeypatch.setattr(RUN_BATCH, submitter)

    outcomes = pipeline.run(config)

    assert len(outcomes) == 2, "skip lets the pipeline finish"
    survivors = {s.id for s in outcomes[1].state.structures}
    assert sid not in survivors
    ran = len(submitter.calls[1].files)
    assert len(survivors) == min(3, ran - 1)


def test_on_failure_best_keeps_the_structure_with_its_best_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, config, _sid = _fail_one_step2_structure(tmp_path)
    config = _with_step_update(
        config, 2, on_failure="best", sample=MinSample(method="min", count=3)
    )
    config = config.model_copy(update={"steps": config.steps[:2]})
    submitter = replay_run_batch(case, allow_missing={_sid})
    monkeypatch.setattr(RUN_BATCH, submitter)

    outcomes = pipeline.run(config)

    assert len(outcomes) == 2, "best lets the pipeline finish"
    ran = len(submitter.calls[1].files)
    assert len(outcomes[1].state.structures) == min(3, ran)


def test_resume_after_stop_resubmits_only_the_failed_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, config, sid = _fail_one_step2_structure(tmp_path)
    submitter = replay_run_batch(case, allow_missing={sid})
    monkeypatch.setattr(RUN_BATCH, submitter)
    with pytest.raises(ChemRefineError, match="halted"):
        pipeline.run(config)

    # Restore the deleted output from a pristine extraction, then resume.
    pristine = extract_case("conformers", tmp_path / "pristine")
    shutil.copytree(pristine.captured / "step2" / sid, case.captured / "step2" / sid)
    resumed = replay_run_batch(case)
    monkeypatch.setattr(RUN_BATCH, resumed)

    outcomes = pipeline.run(config)

    assert len(outcomes) == 3
    step2_resubmissions = [
        sid_ for call in resumed.calls for _i, out, sid_ in call.files if "step2" in str(out)
    ]
    assert step2_resubmissions == [sid], "resume re-attempts only the failed job"
