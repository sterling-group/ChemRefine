"""Tests for the recovery action dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms

from chemrefine import io
from chemrefine.config import Config, StepConfig
from chemrefine.errors import ChemRefineError
from chemrefine.recovery import Action, execute, invalidate_step, resolve_target
from chemrefine.state import FailureKind


def _seeded_config(tmp_path: Path, steps: list[StepConfig]) -> Config:
    seed_dir = tmp_path / "seeds"
    io.write_xyz([Atoms("H"), Atoms("H")], ["a", "b"], step_number=0, output_dir=seed_dir)
    return Config(
        template_dir=tmp_path / "templates",
        scratch_dir=tmp_path / "scratch",
        output_dir=tmp_path / "outputs",
        input=seed_dir,
        steps=steps,
    )


def _two_step_config(tmp_path: Path) -> Config:
    return _seeded_config(
        tmp_path,
        [
            StepConfig(step=1, name="screen", engine="fake", operation="opt_sp"),
            StepConfig(step=2, name="refine", engine="fake", operation="opt_sp"),
        ],
    )


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------


def test_resolve_target_by_number(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    step = resolve_target(cfg, 2)
    assert step.step == 2


def test_resolve_target_by_name(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    step = resolve_target(cfg, "refine")
    assert step.step == 2


def test_resolve_target_by_numeric_string(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    step = resolve_target(cfg, "1")
    assert step.step == 1


def test_resolve_target_missing_raises(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    with pytest.raises(ChemRefineError):
        resolve_target(cfg, "missing")


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


def test_execute_run_invalidates_and_re_executes(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    # First run populates caches.
    execute(cfg, Action.RESUME)
    step1_dir = (cfg.output_dir / "step1_screen").resolve()
    assert (step1_dir / "_cache" / "step.json").is_file()
    # RUN should invalidate and re-execute.
    assert execute(cfg, Action.RUN) == 0
    # Cache should exist again after re-execution.
    assert (step1_dir / "_cache" / "step.json").is_file()


def test_execute_resume_keeps_caches(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    execute(cfg, Action.RESUME)
    # Caches now exist; calling resume again should hit them.
    assert execute(cfg, Action.RESUME) == 0


def test_execute_rebuild_cache_invalidates_only_target(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    execute(cfg, Action.RESUME)
    step1_cache = (cfg.output_dir / "step1_screen").resolve() / "_cache" / "step.json"
    step2_cache = (cfg.output_dir / "step2_refine").resolve() / "_cache" / "step.json"
    # Both caches present.
    assert step1_cache.is_file()
    assert step2_cache.is_file()

    # Invalidate only step 2.
    execute(cfg, Action.REBUILD_CACHE, target=2)
    assert step1_cache.is_file()
    assert step2_cache.is_file()  # re-created by the resume run


def test_execute_rerun_target_by_name(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    execute(cfg, Action.RESUME)
    # rerun by name should also be accepted.
    assert execute(cfg, Action.RERUN, target="screen") == 0


def test_execute_rebuild_cache_no_target_uses_last_step(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    execute(cfg, Action.RESUME)
    assert execute(cfg, Action.REBUILD_CACHE) == 0


def test_invalidate_step_removes_cache(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    execute(cfg, Action.RESUME)
    invalidate_step(cfg, cfg.steps[0])
    step1_dir = (cfg.output_dir / "step1_screen").resolve()
    assert not (step1_dir / "_cache" / "step.json").exists()


def test_execute_unknown_action_raises(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    with pytest.raises(ChemRefineError):
        execute(cfg, "not-an-action")


# ---------------------------------------------------------------------------
# rerun — resubmit only the failed jobs, then clear the ledger
# ---------------------------------------------------------------------------


def _register_flaky():
    """A fake engine: ``fail_ids`` produce no output; ``submitted`` logs each
    submitted structure id so tests can assert what was (re)run."""
    from typing import ClassVar

    import numpy as np

    from chemrefine.engines.api import register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, StepResults, Structure

    @register("flaky")
    class _Flaky:
        name = "flaky"
        fail_ids: ClassVar[set[str]] = set()
        submitted: ClassVar[list[str]] = []

        def prepare(self, ctx):
            ctx.step_dir.mkdir(parents=True, exist_ok=True)
            files = []
            for s in ctx.prev_state.structures:
                step = ctx.step_cfg.step
                inp = structure_artifact_path(ctx.step_dir, step, s.id, "inp")
                out = structure_artifact_path(ctx.step_dir, step, s.id, "out")
                inp.parent.mkdir(parents=True, exist_ok=True)
                inp.write_text("in\n", encoding="utf-8")
                files.append((inp, out, s.id))
            return StepInputs(files=tuple(files))

        def submit(self, inputs, ctx):
            for _inp, out, sid in inputs.files:
                _Flaky.submitted.append(sid)
                if sid not in _Flaky.fail_ids:  # failed jobs produce no output
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(f"FINAL ENERGY: {-1.0 - int(sid) * 1e-3}\n", encoding="utf-8")
            return JobBatch(jobs={})

        def parse(self, inputs, ctx):
            seeds = {s.id: s for s in ctx.prev_state.structures}
            out = []
            for _inp, o, sid in inputs.files:
                energy = float(o.read_text().split("FINAL ENERGY:")[1])
                seed = seeds[sid]
                out.append(
                    Structure(
                        id=sid,
                        atoms=seed.atoms,
                        parent_id=seed.parent_id,
                        energy_hartree=energy,
                        forces_ev_per_a=np.zeros((len(seed.atoms), 3)),
                    )
                )
            return StepResults(structures=tuple(out))

    return _Flaky


def test_resume_is_incremental_resubmits_only_failed(tmp_path: Path):
    """An `on_failure: stop` step fails "1" (ledgered, "0" cached) and halts; after
    the engine recovers, `resume` resubmits ONLY "1" (incremental) and clears it."""
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES
    from chemrefine.errors import ChemRefineError

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path,
            [StepConfig(step=1, name="s", engine="flaky", operation="opt_sp", on_failure="stop")],
        )
        step_dir = (cfg.output_dir / "step1_s").resolve()

        eng.fail_ids = {"1"}
        with pytest.raises(ChemRefineError):  # stop halts after caching "0"
            execute(cfg, Action.RESUME)
        assert [(r.structure_id, r.kind) for r in cache.load_failure_records(step_dir)] == [
            ("1", FailureKind.MISSING_OUTPUT)
        ]
        assert {s.id for s in cache.load(step_dir).results.structures} == {"0"}

        eng.fail_ids = set()  # engine recovers
        eng.submitted = []
        assert execute(cfg, Action.RESUME) == 0  # incremental re-attempt
        assert eng.submitted == ["1"]  # only the failed structure resubmitted
        assert cache.load_failure_records(step_dir) == []  # ledger cleared
        assert {s.id for s in cache.load(step_dir).results.structures} == {"0", "1"}
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)


def test_rerun_redoes_whole_step(tmp_path: Path):
    """`rerun` invalidates the step and re-executes it end-to-end (all structures)."""
    from chemrefine.engines.api import ENGINES

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path, [StepConfig(step=1, name="s", engine="flaky", operation="opt_sp")]
        )
        execute(cfg, Action.RESUME)
        eng.submitted = []
        assert execute(cfg, Action.RERUN, target=1) == 0
        assert sorted(eng.submitted) == ["0", "1"]  # whole step redone
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)


def test_rerun_errors_reattempts_only_the_target_step_failures(tmp_path: Path):
    """`rerun-errors N` re-attempts only step N's pending (stop) failures."""
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES
    from chemrefine.errors import ChemRefineError

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path,
            [StepConfig(step=1, name="s", engine="flaky", operation="opt_sp", on_failure="stop")],
        )
        step_dir = (cfg.output_dir / "step1_s").resolve()
        eng.fail_ids = {"1"}
        with pytest.raises(ChemRefineError):
            execute(cfg, Action.RESUME)  # stop halts, "1" pending
        eng.fail_ids, eng.submitted = set(), []
        assert execute(cfg, Action.RERUN_ERRORS, target=1) == 0
        assert eng.submitted == ["1"]  # only the failed structure
        assert cache.load_failure_records(step_dir) == []
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)


def test_rerun_errors_on_skip_step_reports_nothing_pending(tmp_path: Path, caplog):
    """`rerun-errors` on a skip step must not claim it is re-attempting anything.

    The ledger is visibility-only for ``skip``/``best`` — ``run_step`` never
    resubmits those failures, so the log must say nothing is pending instead
    of announcing a re-attempt that silently never happens.
    """
    import logging

    from chemrefine.engines.api import ENGINES

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path,
            [StepConfig(step=1, name="s", engine="flaky", operation="opt_sp", on_failure="skip")],
        )
        eng.fail_ids = {"1"}
        assert execute(cfg, Action.RESUME) == 0
        eng.fail_ids, eng.submitted = set(), []
        with caplog.at_level(logging.INFO, logger="chemrefine.recovery"):
            assert execute(cfg, Action.RERUN_ERRORS, target=1) == 0
        assert eng.submitted == []  # nothing was actually re-attempted
        assert "nothing is pending" in caplog.text
        assert "re-attempting" not in caplog.text
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)


def test_resume_does_not_reattempt_skip_step(tmp_path: Path):
    """A `skip` step records its failures (visible) but `resume` cache-hits it —
    skipped failures are intentional, not pending."""
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path,
            [StepConfig(step=1, name="s", engine="flaky", operation="opt_sp", on_failure="skip")],
        )
        step_dir = (cfg.output_dir / "step1_s").resolve()
        eng.fail_ids = {"1"}
        assert execute(cfg, Action.RESUME) == 0  # skip → continues, no halt
        assert [(r.structure_id, r.kind) for r in cache.load_failure_records(step_dir)] == [
            ("1", FailureKind.MISSING_OUTPUT)
        ]  # failure is visible
        eng.fail_ids, eng.submitted = set(), []
        assert execute(cfg, Action.RESUME) == 0
        assert eng.submitted == []  # cache-hit — the skipped failure is NOT re-run
        assert cache.load_failure_records(step_dir) != []  # ledger kept for visibility
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)


def _register_fake_nms():
    """A two-round NMS fake driven through the engine hook (``nms_input_info``) + the
    frequency values its ``parse`` attaches to each structure; the generic coordinator does
    the displacement. ``resolved`` controls which parents' displaced children resolve;
    ``fail_round1`` produce no round-1 output; ``submitted`` logs round-1 (parent) submissions
    and ``nms_seen`` the parents that reached the NMS stage."""
    from typing import ClassVar

    import numpy as np
    from ase import Atoms

    from chemrefine.engines.api import NmsInputInfo, register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, StepResults, Structure

    @register("fake-nms2")
    class _FakeNms2:
        name = "fake-nms2"
        resolved: ClassVar[set[str]] = set()
        clean: ClassVar[set[str]] = set()
        fail_round1: ClassVar[set[str]] = set()
        submitted: ClassVar[list[str]] = []
        children_submitted: ClassVar[list[str]] = []
        nms_seen: ClassVar[list[str]] = []

        def prepare(self, ctx):
            ctx.step_dir.mkdir(parents=True, exist_ok=True)
            files = []
            for s in ctx.prev_state.structures:
                step = ctx.step_cfg.step
                inp = structure_artifact_path(ctx.step_dir, step, s.id, "inp")
                out = structure_artifact_path(ctx.step_dir, step, s.id, "out")
                inp.parent.mkdir(parents=True, exist_ok=True)
                inp.write_text("in\n", encoding="utf-8")
                files.append((inp, out, s.id))
            return StepInputs(files=tuple(files))

        def submit(self, inputs, ctx):
            # Outputs are content-faithful: markers say what a parse will find, so
            # promotion (which copies a child's file to the parent's canonical path)
            # is visible to `parse` exactly as a real engine's outputs make it.
            for _inp, out, sid in inputs.files:
                if "_m" not in sid:  # a round-1 (parent) submission, not a displaced child
                    _FakeNms2.submitted.append(sid)
                    marker = "CLEAN\n" if sid in _FakeNms2.clean else ""
                else:
                    _FakeNms2.children_submitted.append(sid)
                    parent = sid.split("_m")[0]
                    marker = "RESOLVED\n" if parent in _FakeNms2.resolved else ""
                if sid not in _FakeNms2.fail_round1:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(f"E -1.0\n{marker}", encoding="utf-8")
            return JobBatch(jobs={})

        def parse(self, inputs, ctx):
            seeds = {s.id: s for s in ctx.prev_state.structures}
            out = []
            for _inp, _o, sid in inputs.files:
                seed = seeds.get(sid)
                text = _o.read_text(encoding="utf-8") if _o.is_file() else ""
                # Frequency values ride on the parsed structure (one pass); NMS reads
                # them — off the file's *content*, so a promoted winner at the canonical
                # path reads as the resolved structure it is.
                if "_m" in sid:  # a displaced child
                    imaginary = {} if "RESOLVED" in text else {3: -9.0}
                    modes = None
                else:  # a round-1 parent reaching NMS
                    _FakeNms2.nms_seen.append(sid)
                    # A real parse always carries the full tensor — trivial modes, the
                    # (possibly) imaginary one, and real vibrations `random` can draw —
                    # and only the imaginary set depends on what the output says.
                    modes = np.zeros((1, 3, 8))
                    modes[0, 0, 5] = 0.1
                    modes[0, 0, 6] = 0.2
                    modes[0, 0, 7] = 0.3
                    imaginary = {} if ("CLEAN" in text or "RESOLVED" in text) else {5: -42.0}
                out.append(
                    Structure(
                        id=sid,
                        atoms=seed.atoms if seed else Atoms("H"),
                        energy_hartree=-1.0,
                        terminated_normally=True,
                        converged=True,
                        imaginary_freqs=imaginary,
                        normal_modes=modes,
                    )
                )
            return StepResults(structures=tuple(out))

        def nms_input_info(self, ctx):
            return NmsInputInfo(is_transition_state=False, computes_frequencies=True)

        def artifact_paths(self, ctx, structure_id):
            step = ctx.step_cfg.step
            return (
                ctx.step_dir / structure_id / f"step{step}_{structure_id}.inp",
                ctx.step_dir / structure_id / f"step{step}_{structure_id}.out",
            )

    return _FakeNms2


def _plain_freq_step() -> StepConfig:
    """The same step as :func:`_nms_step` before anyone turns ``nms: true`` on."""
    return StepConfig(step=1, name="s", engine="fake-nms2", operation="freq")


def test_turning_nms_on_computes_only_the_displacement_children(tmp_path: Path):
    """The flip story: a finished non-NMS run + `nms: true` + resume = children, nothing else.

    The nms flag and its options live in the resolution key, so every round-1 row still
    matches — the outputs on disk are provably this configuration's round 1 — and
    resolution runs on top: the clean parent passes through byte-identical, the
    imaginary parent fans out displacement children, and a stale `attemptK/` planted
    from some other history is ignored in favour of a fresh attempt.
    """
    import json

    from chemrefine import cache
    from chemrefine.engines.api import ENGINES

    eng = _register_fake_nms()
    try:
        eng.clean = {"0"}
        eng.resolved = {"1"}
        step_dir = (tmp_path / "outputs" / "step1_s").resolve()
        execute(_seeded_config(tmp_path, [_plain_freq_step()]), Action.RESUME)
        clean_out = step_dir / "0" / "step1_0.out"
        before = clean_out.read_bytes()

        # Mixed history: a stale resolution label that must not be worn.
        planted = step_dir / "1" / "attempt1"
        planted.mkdir(parents=True)
        (planted / "resolution.json").write_text(
            json.dumps({"resolved_from": "bogus"}), encoding="utf-8"
        )

        eng.submitted, eng.children_submitted, eng.nms_seen = [], [], []
        execute(_seeded_config(tmp_path, [_nms_step(1.0)]), Action.RESUME)

        assert eng.submitted == [], "round 1 must not be resubmitted"
        assert {c.split("_m")[0] for c in eng.children_submitted} == {"1"}
        assert clean_out.read_bytes() == before, "the clean parent's output is untouched"
        cached = cache.load(step_dir)
        by_id = {s.id: s for s in cached.results.structures}
        assert set(by_id) == {"0", "1"}
        assert by_id["0"].resolved_from is None, "no borrowed provenance on a passthrough"
        assert by_id["1"].resolved_from is not None and by_id["1"].resolved_from != "bogus"
        assert cache.load_failure_records(step_dir) == []
    finally:
        eng.resolved, eng.clean, eng.fail_round1 = set(), set(), set()
        eng.submitted, eng.children_submitted, eng.nms_seen = [], [], []
        ENGINES.pop("fake-nms2", None)


def test_the_flip_serves_downstream_rows_whose_parent_did_not_change(tmp_path: Path):
    """After the flip, a follow-up step recomputes only the resolved parent's row.

    The clean parent comes out of resolution bit-identical, so its downstream row key
    still matches and its finished output is adopted; the resolved parent carries the
    promoted child's geometry, so exactly its row recomputes.
    """
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES

    eng = _register_fake_nms()
    try:
        eng.clean = {"0"}
        eng.resolved = {"1"}
        follow = StepConfig(step=2, name="refine", engine="fake", operation="opt_sp")
        steps = [_plain_freq_step(), follow]
        execute(_seeded_config(tmp_path, steps), Action.RESUME)
        step2_dir = (tmp_path / "outputs" / "step2_refine").resolve()

        execute(_seeded_config(tmp_path, [_nms_step(1.0), follow]), Action.RESUME)

        assert (step2_dir / "1" / "attempt1").is_dir(), "the changed row was archived and re-run"
        assert not (step2_dir / "0" / "attempt1").exists(), "the unchanged row was adopted"
        assert {s.id for s in cache.load(step2_dir).results.structures} == {"0", "1"}
    finally:
        eng.resolved, eng.clean, eng.fail_round1 = set(), set(), set()
        eng.submitted, eng.children_submitted, eng.nms_seen = [], [], []
        ENGINES.pop("fake-nms2", None)


def test_an_interrupted_nms_step_resumes_by_adopting_round_1(tmp_path: Path):
    """A driver killed after resolution but before the cache write costs a re-read, not a re-run.

    This deliberately replaces the old doctrine ("an interrupted NMS step falls back to
    the full re-run"): the rows prove round 1, the criterion matches, so the promoted
    winners pass through with their provenance intact and nothing submits at all.
    """
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES

    eng = _register_fake_nms()
    try:
        eng.clean = {"0"}
        eng.resolved = {"1"}
        step_dir = (tmp_path / "outputs" / "step1_s").resolve()
        execute(_seeded_config(tmp_path, [_nms_step(1.0)]), Action.RESUME)
        resolved_from = {s.id: s.resolved_from for s in cache.load(step_dir).results.structures}
        cache.invalidate(step_dir)  # step.json is written last — this is the interruption

        eng.submitted, eng.children_submitted, eng.nms_seen = [], [], []
        execute(_seeded_config(tmp_path, [_nms_step(1.0)]), Action.RESUME)

        assert eng.submitted == [] and eng.children_submitted == []
        survivors = {s.id: s for s in cache.load(step_dir).results.structures}
        assert set(survivors) == {"0", "1"}
        assert survivors["1"].resolved_from == resolved_from["1"], "trusted provenance survives"
    finally:
        eng.resolved, eng.clean, eng.fail_round1 = set(), set(), set()
        eng.submitted, eng.children_submitted, eng.nms_seen = [], [], []
        ENGINES.pop("fake-nms2", None)


def test_a_criterion_change_re_resolves_without_resubmitting_round_1(tmp_path: Path):
    """target: minimum → ts re-runs the *resolution*, and only that.

    The rows are untouched by the criterion, so round 1 adopts; the stale resolutions
    are not trusted (their labels were written for the old criterion); and what cannot
    reach the new target honestly ledgers as unresolved — here both parents, whose
    adopted outputs carry no imaginary mode to displace along.
    """
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES

    eng = _register_fake_nms()
    try:
        eng.clean = {"0"}
        eng.resolved = {"1"}
        step_dir = (tmp_path / "outputs" / "step1_s").resolve()
        execute(_seeded_config(tmp_path, [_nms_step(1.0)]), Action.RESUME)

        eng.resolved = set()
        eng.submitted, eng.children_submitted, eng.nms_seen = [], [], []
        ts_step = _nms_step(1.0)
        ts_step = ts_step.model_copy(
            update={"options": {"target": "ts", "displacement_value": 1.0}}
        )
        execute(_seeded_config(tmp_path, [ts_step]), Action.RESUME)

        assert eng.submitted == [], "round 1 must not be resubmitted"
        records = cache.load_failure_records(step_dir)
        assert {(r.structure_id, r.kind) for r in records} == {
            ("0", FailureKind.UNRESOLVED_NMS),
            ("1", FailureKind.UNRESOLVED_NMS),
        }
    finally:
        eng.resolved, eng.clean, eng.fail_round1 = set(), set(), set()
        eng.submitted, eng.children_submitted, eng.nms_seen = [], [], []
        ENGINES.pop("fake-nms2", None)


def test_a_random_target_flip_fans_out_every_parent(tmp_path: Path):
    """`random` is exploration, not cleanup — the flip fans children for clean parents too."""
    from chemrefine.engines.api import ENGINES

    eng = _register_fake_nms()
    try:
        eng.clean = {"0"}
        execute(_seeded_config(tmp_path, [_plain_freq_step()]), Action.RESUME)

        eng.submitted, eng.children_submitted, eng.nms_seen = [], [], []
        random_step = _nms_step(1.0).model_copy(
            update={"options": {"target": "random", "displacement_value": 1.0}}
        )
        execute(_seeded_config(tmp_path, [random_step]), Action.RESUME)

        assert eng.submitted == []
        assert {c.split("_m")[0] for c in eng.children_submitted} == {"0", "1"}
    finally:
        eng.resolved, eng.clean, eng.fail_round1 = set(), set(), set()
        eng.submitted, eng.children_submitted, eng.nms_seen = [], [], []
        ENGINES.pop("fake-nms2", None)


def _nms_step(displacement: float, on_failure: str = "skip") -> StepConfig:
    return StepConfig(
        step=1,
        name="s",
        engine="fake-nms2",
        operation="freq",
        nms=True,
        options={"target": "minimum", "displacement_value": displacement},
        on_failure=on_failure,
    )


def test_resume_after_tuning_reattempts_only_unresolved(tmp_path: Path):
    """Tuning displacement_value (reuse fingerprint unchanged) + resume reuses
    round-1 and re-runs NMS for ONLY the parent the ledger records as unresolved."""
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES

    eng = _register_fake_nms()
    try:
        eng.resolved = {"0"}  # "1" stays unresolved on the first run
        cfg1 = _seeded_config(tmp_path, [_nms_step(1.0)])
        step_dir = (cfg1.output_dir / "step1_s").resolve()
        execute(cfg1, Action.RESUME)
        assert [(r.structure_id, r.kind) for r in cache.load_failure_records(step_dir)] == [
            ("1", FailureKind.UNRESOLVED_NMS)
        ]
        # Unified model: the survivor keeps the parent's id (resolved geometry), not a child id.
        assert {s.id for s in cache.load(step_dir).results.structures} == {"0"}

        eng.resolved = {"0", "1"}  # new distance resolves "1"
        eng.submitted, eng.children_submitted, eng.nms_seen = [], [], []
        execute(_seeded_config(tmp_path, [_nms_step(2.0)]), Action.RESUME)
        assert eng.submitted == []  # round-1 freq reused, not resubmitted
        # The incremental route re-parses every adopted row (a read, not a job) — the
        # promoted winner at "0"'s canonical path passes straight through — and fans
        # out children for exactly the parent the previous run left unresolved.
        assert eng.nms_seen == ["0", "1"]
        assert {c.split("_m")[0] for c in eng.children_submitted} == {"1"}
        assert cache.load_failure_records(step_dir) == []
        assert {s.id for s in cache.load(step_dir).results.structures} == {"0", "1"}
    finally:
        eng.resolved, eng.clean, eng.fail_round1 = set(), set(), set()
        eng.submitted, eng.children_submitted, eng.nms_seen = [], [], []
        ENGINES.pop("fake-nms2", None)


def test_reattempt_resubmits_missing_round1(tmp_path: Path):
    """A parent whose round-1 output is missing gets its round-1 resubmitted on
    the next resume (NMS-unresolved parents do not)."""
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES
    from chemrefine.errors import ChemRefineError

    eng = _register_fake_nms()
    try:
        eng.resolved = {"0", "1"}
        eng.fail_round1 = {"1"}  # "1" produces no round-1 output
        cfg = _seeded_config(tmp_path, [_nms_step(1.0, on_failure="stop")])
        step_dir = (cfg.output_dir / "step1_s").resolve()
        with pytest.raises(ChemRefineError):  # stop halts on the missing round-1
            execute(cfg, Action.RESUME)
        assert [(r.structure_id, r.kind) for r in cache.load_failure_records(step_dir)] == [
            ("1", FailureKind.MISSING_OUTPUT)
        ]

        eng.fail_round1 = set()  # round-1 recovers
        eng.submitted = []
        execute(cfg, Action.RESUME)  # same config → full-valid + ledger → reattempt_nms
        assert eng.submitted == ["1"]  # round-1 resubmitted only for the missing one
        assert cache.load_failure_records(step_dir) == []
        assert {s.id for s in cache.load(step_dir).results.structures} == {"0", "1"}
    finally:
        eng.resolved, eng.clean, eng.fail_round1 = set(), set(), set()
        eng.submitted, eng.children_submitted, eng.nms_seen = [], [], []
        ENGINES.pop("fake-nms2", None)


def test_rebuild_cache_reparses_without_submitting(tmp_path: Path):
    """`rebuild-cache` rebuilds a step's cache from existing outputs — no submit."""
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path, [StepConfig(step=1, name="s", engine="flaky", operation="opt_sp")]
        )
        step_dir = (cfg.output_dir / "step1_s").resolve()
        execute(cfg, Action.RESUME)  # produces outputs + cache
        cache.invalidate(step_dir)  # drop the cache, keep outputs on disk
        eng.submitted = []
        assert execute(cfg, Action.REBUILD_CACHE, target=1) == 0
        assert eng.submitted == []  # parse-only, nothing resubmitted
        assert {s.id for s in cache.load(step_dir).results.structures} == {"0", "1"}
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)


# ---------------------------------------------------------------------------
# The recovery matrix — this test IS the spec
# ---------------------------------------------------------------------------
#
# Six CLI actions map to a handful of per-step execution paths, and which path a
# given step takes is decided at three different depths (recovery.execute,
# pipeline.run, step.run_step) from two nullable ints. Nothing else states the
# mapping, so it is asserted here in one place: for each action, which steps
# re-execute their engine and which are served from cache.


def _recording_engine():
    """A fake engine that records the steps it actually executed."""
    from typing import ClassVar

    import numpy as np

    from chemrefine.engines.api import register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, StepResults, Structure

    @register("recorder")
    class _Recorder:
        name: ClassVar[str] = "recorder"
        submitted: ClassVar[list[int]] = []

        def prepare(self, ctx):
            ctx.step_dir.mkdir(parents=True, exist_ok=True)
            files = []
            for s in ctx.prev_state.structures:
                step = ctx.step_cfg.step
                inp = structure_artifact_path(ctx.step_dir, step, s.id, "inp")
                out = structure_artifact_path(ctx.step_dir, step, s.id, "out")
                inp.parent.mkdir(parents=True, exist_ok=True)
                inp.write_text("in\n", encoding="utf-8")
                out.write_text("out\n", encoding="utf-8")
                files.append((inp, out, s.id))
            return StepInputs(files=tuple(files))

        def submit(self, inputs, ctx):
            _Recorder.submitted.append(ctx.step_cfg.step)
            return JobBatch(jobs={})

        def parse(self, inputs, ctx):
            seeds = {s.id: s for s in ctx.prev_state.structures}
            out = []
            for _inp, _o, sid in inputs.files:
                seed = seeds[sid]
                out.append(
                    Structure(
                        id=sid,
                        atoms=seed.atoms,
                        parent_id=seed.parent_id,
                        energy_hartree=-1.0 - len(sid) * 1e-3,
                        forces_ev_per_a=np.zeros((len(seed.atoms), 3)),
                    )
                )
            return StepResults(structures=tuple(out))

    return _Recorder


@pytest.mark.parametrize(
    ("action", "target", "expected_submits"),
    [
        # run: every cache is invalidated, so both steps execute again.
        (Action.RUN, None, [1, 2]),
        # resume: both caches are valid and clean, so nothing re-executes.
        (Action.RESUME, None, []),
        # rerun N: only that step's cache is dropped; the other cache-hits. Step 2
        # follows because step 1's survivors are unchanged, so its fingerprint holds.
        (Action.RERUN, 1, [1]),
        (Action.RERUN, 2, [2]),
        # rerun-errors N: nothing is pending, so it degrades to a plain resume.
        (Action.RERUN_ERRORS, 2, []),
        # rebuild-cache N: re-parses from disk. No submission, by definition.
        (Action.REBUILD_CACHE, 2, []),
        # Both scoped actions aimed at a step that is *not* the last one. Every row above
        # targets the final step, which is the one arrangement where "what happens after the
        # target" cannot be observed.
        (Action.RERUN_ERRORS, 1, []),
        (Action.REBUILD_CACHE, 1, []),
    ],
    ids=[
        "run-reexecutes-everything",
        "resume-hits-every-cache",
        "rerun-1-redoes-only-step-1",
        "rerun-2-redoes-only-step-2",
        "rerun-errors-with-nothing-pending-is-resume",
        "rebuild-cache-never-submits",
        "rerun-errors-on-a-non-last-step",
        "rebuild-cache-on-a-non-last-step",
    ],
)
def test_recovery_matrix(tmp_path: Path, action, target, expected_submits):
    """Each action's per-step routing, asserted end to end."""
    from chemrefine.engines.api import ENGINES

    eng = _recording_engine()
    try:
        cfg = _seeded_config(
            tmp_path,
            [
                StepConfig(step=1, name="screen", engine="recorder", operation="opt_sp"),
                StepConfig(step=2, name="refine", engine="recorder", operation="opt_sp"),
            ],
        )
        execute(cfg, Action.RESUME)  # prime both caches
        eng.submitted.clear()

        assert execute(cfg, action, target=target) == 0

        assert sorted(eng.submitted) == expected_submits
    finally:
        eng.submitted.clear()
        ENGINES.pop("recorder", None)


def test_rerun_errors_repairs_the_target_then_runs_the_rest(tmp_path: Path):
    """Repairing step N is only half the command; the run has to go on.

    The steps after the target are exactly the ones that never ran — the halt that left the
    failures pending is what stopped the pipeline there — so they have no cache. Held to
    ``CACHE_ONLY`` they raise for a cache that cannot exist, and the command fails *after*
    repairing what it was pointed at, having done its work and reported an error.
    """
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES
    from chemrefine.errors import ChemRefineError

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path,
            [
                StepConfig(
                    step=1, name="one", engine="flaky", operation="opt_sp", on_failure="stop"
                ),
                StepConfig(step=2, name="two", engine="flaky", operation="opt_sp"),
            ],
        )
        eng.fail_ids = {"1"}
        with pytest.raises(ChemRefineError):
            execute(cfg, Action.RESUME)  # halts at step 1; step 2 never runs
        assert not (cfg.output_dir / "step2_two").exists()

        eng.fail_ids, eng.submitted = set(), []
        assert execute(cfg, Action.RERUN_ERRORS, target=1) == 0

        assert eng.submitted[0] == "1", "the target's failed structure is re-attempted first"
        assert set(eng.submitted[1:]) == {"0", "1"}, "then step 2 runs, for both survivors"
        assert not cache.load_failure_records((cfg.output_dir / "step1_one").resolve())
        assert cache.load((cfg.output_dir / "step2_two").resolve()) is not None
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)


def test_rerun_errors_archives_what_it_replaces_and_leaves_the_rest(tmp_path: Path):
    """A re-attempt seals the prior attempt; a structure it does not re-run is untouched.

    The invariant behind every recovery path: what is about to be overwritten moves into
    ``attemptK/`` first, so no run is lost, and nothing else is disturbed.
    """
    from chemrefine.engines.api import ENGINES
    from chemrefine.errors import ChemRefineError

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path,
            [StepConfig(step=1, name="one", engine="flaky", operation="opt_sp", on_failure="stop")],
        )
        step_dir = (cfg.output_dir / "step1_one").resolve()
        eng.fail_ids = {"1"}
        with pytest.raises(ChemRefineError):
            execute(cfg, Action.RESUME)
        first_input = (step_dir / "1" / "step1_1.inp").read_text(encoding="utf-8")

        eng.fail_ids = set()
        assert execute(cfg, Action.RERUN_ERRORS, target=1) == 0

        archived = step_dir / "1" / "attempt1" / "step1_1.inp"
        assert archived.is_file(), "the re-attempted structure's prior input is sealed away"
        assert archived.read_text(encoding="utf-8") == first_input
        assert (step_dir / "1" / "step1_1.out").is_file(), "and canonical holds the new run"
        assert not list((step_dir / "0").glob("attempt*")), (
            "the structure that succeeded was never re-run, so nothing of it was moved"
        )
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)


def test_rebuild_cache_stops_the_tail_where_it_cannot_serve_and_names_the_repair(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """Past its target, ``rebuild-cache N`` re-reports from caches — submitting nothing —
    and stops quietly at the first cache it cannot serve, naming the cheapest repair.

    Step 2's cache document is deleted while its manifest (whose stamped fingerprint still
    matches this configuration) survives: exactly the state ``rebuild-cache 2`` exists to
    repair, so that is the command the stop must name. A plain ``resume`` would also work
    here, but for an NMS or artifact step it would *recompute* — the hint exists to steer
    away from that.
    """
    import logging

    from chemrefine.engines.api import ENGINES

    eng = _recording_engine()
    try:
        cfg = _seeded_config(
            tmp_path,
            [
                StepConfig(step=1, name="one", engine="recorder", operation="opt_sp"),
                StepConfig(step=2, name="two", engine="recorder", operation="opt_sp"),
            ],
        )
        execute(cfg, Action.RESUME)
        eng.submitted.clear()
        (cfg.output_dir / "step2_two" / "_cache" / "step.json").unlink()

        with caplog.at_level(logging.INFO):
            assert execute(cfg, Action.REBUILD_CACHE, target=1) == 0
        assert eng.submitted == [], "a rebuild submits nothing, target and tail alike"
        assert not (cfg.output_dir / "step2_two" / "_cache" / "step.json").exists(), (
            "the tail walk reads caches; it must not rebuild one for a step it is not targeting"
        )
        assert "rebuild-cache 2" in caplog.text, (
            "outputs on disk still match this configuration, so the stop names the "
            "command that re-adopts them without recomputing"
        )
    finally:
        eng.submitted.clear()
        ENGINES.pop("recorder", None)


#: Actions the matrix above cannot express, and the test that covers each instead. The
#: matrix runs one config for every row; `rebuild-nms` resolves its target by *what a step
#: is*, so it needs a config with an NMS step in it and cannot share that one.
_COVERED_ELSEWHERE = {Action.REBUILD_NMS: "test_rebuild_nms_*"}


def test_recovery_matrix_covers_every_action():
    """A new Action must be given a row above, or named here with the test that covers it.

    Either way it is covered deliberately rather than left to inherit another action's
    routing by accident.
    """
    covered = {
        Action.RUN,
        Action.RESUME,
        Action.RERUN,
        Action.RERUN_ERRORS,
        Action.REBUILD_CACHE,
    }
    assert covered | set(_COVERED_ELSEWHERE) == set(Action)
    assert not covered & set(_COVERED_ELSEWHERE), "an action is covered in two places"


# ---------------------------------------------------------------------------
# rebuild-nms — the rebuild aimed at the NMS step
# ---------------------------------------------------------------------------


def _nms_config(tmp_path: Path, *, second_nms: bool = False) -> Config:
    """A two-step config whose *first* step is the NMS one.

    First on purpose: the last step is what every other action defaults to, so a command that
    finds the NMS step by looking at it can only be told apart from one that takes the last
    when the two are different steps.
    """
    steps = [
        _nms_step(1.0),
        StepConfig(
            step=2,
            name="after",
            engine="fake-nms2",
            operation="opt_sp",
            nms=second_nms,
            options={"target": "minimum", "displacement_value": 1.0} if second_nms else {},
        ),
    ]
    return _seeded_config(tmp_path, steps)


@pytest.mark.parametrize("target", [None, 1], ids=["found-by-nms-flag", "named-explicitly"])
def test_rebuild_nms_rebuilds_the_nms_step_without_submitting(tmp_path: Path, target):
    """The command's whole point: re-read the round-2 children, do not recompute round 1.

    Round 1 is a frequency calculation — the expensive part of an NMS step, and already on
    disk. Re-running it is what `rerun` is for; this re-parses it and re-derives the
    resolution from the displaced children in their `attemptK/`.

    Both ways of reaching the step behave the same, which is the point of naming one: the
    v1 flag took a step number (`--rebuild_nms 2`) and still translates to this.
    """
    from chemrefine.engines.api import ENGINES

    eng = _register_fake_nms()
    try:
        cfg = _nms_config(tmp_path)
        eng.resolved = {"0", "1"}
        execute(cfg, Action.RESUME)
        document = (cfg.output_dir / "step1_s" / "_cache" / "step.json").resolve()
        before = document.stat().st_mtime_ns
        eng.submitted, eng.nms_seen = [], []

        assert execute(cfg, Action.REBUILD_NMS, target=target) == 0

        assert eng.submitted == [], "a rebuild submits nothing"
        assert document.stat().st_mtime_ns != before, "the NMS step's cache was rewritten"
        assert eng.nms_seen, "round 1 was re-parsed, which is where the frequencies come from"
    finally:
        eng.resolved, eng.submitted, eng.nms_seen = set(), [], []
        ENGINES.pop("fake-nms2", None)


def test_rebuild_nms_finds_the_nms_step_rather_than_the_last(tmp_path: Path):
    """With no target it goes by `nms: true`, where every other action takes the last step."""
    from chemrefine.engines.api import ENGINES

    eng = _register_fake_nms()
    try:
        cfg = _nms_config(tmp_path)
        eng.resolved = {"0", "1"}
        execute(cfg, Action.RESUME)
        step2_doc = (cfg.output_dir / "step2_after" / "_cache" / "step.json").resolve()
        before = step2_doc.stat().st_mtime_ns

        assert execute(cfg, Action.REBUILD_NMS) == 0

        assert step2_doc.stat().st_mtime_ns == before, (
            "step 2 is the last step but not the NMS one, so it was never touched"
        )
    finally:
        eng.resolved, eng.submitted, eng.nms_seen = set(), [], []
        ENGINES.pop("fake-nms2", None)


def test_rebuild_nms_refuses_a_step_that_does_no_nms(tmp_path: Path):
    """Naming a plain step is a request `rebuild-cache` answers, so say so."""
    from chemrefine.engines.api import ENGINES

    _register_fake_nms()  # registered so the config validates; the class itself is unused
    try:
        cfg = _nms_config(tmp_path)
        with pytest.raises(ChemRefineError, match="rebuild-cache"):
            execute(cfg, Action.REBUILD_NMS, target=2)
    finally:
        ENGINES.pop("fake-nms2", None)


def test_rebuild_nms_without_an_nms_step_says_so(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    with pytest.raises(ChemRefineError, match="no step sets `nms: true`"):
        execute(cfg, Action.REBUILD_NMS)


def test_rebuild_nms_with_two_nms_steps_asks_which(tmp_path: Path):
    """ "The NMS step" names nothing when there are two, so it asks instead of guessing."""
    from chemrefine.engines.api import ENGINES

    _register_fake_nms()  # registered so the config validates; the class itself is unused
    try:
        cfg = _nms_config(tmp_path, second_nms=True)
        with pytest.raises(ChemRefineError, match="more than one step"):
            execute(cfg, Action.REBUILD_NMS)
    finally:
        ENGINES.pop("fake-nms2", None)


def _coverage_cfg(tmp_path: Path, **step_over) -> Config:
    """A one-step fake-engine Config rooted at tmp_path."""
    return Config(
        output_dir=tmp_path / "outputs",
        steps=[StepConfig(step=1, engine="fake", operation="opt_sp", **step_over)],
    )


# --- recovery: rerun-errors with nothing pending ----------------------------


def test_rerun_errors_logs_when_no_failures(tmp_path: Path, monkeypatch):
    from chemrefine import recovery

    cfg = _coverage_cfg(tmp_path)
    monkeypatch.setattr(recovery.pipeline, "run", lambda *a, **k: [])
    recovery._action_rerun_errors(cfg, None)  # last step, no ledger → "no failures" branch
