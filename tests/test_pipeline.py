"""Tests for the high-level pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms

from chemrefine import io, pipeline
from chemrefine.config import Config, StepConfig
from chemrefine.errors import ChemRefineError, ConfigError
from chemrefine.step import RunPlan, StepMode


def _h2() -> Atoms:
    return Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [0.74, 0, 0]])


def _config(tmp_path: Path, **overrides) -> Config:
    base = {
        "template_dir": tmp_path / "templates",
        "scratch_dir": tmp_path / "scratch",
        "output_dir": tmp_path / "outputs",
        "steps": [StepConfig(step=1, engine="fake", operation="opt_sp")],
    }
    base.update(overrides)
    return Config(**base)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_from_xyz_file(tmp_path: Path):
    io.write_xyz([_h2()], ["seed"], step_number=0, output_dir=tmp_path)
    seed = tmp_path / "step0_seed.xyz"
    cfg = _config(tmp_path, input=seed)
    state = pipeline.bootstrap(cfg)
    assert len(state.structures) == 1
    assert state.structures[0].id == "0"


def test_bootstrap_from_multiframe_xyz_seeds_every_frame(tmp_path: Path):
    """A multi-conformer seed file must yield one structure per frame.

    ASE's default ``read`` keeps only the last frame — relying on it would
    silently drop every other conformer the user supplied.
    """
    seed = tmp_path / "ensemble.xyz"
    frame = "2\nH2 frame {i}\nH 0.0 0.0 0.0\nH {x} 0.0 0.0\n"
    seed.write_text(
        "".join(frame.format(i=i, x=0.70 + 0.02 * i) for i in range(3)), encoding="utf-8"
    )
    cfg = _config(tmp_path, input=seed)
    state = pipeline.bootstrap(cfg)
    assert [s.id for s in state.structures] == ["0", "1", "2"]
    # Frames are distinct geometries, in file order.
    assert state.structures[0].atoms.get_positions()[1][0] == pytest.approx(0.70)
    assert state.structures[2].atoms.get_positions()[1][0] == pytest.approx(0.74)


def test_bootstrap_from_xyz_with_trailing_blank_lines(tmp_path: Path):
    """Editors and ORCA leave trailing blank lines on real .xyz files.

    Seeding must not use ASE's naive ``format="xyz"`` parser, which loops
    ``int(lines.pop(0))`` over every remaining line and dies on the blank tail
    with ``invalid literal for int(): '\\n'``. ``extxyz`` tolerates it.
    """
    seed = tmp_path / "trailing.xyz"
    seed.write_text("2\nsymmetry c1\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n\n\n", encoding="utf-8")
    cfg = _config(tmp_path, input=seed)
    state = pipeline.bootstrap(cfg)
    assert [s.id for s in state.structures] == ["0"]
    assert state.structures[0].atoms.get_chemical_formula() == "H2"


def test_bootstrap_from_directory(tmp_path: Path):
    seed_dir = tmp_path / "seeds"
    io.write_xyz([_h2(), _h2()], ["a", "b"], step_number=0, output_dir=seed_dir)
    cfg = _config(tmp_path, input=seed_dir)
    state = pipeline.bootstrap(cfg)
    assert len(state.structures) == 2
    assert [s.id for s in state.structures] == ["0", "1"]


def test_bootstrap_from_directory_seeds_every_frame_of_multiframe_files(tmp_path: Path):
    """A multi-frame file inside a seed directory yields one structure per frame.

    Directory seeding must not use ASE's default read (last frame only),
    silently dropping every other conformer — the exact bug the single-file
    path already guards against.
    """
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    frame = "2\nH2 frame {i}\nH 0.0 0.0 0.0\nH {x} 0.0 0.0\n"
    (seed_dir / "a.xyz").write_text(
        "".join(frame.format(i=i, x=0.70 + 0.02 * i) for i in range(2)), encoding="utf-8"
    )
    (seed_dir / "b.xyz").write_text(frame.format(i=9, x=0.80), encoding="utf-8")
    cfg = _config(tmp_path, input=seed_dir)
    state = pipeline.bootstrap(cfg)
    assert [s.id for s in state.structures] == ["0", "1", "2"]
    # a.xyz's first frame, then its second, then b.xyz's single frame.
    assert state.structures[0].atoms.get_positions()[1][0] == pytest.approx(0.70)
    assert state.structures[1].atoms.get_positions()[1][0] == pytest.approx(0.72)
    assert state.structures[2].atoms.get_positions()[1][0] == pytest.approx(0.80)


def test_bootstrap_falls_back_to_templates_step1_xyz(tmp_path: Path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    seed_path = template_dir / "step1.xyz"
    # Write a minimal xyz file directly
    seed_path.write_text("2\nH2\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n", encoding="utf-8")
    cfg = _config(tmp_path, input=None)
    state = pipeline.bootstrap(cfg)
    assert len(state.structures) == 1


def test_bootstrap_missing_input_raises(tmp_path: Path):
    cfg = _config(tmp_path, input=None)
    with pytest.raises(ConfigError):
        pipeline.bootstrap(cfg)


def test_bootstrap_unsupported_format_raises(tmp_path: Path):
    bad = tmp_path / "input.json"
    bad.write_text("{}", encoding="utf-8")
    cfg = _config(tmp_path, input=bad)
    with pytest.raises(ConfigError):
        pipeline.bootstrap(cfg)


def test_bootstrap_from_smiles_csv(tmp_path: Path):
    csv = tmp_path / "smiles.csv"
    csv.write_text("smiles\nC\nCC\n", encoding="utf-8")
    cfg = _config(tmp_path, input=csv)
    state = pipeline.bootstrap(cfg)
    assert len(state.structures) == 2


def test_bootstrap_from_smiles_csv_empty_raises(tmp_path: Path):
    """A CSV with only the header (no SMILES rows) must surface as ConfigError."""
    csv = tmp_path / "smiles.csv"
    csv.write_text("smiles\n", encoding="utf-8")
    cfg = _config(tmp_path, input=csv)
    with pytest.raises(ConfigError):
        pipeline.bootstrap(cfg)


def test_run_stops_early_when_a_step_produces_no_survivors(tmp_path: Path):
    """A step whose sample method filters every survivor must halt the pipeline."""
    from chemrefine.engines.api import ENGINES, register
    from chemrefine.state import PipelineState, StepResults

    @register("empty-fake")
    class _EmptyEngine:
        name = "empty-fake"

        def prepare(self, ctx):
            ctx.step_dir.mkdir(parents=True, exist_ok=True)
            from chemrefine.state import StepInputs

            return StepInputs(files=())

        def submit(self, inputs, ctx):
            from chemrefine.state import JobBatch

            return JobBatch(jobs={})

        def parse(self, inputs, ctx):
            return StepResults(structures=())  # no survivors

    try:
        seed_dir = tmp_path / "seeds"
        io.write_xyz([_h2()], ["a"], step_number=0, output_dir=seed_dir)
        cfg = _config(
            tmp_path,
            input=seed_dir,
            steps=[
                StepConfig(step=1, engine="empty-fake", operation="opt_sp"),
                StepConfig(step=2, engine="empty-fake", operation="opt_sp"),
            ],
        )
        outcomes = pipeline.run(cfg)
        # Step 1 produced no survivors → pipeline stops at length 1.
        assert len(outcomes) == 1
        _ = PipelineState  # silence unused import
    finally:
        ENGINES.pop("empty-fake", None)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_executes_each_step_in_order(tmp_path: Path):
    seed = tmp_path / "step0_seed.xyz"
    io.write_xyz([_h2()], ["seed"], step_number=0, output_dir=tmp_path)
    cfg = _config(
        tmp_path,
        input=seed,
        steps=[
            StepConfig(step=1, engine="fake", operation="opt_sp"),
            StepConfig(step=2, engine="fake", operation="opt_sp"),
        ],
    )
    outcomes = pipeline.run(cfg)
    assert len(outcomes) == 2
    assert all(o.cache_hit is False for o in outcomes)


def test_run_writes_steps_csv(tmp_path: Path):
    """A full run emits the cumulative ``steps.csv`` energy summary."""
    seed = tmp_path / "step0_seed.xyz"
    io.write_xyz([_h2()], ["seed"], step_number=0, output_dir=tmp_path)
    cfg = _config(
        tmp_path,
        input=seed,
        steps=[
            StepConfig(step=1, engine="fake", operation="opt_sp"),
            StepConfig(step=2, engine="fake", operation="opt_sp"),
        ],
    )
    pipeline.run(cfg)

    csv_path = cfg.output_dir / "steps.csv"
    assert csv_path.is_file()
    text = csv_path.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    for column in ("Step", "Conformer", "Energy (Hartree)", "% Cumulative"):
        assert column in header
    # Both steps appended (header + one survivor row per step).
    assert "1," in text and "2," in text


def test_run_aligns_csv_and_cache_for_more_than_ten_structures(tmp_path: Path):
    """With ≥10 structures every id maps to its *own* energy, in the cache and in
    ``steps.csv`` alike: a lexical sort puts "10" before "2" and desyncs the rows from
    the energies they name."""
    import pandas as pd
    from fake_engine import _fake_energy

    from chemrefine import cache

    seed_dir = tmp_path / "seeds"
    io.write_xyz(
        [_h2() for _ in range(12)], [str(i) for i in range(12)], step_number=0, output_dir=seed_dir
    )
    cfg = _config(tmp_path, input=seed_dir)

    # Seeds bootstrap in natural order (10/11 after 2), not lexical.
    assert [s.id for s in pipeline.bootstrap(cfg).structures] == [str(i) for i in range(12)]

    pipeline.run(cfg)
    expected = {str(i): _fake_energy(str(i)) for i in range(12)}

    cached = cache.load((cfg.output_dir / "step1").resolve())
    assert cached is not None
    cache_map = {s.id: s.energy_hartree for s in cached.results.structures}
    assert cache_map.keys() == expected.keys()
    for sid, energy in expected.items():
        assert cache_map[sid] == pytest.approx(energy)

    df = pd.read_csv(cfg.output_dir / "steps.csv")
    csv_map = {str(c): e for c, e in zip(df["Conformer"], df["Energy (Hartree)"], strict=True)}
    assert csv_map.keys() == expected.keys()  # all 12, including "10"/"11"
    for sid, energy in expected.items():
        assert csv_map[sid] == pytest.approx(energy)  # each id paired with its own energy


def test_run_threads_state_between_steps(tmp_path: Path):
    # Two seed files in a directory.
    seed_dir = tmp_path / "seeds"
    io.write_xyz([_h2(), _h2()], ["a", "b"], step_number=0, output_dir=seed_dir)
    cfg = _config(
        tmp_path,
        input=seed_dir,
        steps=[
            StepConfig(
                step=1,
                engine="fake",
                operation="opt_sp",
                sample={"method": "min", "count": 1},
            ),
            StepConfig(step=2, engine="fake", operation="opt_sp"),
        ],
    )
    outcomes = pipeline.run(cfg)
    # Step 1 filters 2 -> 1; step 2 keeps that 1.
    assert len(outcomes[0].state.structures) == 1
    assert len(outcomes[1].state.structures) == 1


def test_run_second_call_hits_cache(tmp_path: Path):
    seed_dir = tmp_path / "seeds"
    io.write_xyz([_h2()], ["a"], step_number=0, output_dir=seed_dir)
    cfg = _config(
        tmp_path,
        input=seed_dir,
        steps=[StepConfig(step=1, engine="fake", operation="opt_sp")],
    )
    first = pipeline.run(cfg)
    second = pipeline.run(cfg)
    assert first[0].cache_hit is False
    assert second[0].cache_hit is True


def test_run_after_editing_seed_invalidates_cache(tmp_path: Path):
    """Editing the seed geometry must re-execute the step on the next run.

    Seed IDs are positional, so an edited file yields the *same* parent IDs —
    only the content digest in the fingerprint catches the change. Without it,
    ``resume`` silently reuses results computed from the old geometry.
    """
    seed = tmp_path / "input.xyz"
    seed.write_text("2\nH2\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n", encoding="utf-8")
    cfg = _config(
        tmp_path,
        input=seed,
        steps=[StepConfig(step=1, engine="fake", operation="opt_sp")],
    )
    assert pipeline.run(cfg)[0].cache_hit is False
    # Same path, same structure count, same IDs — different coordinates.
    seed.write_text("2\nH2 stretched\nH 0.0 0.0 0.0\nH 0.90 0.0 0.0\n", encoding="utf-8")
    assert pipeline.run(cfg)[0].cache_hit is False


def test_run_stops_when_no_survivors(tmp_path: Path):
    """pipeline.run raises ConfigError when the seed directory is empty."""
    # Force an empty seed directory — the pipeline has no structures to
    # process and raises ConfigError immediately rather than skipping steps.
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    cfg = _config(tmp_path, input=seed_dir)
    with pytest.raises(ConfigError):
        pipeline.run(cfg)


# ---------------------------------------------------------------------------
# steps.csv reports the energy the step actually filtered on
# ---------------------------------------------------------------------------


def _register_thermo_engine():
    """A fake engine whose Gibbs energies rank the structures *opposite* to electronic.

    The inversion is the point: if the report were still reading the electronic
    energy, its ordering and weights would visibly disagree with the survivors.
    """
    from typing import ClassVar

    import numpy as np

    from chemrefine.engines.api import register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, StepResults, Structure

    @register("fake-thermo")
    class _ThermoEngine:
        name: ClassVar[str] = "fake-thermo"

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
            return JobBatch(jobs={})

        def parse(self, inputs, ctx):
            seeds = {s.id: s for s in ctx.prev_state.structures}
            out = []
            for _inp, _o, sid in inputs.files:
                n = int(sid)
                seed = seeds[sid]
                out.append(
                    Structure(
                        id=sid,
                        atoms=seed.atoms,
                        parent_id=seed.parent_id,
                        energy_hartree=-1.0 - n,  # "2" is lowest electronic
                        gibbs_hartree=-10.0 + n,  # "0" is lowest Gibbs
                        forces_ev_per_a=np.zeros((len(seed.atoms), 3)),
                    )
                )
            return StepResults(structures=tuple(out))

    return _ThermoEngine


def test_steps_csv_reports_the_energy_the_step_filtered_on(tmp_path: Path):
    """A Gibbs-filtered step must not be summarised with electronic energies.

    Reading ``energy_hartree`` while honouring the step's ``temperature_k`` gives an
    ``energy_type: gibbs`` step Boltzmann weights over electronic energies — a table that
    silently contradicts the survivor set it claims to describe.
    """
    import pandas as pd

    from chemrefine.engines.api import ENGINES

    _register_thermo_engine()
    try:
        cfg = _config(
            tmp_path,
            input=None,
            steps=[
                StepConfig(
                    step=1,
                    engine="fake-thermo",
                    operation="opt_sp",
                    sample={"method": "min", "count": 2, "energy_type": "gibbs"},
                )
            ],
        )
        seed_dir = tmp_path / "seeds"
        io.write_xyz([_h2() for _ in range(3)], ["0", "1", "2"], 0, seed_dir)
        cfg = cfg.model_copy(update={"input": seed_dir})

        pipeline.run(cfg)

        df = pd.read_csv(cfg.output_dir / "steps.csv")
        assert list(df["Energy type"]) == ["gibbs", "gibbs"]
        # Survivors are the two lowest *Gibbs* structures, and the reported
        # energies are their Gibbs values — not -1.0/-2.0/-3.0.
        assert sorted(df["Conformer"].astype(str)) == ["0", "1"]
        assert sorted(df["Energy (Hartree)"]) == pytest.approx([-10.0, -9.0])
        # dE is measured in the same currency, so the lowest row is the zero.
        assert min(df["dE (kcal/mol)"]) == pytest.approx(0.0)
    finally:
        ENGINES.pop("fake-thermo", None)


def test_steps_csv_defaults_to_electronic_without_a_sample(tmp_path: Path):
    import pandas as pd

    seed_dir = tmp_path / "seeds"
    io.write_xyz([_h2()], ["0"], 0, seed_dir)
    cfg = _config(tmp_path, input=seed_dir)
    pipeline.run(cfg)
    df = pd.read_csv(cfg.output_dir / "steps.csv")
    assert set(df["Energy type"]) == {"electronic"}


def _preflighted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan: RunPlan) -> list[list[str]]:
    """Run a two-step pipeline under ``plan`` and return what `preflight_backends` was handed."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        pipeline, "preflight_backends", lambda steps: calls.append([s.engine for s in steps])
    )
    cfg = Config(
        template_dir=tmp_path / "templates",
        output_dir=tmp_path / "outputs",
        steps=[
            StepConfig(step=1, engine="fake", operation="opt_sp"),
            StepConfig(step=2, engine="fake", operation="opt_sp"),
        ],
    )
    # It fails later for want of a seed; all that matters is what preflight was handed.
    with pytest.raises(ChemRefineError):
        pipeline.run(cfg, plan)
    return calls


def test_rebuild_cache_requires_no_backend_from_any_of_its_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`rebuild-cache` re-parses outputs on disk; no step of it may demand the backend.

    `preflight_backends` fails fast so a missing MLIP/PySCF env is reported before any job
    submits. `rebuild-cache N` puts N in REBUILD and every *other* step in CACHE_ONLY, and
    neither can submit — so the check has to be keyed on `may_submit`, not on the one named
    step. Exempting only the target leaves the wall standing on all the others, which is the
    whole command: a two-step MLIP config still cannot be rebuilt on a login node, or on any
    machine holding the output tree but not the stack that produced it.
    """
    calls = _preflighted(
        tmp_path,
        monkeypatch,
        RunPlan(default=StepMode.CACHE_ONLY, overrides={1: StepMode.REBUILD}),
    )
    assert calls == [[]], "no step of a rebuild-cache run may be asked for its backend"


def test_a_step_that_can_submit_is_still_preflighted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exemption above is scoped to modes that cannot submit, not a blanket removal.

    `rerun-errors N` runs N in RESUME — it resubmits — so N's backend must still be proven
    available before anything goes out, while the cache-only steps around it are exempt.
    """
    calls = _preflighted(
        tmp_path,
        monkeypatch,
        RunPlan(default=StepMode.CACHE_ONLY, overrides={2: StepMode.RESUME}),
    )
    assert calls == [["fake"]], "the step that will submit must still be checked"


# ---------------------------------------------------------------------------
# run lock — one driver per output tree
# ---------------------------------------------------------------------------


def _write_lock(output_dir: Path, *, host: str, pid: int) -> Path:
    """Plant a lock file as another driver would have left it."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / pipeline.RUN_LOCK_NAME
    lock.write_text(
        json.dumps({"pid": pid, "host": host, "started": "2026-08-10T00:00:00+00:00"}),
        encoding="utf-8",
    )
    return lock


def _dead_pid() -> int:
    """A pid guaranteed to name no live process: a child spawned and already reaped."""
    import subprocess

    proc = subprocess.Popen(["true"])  # a no-op child, only for its pid
    proc.wait()
    return proc.pid


def test_run_lock_is_released_after_a_clean_run(tmp_path: Path):
    seed = tmp_path / "step0_seed.xyz"
    io.write_xyz([_h2()], ["seed"], step_number=0, output_dir=tmp_path)
    cfg = _config(tmp_path, input=seed)
    pipeline.run(cfg)
    assert not (cfg.output_dir / pipeline.RUN_LOCK_NAME).exists()


def test_run_lock_is_released_when_the_run_raises(tmp_path: Path):
    """A failed run must not leave the tree locked — the retry is the very next command."""
    cfg = _config(tmp_path, input=None)  # no seed source: bootstrap raises inside the lock
    with pytest.raises(ConfigError):
        pipeline.run(cfg)
    assert not (cfg.output_dir / pipeline.RUN_LOCK_NAME).exists()


def test_run_lock_is_reentrant_within_one_process(tmp_path: Path):
    """The inner acquisition neither raises nor releases; the outer exit releases.

    This is the ``recovery.execute`` → ``pipeline.run`` shape: the action takes the lock
    around cache invalidation, and the run inside takes it again.
    """
    lock = tmp_path / "outputs" / pipeline.RUN_LOCK_NAME
    with pipeline.run_lock(tmp_path / "outputs"):
        with pipeline.run_lock(tmp_path / "outputs"):
            assert lock.exists()
        assert lock.exists(), "the inner exit must not release the outer's lock"
    assert not lock.exists()


def test_a_live_holders_lock_raises_and_survives(tmp_path: Path):
    """A lock naming a live pid on this host refuses the run and is left in place."""
    import socket
    import subprocess

    from chemrefine.errors import RunLockError

    child = subprocess.Popen(["sleep", "30"])  # a provably-live pid
    try:
        lock = _write_lock(tmp_path / "outputs", host=socket.gethostname(), pid=child.pid)
        with (
            pytest.raises(RunLockError, match=rf"pid {child.pid}"),
            pipeline.run_lock(tmp_path / "outputs"),
        ):
            pass
        assert lock.exists(), "a refused acquisition must not clobber the holder's lock"
    finally:
        child.kill()
        child.wait()


def test_a_dead_holders_lock_is_reclaimed(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """A same-host holder that no longer runs is stale: reclaimed, run proceeds."""
    import socket

    _write_lock(tmp_path / "outputs", host=socket.gethostname(), pid=_dead_pid())
    with caplog.at_level("WARNING"), pipeline.run_lock(tmp_path / "outputs"):
        assert (tmp_path / "outputs" / pipeline.RUN_LOCK_NAME).exists()
    assert "reclaiming stale run lock" in caplog.text
    assert not (tmp_path / "outputs" / pipeline.RUN_LOCK_NAME).exists()
    # The rename-claim is transient: a successful reclaim leaves no residue behind.
    assert not list((tmp_path / "outputs").glob(f"{pipeline.RUN_LOCK_NAME}.reclaim.*"))


def test_two_reclaimers_cannot_both_acquire(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The loser of a reclaim race answers to the winner's lock instead of acquiring.

    Both drivers probe the same dead holder; the winner's `os.replace` takes the stale
    lock's inode and its fresh lock lands before the loser moves. With the old
    `unlink()` reclaim this interleaving deleted the winner's fresh lock and both
    acquired — the one state the lock exists to prevent.
    """
    import os
    import socket

    from chemrefine.errors import RunLockError

    outputs = tmp_path / "outputs"
    _write_lock(outputs, host=socket.gethostname(), pid=_dead_pid())
    lock = outputs / pipeline.RUN_LOCK_NAME
    winner_pid = os.getppid()  # a live pid that is not this process's
    real_replace = os.replace

    def winner_got_there_first(src: object, dst: object) -> None:
        # The winner completes its whole reclaim-and-create inside the loser's window:
        # the stale lock is gone and a live one stands before the loser's rename runs.
        lock.unlink()
        _write_lock(outputs, host=socket.gethostname(), pid=winner_pid)
        raise FileNotFoundError(src)

    monkeypatch.setattr(os, "replace", winner_got_there_first)
    with pytest.raises(RunLockError, match=rf"pid {winner_pid}"), pipeline.run_lock(outputs):
        pass
    monkeypatch.setattr(os, "replace", real_replace)
    holder = pipeline._lock_holder(lock)
    assert holder is not None and holder[1] == winner_pid, "the winner's lock must survive"


def test_a_fresh_lock_swept_up_by_a_reclaim_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A claim that grabbed a lock re-created in the read window is put back, not kept.

    Between reading the stale record and renaming, another driver can finish its own
    reclaim *and* create a live lock at the same path; the rename cannot tell. The
    claim's record no longer matches the one that justified it, so the lock is restored
    (`os.link`, atomic fail-if-exists) and this driver answers to it — instead of
    silently holding a lock that names somebody else.
    """
    import os
    import socket

    from chemrefine.errors import RunLockError

    outputs = tmp_path / "outputs"
    _write_lock(outputs, host=socket.gethostname(), pid=_dead_pid())
    lock = outputs / pipeline.RUN_LOCK_NAME
    winner_pid = os.getppid()
    real_replace = os.replace

    def overtaken(src: object, dst: object) -> None:
        # The winner's reclaim-and-create lands first; the loser's rename then sweeps
        # up the *fresh* lock rather than the stale one it probed.
        _write_lock(outputs, host=socket.gethostname(), pid=winner_pid)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", overtaken)
    with pytest.raises(RunLockError, match=rf"pid {winner_pid}"), pipeline.run_lock(outputs):
        pass
    monkeypatch.setattr(os, "replace", real_replace)
    holder = pipeline._lock_holder(lock)
    assert holder is not None and holder[1] == winner_pid, "the swept-up lock must be restored"
    assert not list(outputs.glob(f"{pipeline.RUN_LOCK_NAME}.reclaim.*"))


def test_release_leaves_a_lock_that_is_no_longer_ours(tmp_path: Path):
    """Exit must not unlink a lock another driver now holds.

    The error message tells users to delete a lock whose run is known dead; followed
    against a run that was in fact alive, the unconditional release then removed the
    *new* holder's lock on exit, reopening the tree to a third driver.
    """
    import os
    import socket

    outputs = tmp_path / "outputs"
    lock = outputs / pipeline.RUN_LOCK_NAME
    with pipeline.run_lock(outputs):
        lock.unlink()  # an operator deletes it, believing this run dead …
        _write_lock(outputs, host=socket.gethostname(), pid=os.getppid())  # … a new run locks
    holder = pipeline._lock_holder(lock)
    assert holder is not None and holder[1] == os.getppid(), "the new holder's lock survives"


def test_a_foreign_hosts_lock_is_never_reclaimed(tmp_path: Path):
    """Liveness cannot be probed across hosts, so a foreign lock is always treated as live.

    Even a pid that is dead *here* proves nothing about the host in the record — the error
    tells the user to delete the lock once that run is known dead.
    """
    from chemrefine.errors import RunLockError

    _write_lock(tmp_path / "outputs", host="some-other-node", pid=_dead_pid())
    with (
        pytest.raises(RunLockError, match="some-other-node"),
        pipeline.run_lock(tmp_path / "outputs"),
    ):
        pass


def test_an_unreadable_lock_raises(tmp_path: Path):
    """A lock with no readable holder cannot be liveness-checked, so it refuses the run."""
    from chemrefine.errors import RunLockError

    (tmp_path / "outputs").mkdir(parents=True)
    (tmp_path / "outputs" / pipeline.RUN_LOCK_NAME).write_text("", encoding="utf-8")
    with pytest.raises(RunLockError, match="unreadable"), pipeline.run_lock(tmp_path / "outputs"):
        pass


def test_pid_alive_reads_permission_denied_as_alive(monkeypatch: pytest.MonkeyPatch):
    """EPERM means the process exists and is somebody else's — alive, not stale.

    Deterministic via monkeypatch: probing a real foreign pid (e.g. 1) answers differently
    for root, and a suite whose verdict depends on who runs it pins nothing.
    """
    import os

    def _deny(pid: int, sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(os, "kill", _deny)
    assert pipeline._pid_alive(12345) is True
