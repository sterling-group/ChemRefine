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


@pytest.mark.parametrize("literal", ["nan", "inf", "-inf"])
def test_bootstrap_refuses_a_non_finite_seed_coordinate(tmp_path: Path, literal: str):
    """A seed geometry that is not numbers must be refused where the file can be named.

    ASE's reader accepts these, and a seed is the one geometry no parse boundary ever
    sees — so left alone it reaches `cache.save`, whose coordinates go to the `arrays.npz`
    sidecar rather than through `write_json`'s `allow_nan=False`. Nothing downstream would
    object: it round-trips the cache and `structure_digest` hashes it to a stable key, so
    every later step would be computed from it silently. `on_failure: best` is the path
    that carries it there, backfilling the seed itself in place of a parse.
    """
    seed = tmp_path / "diverged.xyz"
    seed.write_text(f"2\nseed\nH 0.0 0.0 0.0\nH {literal} 0.0 1.5\n", encoding="utf-8")
    cfg = _config(tmp_path, input=seed)
    with pytest.raises(ConfigError, match="non-finite coordinate"):
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


def test_the_run_walks_every_submittable_steps_own_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``preflight_steps`` rides beside ``preflight_backends``: same list, same t=0.

    The engine-owned refusals (``PreflightChecking``) fire for exactly the steps that
    may submit — keyed on ``may_submit`` for the same rebuild-cache reason the backend
    walk is — and receive the config's charge/multiplicity defaults so each hook can
    resolve its step's effective species.
    """
    calls: list[tuple[list[str], int, int]] = []
    monkeypatch.setattr(
        pipeline,
        "preflight_steps",
        lambda steps, *, charge, multiplicity: calls.append(
            ([s.engine for s in steps], charge, multiplicity)
        ),
    )
    cfg = Config(
        template_dir=tmp_path / "templates",
        output_dir=tmp_path / "outputs",
        steps=[StepConfig(step=1, engine="fake", operation="opt_sp")],
    )
    with pytest.raises(ChemRefineError):
        pipeline.run(cfg)
    assert calls == [(["fake"], 0, 1)]


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
# scoped rebuilds — the best-effort tail
# ---------------------------------------------------------------------------


def _two_step_seeded(tmp_path: Path) -> Config:
    """A two-seed, two-step fake-engine config, ready for `pipeline.run`."""
    seed_dir = tmp_path / "seeds"
    io.write_xyz([_h2(), _h2()], ["a", "b"], 0, seed_dir)
    return _config(
        tmp_path,
        input=seed_dir,
        steps=[
            StepConfig(step=1, engine="fake", operation="opt_sp"),
            StepConfig(step=2, engine="fake", operation="opt_sp"),
        ],
    )


def _rebuild_step1_plan() -> RunPlan:
    """The plan `rebuild-cache 1` resolves to (`recovery._rebuild_plan`)."""
    return RunPlan(default=StepMode.CACHE_ONLY, overrides={1: StepMode.REBUILD}, stop_after=1)


def test_a_scoped_rebuild_re_reports_the_still_valid_tail(tmp_path: Path):
    """``steps.csv`` survives a rebuild whose re-parse changed nothing.

    The report is rewritten from step 1 on every run, so before the best-effort tail walk
    a ``stop_after`` plan deleted every later step's rows even though their caches were
    untouched and valid — after ``rebuild-cache 1`` the report claimed a one-step pipeline
    over a two-step tree.
    """
    cfg = _two_step_seeded(tmp_path)
    pipeline.run(cfg)
    before = (cfg.output_dir / "steps.csv").read_text(encoding="utf-8")

    outcomes = pipeline.run(cfg, _rebuild_step1_plan())

    assert len(outcomes) == 2, "the tail step was served, not skipped"
    assert outcomes[1].cache_hit is True, "served from its cache — nothing recomputed"
    after = (cfg.output_dir / "steps.csv").read_text(encoding="utf-8")
    assert after == before, "same results, same filter — the report is byte-identical"


def test_a_scoped_rebuild_stops_reporting_where_validity_ends(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A rebuild under an edited filter must not resurrect the old tail rows.

    Narrowing step 1's ``sample`` leaves step 1's own cache valid (the filter is excluded
    from its fingerprint) but changes the survivors that feed step 2, so step 2's cache no
    longer matches. The report ends at step 1 and the stop names ``resume`` —
    ``rebuild-cache 2`` would refuse, step 2's outputs having been produced for other
    parents. The stale step-2 artifacts stay on disk for ``resume`` to overwrite; a
    rebuild deletes nothing it did not write.
    """
    import logging

    import pandas as pd

    cfg = _two_step_seeded(tmp_path)
    pipeline.run(cfg)

    narrowed = cfg.model_copy(
        update={
            "steps": [
                StepConfig(
                    step=1,
                    engine="fake",
                    operation="opt_sp",
                    sample={"method": "min", "count": 1},
                ),
                cfg.steps[1],
            ]
        }
    )
    with caplog.at_level(logging.INFO):
        outcomes = pipeline.run(narrowed, _rebuild_step1_plan())

    assert len(outcomes) == 1, "step 2's cache no longer matches; the report ends before it"
    df = pd.read_csv(narrowed.output_dir / "steps.csv")
    assert set(df["Step"]) == {1} and len(df) == 1, "one surviving row under the new filter"
    assert "chemrefine resume" in caplog.text, "upstream changed — recomputation is the repair"
    assert (narrowed.output_dir / "step2" / "step2_ensemble.xyz").is_file(), (
        "stale tail artifacts are left for resume to overwrite, not deleted"
    )


def test_an_unscoped_plan_still_fails_on_an_unusable_earlier_cache(tmp_path: Path):
    """``rerun-errors``' shape — ``CACHE_ONLY`` before the target, no ``stop_after``.

    Best-effort is a property of the region past a rebuild's target, not of ``CACHE_ONLY``
    itself: an earlier step this plan cannot serve is a real failure the command must
    report, never a place for the report to quietly end.
    """
    from chemrefine import cache

    cfg = _two_step_seeded(tmp_path)
    pipeline.run(cfg)
    cache.invalidate((cfg.output_dir / "step1").resolve())

    with pytest.raises(ChemRefineError, match="no cache this configuration can use"):
        pipeline.run(cfg, RunPlan(default=StepMode.CACHE_ONLY, overrides={2: StepMode.RESUME}))


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


def test_a_swept_up_lock_that_cannot_be_restored_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """A restore that loses to a third driver's lock is reported, and its claim kept.

    Once a third lock lands in the rename→link window the swept-up driver is already
    running with no lock on the path — nothing can un-overlap the two, so the residue
    must at least be loud: an error names both drivers, and the claim file survives as
    the only remaining copy of the swept-up record.
    """
    import os
    import socket

    from chemrefine.errors import RunLockError

    outputs = tmp_path / "outputs"
    _write_lock(outputs, host=socket.gethostname(), pid=_dead_pid())
    lock = outputs / pipeline.RUN_LOCK_NAME
    winner_pid = os.getppid()
    third_pid = 1  # a live pid that is neither this process nor its parent
    real_replace = os.replace
    real_link = os.link

    def overtaken(src: object, dst: object) -> None:
        # The winner's reclaim-and-create lands first; the loser's rename then sweeps
        # up the *fresh* lock rather than the stale one it probed.
        _write_lock(outputs, host=socket.gethostname(), pid=winner_pid)
        real_replace(src, dst)

    def third_driver_got_there_first(src: object, dst: object) -> None:
        # A third driver's O_EXCL create lands inside the rename→link window, so the
        # real os.link fails with the real OS answer: FileExistsError.
        _write_lock(outputs, host=socket.gethostname(), pid=third_pid)
        real_link(src, dst)

    monkeypatch.setattr(os, "replace", overtaken)
    monkeypatch.setattr(os, "link", third_driver_got_there_first)
    with (
        caplog.at_level("ERROR"),
        pytest.raises(RunLockError, match=rf"pid {third_pid} on"),
        pipeline.run_lock(outputs),
    ):
        pass
    monkeypatch.setattr(os, "replace", real_replace)
    monkeypatch.setattr(os, "link", real_link)
    assert f"pid {winner_pid}" in caplog.text, "the swept-up run must be named"
    holder = pipeline._lock_holder(lock)
    assert holder is not None and holder[1] == third_pid, "the third driver's lock stands"
    claims = list(outputs.glob(f"{pipeline.RUN_LOCK_NAME}.reclaim.*"))
    assert claims, "the claim must survive as the swept-up record"
    swept = pipeline._lock_holder(claims[0])
    assert swept is not None and swept[1] == winner_pid


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


# ---------------------------------------------------------------------------
# SIGTERM unwinds — scancel must not strand the lock or the local jobs
# ---------------------------------------------------------------------------


def test_sigterm_inside_the_lock_unwinds_and_releases_it(tmp_path: Path):
    """A SIGTERM while the lock is held becomes SystemExit(143) and the lock is released.

    Python's default SIGTERM disposition terminates without unwinding — no ``finally``,
    no ``atexit`` — which is how a ``scancel``-ed driver left the tree locked. Raising
    from the handler is what lets every ``finally`` on the stack do its job.
    """
    import os
    import signal

    lock = tmp_path / "outputs" / pipeline.RUN_LOCK_NAME
    with pytest.raises(SystemExit) as excinfo, pipeline.run_lock(tmp_path / "outputs"):
        assert lock.exists()
        os.kill(os.getpid(), signal.SIGTERM)
    assert excinfo.value.code == 143
    assert not lock.exists()


def test_the_previous_sigterm_disposition_is_restored_on_exit(tmp_path: Path):
    """The handler is scoped to the lock — a host application's own handler survives."""
    import signal

    before = signal.getsignal(signal.SIGTERM)
    with pipeline.run_lock(tmp_path / "outputs"):
        assert signal.getsignal(signal.SIGTERM) is not before, "the unwind handler is active"
    assert signal.getsignal(signal.SIGTERM) is before


def test_the_sigterm_handler_is_not_installed_off_the_main_thread(tmp_path: Path):
    """`signal.signal` raises off the main thread; the lock must still work there.

    A host application driving the pipeline from a worker thread gives up the graceful
    SIGTERM release — that is the signal module's constraint, not a choice — but it must
    not gain a crash for it.
    """
    import signal
    import threading

    before = signal.getsignal(signal.SIGTERM)
    seen: dict[str, object] = {}

    def worker() -> None:
        with pipeline.run_lock(tmp_path / "outputs"):
            seen["during"] = signal.getsignal(signal.SIGTERM)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert seen["during"] is before, "no handler was installed off the main thread"
    assert not (tmp_path / "outputs" / pipeline.RUN_LOCK_NAME).exists()


def test_a_sigtermed_driver_process_releases_the_lock_on_disk(tmp_path: Path):
    """The whole point, end to end: SIGTERM a real driver and the tree is unlocked.

    This is the ``scancel`` shape — the driver runs as its own process, the signal is a
    real one, and the assertion is about what the next driver finds on disk: no lock, and
    exit code 143 saying what stopped the run.
    """
    import subprocess
    import sys
    import textwrap
    import time

    outputs = tmp_path / "outputs"
    script = tmp_path / "driver.py"
    script.write_text(
        textwrap.dedent(f"""
            import time
            from pathlib import Path

            from chemrefine import pipeline

            with pipeline.run_lock(Path({str(outputs)!r})):
                print("locked", flush=True)
                time.sleep(60)
        """),
        encoding="utf-8",
    )
    with subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True) as proc:
        try:
            assert proc.stdout is not None and proc.stdout.readline().strip() == "locked"
            assert (outputs / pipeline.RUN_LOCK_NAME).exists()

            proc.terminate()  # SIGTERM — what scancel and a walltime kill deliver

            assert proc.wait(timeout=30) == 143
            deadline = time.monotonic() + 5.0
            while (outputs / pipeline.RUN_LOCK_NAME).exists() and time.monotonic() < deadline:
                time.sleep(0.05)  # NFS-free here, but give the unlink a beat on slow CI
            assert not (outputs / pipeline.RUN_LOCK_NAME).exists()
        finally:
            if proc.poll() is None:
                proc.kill()
