"""Tests for the high-level pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms

from chemrefine import io, pipeline
from chemrefine.config import Config, StepConfig
from chemrefine.errors import ConfigError


def _h2() -> Atoms:
    return Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [0.74, 0, 0]])


def _config(tmp_path: Path, **overrides) -> Config:
    base = dict(
        template_dir=tmp_path / "templates",
        scratch_dir=tmp_path / "scratch",
        output_dir=tmp_path / "outputs",
        steps=[StepConfig(step=1, engine="fake", operation="opt_sp")],
    )
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


def test_bootstrap_from_directory(tmp_path: Path):
    seed_dir = tmp_path / "seeds"
    io.write_xyz([_h2(), _h2()], ["a", "b"], step_number=0, output_dir=seed_dir)
    cfg = _config(tmp_path, input=seed_dir)
    state = pipeline.bootstrap(cfg)
    assert len(state.structures) == 2
    assert [s.id for s in state.structures] == ["0", "1"]


def test_bootstrap_from_directory_seeds_every_frame_of_multiframe_files(tmp_path: Path):
    """A multi-frame file inside a seed directory yields one structure per frame.

    Regression: directory seeding used ASE's default read (last frame only),
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
    from chemrefine.engines.base import ENGINES, register
    from chemrefine.state import PipelineState, StepResults

    @register("empty-fake")
    class _EmptyEngine:
        name = "empty-fake"
        supports_nms = False

        def prepare(self, ctx):
            ctx.step_dir.mkdir(parents=True, exist_ok=True)
            from chemrefine.state import StepInputs

            return StepInputs(files=())

        def submit(self, inputs, ctx):
            from chemrefine.state import JobBatch

            return JobBatch(jobs={})

        def wait(self, batch):
            return None

        def parse(self, inputs, ctx):
            return StepResults(structures=())  # no survivors

        def normal_mode_sample(self, results, ctx):
            return results

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
