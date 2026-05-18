"""Tests for the FakeEngine round-trip through the full lifecycle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines.base import get_engine
from chemrefine.state import PipelineState, StepContext, Structure


def _ctx_with_seeds(tmp_path: Path, ids: list[str]) -> StepContext:
    """Build a StepContext seeded with one H atom per ID."""
    seeds = tuple(Structure(id=i, atoms=Atoms("H")) for i in ids)
    step_cfg = StepConfig(step=1, engine="fake", operation="opt_sp")
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "step1",
        template_dir=tmp_path / "templates",
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(structures=seeds),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        orca_executable="orca",
    )


def test_fake_engine_prepare_writes_one_inp_per_seed(tmp_path: Path):
    engine = get_engine("fake")
    ctx = _ctx_with_seeds(tmp_path, ["0", "1", "2"])
    inputs = engine.prepare(ctx)
    assert len(inputs.files) == 3
    for inp_path, _out_path, _sid in inputs.files:
        assert inp_path.exists()
        assert inp_path.suffix == ".inp"


def test_fake_engine_submit_writes_output_files(tmp_path: Path):
    engine = get_engine("fake")
    ctx = _ctx_with_seeds(tmp_path, ["0", "1"])
    inputs = engine.prepare(ctx)
    engine.submit(inputs, ctx)
    for _inp, out, _sid in inputs.files:
        assert out.exists()
        assert "FINAL ENERGY" in out.read_text()


def test_fake_engine_parse_returns_structures_with_energy(tmp_path: Path):
    engine = get_engine("fake")
    ctx = _ctx_with_seeds(tmp_path, ["0", "1", "2"])
    inputs = engine.prepare(ctx)
    engine.submit(inputs, ctx)
    engine.wait(engine.submit(inputs, ctx))  # idempotent
    results = engine.parse(inputs, ctx)
    assert len(results.structures) == 3
    for struct in results.structures:
        assert struct.energy_hartree is not None
        assert struct.energy_hartree < 0  # fake energies are negative
        assert struct.forces_eV_per_A is not None
        np.testing.assert_array_equal(struct.forces_eV_per_A, 0)


def test_fake_engine_energy_is_deterministic(tmp_path: Path):
    """Same ID must yield the same energy across separate runs."""
    engine = get_engine("fake")
    ctx_a = _ctx_with_seeds(tmp_path / "a", ["42"])
    inputs_a = engine.prepare(ctx_a)
    engine.submit(inputs_a, ctx_a)
    e_a = engine.parse(inputs_a, ctx_a).structures[0].energy_hartree

    ctx_b = _ctx_with_seeds(tmp_path / "b", ["42"])
    inputs_b = engine.prepare(ctx_b)
    engine.submit(inputs_b, ctx_b)
    e_b = engine.parse(inputs_b, ctx_b).structures[0].energy_hartree

    assert e_a == e_b


def test_fake_engine_does_not_support_nms(tmp_path: Path):
    engine = get_engine("fake")
    assert engine.supports_nms is False


def test_fake_engine_round_trip_preserves_ids(tmp_path: Path):
    engine = get_engine("fake")
    ctx = _ctx_with_seeds(tmp_path, ["0", "1-0", "1-1"])
    inputs = engine.prepare(ctx)
    engine.submit(inputs, ctx)
    results = engine.parse(inputs, ctx)
    assert [s.id for s in results.structures] == ["0", "1-0", "1-1"]
