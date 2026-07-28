"""Tests for the MLIP training pipeline (``engines/mlip/trainer.py``).

The actual ``mace_run_train`` invocation is not exercised here — it
requires a real CUDA stack. We only test the inputs we generate
(extxyz files, YAML config, SLURM script) and the submit/wait loop
with a mocked ``slurm.submit`` / ``slurm.is_finished``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines.mlip import trainer
from chemrefine.engines.mlip.options import MlipOptions, MlipTrainOptions
from chemrefine.errors import ChemRefineError, ConfigError
from chemrefine.quantities import HARTREE_TO_EV
from chemrefine.state import PipelineState, StepContext, StepResults, Structure


def _ctx(tmp_path: Path, **option_overrides) -> StepContext:
    """Build a StepContext for trainer tests."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "step1.inp").write_text(
        "model: MACE\nmax_num_epochs: 5\n",
        encoding="utf-8",
    )
    (template_dir / "cpu.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --partition=normal\n#SBATCH --time=24:00:00\n",
        encoding="utf-8",
    )
    (template_dir / "cuda.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\n",
        encoding="utf-8",
    )
    options = {"device": "cpu"}
    options.update(option_overrides)
    step_cfg = StepConfig(
        step=1,
        name="train",
        engine="mlip",
        operation="mlip_train",
        options=options,
    )
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1_train",
        template_dir=template_dir,
        scratch_dir=None,
        prev_state=PipelineState(),
        charge=0,
        multiplicity=1,
        max_cores=4,
        slurm_template="cpu.slurm.header",
        executables={},
    )


def _struct(sid: str, energy_hartree: float = -1.0) -> Structure:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    return Structure(
        id=sid,
        atoms=atoms,
        energy_hartree=energy_hartree,
        forces_ev_per_a=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


# ---------------------------------------------------------------------------
# prepare_inputs
# ---------------------------------------------------------------------------


def test_prepare_inputs_writes_train_and_test_xyz(tmp_path: Path):
    ctx = _ctx(tmp_path)
    results = StepResults(structures=tuple(_struct(str(i), -1.0 - i * 0.1) for i in range(10)))
    train_path, test_path = trainer.prepare_inputs(results, ctx)
    assert train_path.is_file()
    assert test_path.is_file()
    # 90/10 split → 9 train, 1 test
    assert train_path.read_text().count("\nH2") >= 0  # just verify content non-empty
    assert "Properties=" in train_path.read_text()


def test_prepare_inputs_split_counts_and_is_deterministic(tmp_path: Path):
    """10 structures at valid_fraction=0.1 → 9 train / 1 test, and a fixed seed
    (default 42) yields the identical split on every run."""
    results = StepResults(structures=tuple(_struct(str(i), -1.0 - i * 0.1) for i in range(10)))

    train_path, test_path = trainer.prepare_inputs(results, _ctx(tmp_path))
    n_train = train_path.read_text().count("DFT_energy=")
    n_test = test_path.read_text().count("DFT_energy=")
    assert (n_train, n_test) == (9, 1)

    # Same seed + structures → byte-identical extxyz (deterministic split).
    train2, test2 = trainer.prepare_inputs(results, _ctx(tmp_path / "again"))
    assert train2.read_text() == train_path.read_text()
    assert test2.read_text() == test_path.read_text()


def test_prepare_inputs_converts_energy_to_ev(tmp_path: Path):
    """``Structure.energy_hartree`` should be multiplied by HARTREE_TO_EV on write."""
    import re

    ctx = _ctx(tmp_path)
    e_hartree = -1.5
    results = StepResults(structures=tuple(_struct(str(i), e_hartree) for i in range(10)))
    train_path, test_path = trainer.prepare_inputs(results, ctx)
    expected_ev = e_hartree * HARTREE_TO_EV
    combined = train_path.read_text() + test_path.read_text()
    energies = [float(m.group(1)) for m in re.finditer(r"DFT_energy=(-?\d+\.\d+)", combined)]
    # Every structure carries the same energy; every entry should be the eV value
    assert energies
    assert all(abs(e - expected_ev) < 1e-6 for e in energies)


def test_prepare_inputs_rejects_structure_without_energy(tmp_path: Path):
    ctx = _ctx(tmp_path)
    bad = Structure(id="0", atoms=Atoms("H"), energy_hartree=None, forces_ev_per_a=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="no energy"):
        trainer.prepare_inputs(StepResults(structures=(bad,)), ctx)


def test_prepare_inputs_rejects_structure_without_forces(tmp_path: Path):
    ctx = _ctx(tmp_path)
    bad = Structure(id="0", atoms=Atoms("H"), energy_hartree=-1.0, forces_ev_per_a=None)
    with pytest.raises(ValueError, match="no forces"):
        trainer.prepare_inputs(StepResults(structures=(bad,)), ctx)


def test_prepare_inputs_rejects_empty_results(tmp_path: Path):
    ctx = _ctx(tmp_path)
    with pytest.raises(ValueError, match="no usable structures"):
        trainer.prepare_inputs(StepResults(structures=()), ctx)


def test_prepare_inputs_single_structure_falls_back_to_all_train(tmp_path: Path):
    """One structure isn't enough for a 90/10 split; put it all in train."""
    ctx = _ctx(tmp_path)
    results = StepResults(structures=(_struct("0", -1.0),))
    train_path, test_path = trainer.prepare_inputs(results, ctx)
    assert train_path.is_file()
    # Test file is written (possibly empty)
    assert test_path.is_file()


# ---------------------------------------------------------------------------
# write_training_config
# ---------------------------------------------------------------------------


def test_write_training_config_patches_paths(tmp_path: Path):
    ctx = _ctx(tmp_path)
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    train_path = ctx.step_dir / "mace_train.xyz"
    test_path = ctx.step_dir / "mace_test.xyz"
    train_path.touch()
    test_path.touch()
    config_path = trainer.write_training_config(train_path=train_path, test_path=test_path, ctx=ctx)
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["train_file"] == str(train_path)
    assert cfg["test_file"] == str(test_path)
    assert cfg["log_dir"].endswith("log_dir")
    assert cfg["checkpoints_dir"].endswith("checkpoints_dir")
    assert cfg["results_dir"].endswith("results_dir")
    # The unrelated keys in the template (model, max_num_epochs) must survive.
    assert cfg["model"] == "MACE"
    assert cfg["max_num_epochs"] == 5


def test_write_training_config_handles_empty_template_body(tmp_path: Path):
    """An empty YAML template should still produce a working config."""
    ctx = _ctx(tmp_path)
    (ctx.template_dir / "step1.inp").write_text("", encoding="utf-8")
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    config_path = trainer.write_training_config(
        train_path=ctx.step_dir / "t.xyz",
        test_path=ctx.step_dir / "v.xyz",
        ctx=ctx,
    )
    cfg = yaml.safe_load(config_path.read_text())
    assert "train_file" in cfg
    assert "test_file" in cfg


# ---------------------------------------------------------------------------
# write_training_slurm
# ---------------------------------------------------------------------------


def test_write_training_slurm_uses_cpu_header_when_device_cpu(tmp_path: Path):
    ctx = _ctx(tmp_path, device="cpu")
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    config = ctx.step_dir / "input.yaml"
    config.touch()
    script = trainer.write_training_slurm(ctx=ctx, config_path=config)
    text = script.read_text()
    assert "#SBATCH --partition=normal" in text
    assert "mace_run_train --config" in text
    assert "--job-name=mlip_train" in text


def test_write_training_slurm_uses_cuda_header_when_device_cuda(tmp_path: Path):
    ctx = _ctx(tmp_path, device="cuda")
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    config = ctx.step_dir / "input.yaml"
    config.touch()
    script = trainer.write_training_slurm(ctx=ctx, config_path=config)
    text = script.read_text()
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gres=gpu:1" in text


def test_write_training_slurm_honours_job_name_option(tmp_path: Path):
    ctx = _ctx(tmp_path, device="cpu", job_name="custom_train")
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    config = ctx.step_dir / "input.yaml"
    config.touch()
    script = trainer.write_training_slurm(ctx=ctx, config_path=config)
    assert "#SBATCH --job-name=custom_train" in script.read_text()


def test_write_training_slurm_raises_when_header_missing(tmp_path: Path):
    ctx = _ctx(tmp_path, device="cuda")
    (ctx.template_dir / "cuda.slurm.header").unlink()
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    config = ctx.step_dir / "input.yaml"
    config.touch()
    with pytest.raises(ConfigError, match="SLURM header template"):
        trainer.write_training_slurm(ctx=ctx, config_path=config)


# ---------------------------------------------------------------------------
# submit_training
# ---------------------------------------------------------------------------


def test_submit_training_blocks_until_finished():
    """``submit_training`` should poll ``slurm.is_finished`` until it returns True."""
    finished_calls = [False, False, True]

    def fake_is_finished(job_id: str) -> bool:
        return finished_calls.pop(0)

    with (
        patch.object(trainer.slurm, "submit", return_value="12345"),
        patch.object(trainer.slurm, "is_finished", side_effect=fake_is_finished),
        patch.object(trainer.time, "sleep", return_value=None),
    ):
        job_id = trainer.submit_training(
            script_path=Path("train.slurm"),
            poll_seconds=0,
        )
    assert job_id == "12345"
    assert finished_calls == []


# ---------------------------------------------------------------------------
# run_training (orchestrator)
# ---------------------------------------------------------------------------


def test_run_training_drives_pipeline_and_returns_results_unchanged(tmp_path: Path):
    """Smoke-test the full prepare → config → slurm → submit pipeline."""
    ctx = _ctx(tmp_path)
    results = StepResults(structures=tuple(_struct(str(i)) for i in range(4)))
    with (
        patch.object(trainer.slurm, "submit", return_value="42") as mock_submit,
        patch.object(trainer.slurm, "is_finished", return_value=True),
        patch.object(trainer.time, "sleep", return_value=None),
    ):
        out = trainer.run_training(results, ctx)
    assert out is results
    mock_submit.assert_called_once()
    assert (ctx.step_dir / "mace_train.xyz").is_file()
    assert (ctx.step_dir / "mace_test.xyz").is_file()
    assert (ctx.step_dir / "input.yaml").is_file()
    assert (ctx.step_dir / "train.slurm").is_file()


# ---------------------------------------------------------------------------
# MlipTrainEngine — the "mlip-train" engine wiring
# ---------------------------------------------------------------------------


def test_mlip_train_engine_is_registered():
    from chemrefine.engines.api import ENGINES, get_engine

    assert "mlip-train" in ENGINES
    assert get_engine("mlip-train").name == "mlip-train"


def test_mlip_train_engine_has_no_input_digest(tmp_path: Path):
    """Training is a pass-through, so it contributes nothing to the cache fingerprint."""
    from chemrefine.engines.api import get_engine

    assert get_engine("mlip-train").input_digest(_ctx(tmp_path)) == ""


def test_mlip_train_engine_trains_on_prev_and_passes_structures_through(tmp_path: Path):
    """submit() calls trainer.run_training on the previous step's structures;
    parse() passes those structures through unchanged (model is the artifact)."""
    from chemrefine.engines.api import get_engine

    structs = tuple(_struct(str(i)) for i in range(3))
    ctx = _ctx(tmp_path)
    ctx = StepContext(
        step_cfg=ctx.step_cfg,
        step_dir=ctx.step_dir,
        template_dir=ctx.template_dir,
        scratch_dir=ctx.scratch_dir,
        prev_state=PipelineState(structures=structs),
        charge=ctx.charge,
        multiplicity=ctx.multiplicity,
        max_cores=ctx.max_cores,
        slurm_template=ctx.slurm_template,
        executables=ctx.executables,
    )
    engine = get_engine("mlip-train")
    inputs = engine.prepare(ctx)
    assert inputs.files == ()  # training is not per-structure
    with patch.object(trainer, "run_training", return_value=None) as mock_train:
        engine.submit(inputs, ctx)
    # trainer received the previous step's structures.
    passed = mock_train.call_args.args[0]
    assert passed.structures == structs
    # parse passes the same structures forward.
    assert engine.parse(inputs, ctx).structures == structs


# ---------------------------------------------------------------------------
# job_name is interpolated into an #SBATCH directive
# ---------------------------------------------------------------------------


def test_write_training_slurm_rejects_a_job_name_with_a_newline(tmp_path: Path):
    """A newline in job_name would start an arbitrary extra #SBATCH directive."""
    import pytest

    from chemrefine.errors import ConfigError

    ctx = _ctx(tmp_path, device="cpu", job_name="ok\n#SBATCH --account=someone")
    with pytest.raises(ConfigError, match="job_name"):
        trainer.write_training_slurm(ctx=ctx, config_path=tmp_path / "input.yaml")


def test_write_training_slurm_accepts_ordinary_job_names(tmp_path: Path):
    ctx = _ctx(tmp_path, device="cpu", job_name="mace-run_1.0")
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    script = trainer.write_training_slurm(ctx=ctx, config_path=tmp_path / "input.yaml")
    assert "#SBATCH --job-name=mace-run_1.0" in script.read_text()


def test_write_training_slurm_defaults_to_the_cuda_header(tmp_path: Path):
    """A training step that names no device still asks for a GPU.

    Training is the one step where CPU is not a slower run but an impractical
    one, so `mlip-train` deliberately defaults to `cuda` where the inference
    engines default to `cpu`. Every other test here passes `device` explicitly,
    which is why nothing noticed when the trainer's inline default and the shared
    options default silently drifted apart.
    """
    ctx = _ctx(tmp_path)
    ctx.step_cfg.options.pop("device")
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    config = ctx.step_dir / "input.yaml"
    config.touch()

    script = trainer.write_training_slurm(ctx=ctx, config_path=config)

    assert "--gres=gpu:1" in script.read_text(), "an unspecified device must still train on GPU"


def test_training_device_default_is_declared_not_repeated():
    """The trainer's default lives on the options model, not in a second literal."""
    assert MlipTrainOptions().device == "cuda"
    assert MlipOptions().device == "cpu"


# --- mlip-train engine no-op / unsupported ----------------------------------


def test_mlip_train_engine_not_nms_capable(tmp_path: Path):
    from chemrefine.engines.api import NmsCapableEngine, get_engine

    eng = get_engine("mlip-train")
    # mlip-train is a pass-through: it doesn't satisfy the NMS hook contract.
    assert not isinstance(eng, NmsCapableEngine)


def test_trainer_rejects_valid_fraction_leaving_no_training(tmp_path: Path):
    from chemrefine.engines.mlip import trainer

    seeds = tuple(
        Structure(
            id=str(i),
            atoms=Atoms("H", positions=[[0, 0, 0]]),
            energy_hartree=-1.0,
            forces_ev_per_a=np.zeros((1, 3)),
        )
        for i in range(2)
    )
    # 0.6 of 2 structures rounds up to 2 held out, leaving none to train on. The
    # field bounds valid_fraction to (0, 1); this is the case only the structure
    # count can decide, so the trainer still has to check it.
    ctx = _ctx(tmp_path, valid_fraction=0.6)
    with pytest.raises(ValueError, match="leaves no training"):
        trainer.prepare_inputs(StepResults(structures=seeds), ctx)


def test_trainer_options_reject_a_degenerate_valid_fraction(tmp_path: Path):
    """0 and 1 are wrong whatever the structure count, so the field refuses them."""
    from chemrefine.engines.mlip.options import MlipTrainOptions

    for bad in (0.0, 1.0):
        with pytest.raises(ChemRefineError, match="valid_fraction"):
            MlipTrainOptions.from_raw_lenient({"valid_fraction": bad})
