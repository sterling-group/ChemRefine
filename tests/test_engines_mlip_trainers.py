"""Per-library trainer behaviour — the dataset, the argv, and where the model lands.

These are the tests the trainer this replaced never had. It wrote a dataset MACE could not
read and a command that could not resolve, under 100% line coverage, because every test
asserted on what the code produced rather than on what the library consumes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read as ase_read

from chemrefine.engines.mlip.backends.fairchem import _METADATA_NAME, FairchemTrainer
from chemrefine.engines.mlip.backends.mace import (
    CHARGE_KEY,
    ENERGY_KEY,
    FORCES_KEY,
    SPIN_KEY,
    MaceTrainer,
)
from chemrefine.engines.mlip.training import DatasetSplit, TrainingPlan, split_structures
from chemrefine.errors import ConfigError
from chemrefine.quantities import HARTREE_TO_EV
from chemrefine.state import Structure

_HAS_MACE = importlib.util.find_spec("mace") is not None


def _labelled(sid: str, *, energy: float = -1.5) -> Structure:
    return Structure(
        id=sid,
        atoms=Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
        energy_hartree=energy,
        forces_ev_per_a=np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]]),
    )


def _plan(tmp_path: Path, **overrides: object) -> TrainingPlan:
    base: dict[str, object] = {
        "run_dir": tmp_path / "train",
        "run_name": "train",
        "device": "cpu",
        "gpus": 0,
        "cores": 4,
        "seed": 42,
        "charge": 0,
        "multiplicity": 1,
        "start_from": None,
        "launcher": Path("/envs/mlip-mace/bin/python"),
    }
    base.update(overrides)
    return TrainingPlan(**base)


# ---------------------------------------------------------------------------
# The dataset MACE actually reads
# ---------------------------------------------------------------------------


def test_the_dataset_uses_maces_own_label_keys(tmp_path: Path):
    """The defect that made every training job die at data load.

    The old trainer wrote `DFT_energy` / `DFT_Forces`; MACE's defaults are `REF_energy` /
    `REF_forces`; and the shipped template declared a third pair. MACE refuses a file in which
    it finds none of its keys, so the step failed — and, because nothing checked the job's
    exit status, cached itself as a success.
    """
    plan = _plan(tmp_path)
    split = DatasetSplit(train=(_labelled("0"),), valid=(), test=())

    data = MaceTrainer().write_dataset(plan, split)

    text = data.train.read_text()
    assert f"{ENERGY_KEY}=" in text
    assert FORCES_KEY in text


@pytest.mark.skipif(not _HAS_MACE, reason="needs a mace stack to read its own defaults")
def test_the_label_keys_are_the_ones_mace_declares():
    """Pinned against the library rather than against a literal we chose.

    A copy of MACE's defaults is only correct until MACE moves them, and the failure mode is
    silent — a dataset that loads with zero-weighted labels, or refuses to load at all.
    """
    from mace.tools.default_keys import DefaultKeys

    assert DefaultKeys.ENERGY.value == ENERGY_KEY
    assert DefaultKeys.FORCES.value == FORCES_KEY
    assert DefaultKeys.TOTAL_CHARGE.value == CHARGE_KEY
    assert DefaultKeys.TOTAL_SPIN.value == SPIN_KEY


def test_charge_and_multiplicity_reach_the_dataset(tmp_path: Path):
    """The silent scientific defect: an anion fine-tuned as a neutral singlet.

    MACE defaults `total_charge` to 0.0 and `total_spin` to 1.0 when the keys are absent, so
    the previous trainer — which wrote neither — labelled every charged or open-shell system
    as neutral closed-shell, with nothing said in any log.
    """
    plan = _plan(tmp_path, charge=-1, multiplicity=3)
    split = DatasetSplit(train=(_labelled("0"),), valid=(), test=())

    data = MaceTrainer().write_dataset(plan, split)
    frame = ase_read(str(data.train), index=0)

    assert frame.info[CHARGE_KEY] == -1.0
    assert frame.info[SPIN_KEY] == 3.0, "total_spin is the multiplicity, not the unpaired count"


def test_energies_are_converted_to_electronvolts(tmp_path: Path):
    """Structures carry Hartree; MACE is fitted in eV."""
    plan = _plan(tmp_path)
    split = DatasetSplit(train=(_labelled("0", energy=-1.5),), valid=(), test=())

    data = MaceTrainer().write_dataset(plan, split)
    frame = ase_read(str(data.train), index=0)

    assert frame.info[ENERGY_KEY] == pytest.approx(-1.5 * HARTREE_TO_EV)


def test_forces_survive_the_round_trip_unchanged(tmp_path: Path):
    """Forces are already eV/Å, so they must be written as they are."""
    plan = _plan(tmp_path)
    struct = _labelled("0")
    data = MaceTrainer().write_dataset(plan, DatasetSplit(train=(struct,), valid=(), test=()))

    frame = ase_read(str(data.train), index=0)
    assert np.allclose(frame.arrays[FORCES_KEY], struct.forces_ev_per_a)


def test_an_empty_split_gets_no_file_at_all(tmp_path: Path):
    """ase raises `Empty file` on a zero-byte extxyz, so naming one is worse than omitting it."""
    plan = _plan(tmp_path)
    split = split_structures(
        [_labelled(str(i)) for i in range(10)], valid_fraction=0.1, test_fraction=0.0, seed=42
    )

    data = MaceTrainer().write_dataset(plan, split)

    assert data.train.is_file() and data.valid is not None and data.valid.is_file()
    assert data.test is None
    assert not (plan.run_dir / "test.xyz").exists()


def test_the_pipelines_own_structures_are_left_alone(tmp_path: Path):
    """Force arrays are shared by reference across steps and deliberately read-only."""
    struct = _labelled("0")
    MaceTrainer().write_dataset(_plan(tmp_path), DatasetSplit(train=(struct,), valid=(), test=()))
    assert not struct.forces_ev_per_a.flags.writeable
    assert not struct.atoms.info, "the seed's own info dict must not gain training keys"


@pytest.mark.skipif(not _HAS_MACE, reason="needs a mace stack")
def test_mace_loads_the_dataset_we_write(tmp_path: Path):
    """End to end against the real loader: the labels arrive, and they carry full weight.

    A dataset MACE can open but whose labels it weights at zero trains a model on nothing —
    which is what a key mismatch did on MACE < 0.3.13, before it began refusing outright.
    """
    from mace.data.utils import KeySpecification, load_from_xyz

    plan = _plan(tmp_path, charge=-1, multiplicity=1)
    split = DatasetSplit(train=tuple(_labelled(str(i)) for i in range(4)), valid=(), test=())
    data = MaceTrainer().write_dataset(plan, split)

    _, configs = load_from_xyz(
        file_path=str(data.train), key_specification=KeySpecification.from_defaults()
    )

    assert configs[0].property_weights["energy"] == 1.0
    assert configs[0].property_weights["forces"] == 1.0
    assert configs[0].properties["total_charge"] == -1.0


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def test_the_command_runs_under_the_backends_own_interpreter(tmp_path: Path):
    """A bare `mace_run_train` is on nobody's PATH once MACE has an environment of its own."""
    body = MaceTrainer().command(_plan(tmp_path), tmp_path / "train" / "step1_train.yaml")

    assert body.startswith("/envs/mlip-mace/bin/python -m mace.cli.run_train")
    assert body.endswith("--config step1_train.yaml"), "basename: the job runs inside $WORK_DIR"


def test_a_multi_gpu_step_runs_under_torchrun(tmp_path: Path):
    body = MaceTrainer().command(_plan(tmp_path, gpus=4), tmp_path / "cfg.yaml")

    assert "torch.distributed.run" in body
    assert "--nproc_per_node=4" in body
    assert body.endswith("--distributed")


def test_an_interpreter_path_with_a_space_stays_one_word(tmp_path: Path):
    plan = _plan(tmp_path, launcher=Path("/opt/my envs/mlip-mace/bin/python"))
    assert "'/opt/my envs/mlip-mace/bin/python'" in MaceTrainer().command(plan, Path("c.yaml"))


# ---------------------------------------------------------------------------
# The product
# ---------------------------------------------------------------------------


def test_the_model_is_named_predictably_enough_for_the_next_step(tmp_path: Path):
    """A later step writes this path into its own YAML *before* the training has run.

    MACE also writes a seed-tagged copy under `checkpoints/` — `{name}_run-{seed}…` — which
    the tutorial used to hardcode. `model_dir` (which defaults to `work_dir`) carries the
    seed-free name, and that is the one a config can name in advance.
    """
    run_dir = tmp_path / "train"
    run_dir.mkdir(parents=True)
    assert MaceTrainer().artifact(run_dir, "train") == run_dir / "train.model"


def test_a_two_stage_run_is_found_by_its_own_name(tmp_path: Path):
    """The recommended fine-tuning recipe enables stage two, which renames the product."""
    run_dir = tmp_path / "train"
    run_dir.mkdir(parents=True)
    (run_dir / "train_stagetwo.model").write_bytes(b"weights")

    assert MaceTrainer().artifact(run_dir, "train") == run_dir / "train_stagetwo.model"


# ---------------------------------------------------------------------------
# FAIRChem — a different dataset format, a different launcher, a different product
# ---------------------------------------------------------------------------

_HAS_FAIRCHEM = importlib.util.find_spec("fairchem") is not None


def _fc_plan(tmp_path: Path, **overrides: object) -> TrainingPlan:
    base: dict[str, object] = {
        "run_dir": tmp_path / "train",
        "run_name": "train",
        "device": "cpu",
        "gpus": 0,
        "cores": 4,
        "seed": 42,
        "charge": 0,
        "multiplicity": 1,
        "start_from": None,
        "launcher": Path("/envs/mlip-fairchem/bin/python"),
    }
    base.update(overrides)
    return TrainingPlan(**base)


def _fc_split(n_train: int = 4, n_valid: int = 1) -> DatasetSplit:
    return DatasetSplit(
        train=tuple(_labelled(str(i)) for i in range(n_train)),
        valid=tuple(_labelled(f"v{i}") for i in range(n_valid)),
        test=(),
    )


def test_fairchem_labels_ride_on_a_calculator_not_on_named_keys(tmp_path: Path):
    """The whole reason `write_dataset` is per-library rather than shared.

    MACE reads named info/array keys out of an extxyz; `AseDBDataset` reads whatever the
    stored `SinglePointCalculator` holds. There is no file format that serves both, which is
    what the `Trainer` protocol exists to let each library answer for itself.
    """
    from ase.db import connect

    data = FairchemTrainer().write_dataset(_fc_plan(tmp_path), _fc_split())

    with connect(str(data.train)) as db:
        row = next(iter(db.select()))
    assert row.energy == pytest.approx(-1.5 * HARTREE_TO_EV)
    assert row.natoms == 3


def test_charge_and_spin_ride_in_each_rows_data_mapping(tmp_path: Path):
    """`row.data` is the one channel `AseDBDataset` copies back into `atoms.info`.

    A plain `db.write(atoms)` drops `atoms.info` entirely, and FAIRChem's `common_transform`
    then silently defaults charge to 0 and spin to a singlet — an ion fine-tuned as the wrong
    species with nothing said in any log. The MACE writer has the same obligation with its
    `total_charge`/`total_spin` keys; this is FAIRChem's spelling of it.
    """
    from ase.db import connect

    plan = _fc_plan(tmp_path, charge=-1, multiplicity=2)
    data = FairchemTrainer().write_dataset(plan, _fc_split())

    with connect(str(data.train)) as db:
        for row in db.select():
            assert row.data["charge"] == -1
            assert row.data["spin"] == 2


def test_ranks_per_node_is_the_gpu_width_with_a_floor_of_one(tmp_path: Path):
    """`$NGPUS` cannot serve FAIRChem's `scheduler.ranks_per_node`: zero ranks is no run.

    A CPU step demands zero GPUs, and a scheduler with `ranks_per_node: 0` launches no
    worker at all — so the trainer supplies the distributed width itself, floored at one.
    """
    trainer = FairchemTrainer()
    cpu_plan = _fc_plan(tmp_path)
    cpu = trainer.placeholders(cpu_plan, trainer.write_dataset(cpu_plan, _fc_split()))
    assert cpu["RANKS_PER_NODE"] == "1"

    gpu_plan = _fc_plan(tmp_path, device="cuda", gpus=4)
    gpu = trainer.placeholders(gpu_plan, trainer.write_dataset(gpu_plan, _fc_split()))
    assert gpu["RANKS_PER_NODE"] == "4"


def test_each_fairchem_split_gets_its_own_directory_and_metadata(tmp_path: Path):
    """Without this the two splits would load each other's metadata and fail an assertion.

    FAIRChem resolves a missing `metadata_path` to `metadata.npz` in the database file's
    *parent*, so `train.db` and `valid.db` side by side would both read the first one written
    and trip `Loaded metadata size N and dataset size M mismatch`.
    """
    data = FairchemTrainer().write_dataset(_fc_plan(tmp_path), _fc_split(n_train=4, n_valid=2))

    assert data.valid is not None
    assert data.train.parent != data.valid.parent
    for path, expected in ((data.train, 4), (data.valid, 2)):
        natoms = np.load(path.parent / _METADATA_NAME)["natoms"]
        assert len(natoms) == expected
        assert np.issubdtype(natoms.dtype, np.integer), "the loader asserts an integer dtype"


def test_a_rerun_does_not_stack_duplicate_rows(tmp_path: Path):
    """`ase.db.connect` appends. A resumed or re-run step must not double its own dataset."""
    trainer, plan, split = FairchemTrainer(), _fc_plan(tmp_path), _fc_split(n_train=4)

    trainer.write_dataset(plan, split)
    data = trainer.write_dataset(plan, split)

    assert len(np.load(data.train.parent / _METADATA_NAME)["natoms"]) == 4
    from ase.db import connect

    with connect(str(data.train)) as db:
        assert db.count() == 4


def test_fairchem_training_without_a_validation_set_is_refused(tmp_path: Path):
    """Its runner takes a train *and* an eval dataloader; there is no train-only mode."""
    split = DatasetSplit(train=(_labelled("0"),), valid=(), test=())
    with pytest.raises(ConfigError, match="needs a validation set"):
        FairchemTrainer().write_dataset(_fc_plan(tmp_path), split)


def test_the_device_placeholder_is_uppercased_for_fairchem(tmp_path: Path):
    """Its enums reject their own lowercase values — `'cpu'` is not a valid `DeviceType`.

    A trainer overriding a shared placeholder is legitimate; this is the case it exists for.
    """
    plan = _fc_plan(tmp_path, device="cuda")
    data = FairchemTrainer().write_dataset(plan, _fc_split())

    assert FairchemTrainer().placeholders(plan, data)["DEVICE"] == "CUDA"


def test_fairchem_placeholders_name_every_dataset_and_its_metadata(tmp_path: Path):
    plan = _fc_plan(tmp_path)
    data = FairchemTrainer().write_dataset(plan, _fc_split())

    values = FairchemTrainer().placeholders(plan, data)
    assert values["TRAIN_SET"].endswith("train/train.db")
    assert values["TRAIN_METADATA"].endswith(f"train/{_METADATA_NAME}")
    assert values["VAL_SET"].endswith("valid/valid.db")
    assert values["TEST_SET"] == "", "an absent split renders empty, not as a missing file"


def test_fairchem_runs_its_console_script_not_a_module(tmp_path: Path):
    """`fairchem.core._cli` has no `__main__` guard: `-m` exits 0 having done nothing."""
    body = FairchemTrainer().command(_fc_plan(tmp_path), tmp_path / "train" / "step1_train.yaml")

    assert body == "/envs/mlip-fairchem/bin/fairchem -c step1_train.yaml"


def test_the_fairchem_product_is_the_inference_checkpoint(tmp_path: Path):
    """Under `<run_dir>/<timestamp_id>/checkpoints/final`, which the template pins."""
    assert FairchemTrainer().artifact(tmp_path / "train", "train") == (
        tmp_path / "train" / "train" / "checkpoints" / "final" / "inference_ckpt.pt"
    )


@pytest.mark.skipif(not _HAS_FAIRCHEM, reason="needs a fairchem stack")
def test_fairchem_loads_the_dataset_we_write(tmp_path: Path):
    """End to end against the real loader, as the MACE half is."""
    from fairchem.core.datasets.ase_datasets import AseDBDataset

    plan = _fc_plan(tmp_path)
    data = FairchemTrainer().write_dataset(plan, _fc_split(n_train=4))

    ds = AseDBDataset(
        config={
            "src": str(data.train),
            "metadata_path": str(data.train.parent / _METADATA_NAME),
            "a2g_args": {"r_energy": True, "r_forces": True},
        }
    )
    assert len(ds) == 4
    assert float(ds[0].energy) == pytest.approx(-1.5 * HARTREE_TO_EV, rel=1e-5)
    assert tuple(ds[0].forces.shape) == (3, 3)
    assert list(ds.get_metadata("natoms", [0, 1])) == [3, 3]
