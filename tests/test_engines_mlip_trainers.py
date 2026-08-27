"""Per-library trainer behaviour — the dataset, the argv, and where the model lands.

These are the tests the trainer this replaced never had. It wrote a dataset MACE could not
read and a command that could not resolve, under 100% line coverage, because every test
asserted on what the code produced rather than on what the library consumes.

The cross-validation tests (the ones needing a real MLIP stack) are **integration-tier**
and run their library half in a subprocess under the interpreter the provisioner
resolves — see :func:`_backend_python`. Both halves of that follow from the same fact:
the orchestrator's env deliberately does not hold MACE (the e3nn conflict is why managed
envs exist), so an in-process ``find_spec`` says "absent" on the very machine where
``chemrefine backends install`` has provisioned it — and the managed envs are visible
only to the tier ``conftest._isolate_chemrefine_home`` exempts. Running the half out of
process is also what production does, and it keeps a backend's own import-time warnings
— torch's env-var notices, FAIRChem's pydantic deprecations — out of this suite's
``filterwarnings = error`` regime, which exists for warnings *our* calls trigger.
"""

from __future__ import annotations

import functools
import json
import subprocess
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.io import read as ase_read

from chemrefine.engines import _provision
from chemrefine.engines.mlip.backends.fairchem import _METADATA_NAME, FairchemTrainer
from chemrefine.engines.mlip.backends.mace import (
    CHARGE_KEY,
    ENERGY_KEY,
    FORCES_KEY,
    SPIN_KEY,
    MaceTrainer,
    _to_atoms,
)
from chemrefine.engines.mlip.registry import requirement_from_options
from chemrefine.engines.mlip.train.base import DatasetSplit, TrainingPlan
from chemrefine.errors import ConfigError
from chemrefine.quantities import HARTREE_TO_EV
from chemrefine.state import Structure


@functools.cache
def _backend_python(task_name: str) -> str | None:
    """The interpreter that can import this backend, or ``None`` when nothing here can.

    Resolved through the provisioner itself — the same seam ``launcher_for`` uses in
    production — so the answer is "this interpreter, or the managed env `chemrefine
    backends install` built", never only the first. The final probe (a cheap
    ``find_spec`` in the resolved interpreter, importing nothing) is what keeps the
    suite independent of *fake* provisioned envs: the CI job that plants a bare
    symlink per extra must see a skip, not an ImportError dressed as a failure.
    """
    requirement = requirement_from_options({"task_name": task_name})
    try:
        _provision.require_backend(requirement)
    except ConfigError:
        return None
    python = _provision.resolve_launcher(requirement)
    probe = subprocess.run(
        [
            python,
            "-c",
            f"import importlib.util as u, sys;"
            f"sys.exit(0 if u.find_spec({requirement.import_name!r}) else 1)",
        ],
        capture_output=True,
        check=False,
    )
    return python if probe.returncode == 0 else None


def _require_backend_python(task_name: str) -> str:
    """Skip the calling test when no interpreter — local or managed — has the backend."""
    python = _backend_python(task_name)
    if python is None:
        pytest.skip(f"no {task_name} stack — neither importable here nor in a managed env")
    return python


def _in_backend(python: str, script: str, *args: str) -> dict:
    """Run ``script`` under the backend's interpreter; return the JSON it prints.

    The JSON is the script's *last* stdout line, not its whole stdout: the libraries chat
    on import — MACE prints a cuequivariance notice straight to stdout — and a scripted
    print cannot get in front of one.
    """
    result = subprocess.run(
        [python, "-c", script, *args], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    loaded: dict = json.loads(result.stdout.strip().splitlines()[-1])
    return loaded


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


def _small_split() -> DatasetSplit:
    """Two train / one valid / no test — the shape the per-library writer tests share."""
    return DatasetSplit(
        train=(_labelled("0"), _labelled("1", energy=-1.6)),
        valid=(_labelled("2"),),
        test=(),
    )


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


@pytest.mark.integration
def test_the_label_keys_are_the_ones_mace_declares():
    """Pinned against the library rather than against a literal we chose.

    A copy of MACE's defaults is only correct until MACE moves them, and the failure mode is
    silent — a dataset that loads with zero-weighted labels, or refuses to load at all.
    """
    python = _require_backend_python("mace_off")
    declared = _in_backend(
        python,
        "import json\n"
        "from mace.tools.default_keys import DefaultKeys\n"
        "print(json.dumps({\n"
        '    "energy": DefaultKeys.ENERGY.value,\n'
        '    "forces": DefaultKeys.FORCES.value,\n'
        '    "charge": DefaultKeys.TOTAL_CHARGE.value,\n'
        '    "spin": DefaultKeys.TOTAL_SPIN.value,\n'
        "}))\n",
    )
    assert declared == {
        "energy": ENERGY_KEY,
        "forces": FORCES_KEY,
        "charge": CHARGE_KEY,
        "spin": SPIN_KEY,
    }


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


def test_labelling_copies_rather_than_mutating_the_pipeline_structure(tmp_path: Path):
    """The dataset labels must land on a copy — the structure goes on through the pipeline.

    `_to_atoms` attaches MACE's label keys and the charge/spin; done in place, every
    structure the trainer saw would carry them onward, and the positions buffer it shares
    is the one `structure_digest` hashes into downstream cache keys. The `.copy()` is the
    whole protection (`Structure.atoms` cannot be write-locked the way the force arrays
    are), so this pins it.
    """
    struct = _labelled("0")
    before = struct.atoms.get_positions().copy()
    labelled = _to_atoms(struct, _plan(tmp_path))
    assert struct.atoms.info == {}
    assert FORCES_KEY not in struct.atoms.arrays
    assert np.array_equal(struct.atoms.get_positions(), before)
    assert ENERGY_KEY in labelled.info and FORCES_KEY in labelled.arrays


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


@pytest.mark.integration
def test_mace_loads_the_dataset_we_write(tmp_path: Path):
    """End to end against the real loader: the labels arrive, and they carry full weight.

    A dataset MACE can open but whose labels it weights at zero trains a model on nothing —
    which is what a key mismatch did on MACE < 0.3.13, before it began refusing outright.
    The dataset is written here, in the orchestrator's process, exactly as a run writes it;
    only the *loading* runs under MACE's own interpreter, exactly as a training job loads it.
    """
    python = _require_backend_python("mace_off")
    plan = _plan(tmp_path, charge=-1, multiplicity=1)
    split = DatasetSplit(train=tuple(_labelled(str(i)) for i in range(4)), valid=(), test=())
    data = MaceTrainer().write_dataset(plan, split)

    loaded = _in_backend(
        python,
        "import json, sys\n"
        "from mace.data.utils import KeySpecification, load_from_xyz\n"
        "_, configs = load_from_xyz(\n"
        "    file_path=sys.argv[1], key_specification=KeySpecification.from_defaults()\n"
        ")\n"
        "print(json.dumps({\n"
        '    "energy_weight": configs[0].property_weights["energy"],\n'
        '    "forces_weight": configs[0].property_weights["forces"],\n'
        '    "total_charge": configs[0].properties["total_charge"],\n'
        "}))\n",
        str(data.train),
    )
    assert loaded["energy_weight"] == 1.0
    assert loaded["forces_weight"] == 1.0
    assert loaded["total_charge"] == -1.0


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def test_the_command_runs_under_the_backends_own_interpreter(tmp_path: Path):
    """A bare `mace_run_train` is on nobody's PATH once MACE has an environment of its own."""
    body = MaceTrainer().command(_plan(tmp_path), tmp_path / "train" / "step1_train.yaml")

    assert body.startswith("/envs/mlip-mace/bin/python -m mace.cli.run_train")
    assert body.endswith("--config step1_train.yaml"), "basename: the job runs inside $WORK_DIR"


def test_a_multi_gpu_step_runs_under_torchrun(tmp_path: Path):
    """The base's one torchrun spelling, with MACE's own trainer argv after it."""
    body = MaceTrainer().command(_plan(tmp_path, gpus=4), tmp_path / "cfg.yaml")

    assert "-m torch.distributed.run --standalone --nnodes 1 --nproc_per_node 4" in body
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


@pytest.mark.integration
def test_fairchem_loads_the_dataset_we_write(tmp_path: Path):
    """End to end against the real loader, as the MACE half is."""
    python = _require_backend_python("omol")
    plan = _fc_plan(tmp_path)
    data = FairchemTrainer().write_dataset(plan, _fc_split(n_train=4))

    loaded = _in_backend(
        python,
        "import json, sys\n"
        "from fairchem.core.datasets.ase_datasets import AseDBDataset\n"
        "ds = AseDBDataset(\n"
        "    config={\n"
        '        "src": sys.argv[1],\n'
        '        "metadata_path": sys.argv[2],\n'
        '        "a2g_args": {"r_energy": True, "r_forces": True},\n'
        "    }\n"
        ")\n"
        "print(json.dumps({\n"
        '    "n": len(ds),\n'
        '    "energy0": float(ds[0].energy),\n'
        '    "forces_shape": list(ds[0].forces.shape),\n'
        '    "natoms": [int(n) for n in ds.get_metadata("natoms", [0, 1])],\n'
        "}))\n",
        str(data.train),
        str(data.train.parent / _METADATA_NAME),
    )
    assert loaded["n"] == 4
    assert loaded["energy0"] == pytest.approx(-1.5 * HARTREE_TO_EV, rel=1e-5)
    assert loaded["forces_shape"] == [3, 3]
    assert loaded["natoms"] == [3, 3]


# ---------------------------------------------------------------------------
# SevenNet — extxyz via the reconstructed calculator, the sevenn CLI, checkpoint_best
# ---------------------------------------------------------------------------


def test_sevenn_labels_ride_on_the_reconstructed_calculator(tmp_path: Path):
    """SevenNet's loader takes energy and forces off ``atoms.calc`` — ase's round trip.

    Its reader tries ``get_potential_energy(force_consistent=True)`` first, so
    ``free_energy`` is written alongside ``energy``; both come back on the
    ``SinglePointCalculator`` ase reconstructs from an extxyz frame.
    """
    from chemrefine.engines.mlip.backends.sevenn import SevennTrainer

    files = SevennTrainer().write_dataset(_plan(tmp_path), _small_split())
    frames = ase_read(str(files.train), index=":")
    assert len(frames) == 2
    first = frames[0]
    assert first.get_potential_energy() == pytest.approx(-1.5 * HARTREE_TO_EV, rel=1e-12)
    np.testing.assert_allclose(
        first.get_forces(), [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]]
    )
    assert files.valid is not None and files.valid.is_file()
    assert files.test is None  # an empty split gets no file at all


def test_sevenn_runs_its_console_script_with_the_config_basename(tmp_path: Path):
    """``sevenn train <basename> -s`` from the backend env's own bin, quoted."""
    import shlex

    from chemrefine.engines.mlip.backends.sevenn import SevennTrainer

    plan = _plan(tmp_path, launcher=Path("/envs/my mlip/bin/python"))
    cmd = SevennTrainer().command(plan, tmp_path / "step3_train.yaml")
    assert cmd == f"{shlex.quote('/envs/my mlip/bin/sevenn')} train step3_train.yaml -s"


def test_a_multi_gpu_sevenn_step_runs_under_torchrun(tmp_path: Path):
    """SevenNet's own DDP shape: ``torchrun … --no_python sevenn``, ``-d`` replacing ``-s``."""
    from chemrefine.engines.mlip.backends.sevenn import SevennTrainer

    cmd = SevennTrainer().command(_plan(tmp_path, gpus=2), Path("step3_train.yaml"))
    assert "-m torch.distributed.run --standalone --nnodes 1 --nproc_per_node 2" in cmd
    assert "--no_python" in cmd
    assert cmd.endswith("train step3_train.yaml -d")
    assert " -s" not in cmd


def test_the_sevenn_product_is_the_fixed_best_checkpoint(tmp_path: Path):
    """``checkpoint_best.pth`` — fixed by the library, predictable before the run."""
    from chemrefine.engines.mlip.backends.sevenn import SevennTrainer

    artifact = SevennTrainer().artifact(tmp_path / "train", "train")
    assert artifact == tmp_path / "train" / "checkpoint_best.pth"


# ---------------------------------------------------------------------------
# CHGNet — the driver route: rendered YAML in, fixed-name checkpoint out
# ---------------------------------------------------------------------------


def test_chgnet_runs_the_shipped_driver_under_the_backends_interpreter(tmp_path: Path):
    """``-m …train_driver chgnet`` — the shared driver, dispatching back to this class.

    A managed env is a ``chemrefine[mlip-chgnet]`` install, the same fact that lets the
    ExtOpt server run as ``python -m chemrefine.…server`` from one; the driver resolves
    the trainer through the registry over there, so the library stays one dropped-in
    module. The config rides by basename, for the array-sentinel reason MACE's command
    spells out.
    """
    import shlex

    from chemrefine.engines.mlip.backends.chgnet import ChgnetTrainer

    plan = _plan(tmp_path, launcher=Path("/envs/my chgnet/bin/python"))
    cmd = ChgnetTrainer().command(plan, tmp_path / "step3_train.yaml")
    quoted = shlex.quote("/envs/my chgnet/bin/python")
    assert cmd == f"{quoted} -m chemrefine.engines.mlip.train.driver chgnet step3_train.yaml"


def test_the_chgnet_product_is_the_fixed_named_save(tmp_path: Path):
    """``{run_name}.pth.tar`` — CHGNet's own best checkpoints embed epoch and error."""
    from chemrefine.engines.mlip.backends.chgnet import ChgnetTrainer

    artifact = ChgnetTrainer().artifact(tmp_path / "train", "train")
    assert artifact == tmp_path / "train" / "train.pth.tar"


# ---------------------------------------------------------------------------
# The backend-side hook — chemistry conventions pinned against fakes
# ---------------------------------------------------------------------------


class _FakeStructureData:
    """Records what CHGNet's dataset would be built from."""

    instances: ClassVar[list[_FakeStructureData]] = []

    def __init__(self, *, structures, energies, forces):
        self.structures = structures
        self.energies = energies
        self.forces = forces
        type(self).instances.append(self)


def _install_fake_chgnet_stack(monkeypatch) -> dict:
    """Fake pymatgen / chgnet / torch for ``run_training``; ase and yaml stay real."""
    import sys
    import types
    from unittest.mock import MagicMock

    _FakeStructureData.instances = []
    recorded: dict = {}

    ase_mod = types.ModuleType("pymatgen.io.ase")
    ase_mod.AseAtomsAdaptor = types.SimpleNamespace(
        get_structure=lambda atoms: ("PMG", atoms.get_chemical_formula())
    )
    io_mod = types.ModuleType("pymatgen.io")
    io_mod.ase = ase_mod
    pmg_mod = types.ModuleType("pymatgen")
    pmg_mod.io = io_mod

    dataset_mod = types.ModuleType("chgnet.data.dataset")
    dataset_mod.StructureData = _FakeStructureData
    dataset_mod.get_loader = MagicMock(side_effect=lambda ds, *, batch_size: ("LOADER", ds))
    data_mod = types.ModuleType("chgnet.data")
    data_mod.dataset = dataset_mod

    best = MagicMock()
    best.as_dict.return_value = {"state_dict": "BEST"}
    trainer_instance = MagicMock()
    trainer_instance.best_model = best
    trainer_cls = MagicMock(return_value=trainer_instance)
    trainer_mod = types.ModuleType("chgnet.trainer")
    trainer_mod.Trainer = trainer_cls

    chgnet_cls = MagicMock()
    chgnet_cls.load.return_value = "RELEASED_MODEL"
    chgnet_cls.from_file.return_value = "LOCAL_MODEL"
    model_mod = types.ModuleType("chgnet.model")
    model_mod.CHGNet = chgnet_cls
    chgnet_mod = types.ModuleType("chgnet")
    chgnet_mod.model = model_mod
    chgnet_mod.trainer = trainer_mod
    chgnet_mod.data = data_mod

    torch_mod = types.ModuleType("torch")
    torch_mod.save = MagicMock(
        side_effect=lambda payload, path: recorded.update(saved=(payload, str(path)))
    )

    for name, mod in {
        "pymatgen": pmg_mod,
        "pymatgen.io": io_mod,
        "pymatgen.io.ase": ase_mod,
        "chgnet": chgnet_mod,
        "chgnet.model": model_mod,
        "chgnet.trainer": trainer_mod,
        "chgnet.data": data_mod,
        "chgnet.data.dataset": dataset_mod,
        "torch": torch_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    recorded.update(
        trainer_cls=trainer_cls,
        trainer=trainer_instance,
        chgnet_cls=chgnet_cls,
        get_loader=dataset_mod.get_loader,
        best=best,
    )
    return recorded


def _driver_config(tmp_path: Path, **overrides) -> Path:
    """A rendered driver config over a really-written dataset, as YAML on disk."""
    import yaml

    from chemrefine.engines.mlip.backends.chgnet import ChgnetTrainer

    files = ChgnetTrainer().write_dataset(_plan(tmp_path), _small_split())
    config = {
        "train_set": str(files.train),
        "valid_set": str(files.valid),
        "test_set": "",
        "run_name": "train",
        "device": "cpu",
        "seed": 7,
        "start_from": "",
        "epochs": 3,
        "learning_rate": 1e-3,
        "batch_size": 2,
        "targets": "ef",
    }
    config.update(overrides)
    path = tmp_path / "rendered.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_the_hook_feeds_chgnet_per_atom_energies_without_resplitting(tmp_path: Path, monkeypatch):
    """The two CHGNet facts the hook owns, pinned: eV/atom labels, no second split.

    ``StructureData`` takes **per-atom** energies (CHGNet's own fine-tuning example's
    convention) — a total-energy mistake here mislabels every fine-tune by a factor of
    the atom count. And each split becomes its own ``get_loader`` dataset: CHGNet's
    splitting loader would re-partition what ``split_structures`` already decided.
    Driven through the shared ``train.driver`` entry, exactly as the generated command
    invokes it.
    """
    from chemrefine.engines.mlip.train import driver as train_driver

    recorded = _install_fake_chgnet_stack(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert train_driver.main(["chgnet", str(_driver_config(tmp_path))]) == 0

    train_ds, valid_ds = _FakeStructureData.instances
    assert len(train_ds.structures) == 2 and len(valid_ds.structures) == 1
    # H2O has 3 atoms; the file's energy= field is total eV.
    assert train_ds.energies[0] == pytest.approx(-1.5 * HARTREE_TO_EV / 3, rel=1e-12)
    assert np.asarray(train_ds.forces[0]).shape == (3, 3)
    # Two loaders (no test set), each over one already-split dataset.
    assert recorded["get_loader"].call_count == 2
    train_call = recorded["trainer"].train.call_args
    assert train_call.args[2] is None  # no test loader
    assert train_call.kwargs["save_dir"] == "chgnet_epochs"


def test_the_hook_saves_the_best_model_under_the_promised_name(tmp_path: Path, monkeypatch):
    """``{run_name}.pth.tar`` holding ``{"model": as_dict()}`` — what from_file reads."""
    from chemrefine.engines.mlip.train import driver as train_driver

    recorded = _install_fake_chgnet_stack(monkeypatch)
    monkeypatch.chdir(tmp_path)
    train_driver.main(["chgnet", str(_driver_config(tmp_path))])
    payload, path = recorded["saved"]
    assert path == "train.pth.tar"
    assert payload == {"model": {"state_dict": "BEST"}}
    recorded["chgnet_cls"].load.assert_called_once_with()
    trainer_kwargs = recorded["trainer_cls"].call_args.kwargs
    assert trainer_kwargs["use_device"] == "cpu"
    assert trainer_kwargs["epochs"] == 3
    assert (trainer_kwargs["torch_seed"], trainer_kwargs["data_seed"]) == (7, 7)


def test_the_hook_fine_tunes_from_a_named_checkpoint(tmp_path: Path, monkeypatch):
    """``start_from`` routes through ``from_file`` — the released weights otherwise."""
    from chemrefine.engines.mlip.train import driver as train_driver

    recorded = _install_fake_chgnet_stack(monkeypatch)
    monkeypatch.chdir(tmp_path)
    train_driver.main(["chgnet", str(_driver_config(tmp_path, start_from="/m/prev.pth.tar"))])
    recorded["chgnet_cls"].from_file.assert_called_once_with("/m/prev.pth.tar")
    recorded["chgnet_cls"].load.assert_not_called()


def test_the_hook_falls_back_to_the_final_model_without_a_best(tmp_path: Path, monkeypatch):
    """A run whose metric never improved still saves — ``trainer.model`` stands in."""
    from unittest.mock import MagicMock

    from chemrefine.engines.mlip.backends.chgnet import ChgnetTrainer

    recorded = _install_fake_chgnet_stack(monkeypatch)
    recorded["trainer"].best_model = None
    final = MagicMock()
    final.as_dict.return_value = {"state_dict": "FINAL"}
    recorded["trainer"].model = final
    monkeypatch.chdir(tmp_path)
    import yaml

    config = yaml.safe_load(_driver_config(tmp_path).read_text(encoding="utf-8"))
    assert ChgnetTrainer().run_training(config) == 0
    payload, _path = recorded["saved"]
    assert payload == {"model": {"state_dict": "FINAL"}}


def test_the_hook_refuses_a_config_missing_its_required_keys():
    """The render fills $TRAIN_SET/$VALID_SET/$RUN_NAME; a template naming none fails
    with the placeholders to add, not inside chgnet an hour later."""
    from chemrefine.engines.mlip.backends.chgnet import ChgnetTrainer

    with pytest.raises(SystemExit, match="valid_set"):
        ChgnetTrainer().run_training({"train_set": "/d/t.xyz"})


def test_the_driver_dispatches_by_registry_and_refuses_the_wrong_kind(tmp_path: Path):
    """The shared entry's own refusals: usage, a non-mapping config, a CLI-library task.

    A task whose trainer has no ``run_training`` — MACE and SevenNet drive their own
    CLIs — is told so by name rather than dying on an attribute three frames down; an
    unknown task gets ``trainer_for``'s ordinary vocabulary error.
    """
    import yaml

    from chemrefine.engines.mlip.train import driver as train_driver

    with pytest.raises(SystemExit, match="usage"):
        train_driver.main(["chgnet"])

    listing = tmp_path / "rendered.yaml"
    listing.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a YAML mapping"):
        train_driver.main(["chgnet", str(listing)])

    mapping = tmp_path / "ok.yaml"
    mapping.write_text(yaml.safe_dump({"train_set": "x"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="its library's own CLI"):
        train_driver.main(["mace_off", str(mapping)])


# ---------------------------------------------------------------------------
# ORB — sqlite datasets, the rebuilt loop, a fixed-name state_dict
# ---------------------------------------------------------------------------


def test_orb_datasets_are_ase_sqlite_with_calculator_labels(tmp_path: Path):
    """``AseSqliteDataset`` reads a db path; the labels ride each row's calculator.

    Round-tripped through ase.db itself: the energy comes back off the reconstructed
    calculator in eV, exactly as the adapter will read it. A rerun replaces the file
    rather than appending — ase.db appends by default, and a stacked db would train on
    every previous rerun's rows besides this one's.
    """
    from ase.db import connect

    from chemrefine.engines.mlip.backends.orb import OrbTrainer

    trainer = OrbTrainer()
    files = trainer.write_dataset(_plan(tmp_path), _small_split())
    files = trainer.write_dataset(_plan(tmp_path), _small_split())  # the rerun
    assert files.train is not None and files.train.suffix == ".db"
    with connect(str(files.train)) as db:
        rows = list(db.select())
    assert len(rows) == 2  # replaced, not stacked
    atoms = rows[0].toatoms()
    assert atoms.get_potential_energy() == pytest.approx(-1.5 * HARTREE_TO_EV, rel=1e-12)
    assert files.valid is not None and files.valid.is_file()


def test_orb_runs_the_shared_driver_under_the_backends_interpreter(tmp_path: Path):
    import shlex

    from chemrefine.engines.mlip.backends.orb import OrbTrainer

    plan = _plan(tmp_path, launcher=Path("/envs/my orb/bin/python"))
    cmd = OrbTrainer().command(plan, tmp_path / "step3_train.yaml")
    quoted = shlex.quote("/envs/my orb/bin/python")
    assert cmd == f"{quoted} -m chemrefine.engines.mlip.train.driver orb step3_train.yaml"


def test_the_orb_product_is_the_fixed_named_checkpoint(tmp_path: Path):
    from chemrefine.engines.mlip.backends.orb import OrbTrainer

    assert OrbTrainer().artifact(tmp_path / "train", "train") == tmp_path / "train" / "train.ckpt"


# ---------------------------------------------------------------------------
# The ORB backend-side hook — the rebuilt loop's wiring, pinned against fakes
# ---------------------------------------------------------------------------


class _FakeOrbLoader:
    """Iterable of fake batches; ``len`` is the steps-per-epoch the hook derives."""

    def __init__(self, dataset, *, num_workers, worker_init_fn, collate_fn, batch_sampler):
        from unittest.mock import MagicMock

        self.collate_fn = collate_fn
        batch = MagicMock()
        batch.to.return_value = batch
        self._batches = [batch, batch]

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


def _install_fake_orb_stack(monkeypatch) -> dict:
    """Fake orb_models + torch for ``run_training``; ase and yaml stay real."""
    import sys
    import types
    from unittest.mock import MagicMock

    recorded: dict = {}

    model = MagicMock()
    loss_out = types.SimpleNamespace(loss=MagicMock())
    model.loss.return_value = loss_out
    model.state_dict.return_value = {"w": 1}
    adapter = MagicMock()
    loader_fn = MagicMock(return_value=(model, adapter))

    pretrained = types.ModuleType("orb_models.forcefield.pretrained")
    pretrained.orb_v3_conservative_inf_omat = loader_fn
    forcefield = types.ModuleType("orb_models.forcefield")
    forcefield.pretrained = pretrained

    prop_defs = types.ModuleType("orb_models.common.dataset.property_definitions")
    prop_defs.instantiate_property_config = MagicMock(return_value="TARGETS")
    ase_ds_mod = types.ModuleType("orb_models.common.dataset.ase_sqlite_dataset")
    dataset = MagicMock()
    dataset.__len__ = lambda self: 3
    ase_ds_mod.AseSqliteDataset = MagicMock(return_value=dataset)
    loaders_mod = types.ModuleType("orb_models.common.dataset.loaders")
    loaders_mod.worker_init_fn = MagicMock()
    dataset_pkg = types.ModuleType("orb_models.common.dataset")
    dataset_pkg.property_definitions = prop_defs
    dataset_pkg.ase_sqlite_dataset = ase_ds_mod
    dataset_pkg.loaders = loaders_mod

    optimizer, scheduler = MagicMock(), MagicMock()
    util_mod = types.ModuleType("orb_models.common.training.util")
    util_mod.get_optim = MagicMock(return_value=(optimizer, scheduler))
    # A value distinct from any torch.device result, so an assertion can tell which
    # source the hook took the device from — orb's own pick vs the config's.
    util_mod.init_device = MagicMock(return_value="INIT_DEV")
    training_pkg = types.ModuleType("orb_models.common.training")
    training_pkg.util = util_mod
    utils_mod = types.ModuleType("orb_models.common.utils")
    utils_mod.seed_everything = MagicMock()
    common_pkg = types.ModuleType("orb_models.common")
    common_pkg.dataset = dataset_pkg
    common_pkg.training = training_pkg
    common_pkg.utils = utils_mod
    orb_pkg = types.ModuleType("orb_models")
    orb_pkg.common = common_pkg
    orb_pkg.forcefield = forcefield

    tud = types.ModuleType("torch.utils.data")
    tud.DataLoader = _FakeOrbLoader
    tud.BatchSampler = MagicMock(side_effect=lambda s, *, batch_size, drop_last: ("BS", batch_size))
    tud.RandomSampler = MagicMock(side_effect=lambda ds: ("RS", ds))
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.data = tud
    nn_utils = types.ModuleType("torch.nn.utils")
    nn_utils.clip_grad_norm_ = MagicMock()
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.utils = nn_utils
    torch_mod = types.ModuleType("torch")
    torch_mod.utils = torch_utils
    torch_mod.nn = torch_nn
    torch_mod.device = MagicMock(side_effect=lambda name: f"DEV:{name}")
    torch_mod.save = MagicMock(
        side_effect=lambda payload, path: recorded.setdefault("saves", []).append(str(path))
    )

    for name, mod in {
        "orb_models": orb_pkg,
        "orb_models.common": common_pkg,
        "orb_models.common.dataset": dataset_pkg,
        "orb_models.common.dataset.property_definitions": prop_defs,
        "orb_models.common.dataset.ase_sqlite_dataset": ase_ds_mod,
        "orb_models.common.dataset.loaders": loaders_mod,
        "orb_models.common.training": training_pkg,
        "orb_models.common.training.util": util_mod,
        "orb_models.common.utils": utils_mod,
        "orb_models.forcefield": forcefield,
        "orb_models.forcefield.pretrained": pretrained,
        "torch": torch_mod,
        "torch.utils": torch_utils,
        "torch.utils.data": tud,
        "torch.nn": torch_nn,
        "torch.nn.utils": nn_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    recorded.update(
        loader_fn=loader_fn,
        model=model,
        loss=loss_out.loss,
        optimizer=optimizer,
        scheduler=scheduler,
        get_optim=util_mod.get_optim,
        init_device=util_mod.init_device,
        seed=utils_mod.seed_everything,
        clip=nn_utils.clip_grad_norm_,
        dataset_cls=ase_ds_mod.AseSqliteDataset,
    )
    return recorded


def _orb_config(**overrides) -> dict:
    config = {
        "train_set": "/data/train.db",
        "run_name": "train",
        "base_model": "orb_v3_conservative_inf_omat",
        "device": "cpu",
        "seed": 11,
        "start_from": "",
        "epochs": 2,
        "learning_rate": 1e-4,
        "batch_size": 4,
        "gradient_clip": 0.5,
    }
    config.update(overrides)
    return config


def test_the_orb_hook_rebuilds_the_scripts_loop(monkeypatch, tmp_path: Path):
    """The wiring the unpackaged script established, held by this hook.

    Loader with ``train=True``; the sqlite dataset with the adapter's own collate;
    ``get_optim(lr, epochs*steps, model)``; per-batch backward → clip → step →
    scheduler; a per-epoch ``checkpoint_epoch{n}.ckpt`` and the fixed final save the
    script never had.
    """
    from chemrefine.engines.mlip.backends.orb import OrbTrainer

    recorded = _install_fake_orb_stack(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert OrbTrainer().run_training(_orb_config()) == 0

    # DEV:cpu = the config's own device through torch.device — never orb's
    # init_device(), whose unconditional cuda-if-available pick would put a CPU-booked
    # step on a GPU the scheduler never charged (init_device returns a distinct
    # sentinel here precisely so the source is provable).
    recorded["loader_fn"].assert_called_once_with(device="DEV:cpu", train=True)
    recorded["init_device"].assert_not_called()
    recorded["seed"].assert_called_once_with(11)
    recorded["get_optim"].assert_called_once()
    lr, total_steps, _model = recorded["get_optim"].call_args.args
    assert (lr, total_steps) == (1e-4, 2 * 2)  # epochs x steps_per_epoch
    assert recorded["loss"].backward.call_count == 4
    assert recorded["optimizer"].step.call_count == 4
    assert recorded["scheduler"].step.call_count == 4
    assert recorded["clip"].call_count == 4
    assert recorded["saves"] == [
        "checkpoint_epoch0.ckpt",
        "checkpoint_epoch1.ckpt",
        "train.ckpt",
    ]


def test_the_orb_hook_fine_tunes_from_a_local_checkpoint(monkeypatch, tmp_path: Path):
    """A ``start_from`` file rides the loaders' own ``weights_path`` door."""
    from chemrefine.engines.mlip.backends.orb import OrbTrainer

    recorded = _install_fake_orb_stack(monkeypatch)
    monkeypatch.chdir(tmp_path)
    OrbTrainer().run_training(_orb_config(start_from="/models/prev.ckpt"))
    recorded["loader_fn"].assert_called_once_with(
        device="DEV:cpu", train=True, weights_path="/models/prev.ckpt"
    )


def test_a_config_without_a_device_falls_back_to_orbs_own_pick(monkeypatch, tmp_path: Path):
    """No ``device`` key: the pre-existing behaviour stands — orb's ``init_device``.

    Only a hand-written config can reach this (the engine's render always fills
    ``$DEVICE``); the fallback keeps such a config running rather than refusing it.
    """
    from chemrefine.engines.mlip.backends.orb import OrbTrainer

    recorded = _install_fake_orb_stack(monkeypatch)
    monkeypatch.chdir(tmp_path)
    config = _orb_config(epochs=1)
    del config["device"]
    assert OrbTrainer().run_training(config) == 0

    recorded["init_device"].assert_called_once_with()
    recorded["loader_fn"].assert_called_once_with(device="INIT_DEV", train=True)


def test_the_orb_hook_refuses_what_it_cannot_invent(monkeypatch, tmp_path: Path):
    from chemrefine.engines.mlip.backends.orb import OrbTrainer

    _install_fake_orb_stack(monkeypatch)
    with pytest.raises(SystemExit, match="base_model"):
        OrbTrainer().run_training({"train_set": "/d/t.db", "run_name": "train"})
    with pytest.raises(SystemExit, match="unknown base_model"):
        OrbTrainer().run_training(_orb_config(base_model="orb_not_a_loader"))


def test_the_orb_hook_tolerates_an_optimizer_without_a_scheduler(monkeypatch, tmp_path: Path):
    """``get_optim`` may return no scheduler; the loop must step without one."""
    from chemrefine.engines.mlip.backends.orb import OrbTrainer

    recorded = _install_fake_orb_stack(monkeypatch)
    recorded["get_optim"].return_value = (recorded["optimizer"], None)
    recorded["get_optim"].side_effect = None
    monkeypatch.chdir(tmp_path)
    assert OrbTrainer().run_training(_orb_config(epochs=1)) == 0
    assert recorded["optimizer"].step.call_count == 2
    assert recorded["scheduler"].step.call_count == 0
