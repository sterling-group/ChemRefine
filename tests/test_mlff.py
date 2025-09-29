import sys
import types
import os
from chemrefine.mlff import run_mlff_calculation, MLFFJobSubmitter


class DummyAtoms:
    def __init__(self):
        self.positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        self.symbols = ["H", "H"]
        self.calc = None

    def get_potential_energy(self):
        return 1.0


class DummyOpt:
    def __init__(self, atoms, logfile=None):
        pass

    def run(self, fmax=0.03, steps=200):
        return


def test_run_mlff(monkeypatch, tmp_path):
    xyz = tmp_path / "mol.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")

    def dummy_read(path):
        return DummyAtoms()

    class DummyCalc:
        pass

    def dummy_calculator(**kwargs):
        return DummyCalc()

    monkeypatch.setitem(sys.modules, "ase.io", types.SimpleNamespace(read=dummy_read))
    monkeypatch.setitem(
        sys.modules, "ase.optimize", types.SimpleNamespace(LBFGS=DummyOpt)
    )
    monkeypatch.setitem(
        sys.modules,
        "mace.calculators",
        types.SimpleNamespace(mace_off=dummy_calculator, mace_mp=dummy_calculator),
    )

    coords, energy = run_mlff_calculation(str(xyz), steps=1)
    assert isinstance(coords, list)
    assert abs(energy - (1.0 / 27.211386245988)) < 1e-6


def test_device_selection(monkeypatch, tmp_path):
    xyz = tmp_path / "mol.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")

    def dummy_read(path):
        return DummyAtoms()

    called_devices = []

    def dummy_calculator(**kwargs):
        called_devices.append(kwargs.get("device"))

        class DummyCalc:
            pass

        return DummyCalc()

    monkeypatch.setitem(sys.modules, "ase.io", types.SimpleNamespace(read=dummy_read))
    monkeypatch.setitem(
        sys.modules, "ase.optimize", types.SimpleNamespace(LBFGS=DummyOpt)
    )
    monkeypatch.setitem(
        sys.modules,
        "mace.calculators",
        types.SimpleNamespace(mace_off=dummy_calculator, mace_mp=dummy_calculator),
    )

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
    )
    run_mlff_calculation(str(xyz), steps=1, device=None)
    assert called_devices[0] == "cuda"

    called_devices.clear()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    )
    run_mlff_calculation(str(xyz), steps=1, device=None)
    assert called_devices[0] == "cpu"


def test_mlff_slurm_script(tmp_path):
    header = tmp_path / "mlff.slurm.header"
    header.write_text("#SBATCH --time=01:00:00\nmodule load python\n")
    xyz = tmp_path / "mol.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")

    submitter = MLFFJobSubmitter()
    script = submitter.generate_slurm_script(
        str(xyz), template_dir=tmp_path, output_dir=tmp_path
    )
    assert os.path.exists(script)


def test_env_checkpoint(monkeypatch, tmp_path):
    xyz = tmp_path / "mol.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")
    checkpoint = tmp_path / "mol.model"
    checkpoint.write_text("dummy")

    def dummy_read(path):
        return DummyAtoms()

    captured = {}

    def dummy_calculator(**kwargs):
        captured["cp"] = kwargs.get("model_path")

        class DummyCalc:
            pass

        return DummyCalc()

    monkeypatch.setitem(sys.modules, "ase.io", types.SimpleNamespace(read=dummy_read))
    monkeypatch.setitem(
        sys.modules, "ase.optimize", types.SimpleNamespace(LBFGS=DummyOpt)
    )
    monkeypatch.setitem(
        sys.modules,
        "mace.calculators",
        types.SimpleNamespace(mace_off=dummy_calculator, mace_mp=dummy_calculator),
    )

    monkeypatch.setenv("CHEMREFINE_MLFF_CHECKPOINT", str(checkpoint))
    run_mlff_calculation(str(xyz), steps=1, device="cpu")
    assert captured["cp"] == str(checkpoint)
