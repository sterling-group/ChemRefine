import types
import sys
from chemrefine.mlff import run_mlff_calculation, get_available_device


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

    def dummy_load_model(model_name="mol", device="cpu"):
        class DummyModel:
            def get_calculator(self):
                return object()

        return DummyModel()

    monkeypatch.setitem(sys.modules, "ase.io", types.SimpleNamespace(read=dummy_read))
    monkeypatch.setitem(sys.modules, "ase.optimize", types.SimpleNamespace(BFGS=DummyOpt))
    monkeypatch.setitem(sys.modules, "fairchem.core.models", types.SimpleNamespace(load_model=dummy_load_model))

    coords, energy = run_mlff_calculation(str(xyz), steps=1)
    assert isinstance(coords, list)
    assert abs(energy - (1.0 / 27.211386245988)) < 1e-6


def test_device_selection(monkeypatch, tmp_path):
    xyz = tmp_path / "mol.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")

    def dummy_read(path):
        return DummyAtoms()

    called_devices = []

    def dummy_load_model(model_name="mol", device="cpu"):
        called_devices.append(device)

        class DummyModel:
            def get_calculator(self):
                return object()

        return DummyModel()

    monkeypatch.setitem(sys.modules, "ase.io", types.SimpleNamespace(read=dummy_read))
    monkeypatch.setitem(sys.modules, "ase.optimize", types.SimpleNamespace(BFGS=DummyOpt))
    monkeypatch.setitem(sys.modules, "fairchem.core.models", types.SimpleNamespace(load_model=dummy_load_model))

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)))
    coords, _ = run_mlff_calculation(str(xyz), steps=1, device=None)
    assert called_devices[0] == "cuda"

    called_devices.clear()
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    coords, _ = run_mlff_calculation(str(xyz), steps=1, device=None)
    assert called_devices[0] == "cpu"
