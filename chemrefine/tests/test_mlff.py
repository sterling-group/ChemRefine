import sys
import types
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

    def dummy_get_predict_unit(model_name="mol", device="cpu"):
        class DummyPred:
            pass

        return DummyPred()

    class DummyCalc:
        pass

    def dummy_calculator(pred, task_name="oc20"):
        return DummyCalc()

    monkeypatch.setitem(sys.modules, "ase.io", types.SimpleNamespace(read=dummy_read))
    monkeypatch.setitem(sys.modules, "ase.optimize", types.SimpleNamespace(LBFGS=DummyOpt))
    monkeypatch.setitem(sys.modules, "fairchem.core.pretrained_mlip", types.SimpleNamespace(get_predict_unit=dummy_get_predict_unit))
    monkeypatch.setitem(sys.modules, "fairchem.core", types.SimpleNamespace(FAIRChemCalculator=dummy_calculator, pretrained_mlip=types.SimpleNamespace(get_predict_unit=dummy_get_predict_unit)))

    coords, energy = run_mlff_calculation(str(xyz), steps=1)
    assert isinstance(coords, list)
    assert abs(energy - (1.0 / 27.211386245988)) < 1e-6


def test_device_selection(monkeypatch, tmp_path):
    xyz = tmp_path / "mol.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")

    def dummy_read(path):
        return DummyAtoms()

    called_devices = []

    def dummy_get_predict_unit(model_name="mol", device="cpu"):
        called_devices.append(device)

        class DummyPred:
            pass

        return DummyPred()

    class DummyCalc:
        pass

    def dummy_calculator(pred, task_name="oc20"):
        return DummyCalc()

    monkeypatch.setitem(sys.modules, "ase.io", types.SimpleNamespace(read=dummy_read))
    monkeypatch.setitem(sys.modules, "ase.optimize", types.SimpleNamespace(LBFGS=DummyOpt))
    monkeypatch.setitem(sys.modules, "fairchem.core.pretrained_mlip", types.SimpleNamespace(get_predict_unit=dummy_get_predict_unit))
    monkeypatch.setitem(sys.modules, "fairchem.core", types.SimpleNamespace(FAIRChemCalculator=dummy_calculator, pretrained_mlip=types.SimpleNamespace(get_predict_unit=dummy_get_predict_unit)))

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)))
    run_mlff_calculation(str(xyz), steps=1, device=None)
    assert called_devices[0] == "cuda"

    called_devices.clear()
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    run_mlff_calculation(str(xyz), steps=1, device=None)
    assert called_devices[0] == "cpu"

