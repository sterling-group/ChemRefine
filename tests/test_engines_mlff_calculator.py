"""Tests for the MLFF backend registry in :mod:`chemrefine.engines.mlff.calculator`.

The real ML backends (mace-torch, fairchem-core, chgnet, sevenn) aren't
installed in the dev / CI environment, so each test installs a *fake*
module under the right ``sys.modules`` key before instantiating
:class:`MlffCalculator` (or calling :func:`build_calculator`). That
lets us exercise the real built-in builder bodies and the
``single_point`` / ``optimize`` paths without pulling in a multi-GB
ML stack.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from chemrefine.engines.mlff.calculator import MlffCalculator, build_calculator

# ---------------------------------------------------------------------------
# MACE — mace_off / mace_mp / mace_omol + custom MACE
# ---------------------------------------------------------------------------


def _install_fake_mace(monkeypatch) -> dict[str, MagicMock]:
    """Insert a fake ``mace.calculators`` exposing the three factories."""
    factories = {
        "mace_off": MagicMock(return_value="OFF_CALC"),
        "mace_mp": MagicMock(return_value="MP_CALC"),
        "mace_omol": MagicMock(return_value="OMOL_CALC"),
        "MACECalculator": MagicMock(return_value="CUSTOM_CALC"),
    }
    mod = types.ModuleType("mace.calculators")
    for name, fac in factories.items():
        setattr(mod, name, fac)
    parent = types.ModuleType("mace")
    parent.calculators = mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mace", parent)
    monkeypatch.setitem(sys.modules, "mace.calculators", mod)
    return factories


def test_build_mace_routes_mace_off_to_factory(monkeypatch):
    factories = _install_fake_mace(monkeypatch)
    calc = MlffCalculator(model_name="medium", task_name="mace_off", device="cpu")
    factories["mace_off"].assert_called_once_with(model="medium", device="cpu")
    assert calc.calculator == "OFF_CALC"


def test_build_mace_routes_mace_mp_to_factory(monkeypatch):
    factories = _install_fake_mace(monkeypatch)
    calc = MlffCalculator(model_name="medium", task_name="mace_mp", device="cpu")
    factories["mace_mp"].assert_called_once_with(model="medium", device="cpu")
    assert calc.calculator == "MP_CALC"


def test_build_mace_routes_mace_omol_to_factory(monkeypatch):
    factories = _install_fake_mace(monkeypatch)
    calc = MlffCalculator(model_name="ignored", task_name="mace_omol", device="cpu")
    factories["mace_omol"].assert_called_once_with(device="cpu")
    assert calc.calculator == "OMOL_CALC"


def test_build_calculator_unknown_task_raises(monkeypatch):
    """An unregistered ``task_name`` should raise ``ValueError`` from the dispatcher."""
    _install_fake_mace(monkeypatch)
    with pytest.raises(ValueError, match="unsupported MLFF backend"):
        build_calculator(task_name="mace_weird", model_name="x", device="cpu")


def test_build_custom_mace_constructs_with_fake_module(monkeypatch, tmp_path: Path):
    """``model_path`` short-circuits to the ``custom_mace`` builder."""
    factories = _install_fake_mace(monkeypatch)
    model_file = tmp_path / "fake.model"
    model_file.touch()
    calc = MlffCalculator(
        model_name="ignored",
        task_name="ignored",
        device="cuda",
        model_path=str(model_file),
    )
    factories["MACECalculator"].assert_called_once_with(
        model_path=str(model_file), device="cuda"
    )
    assert calc.calculator == "CUSTOM_CALC"


def test_build_custom_mace_missing_file_raises(tmp_path: Path):
    """The ``custom_mace`` builder validates that the model file exists."""
    missing = tmp_path / "does_not_exist.model"
    with pytest.raises(FileNotFoundError, match="custom MACE model not found"):
        MlffCalculator(
            model_name="ignored", task_name="mace_off", model_path=str(missing)
        )


# ---------------------------------------------------------------------------
# FAIRChem
# ---------------------------------------------------------------------------


def test_build_fairchem_routes_through_predictor(monkeypatch):
    """The FAIRChem builder calls ``pretrained_mlip.get_predict_unit`` then
    constructs a ``FAIRChemCalculator`` from the result."""
    predict = MagicMock()
    predict.get_predict_unit = MagicMock(return_value="PREDICTOR")
    fairchem_calc = MagicMock(return_value="FAIRCHEM_CALC")
    fairchem_mod = types.ModuleType("fairchem.core")
    fairchem_mod.FAIRChemCalculator = fairchem_calc  # type: ignore[attr-defined]
    fairchem_mod.pretrained_mlip = predict  # type: ignore[attr-defined]
    parent = types.ModuleType("fairchem")
    parent.core = fairchem_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fairchem", parent)
    monkeypatch.setitem(sys.modules, "fairchem.core", fairchem_mod)

    calc = MlffCalculator(model_name="uma-s-1", task_name="omol", device="cuda")
    predict.get_predict_unit.assert_called_once_with(
        model_name="uma-s-1", device="cuda"
    )
    fairchem_calc.assert_called_once_with("PREDICTOR", task_name="omol")
    assert calc.calculator == "FAIRCHEM_CALC"


# ---------------------------------------------------------------------------
# CHGNet
# ---------------------------------------------------------------------------


def _install_fake_chgnet(monkeypatch) -> tuple[MagicMock, MagicMock]:
    """Install a fake CHGNet module tree; return ``(loader, calc_class)``."""
    loader = MagicMock(return_value="CHGNET_MODEL")
    chgnet_mod = types.ModuleType("chgnet.model")
    chgnet_mod.CHGNet = MagicMock()  # type: ignore[attr-defined]
    chgnet_mod.CHGNet.load = loader
    calc_class = MagicMock(return_value="CHGNET_CALC")
    chgnet_calculators_mod = types.ModuleType("chgnet.calculators")
    chgnet_calculators_mod.CHGNetCalculator = calc_class  # type: ignore[attr-defined]
    parent = types.ModuleType("chgnet")
    parent.model = chgnet_mod  # type: ignore[attr-defined]
    parent.calculators = chgnet_calculators_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chgnet", parent)
    monkeypatch.setitem(sys.modules, "chgnet.model", chgnet_mod)
    monkeypatch.setitem(sys.modules, "chgnet.calculators", chgnet_calculators_mod)
    return loader, calc_class


def test_build_chgnet_uses_default_when_no_model_path(monkeypatch):
    loader, calc_class = _install_fake_chgnet(monkeypatch)
    calc = MlffCalculator(model_name="ignored", task_name="chgnet", device="cpu")
    loader.assert_called_once_with()
    calc_class.assert_called_once_with(model="CHGNET_MODEL")
    assert calc.calculator == "CHGNET_CALC"


# ---------------------------------------------------------------------------
# SevenN + Orb
# ---------------------------------------------------------------------------


def _install_fake_sevenn(monkeypatch) -> MagicMock:
    factory = MagicMock(return_value="SEVENN_CALC")
    mod = types.ModuleType("sevenn.calculator")
    mod.SevenNetCalculator = factory  # type: ignore[attr-defined]
    parent = types.ModuleType("sevenn")
    parent.calculator = mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sevenn", parent)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", mod)
    return factory


def test_build_sevenn_constructs_calculator(monkeypatch):
    """SevenN routes by ``model_name`` prefix → uses ``model_name`` as the SevenNet model."""
    factory = _install_fake_sevenn(monkeypatch)
    calc = MlffCalculator(
        model_name="sevenn-tiny", task_name="custom_task", device="cpu"
    )
    factory.assert_called_once_with(model="sevenn-tiny", device="cpu")
    assert calc.calculator == "SEVENN_CALC"


def test_build_orb_routes_through_sevenn(monkeypatch):
    """ORB routes through the SevenN entrypoint; uses ``model_name`` as the model."""
    factory = _install_fake_sevenn(monkeypatch)
    calc = MlffCalculator(model_name="orb-d3", task_name="custom_task", device="cpu")
    factory.assert_called_once_with(model="orb-d3", device="cpu")
    assert calc.calculator == "SEVENN_CALC"


# ---------------------------------------------------------------------------
# single_point + optimize — duck-typed fake ASE calculator
# ---------------------------------------------------------------------------


def _calc_with_fake(monkeypatch) -> MlffCalculator:
    """Construct an ``MlffCalculator`` with a fake backend already installed."""
    _install_fake_mace(monkeypatch)
    return MlffCalculator(model_name="medium", task_name="mace_off", device="cpu")


def test_single_point_returns_energy_and_negative_gradient(monkeypatch):
    """``single_point`` returns ``(energy, [-fx, -fy, -fz] per atom)``."""
    calc = _calc_with_fake(monkeypatch)
    fake_forces = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    class _StubAtoms:
        def __init__(self):
            self.calc = None

        def get_potential_energy(self):
            return -42.0

        def get_forces(self):
            return fake_forces

    atoms = _StubAtoms()
    energy, gradient = calc.single_point(atoms)  # type: ignore[arg-type]
    assert energy == -42.0
    assert gradient == [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert atoms.calc is calc.calculator


def test_optimize_invokes_lbfgs_with_fmax_and_steps(monkeypatch):
    """``optimize`` runs ``ase.optimize.LBFGS(...).run(fmax=…, steps=…)``."""
    calc = _calc_with_fake(monkeypatch)

    lbfgs_instance = MagicMock()
    lbfgs_class = MagicMock(return_value=lbfgs_instance)
    fake_optimize_mod = types.ModuleType("ase.optimize")
    fake_optimize_mod.LBFGS = lbfgs_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ase.optimize", fake_optimize_mod)

    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    result = calc.optimize(atoms, fmax=0.05, steps=10)
    lbfgs_class.assert_called_once_with(atoms, logfile=None)
    lbfgs_instance.run.assert_called_once_with(fmax=0.05, steps=10)
    assert result is atoms
    assert atoms.calc is calc.calculator
