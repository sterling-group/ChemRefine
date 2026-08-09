"""Tests for the MLIP backend registry in :mod:`chemrefine.engines.mlip`.

``task_name`` is the registry key (the method/head); ``model_name`` is the
weights handed to the builder. The real ML backends aren't installed in the
dev / CI env, so each test installs a *fake* module under the right
``sys.modules`` key before instantiating :class:`MlipCalculator`, so the real
built-in builder bodies (in ``chemrefine.engines.mlip.backends.*``) run without
a multi-GB ML stack.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from chemrefine.engines.mlip.calculator import MlipCalculator, build_calculator
from chemrefine.errors import ConfigError


def _fake_module(name: str, **attrs: object) -> types.ModuleType:
    """Build a ``types.ModuleType`` with ``attrs`` set (dynamic, no type: ignore)."""
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


# ---------------------------------------------------------------------------
# MACE — task_name selects the variant; model_name is the size
# ---------------------------------------------------------------------------


def _install_fake_mace(monkeypatch) -> dict[str, MagicMock]:
    """Insert a fake ``mace.calculators`` exposing the three factories."""
    factories = {
        "mace_off": MagicMock(return_value="OFF_CALC"),
        "mace_mp": MagicMock(return_value="MP_CALC"),
        "mace_omol": MagicMock(return_value="OMOL_CALC"),
        "MACECalculator": MagicMock(return_value="CUSTOM_CALC"),
    }
    mod = _fake_module("mace.calculators", **factories)
    monkeypatch.setitem(sys.modules, "mace", _fake_module("mace", calculators=mod))
    monkeypatch.setitem(sys.modules, "mace.calculators", mod)
    return factories


def test_build_mace_off_uses_model_name_as_size(monkeypatch):
    factories = _install_fake_mace(monkeypatch)
    calc = MlipCalculator(task_name="mace_off", model_name="medium", device="cpu")
    factories["mace_off"].assert_called_once_with(model="medium", device="cpu")
    assert calc.calculator == "OFF_CALC"


def test_build_mace_mp_uses_model_name_as_size(monkeypatch):
    factories = _install_fake_mace(monkeypatch)
    calc = MlipCalculator(task_name="mace_mp", model_name="large", device="cpu")
    factories["mace_mp"].assert_called_once_with(model="large", device="cpu")
    assert calc.calculator == "MP_CALC"


def test_build_mace_omol_uses_model_name_as_size(monkeypatch):
    factories = _install_fake_mace(monkeypatch)
    calc = MlipCalculator(task_name="mace_omol", model_name="extra_large", device="cpu")
    factories["mace_omol"].assert_called_once_with(model="extra_large", device="cpu")
    assert calc.calculator == "OMOL_CALC"


def test_build_mace_default_size_is_none(monkeypatch):
    """An empty ``model_name`` falls back to the library default (``model=None``)."""
    factories = _install_fake_mace(monkeypatch)
    MlipCalculator(task_name="mace_off", model_name="", device="cpu")
    factories["mace_off"].assert_called_once_with(model=None, device="cpu")


def test_build_calculator_unknown_task_raises(monkeypatch):
    """A `ConfigError` — an unknown task is a config mistake and carries the CLI's exit code."""
    _install_fake_mace(monkeypatch)
    with pytest.raises(ConfigError, match="unsupported MLIP backend"):
        build_calculator(task_name="mace_weird", model_name="x", device="cpu")


@pytest.mark.parametrize("task", ["mace_off", "mace_mp", "mace_omol"])
def test_a_local_checkpoint_goes_through_the_named_familys_own_loader(
    monkeypatch, tmp_path: Path, task: str
):
    """MACE's foundation loaders take a path, so a checkpoint needs no separate task key.

    ``mace_off``'s ``model`` is typed ``str | Path`` and documented as "path to the model";
    only a known name or an ``https:`` URL is treated as a download. That is what lets one
    builder serve released weights and fine-tuned ones alike — and why there is no
    ``custom_fairchem`` to mirror the old ``custom_mace``.
    """
    factories = _install_fake_mace(monkeypatch)
    model_file = tmp_path / "fake.model"
    model_file.touch()
    MlipCalculator(task_name=task, model_name="ignored", device="cuda", model_path=str(model_file))
    factories[task].assert_called_once_with(model=model_file, device="cuda")


def test_a_checkpoint_that_is_not_there_names_the_path(tmp_path: Path):
    """Checked before MACE sees it: its own failure is a `torch.load` traceback.

    That traceback names neither the step nor the option the path came from, which on a
    pipeline whose training step ran overnight is the difference between a typo and a hunt.
    """
    missing = tmp_path / "does_not_exist.model"
    with pytest.raises(FileNotFoundError, match=f"MACE checkpoint not found: {missing}"):
        MlipCalculator(task_name="mace_off", model_name="x", model_path=str(missing))


def test_the_legacy_alias_without_a_checkpoint_says_what_to_write_instead(monkeypatch):
    """``custom_mace`` is kept for old configs, and can only mean "load this file".

    Naming it with nothing to load is the one case it cannot serve, so it says which family to
    name rather than silently picking one — the alias predates the families and has no way to
    know which was meant.
    """
    _install_fake_mace(monkeypatch)
    with pytest.raises(ConfigError, match="mace_off, mace_mp, mace_omol"):
        MlipCalculator(task_name="custom_mace", model_name="medium")


def test_the_legacy_alias_still_loads_a_checkpoint(monkeypatch, tmp_path: Path):
    """A v1 config naming ``custom_mace`` + ``model_path`` keeps working unchanged."""
    factories = _install_fake_mace(monkeypatch)
    model_file = tmp_path / "fake.model"
    model_file.touch()
    MlipCalculator(task_name="custom_mace", model_path=str(model_file), device="cpu")
    factories["mace_off"].assert_called_once_with(model=model_file, device="cpu")


# ---------------------------------------------------------------------------
# FAIRChem (UMA / eSEN) — task_name is the head, model_name the checkpoint
# ---------------------------------------------------------------------------


def _install_fake_fairchem(monkeypatch) -> tuple[MagicMock, MagicMock]:
    """Install a fake ``fairchem.core``; return ``(get_predict_unit, calc_class)``."""
    predict = MagicMock()
    predict.get_predict_unit = MagicMock(return_value="PREDICTOR")
    fairchem_calc = MagicMock(return_value="FAIRCHEM_CALC")
    core = _fake_module("fairchem.core", FAIRChemCalculator=fairchem_calc, pretrained_mlip=predict)
    monkeypatch.setitem(sys.modules, "fairchem", _fake_module("fairchem", core=core))
    monkeypatch.setitem(sys.modules, "fairchem.core", core)
    return predict.get_predict_unit, fairchem_calc


def _install_fake_load_predict_unit(monkeypatch) -> MagicMock:
    """Add the *other* loader — the one that takes a path — to the fake fairchem.

    It lives in a different submodule from `get_predict_unit`, which is why the builder
    imports it inside the branch that uses it: a caller running a named release must not need
    this one to be importable.
    """
    loader = MagicMock(return_value="PATH_PREDICTOR")
    unit = _fake_module("fairchem.core.units.mlip_unit", load_predict_unit=loader)
    monkeypatch.setitem(sys.modules, "fairchem.core.units", _fake_module("fairchem.core.units"))
    monkeypatch.setitem(sys.modules, "fairchem.core.units.mlip_unit", unit)
    return loader


def test_build_fairchem_loads_a_local_checkpoint_through_the_path_api(monkeypatch, tmp_path):
    """A model a training step produced can only be run through `load_predict_unit`.

    `get_predict_unit` resolves a *registry name* and raises `KeyError` for a path — there is
    no path branch in it — so without this the FAIRChem trainer could produce a checkpoint
    that nothing in the pipeline could then load.
    """
    _get_predict_unit, fairchem_calc = _install_fake_fairchem(monkeypatch)
    loader = _install_fake_load_predict_unit(monkeypatch)
    ckpt = tmp_path / "inference_ckpt.pt"
    ckpt.touch()

    calc = MlipCalculator(task_name="omol", model_path=str(ckpt), device="cpu")

    loader.assert_called_once_with(str(ckpt), device="cpu")
    _get_predict_unit.assert_not_called()
    fairchem_calc.assert_called_once_with("PATH_PREDICTOR", task_name="omol")
    assert calc.calculator == "FAIRCHEM_CALC"


def test_build_fairchem_missing_checkpoint_raises(monkeypatch, tmp_path):
    """Named a checkpoint that is not there — say so, rather than fail inside torch.load."""
    _install_fake_fairchem(monkeypatch)
    _install_fake_load_predict_unit(monkeypatch)
    with pytest.raises(FileNotFoundError, match="FAIRChem checkpoint not found"):
        MlipCalculator(task_name="omol", model_path=str(tmp_path / "absent.pt"))


def test_build_fairchem_routes_through_predictor(monkeypatch):
    """The builder loads the checkpoint then constructs a ``FAIRChemCalculator``."""
    get_predict_unit, fairchem_calc = _install_fake_fairchem(monkeypatch)
    calc = MlipCalculator(task_name="omol", model_name="uma-s-1p2", device="cuda")
    get_predict_unit.assert_called_once_with(model_name="uma-s-1p2", device="cuda")
    fairchem_calc.assert_called_once_with("PREDICTOR", task_name="omol")
    assert calc.calculator == "FAIRCHEM_CALC"


@pytest.mark.parametrize("head", ["omat", "odac", "oc20", "oc22", "oc25", "omc"])
def test_build_fairchem_passes_non_omol_head_through(monkeypatch, head):
    """Every non-``omol`` head passes through: ``task_name: omat`` builds the omat head."""
    _get_predict_unit, fairchem_calc = _install_fake_fairchem(monkeypatch)
    MlipCalculator(task_name=head, model_name="uma-s-1p2", device="cpu")
    fairchem_calc.assert_called_once_with("PREDICTOR", task_name=head)


def test_build_fairchem_defaults_model_name(monkeypatch):
    """An empty ``model_name`` falls back to the default UMA checkpoint."""
    get_predict_unit, _calc = _install_fake_fairchem(monkeypatch)
    MlipCalculator(task_name="omol", model_name="", device="cpu")
    get_predict_unit.assert_called_once_with(model_name="uma-s-1p2", device="cpu")


# ---------------------------------------------------------------------------
# CHGNet
# ---------------------------------------------------------------------------


def _install_fake_chgnet(monkeypatch) -> tuple[MagicMock, MagicMock]:
    """Install a fake ``chgnet.model`` (canonical import); return ``(loader, calc)``."""
    loader = MagicMock(return_value="CHGNET_MODEL")
    chgnet_cls = MagicMock()
    chgnet_cls.load = loader
    calc_class = MagicMock(return_value="CHGNET_CALC")
    model_mod = _fake_module("chgnet.model", CHGNet=chgnet_cls, CHGNetCalculator=calc_class)
    monkeypatch.setitem(sys.modules, "chgnet", _fake_module("chgnet", model=model_mod))
    monkeypatch.setitem(sys.modules, "chgnet.model", model_mod)
    return loader, calc_class


def test_build_chgnet_uses_default_when_no_model_path(monkeypatch):
    loader, calc_class = _install_fake_chgnet(monkeypatch)
    calc = MlipCalculator(task_name="chgnet", model_name="ignored", device="cpu")
    loader.assert_called_once_with()
    calc_class.assert_called_once_with(model="CHGNET_MODEL", use_device="cpu")
    assert calc.calculator == "CHGNET_CALC"


# ---------------------------------------------------------------------------
# SevenNet — task_name selects it, model_name is the checkpoint
# ---------------------------------------------------------------------------


def _install_fake_sevenn(monkeypatch) -> MagicMock:
    factory = MagicMock(return_value="SEVENN_CALC")
    mod = _fake_module("sevenn.calculator", SevenNetCalculator=factory)
    monkeypatch.setitem(sys.modules, "sevenn", _fake_module("sevenn", calculator=mod))
    monkeypatch.setitem(sys.modules, "sevenn.calculator", mod)
    return factory


def test_build_sevenn_constructs_calculator(monkeypatch):
    factory = _install_fake_sevenn(monkeypatch)
    calc = MlipCalculator(task_name="sevenn", model_name="7net-0", device="cpu")
    factory.assert_called_once_with(model="7net-0", device="cpu")
    assert calc.calculator == "SEVENN_CALC"


# ---------------------------------------------------------------------------
# ORB — a separate library, model_name names the loader
# ---------------------------------------------------------------------------


def _install_fake_orb(monkeypatch) -> tuple[MagicMock, MagicMock, object]:
    """Install a fake ``orb_models`` tree; return ``(loader, calc_class, orbff)``."""
    orbff = object()
    loader = MagicMock(return_value=orbff)
    pretrained = _fake_module(
        "orb_models.forcefield.pretrained", orb_v3_conservative_inf_omat=loader
    )
    forcefield = _fake_module("orb_models.forcefield", pretrained=pretrained)
    calc_class = MagicMock(return_value="ORB_CALC")
    calc_mod = _fake_module("orb_models.forcefield.inference.calculator", ORBCalculator=calc_class)
    inference = _fake_module("orb_models.forcefield.inference")
    for name, mod in {
        "orb_models": _fake_module("orb_models", forcefield=forcefield),
        "orb_models.forcefield": forcefield,
        "orb_models.forcefield.pretrained": pretrained,
        "orb_models.forcefield.inference": inference,
        "orb_models.forcefield.inference.calculator": calc_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return loader, calc_class, orbff


def test_build_orb_constructs_real_orb_calculator(monkeypatch):
    loader, calc_class, orbff = _install_fake_orb(monkeypatch)
    calc = MlipCalculator(task_name="orb", model_name="orb_v3_conservative_inf_omat", device="cpu")
    loader.assert_called_once_with(device="cpu")
    calc_class.assert_called_once_with(orbff, device="cpu")
    assert calc.calculator == "ORB_CALC"


def test_build_orb_unknown_loader_raises(monkeypatch):
    """A model that isn't a loader in orb_models.forcefield.pretrained errors."""
    _install_fake_orb(monkeypatch)
    with pytest.raises(ValueError, match="unknown ORB model"):
        MlipCalculator(task_name="orb", model_name="orb_not_a_loader", device="cpu")


# ---------------------------------------------------------------------------
# single_point + optimize — duck-typed fake ASE calculator
# ---------------------------------------------------------------------------


def _calc_with_fake(monkeypatch) -> MlipCalculator:
    """Construct an ``MlipCalculator`` with a fake backend already installed."""
    _install_fake_mace(monkeypatch)
    return MlipCalculator(task_name="mace_off", model_name="small", device="cpu")


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
    energy, gradient = calc.single_point(atoms)
    assert energy == -42.0
    assert gradient == [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert atoms.calc is calc.calculator


def test_optimize_invokes_lbfgs_with_fmax_and_steps(monkeypatch):
    """``optimize`` runs ``ase.optimize.LBFGS(...).run(fmax=…, steps=…)``."""
    calc = _calc_with_fake(monkeypatch)

    lbfgs_instance = MagicMock()
    lbfgs_class = MagicMock(return_value=lbfgs_instance)
    monkeypatch.setitem(
        sys.modules, "ase.optimize", _fake_module("ase.optimize", LBFGS=lbfgs_class)
    )

    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    result = calc.optimize(atoms, fmax=0.05, steps=10)
    lbfgs_class.assert_called_once_with(atoms, logfile=os.devnull)
    lbfgs_instance.run.assert_called_once_with(fmax=0.05, steps=10)
    assert result is atoms
    assert atoms.calc is calc.calculator
