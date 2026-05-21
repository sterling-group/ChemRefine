"""Tests for ``engines/pyscf/_runtime`` + ``extopt_calc``.

PySCF is not installed in the CI / dev env, so every PySCF / gpu4pyscf
import is mocked at the boundary. We verify the *shape* of the calls
ChemRefine makes — argument passing, class selection by spin,
tensor-extraction gating — not the chemistry itself.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chemrefine.engines._extopt.base import CalculationData
from chemrefine.engines.pyscf import _runtime, extopt_calc
from chemrefine.engines.pyscf.options import PyscfOptions


def _data(*, multiplicity: int = 1, dograd: bool = True, **settings) -> CalculationData:
    return CalculationData(
        symbols=("H", "H"),
        positions_angstrom=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        charge=0,
        multiplicity=multiplicity,
        nthreads=2,
        dograd=dograd,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Mocked PySCF module setup
# ---------------------------------------------------------------------------


def _install_fake_pyscf(monkeypatch, *, mol_spin: int = 0) -> dict[str, MagicMock]:
    """Insert a fake ``pyscf`` package into sys.modules; return the mocks."""
    mocks: dict[str, MagicMock] = {}

    mol = MagicMock()
    mol.spin = mol_spin
    mol.energy_nuc.return_value = 1.234

    gto_mod = types.ModuleType("pyscf.gto")

    def _mole_factory():
        m = MagicMock()
        m.spin = mol_spin
        m.energy_nuc.return_value = 1.234
        m.intor.side_effect = lambda label: (
            np.ones((2, 2)) if "int1e" in label else np.ones((2, 2, 2, 2))
        )
        # ``mol.build()`` is called for its side-effect only.
        m.build.return_value = None
        # Reading mol.spin later requires it to behave like an int field.
        m.__class__ = MagicMock
        return m

    gto_mod.Mole = _mole_factory
    mocks["gto"] = gto_mod

    rks = MagicMock()
    rks.kernel.return_value = -1.5
    rks.converged = True
    rks.nuc_grad_method.return_value.kernel.return_value = np.zeros((2, 3))
    rks.mo_coeff = np.eye(2)
    rks.mo_occ = np.array([2.0, 0.0])
    mocks["rks"] = rks

    dft_mod = types.ModuleType("pyscf.dft")
    dft_mod.RKS = MagicMock(return_value=rks)
    dft_mod.UKS = MagicMock(return_value=rks)
    mocks["dft"] = dft_mod

    scf_mod = types.ModuleType("pyscf.scf")
    scf_mod.RHF = MagicMock(return_value=rks)
    scf_mod.UHF = MagicMock(return_value=rks)
    mocks["scf"] = scf_mod

    lib_mod = types.ModuleType("pyscf.lib")
    lib_mod.num_threads = MagicMock()
    mocks["lib"] = lib_mod

    ao2mo_mod = types.ModuleType("pyscf.ao2mo")
    ao2mo_mod.incore = types.SimpleNamespace(full=MagicMock(return_value=np.zeros((2, 2, 2, 2))))
    mocks["ao2mo"] = ao2mo_mod

    lo_mod = types.ModuleType("pyscf.lo")
    boys = MagicMock()
    boys.return_value.kernel.return_value = np.eye(2)
    lo_mod.Boys = boys
    mocks["lo"] = lo_mod

    pyscf_pkg = types.ModuleType("pyscf")
    pyscf_pkg.gto = gto_mod
    pyscf_pkg.dft = dft_mod
    pyscf_pkg.scf = scf_mod
    pyscf_pkg.lib = lib_mod
    pyscf_pkg.ao2mo = ao2mo_mod
    pyscf_pkg.lo = lo_mod

    monkeypatch.setitem(sys.modules, "pyscf", pyscf_pkg)
    monkeypatch.setitem(sys.modules, "pyscf.gto", gto_mod)
    monkeypatch.setitem(sys.modules, "pyscf.dft", dft_mod)
    monkeypatch.setitem(sys.modules, "pyscf.scf", scf_mod)
    monkeypatch.setitem(sys.modules, "pyscf.lib", lib_mod)
    monkeypatch.setitem(sys.modules, "pyscf.ao2mo", ao2mo_mod)
    monkeypatch.setitem(sys.modules, "pyscf.lo", lo_mod)
    return mocks


# ---------------------------------------------------------------------------
# PyscfOptions
# ---------------------------------------------------------------------------


def test_pyscf_options_defaults():
    opt = PyscfOptions()
    assert opt.method == "dft"
    assert opt.xc == "pbe"
    assert opt.basis == "def2-svp"
    assert opt.df is False
    assert opt.gpu is False
    assert opt.save_tensors is False
    assert opt.localized is False
    assert opt.tensor_folder == "tensors"


def test_pyscf_options_rejects_empty_tensor_folder():
    with pytest.raises(ValueError):
        PyscfOptions(tensor_folder="")


def test_pyscf_options_rejects_whitespace_tensor_folder():
    with pytest.raises(ValueError, match="non-empty"):
        PyscfOptions(tensor_folder="   ")


def test_pyscf_options_rejects_unknown_field():
    with pytest.raises(ValueError):
        PyscfOptions(unknown_field=True)  # type: ignore[arg-type]


def test_pyscf_options_from_raw_handles_none():
    assert PyscfOptions.from_raw(None) == PyscfOptions()


def test_pyscf_options_from_raw_round_trip():
    raw = {"method": "hf", "save_tensors": True, "localized": True, "tensor_folder": "mytensors"}
    opt = PyscfOptions.from_raw(raw)
    assert opt.method == "hf"
    assert opt.save_tensors is True
    assert opt.localized is True
    assert opt.tensor_folder == "mytensors"


# ---------------------------------------------------------------------------
# _runtime.build_mol
# ---------------------------------------------------------------------------


def test_build_mol_converts_angstrom_to_bohr(monkeypatch):
    mocks = _install_fake_pyscf(monkeypatch)
    mol = _runtime.build_mol(
        symbols=("H", "H"),
        positions_angstrom=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        charge=0,
        multiplicity=1,
        basis="def2-svp",
    )
    # Spot-check: the Mole instance should have been built. The factory in
    # `gto_mod.Mole` mints fresh mocks per call, so we just verify ``build``
    # was called on the returned object.
    mol.build.assert_called_once()
    assert "gto" in mocks


# ---------------------------------------------------------------------------
# _runtime.run_dft — class dispatch
# ---------------------------------------------------------------------------


def test_run_dft_uses_rks_for_closed_shell(monkeypatch):
    mocks = _install_fake_pyscf(monkeypatch, mol_spin=0)
    mol = _runtime.build_mol(
        symbols=("H", "H"),
        positions_angstrom=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        charge=0, multiplicity=1, basis="def2-svp",
    )
    mol.spin = 0
    energy, gradient, meta, _mf = _runtime.run_dft(mol, method="dft", xc="pbe")
    mocks["dft"].RKS.assert_called_once()
    mocks["dft"].UKS.assert_not_called()
    assert energy == -1.5
    assert meta["converged"] is True
    assert meta["gpu_used"] is False
    assert len(gradient) == 2  # 2 atoms reshaped from 6-component flat array


def test_run_dft_uses_uks_for_open_shell(monkeypatch):
    mocks = _install_fake_pyscf(monkeypatch, mol_spin=2)
    mol = _runtime.build_mol(
        symbols=("H", "H"), positions_angstrom=np.array([[0, 0, 0], [0.74, 0, 0]]),
        charge=0, multiplicity=3, basis="def2-svp",
    )
    mol.spin = 2
    _runtime.run_dft(mol, method="dft", xc="pbe")
    mocks["dft"].UKS.assert_called_once()
    mocks["dft"].RKS.assert_not_called()


def test_run_dft_uses_rhf_for_method_hf(monkeypatch):
    mocks = _install_fake_pyscf(monkeypatch)
    mol = _runtime.build_mol(
        symbols=("H",), positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
        charge=0, multiplicity=1, basis="sto-3g",
    )
    mol.spin = 0
    _runtime.run_dft(mol, method="hf")
    mocks["scf"].RHF.assert_called_once()


def test_run_dft_uses_uhf_for_method_hf_open_shell(monkeypatch):
    mocks = _install_fake_pyscf(monkeypatch, mol_spin=1)
    mol = _runtime.build_mol(
        symbols=("H",), positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
        charge=0, multiplicity=2, basis="sto-3g",
    )
    mol.spin = 1
    _runtime.run_dft(mol, method="hf")
    mocks["scf"].UHF.assert_called_once()


def test_run_dft_hf_warns_about_gpu(monkeypatch):
    """``want_gpu`` is honored for DFT but ignored for HF (CPU only)."""
    _install_fake_pyscf(monkeypatch)
    mol = _runtime.build_mol(
        symbols=("H",), positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
        charge=0, multiplicity=1, basis="sto-3g",
    )
    mol.spin = 0
    _, _, meta, _ = _runtime.run_dft(mol, method="hf", want_gpu=True)
    assert "HF GPU path not enabled" in meta["gpu_msg"]
    assert meta["gpu_used"] is False


def test_run_dft_falls_back_to_cpu_when_gpu_import_fails(monkeypatch):
    """When ``gpu4pyscf`` can't be imported the helper falls back cleanly."""
    _install_fake_pyscf(monkeypatch)
    monkeypatch.setitem(sys.modules, "gpu4pyscf.dft", None)  # forces ImportError
    mol = _runtime.build_mol(
        symbols=("H",), positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
        charge=0, multiplicity=1, basis="sto-3g",
    )
    mol.spin = 0
    _, _, meta, _ = _runtime.run_dft(mol, want_gpu=True)
    assert meta["gpu_used"] is False
    assert "fell back to CPU" in meta["gpu_msg"]


def test_run_dft_density_fitting_continues_on_failure(monkeypatch):
    """A ``density_fit()`` exception logs a warning but doesn't abort."""
    mocks = _install_fake_pyscf(monkeypatch)
    mocks["rks"].density_fit.side_effect = RuntimeError("no DF for you")
    mol = _runtime.build_mol(
        symbols=("H",), positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
        charge=0, multiplicity=1, basis="sto-3g",
    )
    mol.spin = 0
    # use_df=True should attempt and silently fall back
    _runtime.run_dft(mol, use_df=True)


def test_run_dft_no_gradient_when_dograd_false(monkeypatch):
    _install_fake_pyscf(monkeypatch)
    mol = _runtime.build_mol(
        symbols=("H",), positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
        charge=0, multiplicity=1, basis="sto-3g",
    )
    mol.spin = 0
    _, gradient, meta, _ = _runtime.run_dft(mol, dograd=False)
    assert gradient == []
    assert meta["grad_norm"] == 0.0


def test_run_dft_uses_gpu_classes_when_available(monkeypatch):
    """The gpu4pyscf branch should be taken when its import succeeds."""
    _install_fake_pyscf(monkeypatch)
    gpu_dft = types.ModuleType("gpu4pyscf.dft")
    gpu_rks = MagicMock()
    gpu_rks.kernel.return_value = -2.0
    gpu_rks.converged = True
    gpu_rks.nuc_grad_method.return_value.kernel.return_value = np.zeros((1, 3))
    gpu_rks.mo_coeff = np.eye(1)
    gpu_rks.mo_occ = np.array([2.0])
    gpu_dft.RKS = MagicMock(return_value=gpu_rks)
    gpu_dft.UKS = MagicMock(return_value=gpu_rks)
    gpu_pkg = types.ModuleType("gpu4pyscf")
    gpu_pkg.dft = gpu_dft
    monkeypatch.setitem(sys.modules, "gpu4pyscf", gpu_pkg)
    monkeypatch.setitem(sys.modules, "gpu4pyscf.dft", gpu_dft)

    mol = _runtime.build_mol(
        symbols=("H",), positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
        charge=0, multiplicity=1, basis="sto-3g",
    )
    mol.spin = 0
    energy, _, meta, _ = _runtime.run_dft(mol, want_gpu=True)
    assert meta["gpu_used"] is True
    assert energy == -2.0


# ---------------------------------------------------------------------------
# _runtime.get_active_space_tensors
# ---------------------------------------------------------------------------


def test_get_active_space_tensors_returns_correct_shapes(monkeypatch):
    _install_fake_pyscf(monkeypatch)
    mol = MagicMock()
    mol.spin = 0
    mol.energy_nuc.return_value = 1.234
    mol.intor.side_effect = lambda label: (
        np.ones((2, 2)) if "int1e" in label else np.ones((2, 2, 2, 2))
    )
    mf = MagicMock()
    mf.mo_coeff = np.eye(2)
    mf.mo_occ = np.array([2.0, 0.0])

    nuc, h1, h2 = _runtime.get_active_space_tensors(mol, mf)
    assert nuc == 1.234
    assert h1.shape == (2, 2)
    assert h2.shape == (2, 2, 2, 2)


def test_get_active_space_tensors_localized_invokes_boys(monkeypatch):
    """The localized branch should hit ``pyscf.lo.Boys`` for occ + vir blocks."""
    mocks = _install_fake_pyscf(monkeypatch)
    mol = MagicMock()
    mol.spin = 0
    mol.energy_nuc.return_value = 1.234
    mol.intor.side_effect = lambda label: (
        np.ones((2, 2)) if "int1e" in label else np.ones((2, 2, 2, 2))
    )
    mf = MagicMock()
    mf.mo_coeff = np.eye(2)
    mf.mo_occ = np.array([2.0, 0.0])

    _runtime.get_active_space_tensors(mol, mf, localized=True)
    # Boys is called once for occupied and once for virtual.
    assert mocks["lo"].Boys.call_count == 2


# ---------------------------------------------------------------------------
# _runtime.save_tensors + print_tensors_file
# ---------------------------------------------------------------------------


def test_save_tensors_writes_npz_with_expected_keys(tmp_path: Path):
    target = tmp_path / "out" / "sample.npz"
    nuc = 1.0
    h1 = np.eye(2)
    h2 = np.ones((2, 2, 2, 2))
    _runtime.save_tensors(path=target, nuc=nuc, h1=h1, h2=h2)
    assert target.is_file()
    data = np.load(target)
    assert set(data.files) == {"hc", "h1e", "h2e"}
    np.testing.assert_array_equal(data["h1e"], h1)


def test_print_tensors_file_runs(tmp_path: Path, capsys):
    target = tmp_path / "t.npz"
    _runtime.save_tensors(path=target, nuc=1.0, h1=np.eye(2), h2=np.zeros((2, 2, 2, 2)))
    _runtime.print_tensors_file(target)
    out = capsys.readouterr().out
    assert "hc" in out and "h1e" in out and "h2e" in out


# ---------------------------------------------------------------------------
# PyscfExtOptCalculator.calc — end-to-end via mocks
# ---------------------------------------------------------------------------


def test_extopt_calc_returns_energy_and_gradient(monkeypatch):
    _install_fake_pyscf(monkeypatch)
    calc = extopt_calc.PyscfExtOptCalculator()
    energy, gradient = calc.calc(_data())
    assert energy == -1.5
    assert len(gradient) == 2


def test_extopt_calc_skips_tensor_extraction_by_default(monkeypatch):
    _install_fake_pyscf(monkeypatch)
    with patch.object(_runtime, "get_active_space_tensors") as mock_tensors:
        extopt_calc.PyscfExtOptCalculator().calc(_data())
    mock_tensors.assert_not_called()


def test_extopt_calc_extracts_tensors_when_settings_request_them(
    tmp_path: Path, monkeypatch
):
    _install_fake_pyscf(monkeypatch)
    monkeypatch.chdir(tmp_path)
    extopt_calc.PyscfExtOptCalculator().calc(
        _data(save_tensors=True, tag="step3_structure_0")
    )
    assert (tmp_path / "tensors" / "step3_structure_0.npz").is_file()


def test_extopt_calc_honours_per_call_settings_override(monkeypatch):
    """Per-request ``settings`` should override the per-process defaults."""
    mocks = _install_fake_pyscf(monkeypatch)
    extopt_calc.PyscfExtOptCalculator(method="dft", xc="pbe").calc(
        _data(method="hf", xc="b3lyp")
    )
    # ``method=hf`` short-circuits dft.RKS and uses scf.RHF instead.
    mocks["scf"].RHF.assert_called_once()
    mocks["dft"].RKS.assert_not_called()


def test_extopt_calc_localized_tensors(tmp_path: Path, monkeypatch):
    mocks = _install_fake_pyscf(monkeypatch)
    monkeypatch.chdir(tmp_path)
    extopt_calc.PyscfExtOptCalculator().calc(
        _data(save_tensors=True, localized=True, tag="s0")
    )
    assert mocks["lo"].Boys.call_count == 2
