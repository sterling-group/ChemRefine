"""Tests for the in-process ``PyscfDirectEngine``.

PySCF is mocked at the import boundary (see
``tests/test_engines_pyscf_extopt_calc.py``); we exercise the engine
lifecycle (``prepare → submit → parse``), the runlog emission, and
the gradient/forces unit conversion.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines.base import get_engine
from chemrefine.engines.pyscf import direct
from chemrefine.engines.pyscf.options import PyscfOptions
from chemrefine.state import JobBatch, PipelineState, StepContext, StepResults, Structure


def _install_fake_pyscf(monkeypatch):
    """Insert mocks for the PySCF imports ``_runtime.run_dft`` performs."""
    rks = MagicMock()
    rks.kernel.return_value = -1.0
    rks.converged = True
    rks.nuc_grad_method.return_value.kernel.return_value = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]
    )
    rks.mo_coeff = np.eye(2)
    rks.mo_occ = np.array([2.0, 0.0])

    gto_mod = types.ModuleType("pyscf.gto")

    def _factory():
        m = MagicMock()
        m.spin = 0
        m.energy_nuc.return_value = 1.0
        m.intor.side_effect = lambda label: (
            np.ones((2, 2)) if "int1e" in label else np.ones((2, 2, 2, 2))
        )
        m.build.return_value = None
        return m

    gto_mod.Mole = _factory

    dft_mod = types.ModuleType("pyscf.dft")
    dft_mod.RKS = MagicMock(return_value=rks)
    dft_mod.UKS = MagicMock(return_value=rks)

    scf_mod = types.ModuleType("pyscf.scf")
    scf_mod.RHF = MagicMock(return_value=rks)
    scf_mod.UHF = MagicMock(return_value=rks)

    lib_mod = types.ModuleType("pyscf.lib")
    lib_mod.num_threads = MagicMock()

    pyscf_pkg = types.ModuleType("pyscf")
    pyscf_pkg.gto = gto_mod
    pyscf_pkg.dft = dft_mod
    pyscf_pkg.scf = scf_mod
    pyscf_pkg.lib = lib_mod

    monkeypatch.setitem(sys.modules, "pyscf", pyscf_pkg)
    monkeypatch.setitem(sys.modules, "pyscf.gto", gto_mod)
    monkeypatch.setitem(sys.modules, "pyscf.dft", dft_mod)
    monkeypatch.setitem(sys.modules, "pyscf.scf", scf_mod)
    monkeypatch.setitem(sys.modules, "pyscf.lib", lib_mod)


def _ctx(tmp_path: Path, structures: tuple[Structure, ...]) -> StepContext:
    step_cfg = StepConfig(
        step=1,
        name="screen",
        engine="pyscf-direct",
        operation="opt_sp",
        options={"method": "dft", "xc": "pbe", "basis": "sto-3g"},
    )
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1_screen",
        template_dir=tmp_path,
        scratch_dir=None,
        prev_state=PipelineState(structures=structures),
        charge=0,
        multiplicity=1,
        max_cores=1,
        slurm_template="cpu.slurm.header",
        orca_executable="orca",
    )


def _seed(sid: str = "0") -> Structure:
    return Structure(
        id=sid,
        atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_pyscf_direct_engine_is_registered():
    engine = get_engine("pyscf-direct")
    assert engine.name == "pyscf-direct"
    assert engine.supports_nms is False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_prepare_writes_per_structure_json(tmp_path: Path):
    ctx = _ctx(tmp_path, structures=(_seed("0"), _seed("1")))
    engine = get_engine("pyscf-direct")
    inputs = engine.prepare(ctx)
    assert len(inputs.files) == 2
    for inp, _out, sid in inputs.files:
        data = json.loads(inp.read_text())
        assert data["id"] == sid
        assert data["engine"] == "pyscf-direct"


def test_submit_writes_runlog_per_structure(tmp_path: Path, monkeypatch):
    _install_fake_pyscf(monkeypatch)
    ctx = _ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("pyscf-direct")
    inputs = engine.prepare(ctx)
    engine.submit(inputs, ctx)
    runlog = ctx.step_dir / "step1_structure_0.runlog"
    text = runlog.read_text()
    assert "ChemRefine pyscf-direct step1_screen starting" in text
    assert "ChemRefine pyscf-direct step1_screen finished" in text
    assert "exit_code=0" in text


def test_submit_records_failure_in_runlog(tmp_path: Path, monkeypatch):
    _install_fake_pyscf(monkeypatch)
    ctx = _ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("pyscf-direct")
    inputs = engine.prepare(ctx)
    with (
        patch.object(direct, "_score_one", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        engine.submit(inputs, ctx)
    runlog = ctx.step_dir / "step1_structure_0.runlog"
    text = runlog.read_text()
    assert "exit_code=1" in text


def test_parse_returns_structures_with_energy_and_forces(tmp_path: Path, monkeypatch):
    _install_fake_pyscf(monkeypatch)
    ctx = _ctx(tmp_path, structures=(_seed("0"),))
    engine = get_engine("pyscf-direct")
    inputs = engine.prepare(ctx)
    engine.submit(inputs, ctx)
    results = engine.parse(inputs, ctx)
    assert len(results.structures) == 1
    s = results.structures[0]
    assert s.id == "0"
    assert s.energy_hartree == -1.0
    assert s.forces_ev_per_a is not None
    assert s.forces_ev_per_a.shape == (2, 3)


def test_wait_is_noop():
    """``wait`` is a no-op once :meth:`submit` runs inline."""
    engine = get_engine("pyscf-direct")
    engine.wait(JobBatch(jobs={}))


def test_normal_mode_sample_not_supported(tmp_path: Path):
    engine = get_engine("pyscf-direct")
    with pytest.raises(NotImplementedError, match="does not support NMS"):
        engine.normal_mode_sample(StepResults(structures=()), _ctx(tmp_path, ()))


def test_pyscf_direct_find_structure_raises_on_unknown_sid(tmp_path: Path):
    """``_find_structure`` raises ``KeyError`` if the SID isn't among seeds."""
    from chemrefine.engines.pyscf.direct import PyscfDirectEngine

    ctx = _ctx(tmp_path, structures=(_seed("0"),))
    engine = PyscfDirectEngine()
    with pytest.raises(KeyError, match="ghost"):
        engine._find_structure(ctx, "ghost")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_score_one_routes_through_runtime(monkeypatch):
    """``_score_one`` should call ``_runtime.run_dft`` exactly once."""
    _install_fake_pyscf(monkeypatch)
    options = PyscfOptions(method="dft", xc="pbe", basis="sto-3g")
    energy, gradient = direct._score_one(
        struct=_seed(),
        options=options,
        charge=0,
        multiplicity=1,
    )
    assert energy == -1.0
    assert len(gradient) == 2


def test_gradient_to_forces_converts_units():
    """Gradient (Hartree/Bohr) flipped to force (eV/A) via HARTREE_PER_BOHR_TO_EV_PER_A."""
    from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A

    gradient = [[1.0, 0.0, 0.0]]
    forces = direct._gradient_to_forces(gradient)
    assert forces is not None
    np.testing.assert_allclose(forces[0], [-HARTREE_PER_BOHR_TO_EV_PER_A, 0.0, 0.0])


def test_gradient_to_forces_handles_none():
    assert direct._gradient_to_forces(None) is None


def test_gradient_to_forces_handles_empty_list():
    assert direct._gradient_to_forces([]) is None
