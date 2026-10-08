"""Portable MO inputs bind to pipeline geometry and participate in file identity."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from chemrefine.config import StepConfig
from chemrefine.engines.qiskit.data import ElectronicStructureData, MolecularMetadata
from chemrefine.engines.qiskit.engine import QiskitEngine
from chemrefine.engines.qiskit.integral_io import save_integrals
from chemrefine.engines.qiskit.integral_source import prepare_integral_job
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.workflow import run_job
from chemrefine.errors import ConfigError, OutputParseError
from chemrefine.input_files import declared_input_files, resolve_input_file_options


@pytest.fixture
def source(tmp_path):
    """Associate independently stored H2 integrals with their exact source geometry."""
    data = ElectronicStructureData(
        **json.loads((Path(__file__).parent / "data/engines/qiskit/h2_integrals.json").read_text())
    )
    data = replace(
        data,
        molecular_metadata=MolecularMetadata(("H", "H"), ((0, 0, 0), (0, 0, 0.735))),
    )
    path = save_integrals(tmp_path / "integrals.json", data)
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nH2\nH 0 0 0\nH 0 0 0.735\n")
    return data, path, xyz


def test_pipeline_integrals_bypass_driver_and_keep_total_energy(source, monkeypatch):
    """A declared MO source reaches the public solver without a hidden SCF rerun."""
    from chemrefine.engines.qiskit import workflow

    def forbidden(*args, **kwargs):
        """Supplying an integral bundle must bypass geometry-driven preparation."""
        pytest.fail("unexpected PySCF call")

    monkeypatch.setattr(workflow, "prepare_pyscf_problem", forbidden)
    _, path, xyz = source
    result = run_job(
        xyz,
        charge=0,
        multiplicity=1,
        options={"integral_source": {"bundle_path": str(path)}, "algorithm": "exact"},
    )
    assert result.energy_hartree == pytest.approx(-1.1373060357534, abs=1e-10)
    assert result.total_energy_hartree == result.energy_hartree
    assert result.metadata["provenance"]["source"] == "stored_pyscf_integrals"


@pytest.mark.parametrize(
    "change", ["geometry", "symbols", "charge", "spin", "metadata", "nuclear", "frames"]
)
def test_integral_pipeline_refuses_mismatched_or_incomplete_identity(source, change):
    """A scalar energy cannot be attributed to a different molecule or energy convention."""
    data, path, xyz = source
    if change == "geometry":
        xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.8\n")
    elif change == "symbols":
        xyz.write_text("2\n\nH 0 0 0\nHe 0 0 0.735\n")
    elif change == "metadata":
        save_integrals(path, replace(data, molecular_metadata=None))
    elif change == "nuclear":
        save_integrals(path, replace(data, nuclear_repulsion_energy=None))
    elif change == "frames":
        xyz.write_text(xyz.read_text() * 2)
    options = QiskitOptions.from_raw({"integral_source": {"bundle_path": str(path)}})
    with pytest.raises(ConfigError, match=r"differs|requires"):
        prepare_integral_job(
            xyz,
            charge=1 if change == "charge" else 0,
            multiplicity=3 if change == "spin" else 1,
            options=options,
        )


def test_missing_source_and_declared_size_limit(source):
    """Explicit input controls are validated before expensive quantum preparation."""
    _, path, xyz = source
    with pytest.raises(ConfigError, match="requires integral_source"):
        prepare_integral_job(xyz, charge=0, multiplicity=1, options=QiskitOptions())
    options = QiskitOptions.from_raw(
        {"integral_source": {"bundle_path": str(path), "max_input_bytes": 1}}
    )
    with pytest.raises(OutputParseError, match="max_bytes"):
        prepare_integral_job(xyz, charge=0, multiplicity=1, options=options)


def test_integral_source_declares_descriptor_and_payload(source):
    """Generic path resolution discovers both files without engine/cache coupling."""
    _, path, _ = source
    engine = QiskitEngine()
    step = StepConfig(
        step=1, engine="qiskit", options={"integral_source": {"bundle_path": str(path)}}
    )
    resolved = resolve_input_file_options(step, engine)
    files = declared_input_files(resolved, engine)
    assert set(files.values()) == {path, next(path.parent.glob("*.npz"))}
    assert engine.input_file_options({}) == ()
    assert engine.input_file_dependencies({}, {"/unrelated": path}) == {}
    assert engine.input_file_dependencies(step.options, {"/unrelated": path}) == {}


@pytest.mark.parametrize(
    "offsets",
    [
        None,
        {"": 1},
        {3: 1},
        {"nuclear_repulsion_energy": 1},
        {"core": True},
        {"core": 1j},
        {"core": float("inf")},
    ],
)
def test_invalid_input_energy_offsets_are_refused(source, offsets):
    """Nuclear constants remain separate and every electronic offset is finite and named."""
    with pytest.raises(ConfigError, match="energy_offsets"):
        replace(source[0], energy_offsets=offsets)


def test_input_offsets_survive_bundle_and_enter_each_energy_once(source):
    """Inactive constants use a separate namespace from later Nature transformations."""
    data, path, xyz = source
    offsets = {"ActiveSpaceTransformer": -2.0, "user_core": 0.25}
    save_integrals(path, replace(data, energy_offsets=offsets))
    offsets["user_core"] = 900
    result = run_job(
        xyz,
        charge=0,
        multiplicity=1,
        options={"integral_source": {"bundle_path": str(path)}, "algorithm": "exact"},
    )
    assert result.energy_hartree == pytest.approx(-1.1373060357534 - 1.75)
    assert result.electronic_energy_hartree == pytest.approx(
        result.energy_hartree - data.nuclear_repulsion_energy
    )
