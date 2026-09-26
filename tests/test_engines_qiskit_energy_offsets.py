"""Imported electronic constants survive transformations and every molecular reporting path."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chemrefine.engines.qiskit.bundles import read_bundle, write_bundle
from chemrefine.engines.qiskit.data import ElectronicStructureData, MolecularMetadata
from chemrefine.engines.qiskit.integral_io import load_integrals, save_integrals
from chemrefine.engines.qiskit.options import ActiveSpaceOptions
from chemrefine.engines.qiskit.problem import prepare_problem
from chemrefine.engines.qiskit.workflow import run_problem
from chemrefine.errors import OutputParseError

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit")


def _h2():
    """Use stored integrals and nuclear energy without executing a chemistry driver."""
    return ElectronicStructureData(
        **json.loads((Path(__file__).parent / "data/engines/qiskit/h2_integrals.json").read_text())
    )


@pytest.mark.parametrize("algorithm", ["exact", "vqd", "qeom", "sqd"])
@pytest.mark.parametrize("nuclear", [None, 0.7137539936876182])
def test_imported_constants_shift_canonical_and_all_root_energies_once(algorithm, nuclear):
    """Active operators/states remain unchanged while electronic and total roots each shift once."""
    data = replace(_h2(), nuclear_repulsion_energy=nuclear)
    controls: dict[str, Any] = {}
    if algorithm == "vqd":
        controls = {"k": 1, "measure_residuals": True}
    elif algorithm == "qeom":
        controls = {"target_root": 2, "measure_residuals": True}
    elif algorithm == "sqd":
        pytest.importorskip("qiskit_addon_sqd")
        controls = {
            "projection": "explicit",
            "counts": {"0101": 10, "0110": 10, "1001": 10, "1010": 10},
            "num_roots": 4,
            "target_root": 2,
            "spin_constraint": "report",
            "configuration_recovery": False,
            "num_batches": 1,
            "samples_per_batch": 4,
        }
    options = {"algorithm": {"name": algorithm, "options": controls}}
    if algorithm in {"vqd", "qeom"}:
        options["optimizer"] = {"name": "slsqp", "options": {"maxiter": 300, "ftol": 1e-12}}
    first = run_problem(prepare_problem(data), options=options)
    changed = replace(data, energy_offsets={"core": -2.0, "environment": 0.25})
    second = run_problem(prepare_problem(changed), options=options)
    assert first.electronic_energy_hartree is not None
    assert second.electronic_energy_hartree is not None
    assert second.energy_hartree == pytest.approx(first.energy_hartree - 1.75, abs=1e-10)
    assert second.electronic_energy_hartree == pytest.approx(
        first.electronic_energy_hartree - 1.75, abs=1e-10
    )
    assert second.as_dict()["energy_hartree"] == second.energy_hartree
    if nuclear is None:
        assert second.total_energy_hartree is None
        assert second.energy_hartree == second.electronic_energy_hartree
    else:
        assert second.total_energy_hartree == second.energy_hartree
        assert second.energy_hartree - second.electronic_energy_hartree == pytest.approx(nuclear)
    if algorithm != "exact":
        assert second.root_energies_hartree is not None
        assert first.root_energies_hartree is not None
        np.testing.assert_allclose(
            np.asarray(second.root_energies_hartree) - first.root_energies_hartree,
            -1.75,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            second.metadata["active_energies_hartree"],
            first.metadata["active_energies_hartree"],
            atol=1e-10,
        )
        assert second.energy_hartree == second.root_energies_hartree[second.target_root]
    for state, original in zip(second.states, first.states, strict=True):
        np.testing.assert_array_equal(state.amplitudes, original.amplitudes)


def test_freeze_core_and_active_reduction_cannot_overwrite_named_input_offsets():
    """An exactly diagonal six-electron model supplies independent known reduction constants."""
    data = ElectronicStructureData(
        3,
        3,
        5,
        np.diag([-5.0, -2.0, -0.8, 0.5, 1.0]),
        np.zeros((5,) * 4),
        molecular_metadata=MolecularMetadata(("C",), ((0, 0, 0),)),
        nuclear_repulsion_energy=0.7,
        energy_offsets={
            "FreezeCoreTransformer": -1.1,
            "ActiveSpaceTransformer": -0.7,
            "core": 0.05,
        },
    )
    prepared = prepare_problem(
        data,
        freeze_core=True,
        active_space=ActiveSpaceOptions(electrons=(1, 1), orbitals=3, active_orbitals=[2, 3, 4]),
    )
    assert prepared.energy_offsets == pytest.approx(
        {
            "nuclear_repulsion_energy": 0.7,
            "input:FreezeCoreTransformer": -1.1,
            "input:ActiveSpaceTransformer": -0.7,
            "input:core": 0.05,
            "FreezeCoreTransformer": -10,
            "ActiveSpaceTransformer": -4,
        }
    )
    assert prepared.active_orbitals == [2, 3, 4] and prepared.num_particles == (1, 1)
    result = run_problem(prepared, options={"algorithm": "exact"})
    assert result.energy_hartree == pytest.approx(-15.6 - 1.75 + 0.7)


@pytest.mark.parametrize("offset", [True, "1.25", None])
def test_integral_bundle_does_not_coerce_invalid_offset_types(tmp_path, offset):
    """Portable JSON follows the same strict numerical offset contract as the Python boundary."""
    source = save_integrals(tmp_path / "source.json", _h2())
    bundle = read_bundle(source)
    altered = write_bundle(
        tmp_path / "altered.json",
        kind="electronic_structure",
        arrays=bundle.arrays,
        metadata={**bundle.metadata, "energy_offsets": {"core": offset}},
    )
    with pytest.raises(OutputParseError, match="energy_offsets"):
        load_integrals(altered)


def test_legacy_integral_bundles_without_offset_metadata_keep_zero_shift(tmp_path):
    """Version-one bundles written before electronic offsets remain readable."""
    source = save_integrals(tmp_path / "source.json", _h2())
    bundle = read_bundle(source)
    metadata = dict(bundle.metadata)
    del metadata["energy_offsets"]
    legacy = write_bundle(
        tmp_path / "legacy.json",
        kind="electronic_structure",
        arrays=bundle.arrays,
        metadata=metadata,
    )
    assert load_integrals(legacy).energy_offsets == {}
