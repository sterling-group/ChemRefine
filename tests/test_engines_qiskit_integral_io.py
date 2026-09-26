"""Portable typed integral bundles preserve complex and unrestricted scientific data."""

from dataclasses import fields
from typing import Any

import numpy as np
import pytest

from chemrefine.engines.qiskit.bundles import read_bundle, write_bundle
from chemrefine.engines.qiskit.data import ElectronicStructureData, MolecularMetadata
from chemrefine.engines.qiskit.integral_io import (
    INTEGRAL_ARRAY_FIELDS,
    load_integrals,
    save_integrals,
)
from chemrefine.errors import ConfigError, OutputParseError


def _data(unrestricted=False):
    """Include every optional owned field in one complex spin-resolved input."""
    one = np.array([[-1, 0.2j], [-0.2j, 0.3]])
    two = np.zeros((2,) * 4)
    two[0, 0, 0, 0] = 0.7
    kwargs: dict[str, Any] = {}
    if unrestricted:
        kwargs.update(
            one_body_integrals_beta=one * 0.8,
            two_body_integrals_beta_beta=two * 0.9,
            two_body_integrals_beta_alpha=two * 0.95,
            overlap_alpha_beta=np.eye(2),
        )
    return ElectronicStructureData(
        1,
        0,
        2,
        one,
        two,
        nuclear_repulsion_energy=0.7,
        orbital_energies=[-1, 0.3],
        orbital_energies_beta=[-0.8, 0.24],
        orbital_occupations=[0, 1],
        orbital_occupations_beta=[0, 0],
        molecular_metadata=MolecularMetadata(("H",), ((0, 0, 0),)),
        provenance={"source": "independent example"},
        metadata={"label": "complex doublet"},
        energy_offsets={"inactive_core": -1.2},
        **kwargs,
    )


@pytest.mark.parametrize("unrestricted", [False, True])
def test_integral_bundles_round_trip_every_owned_field_and_relocation(tmp_path, unrestricted):
    data = _data(unrestricted)
    directory = tmp_path / "original"
    directory.mkdir()
    path = save_integrals(directory / "integrals.json", data)
    relocated = directory.rename(tmp_path / "relocated")
    restored = load_integrals(relocated / path.name)
    for field in fields(ElectronicStructureData):
        expected, actual = getattr(data, field.name), getattr(restored, field.name)
        if field.name in INTEGRAL_ARRAY_FIELDS and expected is not None:
            np.testing.assert_array_equal(actual, expected)
            assert not actual.flags.writeable
        else:
            assert actual == expected
    bundle = read_bundle(relocated / path.name)
    assert bundle.description.kind == "electronic_structure"
    assert bundle.metadata["units"] == "hartree"
    assert bundle.metadata["orbital_order"] == "alpha_then_beta"


def test_integral_bundle_writer_respects_size_budget(tmp_path):
    with pytest.raises(ConfigError, match="max_bytes"):
        save_integrals(tmp_path / "small.json", _data(), max_bytes=1)


@pytest.mark.parametrize(
    "patch",
    [
        {"kind": "states"},
        {"version": 2},
        {"units": "eV"},
        {"arrays": {}},
        {"arrays": {"one_body_integrals": "a", "two_body_integrals": "a"}},
        {"arrays": {"one_body_integrals": "a", "two_body_integrals": "b", "unknown": "c"}},
        {"arrays": {"one_body_integrals": "missing", "two_body_integrals": "other"}},
        {"num_alpha": 7},
        {"molecular_metadata": {"unexpected": 1}},
    ],
)
def test_integral_bundle_rejects_unsupported_or_inconsistent_physical_metadata(tmp_path, patch):
    source = save_integrals(tmp_path / "source.json", _data())
    bundle = read_bundle(source)
    metadata = {**bundle.metadata, **{key: value for key, value in patch.items() if key != "kind"}}
    altered = write_bundle(
        tmp_path / "altered.json",
        kind=patch.get("kind", "electronic_structure"),
        arrays=bundle.arrays,
        metadata=metadata,
    )
    with pytest.raises(OutputParseError, match="invalid electronic structure"):
        load_integrals(altered)


def test_integral_bundle_revalidates_numerical_symmetry(tmp_path):
    source = save_integrals(tmp_path / "source.json", _data())
    bundle = read_bundle(source)
    arrays = dict(bundle.arrays)
    arrays["one_body_integrals"] = np.array([[0, 1j], [1j, 0]])
    altered = write_bundle(
        tmp_path / "bad.json", kind="electronic_structure", arrays=arrays, metadata=bundle.metadata
    )
    with pytest.raises(OutputParseError, match="Hermitian"):
        load_integrals(altered)
