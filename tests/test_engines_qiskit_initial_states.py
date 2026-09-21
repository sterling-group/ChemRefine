"""Reference determinants respect supplied orbital order under each mapper."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from chemrefine.engines.qiskit.components.initial_states import build_hartree_fock
from chemrefine.engines.qiskit.context import ElectronicStructureContext
from chemrefine.engines.qiskit.registry import NoComponentOptions
from chemrefine.errors import ConfigError


def _context(
    alpha: Any = (0, 1),
    beta: Any = (1, 0),
    *,
    mapper: Any = None,
    num_qubits: int = 4,
) -> ElectronicStructureContext:
    """Build the smallest context with explicitly ordered orbital populations."""
    return ElectronicStructureContext(
        problem=SimpleNamespace(orbital_occupations=alpha, orbital_occupations_b=beta),
        mapper=mapper,
        qubit_hamiltonian=None,
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        num_qubits=num_qubits,
        multiplicity=1,
    )


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attributes: Any) -> None:
    """Provide one lazy optional import without requiring the Qiskit extra."""
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("occupations", [None, [1, 0]])
def test_default_and_prefix_occupations_preserve_nature_factory(
    monkeypatch: pytest.MonkeyPatch, occupations: Any
) -> None:
    """Ordinary references keep the existing lazy HartreeFock constructor."""
    calls: list[tuple[Any, ...]] = []
    sentinel = object()

    def hartree_fock(*args: Any) -> object:
        calls.append(args)
        return sentinel

    _install_module(monkeypatch, "qiskit_nature.second_q.circuit.library", HartreeFock=hartree_fock)
    mapper = object()
    result = build_hartree_fock(
        options=NoComponentOptions(),
        context=_context(occupations, occupations, mapper=mapper),
    )
    assert result is sentinel
    assert calls == [(2, (1, 1), mapper)]


@pytest.mark.parametrize(
    ("alpha", "beta", "message"),
    [
        ([0, 1], None, "binary alpha and beta"),
        ([0, 1, 0], [1, 0], "spatial-orbital count"),
        ([0.5, 0.5], [1, 0], "binary alpha and beta"),
        ([float("nan"), 1], [1, 0], "binary alpha and beta"),
        ([0, 1], [1, 1], "particle count"),
    ],
)
def test_invalid_reference_occupations_fail_before_optional_imports(
    alpha: Any, beta: Any, message: str
) -> None:
    """Inconsistent determinant metadata fails before it can select a wrong state."""
    with pytest.raises(ConfigError, match=message):
        build_hartree_fock(options=NoComponentOptions(), context=_context(alpha, beta))


def _install_explicit_reference_fakes(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record the creation product and X gates without loading Qiskit."""
    products: list[Any] = []

    def fermionic_op(data: dict[str, float], *, num_spin_orbitals: int) -> object:
        product = SimpleNamespace(data=data, num_spin_orbitals=num_spin_orbitals)
        products.append(product)
        return product

    class Circuit:
        """Record computational-basis flips on the chosen register."""

        def __init__(self, num_qubits: int) -> None:
            self.num_qubits = num_qubits
            self.flips: list[int] = []

        def x(self, index: int) -> None:
            self.flips.append(index)

    _install_module(monkeypatch, "qiskit", QuantumCircuit=Circuit)
    _install_module(monkeypatch, "qiskit_nature.second_q.operators", FermionicOp=fermionic_op)
    return products


def test_nonprefix_occupations_encode_the_supplied_creation_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Occupied alpha/beta indices retain their supplied orbital ordering."""
    products = _install_explicit_reference_fakes(monkeypatch)
    mapped = SimpleNamespace(num_qubits=4, paulis=SimpleNamespace(x=[[0, 1, 1, 0]] * 4))
    mapped_products: list[Any] = []

    def map_product(product: Any) -> object:
        mapped_products.append(product)
        return mapped

    circuit: Any = build_hartree_fock(
        options=NoComponentOptions(), context=_context(mapper=SimpleNamespace(map=map_product))
    )
    assert products == mapped_products
    assert products[0].data == {"+_1 +_2": 1.0}
    assert products[0].num_spin_orbitals == 4
    assert circuit.num_qubits == 4
    assert circuit.flips == [1, 2]


@pytest.mark.parametrize(
    ("mapped", "message"),
    [
        (None, "incompatible register"),
        (SimpleNamespace(num_qubits=3), "incompatible register"),
        (
            SimpleNamespace(num_qubits=4, paulis=SimpleNamespace(x=[])),
            "one basis state",
        ),
        (
            SimpleNamespace(num_qubits=4, paulis=SimpleNamespace(x=[[0, 1, 0, 0], [1, 0, 0, 0]])),
            "one basis state",
        ),
    ],
)
def test_incompatible_mapped_determinants_fail_descriptively(
    monkeypatch: pytest.MonkeyPatch, mapped: Any, message: str
) -> None:
    """Unsupported mapper encodings do not silently produce a wrong bitstring."""
    _install_explicit_reference_fakes(monkeypatch)
    mapper = SimpleNamespace(map=lambda _: mapped)
    with pytest.raises(ConfigError, match=message):
        build_hartree_fock(options=NoComponentOptions(), context=_context(mapper=mapper))


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity", "reduced_parity", "bravyi_kitaev"])
def test_real_mappers_preserve_reordered_reference_orbital_occupations(mapping: str) -> None:
    """Mapped number operators identify the exact supplied determinant in all bases."""
    mappers = pytest.importorskip("qiskit_nature.second_q.mappers")
    operators = pytest.importorskip("qiskit_nature.second_q.operators")
    quantum_info = pytest.importorskip("qiskit.quantum_info")
    mapper = {
        "jordan_wigner": mappers.JordanWignerMapper(),
        "parity": mappers.ParityMapper(),
        "reduced_parity": mappers.ParityMapper(num_particles=(1, 1)),
        "bravyi_kitaev": mappers.BravyiKitaevMapper(),
    }[mapping]
    width = 2 if mapping == "reduced_parity" else 4
    circuit = build_hartree_fock(
        options=NoComponentOptions(), context=_context(mapper=mapper, num_qubits=width)
    )
    state = quantum_info.Statevector.from_instruction(circuit)
    observed = []
    for orbital in range(4):
        number = operators.FermionicOp({f"+_{orbital} -_{orbital}": 1.0}, num_spin_orbitals=4)
        observed.append(float(state.expectation_value(mapper.map(number)).real))
    np.testing.assert_allclose(observed, [0, 1, 1, 0], atol=1e-12)
    assert sum(observed) == pytest.approx(2.0)
    assert circuit.num_qubits == width
