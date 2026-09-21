"""Independent mapper selection without requiring the optional Qiskit packages."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from chemrefine.engines.qiskit.mapping import map_problem
from chemrefine.engines.qiskit.options import ComponentSelection, QiskitOptions
from chemrefine.engines.qiskit.problem import PreparedProblem
from chemrefine.engines.qiskit.registry import MAPPERS, validate_component_graph


def test_bravyi_kitaev_is_a_lazy_replaceable_mapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new mapping uses the same registry as JW and parity, and no driver."""
    module = ModuleType("qiskit_nature.second_q.mappers")
    expected = object()
    monkeypatch.setattr(module, "BravyiKitaevMapper", lambda: expected, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    options = QiskitOptions(mapper="bravyi-kitaev")
    validate_component_graph(options)
    assert options.mapper.name == "bravyi_kitaev"
    assert MAPPERS.build(ComponentSelection.named("bravyi_kitaev"), problem=None) is expected


@pytest.mark.parametrize(
    "mapper", ["parity", ComponentSelection(name="parity", options={"two_qubit_reduction": False})]
)
def test_mapping_uses_transformed_particles_and_records_original_orbitals_and_offsets(
    monkeypatch, mapper
):
    """Keep problem transformations independent from mapped-register reduction."""
    original_operator = object()
    constants = {"nuclear_repulsion_energy": 0.7, "FreezeCoreTransformer": -3.0}
    native = SimpleNamespace(
        num_spatial_orbitals=3,
        num_particles=(2, 1),
        hamiltonian=SimpleNamespace(constants=constants),
    )
    transformations = [{"kind": "freeze_core", "frozen_orbitals": [0]}]
    prepared = PreparedProblem(
        native,
        original_operator,
        6,
        [1, 3, 4],
        2,
        {"source": "integrals"},
        {"transformations": transformations},
    )
    calls = []
    reduced = mapper == "parity"

    class Hamiltonian:
        """Expose a stable Pauli-term count after simplification."""

        num_qubits = 4 if reduced else 6

        def simplify(self):
            calls.append("simplify")
            return self

        def __len__(self):
            return 8

    qubit_operator = Hamiltonian()

    class Mapper:
        """Capture the exact fermionic operator passed across the mapping boundary."""

        def map(self, operator):
            assert operator is original_operator
            return qubit_operator

    implementation = Mapper()

    def build(selection, *, problem):
        assert problem is native
        assert problem.num_particles == (2, 1)
        assert selection.name == "parity"
        return implementation

    monkeypatch.setattr(MAPPERS, "build", build)
    context = map_problem(prepared, mapper)
    assert calls == ["simplify"]
    assert context.mapper is implementation
    assert context.problem is native
    assert context.fermionic_hamiltonian is original_operator
    assert context.qubit_hamiltonian is qubit_operator
    assert context.num_qubits_before_reduction == 6
    assert context.num_qubits == (4 if reduced else 6)
    assert context.num_spatial_orbitals == 3
    assert context.num_particles == (2, 1)
    assert context.num_pauli_terms == 8
    assert context.multiplicity == 2
    assert context.mapping_metadata["options"] == {"two_qubit_reduction": reduced}
    assert context.mapping_metadata["symmetry_tapering"] is None
    assert context.active_space_metadata == {
        "original_num_spatial_orbitals": 6,
        "active_orbitals": [1, 3, 4],
        "num_particles": [2, 1],
        "energy_offsets": constants,
        "transformations": transformations,
    }
    assert context.provenance == {
        "source": "integrals",
        "problem_metadata": {"transformations": transformations},
    }
    prepared.active_orbitals.append(5)
    assert context.active_space_metadata["active_orbitals"] == [1, 3, 4]
