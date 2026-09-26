"""Reference-aware Pauli symmetry reduction with consistent state/operator transforms.

Automatic discovery finds the full Pauli subgroup shared by a Clifford reference
and the Hamiltonian. General references can supply explicit commuting generators;
their eigenvalues are verified by a bounded statevector calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from chemrefine.engines.qiskit.encodings import (
    binary_nullspace,
    binary_rref,
    independent_paulis,
    pauli_binary_vector,
)
from chemrefine.errors import ConfigError


class Z2TaperingOptions(BaseModel):
    """A base encoding, reference-compatible sector, and explicit reduction budgets."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    base_mapper: Literal["jordan_wigner", "bravyi_kitaev", "parity"] = "jordan_wigner"
    generators: tuple[str, ...] | None = Field(None, min_length=1)
    sectors: tuple[Literal[-1, 1], ...] | None = Field(None, min_length=1)
    min_qubits: StrictInt = Field(1, ge=1)
    max_symmetries: StrictInt = Field(32, ge=1)
    max_qubits: StrictInt = Field(128, ge=1)
    max_statevector_bytes: StrictInt = Field(268435456, ge=1)
    tolerance: float = Field(1e-10, gt=0, le=1e-6)

    @model_validator(mode="after")
    def _explicit_sector(self) -> Self:
        """Explicit eigenvalues require a parallel explicit generator ordering."""
        if self.sectors is not None and (
            self.generators is None or len(self.sectors) != len(self.generators)
        ):
            raise ValueError("sectors require one eigenvalue per explicit symmetry generator")
        if self.min_qubits > self.max_qubits:
            raise ValueError("min_qubits must not exceed max_qubits")
        return self


def _state_budget(width: int, maximum: int) -> None:
    """Bound four complex vectors before allocating any amplitudes."""
    if 64 * (1 << width) > maximum:
        raise ConfigError("tapering state transformation exceeds max_statevector_bytes")


def _reference_clifford(reference: Any, width: int) -> Any | None:
    """Recognize deterministic Clifford circuits without allocating a statevector."""
    from qiskit import QuantumCircuit
    from qiskit.exceptions import QiskitError
    from qiskit.quantum_info import Clifford

    if not isinstance(reference, QuantumCircuit) or reference.num_qubits != width:
        raise ConfigError("tapering reference has an incompatible qubit width")
    if getattr(reference, "num_parameters", 0):
        raise ConfigError("tapering reference must have fixed parameters")
    if getattr(reference, "num_clbits", 0):
        raise ConfigError("tapering reference cannot contain classical bits or measurements")
    if "reset" in reference.count_ops():
        raise ConfigError("tapering reference cannot contain stochastic reset operations")
    try:
        return Clifford(reference)
    except (QiskitError, TypeError, ValueError):
        return None


@dataclass(frozen=True)
class TaperingTransform:
    """One verified Clifford coordinate system and its fixed +1 Z sector.

    The first ``num_tapered`` qubits after ``clifford`` are frozen in zero.
    Metadata retains the original signed generator conventions and eigenvalues.
    Noncommuting observables have a well-defined projected block; evolution
    generators instead require commutation to avoid silently changing dynamics.
    """

    num_qubits: int
    num_tapered: int
    clifford: Any
    metadata: dict[str, Any]
    max_statevector_bytes: int
    tolerance: float

    @property
    def num_reduced_qubits(self) -> int:
        """Return the retained register width."""
        return self.num_qubits - self.num_tapered

    def transform_operator(self, operator: Any) -> Any:
        """Conjugate an encoded operator into the shared symmetry coordinate system."""
        from qiskit.quantum_info import SparsePauliOp

        if not isinstance(operator, SparsePauliOp) or operator.num_qubits != self.num_qubits:
            raise ConfigError("tapering operator has an incompatible qubit width or type")
        if not np.isfinite(operator.coeffs).all():
            raise ConfigError("tapering operator coefficients must be finite")
        return SparsePauliOp(
            operator.paulis.evolve(self.clifford, frame="s"), coeffs=operator.coeffs
        ).simplify(atol=0, rtol=0)

    def project_transformed(self, operator: Any, *, check_commutes: bool = True) -> Any | None:
        """Remove fixed qubits, optionally refusing any term that changes the sector."""
        from qiskit.quantum_info import PauliList, SparsePauliOp

        if not isinstance(operator, SparsePauliOp) or operator.num_qubits != self.num_qubits:
            raise ConfigError("transformed tapering operator has the wrong width")
        if not np.isfinite(operator.coeffs).all():
            raise ConfigError("transformed tapering operator coefficients must be finite")
        simplified = operator.simplify(atol=0, rtol=0)
        changing = np.any(simplified.paulis.x[:, : self.num_tapered], axis=1)
        changing &= simplified.coeffs != 0
        if check_commutes and np.any(changing):
            return None
        retained = ~np.any(simplified.paulis.x[:, : self.num_tapered], axis=1)
        if not np.any(retained):
            return SparsePauliOp("I" * self.num_reduced_qubits, coeffs=[0])
        # PauliList.from_symplectic encodes canonical Hermitian Paulis. Simplify
        # has moved all explicit phases into coefficients before removing bits.
        paulis = PauliList.from_symplectic(
            simplified.paulis.z[retained, self.num_tapered :],
            simplified.paulis.x[retained, self.num_tapered :],
        )
        return SparsePauliOp(paulis, coeffs=simplified.coeffs[retained]).simplify(atol=0, rtol=0)

    def map_operator(self, operator: Any, *, check_commutes: bool = True) -> Any | None:
        """Transform and reduce a base-mapped operator, preserving the requested semantics."""
        return self.project_transformed(
            self.transform_operator(operator), check_commutes=check_commutes
        )

    def taper_statevector(self, state: Any) -> np.ndarray:
        """Reduce a state only after verifying its entire support lies in this sector."""
        from qiskit.quantum_info import Statevector

        _state_budget(self.num_qubits, self.max_statevector_bytes)
        state = Statevector(state)
        if state.num_qubits != self.num_qubits or not state.is_valid(atol=self.tolerance, rtol=0):
            raise ConfigError("tapering state must be normalized and have the full width")
        transformed = state.evolve(self.clifford.to_circuit()).data
        reduced = np.array(transformed[:: 1 << self.num_tapered], dtype=complex, copy=True)
        if not np.isclose(np.vdot(reduced, reduced), 1, atol=self.tolerance, rtol=0):
            raise ConfigError("reference state does not occupy the selected tapering sector")
        reduced.setflags(write=False)
        return reduced

    def lift_statevector(self, state: Any) -> np.ndarray:
        """Restore a reduced state to the original base-mapper register."""
        from qiskit.quantum_info import Statevector

        _state_budget(self.num_qubits, self.max_statevector_bytes)
        state = Statevector(state)
        if state.num_qubits != self.num_reduced_qubits or not state.is_valid(
            atol=self.tolerance, rtol=0
        ):
            raise ConfigError(
                "reduced tapering state must be normalized and have the retained width"
            )
        full = np.zeros(1 << self.num_qubits, dtype=complex)
        full[:: 1 << self.num_tapered] = state.data
        result = np.array(
            Statevector(full).evolve(self.clifford.to_circuit().inverse()).data, copy=True
        )
        result.setflags(write=False)
        return result

    def prepare_reference(self, reference: Any) -> Any:
        """Reduce a Clifford reference polynomially, or a general state within its byte budget."""
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import StatePreparation
        from qiskit.quantum_info import Pauli, SparsePauliOp, StabilizerState, Statevector
        from qiskit.synthesis import synth_circuit_from_stabilizers

        reference_clifford = _reference_clifford(reference, self.num_qubits)
        if reference_clifford is None:
            _state_budget(self.num_qubits, self.max_statevector_bytes)
            state = Statevector.from_instruction(reference)
            circuit = QuantumCircuit(self.num_reduced_qubits)
            circuit.append(
                StatePreparation(self.taper_statevector(state)), range(circuit.num_qubits)
            )
            return circuit
        full = reference_clifford.compose(self.clifford)
        state = StabilizerState(full)
        for qubit in range(self.num_tapered):
            label = "I" * (self.num_qubits - qubit - 1) + "Z" + "I" * qubit
            if not np.isclose(
                state.expectation_value(Pauli(label)), 1, atol=self.tolerance, rtol=0
            ):
                raise ConfigError("reference state does not occupy the selected tapering sector")
        generators = [
            self.project_transformed(SparsePauliOp(pauli), check_commutes=False)
            for pauli in full.to_labels(mode="S")
        ]
        constraints = independent_paulis(generators, self.num_reduced_qubits)
        labels = [
            ("-" if operator.coeffs[0].real < 0 else "+") + operator.paulis[0].to_label()
            for operator in constraints
        ]
        return synth_circuit_from_stabilizers(labels)


def build_tapering_transform(
    hamiltonian: Any, reference: Any, *, options: Z2TaperingOptions | None = None
) -> TaperingTransform:
    """Discover or validate a reference-compatible commuting Pauli symmetry subgroup."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford, Pauli, SparsePauliOp, StabilizerState, Statevector
    from qiskit.synthesis import synth_circuit_from_stabilizers

    options = options or Z2TaperingOptions()
    width = hamiltonian.num_qubits
    if not options.min_qubits <= width <= options.max_qubits:
        raise ConfigError("tapering register lies outside min_qubits/max_qubits")
    if not np.isfinite(hamiltonian.coeffs).all() or np.any(
        np.abs(hamiltonian.coeffs.imag) > options.tolerance
    ):
        raise ConfigError("tapering Hamiltonian must be finite and Hermitian")
    hamiltonian = hamiltonian.simplify(atol=0, rtol=0)
    reference_clifford = _reference_clifford(reference, width)
    allowed = min(options.max_symmetries, width - options.min_qubits)
    generators = []
    if options.generators is None:
        if reference_clifford is None:
            raise ConfigError(
                "automatic tapering requires a Clifford reference; supply explicit generators"
            )
        # In reference coordinates the state is |0>. Its Pauli stabilizer group
        # is precisely the diagonal Z group. A kernel of Hamiltonian X supports
        # gives the complete compatible symmetry subgroup without enumerating it.
        transformed = SparsePauliOp(
            hamiltonian.paulis.evolve(reference_clifford, frame="h"), coeffs=hamiltonian.coeffs
        ).simplify(atol=0, rtol=0)
        nonzero = transformed.coeffs != 0
        basis = binary_nullspace(transformed.paulis.x[nonzero].astype(np.uint8))
        candidates = len(basis)
        for vector in basis[:allowed]:
            pauli = Pauli((vector.astype(bool), np.zeros(width, dtype=bool)))
            mapped = SparsePauliOp(pauli.evolve(reference_clifford, frame="s"))
            # Keep automatic generator labels unsigned and record the actual
            # reference eigenvalue separately, including reordered occupations.
            generators.append(SparsePauliOp(mapped.paulis))
    else:
        candidates = len(options.generators)
        if candidates > allowed:
            raise ConfigError(
                "explicit generators exceed max_symmetries or the retained min_qubits"
            )
        for label in options.generators:
            try:
                operator = SparsePauliOp(label)
            except Exception as exc:
                raise ConfigError(
                    "symmetry generators must be valid Hermitian Pauli labels"
                ) from exc
            if operator.num_qubits != width:
                raise ConfigError("symmetry generator has the wrong qubit width")
            pauli_binary_vector(operator)
            generators.append(operator)
    if reference_clifford is None:
        _state_budget(width, options.max_statevector_bytes)
        state = Statevector.from_instruction(reference)
    else:
        state = StabilizerState(reference_clifford)
    constraints = []
    sectors = []
    for index, generator in enumerate(generators):
        vector = pauli_binary_vector(generator)
        if not np.any(vector):
            raise ConfigError("identity is not an independent tapering generator")
        if not np.all(generator.paulis.commutes(hamiltonian.paulis)):
            raise ConfigError("symmetry generator does not commute with the Hamiltonian")
        for earlier in generators[:index]:
            if not generator.paulis[0].commutes(earlier.paulis[0]):
                raise ConfigError("symmetry generators must mutually commute")
        expectation = complex(generator.coeffs[0] * state.expectation_value(generator.paulis[0]))
        if (
            not np.isclose(abs(expectation.real), 1, atol=options.tolerance, rtol=0)
            or abs(expectation.imag) > options.tolerance
        ):
            raise ConfigError("reference is not an eigenstate of a requested symmetry generator")
        eigenvalue = 1 if expectation.real > 0 else -1
        if options.sectors is not None and options.sectors[index] != eigenvalue:
            raise ConfigError("explicit symmetry sector disagrees with the actual reference")
        constraints.append(eigenvalue * generator)
        sectors.append(eigenvalue)
    if generators and len(
        binary_rref(np.array([pauli_binary_vector(operator) for operator in generators]))[1]
    ) != len(generators):
        raise ConfigError("symmetry generators must be independent")
    signed_labels = [
        ("-" if operator.coeffs[0].real < 0 else "+") + operator.paulis[0].to_label()
        for operator in constraints
    ]
    diagonalizer = (
        synth_circuit_from_stabilizers(signed_labels, allow_underconstrained=True).inverse()
        if constraints
        else QuantumCircuit(width)
    )
    clifford = Clifford(diagonalizer)
    for index, constraint in enumerate(constraints):
        image = SparsePauliOp(
            constraint.paulis.evolve(clifford, frame="s"), coeffs=constraint.coeffs
        ).simplify()
        target = "I" * (width - index - 1) + "Z" + "I" * index
        if image.to_list() != [(target, 1 + 0j)]:
            raise ConfigError(
                "symmetry Clifford synthesis returned an inconsistent coordinate order"
            )
    metadata = {
        "method": "reference_compatible_pauli_z2",
        "discovery": "clifford_reference_stabilizer_intersection"
        if options.generators is None
        else "explicit_generators",
        "compatible_generators_found": candidates,
        "num_tapered_qubits": len(generators),
        "num_reduced_qubits": width - len(generators),
        "generators": [
            {"pauli": operator.paulis[0].to_label(), "sign": round(operator.coeffs[0].real)}
            for operator in generators
        ],
        "sectors": sectors,
        "clifford": clifford.to_dict(),
        "reference_sector_only": True,
    }
    transform = TaperingTransform(
        width, len(generators), clifford, metadata, options.max_statevector_bytes, options.tolerance
    )
    transform.prepare_reference(reference)
    return transform


def build_reference_tapered_mapper(
    problem: Any,
    *,
    options: Z2TaperingOptions,
    reference_selection: Any = None,
    prepared: Any = None,
) -> Any:
    """Build a Nature-compatible mapper from the actually selected untapered reference."""
    from qiskit_nature.second_q.mappers import TaperedQubitMapper

    from chemrefine.engines.qiskit.context import ElectronicStructureContext
    from chemrefine.engines.qiskit.options import ComponentSelection
    from chemrefine.engines.qiskit.registry import INITIAL_STATES, MAPPERS

    selection = reference_selection or ComponentSelection.named("hartree_fock")
    if isinstance(selection, str):
        selection = ComponentSelection.named(selection)
    if not options.min_qubits <= 2 * problem.num_spatial_orbitals <= options.max_qubits:
        raise ConfigError("tapering register lies outside min_qubits/max_qubits")
    base_options = {"two_qubit_reduction": False} if options.base_mapper == "parity" else {}
    base = MAPPERS.build(
        ComponentSelection(name=options.base_mapper, options=base_options), problem=problem
    )
    full_hamiltonian = base.map(problem.hamiltonian.second_q_op()).simplify()
    context = ElectronicStructureContext(
        problem=problem,
        mapper=base,
        qubit_hamiltonian=full_hamiltonian,
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=tuple(problem.num_particles),
        num_qubits=full_hamiltonian.num_qubits,
        multiplicity=getattr(
            prepared, "multiplicity", abs(problem.num_particles[0] - problem.num_particles[1]) + 1
        ),
    )
    reference = INITIAL_STATES.build(selection, context=context)
    transform = build_tapering_transform(full_hamiltonian, reference, options=options)
    transform.metadata["reference_selection"] = selection.model_dump(mode="json")
    transform.metadata["base_mapper"] = options.base_mapper
    reduced_reference = transform.prepare_reference(reference)
    reduced_reference.metadata = dict(reference.metadata or {})

    class ReferenceTaperedMapper(TaperedQubitMapper):
        """Nature's pool/list contract with a reference-validated general Clifford transform."""

        chemrefine_tapering = transform
        chemrefine_reference = reduced_reference
        chemrefine_reference_selection = selection

        def _map_clifford_single(
            self, second_q_op: Any, *, register_length: int | None = None
        ) -> Any:
            """Use precisely the same base mapping and Clifford for every operator."""
            return transform.transform_operator(
                self.mapper.map(second_q_op, register_length=register_length)
            )

        def taper_clifford(
            self, pauli_ops: Any, *, check_commutes: bool = True, suppress_none: bool = True
        ) -> Any:
            """Retain positional holes for Nature's matching excitation-list filtering."""
            if isinstance(pauli_ops, list):
                reduced = [
                    transform.project_transformed(operator, check_commutes=check_commutes)
                    for operator in pauli_ops
                ]
                return (
                    [operator for operator in reduced if operator is not None]
                    if suppress_none
                    else reduced
                )
            if isinstance(pauli_ops, dict):
                mapped = {
                    key: transform.project_transformed(operator, check_commutes=check_commutes)
                    for key, operator in pauli_ops.items()
                }
                return (
                    {key: operator for key, operator in mapped.items() if operator is not None}
                    if suppress_none
                    else mapped
                )
            return transform.project_transformed(pauli_ops, check_commutes=check_commutes)

        def map_observable(self, second_q_op: Any) -> Any:
            """Project observables, including the exactly zero sector-changing contributions."""
            return transform.map_operator(self.mapper.map(second_q_op), check_commutes=False)

        def prepare_occupation_reference(self, occupations: tuple[bool, ...]) -> Any:
            """Prepare actual occupations and verify compatibility with the selected sector."""
            from chemrefine.engines.qiskit.components.initial_states import build_explicit_reference

            full = build_explicit_reference(context, occupations)
            return transform.prepare_reference(full)

    return ReferenceTaperedMapper(base)
