"""Bounded fermionic lattice dynamics, separate from molecular energy workflows.

Qiskit Fermions supplies the fermionic IR and untapered mapping synthesis.
Validated graph BKSF and open-square VC/DK encodings add explicit code sectors,
Clifford reference preparation, and commuting directed-transfer flow synthesis.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from chemrefine.engines.qiskit.encodings import LocalEncodingOptions
from chemrefine.errors import ConfigError


class LatticeEdge(BaseModel):
    """An undirected edge contributing hopping and density-density interaction.

    The oriented amplitude is ``t = hopping + 1j*hopping_imag`` and contributes
    ``-t*a†_source*a_target - t.conjugate()*a†_target*a_source`` per species.
    The interaction is ``density_interaction * n_i * n_j``, where each site's
    density includes both spins for a spinful model. Every edge is listed once.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    source: StrictInt = Field(ge=0)
    target: StrictInt = Field(ge=0)
    hopping: float = 1.0
    hopping_imag: float = 0.0
    density_interaction: float = 0.0


class FermionicLatticeModel(BaseModel):
    """Complex Hermitian hopping graph with Hubbard and extended interactions.

    Spinful mode order is all alpha sites followed by all beta sites. Onsite
    interaction means ``U n_up n_down``; potentials multiply total site density.
    No chemical-potential, half-filling, or constant-energy shifts are implicit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    num_sites: StrictInt = Field(ge=1)
    edges: tuple[LatticeEdge, ...] = ()
    spinful: bool = True
    onsite_interaction: float = 0.0
    site_potentials: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _validate_graph(self) -> Self:
        """Reject duplicated, missing, or physically ambiguous graph entries."""
        seen = set()
        for edge in self.edges:
            if edge.source == edge.target or max(edge.source, edge.target) >= self.num_sites:
                raise ValueError("lattice edges must join different sites below num_sites")
            key = tuple(sorted((edge.source, edge.target)))
            if key in seen:
                raise ValueError("each undirected lattice edge must be specified once")
            seen.add(key)
        if not self.spinful and self.onsite_interaction != 0:
            raise ValueError("onsite Hubbard interaction requires a spinful model")
        if self.site_potentials is not None and len(self.site_potentials) != self.num_sites:
            raise ValueError("site_potentials requires one value per site")
        return self

    @property
    def num_modes(self) -> int:
        """Return the number of spin orbitals before encoding."""
        return self.num_sites * (2 if self.spinful else 1)


def chain_lattice(
    num_sites: int,
    *,
    periodic: bool = False,
    spinful: bool = True,
    hopping: float = 1.0,
    hopping_imag: float = 0.0,
    onsite_interaction: float = 0.0,
    density_interaction: float = 0.0,
    site_potentials: Sequence[float] | None = None,
) -> FermionicLatticeModel:
    """Construct a simple open chain or ring with each undirected edge once."""
    if isinstance(num_sites, bool) or not isinstance(num_sites, Integral) or num_sites < 1:
        raise ConfigError("lattice num_sites must be a positive integer")
    pairs = [(index, index + 1) for index in range(num_sites - 1)]
    if periodic and num_sites > 2:
        pairs.append((0, num_sites - 1))
    return FermionicLatticeModel(
        num_sites=int(num_sites),
        edges=tuple(
            LatticeEdge(
                source=first,
                target=second,
                hopping=hopping,
                hopping_imag=hopping_imag,
                density_interaction=density_interaction,
            )
            for first, second in pairs
        ),
        spinful=spinful,
        onsite_interaction=onsite_interaction,
        site_potentials=None if site_potentials is None else tuple(site_potentials),
    )


def square_lattice(
    rows: int,
    columns: int,
    *,
    periodic: bool = False,
    spinful: bool = True,
    hopping: float = 1.0,
    hopping_imag: float = 0.0,
    onsite_interaction: float = 0.0,
    density_interaction: float = 0.0,
    site_potentials: Sequence[float] | None = None,
) -> FermionicLatticeModel:
    """Construct a row-major rectangular nearest-neighbor simple graph.

    A periodic dimension of length two still has one edge between its sites,
    rather than silently introducing parallel bonds with doubled hopping.
    """
    if any(
        isinstance(size, bool) or not isinstance(size, Integral) or size < 1
        for size in (rows, columns)
    ):
        raise ConfigError("lattice rows and columns must be positive integers")
    pairs = set()
    for row in range(rows):
        for column in range(columns):
            here = row * columns + column
            for neighbor_row, neighbor_column in ((row + 1, column), (row, column + 1)):
                if periodic:
                    neighbor_row %= rows
                    neighbor_column %= columns
                elif neighbor_row >= rows or neighbor_column >= columns:
                    continue
                there = neighbor_row * columns + neighbor_column
                if here != there:
                    pairs.add(tuple(sorted((here, there))))
    return FermionicLatticeModel(
        num_sites=int(rows * columns),
        edges=tuple(
            LatticeEdge(
                source=first,
                target=second,
                hopping=hopping,
                hopping_imag=hopping_imag,
                density_interaction=density_interaction,
            )
            for first, second in sorted(pairs)
        ),
        spinful=spinful,
        onsite_interaction=onsite_interaction,
        site_potentials=None if site_potentials is None else tuple(site_potentials),
    )


class LatticeIntegratorOptions(BaseModel):
    """Ideal simulation and product-formula controls with explicit resource limits."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    steps: StrictInt = Field(1, ge=1)
    order: Literal[1, 2, 4] = 2
    mapping: Literal[
        "jordan_wigner", "bravyi_kitaev", "parity", "bksf_graph", "vc_square", "dk_square"
    ] = "jordan_wigner"
    encoding: LocalEncodingOptions | None = None
    synthesis: Literal["pauli", "flow_sets"] = "pauli"
    exact_reference: bool = False
    max_qubits: StrictInt = Field(16, ge=1)
    max_evolution_blocks: StrictInt = Field(10000, ge=1)
    max_statevector_bytes: StrictInt = Field(268435456, ge=1)
    max_exact_qubits: StrictInt = Field(12, ge=1)

    @model_validator(mode="after")
    def _encoding_controls(self) -> Self:
        """Reject options that a selected mapping cannot consume."""
        local = self.mapping in {"bksf_graph", "vc_square", "dk_square"}
        if self.encoding is not None and (not local or self.encoding.name != self.mapping):
            raise ValueError("encoding.name must match a selected local mapping")
        if self.mapping in {"vc_square", "dk_square"} and self.encoding is None:
            raise ValueError("square mappings require encoding dimensions")
        if self.synthesis == "flow_sets" and not local:
            raise ValueError("flow_sets synthesis requires a local edge/vertex mapping")
        return self


class LatticeDynamicsOptions(LatticeIntegratorOptions):
    """Integrator controls and the physical evolution time for a single circuit."""

    time: float = 1.0


def lattice_encoding_qubits(model: FermionicLatticeModel, options: LatticeIntegratorOptions) -> int:
    """Return the encoded register size without importing any quantum provider."""
    if options.mapping == "bksf_graph":
        return max(1, len(model.edges) * (2 if model.spinful else 1))
    if options.mapping in {"vc_square", "dk_square"}:
        configuration = options.encoding
        if configuration is None or configuration.rows is None or configuration.columns is None:
            raise ConfigError("square mappings require encoding dimensions")
        rows, columns = configuration.rows, configuration.columns
        if rows * columns != model.num_sites:
            raise ConfigError("encoding rectangle must match num_sites per spin species")
        if options.mapping == "vc_square":
            return 2 * model.num_modes
        odd_faces = ((rows - 1) * (columns - 1) + 1) // 2
        return model.num_modes + odd_faces * (2 if model.spinful else 1)
    return model.num_modes


@dataclass(frozen=True)
class LatticeCircuit:
    """Mapped circuit, observables, reference preparation and optional code description."""

    circuit: Any
    fermionic_circuit: Any
    qubit_hamiltonian: Any
    number_observables: tuple[Any, ...]
    initial_circuit: Any
    metadata: dict[str, Any]
    encoding: Any = None


@dataclass(frozen=True)
class LatticeDynamicsResult:
    """Encoded ideal state and lattice observables; never a molecular single-point result."""

    statevector: NDArray[np.complex128]
    mode_occupations: tuple[float, ...]
    particle_number: float
    energy_expectation: float
    energy_drift: float
    return_probability: float
    exact_state_fidelity: float | None
    metadata: dict[str, Any]

    def as_dict(self, *, include_statevector: bool = False) -> dict[str, Any]:
        """Return JSON-compatible observables and optional encoded complex amplitudes."""
        result = {
            "mode_occupations": list(self.mode_occupations),
            "particle_number": self.particle_number,
            "energy_expectation": self.energy_expectation,
            "energy_drift": self.energy_drift,
            "return_probability": self.return_probability,
            "exact_state_fidelity": self.exact_state_fidelity,
            "metadata": self.metadata,
        }
        if include_statevector:
            result["statevector"] = {
                "real": self.statevector.real.tolist(),
                "imag": self.statevector.imag.tolist(),
            }
        return cast("dict[str, Any]", json.loads(json.dumps(result, allow_nan=False)))


def _lattice_blocks(model: FermionicLatticeModel) -> list[Any]:
    """Build Hermitian, number-conserving physical blocks with commuting Pauli images."""
    from qiskit_fermions.operators import FermionOperator

    blocks = []
    spins = (0, model.num_sites) if model.spinful else (0,)
    for edge in model.edges:
        if edge.hopping:
            for offset in spins:
                first, second = edge.source + offset, edge.target + offset
                blocks.append(
                    FermionOperator.from_terms(
                        [
                            ([(True, first), (False, second)], -edge.hopping),
                            ([(True, second), (False, first)], -edge.hopping),
                        ]
                    )
                )
        if edge.hopping_imag:
            for offset in spins:
                first, second = edge.source + offset, edge.target + offset
                blocks.append(
                    FermionOperator.from_terms(
                        [
                            ([(True, first), (False, second)], -1j * edge.hopping_imag),
                            ([(True, second), (False, first)], 1j * edge.hopping_imag),
                        ]
                    )
                )
        if edge.density_interaction:
            terms = []
            for first_spin in spins:
                for second_spin in spins:
                    first, second = edge.source + first_spin, edge.target + second_spin
                    terms.append(
                        (
                            [(True, first), (False, first), (True, second), (False, second)],
                            edge.density_interaction,
                        )
                    )
            blocks.append(FermionOperator.from_terms(terms))
    for site in range(model.num_sites):
        potential = 0.0 if model.site_potentials is None else model.site_potentials[site]
        if potential:
            blocks.append(
                FermionOperator.from_terms(
                    [
                        ([(True, site + offset), (False, site + offset)], potential)
                        for offset in spins
                    ]
                )
            )
        if model.onsite_interaction:
            blocks.append(
                FermionOperator.from_terms(
                    [
                        (
                            [
                                (True, site),
                                (False, site),
                                (True, site + model.num_sites),
                                (False, site + model.num_sites),
                            ],
                            model.onsite_interaction,
                        )
                    ]
                )
            )
    return blocks


def _mapper(mapping: str) -> Any:
    """Return a public Fermions mapper or a Nature-backed generic synthesis mapper."""
    from qiskit.quantum_info import SparseObservable
    from qiskit_fermions.mappers.library import jordan_wigner

    if mapping == "jordan_wigner":
        return jordan_wigner
    from qiskit_nature.second_q.mappers import BravyiKitaevMapper, ParityMapper
    from qiskit_nature.second_q.operators import FermionicOp

    nature_mapper = BravyiKitaevMapper() if mapping == "bravyi_kitaev" else ParityMapper()

    def map_operator(operator: Any, num_qubits: int) -> Any:
        """Translate the released FermionOperator terms without truncation or tapering."""
        labels: dict[str, complex] = {}
        for term, coefficient in operator.iter_terms():
            label = " ".join(f"{'+' if creation else '-'}_{index}" for creation, index in term)
            labels[label] = labels.get(label, 0.0) + coefficient
        mapped = nature_mapper.map(FermionicOp(labels, num_spin_orbitals=num_qubits))
        return SparseObservable.from_sparse_pauli_op(mapped)

    return map_operator


def _product_sequence(block_count: int, order: int, interval: float) -> list[tuple[int, float]]:
    """Return chronological Lie or symmetric Suzuki block intervals through order four."""
    if order == 1:
        return [(index, interval) for index in range(block_count)]
    if order == 2:
        forward = [(index, interval / 2) for index in range(block_count)]
        return forward + list(reversed(forward))
    coefficient = 1 / (4 - 4 ** (1 / 3))
    outer = _product_sequence(block_count, 2, coefficient * interval)
    middle = _product_sequence(block_count, 2, (1 - 4 * coefficient) * interval)
    return outer + outer + middle + outer + outer


def _local_flow_blocks(model: FermionicLatticeModel, encoding: Any) -> list[tuple[Any, Any]]:
    """Split real hopping into directed flows; retain physical imaginary/density blocks."""
    from chemrefine.engines.qiskit.flow import partition_flow_edges, vc_flow_diagonalizer

    weighted = {
        (first + offset, second + offset): edge.hopping
        for edge in model.edges
        if edge.hopping
        for offset in ((0, model.num_sites) if model.spinful else (0,))
        for first, second in ((edge.source, edge.target), (edge.target, edge.source))
    }
    blocks: list[tuple[Any, Any]] = []
    if encoding.metadata["name"] == "vc_square":
        rows, columns = encoding.metadata["rows"], encoding.metadata["columns"]
        for direction in ("east", "west", "south", "north"):
            selected = {
                (first, second): value
                for (first, second), value in weighted.items()
                if (
                    "east"
                    if second - first == 1
                    else "west"
                    if first - second == 1
                    else "south"
                    if second > first
                    else "north"
                )
                == direction
            }
            if not selected:
                continue
            transfer = sum(
                (
                    0.5j * value * encoding.vertices[first] @ encoding.edge_operator(first, second)
                    for (first, second), value in selected.items()
                )
            ).simplify()
            basis = vc_flow_diagonalizer(rows, columns, 2 if model.spinful else 1, direction)
            blocks.append((transfer, basis))
    else:
        for group in partition_flow_edges(tuple(weighted)):
            transfer = sum(
                (
                    0.5j
                    * weighted[first, second]
                    * encoding.vertices[first]
                    @ encoding.edge_operator(first, second)
                    for first, second in group
                )
            ).simplify()
            blocks.append((transfer, None))
    # Imaginary hopping is a separate Hermitian number-conserving current block.
    # It is not passed off as one of the real-transfer flow families.
    remainder = model.model_copy(
        update={"edges": tuple(edge.model_copy(update={"hopping": 0.0}) for edge in model.edges)}
    )
    blocks.extend((encoding.map_operator(block), None) for block in _lattice_blocks(remainder))
    return blocks


def _build_local_dynamics(
    model: FermionicLatticeModel, occupied: tuple[int, ...], options: LatticeDynamicsOptions
) -> LatticeCircuit:
    """Build a code-preserving local-encoding product formula with explicit gauge fixing."""
    from qiskit_fermions.operators import FermionOperator

    from chemrefine.engines.qiskit.encodings import build_local_encoding, pauli_product
    from chemrefine.engines.qiskit.flow import commuting_evolution

    configuration = options.encoding or LocalEncodingOptions(name="bksf_graph")
    configuration = configuration.model_copy(
        update={"max_qubits": min(configuration.max_qubits, options.max_qubits)}
    )
    support = [
        (edge.source + offset, edge.target + offset)
        for edge in model.edges
        for offset in ((0, model.num_sites) if model.spinful else (0,))
    ]
    encoding = build_local_encoding(
        model.num_modes, support, options=configuration, occupied_modes=occupied
    )
    physical = _lattice_blocks(model)
    hamiltonian: Any = FermionOperator.zero()
    for block in physical:
        hamiltonian += block
    mapped_hamiltonian = encoding.map_operator(hamiltonian)
    blocks = (
        _local_flow_blocks(model, encoding)
        if options.synthesis == "flow_sets"
        else [(encoding.map_operator(block), None) for block in physical]
    )
    block_count = (
        len(blocks) * {1: 1, 2: 2, 4: 10}[options.order] * options.steps if options.time else 0
    )
    if block_count > options.max_evolution_blocks:
        raise ConfigError("lattice local synthesis exceeds max_evolution_blocks")
    initial = encoding.prepare_reference(occupied)
    circuit = initial.copy()
    if options.time:
        sequence = _product_sequence(len(blocks), options.order, options.time / options.steps)
        # Cache repeated signed time intervals, especially the repeated Suzuki factors.
        evolutions: dict[tuple[int, float], Any] = {}
        for index, interval in sequence:
            key = (index, interval)
            if key not in evolutions:
                block, basis = blocks[index]
                evolutions[key] = commuting_evolution(block, interval, diagonalizer=basis)
        for _ in range(options.steps):
            for index, interval in sequence:
                circuit.compose(evolutions[index, interval], inplace=True)
    numbers = tuple(
        (pauli_product(encoding.num_qubits) - vertex) / 2 for vertex in encoding.vertices
    )
    constraints = (*encoding.stabilizers, *encoding.gauge_fixers)
    metadata = {
        "model": model.model_dump(mode="json"),
        "options": options.model_dump(mode="json"),
        "occupied_modes": list(occupied),
        "num_modes": model.num_modes,
        "num_qubits": encoding.num_qubits,
        "auxiliary_qubits": max(0, encoding.num_qubits - model.num_modes),
        "stabilizer_constraints": [operator.to_list() for operator in constraints],
        "mapping_provider": "chemrefine_validated_edge_vertex_algebra",
        "encoding": options.mapping,
        "encoding_details": encoding.metadata,
        "spin_orbital_order": "alpha_then_beta" if model.spinful else "site_order",
        "evolution_convention": "exp(-i * time * H), hbar=1",
        "product_formula_split": (
            "directed_transfer_flows_and_physical_current_density_blocks"
            if options.synthesis == "flow_sets"
            else "number_conserving_hopping_current_density_blocks"
        ),
        "flow_particle_number_exact": options.synthesis != "flow_sets",
        "evolution_blocks": block_count,
        "circuit_depth": circuit.depth(),
        "reference_depth": initial.depth(),
        "gate_counts": dict(circuit.count_ops()),
        "depth_convention": "abstract_connectivity_before_hardware_routing",
        "execution": "circuit_construction",
    }
    # Signed Hermitian generators have real unit coefficients; keep metadata JSON-native.
    metadata["stabilizer_constraints"] = [
        {"pauli": operator.paulis[0].to_label(), "eigenvalue": round(operator.coeffs[0].real)}
        for operator in constraints
    ]
    return LatticeCircuit(circuit, None, mapped_hamiltonian, numbers, initial, metadata, encoding)


def build_lattice_dynamics(
    model: FermionicLatticeModel,
    *,
    occupied_modes: Sequence[int],
    options: LatticeDynamicsOptions | None = None,
) -> LatticeCircuit:
    """Prepare actual occupations and synthesize exp(-i H time) approximately.

    Product formulas split Hermitian physical hopping/density blocks. Each
    block's mapped Paulis commute, so their internal Lie synthesis is exact;
    only splitting noncommuting physical blocks contributes Trotter error.
    """
    options = options or LatticeDynamicsOptions()
    modes = model.num_modes
    width = lattice_encoding_qubits(model, options)
    if width > options.max_qubits:
        raise ConfigError(
            f"lattice requires {width} qubits, exceeding max_qubits={options.max_qubits}"
        )
    if any(isinstance(mode, bool) or not isinstance(mode, Integral) for mode in occupied_modes):
        raise ConfigError("lattice occupied_modes must contain integer mode indices")
    occupied = tuple(sorted(int(mode) for mode in occupied_modes))
    if len(set(occupied)) != len(occupied) or any(mode < 0 or mode >= modes for mode in occupied):
        raise ConfigError("lattice occupied_modes must be distinct and within the mode count")
    if options.mapping in {"bksf_graph", "vc_square", "dk_square"}:
        return _build_local_dynamics(model, occupied, options)
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter
    from qiskit_fermions.circuit import FermionicCircuit
    from qiskit_fermions.circuit.library import Evolution
    from qiskit_fermions.operators import FermionOperator
    from qiskit_fermions.transpiler.passes import F2QSynthesis
    from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager

    blocks = _lattice_blocks(model)
    multiplier = {1: 1, 2: 2, 4: 10}[options.order]
    block_count = len(blocks) * multiplier * options.steps if options.time else 0
    if block_count > options.max_evolution_blocks:
        raise ConfigError(
            f"lattice requires {block_count} evolution blocks, exceeding max_evolution_blocks"
        )
    map_operator = _mapper(options.mapping)
    hamiltonian = FermionOperator.zero()
    for block in blocks:
        mapped_block = SparsePauliOp.from_sparse_observable(map_operator(block, modes)).simplify()
        if len(mapped_block.group_commuting()) != 1:
            raise ConfigError("lattice physical block must have commuting mapped Pauli terms")
        hamiltonian = hamiltonian + block
    mapped_hamiltonian = SparsePauliOp.from_sparse_observable(
        map_operator(hamiltonian, modes)
    ).simplify()
    initial = QuantumCircuit(modes)
    if occupied:
        # All three supported encodings are invertible binary changes of the
        # occupation basis. XOR their single-creation X masks instead of mapping
        # a product that would expand into 2**len(occupied) Pauli terms.
        occupation_mask = np.zeros(modes, dtype=bool)
        for mode in occupied:
            creation = FermionOperator.from_terms([([(True, mode)], 1.0)])
            mapped_creation = SparsePauliOp.from_sparse_observable(map_operator(creation, modes))
            masks = mapped_creation.paulis.x
            if not np.all(masks == masks[0]):
                raise ConfigError(
                    "selected lattice encoding does not map a determinant to one bitstring"
                )
            occupation_mask ^= masks[0]
        initial.x(np.flatnonzero(occupation_mask).tolist())
    fermionic = FermionicCircuit(modes)
    if options.time:
        sequence = _product_sequence(len(blocks), options.order, options.time / options.steps)
        for _ in range(options.steps):
            for index, interval in sequence:
                fermionic.append(Evolution(modes, blocks[index], time=interval), range(modes))
    manager = generate_preset_jw_pass_manager(
        basis_gates=["rz", "sx", "x", "cx"], optimization_level=0
    )
    # The preset's layout is one mode per qubit. Replace its synthesis mapper;
    # reference preparation above already uses the same untapered encoding.
    manager.synthesis = F2QSynthesis({"Evolution": ("MapperFn", (map_operator, LieTrotter()))})
    circuit = initial.compose(manager.run(fermionic))
    numbers = tuple(
        SparsePauliOp.from_sparse_observable(
            map_operator(
                FermionOperator.from_terms([([(True, index), (False, index)], 1.0)]), modes
            )
        ).simplify()
        for index in range(modes)
    )
    metadata = {
        "model": model.model_dump(mode="json"),
        "options": options.model_dump(mode="json"),
        "occupied_modes": list(occupied),
        "num_modes": modes,
        "num_qubits": modes,
        "auxiliary_qubits": 0,
        "stabilizer_constraints": [],
        "mapping_provider": "qiskit_fermions"
        if options.mapping == "jordan_wigner"
        else "qiskit_nature",
        "encoding": options.mapping,
        "spin_orbital_order": "alpha_then_beta" if model.spinful else "site_order",
        "evolution_convention": "exp(-i * time * H), hbar=1",
        "product_formula_split": "number_conserving_hopping_and_density_blocks",
        "evolution_blocks": block_count,
        "circuit_depth": circuit.depth(),
        "gate_counts": dict(circuit.count_ops()),
        "execution": "circuit_construction",
    }
    return LatticeCircuit(circuit, fermionic, mapped_hamiltonian, numbers, initial, metadata)


def simulate_lattice_dynamics(
    model: FermionicLatticeModel,
    *,
    occupied_modes: Sequence[int],
    options: LatticeDynamicsOptions | None = None,
) -> LatticeDynamicsResult:
    """Simulate a bounded ideal circuit and optionally compare with exact evolution.

    Energy units are those of the caller's Hamiltonian; time uses their inverse.
    The byte limit accounts for four complex statevectors, not total process RSS.
    The optional exact reference uses sparse exponential action, never a dense
    matrix exponential, and has its own smaller qubit limit.
    """
    options = options or LatticeDynamicsOptions()
    width = lattice_encoding_qubits(model, options)
    working_bytes = 4 * 16 * (1 << width)
    if working_bytes > options.max_statevector_bytes:
        raise ConfigError("lattice working statevectors exceed max_statevector_bytes")
    if options.exact_reference and width > options.max_exact_qubits:
        raise ConfigError("lattice exact reference exceeds max_exact_qubits")
    from qiskit.quantum_info import Statevector

    artifacts = build_lattice_dynamics(model, occupied_modes=occupied_modes, options=options)
    initial = Statevector.from_instruction(artifacts.initial_circuit)
    evolved = Statevector.from_instruction(artifacts.circuit)
    occupations = tuple(
        float(evolved.expectation_value(number).real) for number in artifacts.number_observables
    )
    energy = float(evolved.expectation_value(artifacts.qubit_hamiltonian).real)
    initial_energy = float(initial.expectation_value(artifacts.qubit_hamiltonian).real)
    exact_fidelity = None
    if options.exact_reference:
        from scipy.sparse.linalg import expm_multiply

        generator = -1j * options.time * artifacts.qubit_hamiltonian.to_matrix(sparse=True)
        exact = expm_multiply(generator, initial.data, traceA=generator.diagonal().sum())
        exact_fidelity = float(np.clip(abs(np.vdot(exact, evolved.data)) ** 2, 0.0, 1.0))
    statevector = np.array(evolved.data, dtype=complex, copy=True)
    statevector.setflags(write=False)
    return LatticeDynamicsResult(
        statevector=statevector,
        mode_occupations=occupations,
        particle_number=sum(occupations),
        energy_expectation=energy,
        energy_drift=energy - initial_energy,
        return_probability=float(np.clip(abs(np.vdot(initial.data, evolved.data)) ** 2, 0.0, 1.0)),
        exact_state_fidelity=exact_fidelity,
        metadata={
            **artifacts.metadata,
            "execution": "ideal_local_statevector",
            "working_statevector_bytes": working_bytes,
            "initial_energy": initial_energy,
            "particle_number_drift": sum(occupations) - len(occupied_modes),
            "code_constraint_expectations": [
                float(evolved.expectation_value(operator).real)
                for operator in (
                    (*artifacts.encoding.stabilizers, *artifacts.encoding.gauge_fixers)
                    if artifacts.encoding is not None
                    else ()
                )
            ],
        },
    )
