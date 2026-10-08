"""Validated local fermion encodings with explicit code sectors and Clifford preparation.

The even edge/vertex algebra follows arXiv:1810.05274; square compact encodings
follow arXiv:2003.06939 and the VC transfer convention in Qiskit Fermions' 2D guide.
No claim about routed circuit depth follows from an encoding's operator locality.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from chemrefine.errors import ConfigError


class LocalEncodingOptions(BaseModel):
    """Choose a graph encoding or an open rectangular square-lattice construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: Literal["bksf_graph", "vc_square", "dk_square"] = "bksf_graph"
    rows: StrictInt | None = Field(None, ge=2)
    columns: StrictInt | None = Field(None, ge=2)
    component_parities: tuple[Literal[0, 1], ...] | None = None
    max_qubits: StrictInt = Field(128, ge=1)
    max_generators: StrictInt = Field(2048, ge=1)
    max_expanded_terms: StrictInt = Field(100000, ge=1)

    @model_validator(mode="after")
    def _shape(self) -> Self:
        """Square encodings require dimensions; graph encodings must not ignore them."""
        square = self.name != "bksf_graph"
        if square != (self.rows is not None and self.columns is not None):
            raise ValueError("square encodings require both rows and columns")
        if not square and (self.rows is not None or self.columns is not None):
            raise ValueError("bksf_graph does not consume rectangular dimensions")
        if square and self.component_parities is not None:
            raise ValueError("square encodings retain both parities; omit component_parities")
        return self


def pauli_product(
    width: int, letters: str = "", indices: Sequence[int] = (), sign: float = 1
) -> Any:
    """Construct a Pauli shared with lattice evolution; qubit zero prints on the right."""
    from qiskit.quantum_info import SparsePauliOp

    return SparsePauliOp.from_sparse_list([(letters, list(indices), sign)], num_qubits=width)


def pauli_binary_vector(operator: Any) -> np.ndarray:
    """Return binary Pauli coordinates used by encodings, graph flows and tapering."""
    operator = operator.simplify()
    if len(operator) != 1 or not np.isclose(abs(operator.coeffs[0]), 1):
        raise ConfigError("encoding generators must be single unit Pauli operators")
    if not np.isclose(operator.coeffs[0].imag, 0):
        raise ConfigError("encoding generators must be Hermitian")
    return np.concatenate((operator.paulis.x[0], operator.paulis.z[0]))


def _symplectic(first: np.ndarray, second: np.ndarray) -> bool:
    """Whether two Pauli vectors anticommute."""
    half = len(first) // 2
    return bool((int(first[:half] @ second[half:]) + int(first[half:] @ second[:half])) % 2)


def binary_rref(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduce binary matrices for encodings, graph flows and tapering without float ranks."""
    result = np.array(matrix, dtype=np.uint8, copy=True)
    pivots: list[int] = []
    row = 0
    for column in range(result.shape[1]):
        candidates = np.flatnonzero(result[row:, column])
        if not len(candidates):
            continue
        pivot = row + int(candidates[0])
        result[[row, pivot]] = result[[pivot, row]]
        for other in np.flatnonzero(result[:, column]):
            if other != row:
                result[other] ^= result[row]
        pivots.append(column)
        row += 1
        if row == len(result):
            break
    return result[:row], pivots


def binary_nullspace(matrix: np.ndarray) -> list[np.ndarray]:
    """Compute a deterministic GF(2) kernel basis shared with symmetry tapering."""
    reduced, pivots = binary_rref(matrix)
    result = []
    for column in range(matrix.shape[1]):
        if column in pivots:
            continue
        vector = np.zeros(matrix.shape[1], dtype=np.uint8)
        vector[column] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = reduced[row, column]
        result.append(vector)
    return result


def independent_paulis(operators: Sequence[Any], width: int) -> tuple[Any, ...]:
    """Filter encoding and tapering constraints, rejecting a negative identity."""
    rows: dict[int, tuple[np.ndarray, Any]] = {}
    result: list[Any] = []
    for original in operators:
        operator = original.simplify()
        vector = pauli_binary_vector(operator).astype(np.uint8)
        for pivot, (basis, pauli) in sorted(rows.items()):
            if vector[pivot]:
                vector ^= basis
                operator = (operator @ pauli).simplify()
        nonzero = np.flatnonzero(vector)
        if not len(nonzero):
            if not np.isclose(operator.coeffs[0], 1):
                raise ConfigError("encoding constraints are inconsistent (-identity)")
            continue
        if len(result) >= width:
            raise ConfigError("too many independent commuting encoding constraints")
        rows[int(nonzero[0])] = (vector, operator)
        result.append(original)
    return tuple(result)


def _components(num_modes: int, edges: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    """Order graph components and vertices deterministically, including isolated modes."""
    adjacent: list[list[int]] = [[] for _ in range(num_modes)]
    for first, second in edges:
        adjacent[first].append(second)
        adjacent[second].append(first)
    remaining = set(range(num_modes))
    components = []
    while remaining:
        todo = [min(remaining)]
        component = set()
        while todo:
            vertex = todo.pop()
            if vertex not in component:
                component.add(vertex)
                todo.extend(adjacent[vertex])
        remaining -= component
        components.append(tuple(sorted(component)))
    return tuple(components)


def _path(edges: Mapping[tuple[int, int], Any], source: int, target: int) -> list[int]:
    """Find a deterministic path through the encoding graph, or reject a sector change."""
    neighbors: dict[int, list[int]] = {}
    for first, second in edges:
        neighbors.setdefault(first, []).append(second)
        neighbors.setdefault(second, []).append(first)
    queue = deque([source])
    previous: dict[int, int | None] = {source: None}
    while queue:
        vertex = queue.popleft()
        if vertex == target:
            result = [target]
            predecessor = previous[result[-1]]
            while predecessor is not None:
                result.append(predecessor)
                predecessor = previous[predecessor]
            return list(reversed(result))
        for neighbor in sorted(neighbors.get(vertex, ())):
            if neighbor not in previous:
                previous[neighbor] = vertex
                queue.append(neighbor)
    raise ConfigError("operator changes parity between disconnected encoding components")


@dataclass(frozen=True)
class FermionicEncoding:
    """A verified even-algebra representation with fixed gauge and explicit mode order."""

    num_modes: int
    num_qubits: int
    vertices: tuple[Any, ...]
    edges: Mapping[tuple[int, int], Any]
    stabilizers: tuple[Any, ...]
    gauge_fixers: tuple[Any, ...]
    components: tuple[tuple[int, ...], ...]
    component_parities: tuple[int, ...] | None
    metadata: dict[str, Any]
    max_expanded_terms: int = 100000

    def edge_operator(self, source: int, target: int) -> Any:
        """Represent an edge directly or by a cycle-consistent product along a graph path."""
        if source == target:
            raise ConfigError("an edge operator requires two different modes")
        if (source, target) in self.edges:
            return self.edges[source, target]
        if (target, source) in self.edges:
            return -self.edges[target, source]
        path = _path(self.edges, source, target)
        result = pauli_product(self.num_qubits) * (1j ** (len(path) - 2))
        for first, second in pairwise(path):
            result = result @ self.edge_operator(first, second)
        return result.simplify()

    def _majorana_pair(self, first: int, second: int) -> Any:
        """Map two ordered Majoranas with phases fixed by V=-i gamma_even gamma_odd."""
        left, a = divmod(first, 2)
        right, b = divmod(second, 2)
        if left == right:
            if a == b:
                return pauli_product(self.num_qubits)
            return (1j if a == 0 else -1j) * self.vertices[left]
        result = (1j ** (1 + a + b)) * self.edge_operator(left, right)
        if a:
            result = result @ self.vertices[left]
        if b:
            result = result @ self.vertices[right]
        return result.simplify()

    def map_operator(self, operator: Any) -> Any:
        """Map a number-conserving FermionOperator, including complex quartic observables."""
        from qiskit_fermions.mappers.library import fermion_to_majorana

        expansion = 0
        for term, coefficient in operator.iter_terms():
            if any(mode < 0 or mode >= self.num_modes for _, mode in term):
                raise ConfigError("fermionic operator contains a mode outside the encoding")
            if sum(1 if creation else -1 for creation, _ in term):
                raise ConfigError("local encoding interface requires number-conserving operators")
            if not np.isfinite(complex(coefficient)):
                raise ConfigError("fermionic coefficients must be finite")
            expansion += 1 << len(term)
            if expansion > self.max_expanded_terms:
                raise ConfigError("Majorana expansion exceeds max_expanded_terms")
        component_of = {
            mode: index for index, component in enumerate(self.components) for mode in component
        }
        result = 0 * pauli_product(self.num_qubits)
        for term, coefficient in fermion_to_majorana(operator).iter_terms():
            # Pair within components, retaining the CAR sign of every inter-component swap.
            groups = [component_of[mode // 2] for mode in term]
            inversions = sum(a > b for index, a in enumerate(groups) for b in groups[index + 1 :])
            ordered = sorted(zip(groups, term, strict=True), key=lambda pair: pair[0])
            if any(groups.count(index) % 2 for index in set(groups)):
                raise ConfigError(
                    "operator changes a fixed component parity; connect the encoding graph"
                )
            mapped = pauli_product(self.num_qubits) * (coefficient * (-1) ** inversions)
            for index in range(0, len(ordered), 2):
                mapped = mapped @ self._majorana_pair(ordered[index][1], ordered[index + 1][1])
            result += mapped
        return result.simplify()

    def prepare_reference(self, occupied_modes: Sequence[int]) -> Any:
        """Synthesize a deterministic Clifford preparing a Fock state in the physical code."""
        from qiskit.synthesis import synth_circuit_from_stabilizers

        occupied = set(occupied_modes)
        if len(occupied) != len(occupied_modes) or any(
            isinstance(mode, bool)
            or not isinstance(mode, (int, np.integer))
            or mode < 0
            or mode >= self.num_modes
            for mode in occupied
        ):
            raise ConfigError("occupied_modes must be distinct valid integer mode indices")
        if self.component_parities is not None and any(
            sum(mode in occupied for mode in component) % 2 != parity
            for component, parity in zip(self.components, self.component_parities, strict=True)
        ):
            raise ConfigError("reference occupations disagree with the encoding parity sectors")
        constraints = [*self.stabilizers, *self.gauge_fixers]
        constraints.extend(
            (-1 if mode in occupied else 1) * vertex for mode, vertex in enumerate(self.vertices)
        )
        independent = independent_paulis(constraints, self.num_qubits)
        if len(independent) != self.num_qubits:
            raise ConfigError("encoding leaves unexplained reference-state degrees of freedom")
        labels = []
        for operator in independent:
            operator = operator.simplify()
            labels.append(
                ("-" if operator.coeffs[0].real < 0 else "+") + operator.paulis[0].to_label()
            )
        return synth_circuit_from_stabilizers(labels)

    def decode_occupations(self, bitstring: str) -> tuple[int, ...]:
        """Decode canonical Qiskit computational samples through the vertex observables."""
        if len(bitstring) != self.num_qubits or set(bitstring) - {"0", "1"}:
            raise ConfigError("encoded bitstrings must match the qubit width")
        bits = np.array([int(bit) for bit in reversed(bitstring)], dtype=np.uint8)
        result = []
        for vertex in self.vertices:
            if np.any(vertex.paulis.x):
                raise ConfigError("this encoding's occupations require a measurement basis change")
            parity = int(np.dot(vertex.paulis.z[0].astype(np.uint8), bits)) % 2
            eigenvalue = round(vertex.coeffs[0].real) * (-1) ** parity
            result.append((1 - eigenvalue) // 2)
        return tuple(result)


def validate_encoding(
    vertices: Sequence[Any],
    edges: Mapping[tuple[int, int], Any],
    *,
    name: str = "custom",
    component_parities: tuple[int, ...] | None = None,
    max_generators: int = 2048,
    max_expanded_terms: int = 100000,
) -> FermionicEncoding:
    """Validate a custom generator representation and derive cycle checks and gauge fixing.

    Gauge operators are selected from the full algebra's symplectic commutant. This
    preserves every represented observable, rather than fixing accidental symmetries
    of only one Hamiltonian. No exponential projector or postselection is used.
    """
    from qiskit.quantum_info import Pauli, SparsePauliOp

    if not vertices or len(vertices) + len(edges) > max_generators:
        raise ConfigError("encoding has no modes or exceeds max_generators")
    width = vertices[0].num_qubits
    if width < 1:
        raise ConfigError(
            "custom encodings require at least one qubit, including scalar gauge padding"
        )
    if any(operator.num_qubits != width for operator in [*vertices, *edges.values()]):
        raise ConfigError("encoding generators have inconsistent qubit widths")
    modes = len(vertices)
    if any(not (0 <= first < second < modes) for first, second in edges):
        raise ConfigError("encoding edges must use distinct ordered endpoints within the modes")
    operators = [*vertices, *edges.values()]
    vectors = [pauli_binary_vector(operator).astype(np.uint8) for operator in operators]
    supports = [(index,) for index in range(modes)] + list(edges)
    for index, first_vector in enumerate(vectors):
        for other in range(index):
            expected = bool(set(supports[index]) & set(supports[other]))
            if index < modes and other < modes:
                expected = False
            if _symplectic(first_vector, vectors[other]) != expected:
                raise ConfigError("encoding violates the edge/vertex commutation algebra")
    forest: dict[tuple[int, int], Any] = {}
    cycles = []
    for (first, second), operator in sorted(edges.items()):
        try:
            path = _path(forest, first, second)
        except ConfigError:
            forest[first, second] = operator
            continue
        loop = pauli_product(width) * (1j ** len(path))
        for source, target in pairwise(path):
            loop = loop @ (forest[source, target] if source < target else -forest[target, source])
        cycles.append((loop @ -operator).simplify())
    stabilizers = independent_paulis(cycles, width)
    components = _components(modes, list(edges))
    if component_parities is not None and (
        len(component_parities) != len(components)
        or any(isinstance(parity, bool) or parity not in (0, 1) for parity in component_parities)
    ):
        raise ConfigError("one parity is required per encoding graph component")
    # The physical vertex algebra must retain exactly the declared occupations.
    # Commutation checks alone would accept a representation with spurious fixed
    # occupations or undeclared component-parity restrictions.
    vertex_rank = len(
        binary_rref(
            np.array([pauli_binary_vector(operator) for operator in (*stabilizers, *vertices)])
        )[1]
    )
    physical_bits = modes - (len(components) if component_parities is not None else 0)
    if vertex_rank != len(stabilizers) + physical_bits:
        raise ConfigError(
            "encoding fixes undeclared occupation sectors or has an invalid parity rank"
        )
    if component_parities is not None:
        for component, parity in zip(components, component_parities, strict=True):
            product = pauli_product(width) * (-1) ** parity
            for mode in component:
                product = product @ vertices[mode]
            # The rank and edge algebra already enforce parity independence;
            # signed reduction verifies that the declared eigenvalue is consistent.
            independent_paulis((*stabilizers, product), width)
    # The centralizer of generator rows x|z solves [z|x] g = 0.
    matrix = np.array([np.concatenate((vector[width:], vector[:width])) for vector in vectors])
    commutant = binary_nullspace(matrix)
    gauge = []
    while commutant:
        first_vector = commutant.pop(0)
        partner = next(
            (index for index, vector in enumerate(commutant) if _symplectic(first_vector, vector)),
            None,
        )
        if partner is None:
            continue  # radical: code checks and physical component parity, not gauge
        second_vector = commutant.pop(partner)
        gauge.append(
            SparsePauliOp(
                Pauli((first_vector[width:].astype(bool), first_vector[:width].astype(bool)))
            )
        )
        commutant = [
            vector
            ^ (first_vector if _symplectic(vector, second_vector) else 0)
            ^ (second_vector if _symplectic(vector, first_vector) else 0)
            for vector in commutant
        ]
    return FermionicEncoding(
        modes,
        width,
        tuple(vertices),
        dict(edges),
        stabilizers,
        tuple(gauge),
        components,
        component_parities,
        {
            "name": name,
            "num_modes": modes,
            "num_qubits": width,
            "code_checks": len(stabilizers),
            "gauge_qubits": len(gauge),
            "component_parities": component_parities,
            "preparation": "clifford_stabilizer_synthesis",
        },
        max_expanded_terms,
    )


def build_local_encoding(
    num_modes: int,
    edges: Sequence[tuple[int, int]],
    *,
    options: LocalEncodingOptions | None = None,
    occupied_modes: Sequence[int] = (),
) -> FermionicEncoding:
    """Construct a bounded graph BKSF or open square VC/DK encoding for actual occupations."""
    options = options or LocalEncodingOptions()
    if isinstance(num_modes, bool) or not isinstance(num_modes, Integral) or num_modes < 1:
        raise ConfigError("num_modes must be a positive integer")
    if any(
        len(edge) != 2
        or any(isinstance(mode, bool) or not isinstance(mode, Integral) for mode in edge)
        for edge in edges
    ):
        raise ConfigError("encoding graph edges must contain pairs of integer modes")
    canonical = sorted({tuple(sorted(edge)) for edge in edges})
    if len(canonical) != len(edges) or any(
        not (0 <= first < second < num_modes) for first, second in canonical
    ):
        raise ConfigError("encoding graph edges must be distinct valid pairs")
    components = _components(num_modes, canonical)
    if num_modes + len(canonical) > options.max_generators:
        raise ConfigError("encoding exceeds max_generators")
    parities = options.component_parities
    if options.name == "bksf_graph":
        if parities is None:
            parities = tuple(
                sum(mode in occupied_modes for mode in component) % 2 for component in components
            )
        if len(parities) != len(components):
            raise ConfigError("one parity is required per graph component")
        # Qiskit's Pauli composition does not support a zero-qubit scalar algebra.
        # An edgeless fixed-parity problem gets one explicit, frozen gauge qubit.
        width = max(1, len(canonical))
        if width > options.max_qubits:
            raise ConfigError("local encoding exceeds max_qubits")
        incident = {
            mode: [index for index, edge in enumerate(canonical) if mode in edge]
            for mode in range(num_modes)
        }
        vertices = [
            pauli_product(width, "Z" * len(incident[mode]), incident[mode])
            for mode in range(num_modes)
        ]
        for component, parity in zip(components, parities, strict=True):
            vertices[component[0]] *= (-1) ** parity
        mapped_edges = {}
        for index, (first, second) in enumerate(canonical):
            previous = [
                other for mode in (first, second) for other in incident[mode] if other < index
            ]
            mapped_edges[first, second] = pauli_product(
                width, "X" + "Z" * len(previous), [index, *previous]
            )
    else:
        if options.rows is None or options.columns is None:
            raise ConfigError("square encodings require rows and columns")
        rows, columns = options.rows, options.columns
        sites = rows * columns
        if num_modes % sites:
            raise ConfigError("square dimensions must tile complete mode-species registers")
        square_edges = [
            (site, site + delta)
            for site in range(sites)
            for delta in (1, columns)
            if (delta == 1 and site % columns + 1 < columns)
            or (delta == columns and site // columns + 1 < rows)
        ]
        full_edges = [
            (first + offset, second + offset)
            for offset in range(0, num_modes, sites)
            for first, second in square_edges
        ]
        if num_modes + len(full_edges) > options.max_generators:
            raise ConfigError("square encoding exceeds max_generators")
        if not set(canonical) <= set(full_edges):
            raise ConfigError("square encoding only accepts open nearest-neighbor support edges")
        odd_faces = [
            (row, column)
            for row in range(rows - 1)
            for column in range(columns - 1)
            if (row + column) % 2 == 0
        ]
        auxiliaries = sites if options.name == "vc_square" else len(odd_faces)
        width = num_modes + auxiliaries * (num_modes // sites)
        if width > options.max_qubits:
            raise ConfigError("local encoding exceeds max_qubits")
        vertices = [pauli_product(width, "Z", [mode]) for mode in range(num_modes)]
        mapped_edges = {}
        for offset in range(0, num_modes, sites):
            aux_start = num_modes + (offset // sites) * auxiliaries
            face_qubits = {face: aux_start + index for index, face in enumerate(odd_faces)}
            for first, second in square_edges:
                row, column = divmod(first, columns)
                vertical = second - first == columns
                if options.name == "vc_square":
                    letters, indices = (
                        (
                            "XYXY",
                            [
                                first + offset,
                                second + offset,
                                aux_start + first,
                                aux_start + second,
                            ],
                        )
                        if vertical
                        else ("XXZ", [first + offset, second + offset, aux_start + first])
                    )
                    transfer = pauli_product(width, letters, indices, 0.5)
                    edge = (-2j * vertices[first + offset] @ transfer).simplify()
                else:
                    forward = column % 2 == 0 if vertical else row % 2 == 0
                    tail, head = (first, second) if forward else (second, first)
                    adjacent = (
                        [(row, column - 1), (row, column)]
                        if vertical
                        else [(row - 1, column), (row, column)]
                    )
                    face = next(
                        (face_qubits[face] for face in adjacent if face in face_qubits), None
                    )
                    letters, indices = "XY", [tail + offset, head + offset]
                    if face is not None:
                        letters += "X" if vertical else "Y"
                        indices.append(face)
                    sign = -1 if vertical and not forward else 1
                    edge = pauli_product(width, letters, indices, sign * (1 if forward else -1))
                mapped_edges[first + offset, second + offset] = edge
    result = validate_encoding(
        vertices,
        mapped_edges,
        name=options.name,
        component_parities=parities,
        max_generators=options.max_generators,
        max_expanded_terms=options.max_expanded_terms,
    )
    result.prepare_reference(occupied_modes)
    if options.name != "bksf_graph":
        result.metadata.update(rows=options.rows, columns=options.columns)
    elif not canonical:
        result.metadata["scalar_register_padding"] = 1
    return result
