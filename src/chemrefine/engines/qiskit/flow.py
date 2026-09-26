"""Commuting directed transfer groups and verified Clifford evolution synthesis."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from chemrefine.engines.qiskit.encodings import _rref, _vector
from chemrefine.errors import ConfigError


def partition_flow_edges(
    edges: Sequence[tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Greedily color directed edges into matchings between source and target copies.

    Each color has at most one incoming and one outgoing edge at every vertex, so its
    components are directed paths or cycles. This deterministic coloring need not be
    optimal; no constant number of colors is promised for unbounded-degree graphs.
    """
    if len(set(edges)) != len(edges) or any(first == second for first, second in edges):
        raise ConfigError("flow edges must be distinct directed pairs of different modes")
    groups: list[list[tuple[int, int]]] = []
    for edge in sorted(edges):
        group = next(
            (
                group
                for group in groups
                if all(edge[0] != other[0] and edge[1] != other[1] for other in group)
            ),
            None,
        )
        if group is None:
            group = []
            groups.append(group)
        group.append(edge)
    return tuple(tuple(group) for group in groups)


def commuting_evolution(operator: Any, time: float, *, diagonalizer: Any = None) -> Any:
    """Exactly synthesize a commuting Pauli sum through a checked Clifford basis change.

    Dependent Paulis become multi-Z rotations rather than being incorrectly assigned
    independent qubits. A supplied diagonalizer may instead yield weight-one X/Y/Z
    images, as in square VC flow sets. Its images are checked before any circuit is used.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford, SparsePauliOp
    from qiskit.synthesis import synth_circuit_from_stabilizers

    if not np.all(np.isfinite(operator.coeffs)):
        raise ConfigError("commuting evolution coefficients must be finite")
    if not np.isfinite(time) or np.any(np.abs(operator.coeffs.imag) > 1e-10):
        raise ConfigError("commuting evolution requires finite time and a Hermitian Pauli sum")
    operator = operator.simplify()
    if len(operator.group_commuting()) > 1:
        raise ConfigError("flow-set synthesis requires mutually commuting Pauli terms")
    width = operator.num_qubits
    if diagonalizer is None:
        selected = []
        vectors: list[np.ndarray] = []
        for pauli in operator.paulis:
            vector = _vector(SparsePauliOp(pauli)).astype(np.uint8)
            candidate = [*vectors, vector]
            rank = len(_rref(np.array(candidate))[1])
            if rank > len(vectors):
                selected.append(pauli.to_label())
                vectors.append(vector)
        diagonalizer = (
            synth_circuit_from_stabilizers(selected, allow_underconstrained=True).inverse()
            if selected
            else QuantumCircuit(width)
        )
    if diagonalizer.num_qubits != width:
        raise ConfigError("flow diagonalizer has the wrong qubit width")
    clifford = Clifford(diagonalizer)
    circuit = diagonalizer.copy()
    for pauli, coefficient in zip(operator.paulis, operator.coeffs, strict=True):
        image = SparsePauliOp(pauli.evolve(clifford, frame="s"), coeffs=[coefficient])
        image = image.simplify()
        transformed = image.paulis[0]
        angle = 2 * time * float(image.coeffs[0].real)
        support = np.flatnonzero(transformed.x | transformed.z).tolist()
        if not support:
            circuit.global_phase -= angle / 2
        elif len(support) == 1:
            qubit = support[0]
            if transformed.x[qubit] and transformed.z[qubit]:
                circuit.ry(angle, qubit)
            elif transformed.x[qubit]:
                circuit.rx(angle, qubit)
            else:
                circuit.rz(angle, qubit)
        else:
            if np.any(transformed.x):
                raise ConfigError(
                    "flow diagonalizer did not produce single-qubit or diagonal images"
                )
            target = support[-1]
            for control in support[:-1]:
                circuit.cx(control, target)
            circuit.rz(angle, target)
            for control in reversed(support[:-1]):
                circuit.cx(control, target)
    return circuit.compose(diagonalizer.inverse())


def vc_flow_diagonalizer(rows: int, columns: int, species: int, direction: str) -> Any:
    """Construct the open-square VC two-entangling-layer change of basis.

    The native register is all physical modes followed by one auxiliary per mode.
    The caller verifies every conjugated Pauli with ``commuting_evolution``. These
    layers presume the abstract register connectivity, before hardware routing.
    """
    from qiskit import QuantumCircuit

    if min(rows, columns, species) < 1 or direction not in {"east", "west", "south", "north"}:
        raise ConfigError(
            "VC flow diagonalizer requires positive dimensions and a cardinal direction"
        )
    sites = rows * columns
    modes = species * sites
    circuit = QuantumCircuit(2 * modes)
    horizontal = direction in {"east", "west"}
    lines = [
        [
            offset + row * columns + column
            for row, column in (
                ((index, slot) if horizontal else (slot, index))
                for slot in range(columns if horizontal else rows)
            )
        ]
        for offset in range(0, modes, sites)
        for index in range(rows if horizontal else columns)
    ]
    for line in lines:
        for position, mode in enumerate(line):
            auxiliary = modes + mode
            if horizontal:
                if direction == "west":
                    circuit.s(mode)
                circuit.h(auxiliary)
            elif direction == "south":
                if position % 2 == 0:
                    circuit.sdg(auxiliary)
                    circuit.h(auxiliary)
                else:
                    circuit.h(mode)
                    circuit.s(mode)
                    circuit.h(auxiliary)
                    circuit.s(auxiliary)
            elif position % 2 == 0:
                circuit.h(auxiliary)
                circuit.s(auxiliary)
            else:
                circuit.h(mode)
                circuit.s(mode)
                circuit.sdg(auxiliary)
                circuit.h(auxiliary)
    for line in lines:
        for mode in line[:-1] if horizontal else line:
            circuit.cx(modes + mode, mode)
    for line in lines:
        for position, (first, second) in enumerate(pairwise(line)):
            if horizontal:
                circuit.cx(second, modes + first)
            elif (position % 2 == 0) == (direction == "south"):
                circuit.cx(modes + first, modes + second)
            else:
                source, target = (first, second) if direction == "south" else (second, first)
                circuit.cx(source, target)
    return circuit
