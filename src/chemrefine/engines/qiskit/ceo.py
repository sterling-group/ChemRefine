"""Generalized qubit exchanges and coupled-exchange synthesis in occupation encoding.

The OVP/MVP constructions and 9/13-CX networks follow Figures 4 and 9 of
Ramôa et al., npj Quantum Information 11, 86 (2025),
https://doi.org/10.1038/s41534-025-01039-4 . Gates are expressed for the owned
Hermitian convention U(theta)=exp(-i theta G), including their global phases.
The optimized networks are independently checked against complete matrix exponentials.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any

import numpy as np

from chemrefine.engines.qiskit.context import AnsatzArtifacts, ElectronicStructureContext
from chemrefine.errors import ConfigError


def pauli_support(operator: Any) -> frozenset[int]:
    """Return the union of actual nonidentity qubits in a simplified mapped operator."""
    operator = operator.simplify(atol=1e-12)
    present = np.asarray(operator.paulis.x) | np.asarray(operator.paulis.z)
    present = present[np.abs(operator.coeffs) > 1e-12]
    return frozenset(int(index) for index in np.flatnonzero(np.any(present, axis=0)))


def qubit_exchange(
    source: tuple[int, ...], target: tuple[int, ...], num_qubits: int, *, imaginary: bool = False
) -> Any:
    """Build a qubit-ladder exchange, preserving N and Ms for spin-matched indices.

    Qubit ladder operators have no Jordan-Wigner parity strings. These are
    variational generators in occupation encoding, not fermionic ladder operators.
    The optional symmetric quadrature permits complex variational amplitudes.
    """
    from qiskit.quantum_info import SparsePauliOp

    term = SparsePauliOp("I" * num_qubits)
    for index, creation in [*((i, True) for i in target), *((i, False) for i in source)]:
        ladder = SparsePauliOp.from_sparse_list(
            [("X", [index], 0.5), ("Y", [index], -0.5j if creation else 0.5j)],
            num_qubits=num_qubits,
        )
        term = term @ ladder
    return ((term + term.adjoint()) if imaginary else 1j * (term - term.adjoint())).simplify()


def _occupation_transform(context: ElectronicStructureContext, operator: Any) -> Any:
    """Transform occupation Pauli operators only through a declared JW-compatible mapper."""
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    if isinstance(context.mapper, JordanWignerMapper):
        if context.num_qubits != 2 * context.num_spatial_orbitals:
            raise ConfigError("qubit-exchange pool requires an unreduced occupation register")
        return operator
    transform = getattr(context.mapper, "chemrefine_tapering", None)
    if transform is not None and isinstance(
        getattr(context.mapper, "mapper", None), JordanWignerMapper
    ):
        return transform.map_operator(operator)
    raise ConfigError(
        "qubit-exchange/CEO pools require Jordan-Wigner or a tapered Jordan-Wigner mapper"
    )


def exchange_pool(
    context: ElectronicStructureContext,
    *,
    coupled: bool,
    include_imaginary: bool,
    max_pool_size: int,
) -> AnsatzArtifacts:
    """Enumerate generalized same-spin singles and N/Ms-conserving double exchanges.

    Four same-spin orbitals have three distinct exchanges; two alpha and two beta
    orbitals have two. Each pair of same-support exchanges contributes sum and
    difference OVP candidates. Parent generators remain separately addressable for
    MVP selection. Symmetric and antisymmetric quadratures are never grouped together.
    """
    n = context.num_spatial_orbitals
    singles = 2 * comb(n, 2)
    doubles = 6 * comb(n, 4) + 2 * comb(n, 2) ** 2
    ovps = 12 * comb(n, 4) + 2 * comb(n, 2) ** 2 if coupled else 0
    upper_bound = (singles + doubles + ovps) * (2 if include_imaginary else 1)
    if upper_bound > max_pool_size:
        raise ConfigError("qubit-exchange/CEO pool exceeds max_pool_size before generation")
    width = 2 * n
    operators: list[Any] = []
    metadata: list[dict[str, Any]] = []
    groups: list[list[tuple[tuple[int, ...], tuple[int, ...]]]] = [
        [((a,), (b,))] for spin in (0, n) for a, b in combinations(range(spin, spin + n), 2)
    ]
    source: tuple[int, ...]
    for support in combinations(range(width), 4):
        exchanges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for source in combinations(support, 2):
            target = tuple(i for i in support if i not in source)
            if source < target and sum(i < n for i in source) == sum(i < n for i in target):
                exchanges.append((source, target))
        if exchanges:
            groups.append(exchanges)
    for imaginary in (False, True) if include_imaginary else (False,):
        for group in groups:
            parents = []
            for source, target in group:
                original = qubit_exchange(source, target, width, imaginary=imaginary)
                mapped = _occupation_transform(context, original)
                if mapped is None or not pauli_support(mapped):
                    continue
                index = len(operators)
                operators.append(mapped)
                parents.append(index)
                metadata.append(
                    {
                        "pool_index": index,
                        "label": f"qe_{source}_to_{target}",
                        "family": "ceo" if coupled else "qe",
                        "role": "qe",
                        "source": source,
                        "target": target,
                        "quadrature": "symmetric" if imaginary else "antisymmetric",
                        "original_support": sorted((*source, *target)),
                        "support": sorted(pauli_support(mapped)),
                        "candidate": not coupled or len(source) == 1,
                    }
                )
            for index in parents:
                metadata[index]["group_indices"] = parents.copy()
            if not coupled or len(group[0][0]) == 1:
                continue
            if len(parents) == 1:
                # Projection can leave one allowed exchange; its own QE remains usable.
                metadata[parents[0]]["candidate"] = True
            for left, right in combinations(parents, 2):
                for sign in (1, -1):
                    generator = (operators[left] + sign * operators[right]).simplify(atol=1e-12)
                    if not pauli_support(generator):
                        continue
                    index = len(operators)
                    operators.append(generator)
                    metadata.append(
                        {
                            "pool_index": index,
                            "label": f"ovp_{left}_{sign:+}_{right}",
                            "family": "ceo",
                            "role": "ovp",
                            "parents": [left, right],
                            "weights": [1, sign],
                            "group_indices": parents.copy(),
                            "quadrature": "symmetric" if imaginary else "antisymmetric",
                            "original_support": metadata[left]["original_support"],
                            "support": sorted(pauli_support(generator)),
                            "candidate": True,
                        }
                    )
    if not operators:
        raise ConfigError("qubit-exchange/CEO pool is empty in the selected register/sector")
    return AnsatzArtifacts(operator_pool=tuple(operators), pool_metadata=tuple(metadata))


def single_exchange_circuit(num_qubits: int, source: int, target: int, theta: Any) -> Any:
    """Implement an antisymmetric single excitation with exactly two CX gates."""
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(num_qubits)
    for qubit in (source, target):
        if qubit == source:
            circuit.rz(np.pi / 2, qubit)
        circuit.rx(np.pi / 2, qubit)
    circuit.cx(source, target)
    circuit.rx(theta, source)
    circuit.rz(theta, target)
    circuit.cx(source, target)
    circuit.rx(-np.pi / 2, target)
    circuit.rx(-np.pi / 2, source)
    circuit.rz(-np.pi / 2, source)
    return circuit


def ovp_circuit(
    num_qubits: int, first: dict[str, Any], second: dict[str, Any], sign: int, theta: Any
) -> Any:
    """Implement a canonical sum/difference pair with nine CX gates (paper Figure 9)."""
    from qiskit import QuantumCircuit

    common_source = set(first["source"]) & set(second["source"])
    common_target = set(first["target"]) & set(second["target"])
    if len(common_source) != 1 or len(common_target) != 1 or sign not in (-1, 1):
        raise ConfigError(
            "OVP synthesis requires two canonical exchanges with one common source/target"
        )
    c = next(iter(common_source))
    d = next(iter(common_target))
    a = next(iter(set(first["source"]) - {c}))
    b = next(iter(set(first["target"]) - {d}))
    angle = -theta  # Hermitian G = i(T-T†), U = exp(-i theta G).
    if sign == 1:
        angle = -angle
        a, b, c, d = d, c, b, a
    circuit = QuantumCircuit(num_qubits, global_phase=-np.pi / 4)
    for control, target in ((a, b), (c, d), (a, c)):
        circuit.cx(control, target)
    circuit.h(b)
    circuit.h(d)
    for direction, target in ((1, d), (-1, b), (1, d), (-1, c)):
        circuit.ry(direction * angle / 2, a)
        circuit.cx(a, target)
    circuit.h(d)
    circuit.rz(np.pi / 2, a)
    circuit.ry(-np.pi / 2, b)
    circuit.rz(-np.pi / 2, b)
    circuit.cx(a, b)
    circuit.cx(c, d)
    circuit.rz(-np.pi / 2, b)
    return circuit


def mvp_circuit(num_qubits: int, generators: list[Any], parameters: list[Any]) -> Any:
    """Implement a four-qubit commuting odd-Y Pauli block with thirteen CX gates.

    Each generator may carry its own independent variational parameter. The eight
    coefficient combinations determine the rotation angles in paper Figure 4.
    """
    from qiskit import QuantumCircuit

    support = sorted(pauli_support(generators[0]))
    if len(support) != 4 or any(pauli_support(op) != frozenset(support) for op in generators):
        raise ConfigError("MVP synthesis requires four-qubit generators with identical support")
    strings = ("YXXX", "XYXX", "YYXY", "XXXY", "YXYY", "XYYY", "YYYX", "XXYX")
    coefficients: dict[str, Any] = dict.fromkeys(strings, 0)
    for operator, parameter in zip(generators, parameters, strict=True):
        for label, coefficient in operator.to_list():
            local = "".join(label[-1 - qubit] for qubit in support)
            if local not in coefficients or abs(coefficient.imag) > 1e-12:
                raise ConfigError("MVP synthesis requires real coefficients of odd-Y Pauli strings")
            coefficients[local] += float(coefficient.real) * parameter
    a, b, c, d = support
    circuit = QuantumCircuit(num_qubits, global_phase=-np.pi / 4)
    circuit.rz(-np.pi / 2, a)
    for target in (d, c, b):
        circuit.cx(a, target)
    circuit.h(a)
    for index, (string, sign) in enumerate(zip(strings, (1, 1, -1, 1, -1, -1, -1, 1), strict=True)):
        circuit.rz(2 * sign * coefficients[string], a)
        if index == 3:
            circuit.rz(-np.pi / 2, c)
        if index < 7:
            circuit.cx((b, d, b, c, b, d, b)[index], a)
    circuit.h(a)
    for target in (b, c, d):
        circuit.cx(a, target)
    circuit.rz(np.pi / 2, c)
    return circuit
