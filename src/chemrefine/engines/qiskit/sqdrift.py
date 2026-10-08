"""Seeded, number-conserving SqDRIFT circuits using Qiskit Fermions."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from chemrefine.errors import ConfigError

if TYPE_CHECKING:
    from chemrefine.engines.qiskit.fermionic import FermionicIntegrals


def sqdrift_circuits(
    data: FermionicIntegrals,
    *,
    times: Sequence[float],
    num_groups: int,
    randomizations: int,
    seed: int | None,
) -> Iterator[tuple[Any, dict[str, Any]]]:
    """Yield independent grouped-qDRIFT realizations in canonical JW mode order.

    Diagonal terms are retained: phases accumulated between excitations can affect
    subsequent occupation probabilities. No state-dependent rejection sampling or
    mode relabeling is applied. Each circuit starts from the prepared problem's
    declared alpha/beta occupations.
    """
    from qiskit_fermions.circuit import FermionicCircuit
    from qiskit_fermions.circuit.library import Evolution, InitializeModes
    from qiskit_fermions.operators.terms.grouping import group_terms_by_electronic_structure
    from qiskit_fermions.operators.terms.ordering import canonical_order
    from qiskit_fermions.transpiler import FermionicPassManager
    from qiskit_fermions.transpiler.passes import QDriftTrotterization
    from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager

    from chemrefine.engines.qiskit.fermionic import to_fermion_operator

    modes = 2 * data.norb
    operator: Any = canonical_order(to_fermion_operator(data).normal_ordered().simplify(atol=0))
    group_terms_by_electronic_structure(operator, modes, two_body_physicist_order=False)
    coefficients = np.asarray(operator.get_coeffs())
    if not np.all(np.isfinite(coefficients)):
        raise ConfigError("SqDRIFT Hamiltonian contains non-finite coefficients")
    nonzero = bool(np.any(np.abs(coefficients) > 0))
    occupation = np.zeros(modes, dtype=bool)
    occupation[list(data.occupations[0])] = True
    occupation[[data.norb + i for i in data.occupations[1]]] = True
    rng = np.random.default_rng(seed)
    for time_index, time in enumerate(times):
        for realization in range(randomizations):
            circuit_seed = int(rng.integers(0, 2**31 - 1))
            circuit = FermionicCircuit(modes)
            circuit.append(InitializeModes(occupation.tolist()), circuit.modes)
            if nonzero and time != 0:
                circuit.append(Evolution(modes, operator, time=float(time)), circuit.modes)
            manager = generate_preset_jw_pass_manager(
                optimization_level=1, seed_transpiler=circuit_seed
            )
            manager.optimization = FermionicPassManager(
                [QDriftTrotterization(num_groups, rng=circuit_seed)]
            )
            yield (
                manager.run(circuit),
                {
                    "time_index": time_index,
                    "time": float(time),
                    "randomization": realization,
                    "seed": circuit_seed,
                    "num_groups": num_groups if nonzero and time != 0 else 0,
                },
            )
