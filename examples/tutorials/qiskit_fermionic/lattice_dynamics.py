"""Compare bounded Hubbard evolution under three occupation encodings."""

import json

from chemrefine.engines.qiskit.api import (
    LatticeDynamicsOptions,
    chain_lattice,
    simulate_lattice_dynamics,
)

model = chain_lattice(2, onsite_interaction=2.0)
for mapping in ("jordan_wigner", "bravyi_kitaev", "parity"):
    result = simulate_lattice_dynamics(
        model,
        occupied_modes=[0, 2],
        options=LatticeDynamicsOptions(
            time=0.4, steps=4, order=4, mapping=mapping, exact_reference=True
        ),
    )
    print(json.dumps(result.as_dict(), indent=2))
