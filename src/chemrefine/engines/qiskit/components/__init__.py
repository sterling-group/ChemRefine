"""Built-in modular Qiskit components.

Importing this package registers lightweight builders. The builders themselves import
Qiskit only when invoked inside the optional backend environment.
"""

from chemrefine.engines.qiskit.components import (
    adaptive,
    algorithms,
    ansatze,
    ansatze_extended,
    estimators,
    fermionic_algorithms,
    initial_points,
    initial_states,
    initial_states_extended,
    krylov,
    mappers,
    optimizers,
    optimizers_extended,
    qnspsa,
    samplers,
    spectra,
    subspace_algorithms,
)

__all__ = [
    "adaptive",
    "algorithms",
    "ansatze",
    "ansatze_extended",
    "estimators",
    "fermionic_algorithms",
    "initial_points",
    "initial_states",
    "initial_states_extended",
    "krylov",
    "mappers",
    "optimizers",
    "optimizers_extended",
    "qnspsa",
    "samplers",
    "spectra",
    "subspace_algorithms",
]
