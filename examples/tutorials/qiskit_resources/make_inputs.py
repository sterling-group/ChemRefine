"""Write exactly factorable toy electronic integrals, without a classical SCF run."""

from pathlib import Path

import numpy as np

from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.factorized_resources import THCFactors, save_thc_factors
from chemrefine.engines.qiskit.integral_io import save_integrals

directory = Path(__file__).resolve().parent / "inputs"
directory.mkdir(exist_ok=True)
leaf = np.random.default_rng(81).normal(size=(32, 8))
leaf /= np.linalg.norm(leaf, axis=1)[:, None]
central = np.diag(np.linspace(0.1, 0.6, 32))
eri = np.einsum("Pp,Pq,PQ,Qr,Qs->pqrs", leaf, leaf, central, leaf, leaf)
data = ElectronicStructureData(
    4,
    4,
    8,
    np.diag(np.linspace(-1.0, -0.25, 8)),
    eri,
    provenance={"model": "eight spatial orbitals with exact rank-32 repulsive THC interactions"},
)
save_integrals(directory / "integrals.json", data)
save_thc_factors(directory / "thc.json", THCFactors(leaf, central))
