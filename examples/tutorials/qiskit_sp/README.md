# Qiskit electronic-structure examples

Install `chemrefine[qiskit]`, then run from this directory:

```bash
chemrefine run compare.yaml --dry-run
chemrefine run compare.yaml
```

`compare.yaml` prepares the shipped H2 geometry once with PySCF/STO-3G,
keeps explicit spatial orbitals `[0, 1]`, and uses Jordan–Wigner mapping.
The template solves the same problem exactly, with UCCSD-VQE, and with
UCCSD-pool ADAPT-VQE. Statevector expectations, zero initial parameters,
SLSQP tolerances, and seeds are explicit in the configuration.

The job log prints each total energy, variational errors relative to exact,
qubits, Pauli terms, UCCSD parameters, and the selected ADAPT operator count.
The H2 exact energy is approximately `-1.1373060358` hartree at 0.735 Å;
the deterministic variational calculations should agree within `1e-7` hartree.

A validated run with the supported stack produced:

| Solver | Total energy (hartree) | Error from exact (hartree) |
| --- | ---: | ---: |
| Exact | -1.1373060357534004 | — |
| UCCSD-VQE | -1.1373060357533697 | 3.1e-14 |
| ADAPT-VQE | -1.1373060357533902 | 1.0e-14 |

The mapped Hamiltonian has four qubits and 15 Pauli terms. UCCSD has three
parameters; ADAPT retains the double excitation `((0, 2), (1, 3))` from its
three-operator pool. Its second gradient check satisfies the `1e-6` threshold.
The example uses SLSQP `ftol: 1e-12` so the inner optimization resolves that
gradient test; a looser energy tolerance can stop with a repeated ADAPT
candidate even when the energy error is already very small.

`outputs_compare/steps.csv` records the final ADAPT energy. The raw
per-structure JSON stores all three results under
`engine_metadata.comparison`, including explicit energy units, circuit
metrics, evaluation histories, and ADAPT operator identities. The
normalized `.result.json` follows ChemRefine's usual result contract.

`input.yaml` runs just UCCSD-VQE through the thin `step1.py` template.
`aer_statevector.yaml` and `aer_shots.yaml` demonstrate the existing optional
local Aer estimators and require `chemrefine[qiskit-aer]`.

The [engine documentation](../../../docs/engines/qiskit.md) also shows the
integral-input API, explicit active spaces, and external excitation/operator
pools. The integral API can use `chemrefine[qiskit-core]` without PySCF.
