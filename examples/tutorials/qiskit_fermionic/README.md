# Fermionic research examples

Install `chemrefine[qiskit-fermionic]` or use
`chemrefine backends install qiskit-fermionic`. These examples use local simulators.
From this directory:

```bash
chemrefine run input.yaml --dry-run
chemrefine run input.yaml
chemrefine run lucj.yaml
chemrefine run sqdrift.yaml
python lattice_dynamics.py
```

`input.yaml` samples a fixed UCCSD parameter vector and runs SQD. It does not
optimize that vector first. `lucj.yaml` optimizes a numerically initialized LUCJ
state with ffsim. `sqdrift.yaml` samples grouped fermionic randomized evolution
and applies SQD to the original Hamiltonian. Independent algorithm and sampler
seeds are explicit. Finite samples can miss important determinants; a successful
run is not a certificate of chemical accuracy.

The H₂/STO-3G reference is approximately −1.1373060358 hartree. SQD's classical
subspace diagonalization is part of the hybrid quantum algorithm. These examples
do not add a standalone classical FCI, CASSCF, coupled-cluster, or DMRG workflow.

`lattice_dynamics.py` prints Hubbard-model observables and exact-evolution
fidelity for three supported encodings. Its units follow the model Hamiltonian,
and its result is separate from molecular single-point energies.
