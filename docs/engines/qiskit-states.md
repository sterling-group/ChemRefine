# Sampled quantum states and observables

The `sqd` and `sqdrift` molecular algorithms retain the determinant basis and
complex amplitudes of their selected subspace states. These states describe the
projected quantum-sampled subspace; their RDMs are not direct experimental
measurements or guarantees of exact electronic structure.

## Select a projection and excited root

Existing configurations retain the released SQD Cartesian alpha/beta solver.
Select `projection: explicit` to diagonalize precisely the sampled determinants,
without adding the Cartesian combinations of their alpha and beta halves:

```yaml
algorithm:
  name: sqd
  options:
    projection: explicit
    counts: {"0101": 10, "0110": 10, "1001": 10, "1010": 10}
    configuration_recovery: false
    samples_per_batch: 4
    num_batches: 1
    num_roots: 4
    target_root: 2
    spin_constraint: report
```

Bitstrings are most-significant-first, with beta modes on the left and alpha on
the right. Supplied counts are positive integer frequencies. The explicit
projector uses Python integers, so it does not narrow determinants to 64 bits.
Simulation, RDM allocation, and eigensolver memory budgets still apply.

`num_roots` counts roots in the fixed alpha/beta particle sector. `target_root`
selects the scalar molecular energy, starting at zero. Nuclear and inactive-space
constants are restored once for every root. The default `spin_constraint: require`
checks every returned root against the requested total spin and rejects a
spin-contaminated state. `report` permits different total-spin eigenstates or
mixtures in the fixed magnetization sector and reports their spin expectations
and residuals explicitly; it does not claim to filter a total-spin sector.

Configuration recovery uses the released SQD recovery kernel with weighted
sampling of explicit determinants. Seeds, sampling budgets, and iteration records
remain available. Small projected matrices use dense diagonalization; larger
ones use a matrix-free Hermitian eigensolver with residual validation.

## Use retained states

The Python result exposes `result.states`, one immutable `DeterminantState` per
root, and `root_energies_hartree`, `root_electronic_energies_hartree`, and
`root_total_energies_hartree`. Numerical states are deliberately excluded from
`result.as_dict()` and are persisted by the worker as referenced array artifacts.

```python
state = result.states[result.target_root]
rdms = state.rdms(max_order=2, max_memory_mb=512)
spin_blocks = rdms.spin_blocks()      # alpha-then-beta spatial pairing
spin_sum = rdms.spin_summed()
transition = result.states[1].rdms(bra=result.states[0])
```

The explicit conventions are `gamma[p,q] = <a†p aq>` and
`Gamma[p,q,r,s] = <a†p a†q a_s a_r>`. Transition RDMs use
`<bra|O|state>`. Spin summation assumes the caller intends to pair alpha and beta
spatial indices; unrestricted orbitals need the appropriate cross-orbital
integrals when evaluating spatial observables. Dense RDMs have their own memory
check before allocation.

`FermionicHamiltonian` and `FermionTerm` support sparse complex Hermitian,
number-conserving one- and two-body observables. Term tuples are in written
operator order and coefficients already contain any integral prefactor.
`state.expectation(operator)` contracts the actual state without a full Fock
matrix. Dipoles and other physical observables require their own supplied
integrals; an energy Hamiltonian cannot supply missing property integrals.

The molecular integral API accepts complex Hermitian tensors and unrestricted
alpha/beta blocks. Its existing Nature spin-observable boundary still requires a
real alpha-beta orbital overlap. The released ffsim/Cartesian SQD and grouped
SqDRIFT circuit constructors retain their real shared-spatial-orbital domain;
unsupported inputs fail explicitly. General supplied-count SQD uses the explicit
projector instead.
