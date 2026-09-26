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
The shipped molecular worker writes a `<job>.states.json` descriptor and its
integrity-checked NPZ payload, including any accumulated orbital rotation.
`save_states(path, result.states)` and `load_states(path)` provide the same
versioned format in Python. Paths inside descriptors are relative; amplitudes
are complex numeric arrays and determinant integers use unsigned 64-bit limbs.
Missing or corrupt referenced states invalidate a cached molecular result.
Calling `run_problem` does not write files; `run_job(..., artifact_dir=...)`
explicitly enables persistence at the pipeline boundary.

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

## Optimize active orbitals

For explicit SQD and SqDRIFT, set `orbital_optimization: {}` to alternate orbital
minimization at fixed CI coefficients with rediagonalization of the same
configuration pool. Controls include `max_iterations`,
`max_optimizer_iterations`, `energy_tolerance`, `gradient_tolerance`,
`complex_rotations`, and `spin_mode: shared` or `independent`. Molecular workflows
preserve alpha/beta populations. The Python `optimize_orbitals` API additionally
supports unrestricted spin-orbital rotations when that sector interpretation is
intended.

Only improvements exceeding the energy tolerance are accepted. The result records
a monotone target-root energy history, optimizer evaluation count, convergence,
and the accumulated unitary rotation. Numerical states persist that rotation.
Their `rdms()` and `expectation()` methods default to the original active-orbital
basis; `basis="state"` selects optimized coordinates. Transition RDMs require a
common orbital frame. Frozen and inactive orbitals remain outside the optimization.
`rdms.natural_occupations()` and `rdms.number_correlations()` expose occupation
spectra and `<n_p n_q>` with the correct diagonal.

## Sample Krylov powers

`algorithm: skqd` samples the reference and successive powers of one fixed
approximate evolution circuit. Set `time_step`, `num_steps`, `product_formula`
(`lie` or `suzuki`), `suzuki_order` (2, 4, or 6), and `repetitions`. The circuit at
power zero is always included. All powers repeat the same synthesized step;
the final explicit determinant solve uses the original Hamiltonian, so product
formula error changes the sampled subspace rather than the reported operator.
The ordinary sampler, initial-state selection, shot budgets, recovery controls,
root controls, and orbital optimization remain available.

## Expand sampled states for excitations

`algorithm: extended_sqd` applies occupied-to-virtual excitations defined by the
actual reference occupations to a sampled SQD ground state, then diagonalizes
the explicit union. It follows the determinant expansion in
[extended SQD](https://arxiv.org/html/2411.00468v1). No alpha/beta Cartesian closure
is introduced. `excitation_ranks` defaults to `[1, 2]` and also accepts rank 3;
`minimum_probability` optionally excludes small source coefficients from the
expansion. Excitation-pool size, generated configurations, and final subspace
size have independent bounds that fail explicitly when exceeded.

Physical shot counts retain the source circuit's acquisition totals; generated
basis elements are reported separately. This method defaults to
`spin_constraint: report`, exposing roots across total-spin sectors at fixed
alpha/beta population. Root zero remains the scalar molecular-energy default.
