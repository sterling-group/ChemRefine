# Sampled quantum states and observables

The `sqd`, `sqdrift`, `skqd` and `extended_sqd` molecular algorithms retain the
determinant basis and complex amplitudes of their returned subspace states.
These states describe the
projected quantum-sampled subspace; their RDMs are not direct experimental
measurements or guarantees of exact electronic structure.

Install the `qiskit-fermionic` backend for these algorithms. They all require
the untapered `jordan_wigner` mapper. Supplied-count SQD and extended SQD run on
CPU without submitting samples; acquisition uses the selected sampler and can
use supported Aer CUDA execution. None of these methods consumes an estimator
or optimizer, so leave those components at their defaults.

| Algorithm | Sampling input | Consumed preparation components |
| --- | --- | --- |
| `sqd` | Positive physical `counts`, or one fixed ansatz circuit. | With acquisition: `ansatz`, `initial_state`, `initial_point`, `sampler`. Supplied counts leave all four at their defaults. |
| `sqdrift` | Grouped-qDRIFT circuits from the shared real spatial integrals. | `sampler`; it uses the actual molecular reference and does not consume `ansatz`, `initial_state` or `initial_point`. |
| `skqd` | Reference plus powers of one fixed evolution step. | `initial_state`, `sampler`; leave `ansatz` and `initial_point` at their defaults. |
| `extended_sqd` | SQD counts/acquisition followed by explicit excitation expansion. | The same acquisition components as SQD. Expansion uses actual reference occupations. |

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

The Cartesian projector supports a single spin-validated root and the released
real shared-orbital PySCF/SQD domain. Multiple roots, `spin_constraint: report`
and orbital optimization require `projection: explicit`. Cartesian processing
also requires at least one alpha and one beta electron and at most 63 spatial
orbitals. Explicit supplied-count processing supports the broader
complex/unrestricted integral domain and sectors with an empty spin population.

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

State bundles preserve the final determinant basis, amplitudes and orbital
rotation. Molecular acquisition metadata records budgets, seeds and circuit
summaries; it does not archive the raw sampled count dictionary or each
acquisition circuit. Save those separately in a custom acquisition workflow if
you need to repeat the subspace selection exactly. Supplied counts remain in
the resolved algorithm options. These algorithms do not support
`circuit_export`; their retained CI states do not provide a compiled quantum
state-preparation circuit. See [checkpoint boundaries](qiskit-checkpoints.md#archive-and-recovery-boundaries).

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

`skqd` defaults to `time_step: 0.1`, `num_steps: 4`, `product_formula: suzuki`,
`suzuki_order: 2`, `repetitions: 1` and `max_circuits: 256`. It always uses the
explicit projector. There are `num_steps + 1` circuits, each with `shots` shots;
that total must fit `max_total_shots`. With `product_formula: lie`, leave
`suzuki_order` at its default because that option is unused by the Lie formula.

```yaml
algorithm:
  name: skqd
  options:
    time_step: 0.1
    num_steps: 4
    shots: 4096
    configuration_recovery: false
    samples_per_batch: 100
    num_batches: 1
initial_state: hartree_fock
sampler:
  name: statevector
  options: {seed: 7}
```

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

`max_excitation_operators` (default 10000) bounds the excitation pool,
`max_generated_determinants` (100000) bounds the expanded configuration union,
and the inherited `max_subspace_dimension` (10000) bounds the final solve.
`max_total_diagonalizations` accounts for the additional expansion solve.

```yaml
algorithm:
  name: extended_sqd
  options:
    counts: {"0101": 20, "1010": 20}
    configuration_recovery: false
    samples_per_batch: 2
    num_batches: 1
    excitation_ranks: [1, 2]
    num_roots: 2
    target_root: 1
```

The snippets are molecular step `options` fragments for a compatible supplied
or prepared problem. Keep the integral bundle, resolved configuration and full
state descriptor/payload together for replay. Loading a state supports later
property analysis; it does not resume sampling, configuration recovery or an
interrupted orbital optimizer.
