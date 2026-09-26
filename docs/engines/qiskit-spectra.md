# Molecular spectra and quantum natural SPSA

The `qiskit` molecular engine provides experimental `vqd` and `qeom` algorithms
through the same prepared problem, mapper, estimator, optimizer, and result
contracts as VQE. Both accept real or complex molecular Hamiltonians, including
explicit alpha/beta integral blocks supported by the preparation layer. No
classical SCF calculation is needed when using `run_problem` with supplied
integrals. The XYZ pipeline examples use the existing PySCF preparation adapter.

These methods use the core Qiskit backend. Selecting an Aer estimator or sampler
adds the declared Aer provider requirement. The engine opens every consumed
estimator and sampler and closes both if construction, optimization, measurement,
or a user callback fails. Providers are never replaced by an implicit statevector
calculation.

## Root and energy conventions

`target_root` is zero-based. `QiskitRunResult.energy_hartree` is the selected root,
while `root_energies_hartree` contains all returned roots. The same root index selects
`root_electronic_energies_hartree` and, when a nuclear constant exists,
`root_total_energies_hartree`. Frozen/inactive and nuclear constants are restored
exactly once for every root. Optimizer callbacks contain active-Hamiltonian
objectives and exclude these constants.

Both methods check particle number and spin projection using first **and second**
moments. The default tolerances suit ideal estimators; finite-shot studies must
choose tolerances justified by their measurement uncertainty. Total spin `S²`
means and variances are reported for each root. `target_s2` optionally constrains
all returned roots with `spin_tolerance`; it is unset by default because legitimate
excited states can have a different total spin from the ground state. A mixture of
singlet and triplet components can have a nonintegral mean `S²`.

Solver diagnostics live under `result.metadata["solver"]` and carry
`experimental: true`. Numerical completion does not establish chemical accuracy or
a global variational optimum; `converged` remains unset when the provider optimizer
does not supply enough information to make that claim.

Available optimizer `success`, `status`, `message`, and evaluation/iteration
counts are retained as `optimizer_termination` for VQD roots and
`reference_optimizer_termination` for qEOM. Their convergence verdict describes
optimizer stopping only. Missing success information remains unknown.

### Optional physical Hamiltonian residuals

Set `measure_residuals: true` to additionally measure the active Hamiltonian's
second moment with the selected estimator. The result retains the signed
variance `v = <H²> - <H>²` and, when nonnegative, its square root in hartree.
For an exact state expectation this is `||(H - <H>)|psi>||`; scalar nuclear and
inactive-space offsets leave it unchanged. The default is `false`, so existing
configurations perform no additional second-moment measurements.

VQD reports `root_hamiltonian_residuals` in physical-energy root order. qEOM
reports `reconstructed_root_hamiltonian_residuals`, including its reference,
around each reconstructed state's **Rayleigh energy**. These physical residuals
are distinct from `generalized_eigenpair_residuals`, which test the response
matrix equation. A precise solution of an approximate response problem can
still have substantial physical residuals.

Finite-shot estimates of `v` can be negative. Such results retain the signed
variance, report a null residual, and distinguish negative values within the
configured numerical tolerance from those beyond it. They are never silently
reported as zero residual. Neither the tolerance nor a small positive estimate
is a statistical confidence bound; no residual uncertainty is inferred from
independently measured moments. These diagnostics do not change the reported
energy or certify a particular excited-root index.

| Control | Default | Knob verdict |
| --- | --- | --- |
| `measure_residuals` | `false` | Both runnable spectral examples enable it; tests verify no default extra publications and independent dense-Hamiltonian residuals. |
| `max_residual_product_terms` | `1000000` | Bounds H² product terms before multiplication and after normal ordering; both examples declare it and guard tests reject insufficient budgets before provider execution. |
| `residual_variance_tolerance` | `1e-8` | Hartree squared threshold classifying negative variance estimates; both examples declare it and tests cover both negative statuses. |

Second-moment publications consume `max_measurements` and `max_pauli_terms`.
qEOM reconstructed-state products also consume its existing `max_product_terms`
guard. H² is formed in the full fermionic algebra before mapping or projection.

## Variational quantum deflation

VQD sequentially optimizes a fixed circuit with overlap penalties against previous
roots. The configured sampler evaluates `ComputeUncompute` fidelities with
`fidelity_shots` shots, even when that sampler is the ideal `statevector` sampler.
Use a noise-tolerant optimizer such as COBYLA, SPSA, or QNSPSA for these sampled
objectives. Tiny finite-difference steps, including SLSQP's default step, can give
unreliable gradients of finite-shot overlaps.

```yaml
algorithm:
  name: vqd
  options:
    k: 2
    target_root: 1
    betas: [3.0]
    fidelity_shots: 32768
    overlap_tolerance: 0.05
ansatz: uccsd
estimator:
  name: statevector
  options: {default_precision: 0.0}
sampler:
  name: statevector
  options: {seed: 3}
optimizer:
  name: cobyla
  options: {maxiter: 500, tol: 0.0001}
initial_point:
  name: random
  options: {seed: 31, scale: 1.0}
```

`betas` contains `k - 1` positive penalty strengths in hartree. If omitted, Qiskit
chooses penalties from the Hamiltonian coefficient norm. `initial_points` may
supply `k` explicit parameter vectors; otherwise the selected initial-point
component or Python `initial_point=` supplies the starting vector used for each
root. Conflicting initial-point sources are rejected. Zero starting parameters are
honored and can produce a stationary, duplicate excited-state search.

After optimization, ChemRefine measures the physical Hamiltonian for each root
and checks all pairwise overlaps. A root exceeding `overlap_tolerance` fails with
an actionable error. The reported spectrum is sorted by these measured physical
energies; `root_optimization_order` links it to the deflation search order.
`penalized_objectives`, `optimal_points`, `root_overlap_matrix`, and `root_sectors`
are retained separately. A low overlap alone does not certify an accurate excited
state. The sampled fidelity checks also have finite statistical uncertainty.

## Complex qEOM

qEOM first optimizes a VQE reference with the selected fixed ansatz. It then
constructs spin-projection-preserving excitation and de-excitation operators
relative to the **actual reference occupations**, including explicit orbital
reorderings. `excitation_ranks: [1, 2]` selects singles and doubles.

```yaml
algorithm:
  name: qeom
  options:
    excitation_ranks: [1, 2]
    target_root: 1
    max_excitations: 64
    conditioning_tolerance: 1.0e-10
    residual_tolerance: 1.0e-7
    frequency_tolerance: 1.0e-6
ansatz:
  name: uccsd
  options: {include_imaginary: true}
estimator:
  name: statevector
  options: {default_precision: 0.0}
optimizer:
  name: slsqp
  options: {maxiter: 300, ftol: 1.0e-12}
```

For an expansion basis `B`, ChemRefine measures the symmetrized double-commutator
Hessian and commutator metric,

\[
H_{ij}=\tfrac12\langle[[B_i^\dagger,H],B_j]+[B_i^\dagger,[H,B_j]]\rangle,
\qquad S_{ij}=\langle[B_i^\dagger,B_j]\rangle.
\]

Fermionic products are composed before mapping. Non-Hermitian observables are
measured through their Hermitian and anti-Hermitian parts, preserving complex
matrix elements. This is an owned adapter: it does not use Qiskit Nature's qEOM
path that casts response matrices to real values.

The generalized problem is checked for Hermiticity, metric conditioning, finite
real frequencies, excitation/de-excitation pairing, positive metric norms, and
eigenpair residuals. Singular metrics, zero modes, and unstable complex frequencies
raise `ConfigError`; they are not clipped to zero or removed by a pseudoinverse.
A tapering that removes a required observable's sector must be relaxed.

The reported excited energies equal the optimized reference energy plus positive
response frequencies. They are **linear-response approximations, not variational
upper bounds**. Diagnostics also contain normalized reconstructed-state Rayleigh
energies and spin moments, complex Hessian/metric entries, expansion coefficients,
and conditioning/residual checks. Reconstructed Rayleigh energies can differ from
the response energies when the reference or excitation expansion is approximate.

## QNSPSA

The `qnspsa` optimizer builds its fidelity metric from the same fixed ansatz and
configured sampler as the solve. It accepts `maxiter`, `blocking`,
`allowed_increase`, `learning_rate`, `perturbation`, `resamplings`, `regularization`,
`hessian_delay`, `fidelity_shots`, and `seed`. Scalar `learning_rate` and
`perturbation` must be supplied together. Its private perturbation stream preserves
the ambient Qiskit random stream, including after failures.

```yaml
algorithm: vqe
optimizer:
  name: qnspsa
  options:
    maxiter: 100
    learning_rate: 0.1
    perturbation: 0.1
    fidelity_shots: 4096
    seed: 9
sampler:
  name: statevector
  options: {seed: 5}
```

QNSPSA also works inside VQD and qEOM. The upstream `adapt_vqe` path cannot use this
fixed-circuit metric because its circuit changes between inner solves. Native
ffsim optimization has no qubit circuit for this metric and rejects the combination.
The owned [TETRIS and CEO drivers](qiskit-adaptive.md) rebuild the metric for each
grown circuit and support QNSPSA with the selected sampler.

## Budgets and examples

Both spectral algorithms expose `max_evaluations`, `max_measurements`, and
`max_pauli_terms`. `max_measurements` limits post-optimization estimator publications;
it does not count the optimizer's energy/fidelity calls. The optimizer iteration
budget and `fidelity_shots` separately determine sampling costs. qEOM additionally
checks `max_excitations` before generating its basis and `max_product_terms` around
fermionic operator multiplication. These controls are workload guards, not an
estimate of chemical error or a hard process-memory limit.

Runnable geometry examples share the existing template and H₂ input:

```bash
cd examples/tutorials/qiskit_sp
chemrefine run vqd.yaml --dry-run
chemrefine run qeom.yaml --dry-run
chemrefine run qnspsa.yaml --dry-run
# Remove --dry-run to execute a selected calculation.
```

Real-stack tests compare all four H₂ roots in the fixed `N=2, Ms=0` sector with
exact diagonalization, exercise nonzero imaginary qEOM matrix elements using a
complex hopping Hamiltonian, and check finite-shot VQD against an analytic
one-electron spectrum. They also cover overlap failures, conditioning, spin
constraints, compilation/layout handling, budget failures, and resource cleanup.
