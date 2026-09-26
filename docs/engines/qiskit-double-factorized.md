# Double-factorized evolution

The experimental `double_factorized_evolution` component builds executable
Jordan–Wigner circuits for molecular Hamiltonians and simulates their action on
the declared occupied orbitals. It uses the installed ffsim factorization and
gate synthesis, and reports costs from the actual compiled circuits.

The supported domain is a complex Hermitian one-body tensor shared by alpha and
beta spins, with real shared-spin two-body integrals in chemist order. Explicit
physicist-order inputs are converted according to their declared convention.
Open-shell populations and reordered occupied orbitals are supported. Unequal
unrestricted spin blocks, nonidentity alpha/beta orbital overlap and genuinely
complex two-body tensors are rejected before factorization. Other quantum methods
retain their broader complex/unrestricted domains.

## Portable integral input

`save_integrals(path, data)` and `load_integrals(path)` in the public Qiskit API
write and read an `ElectronicStructureData` object using a versioned native
`electronic_structure` bundle. Integral tensors, overlaps, orbital energies and
occupations are numeric NPZ arrays; the JSON manifest records tensor order,
alpha-then-beta orbital order, hartree units, particle counts, nuclear repulsion,
geometry and provenance. Complex and unrestricted data are preserved by the
bundle even when a consuming algorithm supports a narrower domain.

Every load verifies the array digests, allocation limit, array references and
owned data constructor's physical constraints. Arrays never require pickle.
A YAML input dependency includes the descriptor and all referenced payloads.
Copy the whole bundle when relocating an input.

## YAML trajectory

```yaml
experiment:
  name: double_factorized_evolution
  options:
    integral_bundle_path: double_factorized_integrals.json
    times: [0.0, 0.25, 0.5]
    evolution:
      factorization: eigh
      factorization_tolerance: 1.0e-8
      steps: 4
      order: 2
      exact_reference: true
```

The complete runnable example is
`examples/tutorials/qiskit_experiment/double_factorized.yaml`; its input bundle
contains a small complex hopping and real interaction model. The backend profile
is `qiskit-fermionic`. Each time starts from the same occupied-orbital reference,
including time zero. Time is measured in hbar/hartree. These are ideal local
statevector trajectories, and the experiment requires `device: cpu`.

The builder `build_double_factorized_evolution(data, options, cores=1)` returns a
compiled circuit and factorization arrays without allocating a statevector.
`simulate_double_factorized_evolution` additionally returns a bounded statevector,
physical energy and mode occupations. Its `DoubleFactorizedOptions.time` defaults
to 1 and permits negative times. The YAML adapter uses `times` instead.

## Factorization, accuracy and phases

`eigh` uses nested eigendecomposition and supports real indefinite interaction
matrices. `cholesky` first verifies positive semidefiniteness of the pair-indexed
Coulomb matrix. `compressed` optimizes the factor matrices and reports the actual
optimizer status, iterations, objective and evaluation count. A zero interaction
requires no optimization or two-body factors. An unconverged compression optimizer
is reported; the measured residual, not its convergence flag, determines whether
`max_integral_error` is satisfied.

Truncating interaction factors also changes the reconstructed one-body tensor
through the normal-ordering correction. Both maximum tensor residuals are
reported. `max_factors` can override the requested tensor tolerance; the result
states whether that tolerance was met. The conservative factorization Hamiltonian
error bound is `2 sum(abs(delta_h1)) + 2 sum(abs(delta_h2)) + abs(delta_constant)`;
multiplying by absolute
time and capping at 2 bounds its unitary error. This bound covers factorization
only. Product-formula and Givens synthesis error bounds are explicitly absent.

The public product-formula orders are the physical orders 1, 2 and 4. Their
internal ffsim values are mapped explicitly to 0, 1 and 2. Optional exact-reference
comparisons evolve the **original** Hamiltonian and report phase-sensitive state
error, infidelity and reference energy. These are finite-instance numerical
checks, not universal error bounds.

Nuclear repulsion and the optional `energy_shift_hartree` enter the Hamiltonian
constant once. The same total constant supplies the circuit's global phase and
reported physical energies. Supply `energy_shift_hartree` only for an additional
constant removed from the supplied integrals, such as a separately prepared
inactive-space contribution. The original constant components and the internally
shifted factorized constant are reported separately, including when the Z
representation changes normal ordering.

Named electronic `energy_offsets` in the integral bundle are also added once,
alongside nuclear repulsion and `energy_shift_hartree`. They are recorded as
`input:<name>` entries in the offset metadata. Do not repeat a bundle constant
in `energy_shift_hartree`; that option represents an additional scalar shift.

## Controls and knob verdicts

The experiment requires `integral_bundle_path`. `max_input_bytes` defaults to
268435456. `times` defaults to `[0, 1]` and accepts 1–512 finite values. The nested
`evolution` controls are strict and frozen:

| Field | Default | Meaning and verification |
| --- | --- | --- |
| `steps` | `1` | Product-formula steps per time; convergence tests and example. |
| `order` | `2` | Physical order 1, 2 or 4; every order has an independent convergence test. |
| `factorization` | `eigh` | `eigh`, `cholesky` or `compressed`; all tested numerically. |
| `factorization_tolerance` | `1e-8` | Requested maximum two-body tensor error; tested with exact and truncated factorizations. |
| `max_factors` | `None` | Optional factor-rank cap, which may exceed the requested error; truncation tests. |
| `max_optimizer_iterations` | `100` | Compressed-factor optimization limit; solver diagnostics are retained. |
| `max_integral_error` | `None` | Refuse if either reconstructed tensor's largest entry error exceeds this value; refusal tests. |
| `z_representation` | `false` | Factorization into Z rather than number operators; phase-sensitive equivalence tests. |
| `givens_tolerance` | `1e-12` | Orbital-rotation synthesis threshold, separate from tensor truncation. |
| `energy_shift_hartree` | `0` | Explicit additional Hamiltonian constant; phase and energy-offset tests. |
| `optimization_level` | `1` | Qiskit transpilation level 0–3 in the `rz,sx,x,cx` basis. |
| `seed_transpiler` | `0` | Reproducible synthesis seed. |
| `max_operations` | `1000000` | Preflight synthesis-work estimate and final compiled-operation limit. |
| `max_memory_mb` | `512` | Estimated numerical/synthesis/reference storage guard, not a scheduler reservation. |
| `max_statevector_qubits` | `16` | Explicit simulator width cap; the circuit-only API does not allocate a statevector. |
| `exact_reference` | `false` | Bounded dense exponential of the original Hamiltonian for validation. |

All fields appear in the runnable example. The separate artifact
`max_output_bytes` bounds the stored trajectory. CPU grants reach factorization,
local evolution and transpilation. These estimates do not promise a process-level
RSS limit or reserve scheduler memory.

## Artifacts and validation

The output stores times, statevectors, occupations, energies, factor tensors and
QPY-v13 bytes for every compiled evolution circuit. Saved circuits exclude the
reference preparation, whose exact occupations remain in the manifest. Compiled
gate counts, two-qubit counts and depth accompany each time point. The circuits
are executable; the cost records are not merely an analytical cost graph.

Tests verify complex hopping against an independently mapped original
Hamiltonian, first/second/fourth-order convergence, open-shell and reordered
references, both factor representations, negative time, zero time, explicit
constants, tensor truncation, zero interactions, provider domain refusals and
artifact circuit replay.

The provider implementation and supported tensor domain are documented in the
[ffsim double-factorization API](https://qiskit-community.github.io/ffsim/api/ffsim.linalg.html#ffsim.linalg.double_factorized).
The method follows [Motta et al., arXiv:1808.02625](https://arxiv.org/abs/1808.02625)
and [Cohn, Motta and Parrish, arXiv:2104.08957](https://arxiv.org/abs/2104.08957).
