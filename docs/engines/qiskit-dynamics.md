# Variational quantum dynamics

The experimental `variational_dynamics` component evolves a parameterized circuit
using McLachlan's variational principle. `method: varqite` approximates normalized
imaginary-time evolution; `varqrte` approximates real-time Schrödinger evolution.
The Hamiltonian is a time-independent Hermitian Pauli sum. Complex hopping and
open-shell states are supported when the supplied circuit represents their sector.
A restricted circuit manifold can introduce physical approximation error.

The selected estimator evaluates the Hamiltonian, observables, LCU gradients and
phase-fixed geometric tensor. Derivative ancillas are added before compilation,
and each observable follows its circuit's actual layout. No separate derivative
simulator replaces the selected provider. Local CPU/GPU settings and granted cores
reach the same estimator resource, which closes on success and failure.

The [Qiskit Algorithms variational framework](https://qiskit-community.github.io/qiskit-algorithms/tutorials/11_VarQTE.html)
provides the measurement-based gradients and McLachlan equations. ChemRefine owns
the bounded Euler/RK4 integration and the explicitly regularized metric solve.

## YAML and Python

Use `engine: qiskit-experiment` and `experiment.name: variational_dynamics`.
`circuit_path` names a single parameterized QPY circuit. The file is resolved from
the configuration directory and its contents participate in the cache key.
`max_circuit_bytes` bounds loading (default 32 MiB). `observable` is the Hamiltonian
mapping of Pauli labels to real coefficients; the optional `observables` sequence
contains additional operators. Label widths must match the circuit.

```yaml
experiment:
  name: variational_dynamics
  options:
    circuit_path: ry.qpy
    observable: {Z: 1.0}
    observables: [{X: 1.0}]
    initial_parameters: [1.5707963267948966]
    estimator: statevector
    dynamics:
      method: varqite
      time: 0.5
      steps: 8
      integrator: rk4
```

`initial_parameters` follows the QPY circuit's `circuit.parameters` ordering, which
is retained in the result. The public Python API is
`variational_dynamics(circuit, hamiltonian, initial_parameters, options=...,
estimator=..., observables=..., cores=..., device=...)` and returns immutable
NumPy arrays. The runnable example is
`examples/tutorials/qiskit_experiment/variational.yaml`.

## Accuracy and limits

| Dynamics option | Meaning |
| --- | --- |
| `method` | `varqite` or `varqrte` |
| `time`, `steps` | Positive final time and fixed number of integration intervals |
| `integrator` | `euler` or fourth-order `rk4`; compare refinements to assess error |
| `metric_cutoff` | Absolute eigenvalue cutoff for the geometric tensor |
| `regularization` | Nonnegative ridge added only to retained eigenvalues; introduces bias |
| `numerical_tolerance` | Finite, Hermitian and positive-semidefinite acceptance tolerance |
| `max_velocity` | Maximum parameter-velocity norm before failure |
| `max_parameters`, `max_pauli_terms` | Circuit and observable size limits |
| `max_publications` | Bound on broadcast estimator evaluations, including derivatives |
| `max_memory_mb` | Working-array and known dense-simulator estimate guard |

`max_memory_mb` does not request scheduler memory and is not an exact peak-memory
bound. `max_publications` does not bound provider mitigation/calibration overhead;
finite-shot precision, hardware provider options and their budgets must also be set.
The trajectory includes time zero. The first expectation column is always the
original Hamiltonian, including any supplied identity coefficient exactly once.
Additional observables follow in the supplied order. Energy coefficients determine
time units with hbar set to one; no molecular nuclear energy is inferred or added.

The artifact contains `times`, `parameters`, `expectations` and one
`metric_diagnostics` row per ODE evaluation. Diagnostic columns report retained
rank, extreme eigenvalues, solve residual and velocity norm. Truncated directions
can leave a nonzero residual; inspect it and perform convergence checks. Strongly
negative measured metric eigenvalues fail validation rather than silently becoming
a different metric. Shot noise propagates nonlinearly through integration; these
trajectories do not claim confidence intervals. Molecular `energy_hartree` records
are unchanged by the artifact engine.
