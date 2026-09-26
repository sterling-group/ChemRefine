# Constrained RDM reconstruction

The experimental `rdm_reconstruction` component fits measured complex one- and
two-body reduced density matrices (RDMs) under a selected subset of D, Q and G
positive-semidefinite constraints. CVXPY models the convex problem; the explicitly
selected SCS solver computes a numerical solution. Install the separate
`qiskit-rdm` backend, or use the combined `qiskit-toolkit` environment when both
acquisition and reconstruction must run in one worker environment.

These are necessary representability conditions. Their satisfaction does not
prove that an N-particle state realizes every reconstructed tensor. The default
objective fits measurements. Optional energy regularization is explicitly labeled
and does not establish a variational energy bound. A reconstructed tensor also
does not inherit a statistically valid uncertainty merely because a solver
reports convergence.

## Input and output

`rdm_bundle_path` accepts a native `fermionic_shadows` or `measured_rdms` bundle
containing `one_body` and `two_body` arrays. Their conventions are
`gamma[p,q] = <a†p aq>` and
`Gamma[p,q,r,s] = <a†p a†q a_s a_r>`.
Complex coherences, unequal spin populations and spin mixing are supported; the
constraint is the explicitly supplied **total** particle number. Noise may violate
Hermiticity, antisymmetry, contraction or positivity in the raw arrays.

The output retains `raw_one_body` and `raw_two_body` beside the reconstructed
`one_body` and `two_body`, all in numeric NPZ arrays. A manifest records solver
status, captured convergence warnings, objective values, iterations, solve time, constraint residuals and minimum
eigenvalues. Raw and reconstructed energies are separate when a Hamiltonian is
supplied. Non-Hermitian measurement noise can give a complex raw contraction;
its real and imaginary parts are retained explicitly.

An optional `hamiltonian_bundle_path` reads the same portable electronic-integral
bundle used by [double-factorized evolution](qiskit-double-factorized.md), while
preserving its complex/unrestricted domain. The nuclear-repulsion constant enters
that observable once. RDM modes must already use the integral bundle's alpha-then-beta ordering when an energy is requested. Optional `one_body_weights_array` and
`two_body_weights_array` name arrays in the measurement bundle. Weights must be
finite, real, nonnegative and match the corresponding tensor. Zero weights mask
unmeasured entries; at least one entry must have positive weight.

Both descriptors and every referenced NPZ payload participate in cache identity.
Reconstruction parses only local input artifacts and never queries a provider.

## YAML workflow

```yaml
experiment:
  name: rdm_reconstruction
  options:
    rdm_bundle_path: measured.json
    reconstruction:
      num_particles: 2
      constraints: DQG
      loss: frobenius
      energy_weight: 0.0
```

`examples/tutorials/qiskit_experiment/rdm_reconstruction.yaml` first acquires a
complex orbital-shadow dataset, then references that dataset in a second artifact
step. Input files produced upstream need not exist during configuration validation;
they must exist when the consuming step begins. Each step uses its appropriate
provider profile and passes incoming molecular structures through unchanged.

## Constraints and fitting

The 2-RDM D variable is complex Hermitian in the antisymmetric pair basis p<q.
The complex Q and G matrices are derived by the canonical anticommutation
relations, with the same algebra used in independent numerical checks:

- `D[(p,q),(r,s)] = <a†p a†q a_s a_r>`.
- `Q[(p,q),(r,s)] = <a_q a_p a†r a†s>`.
- `G[(p,q),(r,s)] = <(a†p aq)† (a†r as)>`, using all ordered mode pairs.

All modes also satisfy 0 ≤ gamma ≤ I, trace(gamma)=N, trace(D)=N(N−1)/2 and the
one-/two-RDM contraction identities. The returned point is checked independently
of the solver's status label. Unaccepted statuses, missing primal values and
violations beyond the requested feasibility tolerance fail the experiment.

`frobenius` minimizes weighted squared entry errors. `nuclear` minimizes the
nuclear norms of the weighted one-body matrix and flattened two-body matrix.
Every measured tensor entry contributes, including noisy symmetry-related
entries; the input is not silently symmetrized before fitting.

## Controls and knob verdicts

The adapter fields are required `rdm_bundle_path` and `reconstruction`, optional
`hamiltonian_bundle_path`, optional weight-array names, and `max_input_bytes`
(default 268435456). All appear in the example. Nested options are:

| Field | Default | Meaning and verification |
| --- | --- | --- |
| `num_particles` | required | Integer total particle number; algebra and provider tests cover empty, single-particle and correlated sectors. |
| `constraints` | `DQG` | `D`, `DQ` or `DQG`; independent complex operator-Gram algebra and separate numerical solver cases. |
| `loss` | `frobenius` | Weighted squared error or `nuclear` loss; separate real-solver cases. |
| `energy_weight` | `0` | Add this weight times physical energy to the data objective; requires the Hamiltonian bundle. No variational claim. |
| `solver_tolerance` | `1e-6` | Requested SCS absolute/relative numerical tolerance. |
| `feasibility_tolerance` | `1e-5` | Independently checked residual/eigenvalue tolerance after solving. |
| `max_iterations` | `10000` | SCS iteration cap; failed/inaccurate outcomes retain explicit policies. |
| `max_memory_mb` | `512` | Estimated convex-model allocation guard, separate from scheduler memory. |
| `accept_inaccurate` | `false` | Permit SCS's inaccurate-optimal status only if the primal feasibility checks also pass. |

`max_output_bytes` remains the artifact engine's separate output allocation guard.
Worker CPU limits are established before importing numerical providers. The
Python entry point is `reconstruct_rdms(raw, options, ...)`; it lazily imports
CVXPY/SCS only when called. Vectorized CAR maps avoid scalar expression-tree
expansion, and real one-by-one variables handle empty or single-mode pair spaces.

The SDK-free algebra suite checks every D/Q/G element against independent complex
Fock-space operator Grams. The separate `test_engines_qiskit_rdm_provider.py`
suite requires CVXPY/SCS and exercises actual optimization and artifact round
trips. Algebra checks do not replace numerical provider validation.

The constrained-shadow approach is discussed in
[arXiv:2511.09717](https://arxiv.org/html/2511.09717v1). This implementation uses
complex Hermitian constraints and distinguishes data fitting from optional energy
regularization. Provider conventions follow
[CVXPY's complex constraints](https://www.cvxpy.org/tutorial/constraints/index.html)
and [SCS settings](https://www.cvxgrp.org/scs/api/settings.html).
