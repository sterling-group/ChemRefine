# Fermionic shadows

The experimental `fermionic_shadows` experiment acquires randomized quantum
measurements and reconstructs complex one- and two-particle reduced density
matrices (RDMs). Its two ensembles have distinct circuits, domains and inverse
channels. The catalog marks these domains explicitly.

| `shadows.ensemble` | Measurement setting | Supported state and inversion |
| --- | --- | --- |
| `orbital_haar` | Complex Haar-random U(m) orbital rotation | Fixed total particle number N, supplied as `num_particles`. Low's number-conserving channel is inverted in that sector. |
| `majorana_clifford` | Uniform signed Majorana permutation with determinant +1 | Arbitrary Jordan–Wigner qubit states. Invert each even Majorana degree separately; no fixed-N assumption. |

Here `m` counts fermionic modes, including spin when relevant. Mode i maps to
qubit i. The acquisition code does not infer a spin ordering or assume equal
alpha/beta populations. Spin mixing and complex coherences are allowed. These
experiments measure the supplied preparation circuit; they do not supply a
classical reference workflow.

## YAML acquisition

Supply one bound, unmeasured circuit in Qiskit's QPY format. Input paths resolve
relative to the YAML file and their contents participate in cache identity.
The selected sampler has the same local device and scheduler core grants as
other Qiskit components. The combined backend profile must contain the sampler
and fermionic providers.

```yaml
experiment:
  name: fermionic_shadows
  options:
    circuit_path: one_particle.qpy
    sampler: {name: statevector, options: {seed: 7}}
    shadows:
      ensemble: orbital_haar
      num_particles: 1
      num_settings: 200
      shots_per_setting: 8
      max_order: 2
      seed: 13
```

For Majorana shadows set `ensemble: majorana_clifford` and omit
`num_particles`. Their pairing rotations can change measured particle number.
Every physical outcome is retained, including outcomes outside the preparation's
particle sector. Total-particle postselection is rejected for this ensemble.

Complete runnable pipelines and their small QPY input are in
`examples/tutorials/qiskit_experiment/orbital_shadows.yaml` and
`examples/tutorials/qiskit_experiment/majorana_shadows.yaml`. These examples use
local ideal sampling and require `chemrefine[qiskit-fermionic]`.

## Controls and knob verdicts

The experiment has four fields: required `circuit_path` and `shadows`,
`sampler` (default `statevector`), and `max_circuit_bytes` (default 33554432).
Unknown fields are rejected. The nested controls are:

| Field | Default | Meaning and verification |
| --- | --- | --- |
| `ensemble` | `orbital_haar` | Distinct measurement channel; both appear in runnable examples. |
| `num_particles` | `None` | Required integer N for orbital Haar; forbidden for Majorana. Exercised in both examples and domain tests. |
| `num_settings` | `100` | Number of independently randomized settings; example and reproducibility tests. |
| `shots_per_setting` | `1` | Physical shots for each setting; example and raw count round-trip tests. |
| `max_order` | `2` | Compute 1-RDM alone (`1`) or both RDMs (`2`); both examples and artifact tests. |
| `postselect_particles` | `false` | Explicit fixed-N filtering for orbital shadows only; rejection/acceptance tests. Postselection under noise can bias estimates. |
| `max_total_shots` | `1000000` | Refuse larger acquisition before submission; budget tests. |
| `max_memory_mb` | `512` | Bound estimated settings, count, tensor and supported simulator storage; allocation tests. This does not request scheduler memory. |
| `seed` | `0` | Local generator seed for settings; independent from the sampler seed. Reproducibility tests. |

`max_output_bytes` is the artifact engine's separate payload limit. Both budgets
are checked before sampling. Standalone snapshot functions also accept
`max_memory_mb`. Supplied count batches may contain fewer shots than the declared
per-setting budget but must use positive integer physical frequencies.

## RDM and uncertainty conventions

The arrays follow
`gamma[p,q] = <a†p aq>` and
`Gamma[p,q,r,s] = <a†p a†q a_s a_r>`.
They are dimensionless. Neither finite-sample inverse guarantees positive RDMs.
The orbital inverse uses
`gamma_hat = (m+1) B - N I`, where
`B = (U† diag(outcome) U).T`.
Its 2-RDM weights depend on N as well as m; reusing the unrestricted Majorana
inverse would be incorrect. The Majorana degree-2k inverse multiplier is
`binom(2m,2k) / binom(m,k)`.

The mean weights each independent setting equally. Repeated shots at one setting
form a cluster. The reported real and imaginary standard errors use variation
between settings with a sample-variance denominator. They are empirical standard
errors, not guaranteed confidence intervals. A single setting has no estimated
standard error. `ShadowResult.observable(hamiltonian)` contracts each complete
setting before computing its standard error, preserving covariance between the
RDM entries contributing to the observable.

A dataset retains every actual setting matrix, physical count, setting-specific
RDM, mean RDM and available standard error in NPZ. Counts are packed in Qiskit
bitstring display order; `setting_offsets` identifies each setting's count rows.
The manifest records the mode count, conventions, seed, acceptance by setting,
and sampler metadata. Complex tensors remain complex numeric arrays. Artifacts
can reproduce the inverse without provider calls or another circuit submission.

The Python entry points are `collect_fermionic_shadows` for acquisition and
`estimate_fermionic_shadows` for previously acquired data, in
`chemrefine.engines.qiskit.shadows`.

## Scientific checks

Tests compare complex orbital 2-RDM snapshots against Low's independent pair-space
weights, and average the complete 105 four-mode Majorana matching design for a
correlated complex state against direct Fock-space RDMs. They also compare every
Majorana generator under the synthesized circuits, exercise real statevector and
Aer samplers, and reload/reconstruct complete artifact datasets.

The number-conserving inverse follows [Low, arXiv:2208.08964v2](https://arxiv.org/html/2208.08964v2).
The separate Majorana channel follows [Zhao, Rubin and Miyake,
arXiv:2010.16094](https://arxiv.org/abs/2010.16094).
