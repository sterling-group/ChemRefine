# Quantum experiments

`engine: qiskit-experiment` runs one quantum experiment and writes a versioned
artifact bundle. Incoming molecular structures pass through unchanged. The
experiment's energies, trajectories and arrays are native artifacts; they do not
replace molecular energies in `steps.csv`.

Install `chemrefine[qiskit-fermionic]` for the lattice workflow. Like the molecular
Qiskit engine, this engine uses the managed backend provisioner and ordinary local
or SLURM scheduler. It needs no per-step Python template. Supply the usual CPU
SLURM header even for local dispatch.

## Options

| Key | Default | Meaning |
| --- | --- | --- |
| `experiment` | `ComponentSelection(name='lattice_dynamics', options={})` | Named component, as a string or `{name, options}` mapping. |
| `cores` | `1` | Requested worker threads; capped by the pipeline's `max_cores`. |
| `device` | `cpu` | Local device selection. The lattice experiment requires `cpu`. |
| `backend_python` | `None` | Optional explicit worker interpreter; otherwise use the managed backend. |
| `max_output_bytes` | `268435456` | Maximum total array bytes in the output bundle. This is an allocation guard, not an OS memory reservation. |

All options are validated strictly. A single artifact has no per-structure failure
policy, so use `on_failure: stop`.

## Lattice trajectories

The `lattice_dynamics` component accepts a `model`, `occupied_modes`, `times`, and
`dynamics`. The model and integrator use the same contracts as the
[Python lattice API](qiskit.md#fermionic-lattice-dynamics). Each time evolves the
same reference independently; the result includes time, statevector, occupation
and energy arrays. Energy uses the model's supplied units, and time uses their
inverse with hbar equal to one.

```yaml
steps:
  - step: 1
    engine: qiskit-experiment
    options:
      cores: 2
      max_output_bytes: 268435456
      experiment:
        name: lattice_dynamics
        options:
          model:
            num_sites: 2
            spinful: false
            edges:
              - {source: 0, target: 1, hopping: 1.0}
          occupied_modes: [0]
          times: [0.0, 0.25, 0.5, 1.0]
          dynamics: {steps: 4, order: 2, exact_reference: true}
```

See `examples/tutorials/qiskit_experiment/input.yaml` for a complete pipeline.

## Grouped Pauli measurement

`pauli_measurement` accepts a `circuit_path` containing one bound, unmeasured QPY
circuit, a real-coefficient `observable` mapping Pauli labels to weights, and a
`sampler` component. File paths resolve relative to the configuration file; input
bytes participate in cache identity. `max_circuit_bytes` defaults to 33554432.
Declared bundle inputs also include their referenced NPZ payloads in cache identity.
Typed file declarations may occur inside nested models, lists or dictionaries;
format selection follows the selected experiment and selected union branch. A
same-named field in another registered component does not change the input parser.

```yaml
experiment:
  name: pauli_measurement
  options:
    circuit_path: bell.qpy
    observable: {XX: 1.0, YY: 1.0, ZZ: 1.0}
    sampler: statevector
    measurement: {grouping: commuting, shots: 4096, pilot_shots: 64, seed: 7}
```

`measurement.grouping` selects `none`, qubit-wise commuting `qwc` (the default),
or general `commuting`. General groups use signed Clifford diagonalization;
the reported rotation cost includes those entangling gates. `shots` defaults to
4096 and includes independent pilot shots (default 64 per group). Pilots estimate
variances for allocation and are excluded from the final mean. Every production
group receives at least two shots. Uncertainty uses the full covariance of terms
measured together, not a sum of independent-term errors. It is an empirical
standard error, not a guaranteed confidence interval.

`seed` defaults to zero and splits local pilot/production random streams.
`max_terms` defaults to 4096; `max_memory_mb` defaults to 512 and guards estimated
count, covariance and supported simulator storage, not an OS reservation. Constants
alone require zero shots. Ideal statevector sampling and noisy Aer sampling retain
their existing sampler meanings. CUDA requires an Aer-capable worker and sampler.

Reports retain physical integer counts in NPZ separately from expectations. Packed
bitstrings use Qiskit display order, big-endian packing and trailing zero padding;
`num_qubits` identifies meaningful bits. Per-group covariance arrays, coefficients,
signed Z images and shot allocations make uncertainty reconstruction reproducible.
Measurement format version 2 also retains each rotation's signed binary Clifford
tableau in NPZ. Its rows are the images of `X0,...,Xn-1,Z0,...,Zn-1` under
`C P C†`; columns hold X bits, Z bits and a negative-sign bit, with qubit zero
first. Local recovery verifies binary entries, symplectic relations and each
configured Pauli's signed Z image. Different valid diagonalizing Cliffords are
accepted. This certificate establishes algebraic consistency, not proof that a
provider physically executed the recorded rotation.

Completion, cache consumption and cache rebuilding independently reconstruct
group means, sample covariance, the identity contribution and total standard
error from production counts. Pilots never enter those statistical denominators.
They also check configured coefficients, grouping, bit order and shot budgets.
Reconstruction uses one outcome vector at a time and respects `max_memory_mb`,
including retained arrays and its covariance workspace. Floating comparisons allow
`1e-10` relative and `1e-12` absolute rounding error. Reports predating version 2
must be regenerated: `CACHE_ONLY` reports an unusable cache, `RESUME` may execute
again, and `rebuild-cache` only validates local outputs without submitting jobs.
The same implementation is available as `measure_observable` in the public Python
API. The complete `measurement.yaml` example includes a small QPY input.

## Fermionic shadow datasets

The experimental `fermionic_shadows` component acquires either fixed-N complex
orbital Haar shadows or signed Majorana-Clifford shadows. It retains settings,
physical counts, complex RDMs and uncertainty clustered by randomized setting.
See [fermionic shadows](qiskit-shadows.md) for the distinct channel domains,
controls and runnable examples.

## Double-factorized molecular trajectories

The experimental `double_factorized_evolution` component reads a portable
integral bundle, builds executable ffsim circuits and evolves the declared
occupied orbitals. Its [domain and accuracy controls](qiskit-double-factorized.md)
distinguish integral truncation, product-formula order, Givens synthesis and
constant phases. Complex shared hopping and open-shell references are supported;
released-provider restrictions on two-body integrals are validated explicitly.

## Outputs and recovery

[Variational dynamics](qiskit-dynamics.md) evolves a supplied parameterized QPY
circuit using VarQITE or VarQRTE. Its selected estimator also measures gradients,
geometric tensors and additional observables, with bounded Euler/RK4 integration.

The product is `stepN/experiment/artifact.json`, accompanied by its named NPZ
payload. The manifest records the schema version, content digest, array shapes
and dtypes, units, model, and resolved options. Paths are relative so a complete
run directory can be relocated. Numeric payloads never require pickle.

Scratch cleanup preserves JSON, NPZ, QPY, checkpoint directories, and provider-job
records. A manifest is published only after its payload has been written.
Missing or corrupt payloads invalidate the output. Completion, cache reuse and
`rebuild-cache` also validate the selected experiment's scientific product: its
kind, required arrays, dimensions, numerical storage types and critical metadata.
An unrelated or empty integrity-valid bundle cannot satisfy a trajectory or
measurement step. Resource reports legitimately have no arrays, but must retain
their typed cost report and assumptions.

| Experiment | Required product checks |
| --- | --- |
| `lattice_dynamics` | Selected time grid, encoded state dimension, mode occupations, energies, model, integrator and units. |
| `variational_dynamics` | Time and parameter grids, observable columns, parameter ordering and five diagnostic columns per actual Euler/RK4 derivative evaluation. |
| `double_factorized_evolution` | Compatible factors, orbitals, states, occupations and reference populations; retained QPY header and circuit count. |
| `pauli_measurement` | Complete observable groups, covariance dimensions, packed physical counts and pilot/production shot accounting. |
| `fermionic_shadows` | Ensemble-specific setting dimensions, RDM order, setting clusters, uncertainty arrays and physical shot intervals. |
| `rdm_reconstruction` | Matching raw/fitted RDM dimensions, selected constraints and loss, accepted solver verdict and explicit representability limitations. |
| `spacetime_postselection` | Check registers and identities, circuit header, joint/accepted/rejected physical counts and acceptance accounting. |
| `circuit_cutting` | Observable and signed QPD coefficients, circuit count and every referenced physical measurement register. |
| `pauli_resources` | Analytical query-report discriminator, system width, budget, normalization and query counts. |
| `factorized_resources` | Selected factorization method, provider/version, normalization and positive logical/Toffoli costs. |
| `surface_code_resources` | Physical-report discriminator, matching explicit machine assumptions, physical qubits, runtime and failure bound. |

These checks run without quantum SDKs or provider calls and do not reopen mutable
input circuits or integrals. They establish the stored product's structural and
reporting contract; they do not reproduce the calculation or certify scientific
accuracy.
`CACHE_ONLY` reports unusable output without submission; `RESUME` may recompute.
Keep the complete bundle when copying results elsewhere.

Retained QPY uses the same SDK-free structural validator as molecular circuit
exports. It accepts [published QPY format versions](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qpy)
10–17 and checks complete file/type
headers, circuit counts, declared section lengths, register indices and available
circuit dimensions. Formats 16–17 additionally expose a circuit offset table;
every indexed circuit prefix is checked. Earlier multi-circuit formats expose only
the first prefix without decoding variable-length instructions, so later circuits
receive a total-size lower bound. The symbolic-encoding byte is meaningful only
in formats 10–12. These checks reject header-only stubs and unknown versions;
they do not decode instruction parameters or custom operations. Worker-side
QPY decoding remains necessary before executing a circuit.
Double-factorized and spacetime artifacts also require their known quantum widths
and zero classical bits: these saved evolution/check circuits precede measurement
and hardware layout expansion. Cutting subcircuits may have different widths.

For Python callers, `run_experiment(options, output_path)` executes the same
registry builder and `read_bundle(output_path)` returns validated, read-only
arrays. Third-party builders register in `EXPERIMENTS` and return
`ExperimentResult(kind, arrays, metadata)`. `read_bundle` checks generic integrity;
Python callers can additionally call `validate_experiment_output(name, bundle,
resolved_component_options)` from `chemrefine.engines.qiskit.experiment_outputs`.

Third-party components must also call `register_experiment_output(name, kind,
validator)` in that module. The validator receives a `QuantumBundle` and the
fully defaulted component options, raises `ValueError` on invalid content, and
must remain SDK-free and local. Import both registrations in the orchestrator
and worker. An undeclared output contract fails configuration preflight before
any builder or provider executes, and cannot bypass recovery checks. Contract
names use the same lowercase/hyphen normalization as component names. Compatible
omitted default fields in older lattice metadata are normalized before comparison,
so a valid archived product does not fail merely because the option model grew.
