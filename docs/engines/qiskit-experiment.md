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
The same implementation is available as `measure_observable` in the public Python
API. The complete `measurement.yaml` example includes a small QPY input.

## Fermionic shadow datasets

The experimental `fermionic_shadows` component acquires either fixed-N complex
orbital Haar shadows or signed Majorana-Clifford shadows. It retains settings,
physical counts, complex RDMs and uncertainty clustered by randomized setting.
See [fermionic shadows](qiskit-shadows.md) for the distinct channel domains,
controls and runnable examples.

## Outputs and recovery

The product is `stepN/experiment/artifact.json`, accompanied by its named NPZ
payload. The manifest records the schema version, content digest, array shapes
and dtypes, units, model, and resolved options. Paths are relative so a complete
run directory can be relocated. Numeric payloads never require pickle.

Scratch cleanup preserves JSON, NPZ, QPY, checkpoint directories, and provider-job
records. A manifest is published only after its payload has been written.
Missing or corrupt payloads invalidate the output; `rebuild-cache` validates local
files without executing an experiment. Keep the complete bundle when copying
results elsewhere.

For Python callers, `run_experiment(options, output_path)` executes the same
registry builder and `read_bundle(output_path)` returns validated, read-only
arrays. Third-party builders register in `EXPERIMENTS` and return
`ExperimentResult(kind, arrays, metadata)`.
