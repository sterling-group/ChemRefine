# Circuit cutting

`qiskit-experiment` provides `circuit_cutting` for bound qubit circuits and Hermitian
Pauli observables. It uses Qiskit addon cutting's local-operation QPD method and
V2 sampler reconstruction. The output is a `circuit_cutting` bundle; structures
pass through unchanged and the result is not a molecular-energy record.

Install the optional worker with `chemrefine backends install qiskit-cutting`.
The examples in `examples/tutorials/qiskit_cutting/` generate a GHZ
circuit and exercise manual gates, wires, explicit partitions, and automatic cuts.
Both molecular and lattice preparation circuits can be supplied after mapping and
parameter binding. The observable uses the same qubit mapping as the input QPY.

## Choosing a decomposition

`circuit_path` identifies a QPY file containing exactly one circuit. Input circuits
must have no classical bits and no unbound parameters. Measurements introduced by
the addon are preserved throughout compilation and sampling.

`cutting.mode` selects one of three interfaces:

- `manual`: `gate_cuts` contains indices into the **original** circuit instruction
  list. Each chosen instruction must act on two qubits and have a supported SDK
  decomposition. `wire_cuts` contains `{qubit, before_instruction}` pairs in that
  same original coordinate system. A boundary equal to the original instruction
  count means after its final instruction. Repeated cuts of a wire are allowed
  at distinct boundaries.
- `partition`: `partition_labels` supplies a nonempty string for every original
  qubit. Gates crossing labels are decomposed using the SDK.
- `automatic`: the SDK's `find_cuts` searches for pieces satisfying
  `max_subcircuit_qubits`. `allow_gate_cuts`, `allow_wire_cuts`, `max_backjumps`,
  and `seed` control the search. An unfinished search can return a feasible greedy
  result; `minimum_reached` records whether optimality was proved.

Automatic search requires gates on at most two qubits. Decompose wider gates
before writing QPY if they must participate in automatic search. Barriers are
removed after original-index manual placement because they do not change the
quantum channel. Automatic cut indices in the report refer to the search circuit,
with barriers removed. Idle input wires remain explicit `|0>` factors, including
when an observable acts on them.

The workflow expands wire cuts with the public SDK transformation and maps each
observable to the original wire's final segment. It checks actual partition width
and actual QPD overhead after decomposition. The search's `max_gamma` stopping
condition is not treated as proof that the returned plan satisfies either limit.
These interfaces follow the
[released addon API](https://qiskit.github.io/qiskit-addon-cutting/apidocs/qiskit_addon_cutting.html)
and [wire-cutting construction](https://arxiv.org/abs/2302.03366).

## Distinct sampling controls

`cutting.num_samples` controls QPD decomposition draws. Set it to `exact` to
enumerate exact QPD weights. A finite value may still evaluate sufficiently large
weights exactly and sample only the remaining tail. It is independent of
`cutting.shots`, the physical shots executed for **each generated subexperiment**.
The artifact records actual experiment count, total shots, signed coefficients,
and each coefficient's `EXACT` or `SAMPLED` status.

The addon uses process-global NumPy randomness for QPD generation. ChemRefine
performs search and generation in a dedicated Python subprocess, passing only
trusted QPY and JSON temporary files. The parent process's random state is never
seeded or restored. Sampler execution stays in the original worker, using its
registered provider and closing the resource on success or failure. Set both
the cutting seed and sampler simulation/transpilation seeds for reproducibility.

The default `basic_backend` sampler supports the mid-circuit measurements and
resets that QPD decompositions can introduce. An ideal `statevector` sampler may
reject these operations; use `basic_backend`, `aer`, or a compatible registered
V2 provider. Compilation must preserve the `observable_measurements` and
`qpd_measurements` registers and remain within the selected width bound.

The SDK performs signed reconstruction directly from the physical BitArrays.
Reconstructed estimates may lie outside a Pauli operator's spectral interval at
finite sampling. They are not clipped, converted into physical probabilities,
or represented as negative physical shot counts. No standard error is inferred:
QPD randomness and physical shot noise are distinct sources of uncertainty.
See the [SDK experiment generation and reconstruction source](https://github.com/Qiskit/qiskit-addon-cutting/tree/main/qiskit_addon_cutting).

## Resource limits and artifacts

Before enumerating QPD experiments, the planner bounds exact Cartesian terms,
unique samples, observable groups, total circuits, generated instruction count,
shot bytes, and descriptor size. `max_sampling_overhead` bounds the square of the
product of QPD one-norms. `max_exact_terms` limits full enumeration, including when
finite `num_samples` happens to trigger the SDK's exact branch.

`max_input_qubits`, `max_input_instructions`, `max_observables`, and `max_cuts`
bound input/search complexity. `max_subexperiments`, `max_generated_instructions`,
`max_total_shots`, `max_qpy_bytes`, and `max_working_bytes` bound generated work.
The working-byte forecast includes a conservative instruction-object allowance;
it is not an operating-system process-memory cap. Known dense simulator state
allocations are checked separately. `worker_timeout_seconds` bounds planning;
`publications_per_job` bounds the batch sent to a sampler.

Both original and compiled partition widths are checked. Backend compilation that
adds physical ancillas can therefore require a larger width limit. Output capacity
is checked before sampler execution. Requested budgets may reject an otherwise
mathematically valid decomposition; a high-overhead cut is not made practical by
a narrow circuit alone.

The bundle retains per-register raw unsigned BitArray buffers, register widths,
shots, partition and experiment ordering, signed QPD weights, subobservables, and
the generated logical experiments as QPY bytes. Provider/version and option
provenance accompany the arrays. `reconstruct_cutting_records(bundle.arrays,
bundle.metadata)` repeats the SDK reconstruction without new quantum execution.
`logical_experiments_qpy` can be read with Qiskit's QPY loader for inspection.

Python users can call `plan_cutting`, inspect the resulting `CuttingPlan`, then
call `execute_cutting`. The result carries arrays and metadata suitable for the
common checksummed bundle format. Ordinary ChemRefine input fingerprinting,
relocation, output validation and resume/rebuild behavior apply.
