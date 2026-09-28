# Quantum validation and benchmarks

This directory keeps reproducible local simulation measurements separate from test
durations. No remote quantum jobs are submitted by the benchmark suite.

Start with [dependency ownership](DEPENDENCIES.md) and the run's `REPORT.md`.
Each run under `runs/` retains:

- `validation/`: command logs, environment snapshots and JUnit test results.
- Campaign directories: benchmark manifests, raw JSONL, tidy CSV and summaries.
- `audit/`: direct dependencies, internal import edges and compatibility findings.
- `analysis/`: consolidated measurements, per-case summaries, diagnostics,
  campaign metadata, matched-stack speed ratios and a scaling chart.

The benchmark matrix is finite and explicit. It covers registered components,
supported local primitive methods, selected problem sizes, precision modes, noise
settings and thread counts. It does not claim to exhaust arbitrary circuits,
continuous numerical options, all package versions or remote hardware. Unsupported
combinations and resource limits are retained as rows rather than discarded.

Compare matching case IDs and sizes across environments. CPU-only algorithms are
never reported as GPU measurements. Test durations include fixture and assertion
overhead and are not simulator throughput measurements.

## Matrix and Scientific Scope

The initial execution matrix has 775 configurations per requested device:

| Family | Cases | Dimensions |
| --- | ---: | --- |
| Estimation / sampling | 440 | 4/8/12/16/20 qubits; 1/8 Aer threads; single/double precision; local providers; every exposed Aer method; ideal/depolarizing noise |
| Molecular | 316 | Every algorithm at 4/6 spin orbitals; all 4 mappers x 8 ansatze x 9 optimizers for VQE; all initial states/points |
| Tutorials | 19 | Every YAML in experiment, cutting and resource-estimation tutorial groups |

The current `all` suite additionally includes 48 credential-free IBM Runtime
fake-backend configurations (823 total): 2/4 logical qubits, estimator/sampler,
executor/legacy implementations, job/session/batch modes, and estimator resilience
levels 0/1/2. Unsupported legacy mitigation and CUDA placement are explicit
rejections. These are local simulations, not IBM hardware timings. Per-case job
journals are retained in `journals/`. Remote noise learning and credentials are
never enabled. The 2026-09-28 run keeps this addition in separate campaigns.

The current molecular cases supply explicit UCC excitations and enable spin
symmetrization for the equal-population SKQD examples. The `regressions` suite
contains 69 focused cases for the defects and configuration issues found during
the initial campaigns; historical failing configurations remain unchanged on disk.

Invalid graphs are intentional cells, not failed attempts to execute a supported
configuration. Some components need more than a name: explicit UCC excitations,
determinant occupations and random seeds are examples. Retain follow-up runs with
the actual options rather than treating rejected defaults as complete coverage.
Algorithms that do not consume a configured component do not exercise that
component. The full registered-name audit is **schema validation only**, with
829,440 name/device combinations at fixed canonical options. It is not 829,440
numerical simulations.

This is not every possible combination: continuous parameters, arbitrary circuits,
molecule geometries, noise channels, active spaces, solver tolerances, remote
devices and package versions form an unbounded space. Increasing sizes requires
an explicit memory/runtime budget. Density matrices are capped at a conservative
256 MiB complex128 state estimate; noisy trajectories stop at 12 qubits. The
matrix records those exclusions. Unitary/stabilizer methods are not exposed by
ChemRefine's shot-provider schema and are not invented as supported engine modes.

Primitive circuits use four seeded Ry/Rz/CX layers and 1,024 shots when sampling.
Ideal results are checked against an independent Qiskit Statevector expectation;
sampling tolerance is six times `1/sqrt(shots)`. Noisy results are checked for
finite physical bounds, not incorrectly compared to an ideal answer. Molecular
cases use the stored H2 integrals or a declared three-spatial-orbital hopping
model. Their finite energies are checked against the full-space exact lower
bound; approximation errors and convergence are retained separately. A finite,
variational result does not establish chemical accuracy or the right spin sector.
Tutorials check native bundle decoding and finite output arrays; the test suite
supplies their stronger scientific invariants.

## Timing and Data Dictionary

Each case uses a new subprocess. Setup occurs once, followed by one warmup and
three measured operations by default. Timers use `perf_counter` and wait for
provider jobs to complete. The operation includes component construction,
compilation and execution; it is not GPU-kernel time. Tutorial timing also includes
artifact output. Repeats reuse the same seed to measure runtime variability, not
independent statistical realizations. Molecular default budgets are intentionally
small: most optimizers get 30 iterations; adaptive algorithms get six growth
iterations; sampled subspaces get one iteration and 1,024 shots. Actual resolved
options are recorded per case.

| Field / file | Meaning |
| --- | --- |
| `case_id` | Hash of declared workload config, excluding device; compare source and provider provenance too |
| `campaign` | Independent execution directory; never average different campaigns implicitly |
| `phase` | `warmup`, `measure` or `failure` |
| `status` | `ok`, `unsupported`, `error` or `timeout`; mixed summary statuses remain failures for comparisons |
| `seconds` | Completed operation wall time, excluding setup; absent for failures |
| `setup_seconds` | Imports, input preparation and reference computation; repeated on each row, do not sum |
| `cold_process_seconds` | Whole worker wall time including all repeats and setup; repeated on rows, do not sum |
| `median_seconds` | Median measured successful repeat, excluding warmups |
| `requested_qubits` | Declared input spin/circuit register size, normalized from manifest during analysis |
| `qubits` | Actual result register where reported; native fermionic algorithms may have no qubit register |
| `effective_qubits` | Actual register in consolidated case summaries; their `qubits` field comes from the manifest |
| `spatial_orbitals`, `electrons`, `pauli_terms` | Molecular problem dimensions |
| `depth`, `gates`, `parameters`, `evaluations` | Circuit and variational-work dimensions where available |
| `statevector_bytes` | Theoretical complex128 reference-state storage, not measured GPU memory or single-precision allocation |
| `peak_process_rss_kib` | Worker peak resident host memory; excludes child-process and GPU allocation peaks |
| `array_shapes`, `output_array_bytes`, `input_bytes` | Tutorial data dimensions and payload sizes |
| `provider_metadata`, `resolved_options` | JSON cells preserving execution details and actual numerical controls |
| `cpu_over_gpu` | Matched CPU median / GPU median; above 1 means GPU faster |

JSONL is the raw evidence. CSVs are rectangular with nested structures serialized
as JSON cells and missing values left empty. Times are seconds, energies Hartree,
memory KiB or bytes as named. Logs preserve tracebacks and provider warnings.
`analysis/diagnostics.csv` retains unsupported and failed cells; do not replace
their missing durations with zero. The analysis tool rejects duplicate/out-of-order
iterations, missing warmups/repeats, invalid timings and inconsistent device records.
Speed comparisons require matching source fingerprints, numerical providers (including
solver/addon versions and Conda builds), Python, hardware and resource allocation.
Older or intentionally unmatched campaigns can still be exported with `--tables-only`;
that mode does not generate speedups or plots. Existing artifacts are not removed or
rewritten on validation failure, so use a separate copy of historical campaigns when
regenerating an analysis.
It does not prove equal background load; read each run report's contention caveat.

## Reproduction

Run from the repository root using an isolated interpreter with the required
profile. The CUDA installation recipe is in `docs/engines/installing.md`. Never
install CPU and GPU Aer wheels over each other. `--output` must be a new directory.

```bash
CPU=/path/to/cpu/environment/bin/python
CUDA=/path/to/cuda/environment/bin/python
OUT=benchmarks/quantum/runs/new-run

"$CPU" scripts/quantum_benchmarks.py --output "$OUT/cpu" --device cpu --suite all --timeout 120
CUDA_VISIBLE_DEVICES=0 "$CUDA" scripts/quantum_benchmarks.py --output "$OUT/gpu0" --device cuda --suite all --timeout 120
# Same compiled provider for fair CPU/GPU primitive comparison:
CUDA_VISIBLE_DEVICES=0 "$CUDA" scripts/quantum_benchmarks.py --output "$OUT/cpu-matched" --device cpu --suite primitives --timeout 120
QISKIT_NUM_PROCS=2 "$CPU" scripts/quantum_benchmarks.py --output "$OUT/cpu-fixed" --suite regressions --timeout 120
"$CPU" scripts/quantum_audit.py --output "$OUT/audit" --graph-grid
"$CPU" scripts/quantum_analysis.py "$OUT" --cpu cpu-matched --gpu gpu0 --plots
```

For a smaller matched-stack confirmation, run these commands sequentially after
other validation jobs finish (the first command still requests CPU execution):

```bash
CUDA_VISIBLE_DEVICES=0 QISKIT_NUM_PROCS=1 "$CUDA" scripts/quantum_benchmarks.py --output "$OUT/cpu-serial" --device cpu --suite primitives --match aer_statevector
CUDA_VISIBLE_DEVICES=0 QISKIT_NUM_PROCS=1 "$CUDA" scripts/quantum_benchmarks.py --output "$OUT/gpu0-serial" --device cuda --suite primitives --match aer_statevector
"$CPU" scripts/quantum_analysis.py "$OUT" --cpu cpu-serial --gpu gpu0-serial --plots
```

Run campaigns serially on an otherwise idle host for publication-quality timing.
Pin driver/runtime/builds and numerical thread settings. The driver sets OMP,
OpenBLAS and MKL thread counts per case; Aer cores do not necessarily bound
Qiskit's separate transpiler process pool. GPU ordinal selection is explicit via
`CUDA_VISIBLE_DEVICES`; manifests retain physical GPU UUIDs. No multi-GPU speedup
claim follows from testing two GPUs separately.

## Ongoing Use

Keep small correctness checks in `tests/test_quantum_benchmarks.py` and real
pipeline integration in `tests/test_engines_qiskit_pipeline_live.py`. Run the
bounded campaign after provider upgrades and larger cases in a scheduled dedicated
performance job. Retain raw evidence and environment manifests as immutable CI
artifacts; add a new run directory for reruns. Keep scripts, methodology and compact
summaries in source control, and move large historical raw campaigns to artifact
storage with checksums and a retention policy. Do not add runtime performance
thresholds until stable repeated measurements establish a baseline on fixed hardware.
