# Quantum validation and reproducibility

The local release gate, provider tests and benchmark tools check different claims.
A successful small-system calculation establishes that tested configuration;
it does not establish chemical accuracy, universal CUDA support or successful
quantum-hardware execution. The [support status](../engines/qiskit-status.md)
records those boundaries.

## Select the interpreter and checkout

Run these commands from the source checkout. Use an interpreter with the required
dependencies that imports this checkout's `src/chemrefine`, rather than a different
editable installation. Changing directories does not activate a different environment.

```bash
python -c 'import sys, chemrefine; print(sys.executable); print(chemrefine.__file__)'
```

The local PR gate requires a clean working tree, a `[dev,qiskit-toolkit]` environment
and a Python 3.12 `[qiskit-resources]` worker with `pytest-cov>=7` and `coverage>=7.10`.
Set `PY` and `RESOURCE_PY` to the actual interpreter paths if those are separate.
It checks formatting, types, main and resource coverage, extras resolution,
documentation, package installation and selected scientific mutations:

```bash
scripts/release-check.sh --pr
```

If unrelated local work prevents that gate, validate committed changes in a separate
clean worktree with environments installed against that worktree. Do not use a global
`PYTHONPATH` override for the release gate: its wheel and sdist checks must import the
installed artifacts, not the checkout.

## Check paths the default suite excludes

The default suite deselects `integration` and `perf`. Run the molecular-to-experiment
pipeline and cache scaling tests explicitly when changing those paths:

```bash
python -m pytest -o addopts='' -m 'integration and not gpu' \
  tests/test_engines_qiskit_pipeline_live.py -q
python -m pytest -o addopts='' -m perf tests/test_perf_cache.py -q
```

With a provisioned, compatible CUDA Aer interpreter, run:

```bash
python -m pytest -o addopts='' -m gpu -q \
  tests/test_engines_qiskit_aer_live.py \
  tests/test_engines_qiskit_pipeline_live.py \
  tests/test_quantum_benchmarks.py
```

Use that CUDA interpreter rather than a CPU-only environment. Required GPU checks
must pass without skips. They check actual method/device execution, single/double
arithmetic, pipeline handoff/recovery and a completed primitive workload; device
discovery alone proves less. Known tensor-network failures remain documented in
[installation](../engines/installing.md#aer-cuda-working-paths-and-known-failures).

GitHub CI currently triggers on pushes and PRs targeting `main`, plus explicit dispatch.
A PR targeting another branch does not automatically get that matrix. Hosted CI adds
the Python-version matrix and dependency-floor/provider jobs; the optional CUDA
workflow requires its separately configured runner and reviewed `main`.

## Preserve validation evidence

`scripts/quantum_validation.py` runs a supplied argument vector without a shell and
preserves the command's exit status, full output and environment snapshot. Use the
same interpreter for the wrapper and its tested command:

```bash
python scripts/quantum_validation.py \
  --output /tmp/chemrefine-quantum-evidence --label pipeline -- \
  python -m pytest -o addopts='' -m 'integration and not gpu' \
  tests/test_engines_qiskit_pipeline_live.py -q \
  --junitxml=/tmp/chemrefine-quantum-evidence/pipeline.xml
```

The wrapper writes `pipeline.json` and `pipeline.txt`; pytest writes the JUnit report.
The snapshot includes the source revision and dirty-tree summary, executing package
location and source hashes, installed versions, CPU/GPU information and numerical
thread settings. Conda environments also retain native package build metadata.
The JUnit report and return code establish test outcomes; a snapshot by itself does not.
`tests/provider_reports.py` additionally rejects missing, failed or skipped required
provider tests in CI.

For dependency and import review, `scripts/quantum_audit.py` exports the current
requirements compared with a chosen Git revision and the quantum import edges:

```bash
python scripts/quantum_audit.py --output /tmp/chemrefine-quantum-audit \
  --baseline origin/rewrite
```

Choose a baseline appropriate to the comparison. `--graph-grid` additionally exports
the finite registry-name compatibility grid. This audits the configuration graph;
it does not execute every numerical combination or certify arbitrary component options.

## Bounded quantum benchmarks

`scripts/quantum_benchmarks.py` runs a declared suite in isolated worker processes,
with a per-case timeout and separate warmup, setup, cold-process and operation timings.
Its suites are `primitives`, `molecular`, `tutorials`, `regressions` and `runtime`, or
`all`; `--match` and `--limit` narrow the selected cases. For a small local run:

```bash
python scripts/quantum_benchmarks.py --suite primitives --device cpu \
  --output /tmp/chemrefine-quantum-benchmark --limit 1 \
  --warmups 1 --repeats 3 --timeout 120 --qiskit-processes 1
```

Retain `manifest.json`, `samples.jsonl`, `summary.csv`, logs and provider journals.
Failed, unsupported and timed-out cases remain explicit and never become successful
timings. `--qiskit-processes` bounds transpiler workers independently of Aer's thread
budget. The Runtime suite uses local fake backends and submits no remote quantum
jobs. Real hardware configuration is covered separately in the
[Runtime guide](../engines/qiskit-runtime.md).

`scripts/quantum_workloads.py` supplies the case definitions and worker execution;
it is imported by the benchmark driver, not a separate public CLI. The declared grid
does not exhaust arbitrary circuits, molecule sizes or method options.

`scripts/quantum_analysis.py ROOT --tables-only` consolidates completed evidence
without inferring speed ratios. A CPU/GPU comparison additionally names matching run
directories with `--cpu` and `--gpu`; optional `--plots` exports scaling figures.
Comparison validates source/environment identity, execution options, completed
repetitions and numerical checks. Preserve those inputs with the resulting tables;
operation timing includes compilation and synchronous execution, not GPU kernels alone.

## Preserve the scientific inputs too

Environment evidence complements [integral and preparation checkpoints](../engines/qiskit-checkpoints.md).
Keep geometry, charge/multiplicity, basis, active orbitals, tensor conventions, scalar
energy offsets, random seeds, selected components and provider settings. Save complete
bundles rather than just descriptor JSON. Final preparations and trajectories do not
serialize optimizer state or provide partial solver restart.
