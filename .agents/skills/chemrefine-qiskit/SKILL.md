---
name: chemrefine-qiskit
description: >-
  Configure, run, diagnose, test, or extend ChemRefine's modular Qiskit Nature
  engine, including exact solvers, VQE, ADAPT-VQE, ansatze, estimators,
  optimizers, mappers, active spaces, and component registries. Use when work
  involves a ChemRefine Qiskit-engine step or files under
  src/chemrefine/engines/qiskit.
---

# ChemRefine Qiskit

Operate the Qiskit engine through its typed component graph. Keep scientific choices in
YAML and keep the shipped Python template thin.

## Start with the source of truth

1. Read `docs/user-guide/qiskit.md` for the public configuration and architecture.
2. Inspect `src/chemrefine/engines/qiskit/options.py` and `registry.py` before changing
   option or extension contracts.
3. Inspect the relevant factory in `src/chemrefine/engines/qiskit/components/` before
   claiming an option or capability exists.
4. Reuse `examples/tutorials/qiskit_sp/templates/step1.py` unless a custom component
   must be imported in the worker.

Do not infer behavior from generic Qiskit examples when the repository contract differs.

## Choose a workflow

- For a small reference calculation, select `algorithm: exact` and define a defensible
  active space. Exact diagonalization does not build variational components.
- For fixed VQE, select `algorithm: vqe`, a circuit-capable ansatz, estimator, optimizer,
  and initial point.
- For adaptive VQE, select `algorithm: adapt_vqe` and an ansatz that supplies an
  `operator_pool`. Built-in UCCSD does; EfficientSU2 does not. Keep both ADAPT and UCCSD
  `reps` at `1` with the supported dependency versions.
- For deterministic local validation, prefer `statevector` with
  `default_precision: 0.0`, `initial_point: zeros`, and `slsqp`.
- For local finite-shot studies, use `basic_backend`, set `seed_simulator`, and consider
  seeded SPSA. Do not describe it as hardware, Aer, or a noise model.
- Use `aer_statevector` for Aer-backed exact expectation values. Keep
  `default_precision: 0.0`; positive precision adds Gaussian perturbations, not shots.
- Use `aer_shots` for actual finite-shot sampling through `BackendEstimatorV2`. A
  positive `default_precision` implies approximately `ceil(1 / precision**2)` shots.
  It is ideal unless `noise_model` contains a serialized Aer `NoiseModel`, and even a
  noisy simulator is not quantum hardware.

Set geometry, charge, multiplicity, basis, active space, and every random seed explicitly
when reproducibility matters. Never place provider credentials in ChemRefine YAML.

## Validate before running

From the repository root:

```bash
python -c "from chemrefine.config import load_config; load_config('PATH/TO/input.yaml')"
chemrefine run PATH/TO/input.yaml --dry-run
```

Run the calculation only when the user requested execution. The core stack can be
installed with either `pip install "chemrefine[qiskit]"` or
`chemrefine backends install qiskit`. Aer estimators use the separate
`chemrefine[qiskit-aer]` / `chemrefine backends install qiskit-aer` environment. Request
approval before installing dependencies. Standard Aer is CPU-only; `device: cuda`
requires a compatible Linux `qiskit-aer-gpu` environment and must never be treated as a
hardware submission.

After a run, inspect:

- `outputs/steps.csv` for the total molecular energy;
- `outputs/stepN/<id>/stepN_<id>.json` for fully resolved components, variational
  evaluations, qubit counts, and solver diagnostics;
- the adjacent `.result.json` for ChemRefine's canonical structure record.

Do not compare callback `objective_value_hartree` directly with the final molecular
energy: callback values exclude nuclear-repulsion and active-space constants.

## Extend one component at a time

Use the registry matching the replacement boundary:

| Need | Registry | Required return |
| --- | --- | --- |
| fermion mapping | `MAPPERS` | mapper |
| reference state | `INITIAL_STATES` | circuit |
| circuit or adaptive pool | `ANSATZE` | `AnsatzArtifacts` |
| expectation execution | `ESTIMATORS` | `EstimatorResource` |
| classical search | `OPTIMIZERS` | optimizer |
| starting parameters | `INITIAL_POINTS` | NumPy vector |
| solver strategy | `ALGORITHMS` | `AlgorithmArtifacts` |

For every new component:

1. Define a frozen Pydantic options model with `extra="forbid"`.
2. Register a stable lowercase name and declare ansatz capabilities or algorithm
   requirements accurately.
3. Import optional Qiskit/provider packages inside the builder, never at engine discovery
   time.
4. Return the registry's expected artifact. Put provider/session cleanup and any required
   transpiler in `EstimatorResource`.
   For a separately installed estimator provider, pass
   `backend_requirement=BackendRequirement(extra=..., import_name=...)` to
   `ESTIMATORS.register`; preflight and managed-environment selection derive from it.
5. Ensure the registration module loads in both the orchestrator and backend worker.
   In-tree built-ins are imported from `components/__init__.py`; third-party automatic
   entry-point discovery does not yet exist.
6. Add strict option, compatibility, lifecycle, lazy-import, and failure-path tests.
7. Update `docs/user-guide/qiskit.md` and a runnable example when the public surface changes.

Fail early with `ConfigError` for incompatible artifacts, empty adaptive pools,
zero-parameter VQE circuits, or invalid runtime results.

## Verify changes

Run the focused checks first:

```bash
pytest tests/test_engines_qiskit.py -q
pytest tests/test_engines_contract.py -q
ruff check src/chemrefine/engines/qiskit tests/test_engines_qiskit.py
ruff format --check src/chemrefine/engines/qiskit tests/test_engines_qiskit.py
mypy src/chemrefine/engines/qiskit
interrogate -c pyproject.toml src/chemrefine/engines/qiskit
python scripts/mutation_gate.py -k qiskit
```

Then run the broader suite appropriate to the change. A real-stack smoke test should use a
small molecule such as the shipped H2 example and compare exact, VQE, and/or ADAPT energies
within a stated tolerance. Preserve unrelated worktree changes and report environment-only
dependency failures separately from code regressions.
