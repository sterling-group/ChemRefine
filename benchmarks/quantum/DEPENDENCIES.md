# Dependency and integration audit

## Scope

The audit compares `pyproject.toml` at `df3d37a^` (the parent of the original
Qiskit estimator-provider introduction) with this checkout. "New" means absent
from every dependency group at that baseline, not necessarily added by the most
recent commit. `runs/2026-09-28/audit/dependencies.csv` includes every group,
version constraint, previous constraint and change classification. This work
adds no third-party runtime dependency.

## Newly declared distributions

| Ownership / extra | New distributions |
| --- | --- |
| Core | `packaging` |
| `qiskit-core` | `qiskit`, `qiskit-nature`, `qiskit-algorithms` |
| `qiskit-aer` | `qiskit-aer` |
| `qiskit-fermionic` | `ffsim`, `qiskit-fermions`, `qiskit-addon-sqd` |
| `qiskit-runtime` | `qiskit-ibm-runtime`, `qiskit-mitigation`, `samplomatic` |
| `qiskit-cutting` | `qiskit-addon-cutting` |
| `qiskit-resources` | `openfermion[resources]`, `qualtran` |
| `qiskit-rdm` | `cvxpy`, `scs` |
| `mcp` | `mcp` |
| `agent` | `pydantic-ai-slim[openai]` |
| Test tooling | `coverage`, `nodejs-wheel-binaries` |
| Development tooling | `build`, `pip-audit` |

There are **22 unique new distributions**. Qiskit appears in two extras but is
counted once. `qiskit-fermionic` raises its required Qiskit floor to 2.5, while
`qiskit-core` permits 1.4. New self-references such as `chemrefine[qiskit-aer]`
compose extras; they are not new packages. Existing requirements also changed:
Pydantic's floor, SevenNet's floor, ORB's range and Python markers, and Python
markers for MACE, Torch, e3nn and CHGNet. Consult the CSV for exact changes.

Installed transitive dependencies are captured, with exact versions, in each
validation JSON and benchmark `manifest.json`. CUDA manifests additionally
record Conda build identifiers and available package hashes. Transitive additions
cannot be historically inferred from an old `pyproject.toml` without that old
environment's lockfile, so they are not mislabeled as historical additions.

## Internal Coupling

The Qiskit package imports these shared ChemRefine modules outside its own package:

- `chemrefine.config`
- `chemrefine.engines._execution`, `._input_files`, `._options`, `._provision`
- `chemrefine.engines._script`, `._script.contract`, `._script.render`
- `chemrefine.engines.api`
- `chemrefine.errors`, `chemrefine.input_files`, `chemrefine.io`, `chemrefine.state`

`audit/import_edges.csv` lists the source file and line for each static import,
including reverse consumers. There is **no direct import of another engine's
implementation**, including the PySCF, ORCA, Q-Chem and MLIP engines.

There is a packaging dependency: `qiskit` includes `qiskit-core` and `pyscf`;
`pyscf` in turn includes `server`, bringing Flask and Waitress. The external
PySCF SDK supplies Qiskit Nature's geometry/integral adapter; it does not run the
ChemRefine PySCF engine. Integral-bundle workflows do not need PySCF and can use
the core profile. Keep the distinction between SDK, engine and extra explicit.

Both quantum engines use the common options/schema, worker execution, input
staging, cache and artifact interfaces. The GUI consumes their published schema
and capability metadata. Added GUI tests exercise both real tutorial configs,
the complete component catalog and nested component edits. A live pipeline test
executes molecular preparation, circuit export and measurement on CPU and CUDA,
then verifies cache reuse. This is contract/integration testing, not a visual
browser acceptance test or proof that unavailable external engines work.

## Maintenance Policy

1. Keep `pyproject.toml` as the authoritative list of direct dependencies. Keep
   provider imports lazy and attach provider requirements to component registries.
   Do not move optional quantum stacks into GUI or core imports.
2. Keep worker environments separate: core/Aer, fermionic/toolkit, resource
   estimation, and each MLIP family. `qiskit-toolkit` deliberately excludes the
   resource profile. These profiles have different release cadences and Python
   requirements; MLIP families also have conflicting Torch/e3nn requirements.
   This audit does not claim the resource and toolkit profiles are inherently
   incompatible, or certify a universal environment containing every extra.
3. Maintain reproducible, platform-specific constraints or locks for deployments
   and benchmark machines. Record Python, OS, exact SDK versions, CUDA runtime,
   driver, GPU model and Aer build together. Package metadata should retain tested
   ranges; a developer's full freeze should not become runtime requirements.
4. Continue the existing `.github/workflows/quantum-providers.yml` lower-bound and
   current-version jobs. Include `pip check`, real provider execution, GUI/schema
   contracts, bundle round-trips and SDK-free discovery in upgrade checks. Test
   CUDA builds on a dedicated GPU runner before extending support claims.
5. Keep SDK floor upgrades scoped by profile. The core 1.4 floor and fermionic 2.5
   floor are different contracts. Document Python availability per profile and
   retain separate resource-provider test evidence.
6. Treat GPU Aer as a mutually exclusive compiled provider, not an extra pip wheel
   to layer over CPU Aer. The available PyPI GPU 0.15.1 wheel does not satisfy the
   declared 0.17 series and fails with Qiskit 2.5. This run uses conda-forge's
   CUDA 12.9 Aer 0.17.2 build. The installation guide now contains the verified
   recipe. Do not let pip silently replace that build with the CPU wheel.
7. Keep test-only Node, coverage, build and security tools outside runtime extras.
   Benchmark execution uses the standard library plus installed engine providers;
   optional chart generation uses Matplotlib already available in this validation
   environment. Data analysis is not a core runtime dependency.
8. Review direct SciPy imports when changing numerical profiles. SciPy is currently
   supplied transitively by existing numerical dependencies; declaring it directly
   in a future numerical-provider cleanup would make that contract clearer.
9. Review dependency updates and vulnerability reports regularly. Retain resolver
   reports and deployment locks with release artifacts; do not automatically relax
   caps to resolve one optional provider's conflict.

All four good validation environments passed `pip check`. This does not constitute
a vulnerability scan or certify every possible combination of optional extras.

GPU capability and build references: [Aer simulator documentation](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html),
[conda-forge Aer builds](https://anaconda.org/conda-forge/qiskit-aer/files), and
[Aer build instructions](https://github.com/Qiskit/qiskit-aer/blob/main/CONTRIBUTING.md).
