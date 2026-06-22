# API Reference

Auto-generated reference for the `chemrefine` package, rendered from the module
docstrings. The orchestration core is engine-agnostic; concrete engines plug in
behind the `CalculationEngine` Protocol and the `ENGINES` registry.

```
Config → bootstrap → run_step (cache | engine lifecycle) → filtering → PipelineState
```

## Core

- [Configuration](config.md) — Pydantic models + YAML loader and legacy normalizer
- [Runtime State](state.md) — the frozen dataclasses threaded between stages
- [Errors](errors.md) — the exception hierarchy and exit codes
- [Quantities](quantities.md) — physical constants, conversions, Boltzmann weights

## Pipeline

- [Orchestrator](pipeline.md) — `bootstrap` + `run`
- [Step Lifecycle](step.md) — `run_step` and the cache/rebuild paths
- [Failure Policy](step_failures.md) — `stop` / `skip` / `best` resolution + the
  shared attempt/retry primitive
- [Normal-Mode Sampling](nms.md) — engine-independent two-round NMS coordinator
- [Filtering](filtering.md) — survivor selection
- [Recovery Actions](recovery.md) — the `run` / `resume` / `rerun…` dispatcher

## Engines

- [Contract & Registry](engines_api.md) — the `CalculationEngine` / `NmsCapableEngine` /
  `JobExecutable` Protocols, the DTOs, and the `ENGINES` registry
- [Job & Script Engines](engines_job.md) — the `JobEngine` lifecycle + `build_structures`
  assembler, the `_execution` scheduler, and the `ScriptEngine` kind
- [ORCA](engines_orca.md) — input generation, output parsing, frequencies, the NMS hooks, ExtOpt
- [MLIP](engines_mlip.md) — direct / ExtOpt / training engines + backend dispatcher
- [PySCF](engines_pyscf.md) — direct / ExtOpt engines + the SCF runtime

## Infrastructure

- [SLURM](slurm.md) — script generation, submission, array jobs, polling
- [Throttle](throttle.md) — the CPU+GPU budget throttler
- [Cache](cache.md) — the per-step fingerprint cache, manifest, and failed-job ledger
- [Filesystem I/O](io.md) — XYZ read/write, SMILES→3D, CSV reporting
- [Identifiers](ids.md) — hierarchical ID allocation and canonical filenames
- [Job Log](job_log.md) — the per-job runlog header/footer
