# API Reference

Auto-generated reference for the `chemrefine` package, rendered from the module
docstrings. The orchestration core is engine-agnostic; concrete engines plug in
behind the `CalculationEngine` Protocol and the `ENGINES` registry.

```
Config → bootstrap → run_step (cache | engine lifecycle) → filtering → PipelineState
```

## Core

- [Configuration](config.md) — Pydantic models + YAML loader and legacy normalizer
- [Introspection](introspect.md) — machine-readable view of the config schema and the
  engine registry
- [Validation](validate.md) — a structured report, the non-raising twin of `load_config`
- [Scaffolding](scaffold.md) — starter templates for the files a config names but the
  filesystem lacks
- [Runtime State](state.md) — the frozen dataclasses threaded between stages
- [Errors](errors.md) — the exception hierarchy and exit codes
- [Quantities](quantities.md) — physical constants, conversions, Boltzmann weights

## Pipeline

- [Orchestrator](pipeline.md) — `bootstrap` + `run`
- [Step Lifecycle](step.md) — `run_step` and the cache/rebuild paths
- [Step Lifecycle Body](lifecycle.md) — run, classify, `stop` / `skip` / `best`, persist
- [Attempt Directories](attempts.md) — sealing a structure's state into `attemptK/`
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
- [Q-Chem](engines_qchem.md) — input generation, output parsing, the NMS hooks

## Agent Integration

- [Agent Tools](agent_tools.md) — the tool surface both the MCP server and the
  embedded agent expose
- [MCP Server](mcp_server.md) — those same tools served over the Model Context Protocol
- [Embedded Agent](agent.md) — `chemrefine agent`, the terminal chat that drives them
  without an MCP client

## Infrastructure

- [SLURM](slurm.md) — script generation, submission, array jobs, polling
- [Throttle](throttle.md) — the CPU+GPU budget throttler
- [Cache](cache.md) — the per-step fingerprint cache, manifest, and failed-job ledger
- [Filesystem I/O](io.md) — XYZ read/write, SMILES→3D, CSV reporting
- [Identifiers](ids.md) — hierarchical ID allocation and canonical filenames
- [Job Log](job_log.md) — the per-job runlog header/footer
