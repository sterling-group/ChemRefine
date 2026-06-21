# Architecture & Code Flow

ChemRefine is layered one-directionally: the CLI dispatches a recovery action,
the pipeline iterates steps, each step drives an engine through a fixed lifecycle,
and the engine submits jobs through the SLURM/throttle machinery. State flows
forward as immutable `PipelineState` values.

## Module layering

```
cli → recovery → pipeline → step → {cache, filtering, step_failures, nms}
                                  → engines.base (Protocol + ENGINES registry)
                                       → engines/* (orca, mlip, pyscf, _fake)
                                            → slurm, throttle, io, ids, job_log, quantities
```

Only `cli` touches `sys.argv` / process exit; only `slurm` shells out to
`sbatch` / `squeue`; only `io` and `cache` own the on-disk formats. Engines
receive a focused `StepContext`, never the whole `Config`.

## Run flow

What happens when you run `chemrefine run input.yaml`:

```mermaid
flowchart TD
  CLI["cli.main → _translate_legacy_argv → Typer app"] --> DISP["cli._dispatch"]
  DISP --> REC["recovery.execute (_HANDLERS)"]
  REC --> RUN["pipeline.run"]
  RUN --> BOOT["bootstrap: seed PipelineState\n(.xyz / dir / SMILES csv)"]
  RUN --> LOOP{"for each step"}
  LOOP --> STEP["step.run_step"]
  STEP --> CACHE{"cache.load_if_valid\nfingerprint match?"}
  CACHE -- hit --> FILT["filtering.apply"]
  CACHE -- miss --> LIFE["engine lifecycle:\nprepare → submit → wait → parse"]
  LIFE --> POL["step_failures.apply_failure_policy\n(+ nms.run_nms for nms steps)"]
  POL --> SAVE["cache.save"]
  SAVE --> FILT
  FILT --> CSV["io.save_step_csv → steps.csv"]
  CSV --> NEXT{"survivors?"}
  NEXT -- yes --> LOOP
  NEXT -- no --> STOP["stop early"]
```

## Submit / compute flow

Inside `engine.submit`, jobs are generated and throttled against the CPU+GPU
budget. The ExtOpt engines additionally stand up a gradient server that ORCA
talks to per optimisation step:

```mermaid
flowchart TD
  SUB["SlurmBatchEngine.submit"] --> BUILD["slurm.build_script /\nbuild_array_script"]
  BUILD --> SUBMIT["slurm.submit"]
  SUBMIT -- sbatch present --> SBATCH["sbatch (SLURM)"]
  SUBMIT -- local --> POPEN["bash Popen (background)"]
  SUB --> THR["throttle.Throttler\nwait_for_room (cores + gpus)"]

  subgraph ExtOpt["ExtOpt engines (mlip-extopt / pyscf-extopt)"]
    RB["run_block starts\n_backend_server.server"] --> ORCA["ORCA %method ProgExt"]
    ORCA --> BRIDGE["extopt.bridge (per step)"]
    BRIDGE --> CALC["server /calculate →\nComputeBackend.calc"]
    CALC --> ENGRAD[".engrad → ORCA next step"]
  end
```

## Key types

- `PipelineState` — the tuple of `Structure` survivors threaded between steps.
- `Structure` — one geometry plus energy / forces / status flags; carries its
  `parent_id` so lineage (and `by_parent` filtering) is explicit.
- `StepContext` — the per-step input bundle handed to an engine.
- `StepInputs` / `JobBatch` / `StepResults` — the engine lifecycle hand-offs.

See the [API Reference](../api/index.md) for the full contracts.
