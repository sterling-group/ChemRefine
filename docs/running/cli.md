# CLI Reference

ChemRefine is invoked as `chemrefine <command> CONFIG [TARGET] [flags]`. Every
command takes the YAML config path; the recovery commands optionally take a step
**target** (a step number or its `name:`, defaulting to the latest step).

```bash
chemrefine run input.yaml                    # full pipeline from step 1
chemrefine resume input.yaml                 # honour the cache; retry pending failures
chemrefine rerun input.yaml refine           # redo one whole step (others cache-hit)
chemrefine rerun-errors input.yaml 2         # re-attempt only step 2's failed jobs
chemrefine rebuild-cache input.yaml          # re-parse outputs on disk (no submission)
chemrefine rebuild-nms input.yaml 3          # re-resolve NMS from outputs on disk (no submission)
chemrefine run input.yaml --maxcores 128     # override max_cores from the YAML
chemrefine run input.yaml --dry-run          # validate + describe; submit nothing
```

## Commands

| Command | Argument(s) | What it does |
|---------|-------------|--------------|
| `run` | `CONFIG` | Run the full pipeline from step 1, invalidating any existing cache. |
| `resume` | `CONFIG` | Honour the on-disk cache for unchanged steps and re-attempt the pending failed jobs of any `on_failure: stop` step, then continue. |
| `rerun-errors` | `CONFIG [TARGET]` | Re-attempt one step's pending failed jobs (latest if no target), then continue: earlier steps cache-hit and submit nothing, later ones resume. |
| `rerun` | `CONFIG [TARGET]` | Redo one whole step from scratch (default: latest); every other step resumes, so one whose fingerprint no longer holds runs again. |
| `rebuild-cache` | `CONFIG [TARGET]` | Rebuild one step's cache from outputs already on disk (parse only, no submission). The steps after it are then re-reported from their caches, stopping quietly at the first one that no longer matches the configuration. |
| `rebuild-nms` | `CONFIG [TARGET]` | Re-resolve the NMS step from outputs already on disk (parse only, no submission). With no TARGET it finds the step setting `nms: true` rather than the last one. Round 1 is re-parsed and its displaced children re-read from their `attemptK/`, so re-resolving costs a read rather than a re-run of the frequencies. The steps after it are re-reported from their caches, like `rebuild-cache`. |

## Config tooling (`validate`, `scaffold`, `schema`, `engines`)

| Command | Argument(s) | What it does |
|---------|-------------|--------------|
| `validate` | `CONFIG [--json]` | Validate without running and report **every** finding at once: pydantic errors with their field locations, unknown engines, bad values for declared option knobs, invalid NMS knobs, plus warnings for silent no-ops (undeclared option keys, `nms: true` on an engine that cannot NMS) and for step templates or SLURM headers that do not exist yet. Warnings never affect the exit code; an unrunnable config exits 2. |
| `scaffold` | `CONFIG [--overwrite]` | Write a commented starter into every template file the config expects but lacks — step templates and the SLURM header(s) dispatch would pick. Existing files are kept unless `--overwrite`. |
| `schema` | — | Print the machine-readable schema document as JSON: the config schema, the NMS knob schema, and a descriptor per registered engine (capabilities, declared options schema). What a GUI form or an agent reads instead of this page. |
| `engines` | `[--json]` | List the registered engines: template kind, whether the engine declares an options model or is configured entirely through its template, and its capabilities. |

A first run usually goes: `chemrefine validate input.yaml` → `chemrefine scaffold
input.yaml` → edit the starters → `chemrefine run input.yaml`.

Two more entry points build on the same tooling: [`chemrefine gui`](../workflow/builder.md) (the
click-through workflow builder — [from a cluster](../workflow/builder.md#from-a-cluster) it prints
its own SSH forwarding recipe) and [`chemrefine mcp`](../workflow/agents.md) (the tool server for AI
agents).

## Backend environments (`chemrefine backends`)

Conflicting MLIP stacks each live in one managed environment, provisioned once and
resolved **by name** at run time (see
[Installation → available backends](../engines/installing.md#available-backends)):

| Command | Argument(s) | What it does |
|---------|-------------|--------------|
| `backends install` | `EXTRA…`, `--python` | Provision managed env(s) (e.g. `mlip-mace mlip-fairchem pyscf`), built with the same tool that created the current env (conda / uv / venv) and on the newest Python each backend supports. `--python` (a version, a command name, or a path) overrides that choice when the env is created. |
| `backends list` | — | Every known backend extra and whether its env is provisioned. |
| `backends path` | `EXTRA` | Print the managed env's `python` (exit 1 if not provisioned). |

## Global flags

| Flag | Applies to | Effect |
|------|-----------|--------|
| `--maxcores INT` | all run commands | Override `max_cores` from the YAML (≥ 1). Flag beats YAML. |
| `--maxgpus INT` | all run commands | Override `max_gpus` from the YAML (≥ 0). Flag beats YAML; omit to keep the YAML value (`None` ⇒ auto-resolve). |
| `--dry-run` | all run commands | Load and validate the config and describe the would-be actions; submit nothing. |
| `-v`, `--verbose` | global | Debug-level logging. |
| `--version` | global | Print the ChemRefine version and exit. |

## Failure handling

Per-step `on_failure` (in the YAML) decides in-run behaviour:

- **`stop`** (default) — cache the step's successes, then halt the run so you can fix
  the failures and `resume` (or `rerun-errors N`). The default so failures are never
  silently dropped.
- **`skip`** — drop the failed structures, keep the successes, continue.
- **`best`** — keep every structure, backfilling a failure with the best geometry
  obtained for it. A backfilled structure carries no thermochemistry, so it is
  excluded from a `sample` that ranks on `gibbs` / `enthalpy` /
  `electronic_zero_point` rather than aborting the step.

The `failed_jobs.json` ledger always records which structures failed (so they are
visible), but only `stop` failures are *pending* for `resume` / `rerun-errors` to
re-attempt.

## Legacy flag-style invocations

v1.3.1 flag-style commands are auto-translated (one deprecation warning) to the
subcommands above:

| Legacy | v2 |
|--------|----|
| `chemrefine CONFIG` | `chemrefine run CONFIG` |
| `CONFIG --skip` | `chemrefine resume CONFIG` |
| `CONFIG --rebuild_cache [N]` | `chemrefine rebuild-cache CONFIG [N]` |
| `CONFIG --rebuild_nms [N]` | `chemrefine rebuild-nms CONFIG [N]` |
| `CONFIG --rerun_errors [N]` | `chemrefine rerun-errors CONFIG [N]` |
| `--maxcores N` | unchanged |

## Exit codes

Each failure mode maps to a deterministic exit code (see the
[Errors API](../api/errors.md)): `2` config invalid, `3` unknown engine,
`4` job submission refused, `5` job failed, `6` output unparseable, `7` cache
corrupt, `8` wait deadline expired, `9` backend env could not be built,
`10` another driver holds the output tree's run lock, `1` generic.
