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
chemrefine rebuild-nms input.yaml 3          # re-run the NMS step with current options
chemrefine run input.yaml --maxcores 128     # override max_cores from the YAML
chemrefine run input.yaml --dry-run          # validate + describe; submit nothing
```

## Commands

| Command | Argument(s) | What it does |
|---------|-------------|--------------|
| `run` | `CONFIG` | Run the full pipeline from step 1, invalidating any existing cache. |
| `resume` | `CONFIG` | Honour the on-disk cache for unchanged steps and re-attempt the pending failed jobs of any `on_failure: stop` step, then continue. |
| `rerun-errors` | `CONFIG [TARGET]` | Re-attempt only one step's pending failed jobs (latest if no target); like `resume` but scoped to that step. |
| `rerun` | `CONFIG [TARGET]` | Redo one whole step from scratch (default: latest); others cache-hit. |
| `rebuild-cache` | `CONFIG [TARGET]` | Rebuild one step's cache from outputs already on disk (parse only, no submission). |
| `rebuild-nms` | `CONFIG [TARGET]` | Re-run the NMS step with the current options (a named alias of `rerun`). |

## Backend environments (`chemrefine backends`)

Conflicting MLIP stacks each live in one managed environment, provisioned once and
resolved **by name** at run time (see
[Installation → MLIP backends](installation.md#mlip-backends)):

| Command | Argument(s) | What it does |
|---------|-------------|--------------|
| `backends install` | `EXTRA…` | Provision managed env(s) (e.g. `mlip-mace mlip-fairchem pyscf`), built with the same tool that created the current env (conda / uv / venv). |
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

- **`skip`** (default) — drop the failed structures, keep the successes, continue.
- **`best`** — keep every structure, backfilling a failure with the best geometry
  obtained for it.
- **`stop`** — cache the step's successes, then halt the run so you can fix the
  failures and `resume` (or `rerun-errors N`).

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
corrupt, `8` wait deadline expired, `1` generic.
