# Troubleshooting

## Exit codes

Every ChemRefine failure exits with a code that names the failure mode, so a
wrapper script or a SLURM chain can branch on it without parsing log text.

| Code | Meaning | Usual cause | What to do |
| --- | --- | --- | --- |
| `0` | Success | — | — |
| `1` | Generic failure | A step halted under `on_failure: stop`, or an error with no more specific code | Read the last log lines; if a step halted, see [pending failures](#a-step-halted-with-pending-failures) |
| `2` | Config invalid | A typo in `input.yaml` — unknown key, wrong type, non-contiguous `step:` numbers, a directory path with a shell metacharacter in it | The message quotes the offending field; fix and re-run |
| `3` | Unknown engine | `engine:` names something not in the registry | Check the spelling against the [configuration reference](configuration.md); legacy names (`mlff`, `dft`) are rewritten automatically |
| `4` | Job submission refused | `sbatch` rejected the script — bad partition, over a QoS limit, missing account | The message includes what `sbatch` said. To run without SLURM, set `dispatch: local` |
| `5` | Job ran but failed | A backend reported failure the pipeline could not recover from | Look at the structure's `.runlog` and `.err` under `outputs/stepN/<id>/` |
| `6` | Output unparseable | The calculation produced a file ChemRefine could not read — usually truncated by a time limit or a node failure | Inspect the `.out`; re-run the affected structures with `chemrefine rerun-errors N` |
| `7` | Cache corrupt or unwritable | An interrupted write, a full disk, or a cache written by a different ChemRefine version | `chemrefine rebuild-cache N` re-parses from the outputs already on disk without re-running anything |
| `8` | Wait deadline expired | A job outlived `job_timeout_seconds` | Raise the limit, or check whether the job is stuck in the queue (`squeue -u $USER`). Only reachable when you set `job_timeout_seconds`; the default waits indefinitely |

## A step halted with pending failures

`on_failure: stop` — the default — is deliberate: a step that loses structures
stops the run rather than quietly refining a smaller ensemble than you asked for.
The successes are already cached, and the failures are listed in
`outputs/stepN/_cache/failed_jobs.json`.

```bash
chemrefine resume input.yaml            # re-attempt every pending failure, then continue
chemrefine rerun-errors input.yaml 2    # re-attempt only step 2's failures
chemrefine rerun input.yaml 2           # redo step 2 from scratch
```

`resume` and `rerun-errors` regenerate the failed structures' inputs before
resubmitting, so a template edit made in between actually takes effect. Their
previous artifacts are moved to `outputs/stepN/<id>/attemptK/` rather than
overwritten — nothing is lost, and a job that dies without writing anything can
never be mistaken for a success.

## A structure keeps failing to converge

An unconverged structure is retried once per run from the best geometry it
reached, with the failed attempt archived under `attemptK/`. If it still fails,
it lands in the ledger. Common fixes: loosen the SCF convergence in the template,
give the optimiser more cycles, or start from a better geometry.

## The run reports fewer structures than expected

Check `steps.csv`. Each row records the step, the structure, and the energy the
step filtered on — including an `Energy type` column, so a step sampling on
`gibbs` shows Gibbs energies rather than electronic ones. If a step's survivor
count is lower than you expect, its `sample:` block is usually the reason.

## Nothing runs and the log says a backend is unavailable

MLIP backends install into their own environments because their torch/e3nn trees
conflict. Provision one and re-run:

```bash
chemrefine backends install mlip-mace
chemrefine backends list
```
