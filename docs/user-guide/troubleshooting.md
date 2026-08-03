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
| `8` | Wait deadline expired | Nothing finished for `job_timeout_seconds` | The queue has stalled, not merely slowed — a step that keeps completing jobs restarts the clock each time. Check whether the jobs are stuck (`squeue -u $USER`), or raise the limit. Only reachable when you set `job_timeout_seconds`; the default waits indefinitely |
| `9` | Backend env could not be built | `chemrefine backends install <extra>` failed — no network on the node, a resolver conflict, a full disk, or an env tool that is a shell function rather than a binary on `PATH` | The message quotes the command that failed. Run it on a machine with internet (on HPC: a login node); the half-built env is removed, so a re-run starts clean |

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

## The run was interrupted mid-step

A ChemRefine run is a long-lived process. If you submit it as a batch job (see
the [conformer-sampling tutorial](../tutorials/conformer_sampling.md)) it has a
walltime of its own, and a pipeline that outlives it is killed partway through a
step — as it is by a node failure or a `Ctrl-C`.

Just resume:

```bash
chemrefine resume input.yaml
```

The calculations that had already finished are **re-parsed from disk, not
resubmitted**. Only the structures whose output is missing go back to the
scheduler. ChemRefine can tell the difference because the step's manifest —
written before any job is submitted — records the same fingerprint the cache
would have, so outputs on disk are provably the ones this configuration asked
for. If the config changed in between, the fingerprint no longer matches and the
step is re-run in full rather than mixing results from two configurations.

Two things deliberately do *not* reuse that work:

- `chemrefine run` means start over, and always does.
- An `nms:` step falls back to a full re-run. Its round-2 children need
  re-resolving, not just re-parsing, and half-recovering that is worse than
  redoing it.

If you would rather drive it by hand — to inspect what survived before
continuing — `chemrefine rebuild-cache N` re-parses step N's outputs without
submitting anything, and ledgers whatever is missing.

## A structure keeps failing to converge

An unconverged structure is retried once per run from the best geometry it
reached, with the failed attempt archived under `attemptK/`. If it still fails,
it lands in the ledger. Common fixes: loosen the SCF convergence in the template,
give the optimiser more cycles, or start from a better geometry.

## A structure failed with "the calculation diverged"

```
unparseable: MLIP output …/step1_1.json reports a non-finite 'energy_hartree' (nan);
the calculation diverged
```

The `step{N}.py` template produced a `nan` or `inf` energy — an MLIP that
diverged on a strained geometry, an SCF that blew up. It is refused at the
parse boundary and ledgered like any other failed structure, because a
non-finite energy cannot be ranked: every comparison against it is false, so
it would sort by list position and displace a genuine survivor rather than
being filtered out.

Look at the geometry that produced it (`step{N}_{id}_inp.xyz`). If the model is
simply out of its depth there, `on_failure: skip` drops the structure and keeps
the rest. Gradients are held to the same rule, since a non-finite force is what
an `mlip-train` step would go on to fit.

## A `pyscf-extopt` step failed with "SCF did not converge"

```
PySCF SCF did not converge (E=… Eh, method=dft, xc=pbe, basis=def2-svp);
a gradient from a non-stationary density is not usable.
```

The gradient server refuses to answer with a result ORCA cannot use: ORCA would
otherwise take its next optimisation step on it and report *its own* geometry
convergence in the `.out`, which says nothing about the backend's SCF. The full
message is in `server_<jobid>.log` beside the structure's other artifacts.

Tighten the SCF (a better guess, more cycles, a smaller DIIS space), or set
`strict_scf: false` in the step's `options:` if the loose behaviour is what you
want — the energy is then whatever the last iteration produced.

## An `nms` step says a mode index is not imaginary

```
nms ts_mode_index=7 is not an imaginary mode of this structure;
its imaginary modes are [6].
```

`ts_mode_index` names the mode to *keep* during a `ts` search. It has to be one
this structure's frequency calculation actually flagged imaginary, and it cannot
be checked when the YAML loads — which modes are imaginary is a property of each
result. Use one of the indices in the message, or drop the setting to keep the
most imaginary mode automatically.

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
