# When a run fails

## Exit codes

Every ChemRefine failure exits with a code that names the failure mode, so a
wrapper script or a SLURM chain can branch on it without parsing log text. A
run stopped by a signal exits with the shell's code for that signal.

| Code | Meaning | Usual cause | What to do |
| --- | --- | --- | --- |
| `0` | Success | — | — |
| `1` | Generic failure | A step halted under `on_failure: stop`, or an error with no more specific code | Read the last log lines; if a step halted, see [pending failures](#a-step-halted-with-pending-failures) |
| `2` | Config invalid | A typo in `input.yaml` — unknown key, wrong type, non-contiguous `step:` numbers, a directory path with a shell metacharacter in it | The message quotes the offending field; fix and re-run |
| `3` | Unknown engine | `engine:` names something not in the registry | Check the spelling against the [configuration reference](../workflow/configuration.md); legacy names (`mlff`, `dft`) are rewritten automatically |
| `4` | Job submission refused | `sbatch` rejected the script — bad partition, over a QoS limit, missing account | The message includes what `sbatch` said. To run without SLURM, set `dispatch: local`. An `sbatch` that failed on a controller timeout may have queued the job all the same: the next run finds it in the queue and [stops until it has drained](#a-previous-runs-jobs-are-still-queued-or-running) |
| `5` | Job ran but failed | A backend reported failure the pipeline could not recover from | Look at the structure's `.runlog` and `.err` under `outputs/stepN/<id>/` |
| `6` | Output unparseable | The calculation produced a file ChemRefine could not read — usually truncated by a time limit or a node failure | Inspect the `.out`; re-run the affected structures with `chemrefine rerun-errors N` |
| `7` | Cache corrupt or unreadable | A `_cache/` file or a directory of the run's tree this account cannot read, an interrupted save, or a cache written by a different ChemRefine version | *Unreadable*: restore read access (or run as the tree's owner), then repeat the command. *Corrupt* or another version: `chemrefine rebuild-cache N` re-parses from the outputs already on disk without re-running anything. A job-lease ledger (`active_jobs.json`) is the exception: its message says when to delete it |
| `8` | Wait deadline expired | Nothing finished for `job_timeout_seconds` | The queue has stalled, not merely slowed — a step that keeps completing jobs restarts the clock each time. Check whether the jobs are stuck (`squeue -u $USER`), or raise the limit. Only reachable when you set `job_timeout_seconds`; the default waits indefinitely |
| `9` | Backend env could not be built | `chemrefine backends install <extra>` failed — no network on the node, a resolver conflict, a full disk, or an env tool that is a shell function rather than a binary on `PATH` | The message quotes the command that failed, or names the directory it could not create. Run it on a machine with internet (on HPC: a login node); the half-built env is removed, so a re-run starts clean |
| `10` | Another run holds this output tree | A second `chemrefine` was pointed at an `output_dir` a first one is still working in — a double-submitted driver script, or a second terminal because the first "looks hung" while waiting on the queue | Wait for the other run. If it is known dead, see [another run holds this output tree](#another-run-holds-this-output-tree) |
| `11` | Write refused by the filesystem | A full disk or quota, a read-only mount, or a directory this account cannot write — under `output_dir` during a run, or where `chemrefine scaffold`, the builder or an agent was asked to save a config, a template or a seed | The message names the file, and the one the filesystem refused when that was another. *A file you named* (`WriteRefusedError`): free space, fix the permissions or name another place, then ask again — a config or a template that was there is as it was, while a seed directory may be left without seeds. *A run's output tree* (`OutputTreeError`): when the message also names a run whose lock a reclaim swept up, wait for that run to end or stop it before anything else. Free space, remount read-write or fix the permissions, then run the command that stopped again: `run` and `rerun` begin by discarding caches, so `resume` in their place would serve what they were asked to discard. A `run`, `resume`, `rerun` or `rerun-errors` that stopped inside its steps continues with `chemrefine resume`, which reads back what the jobs had already finished — a trained model included — and runs only the rest |
| `130` | Interrupted | `Ctrl-C` (SIGINT) stopped the driver | On its way out the run stopped its local jobs and released the output tree. `chemrefine resume` continues it — see [the run was interrupted mid-step](#the-run-was-interrupted-mid-step) |
| `143` | Terminated | SIGTERM stopped the driver: `scancel`, the walltime of the batch job it ran in, a plain `kill` | As for `130`. Jobs the run had submitted to SLURM run on, and `chemrefine resume` refuses with code `10` while one of them is still queued or running: wait for them, or `scancel` them |

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
never be mistaken for a success. The one failure they leave alone is an
[`unreadable`](#a-structure-failed-with-unreadable) output, which may be whole.

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
resubmitted**. ChemRefine can tell the difference because the step's manifest —
written before any job is submitted — records each **row's own key**, so an
output on disk is provably the one this configuration would compute for that
structure. Adoption is per row: if the config changed in a way that reaches some
jobs (an edited template, a different effective charge, a changed upstream
parent), exactly those rows re-run and the rest are read back. That is not
mixing two configurations — a row is adopted only under proof its job *is* this
configuration's job. An NMS step continues the same way: adopted round-1 rows
feed the resolution, clean parents pass through, and parents not at the target
fan out fresh displacement children.

A training step continues at its own grain: it has one job and one product. When
the manifest shows the model on disk was trained for this configuration, and the
job's own `.runlog` ends with `exit_code=0` and `scratch_kept=false`, `resume`
adopts the model and trains nothing. A training that was stopped, or whose model
did not come home whole, is trained again, and the log says which it was. A model
nothing vouches for — its runlog is gone, or you copied it home from scratch by
hand — is adopted by `chemrefine rebuild-cache N`, on your word.

One thing deliberately does *not* reuse that work: `chemrefine run` means start
over, and always does.

If you would rather drive it by hand — to inspect what survived before
continuing — `chemrefine rebuild-cache N` re-parses step N's outputs without
submitting anything, and ledgers whatever is missing.

## A job's results did not all come home

A job runs in a scratch directory and copies its results into the structure's
directory when it ends; a file there is no room to copy is moved instead, which
within one filesystem needs none. A result that directory refuses either way —
a full disk with the scratch elsewhere, a quota — is the job's failure, not
something it passes over:

- the job's script exits `74`, or with its program's own status when that had
  already failed, and the system's own message is in the structure's `.err`;
- its `.runlog` ends with `scratch_kept=true`;
- the scratch directory is left in place, because it then holds the only whole
  copy. The runlog's header names its `host` and its `scratch` path.

The `.err` and the `.runlog` sit in the directory that refused the copy, so on a
full disk they can be cut short themselves; the exit status is what survives.
Under SLURM `sacct` shows it, and a local run logs it as a warning.

Whether the structure's result is usable is decided by reading what did arrive:
an output cut short fails its structure like any other that cannot be parsed,
and `chemrefine rerun-errors N` runs it again once there is room. What scratch
kept can also be copied home by hand.

## Another run holds this output tree

Every run takes an advisory lock (`<output_dir>/.chemrefine.lock`) for its whole
duration, and a second run pointed at the same tree exits with code `10` instead
of starting. That refusal is protecting your data: the on-disk state of a *live*
run is indistinguishable from an interrupted one, so a second driver would
re-parse outputs the first one's jobs are still writing, archive their
directories out from under those jobs, and resubmit duplicates.

The lock names its holder — pid, host, start time. What to do depends on that
holder:

- **It is alive**: wait for it, or stop it deliberately, then re-run.
- **It died on *this* host** (killed, OOM): nothing to do — a run on the same
  host detects the dead pid and reclaims the lock by itself.
- **It died on *another* host** (a batch job's node was drained, `kill -9` on a
  different login node): liveness cannot be probed across hosts, so the lock
  stays. Once you know that run is dead, delete the lock file and re-run:

```bash
rm outputs/.chemrefine.lock
chemrefine resume input.yaml
```

## A previous run's jobs are still queued or running

A dead driver is not the whole story: the SLURM jobs it submitted keep running
after it is gone (that is the scheduler's job), and so do locally dispatched
jobs after a `kill -9`. The jobs in flight are therefore recorded in the step's
`_cache/active_jobs.json`, and a new run checks those records — with the same
exit code `10` — before touching the tree, because those jobs' exit traps still
copy results into it.

- **SLURM jobs**: a batch records a tag of its own before it asks `sbatch` for
  its first job, and every job of the batch is submitted under a name that ends
  in it — `step1_0.cr1f3a9c21b0` for structure `0` of step 1, an array as
  `step1_array.cr1f3a9c21b0`. A plain `squeue` cuts a name to its first eight
  characters; `squeue -u $USER -o "%i %j"` prints it whole. Each job's id joins
  the tag in the record as `sbatch` returns it. The new run asks `squeue` once,
  for the recorded ids and for any job named under a recorded tag, so it also
  finds a job whose id never reached the record: the driver was killed, or
  `sbatch` failed, after the scheduler had taken the job. The message names the
  jobs the queue listed. Wait for them to drain (or `scancel` them), then
  re-run. The queue answers for the jobs it lists: after a `kill -9` of a run
  that was submitting, see that no `sbatch` of it is still waiting for the
  scheduler before you run again. When `squeue` itself fails, the run stops and
  says so: run the command again once the scheduler answers. On a host with no
  `squeue` nothing can be checked; the run warns and proceeds, so a tree copied
  off-cluster stays usable.
- **Local jobs**: probed by pid on the host that submitted them. From any other
  host they cannot be probed — once you know that run's jobs are dead, delete
  the named `active_jobs.json` and re-run.

A record the filesystem refuses stops the run with code `11`, and the batch's
SLURM jobs are on record all the same, under the tag written before the first of
them was asked for. A local job stops with the run.

## A structure keeps failing to converge

An unconverged structure is retried once per run from the best geometry it
reached, with the failed attempt archived under `attemptK/`. If it still fails,
it lands in the ledger. Common fixes: loosen the SCF convergence in the template,
give the optimiser more cycles, or start from a better geometry.

## A job's `.err` says its filesystem had no space left

```
ChemRefine: the filesystem holding …/outputs/step1/0 had no space left when this job
ended; what the job wrote there, this file included, may be cut short or missing
```

A job's `.runlog`, its `.err` and the program's output are files in the structure's
directory. While that filesystem is full, nothing the program says about its own failure
can be written to them. The usual reason is the job's own scratch directory, which sits
inside the structure's directory unless `scratch_dir` puts it elsewhere. The job removes
its scratch when it ends, which gives the disk its room back and leaves an empty `.err`
behind — so the job script asks `df` what is left before it removes anything, and adds
this line once the scratch is gone. Where a step's ledgered reason quotes the end of the
`.err`, as an ORCA step's does, the line is in `failed_jobs.json` too.

Point [`scratch_dir`](../workflow/configuration.md) at a filesystem with room for the
program's temporary files, or free space, then `chemrefine resume`.

The line states what `df` reports. A quota `df` does not show leaves no line, and neither
does a disk that is out of inodes, or one the dying program made room on by deleting its
own temporaries: an output that stops mid-line beside an empty `.err` and a non-zero
`exit_code` in the `.runlog` is the same failure without it. And where nothing is freed —
the scratch is on another filesystem, or the job kept it because a result
[did not come home](#a-jobs-results-did-not-all-come-home) — the line cannot be written
either: the run itself then stops with [exit `11`](#exit-codes) at its next write, naming
the file.

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

## A structure failed with "unreadable"

```
unreadable: 3/step2_3.out: Permission denied
```

The job's output is there, or cannot be ruled out, and this account cannot read it: a
tree another account wrote, a directory whose mode changed, an I/O error on a network
filesystem. The file is named relative to the step's directory, with the system's own
reason. It is that structure's failure whichever engine wrote the output, so the step's
other results are cached and its `on_failure` policy decides what follows.

The calculation may be whole, so `resume` and `rerun-errors` do not run it again: they
leave an unreadable output where it is. Restore read access, then have it read:

```bash
chemrefine resume input.yaml            # an on_failure: stop step reads it and carries on
chemrefine rebuild-cache input.yaml 2   # a skip or best step, whose failure is not pending
```

`rebuild-cache` ends at the first later step no cache serves; `chemrefine resume` then
carries on from there.

If the file is beyond recovery, delete it. The structure is then `output missing`, which
`resume` and `rerun-errors` run again for an `on_failure: stop` step; for a `skip` or
`best` step, `chemrefine rerun N` recomputes it.

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

## How recovery decides

One output tree, one driver ([the run lock](caching.md)), six ways to drive it. Every CLI
action resolves to a **plan** — which `StepMode` each step runs in — before anything
executes, so the recovery behaviour is decided once, in `chemrefine.recovery`, rather than
re-derived inside each layer. This page is the decision table; the fingerprints it leans
on are [Caching & Resume](caching.md).

### What each command plans

| command | steps before the target | the target | steps after | run ends at |
|---|---|---|---|---|
| `run` | — | every step `EXECUTE`, its cache and manifest discarded first | — | last step |
| `resume` | — | every step `RESUME` | — | last step |
| `rerun [N]` | `RESUME` (cache-hit when valid) | `RESUME`, after its cache **and manifest** are discarded — so it misses and truly re-executes | `RESUME` | last step |
| `rerun-errors [N]` | `CACHE_ONLY` | `RESUME` | `RESUME` | last step |
| `rebuild-cache [N]` | `CACHE_ONLY` | `REBUILD` | best-effort `CACHE_ONLY` (`stop_after`) | the first step whose cache no longer matches |
| `rebuild-nms [N]` | `CACHE_ONLY` | `REBUILD` — the step setting `nms: true`, not the last | best-effort `CACHE_ONLY` | the first step whose cache no longer matches |

Two asymmetries are deliberate:

- **`rerun-errors` resumes the later steps but cache-hits the earlier ones.** The halt
  that left the target's failures pending is what stopped the later steps from ever
  running, so they have no cache to hit — left `CACHE_ONLY` they would raise for a cache
  that cannot exist, after the command had already repaired what it was pointed at.
- **The rebuilds walk past their target from cache.** They promise to submit nothing, and
  they keep that promise — but `steps.csv` is rewritten from step 1 on every run, so
  *ending* at the target would silently drop the later steps' rows even when their caches
  are still valid. The steps past the target therefore run best-effort (`RunPlan.stop_after`):
  each one is served from its cache — a load, never a re-parse — and the first cache the
  current configuration cannot serve ends the run quietly, with a log line naming the
  cheapest repair (`rebuild-cache` for outputs that still match this configuration,
  `resume` when upstream results changed). The report then covers exactly what the
  current configuration can vouch for. Submitting nothing is not writing nothing: a
  rebuild takes the run lock and rewrites its target's cache, with what it derives beside
  it — a calculation step's manifest and each structure's `.result.json` (an NMS step's
  displaced children excepted), an artifact step's record of its product — and, like every
  run and recovery command, rewrites `steps.csv` and each step's ensemble files. A tree
  it cannot write stops it with [exit `11`](#exit-codes).

`rerun` discards the manifest as well as the cache because the manifest is what makes an
output tree look *interrupted rather than discarded*: kept, a later `resume` would re-parse
the very outputs the user asked to redo.

### What a mode may do

Three questions are asked about a mode, each answered by one predicate on the enum
(`chemrefine.step.StepMode`):

| mode | `may_submit` | `runs_through_run_step` | `can_halt` |
|---|---|---|---|
| `EXECUTE` | yes | yes | yes |
| `RESUME` | yes | yes | yes |
| `CACHE_ONLY` | **no** | yes | **no** |
| `REBUILD` | **no** | no — `rebuild_cache_step` is its own function | yes |

- `CACHE_ONLY` cannot halt because it is the mode every step a scoped action is *not*
  targeting runs in: `rerun-errors 3` has to be able to reach step 3 past step 1's
  pending failures.
- `REBUILD` halts even though it submits nothing: re-parsing from disk cannot make a
  failed structure succeed, and continuing would run the next step against the partial
  survivor set the user asked to stop on.

The predicates are exhaustive `match` statements whose wildcard arm holds only
`assert_never` — a fifth mode added later fails type-checking at all three rather than
silently inheriting the permissive answer.

### The route through one step

`run_step` tries the recovery routes in order, strongest proof first; the first that
applies wins, and `EXECUTE` skips straight to the full run:

1. **Valid cache** — the on-disk fingerprint matches this config and these parents. With
   a pending `on_failure: stop` ledger and a mode that may submit, only the still-failed
   structures are re-attempted (an NMS step goes through `reattempt_nms`, which reuses
   every round-1 output it can still read and re-runs the rest); otherwise it is a plain
   hit and only the `sample:` filter re-runs.
2. **Row provenance** — no cache this key can serve, but the manifest carries each
   row's own key. Rows whose stored key matches are **adopted** (re-parsed from disk);
   the rest — changed or new parents, jobs a config edit actually reaches, jobs with no
   usable output — are computed. An adopted row whose output this account cannot read is
   neither: it stays an `unreadable` failure and is not run again. NMS resolution then runs on the
   assembled round 1: clean parents pass through, `attemptK/` resolutions are honoured
   only under the criterion they were written for, and unresolved parents fan out fresh
   children.
   An artifact step — one job and one product, such as a trained model — is
   continued at its own grain: its product is adopted when the manifest's step
   stamp matches this configuration and its job's runlog says the job finished,
   and otherwise its job runs. A manifest *without* provenance at its grain —
   row keys, or for an artifact step the step stamp — stops the resume by name:
   run `chemrefine rebuild-cache N` to adopt the tree under the current rules
   first, or `chemrefine rerun N` to compute it again.
3. **Full run** — prepare, stamp the manifest with its row provenance, submit, retry
   unconverged once, resolve NMS, apply `on_failure`, cache. A mode that may not submit
   raises here instead, naming `resume` and `rerun` as the fixes.

Each route is strictly cheaper than the next, which is why the order is the safety
argument: a route only runs when a stronger proof than the next one's is in hand.

### What `on_failure` decides

| policy | ledgered in `failed_jobs.json`? | pending for `resume` / `rerun-errors`? | halts the run? |
|---|---|---|---|
| `stop` (default) | yes | **yes** | **yes** — after the step's successes are cached and summarised |
| `skip` | yes, for visibility | no | no |
| `best` | yes, for visibility | no | no |

The halt is a single point — `halt_if_pending`, called from the pipeline after the step's
cache and its `steps.csv` row are written — so a stopped run still records the work it
completed, and the next `resume` re-attempts *only* the ledgered failures. `skip` and
`best` resolve their failures the moment the policy is applied; their ledger entries are a
record, not a queue, which is why `rerun-errors` tells you it has nothing to re-attempt
for them and points at `rerun` instead.

#### Changing the policy over a cached step

The cache stores a step's results **after** the policy is applied — `stop` and `skip`
persist the successes alone, `best` persists the backfilled failures too — and the
fingerprint deliberately excludes `on_failure`, so editing the policy alone never
invalidates a step. What happens instead depends on whether the stored results already
wear the shape the new policy wants:

| edit | what `resume` does |
|---|---|
| `stop` ↔ `skip` | a free cache hit — both store the successes alone, so nothing changes |
| `skip` → `stop` | the ledgered failures become pending and are re-attempted |
| to or from `best`, with a non-empty ledger | the ledgered failures are re-attempted and the step re-finalizes under the new policy — successes are never recomputed |
| any edit with a clean ledger | a free cache hit — with no failures, every policy produces identical results |

The cross-`best` case is the one that needs the machinery: without it, a step halted
under `stop` and switched to `best` would serve the cached successes-only set — `skip`
semantics — with nothing said anywhere. A mode that may not submit (`rebuild-cache` /
`rerun-errors` aimed elsewhere) cannot make that repair and raises the ordinary "no
cache this configuration can use" error instead of serving the wrong survivor set.
