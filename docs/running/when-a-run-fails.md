# When a run fails

## Exit codes

Every ChemRefine failure exits with a code that names the failure mode, so a
wrapper script or a SLURM chain can branch on it without parsing log text.

| Code | Meaning | Usual cause | What to do |
| --- | --- | --- | --- |
| `0` | Success | — | — |
| `1` | Generic failure | A step halted under `on_failure: stop`, or an error with no more specific code | Read the last log lines; if a step halted, see [pending failures](#a-step-halted-with-pending-failures) |
| `2` | Config invalid | A typo in `input.yaml` — unknown key, wrong type, non-contiguous `step:` numbers, a directory path with a shell metacharacter in it | The message quotes the offending field; fix and re-run |
| `3` | Unknown engine | `engine:` names something not in the registry | Check the spelling against the [configuration reference](../workflow/configuration.md); legacy names (`mlff`, `dft`) are rewritten automatically |
| `4` | Job submission refused | `sbatch` rejected the script — bad partition, over a QoS limit, missing account | The message includes what `sbatch` said. To run without SLURM, set `dispatch: local` |
| `5` | Job ran but failed | A backend reported failure the pipeline could not recover from | Look at the structure's `.runlog` and `.err` under `outputs/stepN/<id>/` |
| `6` | Output unparseable | The calculation produced a file ChemRefine could not read — usually truncated by a time limit or a node failure | Inspect the `.out`; re-run the affected structures with `chemrefine rerun-errors N` |
| `7` | Cache corrupt or unwritable | An interrupted write, a full disk, or a cache written by a different ChemRefine version | `chemrefine rebuild-cache N` re-parses from the outputs already on disk without re-running anything |
| `8` | Wait deadline expired | Nothing finished for `job_timeout_seconds` | The queue has stalled, not merely slowed — a step that keeps completing jobs restarts the clock each time. Check whether the jobs are stuck (`squeue -u $USER`), or raise the limit. Only reachable when you set `job_timeout_seconds`; the default waits indefinitely |
| `9` | Backend env could not be built | `chemrefine backends install <extra>` failed — no network on the node, a resolver conflict, a full disk, or an env tool that is a shell function rather than a binary on `PATH` | The message quotes the command that failed. Run it on a machine with internet (on HPC: a login node); the half-built env is removed, so a re-run starts clean |
| `10` | Another run holds this output tree | A second `chemrefine` was pointed at an `output_dir` a first one is still working in — a double-submitted driver script, or a second terminal because the first "looks hung" while waiting on the queue | Wait for the other run. If it is known dead, see [another run holds this output tree](#another-run-holds-this-output-tree) |

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
- **The rebuilds walk past their target read-only.** They promise to submit nothing, and
  they keep that promise — but `steps.csv` is rewritten from step 1 on every run, so
  *ending* at the target would silently drop the later steps' rows even when their caches
  are still valid. The steps past the target therefore run best-effort (`RunPlan.stop_after`):
  each one is served from its cache — a load, never a re-parse — and the first cache the
  current configuration cannot serve ends the run quietly, with a log line naming the
  cheapest repair (`rebuild-cache` for outputs that still match this configuration,
  `resume` when upstream results changed). The report then covers exactly what the
  current configuration can vouch for.

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
   round 1); otherwise it is a plain hit and only the `sample:` filter re-runs.
2. **NMS reuse fingerprint** — only the *search* parameters changed (`displacement_value`,
   `num_random_displacements`, `seed`). Round 1 and the already-resolved children are
   reused; the ledgered-unresolved parents are re-attempted. Nothing pending at all means
   the cache is re-stamped under the new key without recomputing anything.
3. **Manifest fingerprint** — no cache, but a manifest stamped *before submission* proves
   the outputs on disk were produced for this exact configuration: the step was
   interrupted. Everything with a usable result is read back; only the rest is
   resubmitted. (NMS and artifact steps skip this route — an interrupted NMS step needs
   its children re-resolved, not just its outputs re-parsed.)
4. **Full run** — prepare, stamp the manifest, submit, retry unconverged once, resolve
   NMS, apply `on_failure`, cache. A mode that may not submit raises here instead, naming
   `resume` and `rerun` as the fixes.

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
