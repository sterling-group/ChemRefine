# Recovery & Reruns

One output tree, one driver ([the run lock](caching.md)), six ways to drive it. Every CLI
action resolves to a **plan** — which `StepMode` each step runs in — before anything
executes, so the recovery behaviour is decided once, in `chemrefine.recovery`, rather than
re-derived inside each layer. This page is the decision table; the fingerprints it leans
on are [Caching & Resume](caching.md).

## What each command plans

| command | steps before the target | the target | steps after | run ends at |
|---|---|---|---|---|
| `run` | — | every step `EXECUTE`, its cache and manifest discarded first | — | last step |
| `resume` | — | every step `RESUME` | — | last step |
| `rerun [N]` | `RESUME` (cache-hit when valid) | `RESUME`, after its cache **and manifest** are discarded — so it misses and truly re-executes | `RESUME` | last step |
| `rerun-errors [N]` | `CACHE_ONLY` | `RESUME` | `RESUME` | last step |
| `rebuild-cache [N]` | `CACHE_ONLY` | `REBUILD` | not covered (`stop_after`) | the target |
| `rebuild-nms [N]` | `CACHE_ONLY` | `REBUILD` — the step setting `nms: true`, not the last | not covered | the target |

Two asymmetries are deliberate:

- **`rerun-errors` resumes the later steps but cache-hits the earlier ones.** The halt
  that left the target's failures pending is what stopped the later steps from ever
  running, so they have no cache to hit — left `CACHE_ONLY` they would raise for a cache
  that cannot exist, after the command had already repaired what it was pointed at.
- **The rebuilds cover nothing past their target.** They promise to submit nothing, and a
  step the run never reached has no outputs to re-parse — so the plan simply ends there
  (`RunPlan.stop_after`) instead of leaving steps in a mode that must fail.

`rerun` discards the manifest as well as the cache because the manifest is what makes an
output tree look *interrupted rather than discarded*: kept, a later `resume` would re-parse
the very outputs the user asked to redo.

## What a mode may do

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

## The route through one step

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

## What `on_failure` decides

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

### Changing the policy over a cached step

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
