# Normal-Mode Sampling

Normal-mode sampling (NMS) cleans up stationary points: it removes spurious
imaginary frequencies to reach a true minimum, or keeps exactly one to confirm a
first-order saddle (transition state). It is opt-in per step (`nms: true`) and
only runs on NMS-capable engines — the `NMS` column of the [engine
table](../engines/index.md) says which. For the ExtOpt engines it is
ORCA that computes the Hessian, numerically over the backend's gradients, so a `Freq`
template yields a real frequency table.

Because NMS acts on imaginary modes, the step's input **must** compute frequencies:
an NMS step whose input computes none is rejected before any job is submitted with a
`ConfigError`. There is no override — `operation` never changes the generated input
(it only picks the parser), so an explicit value cannot rescue a frequency-less
template. Request the frequencies in the template (e.g. ORCA `! Opt Freq`) instead.

## A generic capability, not an engine feature

The two-round algorithm lives in `chemrefine.nms` and is **engine-independent** — it
drives any NMS-capable engine through one input hook and imports no engine package:

- `nms_input_info(ctx)` — does the input target a TS, and does it compute frequencies?
- the frequency *values* (`imaginary_freqs` mode→cm⁻¹ + the `normal_modes` tensor) ride on
  each parsed `Structure`, filled in the same single pass as geometry/energy — so NMS reads
  them off the structures it already holds, never re-parsing an output.

A new engine becomes NMS-capable by implementing `nms_input_info` and populating those two
structure fields — capability is detected via `isinstance(engine, NmsCapableEngine)`, with
no flag to keep in sync; everything else is shared. That same `isinstance` is what fills
the engine table's `NMS` column, so the documentation cannot fall behind the registry.

## Two rounds + the unified "attempt" model

NMS runs as two rounds sharing **one** core budget:

1. **Round 1** — an opt+freq on each survivor at its canonical `stepN/<id>/`.
2. **Round 2** — for each structure not already at the target, displace
   ±`displacement_value` along the selected mode(s) and re-optimise the ± children
   under `stepN/<id>/attemptK/`, retrying any that fail to converge.

A structure's children are submitted **the moment its own round-1 job finishes**, into the
same queue, alongside whatever is still running — so the slots freed by the early finishers
go to round-2 work instead of idling until the last round-1 job lands. Every parent's
children share that one queue, too: resolving them one parent at a time meant a step with 50
unresolved structures ran 50 sequential batches, each using one parent's worth of the budget.
Raising `max_cores` could not help, because each batch was one parent wide.

Picking each parent's winner still happens once, after the queue drains. It is a decision plus
a file promotion, nothing downstream consumes a resolved structure until the step ends, and
deciding in completion order would give the next step a different cache fingerprint on every
run.

Resolving a structure is an **attempt**, the same shape as the `on_failure`
convergence retry: the exploration is archived under `stepN/<id>/attemptK/` and the
**winner lands at the canonical `stepN/<id>/` with the id unchanged**. For
`minimum`/`ts` the single *best resolved* geometry becomes the survivor (id kept, no
duplicate ± minima); a structure with no resolved child is handed to the step's
`on_failure` policy. `random` is the exception — pure exploration with no resolution
gate, so it fans out to new child structures. Round-1 job failures and NMS-unresolved
parents are recorded in one `failed_jobs.json` write.

### What each directory holds afterwards

A resolved structure ends up with its two calculations kept apart:

```
stepN/<id>/
  stepN_<id>.out .xyz .gbw .hess .opt …   the winning child, promoted whole
  attemptK/
    stepN_<id>.out .xyz .gbw .hess …      round 1, archived as it was
    <id>_m6_pos/  <id>_m6_neg/            every child that was tried
```

**The canonical location always describes one calculation.** Every file there — output,
geometry, orbitals, Hessian, restart — comes from the job that produced the surviving
structure, so parsing any of them agrees with the cached record. Writing back only the
winning *geometry* would leave a `.xyz` from the resolved minimum beside round 1's `.out`
for the saddle it started from, with nothing in the directory to show they describe
different structures.

Round 1 is not discarded: it moves into the same `attemptK/` its children ran in, so one
attempt directory holds both the state that triggered the resolution and everything tried
to resolve it. That is also why `rebuild-cache` stays correct — it re-parses the canonical
path, finds a structure already at the target, and re-derives the same survivor without
re-running the exploration.

## Targets

The `target` defaults to whatever the template implies — an `OptTS` run targets a
`ts`, anything else a `minimum` — and an explicit `options.target` always wins.

| `target` | Resolved when imaginary count = | Behaviour |
|----------|--------------------------------|-----------|
| `minimum` | 0 | Displace ± along **every** imaginary mode (remove them all). |
| `ts` | 1 | Keep the reaction-coordinate mode (`ts_mode_index`, else the largest-magnitude imaginary) and displace along every **other** imaginary mode. |
| `random` | — (no gate) | Displace along `num_random_displacements` modes drawn from all modes (broad exploration). |

`random` selection is seeded (`seed`, default 42) for reproducibility.

## Reuse on resume

The `nms:` flag and every NMS option live in the cache's **resolution key**, never in
a row key — round-1 jobs are byte-identical with NMS on or off — so `resume` always
reuses the round-1 outputs on disk and recomputes only what the edit actually changed:

- **Tuning a search parameter** (`displacement_value`, `num_random_displacements`,
  `seed`) re-reads round 1 and re-attempts just the parents that stayed unresolved;
  resolved structures keep their winners.
- **Changing the criterion** (`target`, `ts_mode_index`) re-runs the *resolution* —
  existing `attemptK/` results are trusted only under the criterion they were written
  for — still without resubmitting a single round-1 job.
- **Changing the parents or the template** recomputes exactly the affected rows, like
  any other step.

## Turning NMS on over an existing tree

Because the flag lives outside the row keys, enabling `nms: true` (with its
`options:`) on a finished step is an ordinary edit: run `chemrefine resume`, and the
step adopts every round-1 output from disk, passes the structures already at the
target straight through — byte-identical, at their canonical paths — and submits
displacement children **only** for the parents that need resolving. A campaign of
1200 finished frequency jobs with 400 imaginary-mode structures costs 400 fan-outs,
not 1200 recomputations. (`random` is the exception by design: it is exploration, so
every parent fans out — round 1 is still reused.) The reverse edit works the same
way: the resolved ensemble simply passes through unresolved semantics no longer ask
about.

`rebuild-nms` re-resolves the NMS step from the outputs already on disk and submits
nothing — the same rebuild `rebuild-cache` performs, but aimed at the step setting
`nms: true` instead of the last one. Reach for it when nothing should be *submitted*
at all (inspection, or adopting a pre-provenance tree); reach for `resume` when the
displacement children should actually run, and for `rerun` when round 1 itself must
be recomputed.

See the [Normal-Mode Sampling API](../api/nms.md) for the coordinator and the two
engine hooks documented in the [ORCA engine API](../api/engines_orca.md).
