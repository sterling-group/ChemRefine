# Normal-Mode Sampling

Normal-mode sampling (NMS) cleans up stationary points: it removes spurious
imaginary frequencies to reach a true minimum, or keeps exactly one to confirm a
first-order saddle (transition state). It is opt-in per step (`nms: true`) and
only runs on NMS-capable engines — ORCA and the ExtOpt engines (`mlip-extopt` /
`pyscf-extopt`), where ORCA computes the Hessian numerically over the backend's
gradients, so a `Freq` template yields a real frequency table.

Because NMS acts on imaginary modes, the step's input **must** compute frequencies:
an NMS step whose input computes none is rejected before any job is submitted with a
`ConfigError` (set `operation` explicitly to override, e.g. when an ORCA template
uses a spelling the inspector doesn't recognise).

## A generic capability, not an engine feature

The two-round algorithm lives in `chemrefine.nms` and is **engine-independent** — it
drives any NMS-capable engine through one input hook and imports no engine package:

- `nms_input_info(ctx)` — does the input target a TS, and does it compute frequencies?
- the frequency *values* (`imaginary_freqs` mode→cm⁻¹ + the `normal_modes` tensor) ride on
  each parsed `Structure`, filled in the same single pass as geometry/energy — so NMS reads
  them off the structures it already holds, never re-parsing an output.

A new engine becomes NMS-capable by implementing `nms_input_info` and populating those two
structure fields — capability is detected via `isinstance(engine, NmsCapableEngine)`, with no flag to keep in sync;
everything else is shared. (Today: ORCA + the ExtOpt engines.)

## Two rounds + the unified "attempt" model

NMS runs as two throttled rounds, never sharing the core budget at once:

1. **Round 1** — an opt+freq on each survivor at its canonical `stepN/<id>/`.
2. **Round 2** — for each structure not already at the target, displace
   ±`displacement_value` along the selected mode(s) and re-optimise the ± children
   under `stepN/<id>/attemptK/`, retrying any that fail to converge.

Resolving a structure is an **attempt**, the same shape as the `on_failure`
convergence retry: the exploration is archived under `stepN/<id>/attemptK/` and the
**winner lands at the canonical `stepN/<id>/` with the id unchanged**. For
`minimum`/`ts` the single *best resolved* geometry becomes the survivor (id kept, no
duplicate ± minima); a structure with no resolved child is handed to the step's
`on_failure` policy. `random` is the exception — pure exploration with no resolution
gate, so it fans out to new child structures. Round-1 job failures and NMS-unresolved
parents are recorded in one `failed_jobs.json` write.

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

NMS distinguishes *search* parameters (`displacement_value`,
`num_random_displacements`, `seed`) from the resolution *criterion* (`target`,
`ts_mode_index`). Tuning only the search parameters lets `resume` reuse the
round-1 frequencies and the already-resolved structures, re-attempting just the
unresolved ones — instead of re-running the whole step. Changing the criterion
(or the parents/template) forces a full re-run.

`rebuild-nms` is a named alias of `rerun` for the NMS-tuning workflow;
`rebuild-cache` re-resolves NMS from the round-2 outputs already on disk.

See the [Normal-Mode Sampling API](../api/nms.md) for the coordinator and the two
engine hooks documented in the [ORCA engine API](../api/engines_orca.md).
