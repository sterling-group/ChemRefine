# Normal-Mode Sampling

Normal-mode sampling (NMS) cleans up stationary points: it removes spurious
imaginary frequencies to reach a true minimum, or keeps exactly one to confirm a
first-order saddle (transition state). It is opt-in per step (`nms: true`) and
only runs on engines that report `supports_nms` (today: ORCA, which produces a
frequency table).

Because NMS acts on imaginary modes, the step's template **must** run a frequency
calc: an ORCA NMS step whose template has no `Freq` keyword is rejected at prepare
time with a `ConfigError` (set `operation` explicitly to override, e.g. when the
template uses a spelling the inspector doesn't recognise).

## Two rounds

NMS runs as two throttled rounds, never sharing the core budget at once:

1. **Round 1** — an `opt+freq` on each survivor. ChemRefine parses the imaginary
   frequencies and the normal-mode displacement tensor from each output.
2. **Round 2** — for each structure that is *not* already at the target, displace
   ±`displacement_value` along the selected mode(s), re-optimise the ± children
   (each in its own directory nested under the parent's, like any "redo this
   structure" re-run), and check whether each child reached the target.

A round-1 structure is *resolved* if it already had the target imaginary count or
any of its displaced children resolved. Unresolved parents are handed to the
step's `on_failure` policy, and both round-1 job failures and NMS-unresolved
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
round-1 frequencies and the already-resolved children, re-attempting just the
unresolved parents — instead of re-running the whole step. Changing the criterion
(or the parents/template) forces a full re-run.

`rebuild-nms` is a named alias of `rerun` for the NMS-tuning workflow;
`rebuild-cache` re-resolves NMS from the round-2 outputs already on disk.

See the [NMS Resolution API](../api/step_nms.md) and the ORCA displacement math in
the [ORCA engine API](../api/engines_orca.md).
