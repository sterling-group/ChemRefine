# Filtering

At the end of each step ChemRefine reduces the parsed results to the structures
that move on to the next step. The orchestrator calls `filtering.apply` once per
step with the engine's `StepResults` and the step's `sample` config; it returns a
`PipelineState` of survivors.

`sample: None` (no `sample:` block) is the identity filter — **every** structure
passes through untouched, including one with no computed energy. That is
deliberate: `on_failure: best` backfills a failed structure from its submitted
input, which on step 1 is a seed with no energy yet, and dropping those here
would silently turn `best` into `skip`. With a `sample:` set, structures without
a computed energy are dropped before ranking — they cannot be sorted or
weighted.

## Methods

| Method | Keeps |
|--------|-------|
| `boltzmann` | Structures (lowest-energy first) until the cumulative Boltzmann weight reaches `percent_cumulative`. |
| `min` | The `count` lowest-energy structures (`count = 0` keeps all), **or** all within `window_kcalmol` of the minimum. Set exactly one. |
| `max` | The `count` *highest*-energy structures, **or** all within `window_kcalmol` of the maximum (PES-style sampling). Set exactly one. |

Boltzmann weights use `temperature_k` (default 298.15 K); the same temperature is
used for the `steps.csv` report so the reported weights match what the step
filtered on.

## Energy type

By default the filter ranks on the **electronic** energy. Set `energy_type` to
`gibbs`, `enthalpy`, or `electronic_zero_point` (aliases `G` / `H` / `E_ZPE`) to
rank on a thermochemical energy instead — these require a frequency calc to have
populated that energy (ORCA writes Gibbs / enthalpy / ZPE in its
`THERMOCHEMISTRY` block; see [`Structure`](../api/state.md)). If **no** structure
carries the chosen energy, filtering raises a `ConfigError` — no frequency calc
ran at all, which is a config mistake. If only *some* lack it (an
`on_failure: best` backfill carries no thermochemistry by construction), those
are excluded from the ranking with a warning naming them, rather than aborting
the one policy whose purpose is to keep going — they stay visible in the cache
and the failure ledger.

## Per-parent grouping

Set `by_parent: true` to apply the chosen method independently within each
parent-ID group instead of globally — useful after a fan-out step (e.g. a GOAT
ensemble) to keep diversity across branches rather than letting one parent's
low-energy children crowd out the rest. Seed structures (no parent) form their
own singleton groups.

## Lineage

Each `Structure` carries its immediate `parent_id`, and fan-out children get
hyphenated IDs (`0` → `0-1` → `0-1-2`). Nested ensembles therefore form a *tree*:
every child has exactly one parent, and `by_parent` groups by that immediate
parent. This is how survivors stay traceable back to their seed across all steps.

See the [Filtering API](../api/filtering.md) and the config
[Sample reference](configuration.md#sample-survivor-filter).
