# Filtering

At the end of each step ChemRefine reduces the parsed results to the structures
that move on to the next step. The orchestrator calls `filtering.apply` once per
step with the engine's `StepResults` and the step's `sample` config; it returns a
`PipelineState` of survivors. A structure with no computed energy is always
dropped first.

`sample: None` (no `sample:` block) is the identity filter — every structure with
an energy passes through.

## Methods

| Method | Keeps |
|--------|-------|
| `boltzmann` | Structures (lowest-energy first) until the cumulative Boltzmann weight reaches `percent_cumulative`. |
| `energy_window` | All structures within `window_kcal` of the lowest-energy one. |
| `integer` | The `count` lowest-energy structures (`count = 0` keeps all). |
| `high_energy` | The `count` *highest*-energy structures (PES-style sampling). |

Boltzmann weights use `temperature_k` (default 298.15 K); the same temperature is
used for the `steps.csv` report so the reported weights match what the step
filtered on.

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
[Sample reference](../user-guide/configuration.md#sample-survivor-filter).
