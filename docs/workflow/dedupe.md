# Dedupe

`sample:` only ever reasons about energy — it never asks whether two survivors
are the *same structure*. That gap matters after any optimization step: two
structures that started as distinct conformers (e.g. from a GOAT step-1 search)
can relax onto the same minimum, and nothing catches it. A
`sample: {method: min, count: 15}` step would then silently return fewer than
15 truly distinct minima, some of the "15" being the same geometry counted
twice.

`dedupe:` closes that gap. The orchestrator calls `dedupe.apply` once per step,
on the engine's raw parsed structures, **before** `sample` ranks and slices
them:

```yaml
steps:
  - step: 3
    engine: orca
    operation: opt_sp
    dedupe: { rmsd_angstrom: 0.125 }
    sample: { method: min, count: 15 }
```

`dedupe: None` (no `dedupe:` block, the default) is the identity — every
structure passes through untouched, matching `sample: None`'s convention.

## How it decides "same structure"

1. **Bucket** structures by their exact atomic-number sequence. Order matters —
   this is also what guarantees two unrelated systems are never compared.
2. Within a bucket, examine structures **lowest-energy first**. A structure is
   dropped the moment its RMSD to any structure already kept — after the
   optimal rigid-body (Kabsch) alignment — falls at or below
   `rmsd_angstrom`. The lowest-energy structure in each cluster of duplicates
   survives.

Because the RMSD is computed after alignment, a duplicate is caught regardless
of how the two structures happen to be translated or rotated relative to each
other — only the internal geometry (bond lengths, angles) is compared.

A cheap pre-filter skips the RMSD computation outright when two structures'
energies differ by more than 5 kcal/mol: the same geometry is the same
electronic-structure calculation, so its energy cannot differ from a real
duplicate's by anywhere near that much. This keeps the check cheap on large
GOAT ensembles (hundreds to ~2000 structures) without changing what counts as
a duplicate — it only rules out comparisons that would never have matched.

## Options

| Key | Default | Description |
|-----|---------|-------------|
| `rmsd_angstrom` | `0.125` | RMSD threshold (Å) below which two structures count as the same geometry. Matches GOAT/CREST's own conformer-dedup default. |
| `include_hydrogens` | `True` | Compare all atoms. Set `False` to compare heavy atoms only (e.g. when hydrogen positions are noisy or irrelevant to what counts as "the same structure"). |
| `by_parent` | `False` | Compare only within each parent-ID group instead of across the whole step. Off by default — collapsing structures that started as *different* parent conformers but converged together is the motivating case. Set `True` to instead keep every parent's own duplicates lineage-local, matching `sample`'s `by_parent`. |

## What it does not touch

- **The cache.** Dedup runs fresh on every cache hit — it is not folded into
  the step's cache key/fingerprint. It reshapes the survivor set, not the
  calculation, so a config-only change to `dedupe:` never invalidates cached
  work the way an engine option change would.
- **`stepN_ensemble.xyz`.** This file still records every structure the step
  actually computed, duplicates included — it is the cache's own record of
  what ran. Only `stepN_survivors.xyz` (and what feeds the next step) reflects
  post-dedup, post-`sample` survivors.

See the [Dedupe API](../api/dedupe.md) and the config
[Dedupe reference](configuration.md#dedupe-structural-duplicate-collapse).
