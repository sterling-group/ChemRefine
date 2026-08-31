![ChemRefine](assets/logo-wordmark.svg#only-light){ width="480" }
![ChemRefine](assets/logo-wordmark-dark.svg#only-dark){ width="480" }

# ChemRefine

--8<-- "README.md:hero"

A YAML file describes a chain of stages (e.g. MLIP screen → DFT refine
→ frequency analysis); ChemRefine writes the engine inputs, submits
SLURM jobs under a global core budget, parses the outputs, filters
survivors by energy, and feeds the next stage. Re-runs are fingerprint-
cached so unchanged steps skip automatically.

[Install it](get-started/install.md){ .md-button .md-button--primary }
[Run something](get-started/first-run.md){ .md-button }

---

## Features

- **Step-based pipeline** driven by a single Pydantic-validated YAML
  config — every knob lives in one place.
- **Engine plugins** behind a narrow Protocol — each a direct engine or an
  ORCA-driven `-extopt` gradient server. The
  [engine table](engines/index.md) is generated from the
  registry, so it is never out of date; new engines drop in via a registry
  decorator (see [Adding an Engine](developer/adding-an-engine.md)).
- **Filtering** by Boltzmann cumulative weight, or the lowest / highest
  structures by count or energy window (`min` / `max`, the latter PES-style),
  with optional per-parent grouping.
- **SLURM-aware**: the PAL/`nprocs` value in each input participates in a
  total-core throttle so concurrent jobs never exceed `max_cores`; whole
  steps can also go out as job arrays.
- **Resumable**: each step's parsed results are cached with a SHA-1
  fingerprint of everything that can reach a job — engine, operation, template
  bytes, effective charge/multiplicity, engine options — plus the parent
  structures (IDs and content). Changing any of those re-runs the step;
  `sample:` and `on_failure` are deliberately excluded, so tuning a filter
  refilters the cached results instead of recomputing them.
- **Hierarchical IDs**: every conformer carries its lineage
  (`0` → `0-1` → `0-1-2`) so survivors trace back to their root structure.

## Quickstart

```bash
pip install chemrefine
```

--8<-- "README.md:quickstart"

[Your first run](get-started/first-run.md) walks this from nothing to results.

## Where to go next

| I want to… | Go to |
|---|---|
| install it, and run something | **[Get started](get-started/install.md)** |
| see a complete study, start to finish | **[Tutorials](tutorials/index.md)** |
| know what can go in the YAML | **[Writing a workflow](workflow/configuration.md)** |
| know which engines and backends there are | **[Engines & backends](engines/index.md)** |
| run it on a cluster, or fix a run that failed | **[Running a workflow](running/cli.md)** |
| understand how it works inside | **[Internals](internals/architecture.md)** |
| move a v1 project to v2 | **[Upgrading from v1](get-started/upgrading-from-v1.md)** |
| add an engine, or send a patch | **[Contributing](developer/contributing.md)** |

## Requirements

--8<-- "README.md:requirements"

## Getting help

- [Project issues](https://github.com/sterling-group/ChemRefine/issues) — search before opening a new one
- [Citing ChemRefine](citation.md)
