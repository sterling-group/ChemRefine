# Migrating from v1.3.1 (main) YAML

ChemRefine v2 loads your existing v1.3.1 workflow YAML **unchanged** — the config
loader rewrites the old keys to the current schema at parse time (you'll see a
one-line deprecation warning per legacy feature). This page documents that
mapping so you can modernise your files when convenient.

Everything below is rewritten automatically **except `calculation_type`**, which
must be replaced by hand with `engine:` + `operation:`.

## Top level

| v1.3.1                          | v2                                |
|---------------------------------|-----------------------------------|
| `orca_executable: /path/orca`   | `executables: { orca: /path/orca }` |
| `initial_xyz: ./seed.xyz`       | `input: ./seed.xyz`               |

## Per step

| v1.3.1                                         | v2                                            |
|------------------------------------------------|-----------------------------------------------|
| `engine: "DFT"`                                | `engine: orca`                                |
| `engine: "MLFF"` + an `mlff:` block            | `engine: mlip-extopt` (ORCA-driven, as before)|
| `engine: "MLFF"` / `mlff` (no block)           | `engine: mlip` (the **new** direct, in-process path) |
| `mlff:` / `pyscf:` / `trainer:` block          | `options: { … }` (the engine block → options) |
| `operation: "OPT+SP"` / `"GOAT"`               | `operation: opt_sp` / `goat`                  |
| `operation: "MLFF_TRAIN"`                      | `engine: mlip-train`                          |
| `normal_mode_sampling: true`                   | `nms: true`                                   |
| `normal_mode_sampling_parameters: { … }`       | folded into `options:` (see NMS table below)  |
| `calculation_type: "DFT"` **(manual)**         | `engine: orca` + `operation: opt_sp`          |

### Normal-mode sampling knobs

`normal_mode_sampling_parameters` keys move into `options:` and are renamed:

| v1.3.1                          | v2 `options` key       | notes |
|---------------------------------|------------------------|-------|
| `calc_type: rm_imag` (default)  | `target: ts`           | keep exactly one imaginary mode (first-order saddle / TS) |
| `calc_type: random`             | `target: random`       | sampling along random modes |
| `displacement_vector`           | `displacement_value`   | ± displacement magnitude (Å) |
| `num_random_displacements`      | `num_random_displacements` | unchanged |

> v2 adds a third target, `target: minimum` (remove **all** imaginary modes → a true minimum), which
> v1.3.1 did not have. A bare `normal_mode_sampling: true` maps to `target: ts` to match main's default.

> **MLFF was ORCA-driven in v1.3.1.** `engine: MLFF` ran ORCA using a machine-learned
> gradient server — that is `mlip-extopt` in v2. The bare `mlip`/`mlff` engine in v2
> is the *new* direct (template-driven, no ORCA) path. A v1.3.1 step is recognised by
> its `mlff:` block and mapped to `mlip-extopt` so it keeps doing what it did.

## Sampling

`sample_type: { method, parameters: { … } }` becomes a flat `sample: { method, … }`,
with these per-method parameter renames:

| method         | v1.3.1 parameter        | v2 key               |
|----------------|-------------------------|----------------------|
| `boltzmann`    | `weight`                | `percent_cumulative` |
| `integer`      | `num_structures`        | `count`              |
| `high_energy`  | `num_structures`        | `count`              |
| `energy_window`| `energy` (+ `unit`)     | `window_kcal`        |

## Example

```yaml
# v1.3.1
orca_executable: /orca
initial_xyz: ./seed.xyz
steps:
  - step: 1
    operation: "OPT+SP"
    engine: "MLFF"
    mlff: { model_name: uma-s-1, task_name: omol, device: cuda }
    sample_type: { method: boltzmann, parameters: { weight: 95 } }

# v2 (what the loader produces; also what you'd write by hand)
executables: { orca: /orca }
input: ./seed.xyz
steps:
  - step: 1
    operation: opt_sp
    engine: mlip-extopt
    options: { model_name: uma-s-1, task_name: omol, device: cuda }
    sample: { method: boltzmann, percent_cumulative: 95 }
```
