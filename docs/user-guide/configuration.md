# Configuration

A ChemRefine run is driven by a single YAML file: top-level defaults, then an
ordered list of `steps`. The file is validated by Pydantic — an unknown key or a
malformed value fails fast with a `ConfigError` before any job is submitted.

A minimal two-step workflow (MLIP screen → DFT refine):

```yaml
template_dir: ./templates
scratch_dir:  ./scratch
output_dir:   ./outputs
input:        ./step1.xyz
charge: 0
multiplicity: 1
max_cores: 64
slurm_template: cpu.slurm.header
executables: { orca: orca }

steps:
  - step: 1
    name: screen
    engine: mlip
    operation: opt_sp
    options: { model_name: medium, task_name: mace_off, device: cuda }
    sample: { method: boltzmann, percent_cumulative: 99 }

  - step: 2
    name: refine
    engine: orca
    operation: opt_sp
    template: dft_opt.inp
    sample: { method: min, window_kcalmol: 3.0 }
```

## Top-level keys

!!! note "Relative paths are resolved against the config file"
    `template_dir`, `output_dir`, `scratch_dir`, and `input` — when given as
    relative paths — resolve against the **directory containing the YAML file**,
    not the process working directory. So `chemrefine run proj/input.yaml` from
    anywhere finds `proj/templates` and writes `proj/outputs`. Absolute paths are
    used as-is.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `template_dir` | path | `./templates` | Directory holding the per-step engine templates and SLURM headers. |
| `scratch_dir` | path | `None` | Fast node-local working directory base. Unset ⇒ a per-calc `_work_…` dir is derived under `output_dir`; on HPC point it at node scratch (e.g. `/scratch/$USER`). Must differ from `output_dir`. |
| `output_dir` | path | `./outputs` | Where per-step results, caches, and `steps.csv` are written. |
| `input` | path | `None` | Seed structures: an `.xyz` (one structure per frame), a directory of `.xyz`, or a `.csv` of SMILES (column `smiles`). Unset falls back to `templates/step1.xyz`. |
| `charge` | int | `0` | Global molecular charge (per-step `charge` overrides). |
| `multiplicity` | int ≥ 1 | `1` | Global spin multiplicity `2S+1` (per-step `multiplicity` overrides). |
| `max_cores` | int ≥ 1 | `4` | Total CPU budget the throttler enforces across concurrent jobs. The `--maxcores` flag overrides this. |
| `max_gpus` | int ≥ 0 / `None` | `None` (auto) | Concurrent-GPU budget. `None` auto-resolves: unlimited under SLURM (the scheduler places GPUs via `--gres`), the detected device count (`nvidia-smi -L`) locally. The `--maxgpus` flag overrides this. |
| `slurm_template` | str | `cpu.slurm.header` | Default SLURM header basename in `template_dir`. A GPU step auto-picks `cuda.slurm.header`. |
| `slurm_array` | bool | `False` | Submit each step as SLURM job array(s) instead of one job per structure (ignored when running locally). |
| `executables` | map | `{}` | Tool → binary-path map for external-binary engines, e.g. `{ orca: /opt/orca/orca }`. Importable backends (mlip, pyscf) need no entry. |
| `steps` | list | — | **Required.** The ordered pipeline stages; `step:` numbers must form a contiguous `1..N`. |

## Per-step keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `step` | int ≥ 1 | — | **Required.** 1-based step number; drives directory naming and order. |
| `name` | str | `None` | Optional filesystem-safe label (letters/digits/`_`/`-`, not all-digits). Directory becomes `stepN_name/`; usable as a CLI target. |
| `engine` | str | — | **Required.** One of `orca`, `mlip`, `mlip-extopt`, `mlip-train`, `pyscf`, `pyscf-extopt`. |
| `operation` | str | — | **Required.** Engine-defined: `opt_sp`, `sp`, `freq`, `pes`, `goat`, `docker`, `solvator`, `mlip_train`. |
| `template` | str | `stepN.{inp,py}` | Engine input template basename (relative to `template_dir` if not absolute). |
| `slurm_template` | str | global | Per-step SLURM header override. |
| `charge` / `multiplicity` | int | global | Per-step overrides of the global values. |
| `options` | map | `{}` | Engine-specific knobs (see below). |
| `sample` | map | `None` | Survivor filter (see below). `None` keeps every structure. |
| `nms` | bool | `False` | Opt-in normal-mode sampling (honoured only when the engine `supports_nms`). |
| `on_failure` | `stop`/`skip`/`best` | `stop` | What to do when some structures fail: `stop` (default) caches the successes then halts so failures are never silently dropped; `skip` drops them and continues; `best` keeps all (backfilling the best geometry). |

## Sample (survivor filter)

`sample` selects which structures advance to the next step. Common knobs:
`by_parent` (default `false` — filter globally; `true` filters within each
parent-ID group), `temperature_k` (default `298.15`, used by Boltzmann), and
`energy_type` (default `electronic`) — the energy the filter sorts/selects on:
`electronic`/`E`, `gibbs`/`G`, `enthalpy`/`H`, or `electronic_zero_point`/`E_ZPE`.
The non-electronic types require a frequency calc (thermochemistry); filtering
raises a clear error if the chosen energy wasn't computed.

`min` and `max` take **exactly one** selector: `count` (keep N) or
`window_kcalmol` (keep all within that energy window).

| `method` | Selector(s) | Keeps |
|----------|-------------|-------|
| `boltzmann` | `percent_cumulative` (default `99`) | Structures until the cumulative Boltzmann weight reaches the percentage. |
| `min` | `count` **or** `window_kcalmol` | The `count` lowest-energy structures (`0` = keep all), or all within `window_kcalmol` of the minimum. |
| `max` | `count` (≥ 1) **or** `window_kcalmol` | The `count` *highest*-energy structures, or all within `window_kcalmol` of the maximum (PES-style sampling). |

## Engine options

`options` is a free per-engine dict; each engine validates its own keys.

=== "mlip / mlip-extopt"

    | Key | Default | Description |
    |-----|---------|-------------|
    | `model_name` (aliases `model`, `size`) | `uma-s-1p1` | Model weights (a MACE size, a FAIRChem checkpoint, a SevenNet/ORB id). |
    | `task_name` (alias `task`) | `omol` | Method/head — selects the backend builder. |
    | `model_path` | `None` | Custom MACE checkpoint (selects the `custom_mace` backend). |
    | `device` | `cuda` | `cuda` or `cpu`. |
    | `cores` | `1` | Per-structure core budget. |

=== "pyscf / pyscf-extopt"

    | Key | Default | Description |
    |-----|---------|-------------|
    | `method` | `dft` | `dft` or `hf`. |
    | `xc` | — (**required** for `dft`) | Exchange-correlation functional. No silent default — name it explicitly. |
    | `basis` | — (**required**) | Orbital basis set. No silent default — name it explicitly. |
    | `df` | `True` | Density fitting / RI (defaults on — large speed-up, negligible cost). |
    | `device` | `cuda` | Compute device; drives `gpu` when `gpu` is unset (`cuda` ⇒ attempt GPU). |
    | `gpu` | derived from `device` | Attempt `gpu4pyscf` if installed (falls back to CPU). Set explicitly to override the `device`-derived default. |
    | `save_tensors` | `False` | Dump 1e/2e MO tensors after the SCF. |
    | `localized` | `False` | Boys-localize before tensor extraction. |
    | `tensor_folder` | `tensors` | Output dir for `save_tensors` `.npz`. **Must be an absolute path when `save_tensors` is set** — a relative path resolves under the per-job scratch and would be deleted. |
    | `cores` | `1` | Per-structure core budget. |

=== "nms (when nms: true)"

    | Key | Default | Description |
    |-----|---------|-------------|
    | `target` | `minimum` | `minimum` removes every imaginary mode; `ts` keeps the reaction coordinate and removes the rest; `random` explores. |
    | `displacement_value` | `1.0` | ± displacement (Å) per selected mode. |
    | `num_random_displacements` | `1` | `random` only: how many modes to draw. |
    | `ts_mode_index` | `None` | `ts` only: which imaginary mode to keep (default: the largest-magnitude one). |
    | `seed` | `42` | RNG seed for reproducible `random` selection. |

## Output layout

```
outputs/
├── step1_screen/      *.inp *.out *.xyz  _cache/
├── step2_refine/      *.inp *.out *.xyz  _cache/
└── steps.csv          Boltzmann summary per surviving structure
```

## Legacy (v1.3.1) configs

Old keys are auto-translated with one deprecation warning each
(`initial_xyz`→`input`, `orca_executable`→`executables`, `sample_type`→`sample`,
the `mlff:`/`pyscf:` engine blocks → `options:`, `normal_mode_sampling`→`nms`, …).
Only `calculation_type` is a hard error. See
[Migrating from v1.3.1](../migrating-from-main.md).
