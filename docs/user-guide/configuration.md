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
| `dispatch` | `auto` / `local` / `slurm` | `auto` | How jobs are executed. `auto` submits via `sbatch` when it is on PATH and runs the generated scripts locally via `bash` otherwise. `local` forces the local runner even when an `sbatch` binary exists (e.g. a workstation with SLURM client tools but no reachable cluster); `slurm` requires `sbatch` and fails fast when it is missing. |
| `job_timeout_seconds` | float > 0 / `None` | `None` (wait forever) | How long to wait for a step's jobs before giving up with exit code `8`. `None` is right under SLURM — the partition's own time limit already bounds the job. Set it when nothing else will: a `dispatch: local` run, or a cluster where a job can sit in `PD` indefinitely. It bounds *waiting*, not compute, so set it well above the longest job you expect. Honoured on both the per-job and `slurm_array` paths. |
| `executables` | map | `{}` | Tool → binary-path map for external-binary engines, e.g. `{ orca: /opt/orca/orca }`. Importable backends (mlip, pyscf) need no entry. |
| `steps` | list | — | **Required.** The ordered pipeline stages; `step:` numbers must form a contiguous `1..N`. |

## Per-step keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `step` | int ≥ 1 | — | **Required.** 1-based step number; drives directory naming and order. |
| `name` | str | `None` | Optional filesystem-safe label (letters/digits/`_`/`-`, not all-digits). Directory becomes `stepN_name/`; usable as a CLI target. |
| `engine` | str | — | **Required.** One of `orca`, `mlip`, `mlip-extopt`, `mlip-train`, `pyscf`, `pyscf-extopt`. |
| `operation` | str | `None` | Engine-defined: `opt_sp`, `sp`, `freq`, `pes`, `goat`, `docker`, `solvator`, `mlip_train`. **Optional** — when omitted, ORCA infers the run type from the template's `!` keyword lines (`GOAT`/`DOCKER`/`SOLVATOR`/a `%geom Scan` block/`Opt`/`OptTS`/`Freq`; `#` comments are ignored, matching is case-insensitive), defaulting to a single point if it finds no run-type keyword. An explicit value always wins — give it when inspection can't decide. |
| `template` | str | `stepN.{inp,py}` | Engine input template basename (relative to `template_dir` if not absolute). |
| `slurm_template` | str | global | Per-step SLURM header override. |
| `charge` / `multiplicity` | int | global | Per-step overrides of the global values. |
| `options` | map | `{}` | Engine-specific knobs (see below). |
| `sample` | map | `None` | Survivor filter (see below). `None` keeps every structure. |
| `nms` | bool | `False` | Opt-in normal-mode sampling (honoured only for an NMS-capable engine: ORCA / ExtOpt). Requires a frequency calc: an ORCA NMS step whose template has no `Freq` keyword is rejected at prepare time (set `operation` explicitly to override). The `target` (`minimum`/`ts`) is inferred from the template — `OptTS` → `ts`, else `minimum` — unless `options.target` is set. |
| `on_failure` | `stop`/`skip`/`best` | `stop` | What to do when some structures fail (after the convergence auto-retry below): `stop` (default) caches the successes then halts so failures are never silently dropped; `skip` drops them and continues; `best` keeps all, backfilling a failed structure with the best geometry obtained for it or, failing that, the input it was submitted with — including on step 1, where that input is a seed with no energy computed yet. A structure that *did not converge* is first retried once from its best geometry — the failed attempt is archived under `stepN/<id>/attemptK/` — before this policy applies. |

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
    | `model_name` (aliases `model`, `size`) | `uma-s-1p2` | Model weights (a MACE size, a FAIRChem checkpoint, a SevenNet/ORB id). |
    | `task_name` (alias `task`) | `omol` | Method/head — selects the backend builder. |
    | `model_path` | `None` | Custom MACE checkpoint (selects the `custom_mace` backend). |
    | `device` | `cuda` | `cuda` or `cpu`. |
    | `cores` | `1` | Per-structure core budget. |
    | `backend_python` | `None` | Explicit interpreter for the backend (escape hatch). Normally unset: the step's managed env is resolved by name — see [Installation → MLIP backends](installation.md#mlip-backends). |

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
    | `tensor_folder` | `tensors` | Output dir for `save_tensors` `.npz`. A relative path (the default) is copied back into the structure's own dir (`outputs/stepN/<id>/tensors/`); an absolute path writes there directly. |
    | `cores` | `1` | Per-structure core budget. |
    | `backend_python` | `None` | Explicit interpreter for the backend (escape hatch). Normally unset: the `pyscf` managed env is resolved by name. |

=== "nms (when nms: true)"

    | Key | Default | Description |
    |-----|---------|-------------|
    | `target` | inferred | `minimum` removes every imaginary mode; `ts` keeps the reaction coordinate and removes the rest; `random` explores. Default: **inferred** from the template (`OptTS` → `ts`, else `minimum`); set this to override. |
    | `displacement_value` | `1.0` | ± displacement (Å) per selected mode. |
    | `num_random_displacements` | `1` | `random` only: how many modes to draw. |
    | `ts_mode_index` | `None` | `ts` only: which imaginary mode to keep (default: the largest-magnitude one). |
    | `seed` | `42` | RNG seed for reproducible `random` selection. |

## Output layout

Each structure gets its **own directory** under the step dir, so a calculation's
files sit together and never collide across structures. The input geometry is
kept as `…_inp.xyz` distinct from the engine's output geometry:

```
outputs/
├── step1_screen/
│   ├── 0/                     step1_0_inp.xyz  step1_0.{inp,out,xyz,...}  step1_0.runlog
│   │   └── 0_m5_pos/          NMS round-2 (a displaced re-run) nests under its parent
│   ├── 1/                     …
│   └── _cache/                step.json + arrays.npz, manifest.json, failed_jobs.json
├── step2_refine/<id>/…
└── steps.csv                  Boltzmann summary per surviving structure
```

NMS round-2 — re-optimising a displaced geometry — is treated like any "redo this
structure" step: the child lives in a sub-directory *inside* its parent's directory
(`stepN/<parent>/<child>/`), the same place a failure re-run would go.

## Legacy (v1.3.1) configs

Old keys are auto-translated with one deprecation warning each
(`initial_xyz`→`input`, `orca_executable`→`executables`, `sample_type`→`sample`,
the `mlff:`/`pyscf:` engine blocks → `options:`, `normal_mode_sampling`→`nms`, …).
Only `calculation_type` is a hard error. See
[Migrating from v1 to v2](../migrating-v1-to-v2.md).
