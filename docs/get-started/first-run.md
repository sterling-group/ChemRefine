# Your first run

A ChemRefine project is a YAML file, a directory of templates, and a seed structure. This
page walks one from nothing to results. It needs only ORCA and a base
[install](install.md) — no backend environment, no GPU: step 1 is ORCA's bundled
GFN2-xTB, step 2 a small DFT optimisation, and the whole thing takes under a minute on a
laptop.

## 1. Describe the pipeline

The project is four small files. Copy
[`examples/first_run/`](https://github.com/sterling-group/ChemRefine/tree/main/examples/first_run)
from the repository, or paste them:

```
first_run/
├── input.yaml
├── step1.xyz                a few conformers, one per frame
└── templates/
    ├── screen.inp           ! XTB2 Opt
    └── dft_opt.inp          ! PBE def2-SVP Opt
```

--8<-- "README.md:quickstart"

`executables:` wants the full path to the ORCA binary — ORCA finds its own helpers
beside it. `max_cores: 4` is the whole budget: with no `%pal` line in the templates, four
serial jobs run at once. On a cluster, add `%pal nprocs N end` to the templates and the
`#SBATCH` lines to `templates/cpu.slurm.header`; nothing else changes.

Every key is in the [configuration reference](../workflow/configuration.md); the engines
you can name are in the [engine table](../engines/index.md).

## 2. Check it before it runs

`validate` reports **every** problem at once rather than stopping at the first — Pydantic
errors with their field locations, unknown engines, bad option values, plus warnings for
things that would silently do nothing:

```bash
chemrefine validate input.yaml
```

A step's template and SLURM header are files you have to supply. `scaffold` writes a
commented starter for each one the config expects but the directory lacks, and leaves
existing files alone:

```bash
chemrefine scaffold input.yaml
```

Edit the starters — that is where the ORCA keywords and the `#SBATCH` lines go — then dry
run, which loads the config, describes what would happen, and submits nothing:

```bash
chemrefine run input.yaml --dry-run
```

## 3. Run it

```bash
chemrefine run input.yaml                # the YAML's max_cores: 4
chemrefine run input.yaml --maxcores 8   # or override the budget
```

The run submits through `sbatch` if it is on `PATH` and with `bash` otherwise; nothing in
the YAML changes between a laptop and a cluster. See [Local & cluster](../running/clusters.md).

## 4. Read the results

```
outputs/
├── step1_screen/
│   ├── 0/                     one directory per structure, named by its id
│   │                          step1_0_inp.xyz (what went in),
│   │                          step1_0.{inp,out,xyz} and .runlog (what came out)
│   ├── 1/                     …
│   ├── step1_ensemble.xyz     every parsed structure, one multi-frame XYZ
│   ├── step1_survivors.xyz    just the ones `sample:` passed to step 2
│   └── _cache/                the fingerprint that lets an unchanged step be skipped,
│                              plus failed_jobs.json when something failed
├── step2_refine/<id>/…
└── steps.csv                  a Boltzmann summary row per surviving structure
```

Both ensemble files are sorted by the step's own ranking energy and carry
`stepN id=<id> E=<hartree> Eh` on each comment line (`G=`, `H=` or `E_ZPE=` when the step
ranks on a thermochemical energy), so any frame traces back to its
directory and its `steps.csv` row. Structure ids are hierarchical (`0` → `0-1` →
`0-1-2`), so every survivor traces back to the seed it came from.

## Next

- Change something and re-run: unchanged steps are served from cache, and only what you
  touched recomputes — [Caching & resume](../running/caching.md).
- Something failed: [When a run fails](../running/when-a-run-fails.md).
- A complete study, start to finish: the [tutorials](../tutorials/index.md).
- Build the YAML by clicking or by asking: the [workflow
  builder](../workflow/builder.md) and the [AI agents](../workflow/agents.md).
