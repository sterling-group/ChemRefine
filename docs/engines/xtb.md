# xtb (`xtb`)

The `xtb` engine runs the [xtb](https://github.com/grimme-lab/xtb) program directly, one job
per structure: GFN2-xTB, GFN1-xTB, GFN0-xTB or GFN-FF, in gas phase or implicit solvent,
with no ORCA install. (ORCA's `! XTB2` keyword runs the same methods through ORCA's own
bundled xtb, configured through the ORCA template; that route is unchanged.)

```yaml
executables: { xtb: /opt/xtb-6.7.1/bin/xtb }
steps:
  - step: 1
    engine: xtb
    operation: opt_freq        # required: the run type
    nms: true                  # the modes are xtb's
    options: { method: gfn2, solvation: alpb, solvent: water, cores: 4 }
```

## The run type is `operation:`

xtb takes its run type on the command line, and its xcontrol input has no key for one, so
the template cannot imply it. **`operation:` is required** on an xtb step — the run's t=0
check refuses a step without one — and it is what runs:

| `operation` | xtb runs | the step reads |
|---|---|---|
| `sp` | `--grad` | energy and forces |
| `opt_sp` | `--opt [opt_level] --grad` | + the optimised geometry and its convergence |
| `freq` | `--hess --grad` | + frequencies, normal modes, thermochemistry, at the input geometry |
| `opt_freq` | `--ohess [opt_level] --grad` | all of the above, at the optimised geometry, in one run |

Every run computes the gradient, so every structure carries forces (for `mlip-train`
labels, say): it costs about as much as the SCC it follows. `sp`, `opt_sp` and `freq` mean
what they mean on every engine; `opt_freq` is xtb's one-run `--ohess`. xtb has no
transition-state optimiser, so an xtb step never makes `ts` the NMS default.

## The template is an xcontrol file

`templates/stepN.xcontrol` is an ordinary xtb xcontrol file (`man 7 xcontrol`), handed to
xtb as `--input` and copied unchanged for every structure. It carries what xtb reads only
there: constraints (`$constrain`, `$fix`), a confining wall (`$wall`), a scan (`$scan`),
and optimiser or SCC settings (`$opt`, `$scc`). The starter `chemrefine scaffold` writes is
comments and `$end`, and runs as it is.

Everything that decides what is computed is on the command line, from the step — and
xtb's command line wins over its xcontrol input. So a template that sets one of these is
refused before anything runs, naming where the setting belongs:

| in the template | set instead by |
|---|---|
| `$chrg` | the config's `charge` (the step's own, or the workflow's) |
| `$spin` | the config's `multiplicity` |
| `method` in `$gfn` | `options.method` |
| `solvent` in `$gbsa` or `$solvation` | `options.solvation` and `options.solvent` |

Charge and multiplicity reach xtb as `--chrg {charge} --uhf {multiplicity − 1}`: xtb counts
unpaired electrons. The geometry is the structure's `stepN_<id>_inp.xyz`, named on the
command line, so it stays in Å both ways.

## Options

| Key | Default | Description |
|-----|---------|-------------|
| `method` | `gfn2` | `gfn2`, `gfn1`, `gfn0` or `gfnff` — `--gfn 2` / `1` / `0` or `--gfnff`, written on the command line every time rather than left to the binary's default. GFN0 reads its parameters from `XTBPATH` (see below). |
| `solvation` | `None` | The implicit-solvation model: `alpb`, `gbsa` or `cpcmx`. No default, and required whenever `solvent` is set, so a model added later never changes what an existing config means. |
| `solvent` | `None` | The solvent, as xtb names it (`water`, `ch2cl2`, `thf`, …; for CPCM-X, a Minnesota Solvation Database name such as `1,2-dichloroethane`). Required with `solvation`; xtb checks the name itself. |
| `solvation_reference` | `None` | ALPB/GBSA only: the reference state of the solvation free energy, spelled as xtb reads it — `bar1M` (1 bar gas → 1 M solution) or `reference`; `gsolv`, xtb's default, adds nothing. xtb 6.7.1 does not know `bar1m`, which is why the spelling is checked here. |
| `opt_level` | `None` | `opt_sp` / `opt_freq` only: `crude`, `sloppy`, `loose`, `lax`, `normal`, `tight`, `vtight` or `extreme`; unset is xtb's `normal`. Checked here because xtb runs an unknown level as `normal` without a word; refused on a run that does not optimise. |
| `accuracy` | `None` | The SCC accuracy, `--acc` (lower is tighter; xtb's default is 1.0). |
| `etemp` | `None` | The electronic temperature in K, `--etemp` (xtb's default is 300). |
| `cores` | `1` | Threads: `-P N` and `OMP_NUM_THREADS=N,1`, allocated as `--ntasks=1 --cpus-per-task=N`. Not part of the cache key. |
| `device` | `cpu` | Inherited and **refused when set to `cuda`**: xtb runs on the CPU, so the knob would only charge a GPU against `max_gpus`. |
| `backend_python` | `None` | Inherited and **refused when set**: xtb is a program named in `executables`, not a Python backend. |

**CPCM-X** gives no gradient in xtb, so `solvation: cpcmx` is accepted on `operation: sp`
alone, which then runs `--scc` and reports no forces. Only some xtb builds include CPCM-X,
and ChemRefine cannot tell from the cluster's config whether yours does: a binary built
without it stops at once, and the structure is ledgered as a run that did not terminate
normally, quoting xtb — *"The CPCM-X library was not included in this version of xTB"*.

## The install

`executables: { xtb: … }` names the binary; a bare `xtb` is looked up on `PATH`. GFN1 and
GFN2 have their parameters built in; GFN0 reads `param_gfn0-xtb.txt` from `XTBPATH`. When
the binary is an absolute `<prefix>/bin/xtb` — the release tarball, a conda environment and
a meson install all look like this — the job sets `XTBPATH=<prefix>/share/xtb`, unless the
environment (a `module load`, the SLURM header) has already set it.

Each job also sets `OMP_STACKSIZE=4G` unless it is set, and raises the stack limit where
the node allows: xtb's own advice, since the Hessian of a larger molecule overflows the
default stack.

## What a step reads

xtb runs in the job's scratch directory with `--namespace` set to the structure's stem,
so it names every file `stepN_<id>.<its usual name>`, and the results come home beside the
stdout (`stepN_<id>.out`): `.gradient`, `.xtbopt.xyz`, `.xtbopt.log` (the optimisation's
trajectory), `.g98.out`, `.hessian`, `.vibspectrum`, `.charges`, `.wbo`. The stem is what
lets an NMS winner's files follow its output when it is promoted to its parent's name.

- **Energy** — the last `TOTAL ENERGY` stdout prints (with implicit solvent, that includes
  the solvation free energy).
- **Forces** — `−∇E` from the `.gradient` file, in eV/Å. The file carries its own energy
  and coordinates, and is refused when they are not the structure's: a gradient left in
  the directory by another calculation is never attached as this one's forces.
- **Geometry** — `.xtbopt.xyz` when stdout says xtb wrote it; otherwise the input geometry,
  since a single point or a Hessian moves no atoms.
- **Convergence** — the last `GEOMETRY OPTIMIZATION CONVERGED` / `FAILED TO CONVERGE` banner.
  Not `.xtboptok`: xtb 6.7.1 leaves that file behind an optimisation that did not converge,
  and exits 0. An unconverged SCC stops xtb with a fatal error, ledgered as such.
- **Termination** — xtb's closing `* finished run on` line, or its fatal-error block, whose
  reasons the ledger quotes. ("normal termination of xtb" goes to stderr, the job's `.err`.)
- **Frequencies and NMS** — from `.g98.out`, which lists the vibrational modes. The table
  and the displacement tensor share one 0-based index over all 3N modes, translations and
  rotations first (six, or five for a linear molecule); an imaginary mode is negative. The
  file prints displacements to two decimals, and each is scaled back to unit length.
- **Thermochemistry** — from xtb's THERMODYNAMIC summary: `gibbs` is its total free energy
  (E + G(RRHO)), `energy_zpe` is E + its zero-point energy, and `enthalpy` is E + H(T) at
  the temperature the summary was computed at.

**Frequencies in implicit solvent** are less certain than in gas phase. With ALPB or GBSA,
the soft modes (a torsion near 300 cm⁻¹, say) of xtb's `--hess` differ from differences of
the same binary's solvated gradients by up to a few tens of wavenumbers, and the solvated
surface itself shifts them by as much when the molecule is merely rotated. Stiff modes
agree to a few wavenumbers. NMS displacements along a soft mode are still fine; free
energies from them carry that noise.

## Checking a build

A miscompiled xtb can give gradients that are simply wrong — conda-forge's xtb 6.7.1
aarch64 gfortran build gives GFN1 and GFN2 gradients off by about their own size, while its
GFN-FF is right. The live tier checks the binary it is given against itself: its gradients
against differences of its energies, for every method in gas and both Born solvation
models, and its Hessian and the frequencies ChemRefine reads against differences of its
gradients. Point it at any build:

```bash
CHEMREFINE_XTB=/path/to/xtb pytest -m integration tests/test_engines_xtb_numerics.py
```

It takes a few seconds; a good build passes all of it.
