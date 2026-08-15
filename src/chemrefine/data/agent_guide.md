# ChemRefine — operating guide for AI agents

ChemRefine runs **multi-step refinement pipelines** for computational chemistry: an
ensemble of candidate structures flows through ordered steps (conformer generation →
cheap screening → expensive refinement), each step runs one *engine* (ORCA, Q-Chem,
PySCF, MLIP potentials, MLIP training), and a *sample* filter between steps keeps only
the survivors worth the next step's cost. The workflow is a **YAML file plus a
`templates/` directory** — a reproducible artifact the user can inspect, version, and
rerun. Your job is to author and triage that artifact; ChemRefine's engine executes it
at HPC scale without you in the loop.

## The working loop

1. **Read the schema first**: `get_schema` returns the exact config schema (generated
   from the validating models — it cannot be stale) plus a descriptor per engine.
   An engine with `options_schema: null` (ORCA) is configured **through its template
   file**, not through `options:`; never invent options keys for it.
2. **Build seeds if needed**: `build_structures` (SMILES or raw XYZ → `.xyz` files;
   heed its warnings — charge/multiplicity parity mistakes produce garbage silently).
   `lookup_smiles` resolves compound names via PubChem (needs network).
3. **Write the config**, then **`validate_config` until `ok` with no surprising
   warnings**. Issues carry the field location (`loc`); warnings name silent no-ops
   (undeclared option keys, `nms: true` on an engine that cannot NMS, missing files).
   Then **`save_config`** puts the reviewed YAML on disk — it re-validates and refuses
   to write anything unrunnable, so it is also the final check.
4. **`scaffold_templates`**, then edit each starter (`read_template` /
   `write_template`) — the template *is* the calculation for template-driven engines:
   ORCA keywords, `%pal`, `%maxcore` all live there.
5. **Show the user the YAML before long runs.** The config is the reviewable protocol;
   that is the point of ChemRefine over agent-executes-everything tools.
6. **`start_run`** — returns immediately; the run is a detached process that owns the
   tree's lock and survives your session. **Never wait busily**: poll `run_status`
   (survivor counts per step, failure counts, log tail) at sensible intervals.
7. **Read results with `get_results`** (paginated `steps.csv`: energies in Hartree and
   kcal/mol, Boltzmann weights at the step's own sampling temperature and energy type).
8. **Triage failures with `get_failures`** — each record has a kind and reason, the
   payload carries the exit-code taxonomy, and the standard recovery is
   `start_run(action="rerun-errors")` (re-attempts only the pending failures, then
   continues; earlier steps cache-hit).

## Recipes

The ORCA-family `operation:` vocabulary (also in `get_schema`'s `operations`):
`opt_sp` (optimize + single point), `sp`, `freq`, `pes` (scans), and the ensemble
generators `goat` (conformers), `docker` (poses), `solvator` (explicit solvent).

**Conformer funnel** (the canonical ChemRefine job): step 1 generates/screens cheaply
(`operation: goat` on an xtb-keyword ORCA template, or an MLIP step), sampling
`{method: boltzmann, percent_cumulative: 99}`; step 2 refines the survivors at DFT
(`operation: opt_sp`) with `{method: min, window_kcalmol: 3}`; a final `sp` or `freq`
step does the high-level energies. Sample on `gibbs` instead of `electronic` when a
frequency calculation provides it and the question is thermodynamic.

**Transition states**: OptTS in the ORCA template (`!OptTS Freq ...`), then **verify
before trusting**: `get_frequencies` must show *exactly one* imaginary mode
(`imaginary_count: 1`; `null` means no frequency table — no evidence at all), and
`analyze_mode` on that mode must show the displacement concentrated on the intended
reaction coordinate — forming/breaking bonds at the top of `bond_changes` (negative
rate = forming, positive = breaking). A large imaginary frequency on a methyl rotor is
the classic false positive. Spurious extra imaginary modes: set `nms: true` with
`options: {target: ts}` — ChemRefine displaces along the spurious modes and reruns
until the count is verified, mechanically.

**Minima**: `imaginary_count: 0` from an explicit frequency run; ChemRefine's
`nms: true` (default `target: minimum`) automates shaking off residual imaginaries.

**MLIP fine-tuning**: the `mlip-train` engine trains on the prior ensemble as a
pipeline step; its template is the backend's own config — start from the shipped
fine-tune example, not from scratch.

**Spin-state ladders** (the shipped spin tutorial's shape): optimise once in the
ground-state guess, then score the same survivors at each multiplicity with per-step
`multiplicity:` overrides and `sample: {method: min, count: 0}` — `count: 0` keeps
*everything*, because a ladder compares energies rather than filtering. Check electron
parity on every rung.

**Redox ladders** (the redox tutorial's shape): per-step `charge:` overrides
(−1 / 0 / +1 with multiplicities 2 / 1 / 2 for a closed-shell parent), `count: 0`
throughout; potentials come from energy differences between the charge-state steps.
Score the ladder on the MLIP surface first (`mlip-extopt` — the `omol` task carries
charge and spin), then repeat the rungs on `orca` to confirm. `pyscf` /
`pyscf-extopt` are the DFT alternatives when ORCA is not available.

**Docking & microsolvation**: `operation: docker` generates guest poses (sample the
best few), `operation: solvator` adds explicit solvent through the template's
`%solvator` block; refine survivors with `mlip-extopt` then `orca`, overriding
`charge` where the guest is an ion. `qchem` steps follow the same pattern with `.in`
templates.

## Failure triage (the ledger's vocabulary)

| ledger kind | it means | the usual move |
|---|---|---|
| `output missing` | crashed, killed, or never started (OOM and walltime look like this) | fix resources, then `rerun-errors` |
| `unparseable` | an output exists but is truncated/corrupt | inspect the file, then `rerun-errors` |
| `did not terminate normally` | the program aborted with its own error | read the log tail via `run_status`, fix the input/template, `rerun-errors` |
| `did not converge` | SCF or geometry unconverged | `rerun-errors` retries from the best geometry; consider a better start or looser thresholds |
| `NMS: target stationary point not reached` | displacing + rerunning never hit the target imaginary count | inspect with `get_frequencies`/`analyze_mode`; adjust `displacement_value` or the template, then `rebuild-nms` or `rerun-errors` |
| `failed` | a generic engine-reported failure | read the step's output via `run_status`'s log tail, then `rerun-errors` |

## Hard rules

* Steps must be numbered contiguously from 1; step/config knobs are validated
  strictly — unknown keys are errors, so write only what the schema names.
* `charge`/`multiplicity` are global with per-step overrides; check electron parity.
* Every template-driven step needs `templates/stepN.<suffix>` (or `template:` naming a
  file); scheduler-run steps need the SLURM header files. `validate_config` warns on
  gaps; `scaffold_templates` fills them.
* One driver per output tree (`start_run` refuses while the lock is held). Recovery
  actions (`resume`, `rerun`, `rerun-errors`, `rebuild-cache`, `rebuild-nms`) are the
  vocabulary for everything after a first `run`.
* Failure payload exit codes: 2 config, 3 unknown engine, 4 submission, 5 job failure,
  6 output parse, 7 cache, 8 throttle timeout, 9 backend provisioning, 10 run lock.
