# Changelog

Notable changes to ChemRefine. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/). Update the Unreleased section
as part of any user-visible change; it becomes the release notes when the
version is tagged.

## [2.0.0] — Unreleased

A ground-up rewrite of the v1.3.1 pipeline. Legacy YAML configs and
flag-style CLI invocations keep working through a translation layer that
warns once per deprecated spelling — see
[migrating from v1 to v2](https://sterling-group.github.io/ChemRefine/migrating-v1-to-v2/)
for the full map.

### Deprecated

- The v1.3.1 compatibility layer — legacy YAML keys (`calculation_type` aside,
  which already raises), the `mlff*`/`dft` engine spellings, the `sample_type`
  block, and the flag-style CLI (`chemrefine CONFIG --rebuild_cache N`) — is
  scheduled for removal in **3.0.0**. It warns once per rewritten feature today.
  See [migrating from v1 to v2](https://sterling-group.github.io/ChemRefine/migrating-v1-to-v2/).

### Added

- Engine plugin system: a `CalculationEngine` protocol plus a registry, with
  four documented base shapes for new engines (`engines/api.py`). Plugins and
  MLIP backends are **auto-discovered** — a new engine package or backend
  module is dropped in and registers itself, with no central import list to
  edit. Bundled engines: `orca`, `mlip`, `mlip-extopt`, `mlip-train`,
  `pyscf`, `pyscf-extopt`.
- Subcommand CLI — `run`, `resume`, `rerun [step]`, `rerun-errors [step]`,
  `rebuild-cache [step]`, `rebuild-nms [step]` — with documented process
  exit codes per failure class.
- Per-step result cache fingerprinted over the step config **and** the
  parent structures' content; `resume` re-executes only what changed.
  Filter-only (`sample:`) edits refilter cached results without re-running
  calculations.
- Per-step failure policy `on_failure: stop | skip | best` with a
  `failed_jobs.json` ledger; `resume` / `rerun-errors` re-attempt only the
  still-failed structures of a `stop` step.
- Two-round normal-mode sampling (`nms: true`) with target-aware
  displacement (`minimum` / `ts` / `random`) and a reuse fingerprint that
  re-attempts only unresolved parents when search parameters are tuned.
- Shared ExtOpt HTTP server: ORCA optimises on gradients served by an MLIP
  or PySCF backend in the same SLURM job (kernel-assigned ports, health
  probe, clean teardown).
- Per-backend MLIP extras (`mlip-fairchem`, `mlip-mace`, `mlip-sevenn`,
  `mlip-orb`, `mlip-chgnet`) and a backend-agnostic calculator factory. Each
  backend installs into its own dedicated environment (their torch/e3nn trees
  conflict); `[mlip]` defaults to FAIRChem / UMA (`uma-s-1p2`, from upstream
  `fairchem-core >= 2.18` on PyPI).
- Managed backend environments: `chemrefine backends {install,list,path}`
  provisions one env per backend (built with the same tool that created the
  current env — conda / uv / venv) and steps resolve them **by name**, so
  conflicting MLIP stacks (e.g. MACE + UMA) run side by side in one pipeline.
  Every run validates its steps' backends before any job submits.
- SLURM-optional execution: the generated scripts run unchanged under
  `bash` with the same core/GPU throttling, runlogs, and artifacts.
- Opt-in job-array submission (`slurm_array: true`): each step goes out as
  one `sbatch --array` per ≤1000 structures, with the scheduler enforcing
  the `max_cores` budget via the array's `%limit` — large ensembles submit
  in seconds instead of one sbatch call per structure. Outputs, runlogs,
  recovery, `job_timeout_seconds` (measured per array *task*, so a draining
  array keeps resetting the clock) and the GPU-budget config checks behave
  identically to the per-job path.
- Seeding from a multi-frame `.xyz`, a directory of `.xyz` files (all
  frames), or a CSV of SMILES (deterministic 3D embedding).
- Engineering gates: 100% line+branch test coverage, 100% docstring
  coverage, ruff, mypy, CodeQL, weekly `pip-audit`, and an
  mkdocstrings-rendered API reference.

### Changed


- **FAIRChem is trainable.** `task_name: omol` (and every other head) now selects a trainer as
  well as a calculator, so the family `[mlip]` installs by default is no longer
  inference-only. Its dataset is an ASE database per split — labels on a
  `SinglePointCalculator`, `metadata.npz` beside each, one directory per split because
  FAIRChem resolves a missing `metadata_path` against the database's *parent*. Verified
  against the real `AseDBDataset`. **The shipped example config is still outstanding**: no
  training config ships in the fairchem wheel to adapt, and the UMA weights are behind a
  gated Hugging Face repo (`403` on the file, model card readable), so it could not be proven
  by a real run.
- **A `task_name` you state wins over the `model_path` shortcut.** A checkpoint with no
  library named still means MACE, as it always has; naming one loads the checkpoint with
  *that* library. The old rule sent any `model_path` to MACE, which was right while MACE was
  the only library that could produce one and would now silently load a FAIRChem model with
  the wrong loader.
- **One MLIP registry, one module per library.** `task_name` selects a library, and that
  library declares the environment it needs **once** — `engines/mlip/backends/<library>.py`
  now holds its ASE calculators *and* its trainer, hanging both off a single `MlipLibrary`.
  Previously inference and training kept parallel registries, so the pip extra, the
  distribution, the import name, the task-key list and the dispatch rule were each spelled
  twice per library with nothing comparing them — and that metadata is what resolves which
  environment a step launches from. A trainable task is now a runnable task by construction,
  and resolves the same environment in both directions. Adding a library, or making one
  trainable, is still one dropped-in module.
- **`mlip-train` is rebuilt.** It runs through the same scheduler and throttle as every
  other engine, so a training job is charged against `max_cores` / `max_gpus`, gets the
  device-aware SLURM header, the runlog, the scratch handling, local dispatch and
  `job_timeout_seconds` — none of which it had. Which library trains is now `task_name`,
  resolved through a registry keyed exactly like the calculator's: one word names the
  library whether a step trains a model or runs one. Adding a trainable backend is one
  dropped-in module under `engines/mlip/backends/`, beside the calculators for the same
  library — the environment it needs is then declared once for both.

  The step's YAML changes: `task_name`, `model_name` and `device` are required (none has a
  default worth guessing — `model_name` is the foundation model a fine-tune starts from),
  `job_name` is gone (the job is named like every other), `valid_fraction` now means what
  it says, and `test_fraction` is the separate held-out set it used to be confused with.
  The template is the trainer's own config `stepN.yaml`, **rendered** through
  `$PLACEHOLDERS` rather than patched — patching is what could not work across libraries,
  since the keys the old code inserted are meaningful to MACE and rejected outright by
  FAIRChem. `operation: mlip_train` is a legacy spelling and still translated.

- **`task_name` is now the only key that selects an MLIP backend.** `model_path` says where
  the named library's weights come from and selects nothing; every library's builder loads a
  local checkpoint itself, so there is no `custom_<library>` task for any of them.

  **Migration:** a step that names a `model_path` and no `task_name` used to resolve MACE,
  and now resolves the `task_name` default (`omol`, FAIRChem). Add the library that produced
  the checkpoint — the same word the `mlip-train` step trained with:

  ```yaml
  options:
    task_name: mace_off          # add this line
    model_path: ./outputs/step4/train/train_stagetwo.model
  ```

  `custom_mace` still resolves, as a MACE alias kept for v1 configs; it now needs a
  `model_path`, since that is the only thing it can mean. The old rule was three-valued —
  task stated, unstated-with-checkpoint, unstated — while every channel it travelled through
  (`model_dump()`, a `$TASK_NAME` template placeholder, the ExtOpt server CLI) is two-valued,
  so the third case was dropped silently at each crossing. It had already made the shipped
  MLIP-training tutorial's last step fail, and could hand a FAIRChem checkpoint to a MACE
  loader as soon as a second library could train one.

- The scoped recovery actions now cover the steps around their target
  deliberately rather than by accident:
  - `rerun-errors N` continues the run after repairing step N. The steps after it
    are exactly the ones the halt stopped from ever running, so holding them to
    the cache they cannot have failed the command *after* it had done its work —
    and told you to run `resume`, which is what you had just run.
  - `rebuild-cache N` ends at the step it rebuilt. Rebuilding step N says nothing
    about the steps after it, and neither available answer was right: serving them
    from a cache they never wrote raises, and resuming them would submit.
  - `rebuild-nms [N]` re-resolves an NMS step from the outputs already on disk and
    submits nothing — the same rebuild `rebuild-cache` performs, aimed at the step
    setting `nms: true` rather than the last one. It had become another spelling of
    `rerun`, which discards the cache and recomputes round 1: the frequency
    calculation is the expensive part of an NMS step, and it is already on disk.
    `--rebuild_nms` from the v1 CLI maps here, and means again what it meant there.
- v2 YAML schema: `engine:` + `operation:` replace `calculation_type`;
  `sample:` replaces `sample_type:`; `nms:` + `options:` replace
  `normal_mode_sampling*`; `executables:` replaces `orca_executable`;
  `input:` replaces `initial_xyz`. Legacy spellings are rewritten with a
  deprecation warning — except `calculation_type`, which raises with a
  pointer to the migration guide.
- `mlff` renamed to `mlip` everywhere (engines, extras, YAML); the old
  spellings remain as aliases.
- The parsed-result field `terminated` is now `terminated_normally`, in the
  cache document, the `*.result.json` records, and the API. `True` always meant
  "the program exited cleanly" — a success marker — and the shorter name read as
  its opposite. Renamed before 2.0.0 ships so no released cache carries the old
  key; the format versions are deliberately unchanged, since there is no released
  reader to protect.
- `options.device` now defaults to `cpu` (was `cuda`). It is read by both the
  rendered script and the scheduler, so the old default asked for a GPU the job
  was never allocated; request one explicitly with `device: cuda`.
- The version is single-sourced in `pyproject.toml`; releases are tag-driven
  (a `vX.Y.Z` tag builds, creates the GitHub Release, and publishes to PyPI
  after a tag↔version consistency check) — and now run the full CI matrix
  first. `ci.yml` triggers on pushes to `main` and on pull requests, neither of
  which a tag is, so the release path had been running metadata validation only.
- `pyscf` gains a `strict_scf` option (default on). Two new CI jobs cover what
  the matrix could not reach: a mutation gate that breaks each critical predicate
  and requires a red test, and a run of the suite with a managed backend
  environment provisioned.

### Fixed


- **A training step no longer swallows the ensemble.** Its structures are the previous
  step's, passed through — but `parse` was never called, because a step's structures came
  from the per-structure ledger and a training step prepares no per-structure jobs. Ten
  structures went in and zero came out, so the pipeline stopped with "produced no
  survivors" at the step *after* training. No shipped workflow had ever got past it.
- **A failed training is no longer cached as a success.** The old step waited for its job
  to leave the scheduler's queue and cached the step either way; a job that exits non-zero
  leaves the queue too. The trained model's existence is now the success test, and failing
  it writes no cache — so `chemrefine resume` retrains rather than serving a model that was
  never produced. A training that *succeeded* before the driver died is adopted by
  `rebuild-cache` without recomputing it.
- **A failed *re*-training no longer adopts the previous run's model.** The existence test
  above cannot tell this run's product from the last one's, so a re-run whose job died
  having written nothing found run 1's model, digested those bytes into the sidecar, and
  cached them under the new fingerprint — a run that was internally consistent and described
  a training that never happened, which the consuming step then cache-hit on too. An
  artifact step now archives its run directory before preparing, as every per-structure step
  already did.
- **A training step under `slurm_array: true` trains on its input.** Both trainers quoted
  the config basename, and the array path renders one run block against `$INP_NAME`
  sentinels that each task assigns — single quotes suppress the expansion, so every task ran
  against a file literally named `$INP_NAME`. Asserted now for every job-executable engine.
- **A FAIRChem training job stops writing inside its own checkpoint directory.** The
  scheduler takes a job's output directory from its output path's parent, which for a
  trainer whose model is nested (`train/checkpoints/final/inference_ckpt.pt`) put the
  runlog, the `.err` and — with no `scratch_dir` — the working directory three levels inside
  the tree it was about to write its final checkpoint to. Its dataset splits moved under
  `data/` for the same reason: the training split and FAIRChem's run id are both `train`.
- **`mlip-train` copies back every pattern its trainers produce.** The engine's list had
  drifted from the trainers': it was missing FAIRChem's `*.yaml` and carried a `*.txt` no
  backend writes. A missing glob is a model left behind when the scratch is cleaned; a test
  now holds the list to the union.
- Model checkpoints are digested by streaming rather than read whole into the driver
  process. This runs on every step's cache key, the files are 1–2 GB, and on a cluster the
  driver is a login node.

- **The training dataset is readable by the trainer.** ChemRefine wrote `DFT_energy` /
  `DFT_Forces` while MACE's defaults are `REF_energy` / `REF_forces`, and the shipped
  template declared a third pair — MACE refuses a file in which it finds none of its keys,
  so every training job died at data load.
- **Charge and multiplicity reach the training set.** MACE reads `total_charge` /
  `total_spin` and silently defaults them to a neutral singlet, so a fine-tune on an ion or
  an open-shell system was fitted against the wrong species with nothing said in any log.
- **The training command resolves.** It emitted a bare `mace_run_train`, which is on
  nobody's `PATH` once MACE lives in its own environment — which it must, since its `e3nn`
  pin cannot share a prefix with FAIRChem's. `mlip-train` is now a provisionable engine:
  its backend is checked by the preflight and its job launches from the managed env.
- **Retraining re-runs the steps that use the model.** A step naming a file in its options
  now has that file's contents in its cache key, so a step consuming a retrained model no
  longer serves a result computed with the previous weights. A training step passes its
  structures through unchanged, which is what left nothing else to move the key.
- An interrupted step that prepared no jobs re-runs instead of "resuming" into nothing. A
  zero-job manifest is a real value, not a missing one, and treating it as missing cached
  an empty result the next run then served.
- The documented default for `device` (`mlip`, `mlip-extopt`, `pyscf`, `pyscf-extopt`) said
  `cuda`; the code says `cpu`. Following the docs got you a silent CPU run on a GPU cluster.
- `mlip-train`'s options are documented at all, in the configuration reference.
- A relative path in a step's `options` — `model_path` — resolves against the **config file's**
  directory, like every other path a config names, instead of the process working directory.
  It reaches a job that runs in a scratch directory, so an unresolved relative path was found
  by nobody: naming the model a previous step produced worked only when you happened to
  invoke `chemrefine` from the config's own directory.
- A FAIRChem checkpoint can be loaded from a path. `get_predict_unit` resolves registry names
  only and raises `KeyError` for a file, so a fine-tuned FAIRChem model could have been
  trained and never run.
- An unknown `task_name` raises `ConfigError` rather than a bare `ValueError`. It is reached
  from the preflight, where only a `ChemRefineError` carries the exit code the CLI maps — a
  `ValueError` there reached the user as a traceback instead of a status.
- A training step that names a runnable-but-untrainable backend is refused by the preflight,
  before any step submits, and told which it is — rather than after its upstream steps have
  spent days computing a dataset.
- A backend can no longer declare a pip extra that `pyproject.toml` does not: `pip install
  "chemrefine[typo]"` warns and exits 0, so the mistake used to provision an empty
  environment that satisfied the preflight and failed at the backend import.

Hardening landed during the 2.0.0 stabilization:

- A `slurm_array` step larger than one array no longer exceeds `max_cores`. Steps past the
  per-array task cap are split into chunks, and every chunk carried the *whole*
  `max_cores // PAL` as its own `%limit` — but they are all queued at once, so the budget
  was granted once per chunk. A 2500-structure step at `pal: 8` ran 1536 cores against a
  `max_cores: 512`. Each chunk now takes a share. The cost is that a share is not handed
  back when a sibling drains early, so the tail of an N-chunk step runs at 1/N of the
  budget.
- `job_timeout_seconds` is reachable again on the `slurm_array` path when `squeue` is
  intermittently failing. A failed poll reported the polled job ids back as though they
  were queue rows; `squeue` prints an array's tasks as `12345_0` and never the bare parent,
  so that set differed from the real rows on every alternation — and the wait, which
  re-anchors its stall deadline when the rows move, saw movement on every tick and never
  timed out. A poll that never reached the scheduler now says so, and the deadline treats
  it as no progress.
- Local GPU jobs are pinned to the devices the run was actually granted. The budget came
  from `nvidia-smi -L` (the whole host) and the pin was the lowest free *index*, so on a
  node that granted `CUDA_VISIBLE_DEVICES=2,3` the throttler admitted a job per host GPU
  and pinned them to `0..N-1` — hardware the run did not own. An inherited allocation is
  now authoritative: `max_gpus` may narrow it and never widen it. Device tokens are carried
  verbatim, so GPU UUIDs and `MIG-…` handles work where a bare index could not name the
  device at all.
- An NMS `resolution.json` that parses but does not carry `resolved_from` now raises
  `CacheError` (exit code 7) naming the file, instead of a bare `KeyError`.

- A structure that fails to converge is re-run as soon as its own job frees a slot,
  instead of after the entire step has drained. Jobs are now parsed as they finish and
  a retry joins the same throttled queue, so it overlaps the rest of the batch. Before,
  nothing was parsed until the last job returned — so no retry could exist yet, and every
  slot freed after the final submission sat idle until then. On a 277-structure step at
  `pal: 16` under `max_cores: 128`, two retries ran as a separate half-hour phase after
  the batch rather than inside it.

  Two consequences worth knowing:

  - **A step that retried will re-run its downstream steps once.** Results now come back
    in manifest order, with a retry's structures beside the ones from the job they
    replace; previously retries were appended after every other structure. That ordering
    feeds `parents_digest`, so the next step's cache fingerprint changes. It settles after
    one run.
  - `job_timeout_seconds` **is now a stall deadline** — the longest a step may go with
    *nothing* finishing — and its clock restarts on every completion. It previously bounded
    a whole drain on some paths and a single wait on others. A batch that keeps draining no
    longer trips it however long the step takes, so a value chosen to catch a stuck queue
    still works and no longer has to be re-tuned as a step grows. Under `slurm_array: true`
    that clock follows the array's *tasks*: a whole step is one job id that does not finish
    until its last task does, so anything coarser would have made the same number mean a
    total-runtime bound there and a stall bound everywhere else.

- NMS round 2 runs in the step's own queue instead of one batch per structure. A structure
  needing displacement gets its `attemptK/` and its ± children submitted the moment its own
  round-1 job finishes, alongside whatever is still running. Two things were serial before.
  Children could not start until *every* round-1 job had drained, so the slots freed by the
  early finishers idled until the last one landed; and each parent's children were their own
  throttled batch, submitted and drained before the next parent's began — a step with 50
  unresolved structures ran 50 sequential batches, each using one parent's worth of
  `max_cores` and idling the rest. Raising `max_cores` could not help, because a batch was
  one parent wide. Picking each winner still happens once, after the queue drains, so
  survivors keep coming back in manifest order and downstream fingerprints do not move.

- `resume` and `rerun-errors` re-run a structure whose output is present but unusable,
  instead of reporting it as failed. They decided what was left to do by asking whether
  the output *file existed*, which a truncated one does. Preparing a re-run archives the
  previous attempt and writes a fresh input, so a run killed in between leaves exactly
  that: a worthless output at the canonical path and the good geometry sitting in
  `attemptK/`, unread. The structure then reached the failure policy having never used
  its second attempt. Both paths now judge by what actually parsed. The window is not
  new, but retries starting as slots free widened it from the tail of a step to nearly
  all of it. Resume also stopped reading every output twice to do this.

- Convergence retries go out as one batch instead of one at a time. Submission is
  budgeted per batch by the throttler, which blocks until that batch's jobs finish,
  so retrying structure by structure handed it a single job per call — the second
  structure was not submitted until the first had *finished*. A step with two
  unconverged structures at `pal: 16` under `max_cores: 128` therefore ran one
  16-core job at a time, left the other 112 cores idle, and cost the sum of the
  retries rather than the longest of them. Raising `max_cores` could not help,
  because the batch size was one.

- A step's cache refuses an `arrays.npz` that a different save wrote. The document
  and its coordinate sidecar are separate atomic writes, so a save interrupted
  between them — a walltime kill, a node failure, Ctrl-C, and `resume` re-saves a
  step whenever it repairs one — left a new sidecar beside the previous document.
  Nothing about that pair is malformed and no check above it could see the
  difference: the records parse, and the fingerprint still matches because it
  covers the step's *inputs*, not what is on disk. Read back, each structure kept
  its own energy and adopted another structure's geometry, which `parents_digest`
  then carried into every step computed from it. The document now names the digest
  of the arrays it was written with, and a mismatch is a rebuild.

- A calculation that diverges to a non-finite energy is refused at the parse
  boundary and ledgered, instead of ranking as a real result. Nothing downstream
  treated `nan` as a failure, and since every comparison against it is false it
  sorted by list position — so a `min, count: 2` step could keep it over the
  genuinely second-best conformer, which then went on to win the next step.
  Gradients are held to the same rule, since a non-finite force is what an
  `mlip-train` step would go on to fit.
- Local dispatch stops the calculation, not just the `bash` wrapper in front of
  it. Each job now runs in its own process group and is signalled as one; before,
  the shell deferred its TERM trap while waiting on the calculation, so the grace
  period expired, the shell was killed and the calculation kept running,
  reparented to init — with no copy-back, no scratch teardown and no runlog
  footer. A 64-job batch also took over five minutes to unwind, which on Ctrl-C
  reads as a hang.
- A structure retried after failing to converge keeps its lineage. The retry
  rebuilt it without `parent_id`, so it came out of the step an orphan — and
  `by_parent` filtering groups on `parent_id or id`, so it formed its own
  singleton group and survived a filter that should have discarded it.
- `pyscf-extopt` refuses to serve a gradient from an SCF that did not converge.
  PySCF returns the last iterate rather than raising, and ORCA's `.out` records
  only its own geometry convergence, so the result ranked against correctly
  converged siblings unmarked. Set `strict_scf: false` to accept it knowingly.
- An `nms` `ts_mode_index` that names no imaginary mode is rejected. The
  exclusion was written as a filter, so an index matching nothing excluded
  nothing and NMS displaced along every imaginary mode — including the reaction
  coordinate the setting exists to preserve — then reported every structure
  unresolved after paying for the whole round-2 batch.
- Editing an `mlip-train` step's MACE config re-runs the training. Its cache key
  omitted the template digest, so retuning epochs or learning rate and running
  `resume` was a cache hit that left the previous model on disk.
- `job_timeout_seconds` now applies to an `mlip-train` step. Its wait loop had no
  deadline, so a training job stuck in `PD` blocked the pipeline indefinitely
  instead of failing with exit code 8.
- ORCA templates using `SloppyOpt` or `VeryTightOpt` are recognised as
  optimisations; the keyword surface is now the one ORCA 6.1.1 actually accepts,
  checked against the binary. A malformed gradient row is reported as a parse
  error for that structure rather than escaping as a traceback that ends the run.
- Frequencies are read from the **last** Hessian in an ORCA output, matching
  the energy, geometry and thermochemistry parsers. A TS search recomputes the
  Hessian as it goes and prints one table per recompute; v1.3.1 accumulated
  across all of them, so imaginary modes a structure had *before* it converged
  were still reported after. A converged transition state came back with
  several imaginary modes instead of one, and normal-mode sampling then went off
  resolving modes that no longer existed — on one real TS run, 66 of 72 round-2
  jobs were spent on structures already at the target.
- Each structure's cache record names the displaced child a normal-mode-sampling
  run resolved it from (`resolved_from`), and the attempt directory keeps a
  `resolution.json` saying the same. A resolved parent keeps its own ID and the
  winning child's files are promoted to the parent's names, so nothing else
  recorded which of the ± children actually produced them.
- A retried normal-mode-sampling child no longer carries its own discarded runs
  into its parent's attempt directory when it wins.
- `rebuild-cache` no longer rewrites the `.result.json` records of NMS round-2
  children while re-reading them. Re-deriving a cache from a finished tree now
  modifies nothing.
- `random` normal-mode sampling no longer displaces along a translation or
  rotation when a molecule has no vibrational modes left to draw from — a
  diatomic could previously be handed a job that moved it and recomputed the
  same energy.
- An engine missing one of the four job primitives now fails when it is
  constructed rather than after a step's jobs have been submitted.
- The step cache is plain data instead of a pickle — loading it can never
  execute code from the file. It is two files: `_cache/step.json` for the
  metadata and `_cache/arrays.npz` for coordinates and forces, which at 10,000
  structures is 31 MB and 0.36 s to load against 69 MB and 1.73 s for a single
  JSON document. `step.json` is written without indentation — read it with `jq`
  or `json.load`; each structure also gets an indented `.result.json` beside its
  output files. Caches from earlier dev builds rebuild automatically.
- ORCA's `.opt` restart file and `.property.txt` are copied back out of the
  scratch directory with the rest of the results. `.opt` is what lets a stalled
  optimisation resume where it stopped rather than start over.
- Normal-mode sampling no longer overwrites the calculation it was launched
  from. The round-1 job is archived into the same `attemptK/` its displaced
  children ran in, and the winning child's artifacts are promoted to the
  structure's canonical path — so `stepN/<id>/` describes one calculation, and
  the geometry, output, orbitals and Hessian there all agree.

- Cluster SLURM headers using `--ntasks-per-node` / `--ntasks-per-core` keep
  those directives in generated scripts.
- An ORCA template requesting more `%pal` ranks than `max_cores` is clamped
  to the budget instead of oversubscribing its allocation.
- A typoed `pyscf-extopt` option fails the step instead of silently running
  with defaults.
- Corrupt output files (overflowed coordinate tokens, malformed ensemble
  frames) land in the failed-jobs ledger instead of crashing the run.
- The ExtOpt tensor-dump tag is sanitized before filename use.
- The ExtOpt server requires a per-run bearer token on `/calculate` (written
  `0600` next to `server.url`), so other users on a shared compute node can
  no longer drive it.
- `on_failure: best` no longer aborts the run when the step samples on `gibbs`,
  `enthalpy`, or `electronic_zero_point`. A backfilled structure carries no
  thermochemistry, and raising on it defeated the one policy meant to keep
  going; it is now excluded from the ranking, and only a step where *nothing*
  has the requested energy is a config error.
- A step's GPU demand, its SLURM header, and the device rendered into its script
  are all read from the engine's own options model, so they cannot disagree.
- The ORCA executable and `$OUTPUT_DIR` are shell-quoted in the ExtOpt run block
  too, not just the plain ORCA one — a path containing a space broke ExtOpt steps.
- Normal-mode sampling seeds its RNG per structure, so `rebuild-cache` re-derives
  the same displaced children a run did even when it skips a structure the run
  visited (previously it reported resolved structures as unresolved).
- A missing input template or SLURM header exits with the documented config-error
  code instead of an uncaught traceback.
- A truncated `.extinp.tmp` is a classified job failure rather than an
  `IndexError` inside the ExtOpt wrapper.
- `squeue` missing from `PATH` no longer raises on every poll; the cache is
  `fsync`ed before its atomic rename.

## [1.3.1] and earlier

The pre-rewrite line; see the
[GitHub releases](https://github.com/sterling-group/ChemRefine/releases)
for its history.
