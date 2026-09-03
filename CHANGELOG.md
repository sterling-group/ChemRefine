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
[migrating from v1 to v2](https://sterling-group.github.io/ChemRefine/get-started/upgrading-from-v1/)
for the full map.

### Deprecated

- The v1.3.1 compatibility layer — legacy YAML keys (`calculation_type` aside,
  which already raises), the `mlff*`/`dft` engine spellings, the `sample_type`
  block, and the flag-style CLI (`chemrefine CONFIG --rebuild_cache N`) — is
  scheduled for removal in **3.0.0**. It warns once per rewritten feature today.
  Flags the translator does not map pass through to the subcommand CLI, so the
  current flags work in the legacy spelling (`chemrefine cfg.yaml --dry-run`)
  and a flag neither CLI knows is refused loudly rather than silently dropped.
  See [migrating from v1 to v2](https://sterling-group.github.io/ChemRefine/get-started/upgrading-from-v1/).

### Added

- **Config tooling** (`chemrefine validate | scaffold | schema | engines`):
  `validate` reports every finding at once — pydantic errors with field locations,
  unknown engines, bad values for declared option knobs, invalid NMS knobs — plus
  warnings for the silent no-ops (undeclared option keys, `nms: true` on an engine
  that cannot NMS, step templates and SLURM headers that do not exist yet, resolved
  the way dispatch resolves them); warnings never block, an unrunnable config exits 2.
  `validate` and `load_config` enforce the same refusals: a non-string top-level
  key — an unquoted `on:`, `yes:` or `no:`, which YAML 1.1 parses as a boolean —
  is the documented validation error, and `template_dir` / `output_dir` /
  `scratch_dir` are refused when they contain a shell metacharacter (`"`, `$`, a
  backtick, a backslash or a newline), checked on the **resolved** paths because
  the generated scripts export them into bash.
  `scaffold` writes commented starter templates into every gap the config expects;
  two steps of different engines naming one template file are refused by name
  rather than the second starter silently replacing the first.
  `schema` prints a machine-readable document generated from the validating models —
  the config schema, the NMS knobs, one descriptor per registered engine (declared
  options schema, capabilities, `operation:` vocabulary) — and `engines` lists the
  registry. What the GUI's forms and any AI agent read instead of the prose docs.
- **MCP server** (`chemrefine mcp`, extra `chemrefine[mcp]`): ChemRefine's agent tools
  served over the Model Context Protocol on stdio, so Claude Code/Desktop, Cursor and
  friends can author, run, and triage workflows (`claude mcp add chemrefine --
  chemrefine mcp`; over SSH for a cluster). The surface: schema/introspection,
  validation, `save_config` (validation gates the write), template read/write/scaffold,
  detached `start_run` with filesystem-read `run_status`/paginated `get_results`/
  `get_failures`, and chemistry grounding — `build_structures` (SMILES/XYZ with
  charge-parity checks), `lookup_smiles` (PubChem), `get_frequencies`
  (minimum-vs-TS from cached imaginary modes) and `analyze_mode` (which atoms and
  bonds a normal mode moves — reaction-coordinate validation). Contracts the tools
  pin down: `start_run` refuses a `target` carrying a `run`/`resume` subcommand,
  `run_status(log_tail_lines=0)` returns an empty tail, a disk failure from
  `save_config`/`write_template`/`scaffold_templates` surfaces as the documented
  `ConfigError`, and a `build_structures` call owns its output directory's seed
  set — earlier `structure_*.xyz` are cleared and a call that fails partway
  leaves none behind, so `input:` directory-seeding reads exactly what the call
  reported. The packaged operating guide ships as the `chemrefine://guide`
  resource, with its vocabulary pinned to the code by tests.
- **Workflow-builder GUI** (`chemrefine gui`, extra `chemrefine[gui]`): a local
  two-pane web app (127.0.0.1 behind a per-session token, on a stable per-user
  port by default so an SSH forwarding setup written once keeps working;
  kernel-assigned when that port is taken) — click-through forms
  rendered from the live schema on the left, the `input.yaml` on the right, emitted
  and parsed server-side only. Validate anchors findings to fields; Save…/scaffold/
  inline template editing; a run dashboard (start/resume/rerun-errors behind
  confirmations, polling status, paginated `steps.csv` results); and, with the
  `[agent]` extra, an agent chat panel whose mutating tool calls arrive as allow/deny
  cards. A browserless session — an HPC login node — prints the SSH forwarding
  recipe instead of hijacking the terminal with a text browser. Run without its
  extra installed, `chemrefine gui` — like `chemrefine agent` — exits with a
  message naming the extra to install. The builder also publishes on the docs
  site as the **Playground** (top navigation) in a build-and-copy static mode.
- **Embedded agent** (`chemrefine agent`, extra `chemrefine[agent]`): a terminal chat
  over the same tool surface, harnessed by PydanticAI — multi-provider (presets for
  local Ollama/vLLM, any OpenAI-compatible endpoint via `--base-url`, or native
  `provider:model` strings; configuration via `CHEMREFINE_LLM_*`). Every mutating tool
  sits behind a y/N confirmation whose refusal is reported to the model as an answer.
  A base URL is accepted only with an `http(s)` scheme — provider resolution
  refuses any other before the URL is paired with `CHEMREFINE_LLM_API_KEY`, on
  every path (CLI flags, the `CHEMREFINE_LLM_*` variables, the GUI chat panel).
  `--check` verifies the configured endpoint and names the fix without downloading
  anything — model choice and site policy stay the user's (see the new
  platforms/model-policy docs page).
- **Q-Chem engine** (`engine: qchem`): per-structure Q-Chem jobs from a `stepN.in`
  template, with the geometry generated into job 1's `$molecule` block (a multi-job
  `@@@` chain's later `$molecule read $end` survives untouched). Parallelism is
  CLI-side, as Q-Chem wants it: `options.cores` renders `-nt N` and is allocated as
  `--ntasks=1 --cpus-per-task=N`; `options.nprocs` opts into MPI (`-mpi -np P [-nt N]`,
  allocated `P×N` — partial method support, so never a default). The install environment
  comes from `executables: {qchem, qc, qcaux}` — `qc` exports `QC`/`PATH` and Q-Chem's
  own documented `QCAUX=$QC/qcaux` default, `qcaux` overrides it for sibling layouts —
  or from a `module load` in the SLURM header. `QCSCRATCH` is the per-job work dir; the
  job always runs with a savename so key scratch (MOs) survives, and `options.save`
  copies it back to the structure dir. The `operation:` vocabulary is
  `sp` / `opt_sp` / `freq` — the GUI dropdown and the schema document pick it up —
  an unknown operation is refused at the run's preflight, and a step that omits
  `operation:` has its run type inferred from the template's `JOBTYPE`. NMS-capable:
  `jobtype ts`/`freq` gate and target the sampling, and the frequency parse maps
  Q-Chem's 3N−6 vibrational modes onto the trivial-modes-first tensor the
  coordinator expects. Geometry is exchanged in Ångström only — a template setting
  `$rem input_bohr` is refused by name, since the writer emits Å and the reader
  assumes it. The output reader is deliberately minimal (final energy + last
  geometry) pending the full parser set.
- **Memory-aware SLURM requests**: an input that declares its memory now shapes its
  allocation. ORCA's `%maxcore` requests `ceil(maxcore × pal ÷ 0.75)` (maxcore is a
  promise ORCA overshoots per process — the 75% rule); Q-Chem's `mem_total` requests its
  declared peak. A header whose own `--mem`/`--mem-per-cpu` already covers the
  requirement stands untouched; a short or absent one is extended to `--mem-per-cpu`,
  with the override logged. Inputs that declare nothing keep the header's policy,
  exactly as before.
- Per-step ensemble XYZ files: every step leaves `stepN_ensemble.xyz` (all of
  its final structures, one multi-frame XYZ) and `stepN_survivors.xyz` (the
  subset the `sample:` filter kept) in its step directory. Frames are sorted
  ascending by the step's own ranking energy and captioned
  `stepN id=<id> E=<hartree> Eh`, so each is traceable to its structure
  directory and `steps.csv` row; `resume` and `rebuild-cache` regenerate both
  files byte-identically.
- One driver per output tree: every run holds an advisory lock
  (`<output_dir>/.chemrefine.lock`) for its whole duration, and a second
  `chemrefine` pointed at the same tree exits with code `10` instead of
  archiving and resubmitting the first one's in-flight work. A lock left by a
  driver that died on the same host is reclaimed automatically; one left on
  another host must be deleted by hand once that run is known dead (the error
  says so). A `scancel`-ed (SIGTERM'd) driver exits in an orderly way (code
  143), releasing the lock and terminating its local jobs — only a genuine
  SIGKILL can strand a lock, and the same-host dead-pid reclaim covers that
  case. `run_status`'s holder payload carries a three-valued `alive` field:
  `true`/`false` for a same-host holder, `null` for a holder on a foreign host
  no status read can probe.
- Engine plugin system: a `CalculationEngine` protocol plus a registry, with
  four documented base shapes for new engines (`engines/api.py`). Plugins and
  MLIP backends are **auto-discovered** — a new engine package or backend
  module is dropped in and registers itself, with no central import list to
  edit. Bundled engines: `orca`, `mlip`, `mlip-extopt`, `mlip-train`,
  `pyscf`, `pyscf-extopt`, `qchem`.
- Subcommand CLI — `run`, `resume`, `rerun [step]`, `rerun-errors [step]`,
  `rebuild-cache [step]`, `rebuild-nms [step]` — with documented process
  exit codes per failure class.
- Per-step result cache fingerprinted over the step config **and** the
  parent structures' content; `resume` re-executes only what changed.
  Filter-only (`sample:`) edits refilter cached results without re-running
  calculations.
- Per-step failure policy `on_failure: stop | skip | best` with a
  `failed_jobs.json` ledger; `resume` / `rerun-errors` re-attempt only the
  still-failed structures of a `stop` step. Changing a step's `on_failure` and
  resuming takes effect even over a cached step: ledgered failures are
  re-attempted and the step is re-finalized under the new policy, and successes
  are never recomputed (`stop` ↔ `skip` edits are a free cache hit, since both
  store the same results). Under `best`, a backfilled structure carries no
  thermochemistry and is simply excluded from rankings on `gibbs`, `enthalpy`
  or `electronic_zero_point`; only a step where *nothing* has the requested
  energy is a config error.
- Automatic convergence retries: a structure that fails to converge is
  re-attempted, and the retry joins the step's own throttled queue the moment
  a slot frees, overlapping the rest of the batch. Results return in manifest
  order, so a step that retried changes `parents_digest` and re-runs its
  downstream steps once, settling after one run. `job_timeout_seconds` is a
  **stall deadline** — the longest a step may go with *nothing* finishing —
  whose clock restarts on every completion (and follows the array's *tasks*
  under `slurm_array: true`); a batch that keeps draining never trips it,
  however long the step takes.
- Two-round normal-mode sampling (`nms: true`) with target-aware displacement
  (`minimum` / `ts` / `random`); tuning the search parameters re-attempts only
  the unresolved parents. `random` draws displacement modes only from a
  molecule's vibrational modes — translations and rotations are never
  candidates, and a structure with none to draw from (a diatomic) simply
  yields no displacements. A `ts_mode_index` that names no imaginary mode of
  the structure is rejected up front, so the mode the setting exists to
  preserve — the reaction coordinate — is never displaced. After sampling,
  `stepN/<id>/` describes a single calculation: the round-1 frequency job is
  archived in the same `attemptK/` as its displaced children, and the winning
  child's geometry, output, orbitals and Hessian are promoted to the
  structure's canonical path, so the files there all agree.
- Shared ExtOpt HTTP server: ORCA optimises on gradients served by an MLIP
  or PySCF backend in the same SLURM job (kernel-assigned ports, health
  probe, clean teardown). A single-environment ExtOpt step requires flask and
  waitress at preflight — the run fails before submission with an error naming
  both fixes (`pip install "chemrefine[server]"`, or
  `chemrefine backends install <extra>`) — and a server that cannot start
  logs one actionable line to its `--log-file`, the file the job's failure
  path tails.
- PySCF engines: `pyscf` runs each structure through the step's own rendered
  Python script, and `pyscf-extopt` serves gradients to ORCA through the ExtOpt
  server. `pyscf-extopt` validates its options strictly — an unrecognized or
  misspelled option fails the step rather than silently running with defaults.
  Both engines require the level of theory to be named — `basis`, and `xc`
  for `method: dft` — the direct engine refusing at preflight: every other
  engine makes the user say it (ORCA in the template's `!` line, Q-Chem in
  `$rem`), and a silent default would compute at a level nobody chose. The
  options model carries no `xc`/`basis` defaults anywhere any more, which
  also re-keys the cache rows of pyscf steps recorded under the old implicit
  `pbe` — `chemrefine rerun` such a step, or strip its manifest's row
  provenance and `rebuild-cache` to adopt the outputs under the new keys.
  `strict_scf`
  (default on) refuses to serve a gradient from an SCF that did not converge:
  PySCF returns the last iterate rather than raising, and ORCA's `.out` records
  only its own geometry convergence, so an unconverged result would otherwise
  rank unmarked against converged siblings; set `strict_scf: false` to accept
  such gradients knowingly. `save_tensors` is restricted-only — an open-shell
  (UHF/UKS) step is refused at prepare, before anything submits, naming the
  knob and the multiplicity. A `device: cuda` step requires the gpu4pyscf
  stack: the preflight derives the requirement from the step's options and
  fails by name, pointing at the `pyscf-gpu` extra, when the environment lacks
  it. Step scripts (`mlip` and `pyscf` alike) return positions and gradients
  as `(N, 3)` arrays — a flat `(3N,)` gradient is an ordinary `UNPARSEABLE`
  ledger entry naming the atom count — and a required output field must be
  present and non-null: a malformed output document is a ledgered per-structure
  parse failure, never a crash.
- Per-backend MLIP extras (`mlip-fairchem`, `mlip-mace`, `mlip-sevenn`,
  `mlip-orb`, `mlip-chgnet`) and a backend-agnostic calculator factory. Each
  backend installs into its own dedicated environment (their torch/e3nn trees
  conflict); `[mlip]` defaults to FAIRChem / UMA (`uma-s-1p2`, from upstream
  `fairchem-core >= 2.18` on PyPI).
- Managed backend environments: `chemrefine backends {install,list,path}`
  provisions one env per backend (built with the same tool that created the
  current env — conda / uv / venv) and steps resolve them **by name**, so
  conflicting MLIP stacks (e.g. MACE + UMA) run side by side in one pipeline.
  Every run validates its steps' backends before any job submits. Each env is
  built on **the Python its backend supports**, not the orchestrator's: every
  backend extra states which Pythons it installs on (`mlip-orb` 3.12 only;
  `mlip-chgnet` up to 3.12; `mlip-mace` up to 3.13), `backends install` builds
  on the newest one it claims — stopping with the ways out when no such
  interpreter is available — and `--python` (a version, a command name, or a
  path) overrides the choice. An env built on a Python its backend excludes is
  refused rather than installed into, because pip succeeds there having
  installed nothing.
- **Every shipped MLIP backend fine-tunes** — v1 trained MACE only.
  `task_name: sevenn | chgnet | orb` join MACE and the FAIRChem heads as
  `mlip-train` selections, so the family `[mlip]` installs by default trains
  too; the `mlip-sevenn` floor is 0.11.1, the release with the unified
  `sevenn train` CLI. Libraries without a charge/spin channel warn instead of
  silently fitting an ion as neutral data, and running what you trained is the
  same `model_path:` line for all five. FAIRChem's dataset is an ASE database
  per split — labels on a `SinglePointCalculator`, `metadata.npz` beside each,
  one directory per split because FAIRChem resolves a missing `metadata_path`
  against the database's *parent*. The worked example ships at
  `examples/tutorials/fairchem_finetune` — label with UMA, fine-tune through
  fairchem's own recipe collapsed into one commented config, run the produced
  checkpoint (the UMA weights themselves stay behind a gated Hugging Face
  repo).
- SLURM-optional execution: the generated scripts run unchanged under
  `bash` with the same core/GPU throttling, runlogs, and artifacts. Local GPU
  jobs run inside the allocation the run inherited: `CUDA_VISIBLE_DEVICES` is
  authoritative, and `max_gpus` may narrow it but never widen it. Device
  tokens are carried verbatim into each job's pin, so GPU UUIDs and `MIG-…`
  handles work as well as bare indices.
- Opt-in job-array submission (`slurm_array: true`): each step goes out as
  one `sbatch --array` per ≤1000 structures, with the scheduler enforcing
  the `max_cores` budget via the array's `%limit` — large ensembles submit
  in seconds instead of one sbatch call per structure. Outputs, runlogs,
  recovery, `job_timeout_seconds` (measured per array *task*, so a draining
  array keeps resetting the clock) and the GPU-budget config checks behave
  identically to the per-job path. A step large enough to split across several
  arrays divides `max_cores` among the chunks, so together they never exceed
  the budget; a share is not returned when a sibling chunk drains early, so
  the tail of an N-chunk step runs at up to 1/N of the budget.
- Seeding from a multi-frame `.xyz`, a directory of `.xyz` files (all
  frames), or a CSV of SMILES (deterministic 3D embedding).

### Changed

- **The cache identity is per-structure, and `resume` is incremental.** A step's key is
  layered the way its work is: a **row key** per structure (engine, template bytes,
  effective charge/multiplicity, the options as the engine's declared model reads them,
  option-file digests, the parent's content), a **resolution key** for what NMS reads
  (criterion and search), and a step fingerprint composed from the ordered rows. The
  manifest records that identity per row, and `resume` adopts every row whose stored
  key matches — re-parsed from disk, never resubmitted — computing only the rest. What
  that means in practice: turning `nms: true` on over a finished frequency step
  submits only the imaginary parents' displacement children (round 1 and the clean
  minima are adopted, byte-identical); follow-up steps recompute only rows whose
  parent actually changed; a search retune re-reads round 1; a criterion change
  re-resolves and nothing more; an interrupted NMS step adopts its finished round 1;
  a typo'd option key nothing declares invalidates nothing (`validate` already warns
  about it). Row keys hash the **effective** charge and multiplicity each job
  renders — inherited workflow-level values included — so editing a workflow-level
  `charge:`/`multiplicity:` invalidates every step that inherits it. A tree from
  before these rules is adopted once, explicitly, by `rebuild-cache` — which
  re-parses under the current rules, submits nothing, and writes the provenance —
  and `resume` names exactly that command instead of silently archiving finished
  work.
- **The manifest moves with the tree.** `_cache/manifest.json` spells its input and
  output files relative to the step directory, so a copied or moved output tree can be
  `rebuild-cache`d, `rerun` and inspected where it lands; it used to record the absolute
  paths of the machine the step ran on. Manifests written before this — and hand-written
  v1 adoption manifests — carry absolute paths and are read exactly as before.
- **A `task_name` you state wins over the `model_path` shortcut.** A checkpoint with no
  library named still means MACE, as it always has; naming one loads the checkpoint with
  *that* library. The old rule sent any `model_path` to MACE, which was right while MACE was
  the only library that could produce one and would now silently load a FAIRChem model with
  the wrong loader.
- **`mlip-train` is rebuilt.** It runs through the same scheduler and throttle as every
  other engine, so a training job is charged against `max_cores` / `max_gpus`, gets the
  device-aware SLURM header, the runlog, the scratch handling, local dispatch and
  `job_timeout_seconds` — none of which it had. Which library trains is now `task_name`,
  resolved through a registry keyed exactly like the calculator's: one word names the
  library whether a step trains a model or runs one. Adding a trainable backend is one
  dropped-in module under `engines/mlip/backends/`, beside the calculators for the same
  library — the environment it needs is then declared once for both.

  The step's YAML changes: `task_name` and `device` are required (neither has a
  default worth guessing); `model_name` is optional — the foundation model a
  fine-tune starts from, and unset means training from scratch (`started_from:
  scratch` in the runlog) rather than silently inheriting a FAIRChem checkpoint
  name as everyone's foundation; `job_name` is gone (the job is named like every
  other); `valid_fraction` now means what it says, and `test_fraction` is the
  separate held-out set it used to be confused with. The template is the
  trainer's own config `stepN.yaml`, **rendered** through `$PLACEHOLDERS` rather
  than patched — patching is what could not work across libraries, since the
  keys the old code inserted are meaningful to MACE and rejected outright by
  FAIRChem. The step's own facts — `device`, `seed`, and the foundation
  weights — are authoritative over the template: for the libraries trained
  through a Python API (CHGNet, ORB) chemrefine delivers them to the training
  process directly and refuses by name a template value that disagrees; for the
  libraries that read their own config file (MACE, SevenNet, FAIRChem) the
  template must reference the placeholders that carry them (`$DEVICE`, and
  `$SEED` where the library reads one), so a `device: cuda` step cannot
  silently train on CPU with the GPU booked. `operation: mlip_train` is a
  legacy spelling and still translated.

  A training step produces no structures of its own — the previous step's
  ensemble passes through unchanged to the following step. Its success test is
  the trained model's existence: `resume` retrains after a failed training, and
  a training that completed before the driver died is adopted by
  `rebuild-cache` without recomputing it. The step's fingerprint includes its
  training-config template, so retuning epochs or learning rate and resuming
  re-runs the training and replaces the model on disk. The preflight refuses a
  training step whose backend can run but not train — before any step
  submits — and names which backends can.

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
  `model_path`, since that is the only thing it can mean. `model_name` names weights
  *within* the selected library and is optional — unset means the library's own
  default (FAIRChem's builder keeps `uma-s-1p2`; SevenNet loads its own default
  release). In v1 the `model_name` prefix (`sevenn…`, `orb…`) doubled as the backend
  selector; it no longer selects anything.

- The scoped recovery actions cover the steps around their target deliberately:
  - `rerun-errors N` repairs step N and then continues the run — the steps after
    it are exactly the ones the halt stopped from ever running.
  - `rebuild-cache N` rebuilds step N, then walks the steps after it read-only:
    each is served from its cache — a load, never a re-parse, and never a
    submission — and the first cache the current configuration cannot serve ends
    the run quietly, its message naming the cheapest repair (`rebuild-cache` for
    outputs that still match this configuration, `resume` when upstream results
    changed). Re-deriving a cache from a finished run tree is read-only: it
    modifies nothing else, including each structure's `.result.json` records.
  - `rebuild-nms [N]` re-resolves an NMS step from the outputs already on disk and
    submits nothing — the same rebuild `rebuild-cache` performs, aimed at the step
    setting `nms: true` rather than the last one. `--rebuild_nms` from the v1 CLI
    maps here, and means what it meant there.
  - `resume` and `rerun-errors` decide what is left to do by what actually
    parsed, not by whether an output file exists — a structure whose output is
    present but truncated or otherwise unusable is re-run rather than reported
    failed.
- v2 YAML schema: `engine:` + `operation:` replace `calculation_type`;
  `sample:` replaces `sample_type:`; `nms:` + `options:` replace
  `normal_mode_sampling*`; `executables:` replaces `orca_executable`;
  `input:` replaces `initial_xyz`. Legacy spellings are rewritten with a
  deprecation warning — except `calculation_type`, which raises with a
  pointer to the migration guide.
- `mlff` renamed to `mlip` everywhere (engines, extras, YAML); the old
  spellings remain as aliases.
- `options.device` defaults to `cpu` on all four MLIP/PySCF engines (`mlip`,
  `mlip-extopt`, `pyscf`, `pyscf-extopt`). It is read by both the rendered
  script and the scheduler, so one setting drives the header and the run;
  request a GPU explicitly with `device: cuda`.
- Keeping scratch artifacts is engine-owned — an engine declares what it copies
  back (`output_dirs`, or its run block's cleanup) — not a flag on the SLURM
  script builders: v1's `save_scratch` has no v2 spelling, and
  `chemrefine.slurm.build_script` takes no such parameter.
- ChemRefine now publishes to PyPI — v1 installed from `git+https` only.
  Supported Python is 3.11 through 3.14 (v1 declared 3.9–3.12). The sdist
  carries the test suite and the shipped examples, so the suite can be run
  from the unpacked tarball; `docs/` stays out of the tarball on size.

### Fixed

- **An MLIP optimisation that runs out of steps is a failure, not a survivor.**
  `MlipCalculator.optimize` discarded the verdict ase's `LBFGS.run` returns, and the
  script output contract had no field to carry one, so the last geometry of an
  unconverged relaxation parsed as a result: ranked against converged siblings, cached,
  and handed to the next step with nothing ledgered. The shared contract now carries
  `converged` (a flag, exempt from the finiteness sweep; a template that never assigns
  it still reports nothing), the helper keeps the verdict on `atoms.info["converged"]`
  and `last_converged`, and the `mlip`/`pyscf` starters and the quickstart template
  assign it — so an exhausted optimiser or a loose SCF is ledgered as a convergence
  failure and retried once from its best geometry, as the retry docs already describe.
- **A run warns about option keys nothing reads, not only `chemrefine validate`.**
  The undeclared-key rule — a key outside the engine's declared model (and NMS's, on an
  `nms: true` step) changes nothing, and a typo of a real knob looks exactly the same —
  was computed by the validate report alone, which a `resume` after an edit, the GUI's
  Run button and an agent's `start_run` never pass through; `target: ts` misspelt on an
  NMS step ran a minimum search in silence. The run's own preflight walk now logs the
  same sentence at its start (a warning, not a refusal: the lenient script-engine read
  is the documented design), and the sentence names the readers instead of hedging on a
  placeholder path that never existed.
- **A v1 `energy_window` value keeps its hartree meaning across migration.** v1 read
  `energy` as hartree unless `unit: kcal/mol` was explicit; the translation layer
  carried the bare number into `window_kcalmol` — kcal/mol by definition — so a config
  that relied on v1's default filtered with a window ~627.5× too narrow, silently, while
  the deprecation warning ("use `window_kcalmol`") read as an endorsement of the value.
  The number now converts (mirroring v1's own rule: only an explicit `kcal/mol` crosses
  unchanged), and the warning names both values so the translation is checkable.
- **`model_path` reaches every MLIP builder — in v1, SevenNet and ORB silently ignored
  it** and loaded the *named release* instead of the checkpoint. SevenNet now takes the
  path through its own `model=` (typed `str | Path`, filesystem checked before release
  names) and ORB through the loaders' `weights_path=`, each behind the existence check
  MACE and FAIRChem already had; an orb-models too old for the keyword is a named
  version limitation, not a `TypeError`. CHGNet's checkpoint support was broken by
  another route — `CHGNet.load(path)` is keyword-only and resolves *release names* —
  and local checkpoints now load through `CHGNet.from_file`, whose `{"model":
  as_dict()}` shape is exactly what the new trainer saves.
- **`model_name` reaches CHGNet.** In v1 the CHGNet branch never consumed it —
  `CHGNet.load()` ran bare, so a pinned `model_name: "0.3.0"` silently served the
  latest release instead. A release name now routes through
  `CHGNet.load(model_name=...)`.
- **An ORCA step under a path containing whitespace fails with its reason, not ORCA's.**
  ORCA reads each geometry through `* xyzfile <path>`, which is whitespace-delimited and not
  a quotable field — it truncates at the first space (`CANNOT OPEN FILE`, naming the prefix,
  with or without quotes around the value) — and it execs an ExtOpt wrapper through `sh`,
  which splits on one (`sh: 1: /path/my: not found`). Both paths derive from `output_dir`, so
  every `orca` / `mlip-extopt` / `pyscf-extopt` step under such a tree failed once per
  structure, naming a path nobody wrote. The ORCA input writer now refuses before it emits
  the directive, `chemrefine validate` warns about the affected steps beforehand, and the
  check runs on the **resolved** path — so a directory reached through a symlinked parent,
  which contains no whitespace anywhere in the YAML, is caught too.
  Deliberately scoped to the engines that write paths into an ORCA input: `mlip`, `pyscf`
  and `qchem` steps run fine from such a tree (Q-Chem inlines the geometry, the script
  engines quote the path, the generated bash quotes everything it interpolates), and so do
  `template_dir` and `scratch_dir`.
- **A diverged calculation's geometry is refused instead of stored.** A non-finite
  energy has long been a parse failure; the *coordinates* were not — a `NaN`
  geometry could be kept, carried through the cache with a perfectly stable
  fingerprint, and every later step computed from coordinates that are not
  numbers with nothing anywhere saying so. Non-finite geometries are now refused
  where they enter — the ORCA, Q-Chem and script parse boundaries raise the same
  error a `*****` overflow token already raises, and a seed `.xyz` is refused by
  name — with the cache as a backstop that refuses to store what did get through.
  The ensemble readers keep skipping rather than failing a whole step for one bad
  conformer; `nan` now simply follows the rule `*****` always had.
- **A non-ASCII token is a 401 from both token gates, not a 500.** The GUI's
  `X-ChemRefine-Token` check and the gradient server's `Authorization` check answer a
  token containing a byte above 0x7F like any other wrong token. Both gates are
  reachable by any user on the node, which makes them exactly the place that has to
  answer plainly whatever they are handed.
- **A step naming a model file works where crypto policy restricts SHA-1.** ChemRefine
  hashes file contents only to decide what to re-run, and says so: every content hash
  is declared `usedforsecurity=False`, so caching and resume work on hosts whose
  crypto policy restricts SHA-1 to non-security use. Cache keys are unchanged — the
  flag is a policy hint, not an input to the hash.
- **`_cache/` documents are readable on a shared tree.** The atomic writer's temp file
  is created 0600 and the rename preserved it, so the cache, manifest and failure
  ledger were owner-only beside world-readable outputs; they now honour the umask like
  any other written file (the server's token sidecar stays 0600 on purpose).
- **An ExtOpt gradient server runs on the step's own core budget.** The server — the
  compute half of an `mlip-extopt` / `pyscf-extopt` job — inherited an uncapped thread
  environment, so torch/MKL took every core the allocation allowed while the scheduler
  charged the job its `%pal`. The job script now exports the pal thread count before
  launching the server, and `1` for ORCA alone, which in ExtOpt mode is only the
  stepper.
- **The training dataset is readable by the trainer.** v1 wrote `DFT_energy` /
  `DFT_Forces` while MACE's defaults are `REF_energy` / `REF_forces`, and the shipped
  template declared a third pair — MACE refuses a file in which it finds none of its
  keys, so every training job died at data load. The dataset now speaks MACE's own
  default keys, so a template needs no `energy_key`/`forces_key` line to be correct.
- **Charge and multiplicity reach the training set.** MACE reads `total_charge` /
  `total_spin` and silently defaults them to a neutral singlet, so a v1 fine-tune on an
  ion or an open-shell system was fitted against the wrong species with nothing said in
  any log.
- **The training command resolves.** v1 emitted a bare `mace_run_train`, which is on
  nobody's `PATH` once MACE lives in its own environment — which it must, since its `e3nn`
  pin cannot share a prefix with FAIRChem's. `mlip-train` is now a provisionable engine:
  its backend is checked by the preflight and its job launches from the managed env.
- **Retraining re-runs the steps that use the model.** A step naming a file in its options
  now has that file's contents in its cache key, so a step consuming a retrained model no
  longer serves a result computed with the previous weights.
- **Editing a template-referenced file re-runs the steps that read it.** The same rule for
  the files an ORCA template names by quoted reference — a `%DOCKER GUEST` geometry, a
  `%pointcharges` file: their bytes are part of the step's cache key, so editing one in
  place makes `resume` re-run the step instead of serving results computed from the old
  file. One-time cost: a tree whose templates reference aux files re-runs those steps once
  when first resumed under this version (steps naming none are keyed exactly as before).
- A relative path in a step's `options` — `model_path` — resolves against the
  **config file's** directory, like every other path a config names, instead of the
  process working directory. A v1 config that named a model relative to the run or
  step directory (as the shipped MLIPTraining tutorial did with
  `../step3/checkpoints_dir/...`) must be rewritten relative to the config file.
- A FAIRChem model can be given as a checkpoint file path as well as a registry
  name — which is how a fine-tuned FAIRChem checkpoint is run. v1 resolved
  registry names only.
- NMS round 2 runs in the step's own queue instead of after the whole step drains.
  A structure needing displacement gets its `attemptK/` and its ± children
  submitted the moment its own round-1 job finishes, alongside whatever is still
  running — in v1 no child started until every round-1 job had drained, so the
  slots freed by early finishers idled until the last one landed. Picking each
  winner still happens once, after the queue drains, so survivors come back in
  manifest order and downstream fingerprints do not move.
- A calculation that diverges to a non-finite energy is refused at the parse
  boundary and ledgered, instead of poisoning the step's energy ranking (every
  comparison against `nan` is false). Gradients are held to the same rule, since
  a non-finite force is what an `mlip-train` step would go on to fit.
- A malformed gradient row in an ORCA output is reported as a parse error for
  that structure instead of being silently mis-read — v1 skipped the row and
  produced a wrong-shaped array. The opt-keyword surface ChemRefine recognises
  is the one ORCA 6.1.1 actually accepts — `SloppyOpt` and `VeryTightOpt`
  included — checked against the binary.
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
- The step cache is plain data instead of a pickle — loading it can never
  execute code from the file. It is two files: `_cache/step.json` for the
  metadata and `_cache/arrays.npz` for coordinates and forces, which at 10,000
  structures is 31 MB and 0.36 s to load against 69 MB and 1.73 s for a single
  JSON document. `step.json` is written without indentation — read it with `jq`
  or `json.load`; each structure also gets an indented `.result.json` beside its
  output files. The document records the digest of the `arrays.npz` it was saved
  with, and a pair from different saves — a save interrupted by a walltime kill
  or Ctrl-C — is refused on read-back and the step rebuilds, rather than
  structures loading another structure's geometry.
- ORCA's `.opt` restart file and `.property.txt` are copied back out of the
  scratch directory with the rest of the results. `.opt` is what lets a stalled
  optimisation resume where it stopped rather than start over.
- Cluster SLURM headers using `--ntasks-per-node` / `--ntasks-per-core` keep
  those directives in generated scripts.
- An ORCA template requesting more `%pal` ranks than `max_cores` is clamped
  to the budget instead of oversubscribing its allocation.
- Corrupt output files (overflowed coordinate tokens, malformed ensemble
  frames) land in the failed-jobs ledger instead of crashing the run.
- The ExtOpt server requires a per-run bearer token on `/calculate` (written
  `0600` next to `server.url`), so other users on a shared compute node can
  no longer drive it.
- The ORCA executable and `$OUTPUT_DIR` are shell-quoted in the generated run
  scripts — the plain ORCA and ExtOpt run blocks alike — so a path containing
  a space no longer breaks either kind of step.
- Normal-mode sampling seeds its RNG per structure, so `rebuild-cache` re-derives
  the same displaced children a run did even when it skips a structure the run
  visited (previously it reported resolved structures as unresolved).
- A missing input template or SLURM header exits with the documented config-error
  code instead of an uncaught traceback.
- A truncated `.extinp.tmp` is a classified job failure rather than an
  `IndexError` inside the ExtOpt wrapper.

## [1.3.1] and earlier

The pre-rewrite line; see the
[GitHub releases](https://github.com/sterling-group/ChemRefine/releases)
for its history.
