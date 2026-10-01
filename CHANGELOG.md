# Changelog

Notable changes to ChemRefine. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/). Update the Unreleased section
as part of any user-visible change; it becomes the release notes when the
version is tagged.

## [2.0.0] — Unreleased

A ground-up rewrite of the v1.3.1 pipeline. A translation layer rewrites
the v1.3.1 spellings in a YAML config or a flag-style CLI invocation to
the current ones and reports every rewrite as a deprecation warning. What
it cannot translate — `calculation_type`, an `engine:` left to v1.3.1's
`dft` default, a training template — is a hand edit; see
[migrating from v1 to v2](https://sterling-group.github.io/ChemRefine/get-started/upgrading-from-v1/)
for those and the full map.

### Deprecated

- The v1.3.1 compatibility layer — legacy YAML keys (`calculation_type` aside,
  which already raises), the `MLFF`/`DFT` engine spellings, the `sample_type`
  block, and the flag-style CLI (`chemrefine CONFIG --rebuild_cache N`) — is
  scheduled for removal in **3.0.0**. Until then it reports every rewrite as a
  deprecation warning. Flags the translator does not map pass through to the
  subcommand CLI, so the current flags work in the legacy spelling
  (`chemrefine cfg.yaml --dry-run`), and a flag neither CLI knows is refused,
  where v1.3.1 ignored an unknown flag silently.
  See [migrating from v1 to v2](https://sterling-group.github.io/ChemRefine/get-started/upgrading-from-v1/).

### Added

- **A per-step gradient timeout for the ExtOpt engines.** `gradient_timeout_seconds`
  (default `600`) bounds one call of the bridge ORCA invokes per geometry, where v1.3.1's
  wrapper waited on its server indefinitely. Expiry is recorded as a timeout that names
  the knob.
- **Every declared knob is a script placeholder, and the whole model is `$OPTIONS_JSON`.**
  A `stepN.py` for `mlip` or `pyscf` reads any field of its engine's options model as
  `$UPPERCASE` (`$MODEL_NAME`, `$BASIS`, `$GPU`, `$CORES`, …; an unset knob renders
  empty), and `options = json.loads("$OPTIONS_JSON")` takes the validated options whole —
  escaped so quotes, backslashes, newlines and `$` in a value survive the literal. Only
  declared knobs are placeholders: any other `$WORD` stays as written, and a key the model
  does not declare never reaches the script. For a template's own settings, `mlip` and
  `pyscf` steps take an `extra:` mapping — rendered as `$EXTRA` (a Python dict literal)
  and inside `$OPTIONS_JSON` — so a key you invent is deliberate and a typo of a real
  knob still warns; the engines that render no template refuse it.
- **Config tooling** (`chemrefine validate | scaffold | schema | engines`):
  `validate` reports every finding at once — pydantic errors with field locations,
  unknown engines, bad values for declared option knobs, invalid NMS knobs — plus
  warnings for the silent no-ops (undeclared option keys, `nms: true` on an engine
  that cannot NMS, a seed `input:`, step templates and SLURM headers that do not exist
  yet, resolved the way dispatch resolves them); warnings never block, an unrunnable
  config exits 2. A run logs the same undeclared-key warning when it starts, so a
  `resume`, the GUI's Run button and an agent's `start_run` report a misspelt knob too.
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
  two-pane web app (127.0.0.1 behind a per-session token, which the page moves out
  of the launch URL into the tab's `sessionStorage` on load so a bookmark or a
  copied address carries no secret; on a stable per-user port by default so an SSH
  forwarding setup written once keeps working; kernel-assigned when that port is
  taken) — click-through forms
  rendered from the live schema on the left, the `input.yaml` on the right, emitted
  and parsed server-side only; the executables row suggests the keys the registered
  engines read. Validate anchors findings to fields; Save…/scaffold/
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
- **A step that omits `operation:` takes it from its template.** v1.3.1 required
  `operation:` on every step. An ORCA-family step reads its run kind from the template's
  `!` keywords and the blocks ORCA accepts in their place — a `%geom … Scan` is a relaxed
  scan, a `%docker` or `%solvator` block a docking or a solvation (a `! GOAT`, `! DOCKER`
  or `! SOLVATOR` keyword outranks either block, and `%solvator` outranks `%docker`), and
  a reaction path, a band of images or a trajectory is named for what it is
  (`irc` / `neb` / `md`) and read as its final structure until a reader of its own joins,
  while a run that ends in a TS optimisation — `OptTS`, `ScanTS`, `NEB-TS` — is read as
  that saddle and targets `ts` for normal-mode sampling;
  a Q-Chem step reads its `JOBTYPE`. An explicit `operation:` always wins; one that
  reads the outputs unlike what the template runs (`opt_sp` over a `! GOAT` template reads
  one structure, never the ensemble) draws a warning when the run starts and from
  `chemrefine validate`.
- **Q-Chem engine** (`engine: qchem`): per-structure Q-Chem jobs from a `stepN.in`
  template, with the geometry generated into job 1's `$molecule` block (only job 1 of a
  multi-job `@@@` chain is edited — a later `$molecule read $end` survives untouched —
  and a block partitioned into fragments is refused by name when the run starts).
  Parallelism is
  CLI-side, as Q-Chem wants it: `options.cores` renders `-nt N` and is allocated as
  `--ntasks=1 --cpus-per-task=N`; `options.nprocs` opts into MPI (`-mpi -np P [-nt N]`,
  allocated `P×N` — partial method support, so never a default). The install environment
  comes from `executables: {qchem, qc, qcaux}` — `qc` exports `QC`/`PATH` and Q-Chem's
  own documented `QCAUX=$QC/qcaux` default, `qcaux` overrides it for sibling layouts —
  or from a `module load` in the SLURM header; `device: cuda` and `backend_python`,
  inherited knobs nothing here reads, are refused at preflight. `QCSCRATCH` is the
  per-job work dir; the job always runs with a savename so key scratch (MOs) survives,
  and `options.save` copies it back to the structure dir. The `operation:` vocabulary is
  `sp` / `opt_sp` / `freq` — the GUI dropdown and the schema document pick it up — an
  unknown operation is refused at the run's preflight, and a step that omits
  `operation:` has its run type inferred from the template's `JOBTYPE`: `pes_scan`
  infers `pes`, `rpath` infers `irc`, `aimd` infers `md` and the string methods keep
  their names — the engine-neutral words for a multi-geometry run, each read as its final
  structure with a warning that says so — and a job type the inspector does not list
  runs as `sp`. NMS-capable: `jobtype ts` targets the sampling and a frequency job —
  `jobtype freq` in an `@@@` chain, or `final_vibrational_analysis true` in the
  optimisation's own `$geom_opt` block — gates it; the frequency parse numbers Q-Chem's
  vibrational modes behind six zero columns, so the tensor and the frequency table share
  one index. Geometry is exchanged in Ångström only — a template setting
  `$rem input_bohr` is refused by name when the run starts, since the writer emits Å and
  the reader assumes it. Each output is read in one pass: the final energy and geometry, the forces,
  thermochemistry and frequency table where the run printed them, and whether it
  converged and terminated normally.
- **Memory-aware SLURM requests**: an input that declares its memory shapes its
  allocation, where v1.3.1 submitted every job with the header's memory as written.
  ORCA's `%maxcore` requests `ceil(maxcore × pal ÷ 0.75)` (maxcore is a promise ORCA
  overshoots per process — the 75% rule); Q-Chem's `mem_total` requests its declared
  peak (the scaffolded Q-Chem starter declares one: without it Q-Chem runs at its own
  2000 MB default, however much SLURM grants). A header whose own `--mem`/`--mem-per-cpu`
  already covers the requirement stands untouched; a short or absent one is extended to
  `--mem-per-cpu`, with the override logged. Inputs that declare nothing keep the
  header's policy.
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
  archiving and resubmitting the first one's in-flight work. The lock is a
  hard link, so the tree must sit on a filesystem that supports them: on one
  that does not (vfat/exfat, some FUSE and SMB shares) the run is refused with
  code `10` before anything runs. A lock left by a driver that died on the
  same host is reclaimed automatically; one left on another host must be
  deleted by hand once that run is known dead (the error says so). A
  `scancel`-ed (SIGTERM'd) driver exits in an orderly way (code 143),
  releasing the lock and terminating its local jobs — only a genuine SIGKILL
  can strand a lock, and the same-host dead-pid reclaim covers that case.
  `run_status`'s holder payload carries a three-valued `alive` field:
  `true`/`false` for a same-host holder, `null` for a holder on a foreign host
  no status read can probe.
- Engine plugin system: a `CalculationEngine` protocol plus a registry, with
  four documented base shapes for new engines (`engines/api.py`). Plugins and
  MLIP backends are **auto-discovered** — a new engine package or backend
  module is dropped in and registers itself, with no central import list to
  edit. Bundled engines: `orca`, `mlip`, `mlip-extopt`, `mlip-train`,
  `pyscf`, `pyscf-extopt`, `qchem`. A template-driven engine declares the starter
  `chemrefine scaffold` writes for it, and registration refuses an engine that
  declares part of a capability: a template engine without its starter, a
  preflight hook without its sentence.
- Subcommand CLI — `run`, `resume`, `rerun [step]`, `rerun-errors [step]`,
  `rebuild-cache [step]`, `rebuild-nms [step]` — with documented process
  exit codes per failure class.
- Per-step result cache: filter-only (`sample:`) edits refilter cached results
  without re-running calculations, and a step cache that is corrupt or cannot
  be read is treated as missing, where v1.3.1 stopped with a traceback; any
  other `_cache/` document that cannot be read or written stops the run as a
  cache error (exit code 7).
- Per-step failure policy `on_failure: stop | skip | best`. Failures are
  ledgered in `failed_jobs.json`, and `resume` / `rerun-errors` re-attempt only
  the still-failed structures of a `stop` step. Changing a step's `on_failure`
  and resuming takes effect even over a cached step: ledgered failures are
  re-attempted and the step is re-finalized under the new policy, and successes
  are never recomputed (`stop` ↔ `skip` edits are a free cache hit, since both
  store the same results). Under `best`, a backfilled structure carries no
  thermochemistry and is simply excluded from rankings on `gibbs`, `enthalpy`
  or `electronic_zero_point`; only a step where *nothing* has the requested
  energy is a config error. A backfill of the submitted input is that input's
  geometry and lineage alone, as a step-1 seed arrives, so it is cached and
  ledgered but never ranked on an energy from the step before.
- Automatic convergence retries: a structure that fails to converge is
  re-attempted, and the retry joins the step's own throttled queue the moment
  a slot frees, overlapping the rest of the batch. Results keep manifest
  order, so a retry never reorders the step's structures. `job_timeout_seconds`
  is a **stall deadline** — the longest a step may go with *nothing*
  finishing — whose clock restarts on every completion (and follows the
  array's *tasks* under `slurm_array: true`); a batch that keeps draining
  never trips it, however long the step takes.
- Two-round normal-mode sampling (`nms: true`) with target-aware displacement
  (`minimum` / `ts` / `random`); tuning the search parameters re-attempts only
  the unresolved parents. `random` draws displacement modes from the frequency
  table the engine's parse names — the vibrations and any imaginary modes;
  translations and rotations are never candidates, and a structure with no
  table yields no displacements, with a warning. Its children are computed in
  the second round like any target's, where v1.3.1's `random` mode handed the
  displaced geometries to the next step without computing them. A
  `ts_mode_index` that names no imaginary mode of the structure is rejected up
  front, so the mode the setting exists to preserve — the reaction coordinate —
  is never displaced. After sampling, `stepN/<id>/` describes a single
  calculation: the round-1 frequency job is archived in the same `attemptK/` as
  its displaced children, and the winning child's geometry, output, orbitals
  and Hessian are promoted to the structure's canonical path, so the files
  there all agree. The structure's cache record names the child it was resolved
  from (`resolved_from`), and its `attemptK/` keeps a `resolution.json` saying
  the same; v1.3.1 named the winner only in its log. An ORCA template requests the
  frequencies sampling needs with a `Freq`, `NumFreq` or `AnFreq` keyword, or with
  `NumFreq true` / `AnFreq true` in a `%freq` block. A step whose input requests none is
  refused when the run starts, and `chemrefine validate` reports it on the step's `nms`
  field, where v1.3.1 ran the step and skipped each structure for want of a frequency
  table.
- Shared ExtOpt HTTP server: ORCA optimises on gradients served by an MLIP or
  PySCF backend in the same SLURM job (kernel-assigned ports, where v1.3.1
  bound every server to the step's `bind` address, `127.0.0.1:8888` by
  default; health probe, clean teardown). The job starts ORCA once the server
  answers its health probe, where v1.3.1 started it after a fixed 10 s; it
  waits up to an hour for a server still loading its model (on a cold cache,
  downloading it) and gives up on one that answers nothing for two minutes.
  The server answers one call at a time from one loaded model, where v1.3.1
  ran four worker threads that could each load their own. `/calculate` serves
  only a request carrying the run's bearer token, written `0600` beside
  `server.url`, so another user on a shared node cannot drive the server;
  v1.3.1's served any process on the node. A single-environment ExtOpt step
  requires flask and waitress at preflight — the run fails before submission
  with an error naming both fixes (`pip install "chemrefine[server]"`, or
  `chemrefine backends install <extra>`) — and a server that cannot start logs
  one actionable line to its `--log-file`, the file the job's failure path
  tails. Both ExtOpt engines read their options strictly, the NMS knobs beside
  the server's own, so a misspelled key is refused before anything is
  submitted, where v1.3.1's `mlff` block ignored a key it did not read. A
  template whose `!` line does not name `ExtOpt` is refused when the run starts,
  where v1.3.1 ran ORCA's own method under the MLFF label while its server idled.
- PySCF engines: `pyscf` runs each structure through the step's own rendered
  Python script, and `pyscf-extopt` serves gradients to ORCA through the
  ExtOpt server. Both engines require the level of theory to be named —
  `basis`, and `xc` for `method: dft` — the direct engine refusing at
  preflight: every other engine makes the user say it (ORCA in the template's
  `!` line, Q-Chem in `$rem`), and a silent default would compute at a level
  nobody chose. `strict_scf` (default on) refuses to serve a gradient from an
  SCF that did not converge: PySCF returns the last iterate rather than
  raising, and ORCA's `.out` records only its own geometry convergence, so an
  unconverged result would otherwise rank unmarked against converged siblings;
  set `strict_scf: false` to accept such gradients knowingly. `save_tensors`
  is restricted-only — an open-shell (UHF/UKS) step is refused when the run
  starts and by `chemrefine validate`, naming the knob and the multiplicity. A
  `device: cuda` step requires the gpu4pyscf stack: the preflight derives the
  requirement from the step's options and fails by name, pointing at the
  `pyscf-gpu` extra, when the environment lacks it. Step scripts (`mlip` and
  `pyscf` alike) return positions and gradients as `(N, 3)` arrays — a flat
  `(3N,)` gradient is an ordinary `UNPARSEABLE` ledger entry naming the atom
  count — and a required output field must be present and non-null: a
  malformed output document is a ledgered per-structure parse failure, never a
  crash. A script reports its own convergence as `converged`, a boolean the
  `mlip` starter takes from its optimiser and the `pyscf` starter from its
  SCF, so an optimiser that runs out of steps, or an SCF that does not
  converge, is ledgered as a convergence failure and retried once from its
  best geometry; a `converged` that is not a boolean is that structure's parse
  failure, and one left unset reports nothing.
- Per-backend MLIP extras (`mlip-fairchem`, `mlip-mace`, `mlip-sevenn`,
  `mlip-orb`, `mlip-chgnet`) and a backend-agnostic calculator factory. Each
  backend installs into its own dedicated environment (their torch/e3nn trees
  conflict); `[mlip]` defaults to FAIRChem / UMA (`uma-s-1p2`, from upstream
  `fairchem-core >= 2.18` on PyPI).
- Managed backend environments: `chemrefine backends {install,list,path}`
  provisions one env per backend (built with the same tool that created the
  current env — conda / uv / venv) and steps resolve them **by name**, so
  conflicting MLIP stacks (e.g. MACE + UMA) run side by side in one pipeline.
  Every run validates its steps' backends before any job submits. That
  check covers the ChemRefine inside each env: an editable install is
  mirrored into it as an editable install of the same checkout, and an env
  holding another source or version is refused, naming the
  `chemrefine backends install <extra>` that reconciles it. Each env is
  built on **the Python its backend supports**, not the orchestrator's:
  every backend extra states which Pythons it installs on (`mlip-orb` 3.12
  only; `mlip-chgnet` up to 3.12; `mlip-mace` up to 3.13),
  `backends install` builds on the newest one it claims — stopping with the
  ways out when no such interpreter is available — and `--python` (a
  version, a command name, or a path) overrides the choice. An env built on
  a Python its backend excludes is refused rather than installed into,
  because pip succeeds there having installed nothing.
- **Every shipped MLIP backend fine-tunes** — v1 trained MACE only.
  `task_name: sevenn | chgnet | orb` join MACE and the FAIRChem heads as
  `mlip-train` selections, so the family `[mlip]` installs by default trains
  too; the `mlip-sevenn` floor is 0.11.1, the release with the unified
  `sevenn train` CLI. Libraries without a charge/spin channel warn instead of
  silently fitting an ion as neutral data, and running what you trained is
  the same `model_path:` line for all five. An ORB model with a stress
  output fine-tunes with that output switched off, and the job warns; the
  `mlip-orb` floor is orb-models 0.6.2, the first release with
  `disable_stress()`. FAIRChem's dataset is an ASE database per split —
  labels on a `SinglePointCalculator`, `metadata.npz` beside each, one
  directory per split because FAIRChem resolves a missing `metadata_path`
  against the database's *parent*. The worked example ships at
  `examples/tutorials/fairchem_finetune` — label with UMA, fine-tune through
  fairchem's own recipe collapsed into one commented config, run the
  produced checkpoint (the UMA weights themselves stay behind a gated
  Hugging Face repo).
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
- Seeding takes every frame as a structure — of a multi-frame `.xyz`, and of
  each file in a directory of `.xyz` files — where v1.3.1 seeded one structure
  per file; a CSV of SMILES embeds deterministically, where v1.3.1's 3D
  embedding was unseeded. Every seed is read when the run starts, and one it
  cannot use (missing, malformed, not text, a CSV without a `smiles` column, a
  file holding no structure) is a config error (exit code 2) naming it, where
  v1.3.1 stopped with a traceback on a missing seed and passed a malformed
  `.xyz` to ORCA unread.

### Changed

- **An explicit `operation:` is held to its engine's vocabulary.** A value outside the
  vocabulary an engine declares is refused when the run starts, before any step computes,
  and `chemrefine validate` reports it on the step's `operation` field; v1.3.1 checked one
  list for every engine, and only once the run reached the step. An operation is read in
  any case with `+` for `_` (`OPT+SP` is `opt_sp`): a rule of the config itself rather than
  of the v1 compatibility layer, so it stays past 3.0.
- **A batch reads its SLURM header once.** Every job script of a step's batch is built from
  the header as the batch read it when it started, where v1.3.1 re-read the file for each
  job; an edit reaches the step's next batch or the next run.
- **An ExtOpt step writes its own wrapper.** The script ORCA runs for each gradient is
  written into the step directory and launches the bridge with the interpreter that runs
  ChemRefine, where v1.3.1 pointed every step's `ProgExt` at the packaged `uma.sh`, which
  ran whichever `python` came first on the job's `PATH`.
- **The ExtOpt wrapper loads no analysis libraries.** ORCA spawns it once per optimizer
  step, and it imports none of pandas, RDKit, scikit-learn or `ase.io`, all of which
  v1.3.1's wrapper imported with the package on every step.
- **A failed ExtOpt gradient call ends the job with a one-line reason.** When the bridge
  ORCA runs for each gradient fails — no reply within `gradient_timeout_seconds`, a server
  it cannot reach, a refused request, a connection reset, a geometry it cannot read — it
  prints the reason on one line and exits with its code, which ORCA stops on and the
  step's ledger quotes, where v1.3.1's client printed a traceback with every failure.
- **A threaded job is one SLURM task with N CPUs.** The `pyscf` and `mlip` script engines
  and `mlip-train` request `--ntasks=1 --cpus-per-task=N` for their single OpenMP/torch
  process, so the allocation is one node by construction; v1.3.1's training script added
  no task shape to its header. An ExtOpt step keeps the ranks' spelling,
  `--ntasks=N --cpus-per-task=1`, as v1.3.1 did, because ORCA runs its `%pal` MPI
  processes in `ProgExt` mode; its script also pins `#SBATCH --nodes=1`, which v1.3.1
  left to the header, since the gradient server threads that same count in one process.
  An engine declares this through `single_node`.
- **The quickstart runs on a base install and ORCA.** `examples/first_run/` is the
  README's two-step pipeline — an xTB screen (ORCA's bundled GFN2-xTB) and a DFT
  refinement of three ethylene-glycol conformers, under a minute on a laptop, nothing
  to install beyond ORCA — and the README, the docs index and *Your first run* show
  those files verbatim (a test holds them together). v1.3.1's quick start listed the
  files to write, and v1.3.1 ran under SLURM only. Its `Examples/` is `examples/`: the
  annotated `input.yaml` is `examples/schema_tour/`, and the tutorials sit under
  `examples/tutorials/` with snake_case names.
- **The cache identity is per-structure, and `resume` is incremental.** A step's key
  is layered the way its work is: a **row key** per structure (engine, template bytes,
  effective charge/multiplicity, the options as the engine's declared model reads
  them, option-file digests, the parent's content), a **resolution key** for what NMS
  reads (criterion and search), and a step fingerprint composed from the ordered rows.
  The manifest records that identity per row, and `resume` adopts every row whose
  stored key matches — re-parsed from disk, never resubmitted — computing only the
  rest. What that means in practice: turning `nms: true` on over a finished frequency
  step submits only the imaginary parents' displacement children (round 1 and the
  clean minima are adopted, byte-identical); follow-up steps recompute only rows whose
  parent actually changed; a search retune re-reads round 1; a criterion change
  re-resolves and nothing more; an interrupted NMS step adopts its finished round 1; a
  retrained model re-runs the steps whose options name its file; an edited file an
  ORCA template names by quoted reference (a `%DOCKER GUEST` geometry, a
  `%pointcharges` file) re-runs the steps that read it; a typo'd option key nothing
  declares invalidates nothing (`validate` already warns about it). Row keys hash the
  **effective** charge and multiplicity each job renders — inherited workflow-level
  values included — so editing a workflow-level `charge:`/`multiplicity:` invalidates
  every step that inherits it. v1.3.1 stored one sha1 per step and never compared it:
  `--skip` restored whatever cache a step held.
- **An output tree has a new layout.** Each structure has a directory of its own
  (`step1/0/step1_0.out` where v1.3.1 wrote `step1/step1_structure_0.out`), and each step
  a `_cache/manifest.json` with paths relative to the step directory. A reason in
  `failed_jobs.json` names an output the same way, as `<structure>/<file>` under the step
  directory, where v1.3.1 quoted a missing output by its absolute path. 2.0 reads neither
  v1.3.1's `stepN_manifest.json` nor its pickle, and
  [adopting a v1 output tree](https://sterling-group.github.io/ChemRefine/get-started/upgrading-from-v1/#adopting-a-v1-output-tree)
  brings a finished v1.3.1 run across for `rebuild-cache` to re-parse.
- **`mlip-train` is rebuilt.** It runs through the same scheduler and throttle as every
  other engine — the `max_cores` / `max_gpus` budgets, the runlog, the scratch handling,
  local dispatch and `job_timeout_seconds` — where v1.3.1 submitted one SLURM script that
  ran a bare `mace_run_train` in whatever environment its header activated, and polled
  until it ended. Its backend is checked by the preflight like any other step's, and the
  trainer runs under that backend's own interpreter. Which library trains is `task_name`,
  resolved through a registry keyed exactly like the calculator's: one word names the
  library whether a step trains a model or runs one. Adding a trainable backend is one
  dropped-in module under `engines/mlip/backends/`, beside the calculators for the same
  library — the environment it needs is then declared once for both.

  The step's YAML changes: `task_name` and `device` are required (neither has a
  default worth guessing); `model_name` is optional — the foundation model a fine-tune
  starts from, and unset means the trainer's own declared start (the docs' backend
  table lists it: a released model, or scratch), recorded as `started_from`;
  `valid_fraction` sizes the validation set and `test_fraction` a separate held-out
  test set, where v1.3.1's `valid_fraction` sized the test set. The template is the
  trainer's own config `stepN.yaml`, **rendered** through `$PLACEHOLDERS`, where
  v1.3.1 read a MACE config from `stepN.inp` and patched dataset and output paths into
  it. A v1.3.1 training step gets `task_name: mace_off` and its device written out,
  and needs a `stepN.yaml` that references its trainer's placeholders. The step's own
  facts — `device`, `seed`, and the foundation weights —
  are authoritative over the template: for the libraries trained through a Python API
  (CHGNet, ORB) chemrefine delivers them to the training process directly and refuses
  by name a template value that disagrees; for the libraries that read their own
  config file (MACE, SevenNet, FAIRChem) the template must reference the placeholders
  that carry them (`$DEVICE`, `$SEED` where the library reads one, and, unless the
  step trains from scratch, its start: `$FOUNDATION_MODEL` for MACE and SevenNet, and
  for FAIRChem `$STARTING_CHECKPOINT`, or `$FOUNDATION_MODEL` for a release), so a
  `device: cuda` step cannot silently train on CPU with the GPU booked. v1.3.1's
  `operation: MLFF_TRAIN` (or `MLIP_TRAIN`) translates to `engine: mlip-train` with a
  deprecation warning.

  A training step's success test is the trained model's existence: `resume`
  retrains after a failed training, and a training that completed before the driver
  died is adopted by `rebuild-cache` without recomputing it. The step's fingerprint
  includes its training-config template, so retuning epochs or learning rate and
  resuming re-runs the training and replaces the model on disk. v1.3.1 counted a
  training step done once its job ended, so `--skip` passed over a failed training
  and an edited template alike. The preflight refuses a training step whose backend
  can run but not train — before any step submits — and names which backends can.

- **`task_name` is now the only key that selects an MLIP backend.** `model_path` says
  where the named library's weights come from and selects nothing; every library's builder
  loads a local checkpoint itself. Under `mace_off` and `mace_omol` a `model_name` is one
  of the family's releases — MACE-OFF's `small`, `medium` or `large`, MACE-OMOL's
  `extra_large` — or is refused when the run starts and by `chemrefine validate`: a
  checkpoint goes in `model_path`. v1.3.1 read no `model_path` from the config, so a
  checkpoint ran only as a MACE `model_name`: one that named no `mace_off` release loaded
  as a file, and `mace_omol` ignored the name.

  **Migration:** v1.3.1 named a checkpoint as `model_name` under `mace_off` or `mace_mp`,
  and the v1 layer moves such a value to `model_path`. A `model_path` with no `task_name`
  resolves the `task_name` default (`omol`, FAIRChem), so name the library that produced
  the checkpoint — the same word the `mlip-train` step trained with:

  ```yaml
  options:
    task_name: mace_off          # add this line
    model_path: ./outputs/step4/train/train_stagetwo.model
  ```

  `model_name` names weights *within* the selected library and is optional — unset
  means the library's own default (FAIRChem's builder keeps `uma-s-1p2`; SevenNet loads
  its own default release). In v1 the `model_name` prefix (`sevenn…`, `orb…`) doubled
  as the backend selector; it no longer selects anything, and the v1 layer translates
  v1.3.1's SevenNet spelling to `task_name: sevenn` with the release as `model_name`.

- **Each recovery command covers the steps around its target.** v1.3.1's flags each acted
  on one step and exited: `--rerun_errors` left the step's cache to `--rebuild_cache`, and
  every flag left the rest of the run to `--skip`.
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
    maps here; v1.3.1's flag re-ran the sampling, resubmitting the displaced jobs
    under `calc_type: rm_imag`, and defaulted to step 2.
  - `resume` and `rerun-errors` decide what is left to do by what actually
    parsed: a structure whose output is present but truncated or otherwise
    unusable is re-run rather than reported failed.
- **v1.3.1's config keys are renamed.** `sample:` replaces `sample_type:` (its methods and
  parameters renamed too), `nms:` + `options:` replace `normal_mode_sampling*`,
  `executables:` replaces `orca_executable`, `input:` replaces `initial_xyz`, and a step's
  `mlff:`, `mlip:` or `trainer:` block becomes its `options:`. The loader rewrites each old
  spelling with a deprecation warning. It refuses `calculation_type`, a v1.0 key v1.3.1
  did not read (it took `engine:` + `operation:`), with a pointer to the migration guide.
  A v1.3.1 `energy_window` keeps its unit: `energy` is read as hartree unless
  `unit: kcal/mol` says otherwise, as v1.3.1 read it, and is converted into
  `window_kcalmol` with a warning that names both values. A rewritten legacy config is
  validated like a current one: a number where `sample_type.parameters` or
  `normal_mode_sampling_parameters` holds a mapping is refused at load, naming the key,
  where v1.3.1 failed with a traceback once that step's jobs had run. A training step
  needs editing by hand (see `mlip-train` above).
- v1.3.1's `engine: MLFF` — ORCA over a machine-learned gradient server — is
  `mlip-extopt`, and its `mlff:` block becomes the step's `options:`. The defaults
  v1.3.1 read a missing key as (`task_name: mace_off`, `model_name: medium` under
  the MACE families whose loader took it, `device: cuda`) are written out, each
  reported, so a translated step runs the model v1.3.1 ran; a block on a step v1.3.1
  never used it on is dropped, reported. v1.3.1's `engine: MLIP` with an `mlip:` block,
  whose server v1.3.1 never started, is translated the same way; a bare `mlip` names
  v2's direct engine. The step-level `model_name` / `task_name` v1.3.1's README put on an
  MLFF step, which v1.3.1 ignored, move into `options:`, so the step runs the model it
  names. A v1.3.1 training step keeps its partition: `trainer.valid_fraction`,
  v1.3.1's *test* split, is `test_fraction`, `valid_fraction: 0` leaves validation to the
  template as before, and `task_name: mace_off` and `device: cuda` are written out where
  the step names none.
- `options.device` defaults to `cpu` on all four MLIP/PySCF engines (`mlip`,
  `mlip-extopt`, `pyscf`, `pyscf-extopt`). It is read by both the rendered
  script and the scheduler, so one setting drives the header and the run;
  request a GPU explicitly with `device: cuda`.
- **Each engine declares what comes back from scratch.** A job's scratch directory is
  removed when it ends, on the per-job and array paths alike, once the results its engine
  declares are copied home — files by pattern, whole directories through `output_dirs`,
  anything else through its run block's own teardown. v1.3.1's `save_scratch`, a
  parameter of its `OrcaJobSubmitter` class that no config key or flag reached, has no
  2.0 equivalent.
- **ChemRefine installs from PyPI.** `pip install chemrefine` replaces v1.3.1's
  `git+https` install. Supported Python is 3.11 through 3.14 (v1.3.1 declared 3.9–3.12).
  The sdist carries the test suite, the shipped examples and the documentation sources
  (`docs/`, `mkdocs.yml`), so the suite runs and the site builds from the unpacked
  tarball.

### Fixed

- **A linear molecule keeps its lowest vibration.** v1.3.1's `random` sampling skipped the
  first six of ORCA's modes for every molecule, but a linear one has five translations and
  rotations, which ORCA states in its output ("The molecule is recognized as being
  linear"): a linear molecule never drew its lowest vibration, and a diatomic, left with
  no mode to draw, ended the run. 2.0 takes the count from ORCA's statement, and `random`
  draws from the frequency table the parse names.
- **Normal-mode sampling no longer ends the run on a structure it cannot resolve.**
  v1.3.1's `rm_imag` raised on a structure with no imaginary mode, and aborted the run
  when neither displaced child had exactly one ("Imaginary frequency removal failed.
  Aborting."). 2.0 passes a structure already at its target through and ledgers one it
  cannot resolve, under the step's `on_failure` policy.
- **An ORB step builds its calculator.** v1.3.1 never built one: it sent a `model_name`
  starting with `orb` to SevenNet's calculator, which it asked for a model named after
  the step's `task_name`, and refused `task_name: orb` with any other name as an
  unsupported task, so an ORB step failed at its first gradient. 2.0 builds orb-models'
  own calculator, finding the `model_name` among orb-models' loaders by function name or
  registry key, with hyphens or underscores, and refusing by name one it does not know or
  has deprecated. The model loads uncompiled: ORB compiles by default, a compile is paid
  once per process, and a ChemRefine process serves one job — on CPU the compile cost
  70 s of an 80 s single point of water.
- **An ORCA structure is read from one evaluation, the last with an energy.** v1.3.1 took
  the last coordinate block, the last `FINAL SINGLE POINT ENERGY` and the last gradient
  block of an output each on its own and checked neither termination nor convergence, so
  an output that did not end on a complete evaluation passed on values from different
  geometries: a job killed mid-optimisation — a walltime, a failed external program —
  carried one cycle's energy on the next cycle's geometry, and an ExtOpt optimisation that
  stopped at its cycle limit, whose later cycles print their energy only as the
  optimiser's `Current Energy`, carried its first cycle's energy on its last geometry. 2.0
  ledgers such a run as failed, and reads every value from the last evaluation ORCA
  printed an energy for, which is the structure `on_failure: best` carries on. A
  coordinate or gradient block cut off by a kill is read in linear time, where v1.3.1's
  parse of a few dozen atoms never finished.
- **An ORCA `Opt` output carries no forces.** v1.3.1 stored an optimisation's last
  gradient block beside its final geometry, but ORCA prints that gradient inside the last
  cycle, at the geometry before the converged one, so a training set labelled from an
  `Opt` step paired each geometry with forces computed at another point. 2.0 reads forces
  only from a gradient printed after the last geometry (an `EnGrad` single point, a
  `Freq`); label a training set with a single point on the optimised geometries, as the
  tutorial does.
- **A periodic-only MLIP runs a molecule in a vacuum box.** v1.3.1 handed CHGNet every
  molecule without a cell, which CHGNet cannot evaluate, so an ExtOpt step on CHGNet
  failed at its first gradient call with an error naming neither the structure nor the
  library. 2.0 evaluates and fits a structure with no cell in a cubic box with 10 Å of
  vacuum on every side, with a warning, and the structure carried forward keeps its own
  coordinates.
- **An output that cannot be read as the molecule it claims is that structure's parse
  failure.** v1.3.1 skipped a PES scan coordinate row it could not read (an overflowed
  `*****` field), so that scan point shipped one atom short with the whole molecule's
  energy, and it carried a dummy centre (ORCA prints `DA` as `XX`) into the next step as
  an atom. 2.0 holds every scan point to the first point's atom count, refuses a row
  whose coordinate overflowed or is not finite, and refuses a symbol that names no
  element, each as that structure's unparseable output.
- **An unknown step target is a config error.** Given a step with no directory,
  v1.3.1's `--rebuild_cache`, `--rebuild_nms` and `--rerun_errors` logged an error and
  exited 0. 2.0 refuses a step number or name the config does not define with exit
  code 2, from every entry point and under `--dry-run` too, before anything touches the
  tree.
- **A negative `seed` is refused before the run starts.** v1.3.1 handed a training step's `seed`
  to its train/test split, which refused a negative one with a traceback once every
  labelling step upstream had run. 2.0 holds the training and NMS `seed` to `>= 0`, so
  `chemrefine validate` and the run's preflight name the field before anything is
  submitted.
- **A float knob refuses `.inf` and `.nan`.** v1.3.1 took any number as the NMS
  displacement (`displacement_vector`), so `.nan`, `.inf` or `0` sent out a full batch of
  displaced children whose coordinates were not finite or were their parent's unchanged.
  2.0 holds `displacement_value` positive and finite, and `window_kcalmol`,
  `temperature_k`, `job_timeout_seconds` and `gradient_timeout_seconds` finite, naming
  the field before anything is submitted. To wait without a bound, write `null` for `job_timeout_seconds`;
  `gradient_timeout_seconds` always has one, so raise it instead.
- **`model_name` reaches CHGNet.** In v1 the CHGNet branch never consumed it —
  `CHGNet.load()` ran bare, so a pinned `model_name: "0.2.0"` silently served the
  default release instead. A release name now routes through
  `CHGNet.load(model_name=...)`.
- **An ORCA step under a path containing whitespace is refused with its reason.** v1.3.1
  wrote the step directory unquoted into each job script and into ORCA's `* xyzfile`
  directive — a whitespace-delimited field ORCA does not treat as quotable — so every job
  of a step under an `output_dir` containing whitespace failed, naming a truncated path.
  2.0 refuses an `orca`, `mlip-extopt` or `pyscf-extopt` step there before it writes an
  input, `chemrefine validate` warns about the step beforehand, and both check the
  **resolved** path, so a directory reached through a symlinked parent with a space in it
  is caught too. `mlip`, `pyscf` and `qchem` steps run from such a tree, and so does a
  step whose `template_dir` or `scratch_dir` contains whitespace.
- **A geometry, energy or gradient that is not a finite number fails its structure.**
  v1.3.1 kept the coordinates it read from an ORCA output as text and checked none of
  them: an overflowed `*****` field put its atom at the origin of the next step's
  geometry, with only a log warning, and a `nan` was written through as is. Its ExtOpt
  wrapper likewise wrote a model's `nan` energy or gradient into ORCA's `.engrad`. 2.0
  refuses an overflowed or non-finite coordinate, and a non-finite energy or gradient,
  wherever it reads one: the structure is ledgered as a parse failure, or, for a gradient
  the ExtOpt server computed, its job fails as a backend failure. A seed `.xyz` holding
  one is refused by name, and the cache refuses to store one that got through.
- **Caching works where crypto policy restricts SHA-1.** v1.3.1 computed each step's
  fingerprint with a plain `hashlib.sha1` call, which a host whose crypto policy restricts
  SHA-1 to non-security use refuses, so a run there failed at its first step. 2.0 hashes
  file and structure contents only to decide what to re-run, and declares every such
  hash `usedforsecurity=False`.
- **`_cache/` files are readable on a shared tree.** v1.3.1 wrote each step's cache —
  `_cache/stepN.pkl` and its `stepN.json` sidecar — through `mkstemp`, whose owner-only
  `0600` survived the rename, so a colleague sharing the tree could read its outputs but
  not its cache. 2.0 writes every `_cache/` document under the umask, like any other file
  it writes; the ExtOpt server's token sidecar stays `0600` on purpose.
- **The training dataset is readable by the trainer.** v1 wrote `DFT_energy` /
  `DFT_Forces` while MACE's defaults are `REF_energy` / `REF_forces`, and the shipped
  template declared a third pair — MACE refuses a file in which it finds none of its
  keys, so every training job died at data load. The dataset now speaks MACE's own
  default keys, so a template needs no `energy_key`/`forces_key` line to be correct.
- **Charge and multiplicity reach the training set.** MACE reads `total_charge` /
  `total_spin` and silently defaults them to a neutral singlet, so a v1 fine-tune on an
  ion or an open-shell system was fitted against the wrong species with nothing said in
  any log.
- **A relative `model_path` resolves against the config file's directory**, like every
  other path a config names. The checkpoint v1.3.1 read from a `model_name` resolved
  against the job's scratch directory instead, so the MLIPTraining tutorial's
  `../step3/checkpoints_dir/…` is moved to `model_path` as written and must be rewritten
  relative to the config file.
- **A FAIRChem checkpoint runs from a path.** A FAIRChem model can be given as a
  checkpoint file path as well as a registry name — which is how a fine-tuned FAIRChem
  checkpoint is run. v1 resolved registry names only.
- **NMS round 2 runs in the step's own queue.** v1.3.1 started no displaced child until
  every round-1 job of the step had finished, so the slots early finishers freed sat idle
  until the last one landed. 2.0 submits a structure's `attemptK/` and its ± children the
  moment its own round-1 job finishes, alongside whatever is still running. Each winner
  is picked once the queue drains, so survivors come back in manifest order whichever
  child finishes first.
- **An unreadable gradient row fails its structure.** v1.3.1 skipped any
  `CARTESIAN GRADIENT` row its pattern could not read — a `*****` overflow, a `nan` — and
  kept a force array shorter than its geometry. 2.0 counts the rows against the atoms and
  reports a short or non-finite gradient as a parse error for that structure.
- **Frequencies are read from the last evaluation of an ORCA output.** A TS search
  recomputes the Hessian as it goes and prints one frequency table per recompute; v1.3.1
  collected the imaginary modes of every table, so a converged transition state came back
  with the imaginary modes it had on the way there, and normal-mode sampling displaced
  along modes that no longer existed. 2.0 reads a structure's frequencies from its last
  evaluation alone, like every other value.
- **The step cache is plain data, not a pickle.** v1.3.1 stored each step's cache as
  `_cache/stepN.pkl` and loaded it with `pickle`, which can execute code from the file.
  2.0 writes `_cache/step.json` for the metadata and `_cache/arrays.npz` for coordinates
  and forces, and loading either executes nothing. `step.json` is written without
  indentation — read it with `jq` or `json.load` — and each structure also gets an
  indented `.result.json` beside its output files. `step.json` records the digest of the
  `arrays.npz` saved with it, so a pair from different saves, left by a walltime kill or
  Ctrl-C mid-save, is refused on read-back and the step rebuilds.
- **ORCA's restart, orbital and property files come back from scratch.** v1.3.1 copied
  only `.out` and `.xyz` files out of the scratch directory before deleting it, so the
  `.gbw`, the `.hess`, the property files and the `.opt` restart file went with it. 2.0
  copies those back too; `.opt` is what lets a stalled optimisation resume where it
  stopped rather than start over.
- **A cluster header keeps its `--ntasks-per-node` and `--ntasks-per-core`.** v1.3.1
  dropped every `#SBATCH` line of the header containing `--ntasks` or `--cpus-per-task`,
  so those directives — and anything else on the same line — never reached the job. 2.0
  strips only the options it sets itself (`--ntasks`, `--cpus-per-task`, `--job-name`,
  `--output`, `--error`) and keeps the rest of each line.
- **An ORCA job starts the ranks it was allocated.** v1.3.1 read a template's rank count
  from `nprocs` alone and capped only the job's `--ntasks` at `max_cores`, leaving the
  input's count as written: a template asking for more ranks than `max_cores` started
  more MPI processes than the job owned, and a `! PALn` template was allocated one task.
  2.0 reads `nprocs` and `! PALn` alike and rewrites a count above `max_cores` down to it
  in the generated input.
- **An output that cannot be read fails its structure, not the run.** v1.3.1 recorded an
  unparseable output in `failed_jobs.json` and then stopped the whole run with a
  traceback before caching any of the step; one malformed frame failed an entire GOAT or
  docking ensemble that way. 2.0 ledgers the structure as a parse failure, caches the
  step's other results and lets the step's `on_failure` policy decide what follows; its
  GOAT, docking and solvation readers skip a frame they cannot read and fail the
  structure only when no frame is left.
- **An ORCA executable whose path contains a space runs.** v1.3.1 wrote the ORCA
  executable and the step's output directory into its job scripts unquoted, so a space
  in the executable's path broke every job of every ORCA and MLFF step. 2.0 quotes both,
  in the plain ORCA and the ExtOpt run blocks alike.
- **Random-mode NMS derives the same displacements every time.** v1.3.1 picked
  random-mode displacements with Python's unseeded global generator, so `--rebuild_nms`
  derived different displaced structures than the run had computed. 2.0 seeds each
  structure's draw from the step's `seed` and the structure's ID, so a run and any
  `rebuild-cache` or `rebuild-nms` of it derive the same children, whichever structures
  the rebuild visits.
- **A step's template, SLURM header and job layout are checked before any step runs.**
  v1.3.1 looked for a step's `stepN.inp` and SLURM header only on reaching that step,
  after every earlier step had run, and reported a missing one as a `FileNotFoundError`
  traceback — or, for a later step's template, a bare exit 1. Before the first step runs,
  2.0 reads the template and SLURM header of every step that may submit, as its jobs
  will, and lays those jobs out against the core and GPU budget: a missing or unreadable
  file, a template its engine's rules reject, or a layout the budget cannot hold is a
  config error, exit 2.
- **A truncated ExtOpt input fails its job by name.** v1.3.1's ExtOpt wrapper read the
  five lines of ORCA's `.extinp.tmp` by index, so a file cut short — ORCA killed
  mid-write, a full disk — ended in an `IndexError` traceback. 2.0's wrapper refuses it
  as a job failure naming the file and how many lines it holds, and the step's ledgered
  reason quotes that line.

## [1.3.1] and earlier

The pre-rewrite line; see the
[GitHub releases](https://github.com/sterling-group/ChemRefine/releases)
for its history.
