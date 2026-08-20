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

- **Config tooling** (`chemrefine validate | scaffold | schema | engines`):
  `validate` reports every finding at once — pydantic errors with field locations,
  unknown engines, bad values for declared option knobs, invalid NMS knobs — plus
  warnings for the silent no-ops (undeclared option keys, `nms: true` on an engine
  that cannot NMS, step templates and SLURM headers that do not exist yet, resolved
  the way dispatch resolves them); warnings never block, an unrunnable config exits 2.
  `scaffold` writes commented starter templates into every gap the config expects.
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
  bonds a normal mode moves — reaction-coordinate validation). The packaged operating
  guide ships as the `chemrefine://guide` resource, with its vocabulary pinned to the
  code by tests.
- **Workflow-builder GUI** (`chemrefine gui`, extra `chemrefine[gui]`): a local
  two-pane web app (127.0.0.1 behind a per-session token) — click-through forms
  rendered from the live schema on the left, the `input.yaml` on the right, emitted
  and parsed server-side only. Validate anchors findings to fields; Save…/scaffold/
  inline template editing; a run dashboard (start/resume/rerun-errors behind
  confirmations, polling status, paginated `steps.csv` results); and, with the
  `[agent]` extra, an agent chat panel whose mutating tool calls arrive as allow/deny
  cards. The builder also publishes on the docs site as the **Playground** (top
  navigation) in a build-and-copy static mode.
- **Embedded agent** (`chemrefine agent`, extra `chemrefine[agent]`): a terminal chat
  over the same tool surface, harnessed by PydanticAI — multi-provider (presets for
  local Ollama/vLLM, any OpenAI-compatible endpoint via `--base-url`, or native
  `provider:model` strings; configuration via `CHEMREFINE_LLM_*`). Every mutating tool
  sits behind a y/N confirmation whose refusal is reported to the model as an answer.
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
  copies it back to the structure dir. NMS-capable: `jobtype ts`/`freq` gate and target
  the sampling, and the frequency parse maps Q-Chem's 3N−6 vibrational modes onto the
  trivial-modes-first tensor the coordinator expects. The output reader is deliberately
  minimal (final energy + last geometry) pending the full parser set.
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
  says so).
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
- Each managed env is built on **the Python its backend supports**, not on the
  orchestrator's. Every backend extra now states which Pythons it installs on
  (`mlip-orb` 3.12 only — orb-models pins `dm-tree==0.1.8`, whose newest wheels
  are cp312; `mlip-chgnet` up to 3.12; `mlip-mace` up to 3.13), and
  `backends install` builds the env on the newest one it claims: conda and uv
  produce that interpreter themselves, a plain venv takes `python3.12` from
  `PATH` or a `uv` binary if there is one, and where there is neither it stops
  with the ways out rather than starting a build that cannot finish. Provisioning
  orb on a 3.13 machine used to end in a thousand lines of C++ from a vendored
  abseil. `--python` (a version, a command name, or a path) overrides the choice;
  an env built on a Python its backend excludes is refused rather than installed
  into, because pip succeeds there having installed nothing.
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
  - `rebuild-cache N` rebuilds step N, then walks the steps after it read-only:
    each is served from its cache — a load, never a re-parse, and never a
    submission — and the first cache the current configuration cannot serve ends
    the run quietly, its message naming the cheapest repair (`rebuild-cache` for
    outputs that still match this configuration, `resume` when upstream results
    changed). The walk exists because `steps.csv` is rewritten from step 1 on
    every run: ending at the target silently dropped the later steps' rows even
    when their caches were untouched and valid. The report now covers exactly
    what the current configuration can vouch for.
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
  The tested matrix covers Python 3.11 through 3.14.
- The sdist carries the test suite **and** the shipped examples, and CI proves the
  combination: the smoke-test job unpacks the built tarball and runs the suite
  inside it, so "a distro packager can run the tests from the sdist" is a gated
  promise rather than a comment. The repo-only guards (docs drift, the mutation
  gate's self-tests) skip themselves there with named reasons; `docs/` stays out
  of the tarball on size.
- `pyscf` gains a `strict_scf` option (default on). Two new CI jobs cover what
  the matrix could not reach: a mutation gate that breaks each critical predicate
  and requires a red test, and a run of the suite with a managed backend
  environment provisioned.
- The live release tier (`scripts/release-check.sh`, tier-3) covers more and runs in
  under half the time. New coverage: a UMA single point (`fairchem_sp` — the default
  backend, previously never run by the gate), numerical frequencies computed through the
  ExtOpt gradient server for both backends (a bare `FREQ` is silently dropped in ExtOpt
  mode — the cases spell `NumFreq`), and `save_tensors` delivery through the
  directory copy-back. The DFT-heavy cases moved to XTB2 where the parsing contract is
  method-agnostic; `nms_minimum` stays PBE/def2-SVP as the one real-DFT parse.

### Fixed


- `chemrefine.slurm.build_script` no longer takes `save_scratch`. It was a v1 concept
  carried into the rewrite's signature and never wired — no config knob and no caller in
  any commit of this line — and `build_array_script` never had it, so wiring it later would
  have silently skipped every `slurm_array` step. Keeping artifacts is engine-owned and
  already works on both paths (`output_dirs`, or `RunBlock.cleanup`). Noted because
  `build_script` is public and rendered on the API page; the runlog's `scratch_kept` field
  is unchanged.
- **Two documentation claims that had stopped being true.** The security page's list of
  hardened boundaries described only the ExtOpt gradient server, so a reader auditing what
  ChemRefine exposes on a shared node never learned that `chemrefine gui` starts a second
  socket — the more consequential one, since it can write files and launch runs. It now has
  its own subsection (loopback-only bind, kernel-assigned port, the per-session token and
  how it is compared, and why `/` is deliberately ungated). Separately, `job_log`'s module
  docstring — published as an API page — still spelled the runlog basename the **v1.3.1**
  way (`step{N}_structure_{ID}`, a convention 2.0 dropped from every artifact name) and gave
  a grep pattern that matched nothing; it now names
  the real per-structure path, including where an attempt's logs land, and a test derives
  that path the way production does and requires the documented glob to match it.
- **A directory name can no longer smuggle a shell metacharacter into a job script.**
  `template_dir` / `output_dir` / `scratch_dir` are refused at load time when they contain
  `"`, `$`, a backtick, a backslash or a newline, because the generated script exports them
  into bash — but the check ran on the paths *as written*, and a relative path is anchored
  to the config file's own directory afterwards, through `model_copy`, which runs no
  validators. So a config containing nothing but `output_dir: ./outputs`, sitting in a
  directory named `$(...)`, produced `export OUTPUT_DIR="…/$(...)/outputs"` — and bash
  performs command substitution inside double quotes, so the directory name ran when the job
  did. The rule is now re-asked on the resolved paths, by `load_config` (raises) and by the
  validation report (a blocking issue, so `save_config` will not write such a config
  either). Present since the paths became resolvable; found while fixing the whitespace
  rule, which had the same shape.
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
  `template_dir` and `scratch_dir`. Enforcing it at config load instead — the first shape of
  this fix — made validity depend on where a project sits on disk: an mlip-only workflow
  under `~/My Drive` was refused although it runs perfectly, and this repository's own
  example suite went red from any checkout path containing a space.
- **A diverged calculation's geometry is refused instead of stored.** A non-finite
  energy has been a parse failure for a while; the *coordinates* were not, and that
  had two costs. Inline in a `.result.json` a `NaN` met the JSON writer's
  `allow_nan=False` and raised a bare `ValueError` no handler caught, so one bad
  structure ended the whole run in a traceback and the step's other successes were
  never cached — and a resume read the same output and died the same way. Worse, on
  the paths that write no record the coordinates go to the `arrays.npz` sidecar
  instead, which has no such check: there a `NaN` was simply kept. It round-trips the
  cache intact and the fingerprint over it is perfectly stable, so every later step
  was computed from coordinates that are not numbers with nothing anywhere saying so.
  Non-finite geometries are now refused where they enter — the ORCA, Q-Chem and script
  parse boundaries raise the same error a `*****` overflow token already raises, and a
  seed `.xyz` is refused by name — with the cache as a backstop that refuses to store
  what did get through. The ensemble readers keep skipping rather than failing a whole
  step for one bad conformer; `nan` now simply follows the rule `*****` always had.
- **A non-ASCII token is a 401 from both token gates, not a 500.** The GUI's
  `X-ChemRefine-Token` check and the gradient server's `Authorization` check had the same
  flaw and were fixed a day apart: headers arrive latin-1-decoded and `compare_digest`
  refuses a `str` holding a non-ASCII character, so any byte above 0x7F raised out of the
  auth gate. Both are reachable by any user on the node, which makes those gates exactly
  the place that has to answer plainly whatever it is handed.
- **A step naming a model file works where crypto policy restricts SHA-1.** ChemRefine
  hashes file contents to decide what to re-run, and says so — every constructor-form
  hash is marked `usedforsecurity=False`. The two *streamed* digests could not say it,
  because `hashlib.file_digest` handed an algorithm name builds the hash with the flag
  defaulted; on a host that permits SHA-1 only for fingerprinting they raised, and the
  one that runs on every step's cache key would have ended the run in a traceback.
  Cache keys are unchanged — the flag is a policy hint, not an input to the hash.
- **A PySCF step that asks for a GPU no longer runs quietly on the CPU.** PySCF is one
  library with two stacks, and the provisioner probed only for `pyscf` — so on an
  environment carrying just the CPU stack a `device: cuda` step started anyway, fell back
  to the CPU inside the job, and recorded the reason in the ExtOpt *server* log: a run that
  succeeded on the wrong hardware and said so nowhere the user was looking. The requirement
  is now derived from the step's own options and probes for `gpu4pyscf`, so such a step
  fails at preflight, by name, pointing at the `pyscf-gpu` extra.
- **A missing optional extra names itself instead of raising `ImportError` from inside.**
  The GUI and the embedded agent guard their optional SDKs, but the imports sat where the
  guard could not see them, so `chemrefine gui` without `chemrefine[gui]` — and
  `chemrefine agent` without `chemrefine[agent]` — surfaced the raw import failure rather
  than the line saying which extra to install.
- **Reading the process umask no longer races other threads.** `os.umask` is the only
  POSIX way to read it and it reads by *setting* it, which is process-global: in that
  window anything another thread created was made with no mask at all, and the GUI serves
  four waitress threads that both write files and `mkdir`. The value now comes from
  `/proc/self/status`, which reading does not disturb.
- **`chemrefine --help` stops swallowing the extra it names.** Rich reads square brackets
  as style markup, so the `[agent]` / `[gui]` / `[mcp]` in each command's install hint was
  consumed as a tag: the one line telling a user which extra to install rendered as a bare
  `chemrefine`. The brackets are escaped now.
- **`chemrefine agent --check` reports a misconfigured endpoint instead of a traceback.**
  A base URL pointing at a web app or a proxy login page answers `200` with HTML, and some
  gateways answer a bare list; every one of those raised out of the preflight, which is the
  command that exists to diagnose exactly that mistake. They are findings now.
- **Two agent-started runs in the same second no longer share a log file.** The run log was
  named to one-second resolution, so two runs started close together opened the same path
  and the second truncated the first while it was still being written — losing the log
  `run_status` hands back when someone asks what went wrong.
- **A failure payload advertising the whole taxonomy no longer leaves a class out.** It was
  built from `ChemRefineError.__subclasses__()`, which is direct subclasses only, so the
  indirect `OutputTerminationError` was missing from every payload that claimed to list
  them all.
- **The declared `pydantic` floor is one the package can actually resolve.** `[mcp]` and
  `[agent]` both require `pydantic>=2.12`, so the old `2.5` floor was a minimum no
  configuration could install.
- **A write that cannot re-mode its temp file no longer strands it.** The atomic writer
  set the file's mode before the guard that cleans up after it, so on a filesystem that
  refuses `fchmod` — shared mounts do — the descriptor leaked and a `.tmp_*.part` was
  left in the directory being written to. Both are now inside the guard.
- **The `.err` tail a dead job's error quotes is decoded as UTF-8.** The tail was read
  with the driver's locale encoding, so under `LANG=C` — a login-node default — every
  non-ASCII byte in the one message that explains why a job died arrived as a
  replacement character. It is now read as UTF-8 like every other text file the
  package touches.
- **A stale-lock reclaim that sweeps up a fresh lock it cannot put back now reports
  it.** If a third driver's lock lands in the reclaim's rename→restore window, the
  restore fails and the swept-up run is left running with no lock on the path — and
  previously nothing said so. The reclaim now logs an error naming both drivers and
  keeps its claim file as the only surviving copy of the swept-up lock record, instead
  of deleting the evidence and passing silently.
- **An ExtOpt step whose environment cannot host the gradient server fails before
  submission, not inside the job.** The preflight accepted "the backend is importable
  here" without checking the server half it implies — so a bare install beside a
  hand-installed backend passed, and the job died hours later on `import waitress`
  with the traceback stranded in a log nothing pointed at. The single-env case now
  requires flask and waitress up front, with the error naming both fixes
  (`chemrefine[server]`, or `chemrefine backends install <extra>`); and a server that
  still cannot start logs one actionable line in its own `--log-file` — the file the
  job's failure path tails — instead of crashing before that file exists.
- **A `scancel`-ed (SIGTERM'd) driver releases the run lock and its local jobs.**
  Python's default SIGTERM disposition terminates without unwinding, so the lock stayed
  behind and a cross-host resume demanded a manual delete for a run that was genuinely
  dead. The lock now scopes a handler that turns SIGTERM into an orderly exit (code
  143); only a genuine SIGKILL can strand a lock, and the same-host dead-pid reclaim
  remains the net for that.
- **`_cache/` documents are readable on a shared tree.** The atomic writer's temp file
  is created 0600 and the rename preserved it, so the cache, manifest and failure
  ledger were owner-only beside world-readable outputs; they now honour the umask like
  any other written file (the server's token sidecar stays 0600 on purpose).
- **Editing `on_failure` over a cached step now takes effect.** The cache stores a step's
  results after the policy is applied, and the fingerprint deliberately excludes
  `on_failure` — so a step halted under `stop` and switched to `best` served the cached
  successes-only set, `skip` semantics, with nothing said anywhere (and a step switched
  away from `best` kept carrying its backfills). A policy edit across the `best` line
  over a non-empty ledger now re-attempts the ledgered failures and re-finalizes under
  the new policy; successes are never recomputed, and `stop` ↔ `skip` stays a free hit
  because both store the same results. The cache document records the policy it was
  finalized under (additive key — existing caches are read as before).
- **Two drivers racing to reclaim the same stale lock can no longer both acquire.**
  Reclaim deleted the dead holder's lock and re-created it, so two `resume`s arriving
  together after a crash could interleave — one deleting the other's fresh lock — and
  both drive the tree, the exact state the lock exists to prevent. Reclaim is now an
  atomic rename that exactly one process can win, verified against the record that
  justified it; and release only deletes the lock file while it still names the exiting
  process, so a driver whose lock was removed out from under it cannot take the new
  holder's with it on exit.
- **A corrupt `failed_jobs.json` now fails like every other corrupt cache file.** Valid
  JSON of the wrong shape escaped the ledger reader as a bare `TypeError` — a traceback
  with the generic exit code naming neither the file nor the fix — where every sibling
  `_cache/` reader raises `CacheError` (exit `7`) with the path. It now does the same.
- **An ExtOpt gradient server now runs on the step's own core budget.** The server — the
  compute half of an `mlip-extopt` / `pyscf-extopt` job — inherited an uncapped thread
  environment, so torch/MKL took every core on the node while the scheduler charged the job
  its `%pal`. Invisible under SLURM's cgroups; an oversubscription on every local run. The
  job script now exports the pal thread count before launching the server, and `1` for ORCA
  alone, which in ExtOpt mode is only the stepper.
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
