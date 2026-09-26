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

- **Structural QPY recovery checks.** Retained circuits now require complete,
  supported file and circuit headers, consistent counts, dimensions and bounded
  sections. SDK-free checks cover known formats 10–17; execution still decodes the
  complete circuit in its worker.

- **Readable boundary-size quantum bundles.** Descriptor limits now count the exact
  published UTF-8 bytes, including the final newline, and preserve prior outputs
  when a replacement exceeds that limit.

- **Imported electronic constants.** Integral bundles preserve named scalar offsets
  through active-space molecular energies and double-factorized evolution. DF/THC
  resource reports list these constants separately from query costs.

- **Physical spectrum residuals.** VQD and qEOM can measure bounded Hamiltonian
  residual diagnostics, retaining signed noisy variances and optimizer stopping
  facts separately from qEOM response-pencil errors.

- **Molecular-to-experiment handoffs.** Circuit-producing solvers can export their
  retained logical preparations for scheduled measurements and shadows, with
  nested payload hashing, scratch copy-back and SDK-free cache validation.

- **Portable quantum preparations.** Bound logical circuits retain parameters,
  mapping, particle sectors, active Hamiltonians, energy offsets and provenance in
  integrity-checked JSON/NPZ artifacts, using QPY compatible with Qiskit 1.4.

- **Shared quantum provenance.** Molecular results and experiment artifacts record
  consumed component graphs, selected provider dependency versions, interpreter and
  platform facts, and sanitized source identifiers through one collector.

- **Runtime restart protection.** Matching unresolved current or archived journal
  requests refuse resubmission by default. An explicit acknowledgement can permit
  new work; bounded, serialized intent publication preserves recovery evidence.

- **Molecular integral handoff.** Qiskit YAML can consume portable MO integral
  bundles, validating molecular geometry, charge, spin and nuclear-energy identity
  before solving. Both descriptor and numeric payload participate in caching.

- **IBM Runtime providers.** Registered estimators and samplers support layouts,
  job/batch/session execution, dynamical decoupling, twirling, TREX, ZNE and PEC.
  Credential-free journals support explicit retrieval; offline provider tests
  distinguish signed mitigation estimates from physical counts and shot budgets.

- **Quantum resource reports.** General Pauli-LCU/QPE budgets, domain-validated
  double-factorized and THC estimates, and explicit surface-code/factory assumptions
  produce versioned artifacts. Resource providers use a separate Python 3.12 worker.

- **Budgeted circuit cutting.** Manual gate/wire cuts, partitions and automated
  width-constrained plans preserve signed reconstruction and physical measurement
  records. Planning runs in an isolated worker; artifacts can be reconstructed
  locally without submitting another experiment.

- **Experimental endpoint Pauli checks.** Clifford payloads support signed,
  noncommuting user checks with explicit Aer noise. Artifacts preserve physical
  data/syndrome counts, accepted and rejected samples, Wilson acceptance intervals,
  conditional observables and the checked circuit.

- **Complex constrained RDM reconstruction.** Experimental D/DQ/DQG fitting keeps
  raw tensors, supports masked weighted losses and explicit energy regularization,
  and independently checks SCS feasibility without representability or energy-bound
  claims. Native artifacts retain solver diagnostics and complex coherences.

- **Durable quantum provider records.** Local and array workers keep credential-free
  request intents and returned job IDs outside scratch. Bounded, validated journals
  retain ambiguous submissions and support explicit retrieval from archived attempts.

- **Typed nested quantum inputs.** File dependencies now traverse nested validated
  models, containers and selected union branches. Bundle payload hashing uses the
  selected component's format declaration, preserving relocated cache identity
  without letting unrelated components reinterpret a same-named file.

- **Variational quantum dynamics.** VarQITE and VarQRTE use the selected estimator
  for observable, gradient and geometric-tensor circuits, including derivative
  ancillas and routed layouts. Bounded trajectories retain metric diagnostics and
  support the Qiskit 1.4 core floor and current providers.

- **Adaptive quantum solvers.** TETRIS and coupled-exchange ADAPT support
  reference-aware tapering, complex exchange quadratures, sector diagnostics,
  energy-increase rollback and freshly bound QNSPSA resources at each circuit growth.

- **Double-factorized quantum evolution.** Portable complex/unrestricted integral
  bundles feed domain-validated ffsim factorization and real compiled circuits.
  Supported complex hopping trajectories retain declared open-shell references,
  constant phases, tensor-compression diagnostics, independent evolution checks
  and reusable QPY circuits in native artifacts.

- **Reference-aware Z₂ tapering.** Molecular mappings discover compatible Pauli
  symmetries and verify sectors against the actual selected reference. Hamiltonians,
  pools, state preparation and observables share one Clifford transformation;
  expectation projection remains distinct from strict generator reduction.

- **Distinct fermionic shadow ensembles.** Quantum experiments acquire complex
  fixed-number orbital Haar shadows or signed Majorana-Clifford shadows with their
  respective inverse channels. Native datasets preserve settings, physical counts,
  complex RDMs and uncertainty grouped by randomized setting for provider-free replay.

- **Grouped quantum measurement.** YAML and Python workflows support ungrouped,
  qubit-wise commuting and general commuting Pauli measurements. Independent pilot
  allocation, joint-shot covariance, physical counts, input-file cache identity and
  array artifact validation share the quantum experiment contract.

- **Local fermion encodings.** Graph BKSF and open-square VC/DK mappings include
  code constraints, actual-reference preparation, complex observables and sample
  decoding. Flow-set evolution uses verified specialized VC circuits and general
  commuting synthesis, with particle-number drift and encoded-width budgets.

- **Quantum spectra and natural-gradient optimization.** VQD separates physical
  root energies from deflation penalties and validates root overlaps. Complex
  qEOM reports conditioned response spectra and reconstructed-state diagnostics.
  QNSPSA uses the selected sampler for fidelity, with declared transitive provider
  requirements and shared resource lifecycle management.

- **Recoverable quantum states.** Molecular workers persist determinant states
  and orbital frames in validated JSON/NPZ bundles. Scratch copy-back includes
  numeric payloads and circuit files; cache reuse checks referenced states before
  accepting a molecular result.

- **Orbital optimization and sampled Krylov states.** Explicit SQD, SqDRIFT,
  SKQD and extended SQD retain complex determinant amplitudes and orbital frames.
  SKQD samples powers of one fixed approximate evolution operator, including
  time zero. Extended SQD expands the actual reference excitation space without
  a Cartesian determinant closure. Root selection and original-basis observables
  share the molecular result contract.

- **Declared engine file dependencies and output validation.** Nested input file
  options and referenced payloads participate in relocation-safe cache identity.
  Engines can validate required native outputs before cache reuse; missing or
  corrupt products cannot masquerade as a successful artifact. Cache-only and
  rebuild operations remain read-only with respect to job submission.

- **Quantum experiment artifacts.** `qiskit-experiment` runs registry-selected
  workflows through the shared scheduler and preserves molecular structures. Lattice
  trajectories write versioned JSON/NPZ bundles with allocation limits, integrity
  checks, relative paths and scratch copy-back support.

- **A per-step gradient timeout for the ExtOpt engines.** `gradient_timeout_seconds`
  (default `600`) bounds one call of the bridge ORCA invokes per geometry. The old bound
  was fixed, and its expiry read as "server unreachable" — or, for a gradient the server
  was still computing, as a traceback in the wrapper. Expiry is now recorded as a timeout
  that names the knob. The cache key of every `mlip-extopt` / `pyscf-extopt` step moves
  with the new knob — the options payload it digests gained a field — so a tree cached
  before it recomputes those steps on `resume`, and `rebuild-cache` refuses such a tree as
  foreign: `rerun N` is the migration.
- **Every declared knob is a script placeholder, and the whole model is `$OPTIONS_JSON`.**
  A `stepN.py` for `mlip` or `pyscf` reads any field of its engine's options model as
  `$UPPERCASE` (`$MODEL_NAME`, `$BASIS`, `$GPU`, `$CORES`, …; an unset knob renders
  empty) where each engine used to list a few names by hand, and
  `options = json.loads("$OPTIONS_JSON")` takes the validated options whole — escaped
  so quotes, backslashes, newlines and `$` in a value survive the literal. Only declared
  knobs are placeholders: any other `$WORD` stays as written, and a key the model does
  not declare never reaches the script. For a template's own settings, `mlip` and
  `pyscf` steps take an `extra:` mapping — rendered as `$EXTRA` (a Python dict literal)
  and inside `$OPTIONS_JSON` — so a key you invent is deliberate and a typo of a real
  knob still warns; the engines that render no template refuse it. The cache key of
  every `mlip`/`pyscf` step moves with the new knob — the options payload it digests
  gained `extra` — so a tree cached before it recomputes those steps on `resume`, and
  `rebuild-cache` refuses such a tree as foreign: `rerun N` is the migration.
- **Config tooling** (`chemrefine validate | scaffold | schema | engines`):
  `validate` reports every finding at once — pydantic errors with field locations,
  unknown engines, bad values for declared option knobs, invalid NMS knobs — plus
  warnings for the silent no-ops (undeclared option keys, `nms: true` on an engine
  that cannot NMS, a seed `input:`, step templates and SLURM headers that do not exist
  yet, resolved the way dispatch resolves them); warnings never block, an unrunnable
  config exits 2.
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
  template, with the geometry generated into job 1's `$molecule` block (only job 1 of a
  multi-job `@@@` chain is edited — a later `$molecule read $end` survives untouched —
  and a block partitioned into fragments is refused by name). Parallelism is
  CLI-side, as Q-Chem wants it: `options.cores` renders `-nt N` and is allocated as
  `--ntasks=1 --cpus-per-task=N`; `options.nprocs` opts into MPI (`-mpi -np P [-nt N]`,
  allocated `P×N` — partial method support, so never a default). The install environment
  comes from `executables: {qchem, qc, qcaux}` — `qc` exports `QC`/`PATH` and Q-Chem's
  own documented `QCAUX=$QC/qcaux` default, `qcaux` overrides it for sibling layouts —
  or from a `module load` in the SLURM header; `device: cuda` and `backend_python`,
  inherited knobs nothing here reads, are refused at preflight. `QCSCRATCH` is the
  per-job work dir; the
  job always runs with a savename so key scratch (MOs) survives, and `options.save`
  copies it back to the structure dir. The `operation:` vocabulary is
  `sp` / `opt_sp` / `freq` — the GUI dropdown and the schema document pick it up —
  an unknown operation is refused at the run's preflight, and a step that omits
  `operation:` has its run type inferred from the template's `JOBTYPE` — `pes_scan`
  infers `pes`, `rpath` infers `irc`, `aimd` infers `md` and the string methods keep
  their names, engine-neutral words that fan a multi-geometry output out into its
  geometries; a job type the inspector does not list runs as `sp`. NMS-capable:
  `jobtype ts` targets the sampling and a frequency job — `jobtype freq` in an `@@@`
  chain, or `final_vibrational_analysis true` in the optimisation's own `$geom_opt`
  block — gates it; the frequency parse maps Q-Chem's 3N−6 vibrational modes onto the
  trivial-modes-first tensor the coordinator expects. Geometry is exchanged in Ångström only — a template setting
  `$rem input_bohr` is refused by name, since the writer emits Å and the reader
  assumes it. The output reader is deliberately minimal (final energy + last
  geometry) pending the full parser set.
- **Memory-aware SLURM requests**: an input that declares its memory now shapes its
  allocation. ORCA's `%maxcore` requests `ceil(maxcore × pal ÷ 0.75)` (maxcore is a
  promise ORCA overshoots per process — the 75% rule); Q-Chem's `mem_total` requests its
  declared peak (the scaffolded Q-Chem starter declares one: without it Q-Chem runs at
  its own 2000 MB default, however much SLURM grants). A header whose own `--mem`/`--mem-per-cpu` already covers the
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
  `pyscf`, `pyscf-extopt`, `qchem`, `qiskit`.
- Modular Qiskit Nature ground-state single points: strict registries make the
  mapper, algorithm, ansatz/operator pool, initial state, estimator, optimizer,
  and initial point independently selectable in YAML. Built-ins cover exact
  diagonalization, fixed VQE, ADAPT-VQE, active-space reduction, UCCSD and
  EfficientSU2, plus four local estimator modes: reference statevector,
  lightweight finite shots, exact-expectation Aer, and finite-shot Aer with an
  optional serialized noise model. The `[qiskit]` extra supplies the core
  Nature/Algorithms/PySCF stack; `[qiskit-aer]` adds pinned CPU Aer, while a
  custom Linux GPU environment can select Aer with `device: cuda`. Runs record
  resolved components, solver diagnostics, and variational evaluations, and a
  shipped H2 tutorial demonstrates the thin-template architecture.
- A driver-independent Qiskit electronic-structure API accepts validated real
  molecular-orbital integrals, preserves explicit active-orbital ordering and
  freeze-core offsets, and reuses prepared problems across exact, VQE, and
  ADAPT-VQE solves. Bravyi–Kitaev mapping, supplied UCC excitations, and external
  mapped operator pools extend the existing component boundaries. Owned results
  record energy contributions, reference errors, resource metrics, and retained
  ADAPT operators/gradient history. The optional `[qiskit-core]` dependency group
  omits PySCF; existing XYZ examples still use `[qiskit]`. The H2 `compare.yaml`
  tutorial reports all three solver results through the normal CLI outputs.
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

- **The ExtOpt wrapper starts in less than half the time.** ORCA spawns it once per
  optimizer step, and its import chain reached `ase.io` twice — through the XYZ helpers
  and through the MACE backend — for half a second it never used. Both import it on the
  first call that reads or writes a frame, so the wrapper never pays for it (0.30 s where
  it took 0.72 s), and a test holds its whole import chain free of it.
- **A threaded job is one SLURM task with N CPUs.** The `pyscf` and `mlip` script engines
  and `mlip-train` used to request `--ntasks=N --cpus-per-task=1` — MPI's shape — for a
  single OpenMP/torch process, which a scheduler may place across nodes, leaving the
  process the first node's share of a budget the throttler charged in full. They now
  request `--ntasks=1 --cpus-per-task=N`, the shape Q-Chem's threaded jobs already used,
  so the allocation is one node by construction. The core count, the thread exports and
  the budget charged are unchanged; only the directive pair moves. An ExtOpt step keeps
  the ranks' spelling — ORCA still runs its `%pal` MPI processes in `ProgExt` mode — and
  its script now pins `#SBATCH --nodes=1`, since the gradient server threads that same
  count in one process; an engine declares this through `single_node`.
- **The quickstart runs on a base install and ORCA.** `examples/first_run/` is the
  README's two-step pipeline — an xTB screen (ORCA's bundled GFN2-xTB) and a DFT
  refinement of three ethylene-glycol conformers, under a minute on a laptop, nothing
  to install beyond ORCA — and the README, the docs index and *Your first run* show
  those files verbatim (a test holds them together). The annotated tour of the config
  schema that sat at `examples/quickstart/` is `examples/schema_tour/`.
- **An ORCA template that walks a reaction path, a band or a trajectory is named for what
  it is.** `! IRC`, the `NEB` family and a `%md` block infer `operation: irc`, `neb` and
  `md` where they used to fall through to `sp`, so the run log and the step's records say
  what the output is. None of the three has a reader of its own yet: the output is read
  as its final structure with a warning, and the step fans out into its geometries the day
  a reader joins the dispatch, with no config change. An explicit `operation:` is still
  held to the readers' vocabulary. Both ExtOpt engines inherit the inspection.
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

- **Tapering at the supported Qiskit floor.** Reference-sector evaluation supports
  Qiskit 1.4 stabilizer states and preserves the signs of explicit Pauli generators.

- **Quantum artifact completeness.** All eleven experiment products now validate
  their kind, required arrays, dimensions and scientific metadata during completion,
  cache reuse and rebuild. Custom experiments declare SDK-free output contracts.

- **Independent quantum acquisition streams.** Shared sampler lifetimes preserve
  cumulative Runtime budgets and retrieval order across shadows, measurement groups
  and sampled evolution. Local requests use distinct reproducible child seeds,
  correcting repeated shot randomness and invalid shadow uncertainty estimates.

- **An API key holding a character HTTP headers cannot carry is refused by name.** A
  curly quote or an em dash pasted into the key made `chemrefine agent --check` blame the
  endpoint ("the reply is not a model listing") for a request it never sent; the key is
  refused where it is resolved, as a newline in it already was.
- **Every read the agent tools and the GUI make answers in the documented shape.** A
  template, `steps.csv` or run log this account cannot read is the tools' `ConfigError`
  naming the file, where it was a 500 with a traceback in the GUI and a generic tool
  failure over MCP; a GET without its query argument is the same `{error, exit_code}`
  refusal the POST endpoints give, not an HTML page the page cannot read; and a run the
  OS refuses to spawn leaves no empty log behind for `run_status` to serve as the newest.
- **A relaxed scan is read inside the `%geom` block that declares it.** The detector
  accepted the word `scan` anywhere after a `%geom` block, so an `Opt Freq` template whose
  `%pointcharges` path, `%base` name or coordinate file contained it was read by the scan
  parser: the final energy survived, the frequency table did not, and NMS refused the step
  for a reason that was not true. The keyword now counts only inside the block's own body.
- **A legacy key that holds a mapping refuses a scalar by name, from both loaders.**
  `executables` behind `orca_executable`, `normal_mode_sampling_parameters` and
  `sample_type.parameters` given a number raised a bare `TypeError` — a traceback out of
  `load_config` and a 500 out of `validate`, whose contract is never to raise. Each is now
  the same field error `options: 3` already was, naming the key.
- **An agent approval is a JSON boolean, never a truthy value.** The GUI's chat endpoint
  read each verdict with `bool()`, so the string `"false"` approved a suspended
  `save_config`, `write_template`, `scaffold_templates` or `start_run`. A verdict that is
  not a JSON boolean is refused in the documented error shape, with nothing run.
- **The developer tooling fails where it used to lie.** The docs viewer hook names the
  mode file when its header is missing or not a count, instead of a bare traceback from
  `mkdocs build --strict`; the mutation gate stops on an entry that would change nothing
  and puts its scratch copy *ahead of* an existing `PYTHONPATH` rather than in its place;
  the logo checker's resemblance gate skips only when numpy or Pillow is missing, not on
  any error inside it; and the logo generator writes the hand-authored SVGs after every
  inkscape step has succeeded, bounds each render with a timeout, and refuses an empty
  glyph query. The security page now says what the cache rule never covered: a model
  checkpoint is unpickled by its library, so it is code.
- **An output tree on a filesystem without hard links is refused by the run lock, with its
  exit code.** The lock is a fail-if-exists `os.link`, and a mount that refuses the call
  (vfat/exfat, some FUSE and SMB shares) escaped as a traceback from inside the claim.
- **A structure's cache digest no longer depends on the Python type of its energy.** The
  digest hashed `repr(energy)`, and a numpy scalar spells itself differently from the
  float it becomes after a cache round trip — a parser handing the driver one would have
  re-keyed every downstream row on every resume. Today's parsers hand over plain floats,
  so no existing key moves.
- **Four engine contracts brought back into line.** An ORCA `Opt` output no longer
  carries forces: its last gradient block is printed inside the last optimisation cycle
  and describes the geometry *before* the converged one, so the forces stored beside the
  result were computed at a different point (label a training set with a single point, as
  the tutorial does). An ORCA frequency banner with no mode under it is now "no data"
  rather than a verified minimum, as the Q-Chem reader already answered. Q-Chem reads its
  knobs strictly, so a misspelled `cores` is refused up front instead of dropping the job
  to one thread. An unknown ORB `model_name` is a `ConfigError` with the exit code every
  config mistake carries, not a bare `ValueError`.
- **The ExtOpt gradient server answers one call at a time, refuses a non-finite answer,
  and is given the time its backend takes to load.** The server served one stateful ASE
  calculator on four worker threads, so two overlapping gradient calls could answer one
  geometry with another's energy and forces; it now runs one worker. A model that could
  not evaluate a geometry answered `nan`, which the bridge accepted and wrote into ORCA's
  `.engrad`; both the server and the bridge now refuse a non-finite energy or gradient as
  the classified backend failure. And the model load — on a cold cache, its download —
  was counted against the two-minute readiness budget, so a first run on a slow link died
  as "did not become ready" for a download that would have finished; the server now
  reports while it is loading and the job waits through that separately.
- **An `on_failure: best` backfill of the submitted input is a geometry alone.** On a
  step after the first, the structure a failed job was given is the previous step's
  result, and the backfill carried it whole — so this step's filter ranked, and
  `steps.csv` reported under this step, an energy from the previous level of theory. The
  backfill now keeps the geometry and the lineage and nothing a calculation of this step
  could have filled, exactly as a step-1 seed arrives; the best geometry obtained keeps
  this step's own values. The next step's cache rows for such backfilled parents move
  once on an existing `best` tree (the parent's digest covers its energy), so a `resume`
  recomputes exactly those rows.
- **A periodic-only library refuses a molecular dataset before anything is written.**
  `mlip-train` with `task_name: chgnet` over the pipeline's own structures — molecules with
  no cell, which is every seed read from `.xyz` or built from SMILES — passed every
  pre-run check, wrote its splits, submitted, and died inside the training job with a
  singular-matrix error naming neither the structure nor the library, after the labelling
  steps were paid for. The trainer now declares that its model is periodic and the shared
  dataset writer refuses, naming the structures and the reason.
- **A native output that cannot be read as the molecule it claims is that structure's
  parse failure, never a smaller molecule or a dead run.** A PES scan point whose
  coordinate row overflowed (`*****`) or parsed to `nan` used to ship with one atom fewer
  and the whole molecule's energy; every scan point is now held to the first point's atom
  count and a corrupt row is refused. A dummy centre in the coordinate table (ORCA prints
  `DA` as `XX`) used to end the whole run in a traceback from inside ASE, discarding the
  step's successes; it is now refused where the structure is assembled, for every engine,
  as an unparseable output naming the symbol. A non-finite component in an ORCA normal-mode
  table withholds the tensor instead of displacing every NMS child along it.
- **A step's cached survivor order no longer depends on how its structures converged.**
  A `resume` or `rerun-errors` that retried a convergence failure cached the retried
  structure *last*, and an NMS re-attempt cached the re-run parents after the kept ones,
  while a fresh run and `rebuild-cache` keep manifest order. That order is part of the
  next step's cache key, so the same results cached under different histories keyed the
  tail differently: after such a resume, `rebuild-cache N` flipped the order back,
  refused `rebuild-cache N+1` as "a different configuration" and left `resume` to
  recompute a tail nothing had changed. Survivors are now put in their parents' order at
  the one point every path ends, so no history can move them — including a fan-out frame
  retried mid-queue, which used to land after its siblings. A tree that already holds a
  retry-ordered cache recomputes its tail once more on its next `rebuild-cache`, which is
  the state it is in today.
- **A run started with a numeric step target launches.** `start_run` — and so the GUI's
  Run panel over `/api/run` — accepted a step number as an integer (the same selector
  `/api/results` takes) but appended it to the child's argv unrendered, so `Popen`
  raised after the run log had been created: a 500 for the request and an empty
  `agent_runs/*.log` that `run_status` then reported as the newest run. The target is
  rendered as text like the two budgets are.
- **A managed backend environment runs the ChemRefine that drives it.** An editable
  orchestrator (`pip install -e .`) got a *snapshot* of its checkout in every managed
  environment, frozen at provisioning: the direct MLIP scripts and the ExtOpt server
  import ChemRefine inside that environment, so the next API change on that side failed
  every job with an `AttributeError` that nothing traced back to the environment, while
  preflight had passed it by name. An editable install is now mirrored as an editable
  install of the same checkout, and every run checks that each managed environment holds
  the same ChemRefine as the one running — same source, or same version from an index —
  refusing up front with `chemrefine backends install <extra>` (which installs into the
  environment that is there) where it does not. Environments built before this change
  are refused once, until that command is run.
- **`chemrefine backends install` accepts a conda-made env for what it is.** conda
  writes a `lib/python3.1 -> python3.12` alias symlink into the envs it creates; read
  first, it made the env "built on Python 3.1", which no backend supports, so installing
  into an existing conda env (a `pyscf-gpu` on top of `pyscf`, a refresh of any env) was
  refused with advice to delete it. The env's one real `lib/` directory is now the answer.
- **An unknown step target exits 2 from every entry point, `--dry-run` included.** The
  same mistake — `rerun cfg.yaml ghost` when no step is called `ghost` — exited 0 under
  `--dry-run` (the target was echoed as though it would run, after the dry run had
  promised to validate), 1 from the CLI (after the run lock was already taken), and
  carried exit code 2 from the agent tools. It is now the documented `ConfigError` (2)
  everywhere, refused before the dry-run summary and before anything touches the tree.
- **A negative `seed` is refused at config load.** The NMS `seed` and the training
  step's `seed` accepted any integer, and both hand it to `numpy.random.default_rng`,
  which refuses a negative with a bare `ValueError`: an NMS step died at its first
  fan-out after round 1 had run, a training step in `prepare` after every labelling
  step upstream — each as a traceback outside the exit-code contract. Both fields are
  now `>= 0`, so `chemrefine validate` and the run's preflight name the field before
  anything is submitted. No cache key moves.
- **A float knob refuses `.inf` and `.nan`.** `gradient_timeout_seconds: .inf` — an
  ordinary spelling of "no timeout" — reached the bridge as a socket timeout Python cannot
  represent, and every ExtOpt geometry step died in the wrapper with a traceback naming
  neither the knob nor the value; an NMS `displacement_value` of `.nan`, `.inf` or `0`
  submitted a full round-2 batch of unusable children. Those two, `window_kcalmol`,
  `temperature_k` and `job_timeout_seconds` are now held finite (and `displacement_value`
  positive) at config load, naming the field. There is no unbounded timeout: write `null`
  for `job_timeout_seconds` to wait indefinitely, and raise `gradient_timeout_seconds`
  rather than removing it.
- **The ExtOpt readiness probe no longer needs `curl`.** The generated job polled
  `/healthz` with `curl`, which nothing declared or checked for: on a node image without
  it every iteration failed, the loop ran its full 120 s, and the job died with "did not
  become ready" — the wrong diagnosis for a missing binary. The probe is now a stdlib
  `urllib` one-liner under the interpreter that hosts the server, which the job resolves
  anyway.
- **A step naming a `model_path` keeps its cache when the tree moves.** The loader
  resolves `model_path` to an absolute path so the job can open it, and that string rode
  into the step's cache row key — so the same config over the same model bytes derived a
  different key at every directory it ran from. A tree copied off a cluster recomputed
  the step that runs the model on `resume`, and `rebuild-cache` refused it as a different
  configuration. The key now carries the option by its basename; the model's bytes were
  always pinned separately, so retraining still re-runs the consumer. The cache key of
  every step naming `model_path` moves once with this change — a tree cached before it
  recomputes those steps on `resume`, and `rebuild-cache` refuses such a tree as foreign:
  `rerun N` is the migration.
- **A script template's `converged` must be a boolean.** The flag had no shape guard, and
  the lifecycle's verdict reads only a literal `false` as a failure — so a `stepN.py`
  assigning `converged = 0` or `"false"` ranked its structure as a converged survivor
  instead of the NOT_CONVERGED retry the field exists to trigger. Anything but `True`,
  `False` or unassigned is now refused at the parse boundary, naming the field, and lands
  in the ledger as that structure's UNPARSEABLE failure. The shipped starters already
  assign a bool (`bool(mf.converged)`, `mlip.last_converged`); a template writing `0`/`1`
  fails loudly from here on.
- **A GUI string field of the wrong JSON type is a 400, not a traceback.** A number,
  list or object sent where an endpoint reads a string — `yaml_text`, `path`,
  `config_path`, `text`, `name`, `base_dir`, a chat `model` or `base_url` — went straight
  into `yaml.safe_load`, `Path()` or `.encode` and raised the stdlib's `TypeError` out of
  the handler as a logged-traceback 500, at fifteen sites, while the counts and step
  selectors beside them answered 400. Every field read now carries the type contract the
  wire-number guard always had, and the refusal names the field and the shape; the agent
  preflight answers such a value as a finding, like every other unusable setting.
- **A GUI request missing a field it needs is a 400 that names the field.** Every
  `payload["…"]` read in the web app raised `KeyError`, re-raised as a logged-traceback
  500 — a rule one test pinned as deliberate while the newer endpoints on the same app
  answered the same class of input with the documented 400 (`/api/structure-file` with an
  empty body, the wire-number guard). One helper now serves every endpoint: a missing key,
  or a body that is not a JSON object at all, is refused as `{error, exit_code}`; a
  genuine bug inside a tool still surfaces as itself.
- **A step selector that is neither a number nor a name is refused, not crashed on.**
  `/api/results`, `/api/failures` and the agent's `get_results`/`get_failures` handed
  their `step` straight to the lookup, so a JSON float, list or mapping raised a bare
  `AttributeError` — a logged-traceback 500 from the GUI — and a JSON `true`, an `int`
  to `isinstance`, quietly selected step 1. `Config.find_step`, the one funnel every
  wire selector passes through, now refuses such a value as a `ConfigError`, so every
  caller answers the documented 400 / exit 2.
- **`rerun-errors` on an NMS step re-runs every round-1 job that left no usable result.**
  The NMS re-attempt resubmitted only the parents with *no* output and re-parsed
  everyone else's round-1 file, so a parent whose output a walltime kill had truncated,
  or whose program had died, failed the same way on every `resume` and `rerun-errors` —
  the command the exit-6 advice names as the repair. It now applies the rule the non-NMS
  resume applies: missing, unreadable and not-terminated round-1 jobs are sealed into
  `attemptK/` and re-run from a regenerated input, unconverged ones are still retried
  from their best geometry, and unresolved parents still keep their round-1 frequency
  output and re-run only round 2.
- **`mlip-extopt` refuses `extra`, as documented.** The engine read the direct `mlip`
  model, whose `extra` mapping is legitimate, and its server command emits only the flags
  it knows — so an `extra:` on an `mlip-extopt` step was accepted and read by nothing,
  the silent no-op the declared-key rule exists to catch. It now reads its own model,
  which refuses the mapping the way `mlip-train` and `pyscf-extopt` already do.
- **A `~name` the host cannot resolve is a refusal at every path a user types.**
  `expanduser` raises `RuntimeError` for an unknown account — neither an `OSError` nor a
  ChemRefine error — and the guard for it had landed on one GUI endpoint of five sites:
  the GUI's browse and save boxes answered a logged-traceback 500, and the
  `read_structure_file` and `save_config` tools the generic crash text. One helper
  (`agent_tools.expand_user_path`) now answers for all of them with the documented 400
  and a typed tool refusal.
- **A traceback that escapes a run starts at the error, not at the run lock.** The
  reentrant acquisition of the run lock (`recovery.execute` holds it, `pipeline.run` takes
  it again) yielded from inside its `except FileExistsError`, so every error raised under
  it was chained to that exception and every escaping traceback opened with "During
  handling of the above exception (FileExistsError: … run.lock)". It yields after the
  handler now.
- **An unreadable or unwritable `_cache/` document is the cache's own error, exit 7.**
  `read_json` refused malformed JSON as a `CacheError` but let a `PermissionError` (a
  mode-000 document, another account's tree) or a `UnicodeDecodeError` (bytes that are
  not text) escape, and no `_cache/` writer converted an `OSError` at all — so a `resume`
  over a colleague's cache, or a `rerun` into a read-only `_cache/`, was a traceback with
  exit 1 where the docs promise "cache corrupt or unwritable", exit 7 and the
  `rebuild-cache` advice. Both halves now raise `CacheError`; `load_if_valid` treats an
  unreadable cache as it treats a corrupt one, and recomputes.
- **A seed the run cannot use is a config error, not a traceback — and never a run that
  computes nothing.** `input:` pointing at a file that is missing, malformed (a frame
  short of its atom count, a coordinate that is not a number, bytes that are not text),
  a CSV without a `smiles` column, or a `.xyz` holding no structure reached the user as
  ASE's or pandas' own exception with exit 1 — outside the exit-code contract the
  template and header checks honour — while an empty file read as zero frames and the
  run exited 0 having computed nothing. All of them are now a `ConfigError` naming the
  seed (exit 2; a 400 from the GUI and the MCP tools), and `chemrefine validate` warns
  about an `input:` that does not exist yet, as it does about a template.
- **NMS knobs on an ExtOpt step are the sampler's, not strangers to the server model.**
  `mlip-extopt` and `pyscf-extopt` read their `options:` strictly — a typoed server knob
  must fail before anything is paid for — but the NMS knobs live in the same mapping, so
  `nms: true` with `target: ts` (or any other knob) was refused at preflight and by
  `chemrefine validate` as "Extra inputs are not permitted": an engine the table marks
  NMS-capable could sample only with every knob at its default. Every strict read now
  takes the engine's share of the options (`StepConfig.engine_options`), and a key
  neither reader declares is still refused.
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
- **Parsing a step is linear in its parent count.** The driver parses one job at a time,
  and every parse rebuilt the parent index from the whole previous state — the assembler
  once per job, a script engine once per structure — so a step over eight thousand
  parents spent seconds indexing per parse and a three-step `rebuild-cache` of such a
  tree tens of seconds on nothing. The state now indexes its structures once
  (`PipelineState.by_id`) and every reader shares it.
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
