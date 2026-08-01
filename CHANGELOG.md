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
  and recovery behave identically to the per-job path.
- Seeding from a multi-frame `.xyz`, a directory of `.xyz` files (all
  frames), or a CSV of SMILES (deterministic 3D embedding).
- Engineering gates: 100% line+branch test coverage, 100% docstring
  coverage, ruff, mypy, CodeQL, weekly `pip-audit`, and an
  mkdocstrings-rendered API reference.

### Changed

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

Hardening landed during the 2.0.0 stabilization:

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
