# Caching & Resume

Every step writes its parsed results to `{step_dir}/_cache/` as two files:

| file | holds |
|------|-------|
| `step.json` | step metadata + each structure's scalars — id, lineage, energies, status flags, symbols |
| `arrays.npz` | the bulk — coordinates and forces, concatenated with an offsets index |

Neither can execute code when loaded, which is the reason this is not pickle:
JSON cannot by construction, and the `.npz` is read with `allow_pickle=False`,
which makes numpy *raise* rather than run an object array. Coordinates round-trip
byte-identically either way, so the fingerprint is stable across save → load.

**Why the split.** Coordinates are 93% of a record. Written as decimal text each
float64 costs 18 bytes on disk, a `strtod` call to parse and 32 bytes live; as a
`.npy` member it costs 8 bytes, a memcpy and 8 bytes. Over 10,000 structures of
10–120 atoms:

| | one JSON document | JSON + `.npz` |
|---|---|---|
| on disk | 69.1 MB | **31.3 MB** |
| save | 2.32 s | **1.06 s** |
| load | 1.73 s | **0.36 s** |
| peak memory | 268 MB | **75 MB** |

`tests/test_perf_cache.py` re-measures this (`pytest -m perf -s`); these
figures come from that run.

An `.npz` is an ordinary ZIP of `.npy` members, and a `.npy` is a short ASCII
header plus the array's raw buffer — inspect it with `unzip -l` or `np.load`.
It is stored uncompressed: float64 coordinates deflate by about 5% and cost
roughly twenty times the encode.

Structures within one step need not share an atom count — `input:` pointing at a
directory or a SMILES CSV seeds different molecules into the same step — so the
arrays are concatenated with an offsets index rather than stacked, and forces
carry a present-mask because an engine may report none.

`step.json` is written **without indentation**; read it with `jq` or
`json.load`, not by eye. Each structure also gets its own indented
`step{N}_{id}.result.json` next to its output files, coordinates included —
that is the artifact meant for reading.

A `step.json` whose `arrays.npz` is missing is an **error**, never a fallback to
reading coordinates inline: that is the one way a tree could end up half in each
format, and it would be silent. The step rebuilds instead.

## The fingerprint

The cache identity is layered the way a step's work is layered:

- a **row key** per structure — everything that determines *that structure's job*:
  the engine and operation, the template's **contents** (a digest of the resolved
  `StepContext.template`, so editing a template in place re-runs its rows even though
  the basename is unchanged), the **effective** charge and multiplicity (the values
  jobs render, whether set on the step or inherited from the workflow), the options
  **as the engine's declared model reads them** (a key nothing declares reaches no job
  and moves no key — `validate` warns about it instead), a digest of any file an
  option names (retraining a model re-runs its consumers), and the parent structure's
  own content (ID, symbols, coordinates, energy);
- a **resolution key** for an NMS step — the flag plus the `nms` options, split into
  the *criterion* (`target`, `ts_mode_index`) and the *search*
  (`displacement_value`, `num_random_displacements`, `seed`). It touches no row key:
  round-1 jobs are byte-identical with NMS on or off, which is what makes turning
  `nms: true` on over a finished run cost only the displacement children;
- the step **fingerprint** — a SHA-1 composing the ordered row keys with the
  resolution key. An exact match serves the whole step from `step.json`; every finer
  question is asked of the rows.

If the seed file, a template's contents, an option a job can read, or any upstream
result changes, the affected **rows** change — and `resume` recomputes exactly those
rows, at that step and at every step downstream, adopting the rest from disk
(see below).

The `sample:` filter is deliberately **excluded** from every key: the cache stores
the *pre-filter* results and filtering re-runs on every load, so tuning a filter is
a cache hit (re-filter), not a re-computation. `on_failure` is likewise excluded —
it has its own repair path. The `executables` map is a machine-local fact and is not
part of a step's identity.

## Auto-retry on non-convergence

Before the `on_failure` policy runs, a structure whose job **did not converge**
(SCF / geometry MaxIter, only ORCA flags this) is retried **once per run** from its
best geometry: the failed attempt's files are archived into a numbered
`stepN/<id>/attemptK/` sub-dir, the input is re-prepared from the last good
geometry, resubmitted, and re-parsed. A crashed or missing-output job is *not*
auto-retried — it goes straight to the policy. The retry is a single inline pass
(at most one per run, never recursive); a later `resume` archives into the next
`attemptK/`, so re-runs keep trying without ever looping or being blocked by an
existing attempt dir.

## Resume and recovery

`resume` honours the cache and re-attempts the pending failed jobs of any
`on_failure: stop` step. Alongside `step.json`, two sidecars under `_cache/`
support recovery:

- **`manifest.json`** — the input → output → structure-ID file layout, so
  `rerun` / recovery can rehydrate which input produced which output after a restart.
  It also carries the step's **provenance** — the fingerprint, the resolution's
  criterion, and each row's own key — written *before* any job is submitted. That is
  what lets a later `resume` prove, structure by structure, that an output on disk is
  the one this configuration would compute: matching rows are **adopted** (re-parsed,
  never resubmitted) and only the rest run. A manifest *without* row provenance — an
  output tree from before these rules, or one hand-written for a v1 adoption — is
  never silently trusted or silently discarded: `resume` stops and names the fix,
  `chemrefine rebuild-cache N`, which re-parses the outputs under the current rules,
  submits nothing, and records the provenance.
- **`failed_jobs.json`** — the ledger of failed structures (`structure_id`,
  `reason`); always written for visibility, but only `stop` failures are *pending*.

`step.json` is written once, at the end of a step, so "no `step.json`" means the
step did not finish. That is deliberately distinct from "the user discarded it":
`run` and `rerun` drop the manifest too, so a step you asked to redo is never
mistaken for one that was merely interrupted.

Writes are atomic (temp file + rename), so an interrupted write never leaves a
half-baked cache.

### One driver per output tree

Everything above assumes a single driver, and the assumption is load-bearing:
the manifest fingerprint proves *what configuration* produced the outputs on
disk, not *whether the run that produced them is still alive*. A live driver
mid-step leaves exactly the state an interrupted one does, so a second driver
resuming over it would read half-written outputs as failures, archive them out
from under the running jobs, and resubmit duplicates.

So each run holds an advisory lock — `<output_dir>/.chemrefine.lock`, naming
its pid, host and start time — for its whole duration, and a second run against
the same tree fails fast with exit code `10` instead. A lock whose holder died
on the same host is detected and reclaimed automatically; one left by a run
killed on *another* host cannot be liveness-checked from here and must be
deleted by hand — see
[troubleshooting](when-a-run-fails.md#another-run-holds-this-output-tree).

## Result records

Beside every parsed output the pipeline drops `step{N}_{id}.result.json` — the
canonical, engine-independent parsed result. Its body is the exact schema the
cache document's `structures` entries use (`structure_record`: one schema, two
envelopes), wrapped in a `result_format` version. A derived artifact for users
and tooling: the pipeline itself re-parses native outputs on rebuild, and ORCA
runs additionally leave ORCA's own `basename.property.json` (requested via
`%output JSONPropFile`) next to the `.out`.

## Cache format version

`step.json` records a `cache_format` version. A document written by an older,
incompatible layout — or a summary-only sidecar carrying no `structures` — is rejected on
load, forcing a clean rebuild rather than a silent wrong read. The document also names the
digest of the `arrays.npz` it was written with, so a sidecar left by a *different* save is
rejected the same way rather than pairing one structure's energy with another's geometry.

## Cost at scale

The cache is rewritten in full on each save, and `parents_digest` re-hashes every parent's
coordinates once per step. Both are linear in the structure count, and both are negligible
next to the calculations they bookkeep. Measured on 30-atom structures
(`tests/test_perf_cache.py`, run with `-m perf`):

| structures | `parents_digest` | `cache.save` | `cache.load` | `steps.csv` | `_cache/` | live state |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.001 s | 0.02 s | 0.01 s | 0.004 s | 0.4 MB | 0.6 MB |
| 2 000 | 0.01 s | 0.15 s | 0.09 s | 0.01 s | 3.6 MB | 6.5 MB |
| 10 000 | 0.06 s | 0.97 s | 0.24 s | 0.05 s | 17.9 MB | 32.6 MB |

A 10 000-structure step spends about a second on all of its bookkeeping, against a step
running 10 000 quantum-chemistry jobs.

The number worth watching is the last column, not the time. The pipeline holds every
structure's `ase.Atoms` in memory for the whole run, so **residency**, not disk, is the
ceiling — and most of it is ASE's object graph rather than the coordinates. A workflow
needing 10⁵ structures in one step would hit that first, and the answer would be to stop
holding them all at once, not a faster serializer.

See the [Cache API](../api/cache.md) for the functions involved.
