# Caching & Resume

Every step writes its parsed results to `{step_dir}/_cache/step.json` — one
human-inspectable JSON document holding the step metadata and every structure
(symbols, coordinates, energy, forces, status flags). It is plain JSON, not
pickle, on purpose: loading it can never execute code from the file, and float
round-tripping keeps coordinates byte-identical so the fingerprint is stable
across save → load.

## The fingerprint

The cache is keyed by a SHA-1 **fingerprint** covering:

- the step config — engine, operation, options, charge, multiplicity, template,
  the NMS flag; and
- the **parent structures** that fed the step — their IDs *and* their content
  (symbols, coordinates, energy), via `parents_digest`; and
- the **template contents** — a digest of the resolved template file
  (`input_digest`), so editing a template *in place* re-runs the step even though
  its basename is unchanged. This also matters because, when `operation` is
  omitted, the template's keywords decide what ORCA does.

If the YAML changes, or the seed file / a template's contents / any upstream
result changes, the fingerprint changes and the next run re-executes the step
(and every downstream step, because its survivor set changed).

The `sample:` filter is deliberately **excluded** from the fingerprint: the cache
stores the *pre-filter* results and filtering re-runs on every load, so tuning a
filter is a cache hit (re-filter), not a re-computation.

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
- **`failed_jobs.json`** — the ledger of failed structures (`structure_id`,
  `reason`); always written for visibility, but only `stop` failures are *pending*.

Writes are atomic (temp file + rename), so an interrupted write never leaves a
half-baked cache.

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
incompatible layout (or the pickle-era summary sidecar) is rejected on load,
forcing a clean rebuild rather than a silent wrong read.

See the [Cache API](../api/cache.md) for the functions involved.
