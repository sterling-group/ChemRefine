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
  (symbols, coordinates, energy), via `parents_digest`.

If the YAML changes, or the seed file / any upstream result changes, the
fingerprint changes and the next run re-executes the step (and every downstream
step, because its survivor set changed).

The `sample:` filter is deliberately **excluded** from the fingerprint: the cache
stores the *pre-filter* results and filtering re-runs on every load, so tuning a
filter is a cache hit (re-filter), not a re-computation.

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

## Cache format version

`step.json` records a `cache_format` version. A document written by an older,
incompatible layout (or the pickle-era summary sidecar) is rejected on load,
forcing a clean rebuild rather than a silent wrong read.

See the [Cache API](../api/cache.md) for the functions involved.
