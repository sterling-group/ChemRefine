# Provider job recovery

Quantum workers receive `CHEMREFINE_PROVIDER_JOURNAL_DIR`, an absolute path to
`provider_jobs` inside their durable job directory. Both local and array scripts
set it before launching Python. Request records are written there immediately,
so they survive a worker failure before scratch output is copied back.

`RequestJournal` records a request digest and a bounded, credential-free summary
before invoking a submission operation. Once the provider returns a job ID, the
record is atomically replaced and synced before execution continues. Records
include the backend, primitive kind, register widths, nominal shot count, time
limit and UTC timestamps. Credentials, account names and circuit payloads are
excluded. Job summaries and nominal shots do not estimate calibration overhead.

The possible states are `intent`, `submitted`, `submission_unknown`, `completed`
and `result_unavailable`. A submission exception can mean the provider accepted
work but its reply was lost. Such a record remains explicitly ambiguous. Before
publishing a new intent, the journal checks current and archived records for the
same request digest. The default `resubmission_policy: refuse_unresolved` refuses
matching `intent`, `submission_unknown`, `result_unavailable`, and prior-owner
`submitted` records. A known job ID is retained when fetching its result fails.
Completed requests may be sampled again deliberately. Repeated asynchronous PUBs
submitted by the same live journal owner are allowed; known session/batch IDs
resolve mode creation and do not imply that their quantum jobs are complete.

Prefer explicit retrieval when the original job ID is known. If investigation
establishes that a fresh submission is appropriate, set the estimator or sampler's
`resubmission_policy: allow_unresolved`. This explicitly acknowledges that accepted
work may be duplicated; it does not retrieve, cancel, or resolve the prior request.
The policy does not bypass record or execution budgets. Neither policy guarantees
exactly-once execution across crashes, separate journal directories or provider-side
retry behavior.

## Local inspection and explicit retrieval

`read_journal(directory)` validates local JSON without importing a provider or
contacting it. `RequestJournal.matching_job(job_id, request_digest)` requires an
explicit ID and matching request. It refuses conflicting records. A provider
adapter can then explicitly retrieve that job during execution.

The shared pipeline archives previous products under `attemptK` before a fresh
run. `RequestJournal` accepts bounded, read-only `history_directories`; a Runtime
adapter using the injected worker path can search that job's archived journals.
A successful retrieval writes its updated record in the current journal while
retaining archived records unchanged. A caller choosing an external `journal_dir`
provides any history locations explicitly.

The default record limit is 10,000 across current and historical directories;
each JSON record is limited to 1 MiB. Journal files are validated against the
versioned schema and their UUID filenames. Atomic temporary writes are flushed
before replacement, followed by directory synchronization on POSIX systems.
A disk failure after acceptance can still prevent the ID from being persisted;
the raised error includes the returned ID for manual provider recovery.

An exclusively created `.submission.lock` file serializes the local-history scan
and new intent publication for writers sharing one current journal directory.
The claim is released before contacting the provider. A crash during publication
can leave it behind; subsequent submissions fail closed. Stop or verify that all
writers have stopped, inspect the durable records, and only then remove that claim
file. Historical attempt directories must remain quiescent during submission.
Independent current journal directories are not mutually locked.

## Pipeline cache behavior

Provider retrieval happens only as an explicit execution choice. Completion,
cache consumption and `rebuild-cache` validate local output bundles and parse
existing products. These paths do not poll, retrieve or submit provider jobs.

A missing or corrupt required payload invalidates a cached quantum result.
`RESUME` can schedule recomputation; `CACHE_ONLY` reports the unusable cache
without submitting. `rebuild-cache` validates and reparses existing local output.
During recomputation, the default Runtime policy still refuses matching unresolved
requests. Resolving that refusal requires known-ID retrieval or the explicit
resubmission acknowledgement above; ordinary resume is not a recovery decision.
Neither a journal entry nor a provider job ID substitutes for a complete validated
result bundle.
