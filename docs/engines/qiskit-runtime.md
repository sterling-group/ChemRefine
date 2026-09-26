# IBM Runtime execution and mitigation

The experimental `ibm_runtime` estimator and sampler use the explicitly selected
Qiskit IBM Runtime **0.50** API. Install the `qiskit-runtime` backend profile and use
`implementation: executor` for the new client-side primitives. The optional
`implementation: legacy_v2` selects the deprecated server-side V2 primitives; it
does not follow the package's changing top-level aliases.

The `examples/tutorials/qiskit_sp/runtime_fake.yaml` example uses a
local noisy fake backend. It requires neither an IBM account nor network access.
To target hardware, replace `fake_backend` with `backend_name` and select an
existing saved account using `account_name`, `channel` and/or `instance`. Credentials
belong in IBM's account storage, never in ChemRefine YAML or job journals. Hardware
jobs consume provider resources; this adapter does not submit during configuration
validation, catalog inspection or result parsing.

## Circuits, execution and limits

Both primitives provide a transpiler with the backend target, `initial_layout`,
`optimization_level` and `seed_transpiler`. The molecular workflow applies the layout
to observables and uses the same compilation resource for fidelity circuits.
Standalone callers must pass ISA circuits and apply their layout to observables.
`seed_simulator` applies only to fake backends. The legacy local SDK ignores seed
zero, so that combination is rejected.

`mode` selects `job`, `session` or `batch`. New remote sessions and batches have their
own journal records. `max_time` bounds a newly created mode, and `max_execution_time`
sets the provider's per-job execution-time limit. Local fake backends do not enforce
remote execution-time limits. `mode_id` reuses a caller-owned mode and requires
`close_mode: false`; owned modes close when their resource exits unless explicitly
configured otherwise.

`max_jobs`, `max_pubs_per_job`, `max_parameter_sets`, `max_nominal_shots` and
`max_request_bytes` bound adapter submissions and input sizes. Batches preserve PUB
order and split at precision/shot changes required by the executor API. A PUB's
explicit precision or shots takes precedence over defaults. For the estimator,
`default_shots` takes precedence over `default_precision` when the PUB does not
specify precision.

The nominal shot budget counts requested PUB values times requested shots (or
`ceil(1 / precision²)`). It is **not a total physical-shot or monetary budget**:
measurement-basis expansion, randomizations, noise learning and mitigation overhead
can add substantial work. PEC has an additional finite `max_overhead`; noise learning
has explicit layer, depth and randomization bounds. Provider execution-time limits
apply separately to every submitted calibration and execution job.

## Mitigation controls

The estimator exposes typed `trex`, `zne`, `pec`, `twirling`,
`dynamical_decoupling` and `noise_learning` options. Resilience levels 1 and 2 supply
TREX and TREX plus ZNE defaults respectively; explicit toggles override these defaults.
The sampler supports twirling and dynamical decoupling. Estimator-only mitigation is
not accepted by the sampler schema.

ZNE supports the released gate-folding variants and probabilistic error amplification
(PEA), distinct noise factors and explicitly selected extrapolators. Configuration
validation rejects underdetermined polynomial fits, incompatible PEC plus ZNE, and
disabling twirling required by a chosen protocol. Mitigated estimates need not be
variational upper bounds, and finite-shot error bars do not include every systematic
error from calibration drift or model misspecification.

The executor requires learned layer models for PEC and PEA. Set
`noise_learning.enabled: true` explicitly. Before each energy batch, the adapter finds
the actual compiled layers, submits and journals a `NoiseLearnerV3` calibration job,
validates the returned model count and widths, and supplies each model with its
corresponding layer. Calibration results are included in the subsequent request's
digest. The current adapter recalibrates each batch; it does not silently reuse an
older model. Runtime 0.50 does not support `NoiseLearnerV3` on local fake backends, so
executor PEC/PEA with a fake backend is rejected. Legacy remote V2 performs its
provider-managed calibration internally; these internal stages have no separately
returned job IDs.

Executor TREX and gate-folding ZNE can run locally with a fake backend. Legacy local
V2 ignores mitigation, twirling and dynamical decoupling, so enabling these options
with that implementation is rejected.

## Durable journals and explicit recovery

Normal and array workers write directly to the durable job output directory's
`provider_jobs` folder through `CHEMREFINE_PROVIDER_JOURNAL_DIR`. Python callers must
provide an absolute `journal_dir` or set that environment variable. Journals contain
a request digest, execution-volume summary, timestamps, state and returned provider
IDs. They omit credentials, account selectors, circuit payloads and provider error
messages. Intents are flushed before provider creation; returned IDs are flushed
immediately afterward. Reading journals makes no provider calls.

A process can fail between provider acceptance and receipt or durable storage of the
ID. Such an intent remains unresolved. The adapter makes **no exactly-once execution
claim** and never resubmits an unresolved request automatically. If an ID was returned
but could not be written, the error reports it for manual provider recovery; an
absent-ID record cannot be matched automatically.

To retrieve known requests, provide `retrieve_job_ids` in the original submission
order with `mode: job`. Include calibration IDs before their corresponding energy
IDs when explicit noise learning was used. Reproduce the original execution options,
compiled circuits, parameter values and observables. Use a fixed transpiler seed.
The adapter verifies each request digest against its recorded ID before fetching a
job; mismatches, missing records and exhausted IDs fail without creating replacements.

Fresh worker attempts archive earlier products. For the injected default journal path,
recovery also searches the same job's `attempt*/provider_jobs` directories without
modifying them. Explicit `journal_dir` uses only that directory plus explicitly listed
`journal_history_dirs`. Searches have finite directory and record budgets and reject
conflicting records. Provider job retention and account access still govern whether a
known ID can be retrieved.

## Validation and sources

Offline integration tests exercise both released primitive implementations against
actual fake-backend noise models, layout-aware expectations, classified sampling,
TREX/ZNE and `ComputeUncompute` fidelity. Adapter tests additionally exercise job
batching, session ownership, failure recovery and calibration-model association.
Synthetic-channel tests substitute an analytically known calibration result while
running the real PEC/PEA executor and postprocessor. They verify PEC's signed
normalization, PEA's physical noise scales and the distinction between nominal and
physical shot totals.
These tests do not validate real hardware calibration quality, provider availability
or remote billing behavior.

The implementation follows the released [Runtime 0.50 source](https://github.com/Qiskit/qiskit-ibm-runtime/tree/f64f2dccc8c16463e3fdb0e67a60fbe1b9bb0209),
including its [executor estimator](https://github.com/Qiskit/qiskit-ibm-runtime/blob/f64f2dccc8c16463e3fdb0e67a60fbe1b9bb0209/qiskit_ibm_runtime/executor_estimator/estimator.py)
and [NoiseLearnerV3](https://github.com/Qiskit/qiskit-ibm-runtime/blob/f64f2dccc8c16463e3fdb0e67a60fbe1b9bb0209/qiskit_ibm_runtime/noise_learner_v3/noise_learner_v3.py).
