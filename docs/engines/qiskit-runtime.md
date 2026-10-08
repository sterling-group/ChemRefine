# IBM Runtime execution and mitigation

The experimental `ibm_runtime` estimator and sampler use the explicitly selected
Qiskit IBM Runtime **0.50** API. Install the `qiskit-runtime` backend profile and use
`implementation: executor` for the new client-side primitives. The optional
`implementation: legacy_v2` selects the deprecated server-side V2 primitives; it
does not follow the package's changing top-level aliases.

**Working locally:** the executor and legacy V2 adapters have numerical
validation against the Runtime 0.50 SDK's noisy fake backends. Executor TREX and
gate-folding ZNE, sampling and layout-aware fidelity run through these local
paths. **Unverified on real hardware:** this validation does not demonstrate a
successful IBM hardware run, remote calibration quality, account access,
session behavior on the service or billing. See the
[support status](qiskit-status.md) for the branch's validation boundaries.

Select `fake_backend` for a local noisy fake backend. It requires neither an IBM
account nor network access.
To target hardware, replace `fake_backend` with `backend_name` and select an
existing saved account using `account_name`, `channel` and/or `instance`. Credentials
belong in IBM's account storage, never in ChemRefine YAML or job journals. Hardware
jobs consume provider resources; this adapter does not submit during configuration
validation, catalog inspection or result parsing.

## Configure a local provider

Install the complete profile with `chemrefine backends install qiskit-runtime`,
or `python -m pip install "chemrefine[qiskit-runtime]"` in the executing
environment. From a source checkout, use `python -m pip install -e ".[qiskit-runtime]"`.
`ibm_runtime` is the component name; `qiskit-runtime` is the install extra and
managed backend name.

This is an `options` fragment for a molecular `qiskit` VQE step using the shipped
single-point template. Its fake backend simulates the released device noise
model, so its energy is a noisy estimate even with mitigation disabled:

```yaml
options:
  algorithm: vqe
  ansatz: uccsd
  initial_state: hartree_fock
  initial_point: zeros
  estimator:
    name: ibm_runtime
    options:
      implementation: executor
      fake_backend: FakeManilaV2
      mode: job
      default_shots: 1024
      resilience_level: 0
      seed_simulator: 1234
      seed_transpiler: 1234
      max_jobs: 1000
      max_nominal_shots: 10000000
  optimizer:
    name: spsa
    options:
      maxiter: 50
      learning_rate: 0.05
      perturbation: 0.1
      seed: 1234
  device: cpu
```

The worker supplies a durable journal path automatically. A standalone Python
caller must set an absolute `journal_dir` in the component options or provide
`CHEMREFINE_PROVIDER_JOURNAL_DIR`. To use Runtime sampling instead, select
`sampler: {name: ibm_runtime, options: {...}}` and supply the shared execution
controls plus `default_shots`; estimator-only `resilience_level`, `trex`, `zne`,
`pec` and `noise_learning` are rejected by its strict sampler schema. Explicit
shots in the consuming workflow override `default_shots`.

For hardware, remove `fake_backend` **and** `seed_simulator`, choose
`backend_name`, and use the existing saved account selectors as needed. Exactly
one of `fake_backend` and `backend_name` is required. `device` describes the local
worker allocation; Runtime components use `device: cpu`, including when they
submit to a quantum device. Configuration validation does not contact that
device. The [typed Runtime API](../api/engines_qiskit.md#sampling-measurement-and-providers)
and `chemrefine schema` list every supported control and default.

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

The unsupported combinations above fail during validation instead of producing
an apparently mitigated result. In particular, local fake-backend PEC/PEA cannot
provide Runtime 0.50's required noise-learning stage. The synthetic calibration
checks described below establish executor mathematics, rather than a working
remote calibration pipeline or an improvement on real hardware.

## Durable journals and explicit recovery

Normal and array workers write directly to the durable job output directory's
`provider_jobs` folder through `CHEMREFINE_PROVIDER_JOURNAL_DIR`. Python callers must
provide an absolute `journal_dir` or set that environment variable. Journals contain
a request digest, execution-volume summary, timestamps, state and returned provider
IDs. They omit credentials, account selectors, circuit payloads and provider error
messages. Intents are flushed before provider creation; returned IDs are flushed
immediately afterward. Reading journals makes no provider calls.

A process can fail between provider acceptance and receipt or durable storage of the
ID. Such an intent remains unresolved. The default
`resubmission_policy: refuse_unresolved` checks current and archived request digests
and refuses a new submission matching an unresolved prior request, including a known
job whose result was not recovered. Completed requests and repeated asynchronous
PUBs owned by the same live adapter remain usable. If an ID was returned but could
not be written, the error reports it for manual provider recovery.

Use explicit retrieval for known IDs. `resubmission_policy: allow_unresolved` instead
acknowledges that a fresh submission may duplicate already accepted work. It does not
cancel or resolve earlier jobs. The default policy refuses unresolved work;
change that field to `allow_unresolved` only when deliberately repeating a request.
These local safeguards make **no exactly-once execution claim**. Separate journal
directories, provider-side retries and lost durable storage remain outside their
scope. See [provider recovery](qiskit-recovery.md) for claim-file crash recovery and
the distinction between resuming a pipeline and retrieving provider work.

To retrieve known requests, provide `retrieve_job_ids` in the original submission
order with `mode: job`. Include calibration IDs before their corresponding energy
IDs when explicit noise learning was used. Reproduce the original execution options,
compiled circuits, parameter values and observables. Use a fixed transpiler seed.
The adapter verifies each request digest against its recorded ID before fetching a
job; mismatches, missing records and exhausted IDs fail without creating replacements.

Save the original integral bundle, complete resolved options and environment
inventory alongside any supported logical bound-circuit export. For retrieval,
also retain the exact submitted ISA circuits, applied observable layout and PUB
inputs when using a standalone client. A logical circuit export is a useful
scientific checkpoint, but recompiling it with a different SDK, backend target
or transpiler settings can produce a different request digest. The journals
deliberately omit circuit payloads and cannot reconstruct these inputs. See
[reproducible checkpoints](qiskit-checkpoints.md) for saving the Hamiltonian and
circuits, and [provider recovery](qiskit-recovery.md) for the separate retrieval
contract.

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
