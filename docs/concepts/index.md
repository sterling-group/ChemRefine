# Concepts

How ChemRefine works under the hood. These pages explain the design so you can
predict its behaviour and extend it confidently.

- [Architecture & Code Flow](architecture.md) — the module layering and a
  diagram of what calls what, from the CLI down to job submission.
- [Caching & Resume](caching.md) — the per-step fingerprint cache that makes
  re-runs skip unchanged work.
- [Recovery & Reruns](recovery.md) — the decision tables behind the six CLI
  actions: what each plans, what a mode may do, and the route through one step.
- [Filtering](filtering.md) — how survivors are selected at the end of each step.
- [Normal-Mode Sampling](nms.md) — the two-round imaginary-mode removal / TS search.
- [Security & Trust Boundaries](security.md) — which inputs run as code, and what
  the ExtOpt server and cache formats do defend against.

The guiding ideas:

- **Immutable state threading.** Each step receives the previous step's survivors
  as a frozen `PipelineState` and returns a new one — there is no mutable shared
  god-object.
- **Engine plugins behind a Protocol.** The orchestrator only ever sees the
  `CalculationEngine` contract and a name→class registry; no concrete engine is
  imported by the core. See [Adding an Engine](../developer/adding-an-engine.md).
- **Fingerprint-cached steps.** A step re-runs only when its config or its parent
  structures change.
