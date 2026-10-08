# Qiskit Engines

The `qiskit` molecular engine, the `qiskit-experiment` artifact engine, and their
reusable Python interfaces. The orchestration package discovers both engines
without importing optional quantum SDKs; workers import the selected providers
when constructing components.

For runnable configurations, see [Qiskit molecular calculations](../engines/qiskit.md)
and [Quantum experiments](../engines/qiskit-experiment.md). The
[support status](../engines/qiskit-status.md) distinguishes supported calculations,
experimental research tools, known bugs, and deferred work. An API appearing here
does not establish hardware validation or chemical accuracy.

## Pipeline engines

The molecular engine renders one Python job per structure. The experiment engine
schedules one artifact-producing job and returns the incoming structures unchanged;
its manifest lives under `<output_dir>/stepN[_name]/experiment/artifact.json`.
Both use the common local/SLURM scheduler and validate required local output
payloads during recovery.

::: chemrefine.engines.qiskit.engine.QiskitEngine

::: chemrefine.engines.qiskit.experiment.QiskitExperimentEngine

## Preparing molecular problems

`ElectronicStructureData` describes caller-supplied molecular-orbital integrals.
`prepare_problem` applies the requested reductions without running a chemistry
driver. `prepare_pyscf_problem` constructs the integrals from a geometry instead.
Mapping is a separate operation, so native fermionic solvers can consume the
prepared problem without constructing a Pauli Hamiltonian.

See [Quantum data handoffs](../engines/qiskit-handoffs.md) for tensor conventions,
orbital ordering, geometry identity, and energy offsets.

::: chemrefine.engines.qiskit.data
    options:
      members: [MolecularMetadata, ElectronicStructureData]

::: chemrefine.engines.qiskit.problem
    options:
      members: [PreparedProblem, prepare_problem, prepare_pyscf_problem]

::: chemrefine.engines.qiskit.mapping.map_problem

## Solving and results

The public convenience imports live in `chemrefine.engines.qiskit.api`.
`solve_exact`, `run_vqe`, and `run_adapt_vqe` select their named algorithm;
`run_problem` accepts the full registered algorithm graph. `run_job` is the thin
geometry-driven entry point used by molecular templates.

Callback records are evaluation records whose objective excludes nuclear and
inactive-space constants. `QiskitRunResult.energy_hartree` restores the available
offsets. `success` records a finite returned result, while `converged` records the
solver's stopping criterion. Neither is a guarantee of chemical accuracy.

::: chemrefine.engines.qiskit.api
    options:
      members: [solve_exact, run_vqe, run_adapt_vqe]

::: chemrefine.engines.qiskit.workflow
    options:
      members: [validate_options, run_problem, run_job]

::: chemrefine.engines.qiskit.result
    options:
      members: [CircuitMetrics, QiskitRunResult]

## Typed configuration

Component names and their nested options are validated before submission. The
models accept named-component shorthand and keep scientific choices in the
configuration rather than in the Python template. Component-specific schemas
come from the registries; inspect the current catalog with `chemrefine schema`.

::: chemrefine.engines.qiskit.options
    options:
      members: [ComponentSelection, ActiveSpaceOptions, IntegralSourceOptions, CircuitExportOptions, QiskitOptions]

## Component registries and contracts

The eight molecular registries share `ComponentRegistry`. Builders import
optional SDKs lazily; registrations must run in both the orchestrator and worker.
Names, capabilities, transitive resource requirements, provider environments,
execution mode, and research domains are declared together. See
[Extending the registries](../engines/qiskit.md#extending-the-registries) for an
extension recipe.

| Registry | Builder product |
| --- | --- |
| `MAPPERS` | Qiskit Nature mapper |
| `INITIAL_STATES` | Reference circuit |
| `ANSATZE` | `AnsatzArtifacts`, containing a fixed circuit and/or adaptive operator pool |
| `ESTIMATORS` | `EstimatorResource`, including provider cleanup and optional compilation |
| `SAMPLERS` | `SamplerResource`, including provider cleanup and optional compilation |
| `OPTIMIZERS` | Classical optimizer |
| `INITIAL_POINTS` | Parameter vector |
| `ALGORITHMS` | `AlgorithmArtifacts` for Nature execution, or `NativeOutcome` for native execution |

`REGISTRIES` maps category names to those objects.
`consumed_component_categories` follows the selected components' transitive
requirements. It determines which execution resources and optional environments
the chosen algorithm actually consumes.

::: chemrefine.engines.qiskit.registry
    options:
      members: [NoComponentOptions, ComponentSpec, ComponentRegistry, consumed_component_categories, validate_component_graph]

::: chemrefine.engines.qiskit.context
    options:
      members: [ElectronicStructureContext, AnsatzArtifacts, SolverComponents, AlgorithmArtifacts, EstimatorResource, SamplerResource]

::: chemrefine.engines.qiskit.native
    options:
      members: [NativeSolveRequest, NativeOutcome]

`BACKEND_PROFILES` declares installable environments and their included provider
extras. It resolves a consumed provider graph to one compatible environment;
it does not merge or install existing environments.

::: chemrefine.engines.qiskit.profiles
    options:
      members: [BackendProfile, BackendProfileRegistry]

::: chemrefine.engines.qiskit.operators.OperatorPool

## Checkpoints and native bundles

Integral, determinant-state, tensor-factor, and bound-circuit checkpoints have
versioned JSON descriptors and hashed numerical payloads. Keep every descriptor
with the payload it names. Readers validate integrity and allocation limits;
plain result JSON excludes the in-memory states and circuits.

These artifacts preserve scientific inputs and selected outputs. They do not
serialize an optimizer's internal state or automatically resume provider jobs.
See [Reproducible quantum checkpoints](../engines/qiskit-checkpoints.md) for a
complete save, reload, solve, and energy-reconstruction workflow.

::: chemrefine.engines.qiskit.bundles
    options:
      members: [ArrayDescription, BundleDescription, QuantumBundle, write_bundle, read_bundle, bundle_dependencies]

::: chemrefine.engines.qiskit.integral_io
    options:
      members: [IntegralDescription, save_integrals, load_integrals]

::: chemrefine.engines.qiskit.state_io
    options:
      members: [StateDescription, save_states, load_states]

::: chemrefine.engines.qiskit.circuit_storage.CircuitDescription

::: chemrefine.engines.qiskit.circuit_io
    options:
      members: [BoundCircuit, bound_circuit, save_circuit, load_circuit, validate_circuit_bundle]

## Determinant states and orbital optimization

These interfaces describe number-conserving fermionic states and projected
Hamiltonians, evaluate reduced density matrices, and rotate or optimize orbital
frames. See [Sampled quantum states](../engines/qiskit-states.md) and
[Quantum spectra](../engines/qiskit-spectra.md) for the solver workflows and
their supported domains.

::: chemrefine.engines.qiskit.determinants
    options:
      members: [FermionTerm, FermionicHamiltonian, ReducedDensityMatrices, DeterminantState, ProjectedEigensystem, projected_operator, projected_eigensystem]

::: chemrefine.engines.qiskit.orbitals
    options:
      members: [OrbitalOptimizationOptions, OrbitalOptimizationResult, rotate_hamiltonian, optimize_orbitals]

## Sampling, measurement, and providers

Sampling accepts an unmeasured circuit, binds any supplied parameter values, and
requires an explicit positive shot count. Returned physical counts must have the
logical register width and sum to the requested shots. Local statevector, Aer,
and Runtime executors have different precision and noise semantics; the
configured executor determines which are available. Remote Runtime hardware
execution requires external authentication and explicit provider settings; local
Runtime fake backends need no credentials. Its durable journal records submission
identity rather than an optimizer checkpoint.

Use one `SamplingSession` for multiple requests so provider budgets and retrieval
cursors cover the whole experiment. Seed-capable local samplers receive a fresh
child stream on each request; a fixed session seed and request order reproduce
those streams. Repeated calls to `sample_circuit` each open a new session and can
repeat the first stream when the component seed is fixed. Providers without local
seed controls manage their own randomness. The session is synchronous and must
not be shared across concurrent threads.

See [IBM Runtime execution](../engines/qiskit-runtime.md) and
[Quantum provider recovery](../engines/qiskit-recovery.md) before submitting or
retrieving a remote job.

::: chemrefine.engines.qiskit.sampling
    options:
      members: [SampleBatch, SamplingSession, sample_circuit]

::: chemrefine.engines.qiskit.measurement
    options:
      members: [MeasurementOptions, measurement_groups, measure_observable]

::: chemrefine.engines.qiskit.runtime_options
    options:
      members: [RuntimeEstimatorOptions, RuntimeSamplerOptions]

::: chemrefine.engines.qiskit.runtime.build_runtime_resource

::: chemrefine.engines.qiskit.journal
    options:
      members: [RequestSummary, JournalRecord, read_journal, RequestJournal]

## Quantum experiment execution

`EXPERIMENTS` is the experiment component registry. `run_experiment` validates
the selected workflow, executes its builder, and publishes an artifact bundle.
An experiment extension also registers an output contract so local parsing and
recovery validate its scientific product without executing SDK code.

See [Quantum experiments](../engines/qiskit-experiment.md) for available names,
configuration, output kinds, and examples.

::: chemrefine.engines.qiskit.experiment
    options:
      members: [ExperimentResult, QiskitExperimentOptions, LatticeExperimentOptions, validate_experiment, run_experiment]

::: chemrefine.engines.qiskit.experiment_outputs
    options:
      members: [register_experiment_output, require_experiment_output, validate_experiment_output]

### Experiment option models

Each selection validates `experiment.options` through its registered model.
These adapters declare file inputs and connect the reusable scientific options
below to scheduled experiment execution. Pauli and surface-code resource
experiments use `PauliResourceOptions` and `SurfaceCodeOptions` directly.

::: chemrefine.engines.qiskit.experiment_measurement
    options:
      members: [PauliCircuitInput, MeasurementExperimentOptions, read_circuit]

::: chemrefine.engines.qiskit.experiment_dynamics.VariationalExperimentOptions

::: chemrefine.engines.qiskit.experiment_double_factorized.DoubleFactorizedExperimentOptions

::: chemrefine.engines.qiskit.experiment_shadows.ShadowExperimentOptions

::: chemrefine.engines.qiskit.experiment_rdm.RDMExperimentOptions

::: chemrefine.engines.qiskit.experiment_spacetime.SpacetimeExperimentOptions

::: chemrefine.engines.qiskit.experiment_cutting.CuttingExperimentOptions

::: chemrefine.engines.qiskit.experiment_resources.FactorizedResourceExperimentOptions

## Dynamics and local encodings

The following research interfaces build and simulate lattice, variational, and
double-factorized evolution, and construct or validate local fermionic encodings.
Their physical domains and allocation limits are described in
[Variational dynamics](../engines/qiskit-dynamics.md),
[Double-factorized evolution](../engines/qiskit-double-factorized.md), and
[Local fermionic encodings](../engines/qiskit-encodings.md). Reference-sector
reduction is documented in [Symmetry tapering](../engines/qiskit-tapering.md).

::: chemrefine.engines.qiskit.lattice
    options:
      members: [LatticeEdge, FermionicLatticeModel, LatticeIntegratorOptions, LatticeDynamicsOptions, LatticeCircuit, LatticeDynamicsResult, chain_lattice, square_lattice, build_lattice_dynamics, simulate_lattice_dynamics, lattice_encoding_qubits]

::: chemrefine.engines.qiskit.dynamics
    options:
      members: [VariationalDynamicsOptions, VariationalDynamicsResult, variational_dynamics]

::: chemrefine.engines.qiskit.double_factorized
    options:
      members: [DoubleFactorizedIntegratorOptions, DoubleFactorizedOptions, DoubleFactorizedEvolution, build_double_factorized_evolution, simulate_double_factorized_evolution]

::: chemrefine.engines.qiskit.encodings
    options:
      members: [LocalEncodingOptions, FermionicEncoding, build_local_encoding, validate_encoding]

::: chemrefine.engines.qiskit.flow
    options:
      members: [partition_flow_edges, vc_flow_diagonalizer, commuting_evolution]

::: chemrefine.engines.qiskit.tapering
    options:
      members: [Z2TaperingOptions, TaperingTransform, build_tapering_transform]

## Shadows, RDM reconstruction, and postselection

See [Fermionic shadows](../engines/qiskit-shadows.md),
[RDM reconstruction](../engines/qiskit-rdm-reconstruction.md), and
[Spacetime postselection](../engines/qiskit-spacetime.md) for estimator assumptions,
statistical uncertainty, reconstruction constraints, and acceptance diagnostics.

::: chemrefine.engines.qiskit.shadows
    options:
      members: [FermionicShadowOptions, ShadowSetting, ShadowResult, random_shadow_settings, shadow_circuit, orbital_shadow_snapshot, majorana_shadow_snapshot, rdm_expectation, estimate_fermionic_shadows, collect_fermionic_shadows]

::: chemrefine.engines.qiskit.rdm_reconstruction
    options:
      members: [RDMReconstructionOptions, RDMReconstructionResult, dqg_matrices, rdm_constraint_diagnostics, reconstruct_rdms]

::: chemrefine.engines.qiskit.spacetime
    options:
      members: [SpacetimeOptions, CheckedCircuit, SpacetimeResult, build_spacetime_circuit, collect_spacetime_counts, postselect_spacetime_counts]

## Circuit cutting and resource estimates

Cutting separates planning, sampling, and reconstruction. Resource estimators
produce model-dependent analytical estimates rather than device benchmarks or
hardware execution promises. See [Circuit cutting](../engines/qiskit-cutting.md)
and [Quantum resource estimates](../engines/qiskit-resources.md) for assumptions
and provider support.

::: chemrefine.engines.qiskit.cutting
    options:
      members: [CuttingOptions, WireCut, CuttingPlan, CuttingResult, plan_cutting, collect_cutting_samples, execute_cutting, reconstruct_cutting_records]

::: chemrefine.engines.qiskit.resources
    options:
      members: [QPEBudget, WalkOracleCost, PauliResourceOptions, qpe_queries, estimate_pauli_resources, MagicFactory, SurfaceCodeOptions, estimate_surface_code]

::: chemrefine.engines.qiskit.factorized_resources
    options:
      members: [FactorizedResourceOptions, THCFactors, save_thc_factors, load_thc_factors, estimate_factorized_resources]
