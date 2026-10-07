# Qiskit capabilities and known limitations

The Qiskit integration supports molecular single-point calculations and a separate
quantum-experiment engine. Local numerical validation covers the standard solver
stack and the research workflows below within their documented domains. It does
not establish convergence, chemical accuracy, or physical quantum-hardware
performance for an arbitrary configuration.

`chemrefine schema` exposes registered components, strict options, provider
requirements, research status and supported domains without importing quantum
SDKs. Use that catalog together with these guides when choosing a workflow.
`standard` describes the supported component interface; `experimental` describes
a research implementation with explicit restrictions. Neither label is an
accuracy certificate. Installation and package ranges are documented in
[Installation](installing.md#qiskit-nature).

## Working capabilities

The following interfaces are implemented and have local validation. Research
methods remain experimental even when their numerical and artifact checks pass.

| Capability | What works | Status and practical limits | Guide |
| --- | --- | --- | --- |
| Molecular preparation and exact reference | XYZ/PySCF or supplied MO integrals; frozen core and explicit active spaces; particle- and total-spin-filtered exact diagonalization. | Standard. Exact solving scales exponentially. Integral order, occupations and scalar energy offsets must be explicit. | [Molecular engine](qiskit.md), [Integral handoffs](qiskit-handoffs.md) |
| VQE and upstream ADAPT-VQE | Fixed-circuit optimization and adaptive UCC/external operator pools; callbacks, resource metrics and selected-operator diagnostics. | Standard. Completion is distinct from convergence. Upstream ADAPT requires `reps: 1` and has the stopping limitation below. | [Molecular engine](qiskit.md#vqe-versus-adapt-vqe) |
| Mappers, references and ansatze | JW, BK, parity reduction; Hartree–Fock and explicit determinants; UCCSD, supplied UCC, rank-selected UCC, EfficientSU2, RealAmplitudes and excitation-preserving circuits. | Standard. Explicit UCC needs excitations. `excitation_preserving` requires untapered JW. Particle conservation alone does not establish total spin. | [Component choices](qiskit.md#built-in-components), [Additional choices](qiskit.md#additional-circuit-and-search-choices) |
| Estimators, samplers and optimizers | Exact local expectations, basic/Aer finite shots, serialized Aer noise models, single/double Aer arithmetic; SLSQP, COBYLA, SPSA, L-BFGS-B, Powell, Nelder–Mead, CG, BFGS and sampler-backed QNSPSA. | Standard local components. Statistical precision and arithmetic precision are different controls. QNSPSA is unsupported by upstream ADAPT but supported by the owned adaptive drivers. | [Estimator semantics](qiskit.md#estimators), [QNSPSA](qiskit-spectra.md#qnspsa) |
| Molecular spectra | VQD roots and complex qEOM; root/sector diagnostics and optional physical Hamiltonian residual measurements. | Experimental. VQD overlaps remain sampled. qEOM response residuals are distinct from physical eigenstate residuals; root ordering and accuracy need validation. | [Spectra](qiskit-spectra.md) |
| TETRIS and CEO adaptive solvers | Batched operator selection, QE/CEO pools, complex quadratures, explicit evolution synthesis, spin checks and energy-increase rollback. | Experimental. QE/CEO pools use JW or JW-based tapering; logical circuit counts exclude hardware routing. | [Adaptive solvers](qiskit-adaptive.md) |
| Native UCJ/LUCJ | `ffsim_vqe` optimizes balanced/unbalanced UCJ and restricted LUCJ in a fixed alpha/beta determinant sector. | Experimental, noiseless classical simulation of a quantum ansatz. Combinatorial state size and a strict final total-spin residual apply. No automatic coupled-cluster initialization. | [Fermionic workflows](qiskit.md#ucj-and-lucj-with-ffsim) |
| Cartesian SQD and grouped SqDRIFT | Fixed-circuit samples or supplied counts, occupancy recovery and selected-subspace diagonalization; grouped fermionic qDRIFT preparation. | Experimental. Released preparation/Cartesian kernels require real shared spatial integrals, untapered JW, nonempty alpha/beta populations and at most 63 spatial orbitals. | [Sampling and SQD](qiskit.md#sampling-and-sqd), [SqDRIFT](qiskit.md#grouped-sqdrift) |
| Explicit SQD and retained determinant states | Exact sampled-determinant projection, multiple roots, complex/unrestricted Hamiltonians and integrity-checked complex state artifacts; state and transition RDMs. | Experimental. Supplied-count explicit projection avoids Cartesian closure and its determinant-width restriction. Memory and subspace budgets still apply; circuit preparation may have narrower limits. | [Sampled states](qiskit-states.md) |
| Active orbitals and sampled Krylov spaces | Active-orbital minimization, SKQD evolution powers and extended-SQD excitation expansion, with retained orbital frames and states. | Experimental. Only active orbitals are optimized. Approximate evolution affects sampled coverage; larger sampled spaces do not guarantee a correct-spin or accurate state. | [Orbitals and Krylov spaces](qiskit-states.md#optimize-active-orbitals) |
| Lattice dynamics | Complex hopping graphs, Hubbard/extended interactions and potentials; JW/BK/parity propagation, Lie/Suzuki formulas and bounded exact comparisons. | Standard API and artifact workflow. Product-formula error needs refinement checks; time and energy units follow the model. | [Lattice API](qiskit.md#fermionic-lattice-dynamics), [Experiment engine](qiskit-experiment.md#lattice-trajectories) |
| Local encodings and flow sets | Graph BKSF and open-rectangle VC/DK; encoded references, constraints, observables and evolution circuits. | Experimental. Graph/component parity and square-lattice support restrictions apply. Pairing and arbitrary spin mixing are outside the interface. | [Local encodings](qiskit-encodings.md) |
| Reference-aware Z₂ tapering | Common transformation of the Hamiltonian, actual reference, pools and observables, including reference-sector verification. | Experimental. Retains one selected symmetry sector; omitted sectors cannot contribute roots. General non-Clifford references need explicit generators. | [Tapering](qiskit-tapering.md) |
| Variational dynamics | Measurement-based VarQITE/VarQRTE with the selected estimator, Euler/RK4 integration and metric diagnostics. | Experimental. Restricted circuit expressivity, metric regularization, step size and shot noise introduce distinct errors. | [Variational dynamics](qiskit-dynamics.md) |
| Grouped Pauli measurement | Commuting groups, pilot/production shot allocation, physical counts, covariance and uncertainty in a validated artifact. | Standard experiment. Finite-shot uncertainty is distinct from preparation error and model accuracy. | [Pauli measurement](qiskit-experiment.md#grouped-pauli-measurement) |
| Fermionic shadows and RDM fitting | Fixed-N orbital-Haar and Majorana-Clifford acquisitions; setting-level uncertainty; complex D/Q/G-constrained CVXPY/SCS reconstruction. | Experimental. Shadow errors are not guaranteed confidence intervals. Necessary RDM constraints do not prove full representability or a variational energy bound. | [Shadows](qiskit-shadows.md), [RDM reconstruction](qiskit-rdm-reconstruction.md) |
| Double-factorized evolution | Actual JW evolution circuits, factor reconstruction, constant phases, compiled costs and bounded exact-reference comparisons. | Experimental. Supports complex shared one-body and real shared two-body integrals; factor truncation, evolution and gate synthesis have separate errors. | [Double factorization](qiskit-double-factorized.md) |
| Resource estimates | Pauli QPE/query bounds, provider DF/THC costs, optional Qualtran cost graphs and explicit surface-code/factory assumptions. | Analytical estimates rather than executable workloads. DF/THC have real shared-integral and provider lookup-table restrictions; THC factors are supplied, not fitted. Physical runtime is a lower bound under supplied assumptions. | [Resource reports](qiskit-resources.md) |
| Circuit cutting | Manual gate/wire cuts, partitions and automatic width planning, followed by physical sampling and signed reconstruction. | Working optional addon workflow. High sampling overhead can defeat narrow widths; automatic search need not prove optimality. No reconstruction standard error is inferred. | [Circuit cutting](qiskit-cutting.md) |
| Endpoint spacetime postselection | Signed coherent Pauli checks around Clifford payloads in an explicitly nonideal Aer model; retained joint counts and conditional estimates. | Experimental local simulation. Implements endpoint checks; distributed check search, non-Markovian learning and hardware execution are unsupported. | [Spacetime checks](qiskit-spacetime.md) |
| IBM Runtime and mitigation | Explicit executor/legacy APIs, layout-aware primitives, batching and mode ownership, typed mitigation, durable journals and known-ID retrieval. | Experimental. Released fake-backend paths and local recovery contracts are validated; physical hardware and remote calibration/billing remain unverified. | [Runtime](qiskit-runtime.md), [Recovery](qiskit-recovery.md) |
| Artifacts, cache and GUI | Portable integrals, bound logical circuits, determinant states and experiment bundles; payload-integrity checks, cache invalidation, provenance and schema-driven GUI editing/errors. | Working local contracts and public CPU/CUDA pipeline paths. Preserve whole bundles; GUI validation is not visual browser acceptance or hardware certification. | [Handoffs](qiskit-handoffs.md), [Artifacts](qiskit-experiment.md#outputs-and-recovery) |

## Known bugs and incomplete behavior

These remain actionable; they are separate from unsupported combinations that
configuration validation deliberately rejects.

| Issue | Current behavior | Practical response |
| --- | --- | --- |
| Aer CUDA tensor-network sampling | Some configurations fail inside cuTensorNet with `CUTENSORNET_STATUS_INTERNAL_ERROR`; others time out. The failure pattern was reproduced through direct Aer calls. Primitive decoding can also encounter unusable shot memory after the provider failure. | Treat the requested method as failed. Explicitly choose a validated statevector method, or density matrix within its memory budget, for affected work. No universal tensor-network support is claimed. |
| Upstream ADAPT initial stopping | Qiskit Algorithms 0.4 raises when every initial pool gradient is already below the threshold, instead of returning the initial state as a successful adaptive result. | Inspect the reference, pool and threshold, or select an owned adaptive driver with its documented stopping rules. |
| Ordinary VQE spin reporting | The standard VQE result does not expose the standardized total-spin eigenstate residual provided by native/adaptive research paths. `preserve_spin` conserves alpha/beta populations, not necessarily total `S²`. | Perform independent total-spin validation before accepting a singlet or other requested multiplicity. |
| Incomplete optimizer verdicts | Some upstream optimizer results omit a convergence flag. ChemRefine retains unknown convergence rather than inferring it from finite energy or normal return. | Inspect available termination data, increase budgets and compare repeated starts with a same-space reference. |
| VQE callback parameter history | The supported upstream callback does not reliably provide the full parameter vector. ChemRefine retains objective records and the final optimal point, without claiming a full parameter trajectory. | Save the final bound preparation circuit and integral bundle for replay. Intermediate optimizer restart requires additional custom checkpointing. |
| Transpiled upstream ADAPT metrics | The supported upstream driver does not expose the final circuit before compilation, so logical metrics can be unavailable while compiled metrics remain present. | Use the retained compiled metrics with their target settings; unavailable logical counts are not zero. |
| Native CPU and memory allocation | Worker OpenMP/BLAS settings and selected provider thread controls constrain planned work. They do not impose a process-wide CPU affinity, native-pool limit or operating-system memory cap. | Use scheduler/cgroup/affinity and memory controls when hard allocation enforcement is required. Algorithm allocation guards are planning checks. |

Successful execution also has scientific limits: insufficient optimization,
sampled-subspace coverage, spin contamination, ansatz bias and finite-shot noise
can produce poor results without a software defect. Strict spin-residual
rejections are scientific failures of the candidate state, not engine crashes.
Equal-spin `symmetrize_spin` is only available when alpha and beta populations
are equal; it is not a general repair for spin contamination. A small energy gap
is meaningful only against a reference using the same Hamiltonian, offsets,
particle sector and relevant spin/symmetry sector.

## Defects already corrected

These corrections are included in the branch and should not be reported as
remaining failures.

| Correction | Resulting behavior |
| --- | --- |
| Reference-aware UCC serialization | Supplied/rank-selected excitations, non-prefix determinants and tapered mapper state survive supported parallel compilation without local-closure or integral-tensor pickling failures. |
| Consumed-component compatibility checks | Known incompatible excitation-preserving mapper combinations fail before worker submission and surface as structured GUI errors. Unused components do not restrict an exact calculation. |
| Aer error preservation | Original provider failure details survive result decoding and remain available in pipeline/GUI failure records. This improves diagnosis; it does not repair cuTensorNet itself. |
| Shadow uncertainty typing | Standard-error tensors use the real `RDMStandardErrors` type while complex measured RDMs retain their own type. |
| Timing and provenance validation | Performance tooling rejects duplicated/incomplete measurements and unmatched source, provider/build, hardware and allocation provenance before calculating comparisons. It bounds its compilation pools and cleans up its isolated workers. |
| Direct numerical dependency | SciPy is declared in the optional Qiskit core rather than relying on transitive installation. |
| CUDA workflow runner paths | The manual workflow initializes temporary runner paths during execution. It requires the configured dedicated runner and reviewed `main`; successful local GPU checks do not establish a hosted workflow run. |

## Unverified and unsupported scope

Physical IBM hardware submission, real remote calibration quality, provider
availability and billing behavior have not been certified by local fake-backend
tests. Synthetic calibration validation for PEC/PEA is not a hardware calibration
claim. Multi-node SLURM, distributed multi-GPU scaling and visual browser
acceptance are also unverified. A successful local CUDA statevector job establishes
that specific simulator path, not every exposed Aer method or precision/workload
combination.

The molecular engine does not provide nuclear gradients, geometry optimization or
frequency analysis. Standalone classical FCI/CASCI, CASSCF, coupled-cluster,
selected-CI reference and DMRG workflows are deferred; the exact and projected
diagonalizations inside quantum workflows remain available. Third-party automatic
entry-point discovery is not implemented: registration modules must load in both
the orchestrator and worker.

Known unsupported configurations fail explicitly. Examples include CUDA with a
non-GPU reference/basic provider, upstream ADAPT with QNSPSA, incompatible QE/CEO
base encodings, non-Clifford spacetime payloads, and out-of-domain provider
factorizations. Runtime legacy fake backends do not implement mitigation, and
executor fake backends do not implement the required PEC/PEA noise learning.
Do not treat these rejections as bugs or silently substitute a different method.

## Reproducible checkpoints

Keep prepared integral bundles, final bound logical circuits, state/experiment
payloads, seeds, resolved options and environment provenance together. Saving only
an optimal parameter vector does not preserve its orbital frame, circuit ordering
or Hamiltonian. [Quantum handoffs](qiskit-handoffs.md) explains portable integral
and circuit formats; [provider recovery](qiskit-recovery.md) explains why a provider
job ID or journal is not a complete numerical checkpoint. Integrity checks detect
missing or corrupt payloads but do not reproduce the scientific calculation.
