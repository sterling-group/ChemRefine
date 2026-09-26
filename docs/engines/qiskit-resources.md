# Quantum resource estimates

The `qiskit-experiment` engine exposes `pauli_resources`, `factorized_resources`, and
`surface_code_resources`. They write a versioned `resource_estimate` bundle, pass the
pipeline structures through unchanged, and do not populate molecular energies.
The estimates are analytical reports. They do not generate executable circuits.

Run the examples in `examples/tutorials/qiskit_resources/`.
DF/THC and Qualtran require a separate Python 3.12 worker:

```bash
chemrefine backends install qiskit-resources --python 3.12
chemrefine run examples/tutorials/qiskit_resources/pauli.yaml --dry-run
chemrefine run examples/tutorials/qiskit_resources/pauli.yaml
```

The worker profile keeps the optional resource stack separate from ordinary Qiskit
workers. Its minimum-Python marker does not select Python 3.12; the explicit
`--python 3.12` argument above does. Record the versions in each report when
comparing estimates.

## Pauli LCU and phase estimation

`pauli_resources` accepts a finite real dictionary of equal-width Pauli strings
in Hartree. Every Hermitian qubit Hamiltonian, including one obtained from complex
or unrestricted fermionic integrals, has real Pauli coefficients. Identity
coefficients are reported separately. The LCU normalization is the sum of the
absolute nonidentity coefficients; zero coefficients are discarded without an
approximation threshold.

`budget.energy_error_hartree` is divided among representation error, synthesis
error, and the remaining phase-estimation allowance. The first two are supplied
bounds. They are not inferred from an executable circuit. `target_overlap` is a
supplied lower bound on the **squared** target-state overlap. It defaults to one,
which assumes an eigenstate is already available. A smaller overlap increases
independent preparations. No state-preparation method or overlap certification is
included.

For a qubitized walk, energy depends on phase as `lambda*cos(2*pi*phase)` up to
an identity shift. The estimator chooses accuracy bits from this Lipschitz bound,
then adds the usual finite-register QPE success bits. Repetitions allocate half
the failure budget to missing the target and half to any inaccurate phase sample.
The reported success event is that all samples are accurate and the target is
sampled at least once; it does not identify which sample is the target. The
requested failure probability and overlap must be at least `1e-15`.

The query estimate deliberately uses conservative standard inverse-QFT QPE.
It is not an optimized phase-estimation cost or a quantum advantage claim.
See [qubitization](https://arxiv.org/abs/1610.06546) and the
[phase-estimation analysis](https://arxiv.org/abs/2111.10430).

An optional `oracle` supplies the cost of one **controlled** walk, including both
prepares and its reflection. Its `logical_qubits` includes system and workspace
but excludes the phase register. Gate categories remain separate, including
unsynthesized rotations. The report multiplies these supplied costs by queries;
inverse-QFT synthesis, initial-state preparation, routing, error correction, and
factories remain excluded. Without an oracle, the output contains no invented
gate count. `max_terms`, `max_phase_bits`, and `max_repetitions` bound allocations.

## DF, THC and provider cost graphs

`factorized_resources` reads the shared `electronic_structure` bundle through
`integral_bundle_path`. Integral and payload content participate in ordinary
ChemRefine cache fingerprints. Relative paths resolve beside the YAML file;
moving the complete input/output directory preserves references.

The current provider domain is at least two spatial orbitals with **real,
shared spatial integrals**. Equal explicitly materialized alpha/beta blocks are
accepted. Unequal unrestricted blocks, nonidentity alpha/beta overlap, or complex
coefficients fail before provider execution. Explicit chemist and physicist
orders are supported. Pauli estimates cover the broader mapped domain.

For `method: df`, OpenFermion's public `df.factorize` produces the retained
factors. The Coulomb matrix must be positive semidefinite within numerical
tolerance. `factorization_threshold` controls truncation. The retained rank is
read from the factor array rather than the provider's loop-index return value.

For `method: thc`, `thc_bundle_path` supplies real leaves `eta[P,p]` and a real
symmetric central matrix `zeta[P,Q]`. Their contraction is
`g[p,q,r,s] = sum_PQ eta[P,p]*eta[P,q]*zeta[P,Q]*eta[Q,r]*eta[Q,s]`.
Use `THCFactors` and `save_thc_factors` to write the portable bundle. This workflow
does not perform a THC fit. Changing the DF threshold under THC is rejected.

Both methods reconstruct the approximate integral tensor and compute the
conservative full-Fock operator bound `2*sum(abs(g-g_approx))`. A factorization
whose bound exceeds `representation_error_hartree` fails. This sufficient bound
can be much stricter than a molecule-specific energy error. The shifted
one-body term and normalization are consistently calculated from the approximate
tensor while retaining the input one-body integrals. Nuclear energy is reported
separately; scalar offsets do not consume queries. Named electronic constants
from an integral bundle are preserved in `input_energy_offsets_hartree`, alongside
the separate nuclear repulsion field, without changing normalization or query costs.

OpenFermion's public cost functions return per-step Toffolis, their own single-run
QPE Toffoli total, and logical qubits **including the system and phase register**.
The released QROM cost implementation also requires each lookup table to have
at least as many entries as its output payload has bits. Small systems or low
factorization ranks can violate this condition, even with otherwise valid
integrals. ChemRefine checks every affected lookup and reports an actionable
unsupported-domain error. It never pads factors or reduces precision to obtain
a cost. Use `pauli_resources` for general query bounds in that case. Successful
reports include the checked table sizes and payload widths. The runnable DF/THC
examples use synthetic eight-orbital, rank-32 factors within this provider domain.

Their `ceil(pi*lambda/(2*epsilon))` iteration formula does not certify the requested
failure probability. The report therefore retains it separately from the
conservative standard-QPE bound. Coefficient/rotation bit counts must be explicit;
their energy error remains the user's supplied synthesis bound.
See the [provider DF implementation](https://github.com/quantumlib/OpenFermion/tree/v1.8.1/src/openfermion/resource_estimates/df),
[THC implementation](https://github.com/quantumlib/OpenFermion/tree/v1.8.1/src/openfermion/resource_estimates/thc),
and [underlying resource formulas](https://arxiv.org/abs/2011.03494).

`qualtran_cost_graph: true` adds the DF `DoubleFactorizationBlockEncoding` graph's
native gate categories and signature width. It does not multiply that graph by
QPE queries: a block encoding and a controlled qubitized walk have different
overheads. Signature width is not a peak-workspace guarantee. The provider's
normalization/error certification and full circuit synthesis are not inferred.
`amplitude_bits_outer` and `amplitude_bits_inner` apply only to this graph;
changing them without enabling the graph fails. See the
[Qualtran DF documentation](https://qualtran.readthedocs.io/en/latest/bloqs/chemistry/df/double_factorization.html).

## Explicit physical assumptions

`surface_code_resources` accepts a separately supplied logical workload:
qubit count, logical cycles, and T/CCZ state counts. It does not infer a schedule
from an analytical gate count. All hardware assumptions are required, including
odd code distance, physical error, threshold, cycle duration, routing patches,
patch footprint, and the logical-error prefactor. Its stated phenomenological
per-patch/per-cycle error is `A*(p/p_threshold)**((d+1)/2)`.

Each consumed magic-state species requires an explicit factory count, footprint,
cycles per state, output error, and provenance. Factory error includes the
factory's internal faults. The runtime is the maximum of the supplied algorithm
cycles and ideal aggregate factory-throughput cycles. This is a **lower bound**
that excludes dependency stalls, warm-up, buffers, and decoder delays.

The fault union bound is evaluated at that runtime lower bound. A passing budget
at this lower bound does not certify a feasible scheduled machine; longer
runtime increases data-fault exposure. This model is for comparing declared
assumptions, not predicting hardware performance. The
[surface-code literature](https://arxiv.org/abs/1208.0928) motivates the scaling;
no hardware parameters are taken from the paper automatically.

Python entry points live in `resources.py` and `factorized_resources.py`. The
YAML adapters own only bundle loading and publication; ordinary artifact-engine
validation, cache-only behavior, and resume/rebuild semantics apply.
