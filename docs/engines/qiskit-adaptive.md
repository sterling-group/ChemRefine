# TETRIS and coupled-exchange adaptive VQE

`algorithm: tetris_adapt` and `algorithm: ceo_adapt` are experimental owned adaptive
drivers. They measure pool gradients using the selected estimator, append generators,
and jointly optimize all retained parameters. Each inner solve constructs a fresh
optimizer for the grown circuit, including QNSPSA's sampler-backed fidelity metric.
They consume the same prepared molecular problem, energy offsets, actual reference,
component resources and result contract as the other molecular algorithms.

These methods can reduce circuit growth costs for some problems; neither guarantees
a global minimum or chemical accuracy. `converged` describes the configured stopping
criterion. Always compare small instances against an exact reference in the same
particle, spin and tapering sector.

## TETRIS

```yaml
algorithm:
  name: tetris_adapt
  options:
    max_iterations: 30
    max_parameters: 128
    gradient_threshold: 1.0e-5
    gradient_norm: l2
    evolution: exact_commuting
ansatz: uccsd
initial_point: zeros
estimator: statevector
optimizer:
  name: slsqp
  options: {maxiter: 200, ftol: 1.0e-12}
```

TETRIS ranks absolute measured gradients and greedily appends operators with disjoint
qubit support in the same iteration. Support is the union of nonidentity Pauli support
**after mapping and tapering**. All new parameters start at zero; earlier optimized
parameters are retained. UCC pools, generalized qubit-excitation pools (`ansatz: qe`),
and Python-supplied `OperatorPool` values are supported. An external pool must already
use the selected reduced register and mapper. Its metadata never substitutes a different
circuit for the supplied operator. UCC `reps` must be one because adaptive execution
consumes its operator pool.

The selection rule follows the [TETRIS-ADAPT-VQE paper](https://arxiv.org/abs/2209.10562).
The original upstream `adapt_vqe` remains available and uses its existing driver.

## Coupled exchanges

```yaml
algorithm:
  name: ceo_adapt
  options:
    variant: adaptive
    tetris: true
    max_iterations: 30
    selection_threshold: 1.0e-12
    spin_constraint: require
ansatz:
  name: ceo
  options: {include_imaginary: false, max_pool_size: 4096}
```

The CEO pool contains generalized same-spin single qubit excitations and doubles
preserving electron number and spin projection. Four same-spin orbitals have three
independent exchanges; two alpha and two beta orbitals have two. Every pair contributes
sum/difference one-parameter (OVP) candidates. These are occupation-qubit ladder
operators without fermionic Jordan–Wigner parity strings.

`variant: adaptive` ranks OVP gradients. If more than one underlying QE gradient on the
selected support exceeds `selection_threshold`, it adds independently parameterized
QE generators as a multiple-parameter (MVP) block; otherwise it adds the selected OVP.
`variant: ovp` always uses one-parameter candidates. `variant: mvp` ranks support groups
by the sum of absolute underlying gradients and retains all their independent
parameters. `tetris: true` greedily batches disjoint groups; false selects one group.
This implementation follows [Ramôa et al.](https://doi.org/10.1038/s41534-025-01039-4).

For an unreduced Jordan–Wigner occupation register, antisymmetric single exchanges use
two CX gates, OVP doubles use nine, and the joint MVP double circuit uses thirteen,
following Figures 4 and 9 of that paper. Complete unitary tests check signs, independent
angles and global phases. These are logical CX counts before hardware routing; they
are not resource guarantees for a target device.

The built-in QE/CEO pools support `jordan_wigner` and `z2_tapered` with a Jordan–Wigner
base. Tapered generators are consistently transformed and sector-changing or zero
generators are removed. Reduced circuits use Pauli synthesis instead of the original
9/13-CX networks. Parity and Bravyi–Kitaev bases are explicitly rejected for these pools;
TETRIS with a correctly mapped UCC/external pool can use them.

`include_imaginary: true` adds symmetric Hermitian exchange quadratures, allowing
complex amplitudes. These are kept separate from antisymmetric groups and use generic
Pauli synthesis; the optimized odd-Y circuit counts do not apply. Real and complex
integrals and unrestricted orbital inputs accepted by the molecular preparation API
can be used subject to the chosen pool's expressivity and the final sector checks.

## Evolution, spin and stopping criteria

`evolution: exact_commuting` requires pairwise commuting Pauli terms within each
appended generator. Noncommuting external generators must explicitly select
`lie_trotter` or `suzuki`, with `repetitions` and `suzuki_order` (2, 4 or 6).
The selected decomposition is exposed to both simulated and hardware execution.
Approximate product formulas may leak a desired sector; final checks still apply.

N and spin projection are checked using both first and second moments, so a mixture
with the correct mean does not pass. S² and its variance are always reported.
`spin_constraint: require` also checks the configured molecular multiplicity, with
`spin_tolerance`. `report` permits spin mixing but records it. QE and CEO generators
do not generally preserve total S², and a successful inner optimization alone does
not establish the desired spin state.

Growth stops for a small pool gradient (`gradient_norm: l2` or `max`) or a sufficiently
small accepted energy change (`eigenvalue_threshold`; zero disables this check).
An iteration cap, repeated immediate selection, or no candidate above
`selection_threshold` reports nonconvergence. If independent evaluation of a proposed
state raises energy by more than `energy_increase_tolerance`, the previous circuit,
parameters and energy are retained and termination reports `energy_increase_rollback`.
No failed candidate is presented as the final optimized state.

The result's `metadata.solver` records pool identities, per-iteration gradients,
selected blocks, whether a candidate was retained, final sector moments, parameters,
logical circuit metrics and optimizer rebuilding. Callback records identify their
`inner_run`; callback energies exclude nuclear and inactive-space constants. Final
molecular energies include these constants exactly once.

## Resources and examples

`max_pool_size` limits pool entries; pool builders bound generalized enumeration before
materializing it. `max_parameters`, `max_iterations` and `max_evaluations` bound adaptive
growth and recorded energy objective evaluations. `max_product_terms` bounds each
Hamiltonian–generator product before gradient expansion. `max_pauli_terms` limits each
measured observable. `max_measurements` counts gradient, independent energy and sector
estimator publications; it does not count inner optimizer energy or fidelity calls.
Optimizer iteration settings, sampling precision and QNSPSA `fidelity_shots` must also
be budgeted. Resources close on both solver and callback failures.

QNSPSA requires a selected sampler. It rebuilds its fidelity circuit and Hessian state
at each inner solve, with the configured random seed. The upstream `adapt_vqe` still
rejects QNSPSA because that driver does not provide this rebuilding contract.

Runnable examples: `examples/tutorials/qiskit_sp/tetris_adapt.yaml`,
`ceo_adapt.yaml`, and `ceo_qnspsa.yaml`. The QNSPSA example is a stochastic experiment;
it does not promise the deterministic SLSQP example's final accuracy.
