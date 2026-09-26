# Local fermionic encodings and flow sets

The lattice API supports graph Bravyi–Kitaev superfast (BKSF), open rectangular
Verstraete–Cirac (VC), and Derby–Klassen (DK) encodings. These are experimental
scientific implementations. They construct actual encoded Hamiltonians,
Clifford reference circuits, physical observables, and evolution circuits.
The [quantum experiment engine](qiskit-experiment.md) exposes the same choices in
YAML and records their trajectories as native artifacts.

## Supported domains

| Mapping | Domain | Encoded register and sector |
| --- | --- | --- |
| `bksf_graph` | Any finite simple undirected support graph, including disconnected graphs and isolated modes. | One qubit per edge; one fixed fermion parity per connected component. Cycles supply code constraints. An edgeless graph uses one explicitly recorded, frozen gauge qubit because the SDK cannot compose zero-qubit Pauli operators. |
| `vc_square` | Open rectangles with at least two rows and columns, optionally a subset of their nearest-neighbor bonds; separate species registers. | One physical and one auxiliary qubit per mode. Both physical parities are retained. Cycle constraints and derived gauge fixers select a complete code state. |
| `dk_square` | The same open rectangles and species convention as VC. | Physical mode qubits plus an auxiliary on each odd checkerboard face. The odd faces are chosen as the larger color class when the counts differ, retaining both physical parities; any extra gauge degree is fixed explicitly. |

A spinful lattice orders all alpha sites before all beta sites. Onsite Hubbard
terms and intersite density interactions can couple the two registers. An
operator transferring a fermion between disconnected encoding components is
rejected. To represent such transport with BKSF, include connecting support edges
in the encoding graph. Square encodings do not accept periodic wraparound or
non-nearest-neighbor support bonds; choose a general graph encoding instead.

All three encodings map number-conserving fermionic observables, including
complex hopping, currents, and quartic terms. Nonlocal observables within a
connected component use products along graph paths. Pairing operators are
outside this public interface. The lattice model uses a shared hopping amplitude
for its two species; arbitrary spin mixing is not implied by `spinful: true`.

## Configuration

For a square selection, the mapping and nested encoding name must agree:

```yaml
dynamics:
  mapping: vc_square
  encoding: {name: vc_square, rows: 2, columns: 2}
  synthesis: flow_sets
  steps: 4
  order: 2
  exact_reference: true
  max_qubits: 12
  max_exact_qubits: 12
```

The graph mapping can be selected with `mapping: bksf_graph` alone. Its optional
nested `encoding` controls explicit component parities and construction budgets.
Components are ordered by their smallest mode index. Omitted parities are inferred
from `occupied_modes`, including odd-particle references; supplied parities must
agree with those actual occupations.

| Control | Default | Effect and validity |
| --- | --- | --- |
| `mapping` | `jordan_wigner` | Also accepts `bravyi_kitaev`, `parity`, and the three local names above. |
| `encoding` | omitted | Consumed only by a local mapping. A mismatched name or a nested encoding with an ordinary mapper is rejected. |
| `encoding.rows`, `encoding.columns` | omitted | Required together for VC/DK; their product must equal `model.num_sites`. Rejected for graph BKSF. |
| `encoding.component_parities` | inferred from reference | A list of 0/1 values for graph BKSF. Rejected for square mappings, which retain both parities. |
| `encoding.max_qubits` | `128` | Construction guard, additionally bounded by `dynamics.max_qubits`. |
| `encoding.max_generators` | `2048` | Limits the total vertex/edge generators before constructing their Pauli representation. Square encodings count the complete rectangle, even when physical bonds are omitted. |
| `encoding.max_expanded_terms` | `100000` | Caps the prospective Majorana expansion of each mapped fermionic operator. |
| `synthesis` | `pauli` | `pauli` evolves Hermitian physical hopping/current/density blocks. `flow_sets` groups directed transfers and requires a local mapping. |
| `steps`, `order` | `1`, `2` | Product-formula step count and order 1, 2, or 4; negative fourth-order substeps are retained. |
| `max_evolution_blocks` | `10000` | Limits the expanded product-formula sequence, including repeated Suzuki blocks. |
| `max_qubits` | `16` | Limits the actual encoded register, including auxiliary qubits. |
| `max_statevector_bytes` | `268435456` | Bounds four complex statevector arrays; this is not a total process-memory guarantee. |
| `exact_reference`, `max_exact_qubits` | `false`, `12` | Optional sparse exact evolution for a fidelity check, subject to the actual encoded width. |

Each lattice edge has real `hopping` (default 1) and `hopping_imag` (default 0).
Writing `t = hopping + i*hopping_imag`, the contribution is
`-t a†_source a_target - conjugate(t) a†_target a_source` for each species.
Changing the written edge orientation conjugates the physical amplitude. Each
undirected edge must appear once. All numeric model parameters must be finite.
The chain and square Python helpers also accept `hopping_imag`.

## Approximation and diagnostics

With `synthesis: pauli`, the real hopping, imaginary current, and density terms
form separate Hermitian blocks. Each block is synthesized exactly, preserves
particle number, and preserves the encoding constraints. Splitting different
blocks introduces product-formula error.

With `synthesis: flow_sets`, real hopping is divided into directed transfers.
These commute within each selected flow. VC uses its four cardinal transfer
families and verified Clifford changes of basis; BKSF and DK use deterministic
matching groups on source and target copies of the graph. Generic groups need
not use the minimum number of colors. Imaginary hopping remains a separate
physical current block.

Individual directed transfers need not conserve particle number. A finite-step
flow formula can therefore exhibit particle-number drift even for a
number-conserving Hamiltonian. Results report that drift, energy drift, the code
and gauge constraint expectations, and optional exact-state fidelity. Refining
the product formula controls this approximation; constraint preservation alone
does not establish its accuracy.

VC's changes of basis use two entangling layers on the abstract encoding
connectivity. Reference preparation and hardware routing have separate costs.
Reported circuit depth is before routing. Neither arbitrary graphs nor routed
circuits are promised constant depth. These experiment trajectories use bounded
ideal local statevector simulation; selecting a local encoding does not submit a
hardware job or add noise mitigation.

## Python extension boundary

`build_local_encoding(num_modes, edges, options=..., occupied_modes=...)` returns
a `FermionicEncoding`. Its `map_operator` consumes the released Qiskit Fermions
`FermionOperator`; `prepare_reference` builds a Clifford circuit from actual
occupations. `decode_occupations` interprets canonical Qiskit bitstrings with
qubit zero on the right through the encoded vertex observables. Custom encodings
with non-diagonal occupations must perform a measurement basis change first.

`validate_encoding(vertices, edges, component_parities=...)` accepts custom
Hermitian unit Pauli generators. It checks edge/vertex commutation relations,
independent physical occupation sectors, signed cycle constraints, and declared
parities. Gauge fixing comes from the commutant of the full represented algebra,
so it preserves all represented observables. Reference preparation requires a
complete consistent set of constraints and never uses exponential stabilizer
projection. A custom encoding still needs its own domain and resource analysis.

`lattice_encoding_qubits(model, integrator_options)` computes the register size
without loading a quantum SDK. The experiment engine uses this value for both
per-state and aggregate trajectory budgets. Molecular single-point mapper
registries remain separate from this lattice encoding API.

## Examples and validation

Complete runnable pipelines share the existing `qiskit_experiment` tutorial's
molecule and scheduler header:

- `examples/tutorials/qiskit_experiment/bksf_graph.yaml`: a triangle with complex
  hopping, a fixed odd-particle sector, and generic flow synthesis.
- `examples/tutorials/qiskit_experiment/vc_square.yaml`: open 2-by-2 VC flow sets.
- `examples/tutorials/qiskit_experiment/dk_square.yaml`: the compact DK rectangle
  with physical-block synthesis.

The structures are pass-through pipeline inputs; they do not define these lattice
Hamiltonians. Validate from the repository root, for example:

```bash
chemrefine run examples/tutorials/qiskit_experiment/vc_square.yaml --dry-run
```

Tests compare encoded dynamics with independent Jordan–Wigner results for
complex hopping and quartic observables across particle sectors, check signed
code constraints and reference preparation, and verify flow refinement and
allocation failures.

The construction follows the [BKSF edge/vertex algebra](https://arxiv.org/abs/1810.05274),
[DK compact mapping](https://arxiv.org/abs/2003.06939),
[Qiskit Fermions VC guide](https://qiskit.github.io/qiskit-fermions/stable/0.1/guides/2d_fermi_hubbard.html),
and [flow-set synthesis paper](https://arxiv.org/abs/2512.11418).
