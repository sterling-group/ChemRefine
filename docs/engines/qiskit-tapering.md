# Reference-aware symmetry tapering

The experimental `z2_tapered` molecular mapper reduces Pauli symmetries only after
verifying their eigenvalues in the selected initial state. It applies one
Clifford coordinate transformation to the Hamiltonian, reference, operators and
observables. The underlying encoding can be Jordan–Wigner, Bravyi–Kitaev, or
untapered parity.

```yaml
mapper:
  name: z2_tapered
  options:
    base_mapper: jordan_wigner
    min_qubits: 1
initial_state: hartree_fock
```

The actual transformed orbital occupations determine the Hartree–Fock reference.
An explicit `initial_state: {name: determinant, options: ...}` takes precedence
when selecting the sector. No Aufbau occupation pattern is inferred after an
orbital permutation. The mapper consumes the initial-state selection even for
an exact solver, because that state defines the retained symmetry sector.

## Scope and sector choice

For a Clifford reference, automatic discovery finds the complete Pauli subgroup
shared by the Hamiltonian and that reference's stabilizer group. In reference
coordinates the state is all-zero; a binary nullspace of the Hamiltonian's X
supports determines its compatible Z symmetries. Conjugating them back supports
general X/Y/Z Pauli symmetries, not only particle-parity strings. No Hamiltonian
coefficients are truncated during discovery. Reduction limits can retain only a
subset of the discovered independent generators.

A general non-Clifford reference must supply explicit `generators`. Its
statevector is used to verify each eigenvalue, subject to an allocation guard.
Every generator must be a Hermitian, nonidentity Pauli on the full **base-mapper
register**, commute with the Hamiltonian and the other generators, and be
independent. `sectors`, when supplied, must agree with the actual reference;
they cannot override an incompatible state. Unknown eigenvalues, parameterized
references, measurements, and stochastic reset operations are rejected.

Tapering retains one reference-selected symmetry sector. Its minimum energy is
the minimum within that sector. It is not a guarantee that the molecule's global
ground state lies there, and a spectrum computed entirely in the reduced
register cannot recover roots in omitted sectors. Choose the reference and
retained symmetries for the physical target, especially when comparing
symmetry-changing excited states. The usual particle and total-spin diagnostics
remain separate checks.

## Options

| Option | Default | Effect |
| --- | --- | --- |
| `base_mapper` | `jordan_wigner` | Also accepts `bravyi_kitaev` and `parity`. The parity base is explicitly untapered; two separate reduction mechanisms are not composed implicitly. |
| `generators` | omitted | Automatic compatible-subgroup discovery for a Clifford reference. Otherwise a list of explicit Pauli labels, with Qiskit qubit zero on the right. |
| `sectors` | omitted | Infer each eigenvalue from the actual reference. An explicit list contains one ±1 value per supplied generator and is checked against that reference. |
| `min_qubits` | `1` | Retain at least this many qubits. Zero-qubit SDK paths are not exposed. |
| `max_symmetries` | `32` | Maximum number of independent qubits to remove. Automatic discovery records both the number found and the number used; an oversized explicit list is rejected. |
| `max_qubits` | `128` | Bound the base register before molecular Pauli mapping and Clifford construction. |
| `max_statevector_bytes` | `268435456` | Bound four full-register complex statevector arrays before a general-reference or explicit state transformation. This is not a total process-memory bound. Clifford reference preparation uses polynomial stabilizer algebra and does not allocate those arrays. |
| `tolerance` | `1e-10` | Eigenvalue, normalization, and Hermiticity validation tolerance, restricted to `(0, 1e-6]`. It does not prune Hamiltonian terms or permit a user-selected wrong sector. |

All options reject unknown keys. The component catalog labels this mapper as
experimental and declares its reference-dependent domain.

## Consistent circuits, pools and observables

The shared assembly path reuses the reference actually transformed by the
mapper. It preserves explicit occupation metadata for reference-relative
excitation generation. Reconstructing a different determinant later verifies
that it belongs to the same symmetry sector.

Nature UCC and UCCSD map their excitation operators through the same transform.
Sector-changing generators are excluded with positional placeholders so that
excitation labels and surviving pool entries remain aligned. A pool can become
empty or an ansatz can have no free parameters after reduction; those cases
remain subject to the algorithm's usual validation. Ansatz implementations that
require unreduced occupation qubits do not become taper-compatible merely by
selecting a different mapper.

For evolution generators, `transform.map_operator(full_operator)` returns
`None` if any nonzero Pauli term changes the sector. For observables,
`transform.map_operator(full_operator, check_commutes=False)` instead returns
the projected block `P O P`. Terms changing the sector contribute exactly zero
to expectation values within it; commuting contributions of a mixed observable
are retained. This distinction prevents projection from silently changing a
requested evolution generator.

The mapper supplies `map_observable(fermionic_operator)` for this expectation
semantics. Compose products or response commutators in the full fermionic
algebra before projecting them: in general `P A B P` is not equal to
`(P A P)(P B P)`. Ordinary strict mapping can return an unavailable observable
when it changes sector; do not treat that as a measured zero without the explicit
projection semantics.

Arbitrary externally supplied pools still use their declared prepared mapper
and qubit ordering. To convert a pool expressed in the base encoding, transform
each generator with the helper and explicitly handle rejected or zero entries.
An operator from an unrelated encoding cannot be identified by width alone.

## Python interface and provenance

`build_tapering_transform(hamiltonian, reference_circuit, options=...)` returns a
`TaperingTransform`. Its methods provide:

- `transform_operator`: the full-width Clifford conjugation;
- `map_operator`: conjugation and strict reduction or observable projection;
- `prepare_reference`: sector-validated reduced reference preparation;
- `taper_statevector` and `lift_statevector`: inverse state conversions within
  the byte budget, preserving normalization and relative phases.

For a prepared molecular problem,
`map_problem(prepared, mapper_selection, initial_state=reference_selection)`
uses the same path as the solver. The resulting mapper's
`chemrefine_tapering` attribute exposes the transform, while its `mapper`
attribute is the underlying untapered Nature mapper.

The mapping metadata records the base mapper, selected initial-state component,
signed generators, actual eigenvalues, discovered and used symmetry counts,
retained width, and the Clifford tableau. These records describe the scientific
sector restriction; they do not claim convergence of a variational solver.

`examples/tutorials/qiskit_sp/tapered.yaml` is a complete H₂ VQE pipeline using the
existing thin template. Validate it with:

```bash
chemrefine run examples/tutorials/qiskit_sp/tapered.yaml --dry-run
```

Tests compare exact and VQE H₂ energies for all three base encodings, verify
actual non-Aufbau reference sectors and UCC pool alignment, and test general
non-Z symmetry dynamics and arbitrary observable projection against the full
register.
