# Quantum data and circuit handoffs

The public `chemrefine.engines.qiskit.api` interface exchanges electronic integrals
and bound preparation circuits through versioned JSON/NPZ bundles. Descriptors
record conventions and integrity digests; numeric payloads load without pickle.
Both files must travel together. Relative payload names make a bundle relocatable.

## Molecular integral inputs

Set `options.integral_source.bundle_path` on a molecular `qiskit` step to reuse
integrals without running its PySCF driver. The bundle must include
`MolecularMetadata` with the same ordered atoms, geometry, charge and spin as the
incoming structure, and an explicit nuclear repulsion energy. The default geometry
tolerance is `1e-7` angstrom. `basis` remains a driver option and does not transform
supplied integrals; the bundle declares its own basis and tensor conventions.

`save_integrals` and `load_integrals` are the Python exchange functions. A producing
engine must explicitly supply valid MO tensors; this feature does not extract them
automatically from an ORCA/PySCF output or track orbital changes after a geometry
update. The [integral-input example](https://github.com/Sterling-Group/ChemRefine/tree/main/examples/tutorials/qiskit_integrals)
uses a supplied H₂ bundle and the exact quantum reference solver.

Paths resolve against the YAML file. The scheduler/cache layer hashes both the
descriptor and its referenced payload, so changing only the numeric data changes
the input identity. Copy the whole input/output tree when relocating a run.

## Bound preparation circuits

`bound_circuit(context, circuit, parameters, root=0)` binds a logical circuit using
its original Qiskit parameter order. Supply either the ordered values or a mapping
from the original parameter objects. The returned `BoundCircuit` contains the
bound circuit and its `CircuitDescription`.

```python
from pathlib import Path
from chemrefine.engines.qiskit.api import bound_circuit, load_circuit, save_circuit

# context is a mapped ElectronicStructureContext; circuit is its logical ansatz.
preparation = bound_circuit(context, circuit, optimal_parameters)
save_circuit(Path("root0.circuit.json"), preparation, max_bytes=33554432)
restored = load_circuit(Path("root0.circuit.json"), max_bytes=33554432)
```

Each descriptor preserves parameter names and values, particle populations,
alpha-then-beta spin-orbital ordering, mapper and tapering metadata, active-space
selection, the mapped active Hamiltonian, separately named energy offsets, and
provenance. Qiskit Pauli labels place qubit zero on the right. The saved circuit is
bound and has no classical registers. It describes the logical state before
hardware routing; a later provider can compile it for another layout.

QPY version 13 is stored as a byte array inside the NPZ payload. Its compatibility
is numerically checked from Qiskit 2.5 to the supported Qiskit 1.4 floor. Older
Qiskit may warn about the newer producer version. Artifact recovery verifies the
descriptor, payload digest, byte-array shape and QPY signature without importing
Qiskit. Full circuit deserialization happens only in the selected worker.

For this preparation, the total molecular energy is the expectation of
`description.active_hamiltonian` plus the sum of
`description.energy_offsets.values()`. Add those constants exactly once. A
measurement of another supplied observable does not automatically include them.

## Export and consume in a pipeline

Set `options.circuit_export: {max_bytes: 1048576}` to retain supported preparations.
VQE, ADAPT-VQE, TETRIS-ADAPT, CEO-ADAPT and VQD advertise the `bound_circuit`
capability in their component catalog. VQD exports every computed physical root
after energy ordering. ADAPT exports the retained circuit after any rollback.
Exact, sampled-subspace and qEOM response solvers do not currently provide this
export and reject the option during validation.

The normal molecular worker saves `stepN_ID_inp.rootR.circuit.json` and its NPZ
payload beside the raw result, and records their descriptors in
`engine_metadata.quantum_artifacts`. Scratch copy-back preserves both files.
Completion and cache reuse validate the required bundles locally. A malformed or
missing payload invalidates reuse; parsing and cache rebuilding do not contact a
provider. In the Python API, `run_problem(..., options={"circuit_export": {}})`
returns transient `result.circuits`; call `save_circuit` to persist them.

The [two-step example](https://github.com/Sterling-Group/ChemRefine/tree/main/examples/tutorials/qiskit_handoffs)
solves H₂ from stored integrals, exports its VQE preparation, then measures `IIZZ`
(alpha-particle parity) through `qiskit-experiment`. Run it from its directory with
`chemrefine run input.yaml`. The configured environment must include the selected
providers, or set `backend_python` to a compatible interpreter on both steps.
The artifact step passes molecular structures and their canonical energy through
unchanged; its measured observable lives in the experiment artifact.

Circuit inputs accept either these JSON/NPZ bundles or raw QPY. Typed dependency
discovery includes the bundle's numeric payload. Orbital and Majorana shadow
experiments require a full Jordan–Wigner occupation register and refuse exported
parity/tapered preparations; their inversion cannot infer occupations from a
reduced register. Other experiments retain their own circuit-domain checks, such
as Clifford requirements for endpoint Pauli checks. Raw QPY callers remain
responsible for declaring a compatible scientific interpretation.
