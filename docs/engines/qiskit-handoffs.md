# Quantum data and circuit handoffs

The public `chemrefine.engines.qiskit.api` interface exchanges electronic integrals
and bound preparation circuits through versioned JSON/NPZ bundles. Descriptors
record conventions and integrity digests; numeric payloads load without pickle.
Both files must travel together. Relative payload names make a bundle relocatable.

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
