# Circuit cutting examples

Generate the bound three-qubit GHZ circuit, then choose a decomposition:

```bash
chemrefine backends install qiskit-cutting
python examples/tutorials/qiskit_cutting/make_inputs.py
chemrefine run examples/tutorials/qiskit_cutting/manual.yaml --dry-run
chemrefine run examples/tutorials/qiskit_cutting/manual.yaml
chemrefine run examples/tutorials/qiskit_cutting/wire.yaml
chemrefine run examples/tutorials/qiskit_cutting/partition.yaml
chemrefine run examples/tutorials/qiskit_cutting/automatic.yaml
```

Each example samples partitions of at most two qubits. The uncut ideal observable
has expectation 2; finite-shot and sampled-QPD estimates fluctuate. QPD
`num_samples` and physical `shots` are separate controls. Results retain unsigned
physical measurement bytes and signed reconstruction weights.

The supplied geometry only carries the pipeline's structure records. No quantum
chemistry energy is calculated from it. See the
[cutting contract](../../../docs/engines/qiskit-cutting.md) for limitations and
resource guards.
