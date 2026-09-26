# Resource-estimation examples

The geometry is a pipeline carrier only. These steps preserve it and write resource
reports, not molecular energies. The Pauli Hamiltonian and factory costs are
illustrative supplied assumptions. DF and THC use exactly factorable toy electronic
integrals, not integrals computed from the carrier geometry.

```bash
chemrefine run examples/tutorials/qiskit_resources/pauli.yaml --dry-run
chemrefine run examples/tutorials/qiskit_resources/pauli.yaml
chemrefine run examples/tutorials/qiskit_resources/surface_code.yaml
```

DF/THC need their optional isolated worker and generated numeric input bundles:

```bash
chemrefine backends install qiskit-resources --python 3.12
python examples/tutorials/qiskit_resources/make_inputs.py
chemrefine run examples/tutorials/qiskit_resources/df.yaml --dry-run
chemrefine run examples/tutorials/qiskit_resources/df.yaml
chemrefine run examples/tutorials/qiskit_resources/thc.yaml
```

The DF example includes Qualtran's analytical cost graph. The THC example uses
OpenFermion's public formula with supplied factors. Neither is an executable
fault-tolerant circuit. See [resource-estimation contracts](../../../docs/engines/qiskit-resources.md)
for error-budget and hardware-model limits. Generated `inputs/` and `outputs-*`
directories may be relocated together with these YAML files.
