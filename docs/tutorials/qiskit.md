# Qiskit molecular energies

Run the shipped H2 workflow to calculate a molecular single-point energy with
Qiskit Nature. The geometry stays fixed; the classical PySCF preparation constructs
the electronic Hamiltonian and VQE optimizes its quantum circuit parameters.
This example uses exact local statevector expectations, with no shot noise or
hardware submission.

## Install the molecular stack

From a source checkout, install the Qiskit extra into the Python environment you
will use to launch ChemRefine:

```bash
python -m pip install -e '.[qiskit]'
```

Alternatively, keep the scientific stack in a managed worker:

```bash
chemrefine backends install qiskit
```

The main environment needs ChemRefine installed either way. See
[installation](../engines/installing.md#qiskit-nature) for Aer, fermionic and Runtime
profiles and their Python requirements.

## Use the shipped files

Work from `examples/tutorials/qiskit_sp/` in the checkout. It contains the H2
geometry, `templates/step1.py`, a local CPU header and this complete configuration:

```yaml
--8<-- "examples/tutorials/qiskit_sp/input.yaml"
```

The two-electron, two-orbital active space maps to four Jordan–Wigner qubits.
The `uccsd` circuit starts from Hartree–Fock, with its variational parameters
initialized to zero;
SLSQP searches for its ground-state energy. The named components and their
settings live in YAML, while the worker template calls the shared runner.

## Validate and run

```bash
cd examples/tutorials/qiskit_sp
chemrefine validate input.yaml
chemrefine run input.yaml --dry-run
chemrefine run input.yaml
```

`dispatch: local` runs the generated job script through Bash. To use SLURM, choose
`dispatch: slurm` or `auto` and supply headers for your cluster; see
[local and cluster execution](../running/clusters.md).

Read the final total molecular energy in `outputs/steps.csv`. The worker's
`outputs/step1_vqe/0/step1_0.json` retains `engine_metadata`, including component
choices and solver diagnostics; `step1_0.result.json` is the canonical pipeline
record. Callback objective values exclude scalar energy constants, so compare
the reported total energies rather than mixing callbacks with molecular results.

## Compare exact and adaptive solvers

The same directory includes a comparison workflow that prepares the molecular
problem once and solves it with exact diagonalization, UCCSD-VQE and ADAPT-VQE:

```bash
chemrefine validate compare.yaml
chemrefine run compare.yaml --dry-run
chemrefine run compare.yaml
```

The [molecular guide](../engines/qiskit.md#install-and-run-the-example) explains
the reported comparison and energy conventions. Agreement on this small H2 case
does not establish convergence or chemical accuracy for a larger molecule or
active space.

## Keep a reproducible checkpoint

Keep the config, geometry, exact package versions and source revision. Follow the
[checkpoint guide](../engines/qiskit-checkpoints.md) to save the MO integrals and
export supported final bound circuits or determinant states, then use the
[handoff guide](../engines/qiskit-handoffs.md) to reuse them in later experiments.
A retained state is a final preparation; it does not resume an optimizer's internal
state or replace the full molecular Hamiltonian record.

Use [support status](../engines/qiskit-status.md) before changing providers or
methods. The molecular engine has no geometry optimization, gradients, frequencies
or NMS. Additional [quantum experiments](../engines/qiskit-experiment.md) produce
artifact bundles while passing the pipeline structures through unchanged.
