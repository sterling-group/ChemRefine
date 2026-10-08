# Reproducible Qiskit calculations and checkpoints

Save the molecular-orbital integrals **before** solving, then save the final bound
logical circuit and result diagnostics. Replaying those artifacts preserves the
Hamiltonian and state representation used for the calculation. Geometry, basis,
seeds and final parameters alone are insufficient: a fresh SCF calculation can
change orbital phases, ordering or rotations within a degenerate subspace. The
old parameters then refer to a different orbital frame.

These checkpoints support repeating a solve and independently checking a final
state's energy. They do not resume the internal state of a classical optimizer.
The APIs below belong to `chemrefine.engines.qiskit.api`; see also
[quantum data handoffs](qiskit-handoffs.md) for their artifact contracts.

## Save the original integral input

`save_integrals(path, data)` accepts `ElectronicStructureData`, not
`PreparedProblem`. Save the producer's original orthonormal MO tensors before
calling `prepare_problem`, applying frozen-core/active-space transformations, or
optimizing orbitals. `run_job` and `prepare_pyscf_problem` do not automatically
export their integrals. A caller using the XYZ driver must explicitly obtain and
package the producer's MO tensors, or start from an existing integral bundle.

The following small molecular example uses the optional PySCF installation to
create the initial checkpoint. Run its Python blocks in order from one directory;
all generated artifacts go into `checkpoints/`.

```python
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pyscf import ao2mo, gto, scf

from chemrefine.engines.qiskit.api import (
    ElectronicStructureData,
    MolecularMetadata,
    save_integrals,
)

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)
symbols = ("H", "H")
coordinates = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.735))
basis = "sto-3g"
mol = gto.M(
    atom=list(zip(symbols, coordinates)), basis=basis,
    unit="Angstrom", charge=0, spin=0, verbose=0,
)
mean_field = scf.RHF(mol)
mean_field.conv_tol = 1e-12
mean_field.kernel()
if not mean_field.converged:
    raise RuntimeError("SCF did not converge; inspect the preparation")

coefficients = mean_field.mo_coeff
orbital_count = coefficients.shape[1]
data = ElectronicStructureData(
    num_alpha=mol.nelec[0], num_beta=mol.nelec[1],
    num_spatial_orbitals=orbital_count,
    one_body_integrals=coefficients.conj().T @ mean_field.get_hcore() @ coefficients,
    two_body_integrals=ao2mo.kernel(mol, coefficients, compact=False).reshape(
        (orbital_count,) * 4
    ),
    two_body_order="chemist",
    nuclear_repulsion_energy=float(mol.energy_nuc()),
    orbital_energies=mean_field.mo_energy,
    orbital_occupations=np.asarray(mean_field.mo_occ) / 2,
    orbital_occupations_beta=np.asarray(mean_field.mo_occ) / 2,
    multiplicity=1,
    molecular_metadata=MolecularMetadata(symbols, coordinates, charge=0),
    provenance={"producer": "PySCF RHF", "producer_version": version("pyscf")},
    metadata={"basis": basis, "orbital_representation": "orthonormal MO"},
)
save_integrals(checkpoint_dir / "molecule.integrals.json", data)
```

This producer example is for real restricted orbitals. The integral API also
supports complex Hermitian and unrestricted inputs; unrestricted data requires
all beta integral blocks and the alpha-beta orbital overlap for spin observables.
The overlap array is currently real; general complex alpha/beta orbital overlaps
are outside the supported input contract. Occupations are `0`/`1` per spin,
rather than PySCF's restricted `0`/`2` total occupations; the example divides the
RHF occupations accordingly.
Declare `two_body_order` explicitly and preserve alpha-then-beta spin-orbital
ordering, occupations, particle counts and multiplicity.

Keep nuclear repulsion in `nuclear_repulsion_energy`. For already reduced inputs,
put electronic constants excluded from their tensors in `energy_offsets`.
Preparation namespaces them as `input:<name>` and adds any new transformation
constants separately. Do not apply the same reduction twice or put a constant
both into the tensors and into an offset. If only reduced tensors are saved,
the omitted orbitals cannot subsequently be recovered from that checkpoint.

The tensors preserve the MO Hamiltonian representation but do not include an
AO-to-MO coefficient matrix or arbitrary property integrals. Archive those
separately when later work needs spatial observables or a relationship to the
original atomic orbitals. Save each deliberately changed orbital frame as a new
checkpoint with its transformation recorded; keep the original checkpoint.

## Reload, optimize, and retain the final state

This block starts from the saved integrals and does not run SCF. Active-space
indices refer to the original input spatial orbitals, even with frozen core.
For a larger input, choose and record an appropriate reduction explicitly.

```python
import json
from pathlib import Path

from chemrefine.engines.qiskit.api import (
    QiskitOptions,
    load_integrals,
    prepare_problem,
    run_problem,
    save_circuit,
)

checkpoint_dir = Path("checkpoints")
options = QiskitOptions.from_raw({
    "algorithm": "vqe",
    "mapper": "jordan_wigner",
    "ansatz": "uccsd",
    "initial_state": "hartree_fock",
    "initial_point": "zeros",
    "estimator": {
        "name": "statevector",
        "options": {"default_precision": 0.0, "seed": 1234},
    },
    "optimizer": {
        "name": "slsqp",
        "options": {"maxiter": 200, "ftol": 1e-10},
    },
    "circuit_export": {"max_bytes": 33554432},
})
restored_data = load_integrals(checkpoint_dir / "molecule.integrals.json")
prepared = prepare_problem(
    restored_data, active_space=options.active_space, freeze_core=options.freeze_core,
)
result = run_problem(prepared, options=options)
save_circuit(checkpoint_dir / "ground-state.circuit.json", result.circuits[0])
(checkpoint_dir / "ground-state.result.json").write_text(
    json.dumps(result.as_metadata(), indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
(checkpoint_dir / "solver-options.json").write_text(
    json.dumps(options.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)
```

`run_problem` returns circuits in memory only when `circuit_export` is enabled.
Call `save_circuit` to persist them. `result.as_metadata()` deliberately excludes
the transient SDK circuits and numerical states; saving only that JSON does not
save a state checkpoint. Its component diagnostics resolve default component
options and retain the execution environment and source provenance.

`circuit_export` is supported by VQE, ADAPT-VQE, TETRIS-ADAPT, CEO-ADAPT and VQD.
VQD retains each physical root after energy ordering. Exact, sampled-subspace
and qEOM solvers currently reject this option. Sampled-subspace states have their
own [state bundles](qiskit-states.md).

## Independently reconstruct the saved energy

The circuit bundle includes the original mapped active Pauli Hamiltonian and
separately named energy offsets. Check the final energy directly against these
saved quantities instead of rebuilding SCF or inferring a Hamiltonian from a
new geometry calculation.

```python
import json
from pathlib import Path

from qiskit.quantum_info import SparsePauliOp, Statevector

from chemrefine.engines.qiskit.api import load_circuit

checkpoint_dir = Path("checkpoints")
saved = load_circuit(checkpoint_dir / "ground-state.circuit.json")
hamiltonian = SparsePauliOp.from_list(list(saved.description.active_hamiltonian.items()))
state = Statevector.from_instruction(saved.circuit)
active_energy = complex(state.expectation_value(hamiltonian))
if abs(active_energy.imag) > 1e-10:
    raise RuntimeError("Expected a real energy for the saved Hermitian Hamiltonian")
energy_hartree = active_energy.real + sum(saved.description.energy_offsets.values())
recorded = json.loads((checkpoint_dir / "ground-state.result.json").read_text())
error_hartree = energy_hartree - recorded["result"]["energy_hartree"]
assert abs(error_hartree) < 1e-8, error_hartree
print(f"Replayed energy: {energy_hartree:.12f} hartree")
```

This check uses exact ideal statevector expectations for the deterministic
example above. With finite shots, nonzero estimator precision, or a noise model,
the reported estimate can differ from the ideal replay. Use a justified
uncertainty tolerance for a statistical estimate and record that distinction.
Replay is a total molecular energy only if nuclear repulsion is present in the
saved offsets. Add every offset once. Qiskit Pauli labels place qubit zero on
the right; retain the saved mapping and tapering metadata.

Energy replay alone does not establish particle-sector or spin correctness,
optimizer convergence, or chemical accuracy. Save separately computed particle
and spin diagnostics with their target values and tolerances. The ordinary VQE
result JSON currently does not retain the Nature spin auxiliary results, and
the circuit bundle does not contain those observables. Reconstruct them from the
same saved integrals and mapping when validating the state; unavailable
diagnostics must remain explicitly unknown.

## Use the same checkpoints in a pipeline

The following is an `options` fragment for a molecular `qiskit` step. Paths are
resolved against its YAML file. The incoming structure must have matching
ordered atoms, geometry, charge and multiplicity; its bundle must include
`MolecularMetadata` and nuclear repulsion. The default geometry tolerance is
`1e-7` angstrom. `basis` does not transform supplied integral tensors.

```yaml
options:
  integral_source:
    bundle_path: checkpoints/molecule.integrals.json
    max_input_bytes: 268435456
  algorithm: vqe
  mapper: jordan_wigner
  ansatz: uccsd
  initial_state: hartree_fock
  initial_point: zeros
  estimator:
    name: statevector
    options:
      default_precision: 0.0
      seed: 1234
  optimizer:
    name: slsqp
    options:
      maxiter: 200
      ftol: 1.0e-10
  circuit_export:
    max_bytes: 33554432
```

The normal worker calls `run_job(..., artifact_dir=...)`, which saves each
retained circuit bundle and records its descriptor in
`engine_metadata.quantum_artifacts`. A direct `run_job` caller must provide
`artifact_dir` for automatic persistence. The pipeline hashes both the input
descriptor and referenced payload for cache identity, preserves output bundles
on scratch copy-back, and validates required bundles before cache reuse.
A missing or corrupt required payload invalidates reuse. Loading or validating
local bundles does not contact a provider.

## Archive and recovery boundaries

Each integral or circuit bundle has a JSON descriptor and a uniquely named NPZ
payload beside it. Preserve **both files**, using the descriptor's `payload`
field rather than guessing its filename. Readers verify its SHA-256 digest,
array dimensions, numeric conventions and finite values without pickle. The
writer publishes the descriptor last, keeping the previous complete artifact
usable if a new write fails. Payload digests detect corruption; they do not
authenticate the producer or prove scientific correctness.

Integral save/load defaults to a 256 MiB uncompressed array budget. Pipeline
`integral_source.max_input_bytes` defaults to 512 MiB; set explicit compatible
budgets at both boundaries when needed. Circuit save/load and export default
to 32 MiB per root for QPY plus Hamiltonian and parameter arrays. Descriptors
have a separate 1 MiB limit. Fixed interpretation sizes can be checked before
optimization; adaptive circuit growth and final QPY size remain runtime checks.
These limits do not reserve scheduler memory.

Circuit payloads use QPY version 13. Interpretation storage version 2 keeps large
Hamiltonians and parameter bindings in numeric arrays; readers also accept
version 1 interpretation data. Use the supported Qiskit package ranges and keep
the producer and replay environments recorded. A valid bundle cannot guarantee
compatibility with arbitrary future SDK versions or custom circuit instructions.

Keep the result JSON, resolved options, original geometry, integral producer
settings, basis and active-orbital choice alongside the bundles. Set every
relevant seed explicitly: random initialization, stochastic optimizer, estimator,
sampler, simulator and transpiler. Record precision, shots/noise settings,
provider/backend choices, package versions and source revision. Execution
provenance records selected dependency families, Python/platform, ChemRefine
version and source identity; circuit provenance carries preparation facts and
is not a substitute for the full result metadata. A Git dirty flag does not
archive the changes, so retain the patch or exact source snapshot separately.
Fixed seeds, a fixed Hamiltonian and a recorded software stack support
repeatability; they do not guarantee bitwise equality across platforms or
provider calibrations.

| Saved material | Working recovery capability | Current limit |
| --- | --- | --- |
| Original integral bundle | Prepare and solve the same MO Hamiltonian without fresh SCF | No automatic integral export from the XYZ workflow; absent original orbitals/property integrals cannot be recovered |
| Bound circuit bundle | Reconstruct its ideal state and energy, or submit it to a compatible later experiment | Logical preparation only; no hardware calibration snapshot or transpiled layout |
| Parameter bindings and identical fixed ansatz | Start a new VQE with `run_problem(..., initial_point=values)` after checking parameter order | A new optimization; momentum, Hessian estimates, RNG state and other optimizer internals are not restored |
| Completed pipeline outputs | Validate and reuse a complete cached result | Cache reuse is not an in-progress optimizer checkpoint |
| Provider job journal | Inspect durable submission records and explicitly retrieve known jobs when supported | Does not preserve optimizer state or guarantee exactly-once execution; see [provider recovery](qiskit-recovery.md) |

There is currently **no automatic full optimizer checkpoint or continuation**.
Evaluation callbacks record objective values and metadata, not a complete
parameter trajectory. Adaptive replay does not reconstruct the optimizer's
growth/selection state for continuation. Preserve the integral and final-state
checkpoints for independent review and deliberate new runs.
