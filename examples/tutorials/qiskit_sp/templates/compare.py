import json

from chemrefine.engines.qiskit.api import (
    prepare_pyscf_problem,
    run_adapt_vqe,
    run_vqe,
    solve_exact,
)
from chemrefine.engines.qiskit.options import QiskitOptions

options = QiskitOptions.from_raw(json.loads("$OPTIONS_JSON"))
problem = prepare_pyscf_problem(
    "$XYZ_PATH",
    charge=int("$CHARGE"),
    multiplicity=int("$MULTIPLICITY"),
    options=options,
)
exact = solve_exact(problem, options=options)
vqe = run_vqe(problem, options=options, reference_energy_hartree=exact.energy_hartree)
adapt = run_adapt_vqe(problem, options=options, reference_energy_hartree=exact.energy_hartree)

print(f"Exact energy: {exact.energy_hartree:.12f} hartree")
print(f"VQE energy: {vqe.energy_hartree:.12f} hartree")
print(f"VQE error: {vqe.energy_error_hartree:.3e} hartree")
print(f"ADAPT-VQE energy: {adapt.energy_hartree:.12f} hartree")
print(f"ADAPT-VQE error: {adapt.energy_error_hartree:.3e} hartree")
print(f"Qubits: {adapt.num_qubits}")
print(f"Pauli terms: {adapt.num_pauli_terms}")
print(f"UCCSD parameters: {vqe.parameter_count}")
print(f"ADAPT operators selected: {len(adapt.adapt_selected_operators or ())}")

energy_hartree = adapt.energy_hartree
engine_metadata = adapt.as_metadata()
engine_metadata["comparison"] = {
    "exact": exact.as_dict(),
    "vqe": vqe.as_dict(),
    "adapt_vqe": adapt.as_dict(),
}
if adapt.converged is not None:
    converged = adapt.converged
