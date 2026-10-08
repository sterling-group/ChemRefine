import json

from chemrefine.engines.qiskit.workflow import run_job

result = run_job(
    "$XYZ_PATH",
    charge=int("$CHARGE"),
    multiplicity=int("$MULTIPLICITY"),
    options=json.loads("$OPTIONS_JSON"),
    artifact_dir=".",
)

energy_hartree = result.energy_hartree
engine_metadata = result.as_metadata()
if result.converged is not None:
    converged = result.converged
