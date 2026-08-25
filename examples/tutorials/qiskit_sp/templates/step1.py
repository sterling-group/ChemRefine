import json

from chemrefine.engines.qiskit.workflow import run_job

result = run_job(
    "$XYZ_PATH",
    charge=int("$CHARGE"),
    multiplicity=int("$MULTIPLICITY"),
    options=json.loads("$QISKIT_OPTIONS_JSON"),
)

energy_hartree = result.energy_hartree
engine_metadata = result.metadata
