# Direct-MLIP screening step. ChemRefine renders this per structure:
# $XYZ_PATH / $CHARGE / $MULTIPLICITY come from the pipeline, and
# $MODEL_NAME / $TASK_NAME / $DEVICE from the YAML step options.
# Assign `energy_hartree` (optionally `positions_angstrom` /
# `gradient_hartree_per_bohr` / `converged`) — the appended output footer harvests them.
from ase.io import read
from ase.units import Hartree

from chemrefine.engines.mlip.calculator import MlipCalculator

mlip = MlipCalculator(model_name="$MODEL_NAME", task_name="$TASK_NAME", device="$DEVICE")
atoms = mlip.optimize(read("$XYZ_PATH"), fmax=0.03)

energy_hartree = atoms.get_potential_energy() / Hartree
positions_angstrom = atoms.get_positions()
# False when LBFGS ran out of steps before reaching fmax: ledgered as a convergence
# failure and retried from this geometry, instead of ranking as a survivor.
converged = mlip.last_converged
