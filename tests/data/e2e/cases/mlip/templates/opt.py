# A short MLIP relaxation: 4 LBFGS steps cannot reach fmax from the stretched seed, so the
# first attempt reports converged = False and the retry continues from its geometry.
import numpy as np
from ase.io import read

from chemrefine.engines.mlip.calculator import MlipCalculator
from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A, HARTREE_TO_EV

mlip = MlipCalculator(
    model_name="$MODEL_NAME",
    task_name="$TASK_NAME",
    device="$DEVICE",
    charge=$CHARGE,
    multiplicity=$MULTIPLICITY,
)
atoms = mlip.optimize(read("$XYZ_PATH"), fmax=0.5, steps=4)
energy_ev, gradient_ev_per_a = mlip.single_point(atoms)

energy_hartree = energy_ev / HARTREE_TO_EV
gradient_hartree_per_bohr = np.asarray(gradient_ev_per_a) / HARTREE_PER_BOHR_TO_EV_PER_A
positions_angstrom = atoms.get_positions()
converged = mlip.last_converged
