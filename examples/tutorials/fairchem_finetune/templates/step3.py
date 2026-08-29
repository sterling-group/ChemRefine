# Run the model step 2 just trained.
import numpy as np
from ase.io import read

from chemrefine.engines.mlip.calculator import MlipCalculator
from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A, HARTREE_TO_EV

mlip = MlipCalculator(
    task_name="$TASK_NAME", model_name="", model_path="$MODEL_PATH", device="$DEVICE"
)
atoms = read("$XYZ_PATH")
energy_ev, gradient_ev_per_a = mlip.single_point(atoms)

energy_hartree = energy_ev / HARTREE_TO_EV
gradient_hartree_per_bohr = np.asarray(gradient_ev_per_a) / HARTREE_PER_BOHR_TO_EV_PER_A
positions_angstrom = atoms.get_positions()
