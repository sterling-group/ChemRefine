# Direct-MLIP capture step: optimise each seed frame with MACE-OFF on CPU.
from ase.io import read
from ase.units import Hartree

from chemrefine.engines.mlip.calculator import MlipCalculator

mlip = MlipCalculator(model_name="$MODEL_NAME", task_name="$TASK_NAME", device="$DEVICE")
atoms = mlip.optimize(read("$XYZ_PATH"), fmax=0.05)

energy_hartree = atoms.get_potential_energy() / Hartree
positions_angstrom = atoms.get_positions()
