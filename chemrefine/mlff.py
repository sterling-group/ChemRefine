import logging
from typing import List, Tuple


def run_mlff_calculation(
    xyz_path: str,
    model_name: str = "mol",
    device: str = "cpu",
    fmax: float = 0.03,
    steps: int = 200,
) -> Tuple[List[List], float]:
    """Optimize geometry using farichem-core.

    Parameters
    ----------
    xyz_path : str
        Path to an XYZ file containing the starting geometry.
    model_name : str, optional
        Name of the MLFF model to load. Defaults to ``"mol"``.
    device : str, optional
        Device for model evaluation ("cpu" or "cuda"). Defaults to "cpu".
    fmax : float, optional
        Force convergence criterion for the optimiser. Defaults to ``0.03``.
    steps : int, optional
        Maximum optimisation steps. Defaults to ``200``.

    Returns
    -------
    tuple
        Optimised coordinates ``[[symbol, x, y, z], ...]`` and the energy in Hartree.
    """
    try:
        from ase.io import read
        from ase.optimize import BFGS
        from fairchem.core import models
    except ImportError as exc:  # pragma: no cover - dependency missing at runtime
        raise ImportError(
            "MLFF calculations require the 'fairchem-core' and 'ase' packages"
        ) from exc

    logging.info(f"Loading MLFF model '{model_name}' on {device}.")
    atoms = read(xyz_path)
    model = models.load_model(model_name=model_name, device=device)
    atoms.calc = model.get_calculator()

    optimizer = BFGS(atoms, logfile=None)
    optimizer.run(fmax=fmax, steps=steps)

    energy_ev = atoms.get_potential_energy()
    energy_hartree = energy_ev / 27.211386245988
    coords = [
        [atom.symbol, atom.position[0], atom.position[1], atom.position[2]]
        for atom in atoms
    ]
    return coords, energy_hartree
