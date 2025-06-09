import logging
from typing import List, Tuple


def get_available_device() -> str:
    """Return ``"cuda"`` if a GPU is available, otherwise ``"cpu"``."""
    try:  # pragma: no cover - optional dependency
        import torch

        if torch.cuda.is_available():
            logging.info("GPU detected for MLFF calculations.")
            return "cuda"
    except Exception:
        pass

    logging.info("No GPU detected; falling back to CPU for MLFF calculations.")
    return "cpu"


def run_mlff_calculation(
    xyz_path: str,
    model_name: str = "mol",
    device: str | None = None,
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
    device : str or None, optional
        Device for model evaluation ("cpu" or "cuda"). If ``None``,
        ``get_available_device()`` is used to automatically select the
        device.
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
        from ase.optimize import LBFGS
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
    except ImportError as exc:  # pragma: no cover - dependency missing at runtime
        raise ImportError(
            "MLFF calculations require the 'fairchem-core' and 'ase' packages"
        ) from exc

    if device is None:
        device = get_available_device()

    logging.info(f"Loading MLFF model '{model_name}' on {device}.")
    atoms = read(xyz_path)
    predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
    calc = FAIRChemCalculator(predictor, task_name="oc20")
    atoms.calc = calc

    optimizer = LBFGS(atoms, logfile=None)
    optimizer.run(fmax=fmax, steps=steps)

    energy_ev = atoms.get_potential_energy()
    energy_hartree = energy_ev / 27.211386245988
    coords = [
        [atom.symbol, atom.position[0], atom.position[1], atom.position[2]]
        for atom in atoms
    ]
    return coords, energy_hartree
