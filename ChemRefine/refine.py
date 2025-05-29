import numpy as np
import logging
from .constants import HARTREE_TO_KCAL_MOL, R_KCAL_MOL_K, DEFAULT_TEMPERATURE, DEFAULT_ENERGY_WINDOW, DEFAULT_BOLTZMANN_PERCENTAGE

def filter_structures(coordinates, energies, ids, method, **kwargs):
    if len(coordinates) != len(energies) or len(ids) != len(energies):
        raise ValueError("List length mismatch.")

    energies = np.array([float(e) for e in energies])
    sorted_indices = np.argsort(energies)
    parameters = kwargs.get('parameters', {})

    if method == 'energy_window':
        window = parameters.get('energy', DEFAULT_ENERGY_WINDOW)
        if parameters.get('unit') == 'kcal/mol':
            window /= HARTREE_TO_KCAL_MOL
        min_energy = np.min(energies)
        indices = [i for i in sorted_indices if energies[i] <= min_energy + window]

    elif method == 'boltzmann':
        temperature = DEFAULT_TEMPERATURE
        percent = parameters.get('weight', DEFAULT_BOLTZMANN_PERCENTAGE)
        kcal = energies * HARTREE_TO_KCAL_MOL
        delta_E = kcal - np.min(kcal)
        probs = np.exp(-delta_E / (R_KCAL_MOL_K * temperature))
        probs /= np.sum(probs)
        cum_probs = np.cumsum(probs)
        cutoff = percent / 100
        limit = np.searchsorted(cum_probs, cutoff) + 1
        indices = sorted_indices[:limit]

    elif method == 'integer':
        n = parameters.get('num_structures', len(coordinates))
        indices = sorted_indices[:min(n, len(coordinates))]
    else:
        raise ValueError("Unknown filter method.")

    return [coordinates[i] for i in indices], [ids[i] for i in indices]
