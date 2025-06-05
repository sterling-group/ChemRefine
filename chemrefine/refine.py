import numpy as np
import logging
from .constants import HARTREE_TO_KCAL_MOL, R_KCAL_MOL_K, DEFAULT_TEMPERATURE

class StructureRefiner:
    def filter(self, coordinates, energies, ids, method, parameters):
        """
        Filters structures based on the provided method while preserving original ID order.

        Args:
            coordinates (list): List of coordinate blocks (parsed from ORCA output).
            energies (list): List of energies corresponding to each structure.
            ids (list): List of IDs corresponding to each structure.
            method (str): Filtering method ('energy_window', 'boltzmann', 'integer').
            parameters (dict): Additional parameters for filtering methods.

        Returns:
            tuple: (filtered_coordinates, filtered_ids)
        """
        if len(coordinates) != len(energies) or len(energies) != len(ids):
            raise ValueError(
                f"Mismatch in list lengths: coordinates ({len(coordinates)}), "
                f"energies ({len(energies)}), and ids ({len(ids)}) must have the same length."
            )

        # Defensive check for empty energies
        if not energies or all(e is None for e in energies):
            logging.warning("No valid energies found to filter. Returning empty results.")
            return [], []

        energies = np.array([float(e) for e in energies])
        sorted_indices = np.argsort(energies)

        if method == 'energy_window':
            return self._filter_energy_window(coordinates, energies, ids, sorted_indices, parameters)
        elif method == 'boltzmann':
            return self._filter_boltzmann(coordinates, energies, ids, sorted_indices, parameters)
        elif method == 'integer':
            return self._filter_integer(coordinates, energies, ids, sorted_indices, parameters)
        else:
            raise ValueError("Invalid method. Choose from 'energy_window', 'boltzmann', or 'integer'.")

    def _filter_energy_window(self, coordinates, energies, ids, sorted_indices, parameters):
        logging.info("Filtering structures based on energy window.")
        energy = parameters.get('energy', 0.5)  # Default is 0.5 Hartrees
        unit = parameters.get('unit', 'hartree').lower()
        logging.info(f"Filtering Energy window: {energy} {unit}")

        if unit == 'kcal/mol':
            energy /= HARTREE_TO_KCAL_MOL
            logging.info(f"Converted energy window to Hartrees: {energy:.6f}")

        min_energy = np.min(energies)
        favored_indices = [i for i in sorted_indices if energies[i] <= min_energy + energy]

        return self._apply_mask(coordinates, ids, favored_indices)

    def _filter_boltzmann(self, coordinates, energies, ids, sorted_indices, parameters):
        logging.info("Filtering structures based on Boltzmann probability.")
        if len(energies) == 0:
            logging.warning("No structures available for Boltzmann filtering. Returning empty lists.")
            return [], []

        if len(energies) == 1:
            logging.info("Only one structure available; returning it.")
            return coordinates, ids

        temperature = parameters.get('temperature', DEFAULT_TEMPERATURE)
        percentage = parameters.get('weight', 99)
        logging.info(f"Filtering Boltzmann probability: {percentage}% at {temperature} K.")

        energies_kcalmol = energies * HARTREE_TO_KCAL_MOL
        min_energy = np.min(energies_kcalmol)
        delta_E = energies_kcalmol - min_energy

        boltzmann_weights = np.exp(-delta_E / (R_KCAL_MOL_K * temperature))
        boltzmann_probs = boltzmann_weights / np.sum(boltzmann_weights)
        cumulative_probs = np.cumsum(boltzmann_probs)

        cutoff_prob = percentage / 100.0
        favored_indices_sorted = [i for i, prob in enumerate(cumulative_probs) if prob <= cutoff_prob]
        if favored_indices_sorted and favored_indices_sorted[-1] < len(cumulative_probs) - 1:
            favored_indices_sorted.append(favored_indices_sorted[-1] + 1)

        favored_indices = sorted_indices[favored_indices_sorted]
        return self._apply_mask(coordinates, ids, favored_indices)

    def _filter_integer(self, coordinates, energies, ids, sorted_indices, parameters):
        logging.info("Filtering structures based on integer count.")
        num_structures = parameters.get('num_structures', len(coordinates))
        logging.info(f"Number of structures to select: {num_structures}")

        if num_structures <= 0 or num_structures >= len(coordinates):
            logging.info("Input count exceeds available structures; taking all structures.")
            favored_indices = sorted_indices
        else:
            favored_indices = sorted_indices[:num_structures]

        return self._apply_mask(coordinates, ids, favored_indices)

    def _apply_mask(self, coordinates, ids, favored_indices):
        mask = [i in favored_indices for i in range(len(coordinates))]
        filtered_coordinates = [coord for coord, keep in zip(coordinates, mask) if keep]
        filtered_ids = [id_ for id_, keep in zip(ids, mask) if keep]
        logging.info(f"Selected {len(filtered_coordinates)} structures after filtering.")
        return filtered_coordinates, filtered_ids
