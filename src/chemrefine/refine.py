import numpy as np
import logging
from .constants import HARTREE_TO_KCAL_MOL, R_KCAL_MOL_K, DEFAULT_TEMPERATURE


class StructureRefiner:
    def filter(
        self, coordinates, energies, ids, method, parameters, *, by_parent=False
    ):
        """
        Filters structures globally or per-parent depending on `by_parent`.

        Args:
            coordinates (list): List of coordinates.
            energies (list): List of energies.
            ids (list): Persistent IDs.
            method (str): Filtering method.
            parameters (dict): Filtering parameters.
            by_parent (bool): If True, apply filtering per-parent group.

        Returns:
            tuple[list, list]: (filtered_coordinates, filtered_ids)
        """
        if not by_parent:
            return self._filter_global(coordinates, energies, ids, method, parameters)
        else:
            return self._filter_by_parent(
                coordinates, energies, ids, method, parameters
            )

    def _filter_global(self, coordinates, energies, ids, method, parameters):
        """Global filtering (current behavior)."""
        logging.info("Filtering structures globally.")
        return self._dispatch(coordinates, energies, ids, method, parameters)

    def _filter_by_parent(self, coordinates, energies, ids, method, parameters):
        """
        Filter structures grouped by parent ID.

        Supports both:
        - Integer block allocation scheme (stride = 1000).
        - Hyphen-separated string scheme ("0-1-2").
        """
        import logging
        from collections import defaultdict

        logging.info("Filtering structures by parent ID groups.")

        groups = defaultdict(lambda: {"coords": [], "energies": [], "ids": []})

        for coord, e, sid in zip(coordinates, energies, ids):
            if isinstance(sid, int):
                # old block allocation scheme
                parent = sid // 1000
            elif isinstance(sid, str):
                # new hyphen scheme: parent is everything before the last "-"
                parent = sid.rsplit("-", 1)[0] if "-" in sid else sid
            else:
                raise TypeError(f"Unsupported structure_id type: {type(sid)}")

            groups[parent]["coords"].append(coord)
            groups[parent]["energies"].append(e)
            groups[parent]["ids"].append(sid)

        all_coords, all_ids = [], []
        for parent, g in groups.items():
            f_coords, f_ids = self._dispatch(
                g["coords"], g["energies"], g["ids"], method, parameters
            )
            all_coords.extend(f_coords)
            all_ids.extend(f_ids)

        return all_coords, all_ids

    def _dispatch(self, coordinates, energies, ids, method, parameters):
        """Your existing filtering logic moved here."""
        import numpy as np
        import logging

        logging.info("Starting structure filtering process.")
        logging.info(
            f"Method: {method}, Parameters: {parameters} starting {len(coordinates)} structures."
        )

        if len(coordinates) != len(energies) or len(energies) != len(ids):
            raise ValueError("Mismatch in list lengths")

        if not energies or all(e is None for e in energies):
            logging.warning(
                "No valid energies found to filter. Returning empty results."
            )
            return [], []

        energies = np.array([float(e) for e in energies])
        sorted_indices = np.argsort(energies)

        if method == "energy_window":
            return self._filter_energy_window(
                coordinates, energies, ids, sorted_indices, parameters
            )
        elif method == "boltzmann":
            return self._filter_boltzmann(
                coordinates, energies, ids, sorted_indices, parameters
            )
        elif method == "integer":
            return self._filter_integer(
                coordinates, energies, ids, sorted_indices, parameters
            )
        elif method == "high_energy":
            return self._filter_high_energy(
                coordinates, energies, ids, sorted_indices, parameters
            )
        else:
            raise ValueError(
                "Invalid method. Choose from 'energy_window', 'boltzmann', 'integer', 'high_energy'."
            )

    def _filter_energy_window(
        self, coordinates, energies, ids, sorted_indices, parameters
    ):
        logging.info("Filtering structures based on energy window.")
        energy = parameters.get("energy", 0.5)  # Default is 0.5 Hartrees
        unit = parameters.get("unit", "hartree").lower()
        logging.info(f"Filtering Energy window: {energy} {unit}")

        if unit == "kcal/mol":
            energy /= HARTREE_TO_KCAL_MOL
            logging.info(f"Converted energy window to Hartrees: {energy:.6f}")

        min_energy = np.min(energies)
        favored_indices = [
            i for i in sorted_indices if energies[i] <= min_energy + energy
        ]

        return self._apply_mask(coordinates, ids, favored_indices)

    def _filter_boltzmann(self, coordinates, energies, ids, sorted_indices, parameters):
        logging.info("Filtering structures based on Boltzmann probability.")
        if len(energies) == 0:
            logging.warning(
                "No structures available for Boltzmann filtering. Returning empty lists."
            )
            return [], []

        if len(energies) == 1:
            logging.info("Only one structure available; returning it.")
            return coordinates, ids

        temperature = parameters.get("temperature", DEFAULT_TEMPERATURE)
        percentage = parameters.get("weight", 99)
        logging.info(
            f"Filtering Boltzmann probability: {percentage}% at {temperature} K."
        )

        energies_kcalmol = energies * HARTREE_TO_KCAL_MOL
        min_energy = np.min(energies_kcalmol)
        delta_E = energies_kcalmol - min_energy

        boltzmann_weights = np.exp(-delta_E / (R_KCAL_MOL_K * temperature))
        boltzmann_probs = boltzmann_weights / np.sum(boltzmann_weights)
        cumulative_probs = np.cumsum(boltzmann_probs)

        cutoff_prob = percentage / 100.0
        favored_indices_sorted = [
            i for i, prob in enumerate(cumulative_probs) if prob <= cutoff_prob
        ]
        if (
            favored_indices_sorted
            and favored_indices_sorted[-1] < len(cumulative_probs) - 1
        ):
            favored_indices_sorted.append(favored_indices_sorted[-1] + 1)

        favored_indices = sorted_indices[favored_indices_sorted]
        return self._apply_mask(coordinates, ids, favored_indices)

    def _filter_integer(self, coordinates, energies, ids, sorted_indices, parameters):
        logging.info("Filtering structures based on integer count.")
        num_structures = parameters.get("num_structures", len(coordinates))
        logging.info(f"Number of structures to select: {num_structures}")

        # If user specifies 0 or a value greater than available structures, take all structures
        if not num_structures or num_structures >= len(coordinates):
            logging.info(
                "Input count is 0 or exceeds available structures; taking all structures."
            )
            favored_indices = sorted_indices
        else:
            favored_indices = sorted_indices[:num_structures]

        return self._apply_mask(coordinates, ids, favored_indices)

    def _apply_mask(self, coordinates, ids, favored_indices):
        """
        Apply a mask to select structures based on favored indices.
        Ensures at least one structure is always selected.
        """
        if len(favored_indices) == 0:
            logging.warning(
                "No structures selected after filtering. Taking the lowest-energy structure by default."
            )
            favored_indices = [0]

        mask = [i in favored_indices for i in range(len(coordinates))]
        filtered_coordinates = [coord for coord, keep in zip(coordinates, mask) if keep]
        filtered_ids = [id_ for id_, keep in zip(ids, mask) if keep]
        logging.info(
            f"Selected {len(filtered_coordinates)} structures after filtering."
        )
        return filtered_coordinates, filtered_ids

    def _filter_high_energy(
        self, coordinates, energies, ids, sorted_indices, parameters
    ):
        """
        Selects the highest-energy structures from PES output.

        Args:
            coordinates (list): List of atomic coordinates.
            energies (list): List of energies.
            ids (list): List of structure IDs.
            sorted_indices (list): Not used here.
            parameters (dict): Must contain 'num_structures'.

        Returns:
            tuple: (filtered_coordinates, filtered_ids)
        """
        num_structures = parameters.get("num_structures", 1)

        if len(energies) == 0 or all(e is None for e in energies):
            raise ValueError("Energy list is empty or contains only None values.")

        energy_tuples = [(i, e) for i, e in enumerate(energies) if e is not None]
        sorted_by_energy = sorted(energy_tuples, key=lambda x: x[1], reverse=True)
        selected = sorted_by_energy[:num_structures]
        selected_indices = [i for i, _ in selected]

        filtered_coords = [coordinates[i] for i in selected_indices]
        filtered_ids = [ids[i] for i in selected_indices]

        return filtered_coords, filtered_ids
