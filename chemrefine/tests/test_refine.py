import pytest
import numpy as np
from chemrefine.refine import StructureRefiner

def test_filter_structures():
    """
    Test StructureRefiner.filter() reduces structures based on dummy energy thresholds.
    """
    refiner = StructureRefiner()
    coordinates = np.random.rand(10, 3)
    energies = np.linspace(-100, -90, 10)
    ids = list(range(10))
    filtered_coords, filtered_ids = refiner.filter(coordinates, energies, ids, 'integer', {'num_structures': 1})
    assert len(filtered_coords) <= len(coordinates)
    assert all(i in ids for i in filtered_ids)
