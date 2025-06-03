import pytest
import pandas as pd
from chemrefine.utils import Utility

def test_save_step_csv(tmp_path):
    """
    Test Utility.save_step_csv() writes CSV with correct structure.
    """
    energies = [-75.0, -74.5, -74.2]
    ids = [0, 1, 2]
    utility = Utility()
    step_number = 1
    utility.save_step_csv(energies, ids, step_number)

    csv_file = tmp_path / f"step{step_number}.csv"
    assert csv_file.exists()
    df = pd.read_csv(csv_file)
    assert 'energy' in df.columns
    assert len(df) == len(energies)
