import pytest
from unittest import mock
from chemrefine.core import ChemRefiner

def test_parse_and_filter_outputs(monkeypatch):
    """
    Test ChemRefiner.parse_and_filter_outputs() for correct output parsing and filtering.
    """
    chem = ChemRefiner()

    output_files = ["/fake/path/output1.out", "/fake/path/output2.out"]
    calculation_type = "DFT"
    step_number = 1
    sample_method = "boltzmann"
    parameters = {}
    step_dir = "/fake/path/step1"

    # Patch OrcaInterface.parse_output to return dummy coordinates and energies
    dummy_coords = [[0, 0, 0], [1, 1, 1]]
    dummy_energies = [-100.0, -99.5]
    chem.orca = mock.MagicMock()
    chem.orca.parse_output.return_value = (dummy_coords, dummy_energies)

    # Patch utils and refiner
    chem.utils = mock.MagicMock()
    chem.refiner = mock.MagicMock()
    chem.refiner.filter.return_value = (dummy_coords, [0, 1])

    # Call the function
    filtered_coords, filtered_ids = chem.parse_and_filter_outputs(
        output_files, calculation_type, step_number, sample_method, parameters, step_dir
    )

    # Assertions
    chem.orca.parse_output.assert_called_once_with(output_files, calculation_type, dir=step_dir)
    chem.utils.save_step_csv.assert_called_once()
    chem.refiner.filter.assert_called_once()
    chem.utils.move_step_files.assert_called_once()
    assert filtered_coords == dummy_coords
    assert filtered_ids == [0, 1]
