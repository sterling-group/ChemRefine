import pytest
from pathlib import Path
from chemrefine.orca_interface import OrcaInterface

def test_parse_output_real_data(tmp_path):
    """
    Test OrcaInterface.parse_output() parses a real ORCA output file.
    """
    # Copy the example ORCA output to a temporary test directory
    src_file = Path(__file__).parent / "data" / "orca.out"
    test_file = tmp_path / "orca.out"
    test_file.write_text(src_file.read_text())

    # Initialize OrcaInterface
    orca = OrcaInterface()

    # Parse the output
    coordinates, energies = orca.parse_output([test_file], calculation_type="DFT")

    # Perform assertions
    assert isinstance(coordinates, list)
    assert isinstance(energies, list)
    assert len(energies) > 0
    assert isinstance(energies[0], float)
