import pytest
from unittest import mock
from chemrefine.core import ChemRefiner

def test_submit_orca_jobs(monkeypatch):
    """
    Test ChemRefiner.submit_orca_jobs() to ensure each input file is processed and a job is submitted.
    """
    # Initialize ChemRefiner
    chem = ChemRefiner()

    # Dummy input files
    input_files = ["/fake/path/input1.inp", "/fake/path/input2.inp"]
    cores = 8
    step_dir = "/fake/path/step1"

    # Patch OrcaJobSubmitter methods
    chem.orca_submitter = mock.MagicMock()
    chem.orca_submitter.parse_pal_from_input.return_value = 4
    chem.orca_submitter.generate_slurm_script.return_value = "/fake/path/job.slurm"
    chem.orca_submitter.submit_job.return_value = "12345"

    # Call the function
    chem.submit_orca_jobs(input_files, cores, step_dir)

    # Assertions
    assert chem.orca_submitter.parse_pal_from_input.call_count == len(input_files)
    assert chem.orca_submitter.adjust_pal_in_input.call_count == len(input_files)
    assert chem.orca_submitter.generate_slurm_script.call_count == len(input_files)
    assert chem.orca_submitter.submit_job.call_count == len(input_files)
