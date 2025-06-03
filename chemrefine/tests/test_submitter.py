import pytest
from chemrefine.orca_interface import OrcaJobSubmitter
from pathlib import Path

def test_generate_slurm_script(tmp_path):
    """
    Test OrcaJobSubmitter.generate_slurm_script() with user-provided header.
    """
    submitter = OrcaJobSubmitter(orca_executable="orca", scratch_dir="/tmp/scratch")
    input_file = tmp_path / "test.inp"
    input_file.write_text("! Some ORCA input")

    header_template = tmp_path / "orca.slurm.header"
    header_template.write_text("#SBATCH --partition=short\n#SBATCH --time=1:00:00\nORCA_EXEC=/path/to/orca\n")

    slurm_script = submitter.generate_slurm_script(input_file, pal_value=4, template_dir=tmp_path)
    assert slurm_script.exists()
    content = slurm_script.read_text()
    assert "--job-name=test" in content
    assert "ORCA_EXEC" in content

def test_parse_pal_from_input(tmp_path):
    """
    Test OrcaJobSubmitter.parse_pal_from_input() parses PAL value.
    """
    input_file = tmp_path / "test.inp"
    input_file.write_text("""
    ! DFT calculation
    %pal nprocs 8 end
    """)
    submitter = OrcaJobSubmitter()
    pal_value = submitter.parse_pal_from_input(input_file)
    assert pal_value == 8
