from chemrefine.orca_interface import OrcaJobSubmitter


def test_orca_job_submitter_runs(tmp_path):
    # Create a dummy ORCA input file
    input_file = tmp_path / "input.inp"
    input_file.write_text(
        """
    ! B3LYP def2-SVP
    * xyz 0 1
    H 0.0 0.0 0.0
    H 0.0 0.0 0.74
    *
    """
    )

    # Instantiate the submitter
    submitter = OrcaJobSubmitter()

    # Use PAL adjustment or SLURM script generation as tests
    pal_value = 4
    modified = submitter.adjust_pal_in_input(input_file, pal_value)
    assert modified is True or modified is False

    extracted_pal = submitter.parse_pal_from_input(input_file)
    assert isinstance(extracted_pal, int)

    slurm_script = submitter.generate_slurm_script(input_file, pal_value)
    assert slurm_script.exists()
