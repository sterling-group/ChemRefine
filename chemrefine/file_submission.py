from .orca_interface import OrcaJobSubmitter
import logging
class FileSubmitter:
    def __init__(self):
        self.orca_submitter = OrcaJobSubmitter()

    def submit_files(self, input_files, max_cores, qorca_flags=None):
        for input_file in input_files:
            input_path = Path(input_file)
            
            # 1 Read the PAL value from input
            pal_value = self.orca_submitter.parse_pal_from_input(input_path)
            pal_value = min(pal_value, max_cores)

            # 2️ Adjust the PAL value in the input file
            self.orca_submitter.adjust_pal_in_input(input_path, pal_value)

            # 3️ Generate SLURM script
            slurm_script = self.orca_submitter.generate_slurm_script(input_path, pal_value)

            # 4️ Submit the job
            job_id = self.orca_submitter.submit_job(slurm_script)
            logging.info(f"Job submitted with ID: {job_id}")

