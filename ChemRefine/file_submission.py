import os, subprocess, time, getpass, re, logging
from .utils import extract_pal_from_qorca_output
from .constants import JOB_CHECK_INTERVAL, DEFAULT_CORES, DEFAULT_MAX_CORES

def submit_qorca(input_file, qorca_flags=None):
    qorca_path = os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "qorca", "qorca")
    command = ["python3", qorca_path, input_file]
    if qorca_flags:
        command.extend(qorca_flags)
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    jobid_match = re.search(r'Submitted job (\d+)', result.stdout)
    jobid = jobid_match.group(1) if jobid_match else result.stdout.strip().split()[-1]
    pal = extract_pal_from_qorca_output(result.stdout + "\n" + result.stderr)
    return jobid, pal

def is_job_finished(job_id):
    try:
        username = getpass.getuser()
        result = subprocess.run(["squeue", "-u", username, "-o", "%i", "-h"],
                                capture_output=True, text=True, timeout=30)
        return job_id not in result.stdout.strip().splitlines()
    except Exception:
        return True

def submit_files(input_files, max_cores=DEFAULT_MAX_CORES, qorca_flags=None):
    total_cores, active_jobs = 0, {}
    for input_file in input_files:
        cores_needed = DEFAULT_CORES
        while total_cores + cores_needed > max_cores:
            time.sleep(JOB_CHECK_INTERVAL)
            finished = [jid for jid, cores in active_jobs.items() if is_job_finished(jid)]
            for jid in finished:
                total_cores -= active_jobs.pop(jid)
        job_id, pal = submit_qorca(input_file, qorca_flags)
        actual_cores = pal if pal else cores_needed
        active_jobs[job_id] = actual_cores
        total_cores += actual_cores
    while active_jobs:
        time.sleep(JOB_CHECK_INTERVAL)
        for jid in list(active_jobs):
            if is_job_finished(jid):
                total_cores -= active_jobs.pop(jid)
