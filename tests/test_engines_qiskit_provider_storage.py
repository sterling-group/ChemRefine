"""Provider request records survive worker failure outside transient scratch storage."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chemrefine.config import StepConfig
from chemrefine.engines.qiskit.engine import QiskitEngine
from chemrefine.engines.qiskit.experiment import QiskitExperimentEngine
from chemrefine.slurm.script import build_array_script, build_script, write_array_manifests
from chemrefine.state import PipelineState, StepContext


@pytest.mark.parametrize("engine_cls", [QiskitEngine, QiskitExperimentEngine])
@pytest.mark.parametrize("array", [False, True])
def test_provider_records_are_durable_before_failed_worker_copyback(tmp_path, engine_cls, array):
    """Execute both real shell paths without submitting to SLURM or any quantum provider."""
    engine = engine_cls()
    output = tmp_path / "durable job"
    output.mkdir()
    source = output / "worker.py"
    source.write_text(
        "import json, os\nfrom pathlib import Path\n"
        "directory = Path(os.environ['CHEMREFINE_PROVIDER_JOURNAL_DIR'])\n"
        "directory.mkdir(parents=True)\n"
        "record = {'cwd': str(Path.cwd()), 'threads': os.environ['OMP_NUM_THREADS']}\n"
        "(directory / 'request.json').write_text(json.dumps(record))\n"
        "Path('result.json').write_text('{}')\nraise SystemExit(7)\n"
    )
    header = tmp_path / "header"
    header.write_text("#!/bin/bash\n")
    ctx = StepContext(
        step_cfg=StepConfig(
            step=1, engine=engine.name, options={"backend_python": sys.executable, "cores": 8}
        ),
        step_dir=tmp_path,
        template_dir=tmp_path,
        template=None,
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="header",
    )
    script = tmp_path / "job.sh"
    args = {
        "step_label": "quantum",
        "ntasks": 1,
        "cpus_per_task": 2,
        "template_path": header,
        "script_path": script,
        "output_dir": output,
        "scratch_dir": ctx.scratch_dir,
        "engine": engine.name,
        "operation": "sp",
        "step": 1,
        "output_globs": engine.output_globs,
        "output_dirs": engine.output_dirs(ctx),
    }
    target = output / "result.json"
    command = ["bash", str(script)]
    if array:
        build_array_script(
            **args, run_block=engine.run_block(ctx, Path("$INP_NAME"), Path("$OUT_NAME"))
        )
        manifest = write_array_manifests([(source, target, "0")], tmp_path, step_label="quantum")
        command.append(str(manifest[0][0]))
    else:
        build_script(
            **args,
            job_name="quantum",
            input_path=source,
            structure_id="0",
            run_block=engine.run_block(ctx, source, target),
        )
    result = subprocess.run(
        command, env={**os.environ, "SLURM_ARRAY_TASK_ID": "0"}, capture_output=True, check=False
    )
    assert result.returncode == 7, result.stderr.decode()
    journal = output / "provider_jobs" / "request.json"
    record = json.loads(journal.read_text())
    assert Path(record["cwd"]).parent == ctx.scratch_dir
    assert record["threads"] == "2"
    assert journal.is_file() and target.is_file()
