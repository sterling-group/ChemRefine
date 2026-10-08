"""Qiskit worker budgets agree with the shared scheduler's grant."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from chemrefine.config import StepConfig
from chemrefine.engines.qiskit.engine import QiskitEngine
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.state import PipelineState, StepContext


@pytest.mark.parametrize("estimator", ["aer_statevector", "aer_shots"])
@pytest.mark.parametrize("requested, granted", [(8, 3), (2, 2)])
def test_rendered_aer_options_use_the_granted_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    estimator: str,
    requested: int,
    granted: int,
) -> None:
    """Both template spellings receive the grant without changing scientific choices."""
    raw = {
        "cores": requested,
        "basis": "6-31g",
        "algorithm": "vqe",
        "estimator": estimator,
        "active_space": {"electrons": [1, 1], "orbitals": 2},
        "optimizer": {"name": "slsqp", "options": {"maxiter": 17}},
    }
    original = QiskitOptions.from_raw(raw).as_job_spec()
    step = StepConfig(step=1, engine="qiskit", options=raw)
    ctx = StepContext(
        step_cfg=step,
        step_dir=tmp_path,
        template_dir=tmp_path,
        template=None,
        scratch_dir=None,
        prev_state=PipelineState(),
        charge=0,
        multiplicity=1,
        max_cores=3,
        slurm_template="cpu.slurm.header",
    )
    template = tmp_path / "template.py"
    template.write_text(
        "import json\n"
        'worker = json.loads("$OPTIONS_JSON")\n'
        'legacy_worker = json.loads("$QISKIT_OPTIONS_JSON")\n'
        "cores_placeholder = $CORES\n"
        "energy_hartree = -1.0\n",
        encoding="utf-8",
    )
    engine = QiskitEngine()
    rendered = tmp_path / "step1_0.py"
    output = tmp_path / "step1_0.json"
    engine.build_input(
        xyz_path=tmp_path / "seed.xyz",
        template_path=template,
        input_path=rendered,
        output_path=output,
        ctx=ctx,
    )
    monkeypatch.chdir(tmp_path)
    executed = runpy.run_path(str(rendered))

    assert executed["worker"] == {**original, "cores": granted}
    assert executed["legacy_worker"] == executed["worker"]
    assert executed["cores_placeholder"] == granted
    assert json.loads(output.read_text(encoding="utf-8")) == {"energy_hartree": -1.0}
    assert engine.slurm_layout(ctx) == (1, granted)
    run_block = engine.run_block(ctx, rendered, output).body
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f"export {variable}={granted}\n" in run_block
    assert raw["cores"] == requested
    assert ctx.step_cfg is step
    assert QiskitOptions.from_raw(step.options).as_job_spec() == original
    assert engine.pal(ctx) == requested
