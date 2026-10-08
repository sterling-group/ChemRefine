"""Provider compatibility jobs must prove numerical cases ran rather than skipped."""

import runpy
import subprocess
from pathlib import Path

import pytest

CHECK = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tests/provider_reports.py"))


def test_provider_report_requires_each_selected_file(tmp_path):
    """Mixed function/class cases count, but an omitted provider file does not."""
    report = tmp_path / "report.xml"
    report.write_text(
        '<testsuites><testsuite><testcase classname="tests.test_alpha.Suite"/>'
        '<testcase classname="test_beta"/></testsuite></testsuites>'
    )
    CHECK["validate_report"](report, ["tests/test_alpha.py", "tests/test_beta.py"])
    with pytest.raises(ValueError, match=r"test_missing\.py"):
        CHECK["validate_report"](report, ["tests/test_missing.py"])


@pytest.mark.parametrize("tag", ["skipped", "error", "failure", "empty"])
def test_provider_report_rejects_skips_failures_and_empty_runs(tmp_path, tag):
    """Successful collection or an SDK import never substitutes for numerical execution."""
    report = tmp_path / "report.xml"
    report.write_text(
        "<testsuite/>"
        if tag == "empty"
        else f'<testsuite><testcase classname="test_alpha"><{tag}/></testcase></testsuite>'
    )
    with pytest.raises(ValueError, match="without skips"):
        CHECK["validate_report"](report, ["test_alpha.py"])


@pytest.fixture
def cuda_workflow():
    """Read the shipped workflow without YAML 1.1 boolean coercion of its trigger."""
    import yaml

    root = Path(__file__).resolve().parents[1]
    return yaml.load(
        (root / ".github/workflows/quantum-cuda.yml").read_text(), Loader=yaml.BaseLoader
    )


def test_cuda_workflow_is_opt_in_protected_and_rejects_skips(cuda_workflow):
    """Runner access is manual, main-only, environment-gated and evidence-producing."""
    import re

    workflow = cuda_workflow
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    readiness = workflow["jobs"]["readiness"]["steps"][0]
    assert "refs/heads/main" in readiness["run"]
    assert 'test "$ENABLED" = true' in readiness["run"]
    cuda = workflow["jobs"]["cuda"]
    assert cuda["needs"] == "readiness"
    assert cuda["environment"] == "quantum-cuda"
    assert "chemrefine-cuda" in cuda["runs-on"]
    commands = "\n".join(step.get("run", "") for step in cuda["steps"])
    assert "-m gpu" in commands and "tests/provider_reports.py" in commands
    assert "quantum_validation.py" in commands and "pip check" in commands
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r".+@[0-9a-f]{40}", step["uses"])
            if "run" in step:
                subprocess.run(["bash", "-n"], input=step["run"], text=True, check=True)
    checkout = cuda["steps"][0]
    assert checkout["with"]["persist-credentials"] == "false"
    assert cuda["steps"][-1]["if"] == "always()"


def test_cuda_workflow_initializes_paths_after_runner_assignment(cuda_workflow, tmp_path):
    """Runner paths are expanded at execution time, not in invalid job-level expressions."""
    workflow = cuda_workflow
    environments = [workflow.get("env", {})]
    environments.extend(job.get("env", {}) for job in workflow["jobs"].values())
    assert all("runner." not in value for env in environments for value in env.values())
    steps = workflow["jobs"]["cuda"]["steps"]
    init_index, initialize = next(
        (index, step)
        for index, step in enumerate(steps)
        if step.get("name") == "Initialize runner paths"
    )
    consumers = [index for index, step in enumerate(steps) if "$EVIDENCE" in step.get("run", "")]
    assert consumers and all(init_index < index for index in consumers)
    runner_temp = tmp_path / "runner temp with spaces"
    github_env = tmp_path / "github env"
    github_env.write_text("EXISTING=retained\n")
    subprocess.run(
        ["bash", "-euo", "pipefail", "-c", initialize["run"]],
        env={"RUNNER_TEMP": str(runner_temp), "GITHUB_ENV": str(github_env)},
        check=True,
    )
    values = dict(line.split("=", 1) for line in github_env.read_text().splitlines())
    assert values == {
        "EXISTING": "retained",
        "CHEMREFINE_HOME": str(runner_temp / "chemrefine-cuda-home"),
        "EVIDENCE": str(runner_temp / "quantum-cuda-evidence"),
    }
    assert (
        steps[-1]["with"]["path"].replace("${{ runner.temp }}", str(runner_temp))
        == values["EVIDENCE"]
    )


@pytest.mark.parametrize(
    ("ref", "enabled", "python", "allowed"),
    [
        ("refs/heads/main", "true", "/opt/cuda/bin/python", True),
        ("refs/heads/feature", "true", "/opt/cuda/bin/python", False),
        ("refs/heads/main", "", "/opt/cuda/bin/python", False),
        ("refs/heads/main", "true", "", False),
    ],
)
def test_cuda_readiness_requires_main_and_explicit_setup(
    cuda_workflow, ref, enabled, python, allowed
):
    """The hosted guard refuses to request the GPU runner until all prerequisites are set."""
    step = cuda_workflow["jobs"]["readiness"]["steps"][0]
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        env={"SELECTED_REF": ref, "ENABLED": enabled, "CUDA_PYTHON": python},
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == allowed
