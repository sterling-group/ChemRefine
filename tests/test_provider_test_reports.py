"""Provider compatibility jobs must prove numerical cases ran rather than skipped."""

import runpy
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


def test_cuda_workflow_is_opt_in_protected_and_rejects_skips():
    """Runner access is manual, main-only, environment-gated and evidence-producing."""
    import re
    import subprocess

    import yaml

    root = Path(__file__).resolve().parents[1]
    workflow = yaml.load(
        (root / ".github/workflows/quantum-cuda.yml").read_text(), Loader=yaml.BaseLoader
    )
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
