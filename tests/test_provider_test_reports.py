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
