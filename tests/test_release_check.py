"""The release gate's sections: each is listed, selectable, and a selection says what it is.

``scripts/release-check.sh`` runs as named sections, and ``--list`` prints what a set of
options would run without running any of it. Option parsing and selection are the part of
the script that can be wrong with every tool in it green — a section silently dropped, a
partial run announcing "ready to tag" — so that is the seam these tests drive.

Each call runs a copy of the script in a scratch directory with ``PY`` pointing nowhere and
no ``release-check.env`` beside it: a path that failed to stop at ``--list`` or at a refusal
ends at "no Python found" instead of starting the gate from inside the suite.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_SCRIPT = REPO / "scripts" / "release-check.sh"
_BASH = shutil.which("bash")

# The same reason as tests/test_mutation_gate.py: neither the sdist nor the mutation gate's
# scratch copy carries scripts/.
pytestmark = pytest.mark.skipif(
    not _SCRIPT.is_file() or _BASH is None,
    reason="no scripts/ — neither the sdist nor the gate's own scratch copy carries it",
)

_CASES = sorted(
    case.name
    for case in (REPO / "tests/data/e2e/cases").iterdir()
    if (case / "input.yaml").is_file()
)
"""The live cases, by the rule ``tests/test_e2e_live.py`` collects them."""


@pytest.fixture
def gate(tmp_path: Path):
    """Run a sandboxed copy of the script; returns ``(exit code, stdout, stderr)``."""
    (tmp_path / "scripts").mkdir()
    shutil.copy2(_SCRIPT, tmp_path / "scripts" / _SCRIPT.name)
    (tmp_path / "tests").symlink_to(REPO / "tests", target_is_directory=True)
    env = {"PATH": os.environ["PATH"], "PY": str(tmp_path / "no-python")}

    def run(*args: str) -> tuple[int, str, str]:
        assert _BASH is not None
        done = subprocess.run(
            [_BASH, str(tmp_path / "scripts" / _SCRIPT.name), *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        return done.returncode, done.stdout, done.stderr

    return run


def _marked(listing: str) -> list[str]:
    """The sections a ``--list`` marks as selected, in the order it prints them."""
    return re.findall(r"^\* (\S+)", listing, re.MULTILINE)


def _listed(listing: str) -> list[str]:
    return re.findall(r"^[* ] (\w+) +(?:--pr and release|release only) ", listing, re.MULTILINE)


def test_every_section_the_script_runs_is_listed_and_the_release_gate_runs_them_all(gate):
    """The table ``--list`` prints and the ``section_*`` functions are one roster, in order."""
    code, out, _ = gate("--list")
    assert code == 0
    defined = re.findall(r"^section_(\w+)\(\)", _SCRIPT.read_text(encoding="utf-8"), re.MULTILINE)
    assert _listed(out) == defined
    assert _marked(out) == defined
    assert "passes as: the release gate" in out


def test_the_pr_gate_is_the_release_gate_without_its_workstation_half(gate):
    _, release, _ = gate("--list")
    code, out, _ = gate("--list", "--pr")
    assert code == 0
    assert "live" not in _marked(out) and "audit" not in _marked(out)
    assert set(_marked(out)) < set(_marked(release))
    assert "passes as: the pr gate" in out


def test_a_selection_runs_in_gate_order_and_never_passes_as_the_gate(gate):
    code, out, _ = gate("--list", "--only", "live,lint")
    assert code == 0
    assert _marked(out) == ["lint", "live"]
    assert "passes as: a selection, not the gate" in out


def test_skipping_a_section_is_a_selection_too(gate):
    _, out, _ = gate("--list", "--skip", "mutation")
    assert "mutation" not in _marked(out)
    assert "passes as: a selection, not the gate" in out


@pytest.mark.parametrize("section", ["sdist", "matrix"])
def test_a_section_reading_the_built_package_builds_it(gate, section: str):
    _, out, _ = gate("--list", "--only", section)
    assert _marked(out) == [section]
    assert "built first" in out


def test_a_case_runs_the_live_section_alone(gate):
    code, out, _ = gate("--list", "--case", _CASES[0], "--", "--record")
    assert code == 0
    assert _marked(out) == ["live"]
    assert f"live runs: {_CASES[0]}" in out and "--record" in out
    assert "passes as: a selection, not the gate" in out


@pytest.mark.parametrize(
    ("args", "valid"),
    [(("--only", "nope"), "lint"), (("--skip", "nope"), "live"), (("--case", "nope"), _CASES[0])],
)
def test_an_unknown_name_is_refused_naming_the_real_ones(gate, args: tuple[str, ...], valid: str):
    code, out, err = gate(*args)
    assert code == 2 and out == ""
    assert "'nope'" in err and valid in err


@pytest.mark.parametrize(
    "args",
    [
        ("--pr", "--release"),
        ("--pr", "--only", "lint"),
        ("--pr", "--case", _CASES[0]),
        ("--pr", "--no-heavy"),
        ("--only", "docs", "--", "-x"),
        ("--case", _CASES[0], "--skip", "live"),
        ("--only",),
        ("--frobnicate",),
    ],
    ids=lambda args: " ".join(args),
)
def test_options_that_contradict_each_other_are_refused_before_anything_runs(gate, args):
    code, out, _ = gate(*args)
    assert code == 2 and out == ""


def test_help_is_the_header_and_names_the_section_options(gate):
    code, out, _ = gate("--help")
    assert code == 0
    assert "--only" in out and "--case" in out and "PY" in out
    assert "set -euo" not in out
