"""The release gate's sections: each is listed, selectable, and a selection says what it is.

``scripts/release-check.sh`` runs as named sections, and ``--list`` prints what a set of
options would run without running any of it. Option parsing and selection are the part of
the script that can be wrong with every tool in it green — a section silently dropped, a
partial run announcing "ready to tag" — so that is the seam these tests drive.

Each call runs a copy of the script in a scratch directory with ``PY`` pointing nowhere and
no ``release-check.env`` beside it: a path that failed to stop at ``--list`` or at a refusal
ends at "no Python found" instead of starting the gate from inside the suite. The run's own
half — its preconditions and its verdict — is driven through stand-ins for the tools it
calls (:func:`stubbed`), so what a run passes as is held where the run says it.
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


_STUBS = {
    "python": """#!/bin/sh
# The interpreter, as far as the gate asks it: the import check, versions, one build.
case "$*" in
    *"import chemrefine, pathlib"*) printf '%s/src\\n' "$(cd "$(dirname "$0")/.." && pwd)" ;;
    *"import chemrefine;"*) echo 0.0.stub ;;
    *"version_info[:3]"*) echo 3.13.0 ;;
    *"version_info[:2]"*) echo 3.13 ;;
    "-m build --sdist --wheel")
        mkdir -p dist/chemrefine-0 && tar -czf dist/chemrefine-0.tar.gz -C dist chemrefine-0 ;;
    -) cat >/dev/null; echo 3.13 3.98 ;;
    *) exit 3 ;;
esac
""",
    "pytest": """#!/bin/sh
# tier-3's collection: a light case, and the heavy one whenever an -m after the script's
# own, in any spelling pytest reads, selects integration again.
light=tests/test_e2e_live.py::test_live_case[mlip]
heavy=tests/test_e2e_live.py::test_live_case[fairchem_sp]
case "$*" in
    *"-m heavy"*) echo "$heavy" ;;
    *"not heavy"*integration*) printf '%s\\n' "$light" "$heavy" ;;
    *) echo "$light" ;;
esac
""",
    "mypy": "#!/bin/sh\nexit 0\n",
    "pyproject-build": "#!/bin/sh\nexit 0\n",
    "systemd-run": "#!/bin/sh\nexit 0\n",
    "orca": "#!/bin/sh\nexit 0\n",
    "orca_2json": "#!/bin/sh\nexit 0\n",
}
"""Stand-ins for what a run calls, so the run half executes in a scratch copy in a second.

The interpreter answers the import check with the copy's own ``src/`` and builds an empty
sdist; the classifiers it reports include a version no machine has, so the matrix has a
leg to name. ``systemd-run`` succeeds without running anything, so the live section ends
where it would hand the tier to the scheduler."""


@pytest.fixture
def stubbed(tmp_path: Path):
    """Run the scratch copy's run half against :data:`_STUBS`; ``(exit code, out, err)``."""
    (tmp_path / "scripts").mkdir()
    shutil.copy2(_SCRIPT, tmp_path / "scripts" / _SCRIPT.name)
    (tmp_path / "tests").symlink_to(REPO / "tests", target_is_directory=True)
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in _STUBS.items():
        (stubs / name).write_text(body, encoding="utf-8")
        (stubs / name).chmod(0o755)
    env = {
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "PY": str(stubs / "python"),
        "ORCA": str(stubs / "orca"),
    }

    def run(*args: str) -> tuple[int, str, str]:
        assert _BASH is not None
        done = subprocess.run(
            [_BASH, str(tmp_path / "scripts" / _SCRIPT.name), *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmp_path,
            timeout=60,
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
    """Exactly that: a row moved to release-only would drop a check CI runs on every PR."""
    _, release, _ = gate("--list")
    code, out, _ = gate("--list", "--pr")
    assert code == 0
    assert _marked(out) == [name for name in _marked(release) if name not in ("audit", "live")]
    assert "passes as: the pr gate" in out


def test_a_selection_runs_in_gate_order_and_never_passes_as_the_gate(gate):
    """The order is the run loop's own: ``--list`` prints the variable the loop iterates."""
    code, out, _ = gate("--list", "--only", "live,lint")
    assert code == 0
    assert _marked(out) == ["lint", "live"]
    assert "runs, in order: lint live" in out
    assert "passes as: a selection, not the gate" in out


def test_a_partial_run_passes_as_that_selection_and_never_as_ready_to_tag(stubbed):
    """The run's own verdict, where ``--list`` only previews it."""
    code, out, err = stubbed("--only", "types")
    assert code == 0, err
    assert "✓ passed: types" in out
    assert "A selection, not the gate: before tagging, run it whole." in out
    assert "ready to tag" not in out


def test_a_partial_run_of_the_matrix_names_the_legs_it_could_not_run(stubbed):
    """The whole gate names them among what only CI runs; a selection did not say at all.

    On a machine with no second interpreter, ``--only matrix`` built the package, ran no
    leg, and ended ``✓ passed: matrix``.
    """
    code, out, err = stubbed("--only", "matrix")
    assert code == 0, err
    assert "✓ passed: matrix" in out
    assert "no interpreter here (3.98) did not run" in out


@pytest.mark.parametrize(
    "passed_through",
    [("-m", "integration"), ("-mintegration",), ("-xm", "integration")],
    ids=lambda args: " ".join(args),
)
def test_no_heavy_undone_by_the_pytest_arguments_is_refused_before_anything_runs(
    stubbed, passed_through: tuple[str, ...]
):
    """pytest decides what an -m after ``--`` selects, in every spelling it reads, so it is asked.

    A pattern over the arguments missed ``-xm integration``, which pytest reads as ``-x -m
    integration``: the heavy case ran, under a start threshold sized without it.
    """
    code, out, err = stubbed("--only", "live", "--no-heavy", "--", *passed_through)
    assert code == 2
    assert "test_live_case[fairchem_sp]" in err
    assert "tier-3" not in out, "refused in the preconditions, before the live section"


def test_no_heavy_that_holds_reaches_the_live_section(stubbed):
    """The control: with nothing passed through, the preconditions pass and the tier starts."""
    code, out, err = stubbed("--only", "live", "--no-heavy")
    assert "--no-heavy, but" not in err
    assert "tier-3: the real binaries, end to end" in out
    assert code == 0 or "of memory available" in err


def test_skipping_a_section_is_a_selection_too(gate):
    _, out, _ = gate("--list", "--skip", "mutation")
    assert "mutation" not in _marked(out)
    assert "passes as: a selection, not the gate" in out


@pytest.mark.parametrize("section", ["sdist", "matrix"])
def test_a_section_reading_the_built_package_builds_it(gate, section: str):
    """Run alone, it builds the package itself, and ``--list`` says so."""
    _, out, _ = gate("--list", "--only", section)
    assert _marked(out) == [section]
    assert "built first" in out
    body = re.search(
        rf"^section_{section}\(\) {{\n(.*?)^}}", _SCRIPT.read_text(encoding="utf-8"), re.M | re.S
    )
    assert body is not None and re.search(r"^\s*build_package$", body[1], re.M)


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
        ("--case", _CASES[0], "--no-heavy"),
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
