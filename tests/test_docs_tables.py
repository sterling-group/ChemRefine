"""Guard: the generated docs tables must actually name everything the registry holds.

``docs/hooks/tables.py`` replaces the hand-written engine / backend / extra rosters with
tables built from the registry at docs-build time, and ``mkdocs build --strict`` fails if
the hook raises. What ``--strict`` cannot see is a hook that *succeeds* and emits an empty
or partial table — the build passes and the page silently claims the project has no
engines. These tests are that missing half.

The exit-code table stays hand-written and is guarded rather than generated: two of its
four columns (``Usual cause`` / ``What to do``) are teaching that exists nowhere in the
code, so generating ``Code`` + ``Meaning`` would either delete them or leave a row that
still has to be authored by hand — no saving, and a worse page. What the code *does* own
is the set of codes, which is exactly what went stale (``RunLockError``'s ``10`` was
missing from two pages until ``4390ebf``), so that is what is checked here.

``docs/`` ships in the repository but not in the sdist, so every test skips rather than
fails when the directory is absent — the same rule as ``test_docs_drift``.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from chemrefine.engines import known_backend_extras
from chemrefine.engines.api import ENGINES
from chemrefine.engines.mlip.registry import registered_backends
from chemrefine.errors import ChemRefineError

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HOOK = _REPO_ROOT / "docs" / "hooks" / "tables.py"
_RUN_FAILS = _REPO_ROOT / "docs" / "running" / "when-a-run-fails.md"

pytestmark = pytest.mark.skipif(
    not _HOOK.is_file(), reason="docs/ is a repository artifact and does not ship in the sdist"
)


def _hook() -> Any:
    """Import the hook the way MkDocs does — by path, not as a package module."""
    spec = importlib.util.spec_from_file_location("chemrefine_docs_tables", _HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _render(directive: str) -> str:
    """The markdown one directive expands to, as a page would receive it."""
    return str(_hook().on_page_markdown(f"<!-- chemrefine:{directive} -->"))


def _rows(table: str) -> list[str]:
    """The table's body rows — header and separator dropped."""
    return [line for line in table.splitlines() if line.startswith("|")][2:]


@pytest.mark.parametrize("directive", ["engines", "backends", "extras"])
def test_each_table_has_a_header_and_at_least_one_row(directive: str):
    """The empty-table failure the build itself cannot see."""
    table = _render(directive)
    assert table.startswith("| "), f"{directive} did not render a markdown table:\n{table}"
    assert _rows(table), f"{directive} rendered a header with no rows"


def test_every_registered_engine_is_in_the_engine_table():
    """The roster that went stale: ``qchem`` was missing from seven pages."""
    table = _render("engines")
    missing = sorted(name for name in ENGINES if f"`{name}`" not in table)
    assert not missing, f"engines {missing} are registered but absent from the generated table"
    assert len(_rows(table)) == len(ENGINES)


def test_every_backend_extra_and_task_name_is_in_the_backend_table():
    """``chemrefine backends install`` accepts exactly these; the page must say so."""
    table = _render("backends")
    missing = sorted(e for e in known_backend_extras() if f"`{e}`" not in table)
    assert not missing, f"backend extras {missing} are installable but absent from the table"
    absent = sorted(t for t in registered_backends() if f"`{t}`" not in table)
    assert not absent, f"task names {absent} resolve to a backend but are absent from the table"


def test_the_extras_table_names_every_extra_pyproject_declares():
    """The gap this table closes: the install page named 11 of the 16 extras."""
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(data["project"]["optional-dependencies"])
    table = _render("extras")
    missing = sorted(e for e in declared if f"`[{e}]`" not in table)
    assert not missing, f"pyproject declares extras {missing} that the generated table omits"


def test_a_training_engine_is_not_advertised_against_an_untrainable_backend():
    """``trainer_for`` raises for chgnet / orb / sevenn — the table must not contradict it.

    Guarding the one derived judgement in the hook. Everything else it prints is copied
    from the registry; this row is filtered, so it is the row that can be wrong.
    """
    row = next(line for line in _rows(_render("backends")) if "`mlip-chgnet`" in line)
    assert "`mlip-train`" not in row, (
        "the backend table lists `mlip-train` against a backend with no trainer; "
        "a step naming it fails with ConfigError"
    )


def test_an_unknown_directive_is_a_build_failure_not_a_silent_comment():
    """A typo must not render as an HTML comment nobody sees."""
    with pytest.raises(ValueError, match="unknown directive"):
        _hook().on_page_markdown("<!-- chemrefine:enignes -->")


def test_a_directive_inside_prose_is_left_alone():
    """The hook's own documentation mentions the directives; they must not expand."""
    prose = "write `<!-- chemrefine:engines -->` on a line of its own"
    assert _hook().on_page_markdown(prose) == prose


@pytest.mark.skipif(not _RUN_FAILS.is_file(), reason="docs/ does not ship in the sdist")
def test_every_exit_code_is_documented():
    """The hand-written table's one machine-checkable claim: that it covers every code.

    Not generated — see this module's docstring. ``0`` is documented too and belongs to no
    exception, so the check runs one way: every code the package can exit with must appear.
    """
    text = _RUN_FAILS.read_text(encoding="utf-8")
    codes = {
        obj.exit_code
        for obj in vars(sys.modules["chemrefine.errors"]).values()
        if inspect.isclass(obj) and issubclass(obj, ChemRefineError)
    }
    missing = sorted(c for c in codes if not re.search(rf"^\| `{c}` \|", text, re.MULTILINE))
    assert not missing, (
        f"exit code(s) {missing} are raised by chemrefine.errors but have no row in "
        f"{_RUN_FAILS.name}"
    )
