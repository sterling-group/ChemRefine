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
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from chemrefine import agent_tools
from chemrefine.engines import known_backend_extras
from chemrefine.engines.api import ENGINES
from chemrefine.engines.mlip.registry import registered_backends
from chemrefine.errors import EXIT_CODES

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HOOK = _REPO_ROOT / "docs" / "hooks" / "tables.py"
_RUN_FAILS = _REPO_ROOT / "docs" / "running" / "when-a-run-fails.md"
_AGENTS = _REPO_ROOT / "docs" / "workflow" / "agents.md"

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


def test_the_training_engine_is_advertised_exactly_where_a_trainer_exists():
    """``mlip-train`` appears on a backend's row iff one of its tasks has a trainer.

    Guarding the one derived judgement in the hook — everything else it prints is copied
    from the registry; this row is filtered, so it is the row that can be wrong, in either
    direction: advertising a step that would fail with ``ConfigError``, or hiding one that
    runs. Derived from the registry rather than naming a backend, because *which* backends
    train is the roster's business and has already changed once under a test that froze it.
    """
    from chemrefine.engines.mlip.registry import (
        backend_spec,
        registered_backends,
        registered_trainers,
    )

    trainable_extras = {backend_spec(t).extra for t in registered_trainers()}
    mlip_extras = {backend_spec(t).extra for t in registered_backends()}
    checked = 0
    for line in _rows(_render("backends")):
        extra = next((e for e in mlip_extras if f"`{e}`" in line), None)
        if extra is None:
            continue
        checked += 1
        assert ("`mlip-train`" in line) == (extra in trainable_extras), (
            f"{extra}: the table and trainer_for disagree about mlip-train"
        )
    assert checked == len(mlip_extras), "some registered backend never appeared in the table"


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
    missing = sorted(
        c for c in set(EXIT_CODES.values()) if not re.search(rf"^\| `{c}` \|", text, re.MULTILINE)
    )
    assert not missing, (
        f"exit code(s) {missing} are raised by chemrefine.errors but have no row in "
        f"{_RUN_FAILS.name}"
    )


@pytest.mark.skipif(not _AGENTS.is_file(), reason="docs/ does not ship in the sdist")
def test_the_confirmation_gate_names_every_mutating_tool_and_no_others():
    """Guarded, not generated — the roster is a clause inside a sentence that teaches.

    The page stated it twice, a hundred lines apart, and the two copies had drifted in
    opposite directions: one omitted ``build_structures``, the other ``save_config``, and
    both claimed to be exhaustive. A directive cannot fix that without breaking two
    sentences in half for a five-item list, so one copy is gone and the survivor is pinned
    to ``MUTATING_TOOLS`` in both directions. Backticked names are filtered by membership
    in ``TOOLS`` so the ``allow start_run(...)`` sample and ordinary prose are ignored.
    """
    section = _AGENTS.read_text(encoding="utf-8").split("## The confirmation gate", 1)[1]
    section = section.split("\n## ", 1)[0]
    tools = {tool.__name__ for tool in agent_tools.TOOLS}
    named = {name for name in re.findall(r"`([a-z_]+)`", section) if name in tools}
    assert named == set(agent_tools.MUTATING_TOOLS), (
        "the confirmation-gate section must name exactly the mutating tools; it names "
        f"{sorted(named)}, MUTATING_TOOLS is {sorted(agent_tools.MUTATING_TOOLS)}"
    )
