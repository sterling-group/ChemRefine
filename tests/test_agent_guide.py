"""The packaged agent guide must speak only vocabulary the code actually has.

The guide rides every agent system prompt and the ``chemrefine://guide`` MCP resource,
so a stale line in it becomes a model confidently calling a tool that does not exist or
recommending an operation nothing parses. Same spirit as the docs drift guard: each
closed vocabulary the guide draws on — tool names, operations, recovery actions, the
failure-kind ledger wording, the exit codes — is pinned both ways where it matters,
with the phantom direction (guide names something the code lost) covered for the
riskiest class, tool names.
"""

from __future__ import annotations

import re

import pytest

from chemrefine import agent_tools
from chemrefine.engines.orca.output.coordinator import known_operations
from chemrefine.state import FailureKind

GUIDE = agent_tools.guide_text()

_TOOL_SHAPED = re.compile(
    r"\b(?:get|list|validate|save|read|write|scaffold|start|run|build|lookup|analyze)"
    r"_[a-z_]+\b"
)


def test_every_operation_is_taught():
    """A model that never hears of `solvator` will never use it."""
    for operation in sorted(known_operations()):
        assert f"`{operation}`" in GUIDE or f"`operation: {operation}`" in GUIDE, operation


def test_every_action_start_run_accepts_is_taught():
    """`assert "run" in GUIDE` is true of any prose in English.

    ``_ACTIONS`` is the vocabulary ``start_run`` validates against, so a model that never
    hears ``rebuild-nms`` will never ask for it. The backticks are what make this a check
    about the word rather than about three letters that happen to occur. Named for
    ``start_run`` rather than for "recovery" because ``run`` is in the tuple and is not a
    recovery action — the old name made correct prose look like a bug.
    """
    for action in agent_tools._ACTIONS:
        assert f"`{action}`" in GUIDE, action


def test_the_failure_vocabulary_matches_the_ledger():
    """The triage table quotes the ledger's exact wording — the strings agents will see."""
    for kind in FailureKind:
        assert f"`{kind.value}`" in GUIDE, kind


def test_the_guide_points_at_the_exit_code_map_instead_of_copying_it():
    """A prose copy of a shipped map is a copy that drifts, and this one had.

    The guide listed codes 2-10 and never named the two classes that exit 1, while the
    check that used to be here — ``str(cls.exit_code) in GUIDE`` — could not see it: every
    digit it looked for already appears in the numbered working loop above. It also read
    ``__subclasses__()``, direct-only, so it never asked about ``OutputTerminationError``
    at all. ``get_failures`` ships the map itself, so what is left for the guide is to
    name the key; the human-facing table with causes and remedies lives in
    ``docs/running/when-a-run-fails.md``, guarded there.
    """
    assert "`exit_codes`" in GUIDE


@pytest.mark.parametrize("mention", sorted(set(_TOOL_SHAPED.findall(GUIDE))))
def test_no_tool_shaped_mention_is_a_phantom(mention: str):
    """Every tool-looking name in the guide must be a registered tool.

    The phantom direction is the dangerous one: prose renaming survived every other
    gate once (`MUTATING_TOOLS`'s "save" entry), and a model told about a tool that
    does not exist will call it and burn a retry loop learning otherwise.
    """
    real = {tool.__name__ for tool in agent_tools.TOOLS}
    assert mention in real, f"the guide mentions {mention!r}; registered tools: {sorted(real)}"
