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
from chemrefine.errors import ChemRefineError
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


def test_every_recovery_action_is_taught():
    for action in agent_tools._ACTIONS:
        assert action in GUIDE, action


def test_the_failure_vocabulary_matches_the_ledger():
    """The triage table quotes the ledger's exact wording — the strings agents will see."""
    for kind in FailureKind:
        assert f"`{kind.value}`" in GUIDE, kind


def test_the_exit_codes_listed_are_the_real_ones():
    for cls in ChemRefineError.__subclasses__():
        assert str(cls.exit_code) in GUIDE, cls.__name__


@pytest.mark.parametrize("mention", sorted(set(_TOOL_SHAPED.findall(GUIDE))))
def test_no_tool_shaped_mention_is_a_phantom(mention: str):
    """Every tool-looking name in the guide must be a registered tool.

    The phantom direction is the dangerous one: prose renaming survived every other
    gate once (`MUTATING_TOOLS`'s "save" entry), and a model told about a tool that
    does not exist will call it and burn a retry loop learning otherwise.
    """
    real = {tool.__name__ for tool in agent_tools.TOOLS}
    assert mention in real, f"the guide mentions {mention!r}; registered tools: {sorted(real)}"
