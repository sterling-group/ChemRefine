"""Assemble the PydanticAI agent: shared tools, gated mutations, packaged knowledge.

The harness registers :data:`chemrefine.agent_tools.TOOLS` verbatim — the same
functions the MCP server serves, schemas derived from the same signatures — and wraps
exactly the :data:`~chemrefine.agent_tools.MUTATING_TOOLS` in a confirmation callback:
where an MCP client's own approval dialog gates a mutating call, the embedded chat has
no client in front of it, so the gate lives here. A declined call returns a structured
``{"denied": …}`` result the model can read and continue from — a refusal is an answer,
not a crash — which also bounds what a prompt-injected instruction can do: nothing
mutating happens without the human's yes.

Instructions = the packaged agent guide (:func:`chemrefine.agent_tools.guide_text`,
identical bytes to the MCP resource) plus the session context (which config file, if
any). The model is a parameter, never constructed here — that is what lets the tests
run the entire harness against ``TestModel``/``FunctionModel`` offline.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from chemrefine import agent_tools

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.models import Model

ConfirmFn = Callable[[str, str], bool]
"""``(tool_name, rendered_args) -> allow?`` — the chat asks the human, tests script it."""


def _gated(fn: Callable[..., Any], confirm: ConfirmFn) -> Callable[..., Any]:
    """Wrap a mutating tool so nothing happens before the human says yes.

    ``functools.wraps`` keeps the name, docstring and (via ``__wrapped__``) the
    signature, so the model sees the identical tool schema either way.
    """

    @functools.wraps(fn)
    def gate(*args: Any, **kwargs: Any) -> Any:
        rendered = json.dumps(kwargs) if kwargs else json.dumps(list(args))
        if not confirm(fn.__name__, rendered):
            return {"denied": f"the user declined {fn.__name__}; ask what they would like instead"}
        return fn(*args, **kwargs)

    return gate


def instructions(config_path: str | None) -> str:
    """The system prompt: the packaged guide, the gate contract, the session context."""
    parts = [
        agent_tools.guide_text(),
        (
            "You are running inside `chemrefine agent`, a terminal chat. Mutating tools "
            "ask the user for confirmation before executing; a `denied` result means "
            "they said no — adjust course, never retry the same call unprompted. Show "
            "the YAML and wait for a go-ahead before start_run."
        ),
    ]
    if config_path is not None:
        parts.append(
            f"The user's config file for this session: {config_path} — summarize_config "
            "and run_status are the right first calls."
        )
    return "\n\n".join(parts)


def build_agent(
    model: Model | str,
    *,
    confirm: ConfirmFn,
    config_path: str | None = None,
) -> Agent[None, str]:
    """The assembled agent — every shared tool registered, mutations behind ``confirm``."""
    from pydantic_ai import Agent

    agent: Agent[None, str] = Agent(model, instructions=instructions(config_path))
    for tool in agent_tools.TOOLS:
        if tool.__name__ in agent_tools.MUTATING_TOOLS:
            agent.tool_plain(_gated(tool, confirm))
        else:
            agent.tool_plain(tool)
    return agent


def build_web_agent(
    model: Model | str,
    *,
    config_path: str | None = None,
) -> Agent[None, Any]:
    """The deferred-approval variant, for harnesses that cannot block on a prompt.

    The terminal chat's gate is a blocking ``confirm`` — impossible in the middle of an
    HTTP request. Here the mutating tools are registered with
    ``requires_approval=True`` instead: a gated call *suspends* the run, the result
    comes back as a ``DeferredToolRequests`` naming each call and its arguments (the
    GUI renders allow/deny cards), and the next request resumes the same run with a
    ``DeferredToolResults`` verdict — the tool executes only on an explicit yes,
    exactly the guarantee the terminal gate gives, enforced by the SDK rather than a
    wrapper.
    """
    from pydantic_ai import Agent, DeferredToolRequests

    agent: Agent[None, Any] = Agent(
        model,
        instructions=instructions(config_path),
        output_type=[str, DeferredToolRequests],
    )
    for tool in agent_tools.TOOLS:
        register = agent.tool_plain(requires_approval=tool.__name__ in agent_tools.MUTATING_TOOLS)
        register(tool)
    return agent
