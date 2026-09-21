"""The MCP server — :mod:`chemrefine.agent_tools` served over the Model Context Protocol.

Thin by design: every tool is a function from :mod:`chemrefine.agent_tools` registered
verbatim (the SDK derives each tool's schema from the signature and its description from
the docstring), so the MCP surface and the embedded agent's surface are one list —
:data:`chemrefine.agent_tools.TOOLS` — and cannot drift. The only content of its own
this module serves is the ``chemrefine://guide`` resource: the packaged operating guide
(``data/agent_guide.md``), knowledge shipped *beside* the tools so any client model can
load ChemRefine's recipes without them being welded into a harness.

Run it with ``chemrefine mcp`` (stdio transport — what ``claude mcp add`` and every
desktop client speak; launch over SSH for a cluster: ``ssh login-node chemrefine mcp``).
This module imports the ``mcp`` SDK at import time and is itself imported lazily by the
CLI, which turns an absent SDK into the actionable "pip install 'chemrefine[mcp]'"
message instead of a traceback.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from chemrefine import __version__

# Explicit re-export aliases: tests and docs address the surface as mcp_server.TOOLS /
# mcp_server.guide_text, and `no_implicit_reexport` requires the spelling to say so.
from chemrefine.agent_tools import TOOLS as TOOLS
from chemrefine.agent_tools import guide_text as guide_text
from chemrefine.errors import ChemRefineError


def actionable(tool: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a tool so a :class:`~chemrefine.errors.ChemRefineError` reaches the model.

    The SDK distinguishes an error a tool raised *on purpose* from a crash, and only the
    first keeps its text: a ``ToolError`` is relayed as an ``is_error`` result carrying the
    message, while any other exception becomes a generic "Error executing tool <name>" with
    the detail deliberately left on the server. That is the right default — an arbitrary
    traceback can carry paths and environment — and it is the wrong answer for
    :class:`ChemRefineError`, which exists precisely to be shown: it is the class every
    config mistake, missing template and unknown action is raised as, and its message names
    what to change.

    Unwrapped, ``action: explode`` reaches an agent as "Error executing tool start_run" and
    the sentence listing the valid actions never leaves the process, so the model has nothing
    to act on and no reason to believe a retry would differ.

    Only ``ChemRefineError`` is translated. A genuine crash keeps the generic message, which
    is the SDK's hygiene and worth keeping — this says which of our exceptions are answers to
    the caller rather than failures of the server.

    :func:`functools.wraps` carries ``__wrapped__``, so the SDK still derives the tool's
    schema from the original signature and its description from the original docstring.
    """

    @functools.wraps(tool)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return tool(*args, **kwargs)
        except ChemRefineError as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


def build_server() -> MCPServer:
    """Assemble the server: all :data:`TOOLS` plus the ``chemrefine://guide`` resource."""
    server = MCPServer(
        name="chemrefine",
        version=__version__,
        instructions=(
            "Author and drive reproducible ChemRefine refinement pipelines. Read the "
            "chemrefine://guide resource first: it carries the working loop "
            "(schema → validate → scaffold → run → status → results/failures) and the "
            "chemistry recipes (conformer funnel, TS validation via imaginary-mode "
            "analysis). start_run is detached — poll run_status; never busy-wait."
        ),
    )
    for tool in TOOLS:
        server.tool()(actionable(tool))

    @server.resource(
        "chemrefine://guide",
        name="ChemRefine agent guide",
        description="Operating loop, chemistry recipes, and hard rules for driving ChemRefine.",
        mime_type="text/markdown",
    )
    def guide() -> str:
        """Serve the packaged guide verbatim."""
        return guide_text()

    return server


def main() -> None:
    """Serve on stdio — the transport every MCP client launcher speaks."""
    build_server().run(transport="stdio")
