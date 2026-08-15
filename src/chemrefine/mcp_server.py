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

from mcp.server import MCPServer

from chemrefine import __version__

# Explicit re-export aliases: tests and docs address the surface as mcp_server.TOOLS /
# mcp_server.guide_text, and `no_implicit_reexport` requires the spelling to say so.
from chemrefine.agent_tools import TOOLS as TOOLS
from chemrefine.agent_tools import guide_text as guide_text


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
        server.tool()(tool)

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
