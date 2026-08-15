"""The MCP server — :mod:`chemrefine.agent_tools` served over the Model Context Protocol.

Thin by design: every tool is a function from :mod:`chemrefine.agent_tools` registered
verbatim (the SDK derives each tool's schema from the signature and its description from
the docstring), so the MCP surface and the embedded agent's surface are one list —
:data:`TOOLS` — and cannot drift. The only content of its own this module serves is the
``chemrefine://guide`` resource: the packaged operating guide
(``data/agent_guide.md``), knowledge shipped *beside* the tools so any client model can
load ChemRefine's recipes without them being welded into a harness.

Run it with ``chemrefine mcp`` (stdio transport — what ``claude mcp add`` and every
desktop client speak; launch over SSH for a cluster: ``ssh login-node chemrefine mcp``).
This module imports the ``mcp`` SDK at import time and is itself imported lazily by the
CLI, which turns an absent SDK into the actionable "pip install 'chemrefine[mcp]'"
message instead of a traceback.
"""

from __future__ import annotations

from importlib import resources

from mcp.server import MCPServer

from chemrefine import __version__, agent_tools

TOOLS = (
    agent_tools.get_schema,
    agent_tools.list_engines,
    agent_tools.validate_config,
    agent_tools.validate_config_path,
    agent_tools.summarize_config,
    agent_tools.read_template,
    agent_tools.write_template,
    agent_tools.scaffold_templates,
    agent_tools.start_run,
    agent_tools.run_status,
    agent_tools.get_results,
    agent_tools.get_failures,
    agent_tools.lookup_smiles,
    agent_tools.build_structures,
    agent_tools.get_frequencies,
    agent_tools.analyze_mode,
)
"""Every tool the server exposes — one shared list, imported by the embedded agent too."""


def guide_text() -> str:
    """The packaged agent guide, read from the wheel's ``data/`` directory."""
    return (
        resources.files("chemrefine").joinpath("data/agent_guide.md").read_text(encoding="utf-8")
    )


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
