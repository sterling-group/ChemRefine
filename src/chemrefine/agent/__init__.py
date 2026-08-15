"""The embedded chat agent (``chemrefine agent``) — the same tools, no MCP client needed.

For the user without an agent client: a terminal chat that authors and triages
ChemRefine workflows through exactly the surface the MCP server exposes
(:data:`chemrefine.agent_tools.TOOLS`), harnessed by PydanticAI. Split by concern:
:mod:`.providers` resolves *which model* (flags → environment → preset, any
OpenAI-compatible endpoint including local Ollama/vLLM), :mod:`.harness` builds the
Agent (tools registered verbatim, the mutating ones behind a confirmation callback,
instructions = the packaged guide), :mod:`.chat` is the REPL. The model is
interchangeable by construction — tests swap in PydanticAI's ``TestModel`` /
``FunctionModel`` and run the whole harness offline.
"""
