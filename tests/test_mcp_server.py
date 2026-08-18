"""The MCP surface: same tools as the library, served with schemas, over a real session.

Driven through the SDK's in-memory ``Client`` — a genuine MCP session (initialize,
list, call, read) without a subprocess — so what these tests prove is what Claude Code
or any other client negotiates: every :data:`chemrefine.mcp_server.TOOLS` entry is
listed with a derived input schema and a docstring-sourced description, results come
back as structured content, a :class:`~chemrefine.errors.ChemRefineError` becomes a
tool *error result* (not a dead session), and the packaged guide is readable at
``chemrefine://guide``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from mcp.client import Client

from chemrefine import mcp_server
from chemrefine.cli import app


@pytest.fixture
def anyio_backend() -> str:
    """Run the anyio-marked tests on asyncio (the backend the CLI serves with)."""
    return "asyncio"


@pytest.mark.anyio
async def test_every_shared_tool_is_listed_with_schema_and_description():
    async with Client(mcp_server.build_server()) as client:
        listed = {t.name: t for t in (await client.list_tools()).tools}
    assert set(listed) == {t.__name__ for t in mcp_server.TOOLS}
    validate = listed["validate_config"]
    assert list(validate.input_schema["properties"]) == ["yaml_text", "base_dir"]
    assert validate.description is not None
    assert "never raises" in validate.description


@pytest.mark.anyio
async def test_a_call_rides_the_wire_and_returns_structured_content(tmp_path: Path):
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump({"steps": [{"step": 1, "engine": "fake"}]}), "utf-8")
    async with Client(mcp_server.build_server()) as client:
        result = await client.call_tool("validate_config_path", {"config_path": str(config)})
    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["ok"] is True


@pytest.mark.anyio
async def test_a_chemrefine_error_is_a_tool_error_not_a_dead_session():
    async with Client(mcp_server.build_server()) as client:
        bad = await client.call_tool(
            "start_run", {"config_path": "/nope.yaml", "action": "explode"}
        )
        assert bad.is_error is True
        assert "unknown action" in bad.content[0].text
        # The session survived the error — the next call answers normally.
        follow_up = await client.call_tool("list_engines", {})
        assert follow_up.is_error is False


@pytest.mark.anyio
async def test_the_guide_resource_serves_the_packaged_markdown():
    async with Client(mcp_server.build_server()) as client:
        resources = await client.list_resources()
        assert any(str(r.uri) == "chemrefine://guide" for r in resources.resources)
        read = await client.read_resource("chemrefine://guide")
    text = read.contents[0].text
    assert text == mcp_server.guide_text()
    assert "operating guide" in text
    assert "start_run is detached" not in text  # instructions live on the server, not the guide


def test_main_serves_stdio(monkeypatch: pytest.MonkeyPatch):
    """`main` runs the assembled server on stdio — the transport clients launch."""
    served: list[str] = []

    class _Stub:
        def run(self, transport: str) -> None:
            served.append(transport)

    monkeypatch.setattr(mcp_server, "build_server", _Stub)
    mcp_server.main()
    assert served == ["stdio"]


def test_cli_mcp_serves_stdio_via_the_module(monkeypatch: pytest.MonkeyPatch):
    """`chemrefine mcp` hands off to mcp_server.main — patched here so nothing blocks."""
    from typer.testing import CliRunner

    called: list[bool] = []
    monkeypatch.setattr(mcp_server, "main", lambda: called.append(True))
    result = CliRunner().invoke(app, ["mcp"])
    assert result.exit_code == 0
    assert called == [True]


def test_cli_mcp_names_the_missing_extra(without_extra, caplog):
    """Without the SDK the command says exactly what to install, and exits 1.

    This guard was the one that worked — ``mcp_server`` imports the SDK at module scope,
    which is the rule its docstring states and the other two commands had drifted from.
    It is tested honestly now for the same reason they are: refusing our own module, and
    leaving it cached, let the command run for real and still report success.
    """
    from typer.testing import CliRunner

    without_extra("mcp", purge=("chemrefine.mcp_server",))
    result = CliRunner().invoke(app, ["mcp"])
    assert result.exit_code == 1
    assert "chemrefine[mcp]" in caplog.text
