"""The terminal REPL around the harness — read, run, print, remember.

Deliberately small: history is PydanticAI's own message list threaded between turns,
confirmation is a ``typer.confirm`` naming the tool and its arguments, and the loop
ends on ``exit`` / ``quit`` / EOF. Everything with judgment in it lives in the harness
and the tools; everything with a network in it lives in the model the caller resolved.
"""

from __future__ import annotations

import typer

from chemrefine.agent.harness import build_agent
from chemrefine.agent.providers import ProviderConfig
from chemrefine.errors import ChemRefineError


def _confirm(tool_name: str, rendered_args: str) -> bool:
    """The human gate: name the mutation, show the arguments, ask."""
    return typer.confirm(f"allow {tool_name}({rendered_args})?", default=False)


def main(
    provider: str = "custom",
    model: str | None = None,
    base_url: str | None = None,
    config_path: str | None = None,
) -> None:
    """Run the chat until the user leaves; one agent, one growing history."""
    resolved = ProviderConfig.resolve(provider, model=model, base_url=base_url)
    agent = build_agent(resolved.build_model(), confirm=_confirm, config_path=config_path)
    typer.echo(f"ChemRefine agent — model {resolved.model} (exit/quit to leave)")
    history = None
    while True:
        try:
            prompt = typer.prompt("chemrefine", prompt_suffix="> ")
        except (EOFError, typer.Abort):
            break
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        # A failed turn must not end the session: the likeliest failure here is the model
        # endpoint dying mid-chat (the very case `--check` preflights), and before this
        # guard it took the whole conversation down as a traceback — where the web harness
        # answers the same failures per turn (:func:`chemrefine.gui.app.create_app`).
        try:
            result = agent.run_sync(prompt, message_history=history)
        except ChemRefineError as e:
            # A tool's own failure keeps its documented name-and-message shape.
            typer.echo(f"{type(e).__name__}: {e}", err=True)
            continue
        except Exception as e:  # the model endpoint is the outside world
            typer.echo(f"model endpoint failed: {e}", err=True)
            continue
        typer.echo(result.output)
        history = result.all_messages()
