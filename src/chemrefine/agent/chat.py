"""The terminal REPL around the harness — read, run, print, remember.

Deliberately small: history is PydanticAI's own message list threaded between turns,
confirmation is a ``typer.confirm`` naming the tool and its arguments, and the loop
ends on ``exit`` / ``quit`` / EOF. Everything with judgment in it lives in the harness
and the tools; everything with a network in it lives in the model the caller resolved.
"""

from __future__ import annotations

import typer

from chemrefine.agent.harness import TurnFailed, build_agent, run_turn
from chemrefine.agent.providers import ProviderConfig


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
        # endpoint dying mid-chat (the very case `--check` preflights), which the web
        # harness answers per turn too (:func:`chemrefine.gui.app.create_app`). A tool's
        # refusal never lands here — it is the tool's answer to the model. What the turn
        # did before it failed is the conversation the next one continues.
        try:
            result = run_turn(agent, prompt, message_history=history)
        except TurnFailed as e:
            typer.echo(f"model endpoint failed: {e}", err=True)
            history = e.messages
            continue
        typer.echo(result.output)
        history = result.all_messages()
