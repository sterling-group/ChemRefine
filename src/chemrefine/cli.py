"""Typer-based command-line interface for ChemRefine.

This is the **only** module that calls :func:`sys.exit` or reads
``sys.argv``. Subcommands map one-to-one to
:class:`~chemrefine.recovery.Action` values:

* ``chemrefine run CONFIG`` — full pipeline from step 1 (caches invalidated).
* ``chemrefine resume CONFIG`` — honor existing cache and re-attempt the pending
  failed jobs of any ``on_failure: stop`` step, then continue.
* ``chemrefine rerun-errors CONFIG [STEP]`` — re-attempt only one step's pending
  failed jobs (latest if no STEP); like ``resume`` but scoped to that step.
* ``chemrefine rerun CONFIG [STEP]`` — redo one whole step from scratch
  (others cache-hit).
* ``chemrefine rebuild-cache CONFIG [STEP]`` — rebuild one step's cache from
  outputs already on disk (parse only, no submission).
* ``chemrefine rebuild-nms CONFIG [STEP]`` — re-run the NMS step with the
  current options (a named alias of ``rerun``).

Per-step ``on_failure: stop | skip | best`` (in the YAML) decides in-run
behaviour: ``skip`` (default) drops the failures and continues, ``best`` keeps
all (backfilling the best geometry), ``stop`` halts the run after caching the
step's successes. The ``failed_jobs.json`` ledger always records which
structures failed (so they're visible), but only ``stop`` failures are pending
for ``resume`` / ``rerun-errors`` to re-attempt.

Legacy v1.3.1 flag-style invocations (``chemrefine CONFIG --rebuild_cache N``,
``--rerun_errors N``, ``--skip``, …) are translated to these subcommands by
:func:`_translate_legacy_argv` before Typer parses.

Global flags: ``--maxcores INT`` overrides ``max_cores`` in the YAML;
``--dry-run`` loads and validates the config without executing; ``-v``
turns on debug logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from chemrefine import __version__
from chemrefine.errors import ChemRefineError

logger = logging.getLogger("chemrefine")

app = typer.Typer(
    name="chemrefine",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
    help="Automated and interoperable manager for computational chemistry workflows.",
)


# ---------------------------------------------------------------------------
# Shared callback (global flags) + version
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    """Eager Typer callback for ``--version`` — print and exit."""
    if value:
        typer.echo(f"ChemRefine {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable debug-level logging.")
    ] = False,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            help="Show ChemRefine version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Configure logging for the rest of the invocation."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def execute(config, action, target: str | None = None) -> int:
    """Run a recovery action — a thin, patchable indirection to
    :func:`chemrefine.recovery.execute` (``config`` is a
    :class:`~chemrefine.config.Config`, ``action`` a
    :class:`~chemrefine.recovery.Action`; both stay un-annotated so the heavy
    imports remain lazy rather than relying on a ``TYPE_CHECKING`` block).

    Kept at module scope (rather than imported inside ``_dispatch``) so the heavy
    ``recovery`` import stays lazy yet ``cli.execute`` remains a stable monkeypatch
    target for tests.
    """
    from chemrefine import recovery

    return recovery.execute(config, action, target=target)


def _load(config_path: Path, *, maxcores: int | None):
    """Load + validate ``config_path`` and apply the ``--maxcores`` override.

    Returns a :class:`~chemrefine.config.Config` (un-annotated to keep the
    pydantic/config import lazy — it loads only when a command actually runs).
    """
    from chemrefine.config import load_config

    cfg = load_config(config_path)
    if maxcores is not None:
        cfg = cfg.model_copy(update={"max_cores": maxcores})
    return cfg


def _dispatch(
    action_name: str,
    config_path: Path,
    *,
    maxcores: int | None,
    target: str | None,
    dry_run: bool,
) -> int:
    """Load the config, then either describe the would-be execution (``--dry-run``) or run it.

    ``action_name`` is the subcommand string (== the :class:`Action` value); the
    enum is resolved lazily so a plain ``--dry-run`` never imports ``recovery``.
    Loading happens inside the same handler as execution so **every**
    :class:`ChemRefineError` — a malformed config included — exits with its
    documented ``exit_code`` (see :mod:`chemrefine.errors`) instead of escaping
    as a traceback.
    """
    try:
        config = _load(config_path, maxcores=maxcores)
        if dry_run:
            typer.echo(f"[dry-run] action={action_name}")
            typer.echo(f"[dry-run] output_dir={config.output_dir}")
            typer.echo(f"[dry-run] max_cores={config.max_cores}")
            for step_cfg in config.steps:
                typer.echo(
                    f"[dry-run] step {step_cfg.step}: "
                    f"{step_cfg.dir_name()} engine={step_cfg.engine} op={step_cfg.operation}"
                )
            if target is not None:
                typer.echo(f"[dry-run] target step: {target}")
            return 0
        from chemrefine.recovery import Action

        return execute(config, Action(action_name), target=target)
    except ChemRefineError as e:
        logger.error("%s: %s", type(e).__name__, e)
        raise typer.Exit(code=e.exit_code) from e


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


ConfigArg = Annotated[Path, typer.Argument(..., exists=True, dir_okay=False, readable=True)]
MaxCoresOpt = Annotated[
    int | None,
    # min=1 mirrors Config's ``max_cores: ge=1`` — the override is applied via
    # ``model_copy`` (no re-validation), so the flag must reject 0/negative itself.
    typer.Option("--maxcores", min=1, help="Override max_cores from the YAML."),
]
DryRunOpt = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        help="Validate the config and describe actions; do not execute.",
    ),
]
TargetArg = Annotated[
    str | None,
    typer.Argument(help="Step number or name to target. Defaults to the last step."),
]


@app.command()
def run(
    config_path: ConfigArg,
    maxcores: MaxCoresOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Run the full pipeline from step 1, invalidating any existing cache."""
    raise typer.Exit(_dispatch("run", config_path, maxcores=maxcores, target=None, dry_run=dry_run))


@app.command()
def resume(
    config_path: ConfigArg,
    maxcores: MaxCoresOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Resume the pipeline, hitting the on-disk cache for unchanged steps."""
    raise typer.Exit(
        _dispatch("resume", config_path, maxcores=maxcores, target=None, dry_run=dry_run)
    )


@app.command("rebuild-cache")
def rebuild_cache(
    config_path: ConfigArg,
    target: TargetArg = None,
    maxcores: MaxCoresOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Rebuild one step's cache from existing outputs (default: latest); no submission."""
    raise typer.Exit(
        _dispatch("rebuild-cache", config_path, maxcores=maxcores, target=target, dry_run=dry_run)
    )


@app.command("rebuild-nms")
def rebuild_nms(
    config_path: ConfigArg,
    target: TargetArg = None,
    maxcores: MaxCoresOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Re-run the NMS step with the current options (alias of rerun)."""
    raise typer.Exit(
        _dispatch("rebuild-nms", config_path, maxcores=maxcores, target=target, dry_run=dry_run)
    )


@app.command()
def rerun(
    config_path: ConfigArg,
    target: TargetArg = None,
    maxcores: MaxCoresOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Redo one whole step from scratch (default: latest); others cache-hit."""
    raise typer.Exit(
        _dispatch("rerun", config_path, maxcores=maxcores, target=target, dry_run=dry_run)
    )


@app.command("rerun-errors")
def rerun_errors(
    config_path: ConfigArg,
    target: TargetArg = None,
    maxcores: MaxCoresOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Re-attempt only one step's pending failed jobs (default: latest)."""
    raise typer.Exit(
        _dispatch("rerun-errors", config_path, maxcores=maxcores, target=target, dry_run=dry_run)
    )


# ---------------------------------------------------------------------------
# Legacy (v1.3.1) flag-style CLI → subcommand translation
# ---------------------------------------------------------------------------

_SUBCOMMANDS = frozenset({"run", "resume", "rerun", "rerun-errors", "rebuild-cache", "rebuild-nms"})


def _translate_legacy_argv(argv: list[str]) -> list[str]:
    """Map a v1.3.1 flag-style invocation to the new subcommand argv.

    The single home for the old flags. New-style argv (first positional is a
    known subcommand, or ``--version`` / ``--help`` / no positional) is returned
    unchanged. Otherwise the old flags are parsed and rewritten:
    ``CONFIG``→``run``; ``--skip``→``resume``; ``--rebuild_cache [N]``→
    ``rebuild-cache``; ``--rebuild_nms [N]``→``rebuild-nms``; ``--rerun_errors
    [N]``→``rerun-errors``; ``--maxcores`` carried through.
    """
    if any(a in ("--version", "--help", "-h") for a in argv):
        return argv
    first_positional = next((a for a in argv if not a.startswith("-")), None)
    if first_positional is None or first_positional in _SUBCOMMANDS:
        return argv

    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("input_yaml")
    parser.add_argument("--maxcores", type=int)
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--rebuild_cache", nargs="?", const=True, type=int, default=False)
    parser.add_argument("--rebuild_nms", nargs="?", const=True, type=int, default=False)
    parser.add_argument("--rerun_errors", nargs="?", const=True, type=int, default=False)
    parser.add_argument("-v", "--verbose", action="store_true")
    try:
        ns, _unknown = parser.parse_known_args(argv)
    except SystemExit:
        return argv  # malformed legacy args — let Typer surface the error

    def _step(value: object) -> list[str]:
        return [str(value)] if isinstance(value, int) and not isinstance(value, bool) else []

    if ns.rebuild_cache is not False:
        command, step = "rebuild-cache", _step(ns.rebuild_cache)
    elif ns.rebuild_nms is not False:
        command, step = "rebuild-nms", _step(ns.rebuild_nms)
    elif ns.rerun_errors is not False:
        command, step = "rerun-errors", _step(ns.rerun_errors)
    elif ns.skip:
        command, step = "resume", []
    else:
        command, step = "run", []

    new_argv = (["-v"] if ns.verbose else []) + [command, ns.input_yaml, *step]
    if ns.maxcores is not None:
        new_argv += ["--maxcores", str(ns.maxcores)]
    logger.warning("legacy CLI flags detected; mapped to `chemrefine %s`", " ".join(new_argv))
    return new_argv


def main() -> None:
    """Entry point: translate any legacy flag-style argv, then run the Typer app."""
    import sys

    sys.argv[1:] = _translate_legacy_argv(sys.argv[1:])
    app()
