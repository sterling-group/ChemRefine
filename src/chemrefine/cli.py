"""Typer-based command-line interface for ChemRefine.

This is the **only** module that calls :func:`sys.exit` or reads
``sys.argv``. Subcommands map one-to-one to
:class:`~chemrefine.recovery.Action` values:

* ``chemrefine run CONFIG`` — full pipeline from step 1 (caches invalidated).
* ``chemrefine resume CONFIG`` — honor existing cache where possible.
* ``chemrefine rebuild-cache CONFIG [STEP]`` — invalidate one step's
  cache, then resume.
* ``chemrefine rebuild-nms CONFIG [STEP]`` — same as rebuild-cache but
  semantically scoped to normal-mode-sampling rebuilds (engines may
  treat it differently in their parse step).
* ``chemrefine rerun CONFIG [STEP]`` — re-execute a step (alias of
  rebuild-cache for now; will resubmit failed jobs once the recovery
  module learns about ``failed_jobs.json``).

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
from chemrefine.config import Config, load_config
from chemrefine.errors import ChemRefineError
from chemrefine.recovery import Action, execute

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


def _load(config_path: Path, *, maxcores: int | None) -> Config:
    """Load + validate ``config_path`` and apply the ``--maxcores`` override."""
    cfg = load_config(config_path)
    if maxcores is not None:
        cfg = cfg.model_copy(update={"max_cores": maxcores})
    return cfg


def _dispatch(action: Action, config: Config, *, target: str | None, dry_run: bool) -> int:
    """Either describe the would-be execution (``--dry-run``) or run it."""
    if dry_run:
        typer.echo(f"[dry-run] action={action.value}")
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
    try:
        return execute(config, action, target=target)
    except ChemRefineError as e:
        logger.error("%s: %s", type(e).__name__, e)
        raise typer.Exit(code=e.exit_code) from e


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


ConfigArg = Annotated[Path, typer.Argument(..., exists=True, dir_okay=False, readable=True)]
MaxCoresOpt = Annotated[
    int | None,
    typer.Option("--maxcores", help="Override max_cores from the YAML."),
]
DryRunOpt = Annotated[
    bool, typer.Option("--dry-run", help="Validate the config and describe actions; do not execute.")
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
    cfg = _load(config_path, maxcores=maxcores)
    raise typer.Exit(_dispatch(Action.RUN, cfg, target=None, dry_run=dry_run))


@app.command()
def resume(
    config_path: ConfigArg,
    maxcores: MaxCoresOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Resume the pipeline, hitting the on-disk cache for unchanged steps."""
    cfg = _load(config_path, maxcores=maxcores)
    raise typer.Exit(_dispatch(Action.RESUME, cfg, target=None, dry_run=dry_run))


@app.command("rebuild-cache")
def rebuild_cache(
    config_path: ConfigArg,
    target: TargetArg = None,
    maxcores: MaxCoresOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Invalidate one step's cache (default: latest step), then resume."""
    cfg = _load(config_path, maxcores=maxcores)
    raise typer.Exit(_dispatch(Action.REBUILD_CACHE, cfg, target=target, dry_run=dry_run))


@app.command("rebuild-nms")
def rebuild_nms(
    config_path: ConfigArg,
    target: TargetArg = None,
    maxcores: MaxCoresOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Rebuild one step's normal-mode-sampling output, then resume."""
    cfg = _load(config_path, maxcores=maxcores)
    raise typer.Exit(_dispatch(Action.REBUILD_NMS, cfg, target=target, dry_run=dry_run))


@app.command()
def rerun(
    config_path: ConfigArg,
    target: TargetArg = None,
    maxcores: MaxCoresOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Re-execute one step (default: latest), resubmitting jobs as needed."""
    cfg = _load(config_path, maxcores=maxcores)
    raise typer.Exit(_dispatch(Action.RERUN, cfg, target=target, dry_run=dry_run))


# ---------------------------------------------------------------------------
# pyproject ``[project.scripts]`` entry point
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover - re-export for legacy entry-point imports
    """Backstop entry point for environments that import ``chemrefine.cli:main``."""
    app()


if __name__ == "__main__":
    app()
