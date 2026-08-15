"""Typer-based command-line interface for ChemRefine.

This is the **only** module that calls :func:`sys.exit` or reads
``sys.argv``. Subcommands map one-to-one to
:class:`~chemrefine.recovery.Action` values:

* ``chemrefine run CONFIG`` — full pipeline from step 1 (caches invalidated).
* ``chemrefine resume CONFIG`` — honor existing cache and re-attempt the pending
  failed jobs of any ``on_failure: stop`` step, then continue.
* ``chemrefine rerun-errors CONFIG [STEP]`` — re-attempt one step's pending
  failed jobs (latest if no STEP), then continue: earlier steps cache-hit and
  submit nothing, later ones resume.
* ``chemrefine rerun CONFIG [STEP]`` — redo one whole step from scratch; every
  other step resumes, so one whose fingerprint no longer holds runs again.
* ``chemrefine rebuild-cache CONFIG [STEP]`` — rebuild one step's cache from
  outputs already on disk (parse only, no submission); the run ends there.
* ``chemrefine rebuild-nms CONFIG [STEP]`` — the same rebuild aimed at the NMS
  step: with no STEP it finds the one setting ``nms: true`` rather than taking
  the last. Round 1 is re-parsed and its displaced children re-read from disk,
  so re-resolving costs a read rather than a re-run of the frequencies.

``chemrefine backends {install,list,path}`` manages the per-backend environments
(conflicting MLIP stacks live in one managed env each, resolved by name — see
:mod:`chemrefine.engines._provision`).

``chemrefine schema`` / ``chemrefine engines [--json]`` print the machine-readable
config schema document and the engine registry (see :mod:`chemrefine.introspect`) —
what a GUI form or an agent reads instead of the prose docs.

Per-step ``on_failure: stop | skip | best`` (in the YAML) decides in-run
behaviour: ``stop`` (default) halts the run after caching the step's successes,
``skip`` drops the failures and continues, ``best`` keeps all (backfilling the
best geometry). The ``failed_jobs.json`` ledger always records which structures
failed (so they're visible), but only ``stop`` failures are pending for
``resume`` / ``rerun-errors`` to re-attempt.

Legacy v1.3.1 flag-style invocations (``chemrefine CONFIG --rebuild_cache N``,
``--rerun_errors N``, ``--skip``, …) are translated to these subcommands by
:func:`chemrefine.cli_legacy.translate_argv` before Typer parses.

Every run command takes ``--maxcores INT`` / ``--maxgpus INT``, which override
``max_cores`` / ``max_gpus`` in the YAML, and ``--dry-run``, which loads and
validates the config without executing. ``-v`` is the one global flag and goes
before the subcommand; it turns on debug logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from chemrefine import __version__, cli_legacy
from chemrefine.errors import ChemRefineError

if TYPE_CHECKING:
    from chemrefine.config import Config
    from chemrefine.recovery import Action

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


def execute(config: Config, action: Action, target: str | None = None) -> int:
    """Run a recovery action — a thin, patchable indirection to
    :func:`chemrefine.recovery.execute`. The annotations resolve under
    ``TYPE_CHECKING`` only, so the heavy imports stay lazy.

    Kept at module scope (rather than imported inside ``_dispatch``) so the heavy
    ``recovery`` import stays lazy yet ``cli.execute`` remains a stable monkeypatch
    target for tests.
    """
    from chemrefine import recovery

    return recovery.execute(config, action, target=target)


def _load(config_path: Path, *, maxcores: int | None, maxgpus: int | None) -> Config:
    """Load + validate ``config_path`` and apply the ``--maxcores`` / ``--maxgpus`` overrides.

    The return annotation resolves under ``TYPE_CHECKING`` only — the
    pydantic/config import stays lazy (it loads only when a command actually
    runs). Both flags beat the YAML; ``model_copy`` skips re-validation, so the
    flags enforce their own bounds (see :data:`MaxCoresOpt` / :data:`MaxGpusOpt`).
    """
    from chemrefine.config import load_config

    cfg = load_config(config_path)
    updates: dict[str, int] = {}
    if maxcores is not None:
        updates["max_cores"] = maxcores
    if maxgpus is not None:
        updates["max_gpus"] = maxgpus
    if updates:
        cfg = cfg.model_copy(update=updates)
    return cfg


def _dispatch(
    action_name: str,
    config_path: Path,
    *,
    maxcores: int | None,
    maxgpus: int | None,
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
        config = _load(config_path, maxcores=maxcores, maxgpus=maxgpus)
        if dry_run:
            typer.echo(f"[dry-run] action={action_name}")
            typer.echo(f"[dry-run] output_dir={config.output_dir}")
            typer.echo(f"[dry-run] max_cores={config.max_cores}")
            typer.echo(
                f"[dry-run] max_gpus={config.max_gpus if config.max_gpus is not None else 'auto'}"
            )
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
MaxGpusOpt = Annotated[
    int | None,
    # min=0 mirrors Config's ``max_gpus: ge=0``; applied via ``model_copy`` (no
    # re-validation), so the flag enforces its own bound. Beats the YAML, like
    # --maxcores; omit to keep the YAML's value (``None`` ⇒ auto-resolve).
    typer.Option("--maxgpus", min=0, help="Override max_gpus from the YAML."),
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
    maxgpus: MaxGpusOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Run the full pipeline from step 1, invalidating any existing cache."""
    raise typer.Exit(
        _dispatch(
            "run", config_path, maxcores=maxcores, maxgpus=maxgpus, target=None, dry_run=dry_run
        )
    )


@app.command()
def resume(
    config_path: ConfigArg,
    maxcores: MaxCoresOpt = None,
    maxgpus: MaxGpusOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Resume the pipeline, hitting the on-disk cache for unchanged steps."""
    raise typer.Exit(
        _dispatch(
            "resume", config_path, maxcores=maxcores, maxgpus=maxgpus, target=None, dry_run=dry_run
        )
    )


@app.command("rebuild-cache")
def rebuild_cache(
    config_path: ConfigArg,
    target: TargetArg = None,
    maxcores: MaxCoresOpt = None,
    maxgpus: MaxGpusOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Rebuild one step's cache from existing outputs (default: latest); no submission.

    The run ends at that step.
    """
    raise typer.Exit(
        _dispatch(
            "rebuild-cache",
            config_path,
            maxcores=maxcores,
            maxgpus=maxgpus,
            target=target,
            dry_run=dry_run,
        )
    )


@app.command("rebuild-nms")
def rebuild_nms(
    config_path: ConfigArg,
    target: TargetArg = None,
    maxcores: MaxCoresOpt = None,
    maxgpus: MaxGpusOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Re-resolve the NMS step from outputs already on disk; no submission.

    Defaults to the step setting `nms: true` rather than the last step. Round 1 is re-parsed
    and its displaced children re-read from the `attemptK/` they ran in, so re-resolving
    costs a read, not a re-run of the frequencies. The run ends at that step.
    """
    raise typer.Exit(
        _dispatch(
            "rebuild-nms",
            config_path,
            maxcores=maxcores,
            maxgpus=maxgpus,
            target=target,
            dry_run=dry_run,
        )
    )


@app.command()
def rerun(
    config_path: ConfigArg,
    target: TargetArg = None,
    maxcores: MaxCoresOpt = None,
    maxgpus: MaxGpusOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Redo one whole step from scratch (default: latest); the others resume."""
    raise typer.Exit(
        _dispatch(
            "rerun", config_path, maxcores=maxcores, maxgpus=maxgpus, target=target, dry_run=dry_run
        )
    )


@app.command("rerun-errors")
def rerun_errors(
    config_path: ConfigArg,
    target: TargetArg = None,
    maxcores: MaxCoresOpt = None,
    maxgpus: MaxGpusOpt = None,
    dry_run: DryRunOpt = False,
) -> None:
    """Re-attempt one step's pending failed jobs (default: latest), then continue the run."""
    raise typer.Exit(
        _dispatch(
            "rerun-errors",
            config_path,
            maxcores=maxcores,
            maxgpus=maxgpus,
            target=target,
            dry_run=dry_run,
        )
    )


# ---------------------------------------------------------------------------
# Introspection — the machine-readable schema and registry views
# ---------------------------------------------------------------------------


@app.command()
def agent(
    config_path: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Config file this session works on (optional).",
        ),
    ] = None,
    provider: Annotated[
        str,
        typer.Option("--provider", help="Endpoint preset: openai, ollama, vllm, or custom."),
    ] = "custom",
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model name, or a provider:model string."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="OpenAI-compatible endpoint URL override."),
    ] = None,
) -> None:
    """Chat with an embedded agent that authors and triages workflows (terminal REPL).

    The same tools as the MCP server, no client required: multi-provider via
    OpenAI-compatible endpoints (local Ollama/vLLM included) or PydanticAI's native
    provider strings. Mutating tools always ask for confirmation first. Needs the
    ``chemrefine[agent]`` extra; configuration also via CHEMREFINE_LLM_MODEL /
    _BASE_URL / _API_KEY.
    """
    try:
        from chemrefine.agent import chat
    except ImportError as e:
        logger.error("the embedded agent needs PydanticAI: pip install 'chemrefine[agent]' (%s)", e)
        raise typer.Exit(code=1) from e
    try:
        chat.main(
            provider=provider,
            model=model,
            base_url=base_url,
            config_path=str(config_path) if config_path is not None else None,
        )
    except ChemRefineError as e:
        logger.error("%s: %s", type(e).__name__, e)
        raise typer.Exit(code=e.exit_code) from e


@app.command()
def gui(
    config_path: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Existing config to load into the builder.",
        ),
    ] = None,
    port: Annotated[
        int, typer.Option("--port", min=0, help="Port to bind on 127.0.0.1 (0 = pick free).")
    ] = 0,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Print the URL instead of opening a browser.")
    ] = False,
) -> None:
    """Open the click-through YAML builder in a browser (local web app).

    Left pane: steps, engines and options as forms driven by the live schema; right
    pane: the YAML being built. Binds 127.0.0.1 behind a per-session token — reach a
    cluster with SSH port forwarding. Needs the ``chemrefine[gui]`` extra.
    """
    try:
        from chemrefine.gui.serve import launch
    except ImportError as e:
        logger.error("the GUI needs flask/waitress: pip install 'chemrefine[gui]' (%s)", e)
        raise typer.Exit(code=1) from e
    launch(config_path, port=port, open_browser=not no_browser)


@app.command()
def mcp() -> None:
    """Serve ChemRefine's agent tools over the Model Context Protocol (stdio).

    Register it with an MCP client, e.g. ``claude mcp add chemrefine -- chemrefine mcp``
    (or over SSH for a cluster: ``-- ssh login-node chemrefine mcp``). Needs the
    ``chemrefine[mcp]`` extra; see :mod:`chemrefine.mcp_server`.
    """
    try:
        from chemrefine import mcp_server
    except ImportError as e:
        logger.error(
            "the MCP server needs the Model Context Protocol SDK: "
            "pip install 'chemrefine[mcp]' (%s)",
            e,
        )
        raise typer.Exit(code=1) from e
    mcp_server.main()


@app.command()
def schema() -> None:
    """Print the machine-readable config schema document as JSON.

    The document assembled by :func:`chemrefine.introspect.schema_document`: the config
    schema, the NMS knob schema, and a descriptor per registered engine. Consumers that
    cannot import chemrefine (the GUI's forms, an agent writing a config) read this
    instead of the docs, so it is generated from the validating models and cannot drift.
    """
    import json

    from chemrefine.introspect import schema_document

    typer.echo(json.dumps(schema_document(), indent=2))


@app.command()
def validate(
    config_path: ConfigArg,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the structured validation report as JSON.")
    ] = False,
) -> None:
    """Validate a config and report every finding at once; exit 2 if it cannot run.

    The non-raising twin of loading: pydantic errors keep their field locations,
    registry-aware checks (unknown engine, bad option values, missing templates) are
    included, and warnings do not affect the exit code. See :mod:`chemrefine.validate`.
    """
    import json

    from chemrefine.validate import validate_config_file

    report = validate_config_file(config_path)
    if as_json:
        typer.echo(json.dumps(report.to_json(), indent=2))
    else:
        for issue in report.issues:
            where = ".".join(str(part) for part in issue.loc) or "config"
            typer.echo(f"error [{issue.kind}] at {where}: {issue.message}")
        for warning in report.warnings:
            where = ".".join(str(part) for part in warning.loc) or "config"
            typer.echo(f"warning [{warning.kind}] at {where}: {warning.message}")
        if report.ok:
            steps = report.config.steps if report.config is not None else ()
            typer.echo(f"OK: {len(steps)} step(s) validated")
    if not report.ok:
        raise typer.Exit(code=2)


@app.command()
def scaffold(
    config_path: ConfigArg,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Rewrite existing template files with starters."),
    ] = False,
) -> None:
    """Write starter templates for every file the config expects but lacks.

    Step templates for each template-driven step and the SLURM header(s) dispatch would
    pick. Existing files are kept unless --overwrite. See :mod:`chemrefine.scaffold`.
    """
    try:
        from chemrefine.config import load_config
        from chemrefine.scaffold import plan_templates, scaffold_templates

        config = load_config(config_path)
        written = set(scaffold_templates(config, overwrite=overwrite))
        for plan in plan_templates(config):
            state = "wrote" if plan.path in written else "kept"
            typer.echo(f"{state} {plan.path}")
    except ChemRefineError as e:
        logger.error("%s: %s", type(e).__name__, e)
        raise typer.Exit(code=e.exit_code) from e


@app.command()
def engines(
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the full engine descriptors as JSON.")
    ] = False,
) -> None:
    """List registered engines: template kind, options model, capabilities."""
    import dataclasses
    import json

    from chemrefine.introspect import describe_engines

    descriptors = describe_engines()
    if as_json:
        typer.echo(json.dumps([dataclasses.asdict(d) for d in descriptors], indent=2))
        return
    for d in descriptors:
        template = f"template .{d.template_suffix}" if d.template_driven else "no template"
        options = "options model" if d.options_schema is not None else "template-configured"
        caps = ", ".join(d.capabilities) if d.capabilities else "-"
        typer.echo(f"{d.name:14} {template:14} {options:20} {caps}")


# ---------------------------------------------------------------------------
# backends — manage per-backend environments (provision once, reuse every run)
# ---------------------------------------------------------------------------

backends_app = typer.Typer(
    no_args_is_help=True,
    help="Manage per-backend environments (conflicting MLIP stacks, one env each).",
)
app.add_typer(backends_app, name="backends")


@backends_app.command("install")
def backends_install(
    extras: Annotated[
        list[str], typer.Argument(help="Backend extra(s), e.g. mlip-mace mlip-fairchem pyscf.")
    ],
) -> None:
    """Provision managed env(s) so steps can run these backends side by side.

    Each env is built with the same tool that created the current environment
    (conda / uv / venv) under ``$CHEMREFINE_HOME`` and reused by every later run;
    run this once (on HPC: on a login node with internet) per backend you use.
    """
    from chemrefine.engines import build_backend_env, known_backend_extras

    known = known_backend_extras()
    unknown = [e for e in extras if e not in known]
    if unknown:
        raise typer.BadParameter(f"unknown backend(s) {unknown}; known: {sorted(known)}")
    for extra in extras:
        typer.echo(f"provisioning {extra} …")
        try:
            python = build_backend_env(extra)
        except ChemRefineError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(code=e.exit_code) from e
        typer.echo(f"{extra}: {python}")


@backends_app.command("list")
def backends_list() -> None:
    """List every known backend extra and whether its managed env is provisioned."""
    from chemrefine.engines import backend_env_python, known_backend_extras

    for extra in sorted(known_backend_extras()):
        python = backend_env_python(extra)
        status = str(python) if python.is_file() else "not provisioned"
        typer.echo(f"{extra:16} {status}")


@backends_app.command("path")
def backends_path(
    extra: Annotated[str, typer.Argument(help="Backend extra, e.g. mlip-fairchem.")],
) -> None:
    """Print the managed env's python for one backend (exit 1 if not provisioned)."""
    from chemrefine.engines import backend_env_python

    python = backend_env_python(extra)
    typer.echo(str(python))
    if not python.is_file():
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Legacy (v1.3.1) flag-style CLI → subcommand translation
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: translate any legacy flag-style argv, then run the Typer app."""
    import sys

    sys.argv[1:] = cli_legacy.translate_argv(sys.argv[1:])
    app()
