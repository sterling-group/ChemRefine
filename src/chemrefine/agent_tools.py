"""The tool layer agents drive ChemRefine through — framework-neutral, JSON-shaped.

One module holds every operation an AI agent (or any remote caller) performs against a
ChemRefine tree, so the MCP server and the embedded chat agent register *the same
functions* and cannot drift apart. Nothing here imports an
agent framework; every function takes JSON-compatible arguments, returns a
JSON-serializable dict, and raises :class:`~chemrefine.errors.ChemRefineError`
subclasses — whose documented ``exit_code`` taxonomy doubles as the structured failure
signal a caller branches on.

Two design rules, both taken from what breaks agent/HPC integrations in practice:

* **Submitting never blocks.** :func:`start_run` launches a *detached* ``python -m
  chemrefine`` child that owns the run lock and outlives the caller, and returns
  immediately; progress is read back by :func:`run_status` / :func:`get_results` /
  :func:`get_failures`, which are pure filesystem reads of what the pipeline already
  persists (``steps.csv``, ``failed_jobs.json``, the run lock, the child's log).
* **Results are paginated.** A refinement tree can hold thousands of structures;
  :func:`get_results` returns a slice with a total, never the whole ensemble, so a tool
  result cannot flood a model's context window.

Everything else delegates to the library seams built for exactly this:
:mod:`chemrefine.introspect` (schema), :mod:`chemrefine.validate` (structured report),
:mod:`chemrefine.scaffold` (starter templates), :func:`chemrefine.pipeline.lock_status`.
"""

from __future__ import annotations

import csv
import dataclasses
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chemrefine import introspect, pipeline, scaffold
from chemrefine.cache import load_failure_records
from chemrefine.config import Config, StepConfig, load_config
from chemrefine.errors import ChemRefineError, ConfigError, RunLockError
from chemrefine.validate import validate_config_file, validate_config_text

_ACTIONS = ("run", "resume", "rerun", "rerun-errors", "rebuild-cache", "rebuild-nms")
"""CLI actions :func:`start_run` may launch — the recovery vocabulary, nothing else."""

_EXIT_CODES: dict[str, int] = {
    cls.__name__: cls.exit_code
    for cls in (ChemRefineError, *ChemRefineError.__subclasses__())
}
"""The documented failure taxonomy, shipped with every failure payload."""


# ---------------------------------------------------------------------------
# Introspection + validation (thin re-exposures of the library seams)
# ---------------------------------------------------------------------------


def get_schema() -> dict[str, Any]:
    """The machine-readable schema document — read this before writing a config."""
    return introspect.schema_document()


def list_engines() -> list[dict[str, Any]]:
    """One descriptor per registered engine (capabilities, declared options schema)."""
    return [dataclasses.asdict(d) for d in introspect.describe_engines()]


def validate_config(yaml_text: str, base_dir: str | None = None) -> dict[str, Any]:
    """Validate config YAML text; every finding at once, never raises.

    ``base_dir`` resolves relative paths (and locates templates) as if the text lived in
    that directory — pass the directory the config will be saved to.
    """
    base = Path(base_dir) if base_dir is not None else None
    return validate_config_text(yaml_text, base_dir=base).to_json()


def validate_config_path(config_path: str) -> dict[str, Any]:
    """Validate a config file on disk — the report shape, not an exception."""
    return validate_config_file(Path(config_path)).to_json()


def summarize_config(config_path: str) -> dict[str, Any]:
    """A loaded config's execution summary: settings plus one row per step."""
    config = load_config(Path(config_path))
    return {
        "output_dir": str(config.output_dir),
        "template_dir": str(config.template_dir),
        "input": str(config.input) if config.input is not None else None,
        "charge": config.charge,
        "multiplicity": config.multiplicity,
        "max_cores": config.max_cores,
        "max_gpus": config.max_gpus,
        "dispatch": config.dispatch,
        "steps": [
            {
                "step": s.step,
                "name": s.name,
                "dir": s.dir_name(),
                "engine": s.engine,
                "operation": s.operation,
                "template": s.template,
                "options": s.options,
                "sample": s.sample.model_dump() if s.sample is not None else None,
                "nms": s.nms,
                "on_failure": s.on_failure,
            }
            for s in config.steps
        ],
    }


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def _step_template_plan(config: Config, step: int | str) -> scaffold.TemplatePlan:
    """The template plan row for one step, or the :class:`ConfigError` saying why not."""
    step_cfg = config.find_step(step)
    if step_cfg is None:
        raise ConfigError(f"no step matches {step!r}")
    for plan in scaffold.plan_templates(config):
        if plan.kind == "step" and plan.step == step_cfg.step:
            return plan
    raise ConfigError(
        f"step {step_cfg.step} (engine {step_cfg.engine!r}) does not read a template"
    )


def read_template(config_path: str, step: int | str) -> dict[str, Any]:
    """One step's template text — the file the engine will actually render."""
    plan = _step_template_plan(load_config(Path(config_path)), step)
    if not plan.exists:
        raise ConfigError(
            f"template {plan.path} does not exist yet (scaffold_templates writes a starter)"
        )
    return {"path": str(plan.path), "text": plan.path.read_text(encoding="utf-8")}


def write_template(config_path: str, step: int | str, text: str) -> dict[str, Any]:
    """Replace one step's template with ``text`` (creating template_dir if needed)."""
    plan = _step_template_plan(load_config(Path(config_path)), step)
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    plan.path.write_text(text, encoding="utf-8")
    return {"path": str(plan.path), "bytes": len(text.encode("utf-8"))}


def scaffold_templates(config_path: str, overwrite: bool = False) -> dict[str, Any]:
    """Write starter templates into every gap the config expects; report both sides."""
    config = load_config(Path(config_path))
    written = set(scaffold.scaffold_templates(config, overwrite=overwrite))
    plans = scaffold.plan_templates(config)
    return {
        "written": sorted(str(p) for p in written),
        "kept": sorted(str(p.path) for p in plans if p.path not in written),
    }


# ---------------------------------------------------------------------------
# Runs — detached submit, filesystem-read status
# ---------------------------------------------------------------------------


def start_run(
    config_path: str,
    action: str = "run",
    target: str | None = None,
    max_cores: int | None = None,
    max_gpus: int | None = None,
) -> dict[str, Any]:
    """Launch a detached ``python -m chemrefine <action>`` and return immediately.

    The child owns the run lock, logs to ``output_dir/agent_runs/``, and survives this
    process exiting — an agent session ending must not kill a three-day refinement.
    Refuses while a live driver holds the lock (:func:`~chemrefine.pipeline.lock_status`),
    and validates ``action`` / ``target`` before anything launches, so the likeliest
    mistakes fail here with a message rather than in a log nobody is watching yet.
    """
    if action not in _ACTIONS:
        raise ConfigError(f"unknown action {action!r}; one of {list(_ACTIONS)}")
    path = Path(config_path).resolve()
    config = load_config(path)
    if target is not None and config.find_step(target) is None:
        raise ConfigError(f"no step matches target {target!r}")
    status = pipeline.lock_status(config.output_dir)
    if status.held:
        raise RunLockError(
            f"a driver already holds {config.output_dir} "
            f"(pid {status.pid} on {status.host}, started {status.started})"
        )
    log_dir = config.output_dir / "agent_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{action}.log"
    argv = [sys.executable, "-m", "chemrefine", action, str(path)]
    if target is not None:
        argv.append(target)
    if max_cores is not None:
        argv += ["--maxcores", str(max_cores)]
    if max_gpus is not None:
        argv += ["--maxgpus", str(max_gpus)]
    with log_path.open("wb") as log:
        proc = subprocess.Popen(  # noqa: S603
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    return {"pid": proc.pid, "log": str(log_path), "output_dir": str(config.output_dir)}


def _steps_csv_rows(output_dir: Path) -> list[dict[str, str]]:
    """Every row of the cumulative ``steps.csv`` (``[]`` before the first step reports)."""
    path = output_dir / "steps.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _latest_log(output_dir: Path) -> Path | None:
    """The newest ``agent_runs`` log, or ``None`` when no agent ever started a run."""
    log_dir = output_dir / "agent_runs"
    logs = sorted(log_dir.glob("*.log")) if log_dir.is_dir() else []
    return logs[-1] if logs else None


def run_status(config_path: str, log_tail_lines: int = 40) -> dict[str, Any]:
    """Where the tree stands: lock holder, per-step progress, the latest log's tail.

    Everything is read from what the pipeline persists — nothing here talks to the
    driver, so the answer is the same whether the run is live, finished, or died.
    """
    config = load_config(Path(config_path))
    status = pipeline.lock_status(config.output_dir)
    reported: dict[int, int] = {}
    for row in _steps_csv_rows(config.output_dir):
        step = int(row["Step"])
        reported[step] = reported.get(step, 0) + 1
    steps = []
    for s in config.steps:
        step_dir = config.step_dir(s)
        steps.append(
            {
                "step": s.step,
                "dir": s.dir_name(),
                "engine": s.engine,
                "reported_survivors": reported.get(s.step, 0),
                "failures": len(load_failure_records(step_dir)),
                "cached": (step_dir / "_cache").is_dir(),
            }
        )
    log_path = _latest_log(config.output_dir)
    tail: list[str] | None = None
    if log_path is not None:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
            -log_tail_lines:
        ]
    return {
        "running": status.held,
        "holder": (
            {"host": status.host, "pid": status.pid, "started": status.started}
            if status.host is not None
            else None
        ),
        "steps": steps,
        "log": str(log_path) if log_path is not None else None,
        "log_tail": tail,
    }


def _steps_for(config: Config, step: int | str | None) -> tuple[StepConfig, ...]:
    """The step(s) a query names — all of them, or exactly the one that matches."""
    if step is None:
        return tuple(config.steps)
    step_cfg = config.find_step(step)
    if step_cfg is None:
        raise ConfigError(f"no step matches {step!r}")
    return (step_cfg,)


def get_results(
    config_path: str,
    step: int | str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """A paginated slice of ``steps.csv`` — survivors with energies and weights.

    Rows are exactly what the pipeline reported (already sorted by energy within each
    step, with the ``Energy type`` column naming which energy that step filtered on).
    ``total`` counts the filtered rows so a caller pages without fetching everything.
    """
    config = load_config(Path(config_path))
    rows = _steps_csv_rows(config.output_dir)
    if step is not None:
        wanted = {s.step for s in _steps_for(config, step)}
        rows = [row for row in rows if int(row["Step"]) in wanted]
    return {"total": len(rows), "offset": offset, "rows": rows[offset : offset + limit]}


def get_failures(config_path: str, step: int | str | None = None) -> dict[str, Any]:
    """Every ledgered failure, with the exit-code taxonomy and the suggested recovery.

    The ledger records *all* failures for visibility; only an ``on_failure: stop``
    step's failures are pending for re-attempt, which is why the suggestion is
    ``rerun-errors`` (re-attempt just those, then continue) rather than a full rerun.
    """
    config = load_config(Path(config_path))
    failures = [
        {
            "step": s.step,
            "structure_id": record.structure_id,
            "kind": record.kind.value,
            "reason": record.reason,
        }
        for s in _steps_for(config, step)
        for record in load_failure_records(config.step_dir(s))
    ]
    return {
        "failures": failures,
        "exit_codes": _EXIT_CODES,
        "suggested_action": "rerun-errors" if failures else None,
    }
