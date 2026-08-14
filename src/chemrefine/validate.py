"""Config validation as a structured report — the non-raising twin of ``load_config``.

:func:`chemrefine.config.load_config` answers "give me a runnable config or raise"; a
GUI highlighting the offending field and an agent deciding what to fix next both need
the *other* shape: every finding at once, each anchored to the location that caused it.
:func:`validate_config_text` / :func:`validate_config_file` produce that shape without
ever raising — parse and model failures become :class:`ValidationIssue` rows (pydantic's
``loc`` tuples survive structurally), and the checks a run would only surface later
become issues or warnings up front:

* an ``engine:`` name not in this environment's registry (the run would exit with
  :class:`~chemrefine.errors.EngineNotFoundError`),
* a bad value for a knob the engine's declared options model reads — through
  ``from_raw_lenient``, the engine's own read, so validation and the run cannot
  disagree,
* invalid NMS knobs on an ``nms: true`` step — and ``nms: true`` on an engine that
  cannot NMS, which the run would silently skip (warning),
* option keys no reader of this step declares (warning — a script-engine template may
  read them as placeholders, but a typo looks exactly the same),
* step templates and SLURM headers that do not exist yet (warnings —
  ``chemrefine scaffold`` creates starters; the header is resolved the way dispatch
  resolves it, per-step override first, cuda header when the step's validated options
  request a GPU).

Registry-aware checks live here rather than in :mod:`chemrefine.config` because config
must stay importable by the engine subsystem — importing the registry from there would
be a cycle.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from chemrefine import slurm
from chemrefine.config import Config, StepConfig, _resolve_relative_paths
from chemrefine.engines._job import gpus_from_options
from chemrefine.engines.api import (
    ENGINES,
    CalculationEngine,
    JobExecutable,
    NmsCapableEngine,
    OptionsDeclaring,
    TemplateDriven,
    get_engine,
)
from chemrefine.errors import ConfigError
from chemrefine.ids import step_template_path
from chemrefine.nms import NmsOptions


@dataclasses.dataclass(frozen=True)
class ValidationIssue:
    """One finding, anchored to the config location that caused it.

    ``loc`` follows pydantic's convention — a path of keys/indices into the YAML
    (``("steps", 1, "options")``) — so a GUI walks it to the field to highlight and an
    agent quotes it verbatim. ``kind`` is a short machine-checkable class of finding
    (``yaml`` / ``legacy`` / a pydantic error type / ``engine`` / ``options`` / ``nms``
    / ``template`` / ``slurm-header``); the message alone is for humans.
    """

    loc: tuple[str | int, ...]
    kind: str
    message: str


@dataclasses.dataclass(frozen=True)
class ValidationReport:
    """Everything one validation pass found, split by severity.

    ``issues`` mean the config cannot run (``ok`` is their absence); ``warnings`` mean
    it can, but something is worth a look before it does — a template that does not
    exist yet, an option key nothing declares. ``config`` is the validated model when
    the issues list is empty, so a caller that validated text does not parse it twice.
    """

    issues: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]
    config: Config | None

    @property
    def ok(self) -> bool:
        """Whether the config is runnable — no issues (warnings do not block)."""
        return not self.issues

    def to_json(self) -> dict[str, Any]:
        """The report as a JSON-serializable dict (the MCP/GUI wire shape)."""
        return {
            "ok": self.ok,
            "issues": [dataclasses.asdict(issue) for issue in self.issues],
            "warnings": [dataclasses.asdict(warning) for warning in self.warnings],
        }


def _failed(kind: str, message: str) -> ValidationReport:
    """A report that died before a model existed — one issue, no config."""
    issue = ValidationIssue(loc=(), kind=kind, message=message)
    return ValidationReport(issues=(issue,), warnings=(), config=None)


def validate_config_text(text: str, *, base_dir: Path | None = None) -> ValidationReport:
    """Validate YAML config text; never raises.

    The same pipeline as :func:`~chemrefine.config.load_config` — ``yaml.safe_load``,
    mapping check, ``Config(**raw)`` (legacy normalization included), path resolution
    against ``base_dir`` when given — but every failure becomes a report row. Model
    validation errors keep pydantic's per-error ``loc``; the step-level checks described
    in the module docstring run only once a model exists.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return _failed("yaml", f"malformed YAML: {e}")
    if not isinstance(raw, dict):
        return _failed("yaml", "config is not a YAML mapping")
    try:
        config = Config(**raw)
    except ValidationError as e:
        issues = tuple(
            ValidationIssue(loc=tuple(err["loc"]), kind=str(err["type"]), message=str(err["msg"]))
            for err in e.errors()
        )
        return ValidationReport(issues=issues, warnings=(), config=None)
    except ConfigError as e:
        # The legacy normalizer refuses some v1 spellings before pydantic runs.
        return _failed("legacy", str(e))
    if base_dir is not None:
        config = _resolve_relative_paths(config, base=base_dir.resolve())
    issues_list, warnings_list = _inspect_steps(config)
    report_config = config if not issues_list else None
    return ValidationReport(
        issues=tuple(issues_list), warnings=tuple(warnings_list), config=report_config
    )


def validate_config_file(path: Path) -> ValidationReport:
    """Validate a config file on disk — :func:`validate_config_text` plus the read.

    Relative paths resolve against the file's own directory, exactly as
    :func:`~chemrefine.config.load_config` resolves them.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return _failed("io", f"could not read {path}: {e}")
    return validate_config_text(text, base_dir=path.parent)


def _inspect_steps(config: Config) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    """The registry-aware per-step checks; returns ``(issues, warnings)``."""
    issues: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    headers: dict[str, list[int]] = {}
    for index, step in enumerate(config.steps):
        engine_cls = ENGINES.get(step.engine)
        if engine_cls is None:
            issues.append(
                ValidationIssue(
                    loc=("steps", index, "engine"),
                    kind="engine",
                    message=(
                        f"unknown engine {step.engine!r}; registered in this environment: "
                        f"{sorted(ENGINES)}"
                    ),
                )
            )
            continue
        engine = get_engine(step.engine)
        declared: set[str] = set()
        options_ok = True
        if isinstance(engine, OptionsDeclaring):
            declared |= engine.options_cls._accepted_names()
            try:
                engine.options_cls.from_raw_lenient(step.options)
            except ConfigError as e:
                options_ok = False
                issues.append(
                    ValidationIssue(loc=("steps", index, "options"), kind="options", message=str(e))
                )
        if step.nms:
            declared |= set(NmsOptions.model_fields)
            try:
                NmsOptions.from_raw(step.options)
            except ValidationError as e:
                issues.append(
                    ValidationIssue(loc=("steps", index, "options"), kind="nms", message=str(e))
                )
            if not isinstance(engine, NmsCapableEngine):
                # The run silently skips NMS for an incapable engine (step.py narrows and
                # moves on) — the one silent no-op a pre-run check can catch.
                warnings.append(
                    ValidationIssue(
                        loc=("steps", index, "nms"),
                        kind="nms",
                        message=(
                            f"engine {step.engine!r} cannot normal-mode sample; "
                            "nms: true will be ignored"
                        ),
                    )
                )
        undeclared = sorted(set(step.options or {}) - declared)
        if undeclared:
            warnings.append(
                ValidationIssue(
                    loc=("steps", index, "options"),
                    kind="options",
                    message=(
                        f"keys {undeclared} are not declared by any reader of this step's "
                        "options; a script template may read them as placeholders, but a "
                        "typo looks exactly the same"
                    ),
                )
            )
        if isinstance(engine, TemplateDriven):
            template = step_template_path(
                config.template_dir,
                step.step,
                template=step.template,
                suffix=engine.template_suffix,
            )
            if not template.is_file():
                warnings.append(
                    ValidationIssue(
                        loc=("steps", index, "template"),
                        kind="template",
                        message=(
                            f"{engine.label} template {template} does not exist yet "
                            "(chemrefine scaffold creates starters)"
                        ),
                    )
                )
        if isinstance(engine, JobExecutable):
            # Only scheduler-run engines resolve a header (locally too); an inline engine
            # like the test fake never reads one, and warning about it would be noise.
            headers.setdefault(_effective_header(config, step, engine, options_ok), []).append(
                step.step
            )
    for name, steps in sorted(headers.items()):
        path = config.template_dir / name
        if not path.is_file():
            warnings.append(
                ValidationIssue(
                    loc=("template_dir",),
                    kind="slurm-header",
                    message=(
                        f"SLURM header {path} (used by step(s) {steps}) does not exist yet "
                        "(chemrefine scaffold creates a starter)"
                    ),
                )
            )
    return issues, warnings


def _effective_header(
    config: Config, step: StepConfig, engine: CalculationEngine, options_ok: bool
) -> str:
    """The header dispatch would pick — per-step override, cuda on GPU demand, else global.

    Mirrors ``_execution._header_name`` without needing a :class:`StepContext`: the GPU
    demand is read through the engine's declared options model via
    :func:`~chemrefine.engines._job.gpus_from_options`, the same read the scheduler
    performs. Skipped (global header assumed) when the step's options already failed
    validation — the failure is reported on its own line, and a second exception here
    would bury it.
    """
    if step.slurm_template:
        return step.slurm_template
    if (
        options_ok
        and isinstance(engine, OptionsDeclaring)
        and gpus_from_options(step.options, engine.options_cls) > 0
    ):
        return slurm.header_name_for_device("cuda")
    return config.slurm_template
