"""Starter templates for the files a config names but the filesystem lacks.

A validated config is not yet a runnable one: every template-driven step needs its
``templates/stepN.<suffix>`` and every scheduler-run step a SLURM header, and a first
run's likeliest failure is exactly one of those files missing
(:func:`chemrefine.ids.require_template`'s reason to exist). :func:`plan_templates`
answers *which files this config expects and whether they exist* — the view the GUI's
template chips and an agent's next-action decision render from — and
:func:`scaffold_templates` writes a commented starter into each gap so the user edits a
working file instead of authoring one from a blank page.

Starters are deliberately minimal echoes of the shipped examples (``examples/quickstart``
et al.). Each engine declares its own (:class:`~chemrefine.engines.api.StarterProviding`),
so the text travels with the engine and adding one edits nothing here; an engine that
declares none gets a suffix-shaped fallback, so a drop-in still scaffolds something
useful. The trainer's starter is honest about its limits: a real ``mlip-train`` template is
load-bearing in ways a stub cannot be, so it points at the worked example rather than
pretending.

Which header a step needs is :func:`chemrefine.validate.effective_header`'s answer —
the same resolution dispatch uses — so the plan and the run cannot disagree about the
file's name.
"""

from __future__ import annotations

import dataclasses
import textwrap
from contextlib import suppress
from pathlib import Path
from typing import Literal

from chemrefine.config import Config
from chemrefine.engines.api import JobExecutable, StarterProviding, TemplateDriven, get_engine
from chemrefine.errors import ConfigError, EngineNotFoundError
from chemrefine.ids import step_template_path
from chemrefine.validate import effective_header

_SCRIPT_FALLBACK = (
    "# Script starter. Rendered per structure: $XYZ_PATH / $CHARGE / $MULTIPLICITY\n"
    "# come from the pipeline; step options render as $UPPERCASE placeholders.\n"
    "$OUTPUT_CONTRACT"
)
"""The starter for a ``.py`` template of an engine that declares none of its own."""
_GENERIC_STARTER = "# ChemRefine step template — this engine documents its own format.\n"

_HEADER_STARTERS: dict[str, str] = {
    "cuda.slurm.header": (
        "#!/bin/bash\n#SBATCH --partition=EDIT_ME\n#SBATCH --time=24:00:00\n#SBATCH --gres=gpu:1\n"
    ),
}
_HEADER_DEFAULT = "#!/bin/bash\n#SBATCH --partition=EDIT_ME\n#SBATCH --time=24:00:00\n"


@dataclasses.dataclass(frozen=True)
class TemplatePlan:
    """One file the config expects: where it belongs and whether it is there.

    ``kind`` separates a step's input template from a SLURM header; ``step`` / ``engine``
    anchor a step template to the step that needs it (both ``None`` for a header, which
    any number of steps may share).
    """

    path: Path
    exists: bool
    kind: Literal["step", "slurm-header"]
    step: int | None
    engine: str | None


def plan_templates(config: Config) -> tuple[TemplatePlan, ...]:
    """Every template file this config expects, in step order, headers last.

    Reads the engines exactly as the run does (registry + capability checks), so an
    unknown engine raises :class:`~chemrefine.errors.EngineNotFoundError` here rather
    than producing a plan for files no engine would read — run ``chemrefine validate``
    first for the report shape.

    Steps may share a ``template:`` name — reusing one input across steps is ordinary —
    and each sharing step keeps a plan of its own, so per-step lookups
    (:func:`chemrefine.agent_tools.read_template`) keep answering for every step. What a
    share must *not* cross is an engine boundary: the sharers' starters then differ, and
    ``scaffold_templates`` snapshots ``exists`` before writing anything, so the second
    starter silently replaced the first with ``overwrite`` still False. Refused here, at
    planning time, so every consumer of this seam — the CLI, the GUI's chips, the agent
    tools — inherits the refusal before a byte is written.
    """
    plans: list[TemplatePlan] = []
    step_plans: dict[Path, TemplatePlan] = {}
    headers: dict[str, None] = {}
    for step in config.steps:
        engine = get_engine(step.engine)
        if isinstance(engine, TemplateDriven):
            path = step_template_path(
                config.template_dir,
                step.step,
                template=step.template,
                suffix=engine.template_suffix,
            )
            earlier = step_plans.get(path)
            if earlier is not None and earlier.engine != step.engine:
                raise ConfigError(
                    f"steps {earlier.step} ({earlier.engine}) and {step.step} "
                    f"({step.engine}) both name {path.name} as their template, and the "
                    f"two engines read different formats — one starter would silently "
                    f"overwrite the other. Give each engine's steps a template of its own."
                )
            plan = TemplatePlan(
                path=path,
                exists=path.is_file(),
                kind="step",
                step=step.step,
                engine=step.engine,
            )
            step_plans.setdefault(path, plan)
            plans.append(plan)
        if isinstance(engine, JobExecutable):
            headers.setdefault(effective_header(config, step, engine), None)
    for name in headers:
        path = config.template_dir / name
        plans.append(
            TemplatePlan(
                path=path, exists=path.is_file(), kind="slurm-header", step=None, engine=None
            )
        )
    return tuple(plans)


def _output_contract_comment(engine: object) -> str:
    """The "assign these names" comment, written from the engine's own output contract.

    Generated rather than typed into each starter, for the reason the contract is declared at
    all: a roster restated here as prose can tell a user to assign a name the footer does not
    harvest, or fail to mention one it does. An engine that extends
    :attr:`~chemrefine.engines._script.engine.ScriptEngine.output_fields` gets its extra names
    into its own starter with no edit to this module.

    Takes the engine rather than its name because describing a contract and finding the
    engine that declares one are two jobs; the caller already has the object.
    """
    fields = getattr(engine, "output_fields", None)
    if not fields:
        return ""
    required = [f.name for f in fields if f.required]
    optional = [f.name for f in fields if not f.required]
    line = f"# Assign {', '.join(f'`{n}`' for n in required)}"
    if optional:
        line += f" (optionally {', '.join(f'`{n}`' for n in optional)})"
    line += " — the appended output footer harvests them."
    # Wrapped, because this lands in a file a person opens: the roster grows with the
    # engine's contract, and one starter comment should not run off the side of an editor.
    return "".join(f"{chunk}\n" for chunk in textwrap.wrap(line, width=88, subsequent_indent="# "))


def _starter_for(plan: TemplatePlan) -> str:
    """The starter body for one planned file — the engine's own, else the suffix fallback."""
    if plan.kind == "slurm-header":
        return _HEADER_STARTERS.get(plan.path.name, _HEADER_DEFAULT)
    # `plan_templates` resolves every engine before any starter is chosen, so a name that
    # does not resolve reaches here only from a direct caller — and an engine nobody can look
    # up has no starter and no contract to describe, which is what a non-script engine
    # without a starter answers too.
    engine: object = None
    if plan.engine is not None:
        with suppress(EngineNotFoundError):
            engine = get_engine(plan.engine)
    starter = (
        engine.template_starter
        if isinstance(engine, StarterProviding)
        else _fallback_starter(plan.path.suffix.lstrip("."))
    )
    # Only this one placeholder is filled: a starter is a *template*, and its `$XYZ_PATH`,
    # `$CHARGE` and option placeholders belong to the renderer that runs per structure.
    return starter.replace("$OUTPUT_CONTRACT", _output_contract_comment(engine))


def _fallback_starter(suffix: str) -> str:
    """A starter for an engine that declares none, from the suffix alone.

    ``.py`` gets the neutral script starter — MLIP's and PySCF's bodies are their own, so
    neither would serve a stranger. ``.inp`` is ORCA-shaped, the one input format other
    programs mimic, so ORCA's own starter serves a third-party ``.inp`` engine. Anything else
    gets a comment saying the engine documents its own format.
    """
    if suffix == "py":
        return _SCRIPT_FALLBACK
    if suffix == "inp":
        orca = get_engine("orca")
        if isinstance(orca, StarterProviding):  # it is; the check is for the type checker
            return orca.template_starter
    return _GENERIC_STARTER


def scaffold_templates(config: Config, *, overwrite: bool = False) -> tuple[Path, ...]:
    """Write a starter into every planned gap; return the paths written.

    Existing files are left alone unless ``overwrite`` — scaffolding is for the blank
    page, and a template the user has edited is exactly the file this must never touch
    by default. ``template_dir`` is created if missing.

    An unwritable destination — a read-only tree, an exhausted quota (ENOSPC on HPC
    scratch is the everyday case) — is a :class:`~chemrefine.errors.ConfigError` naming
    the path, not a raw :class:`OSError`: this seam serves the CLI, the GUI and the MCP
    tools, and all three promise failures that carry the documented exit code — the GUI's
    error handler re-raises anything else as a 500 with a logged traceback.
    """
    written: list[Path] = []
    try:
        config.template_dir.mkdir(parents=True, exist_ok=True)
        for plan in plan_templates(config):
            # Steps sharing a template each carry a plan (per-step lookups need one), so
            # the file itself is written once: `exists` was snapshotted before any write,
            # and re-writing per sharing step re-did identical work at best.
            if plan.path in written or (plan.exists and not overwrite):
                continue
            # A `template:` override may name a subdirectory (or an absolute path elsewhere)
            # — the same shape `agent_tools.write_template` already creates parents for.
            plan.path.parent.mkdir(parents=True, exist_ok=True)
            plan.path.write_text(_starter_for(plan), encoding="utf-8")
            written.append(plan.path)
    except OSError as e:
        # `written` is in the message because the failure is mid-loop: the starters
        # already on disk stay there, and a retry would read a half-written last file as
        # "exists — kept". Naming what landed makes the partial state inspectable.
        landed = ", ".join(str(p) for p in written) or "nothing"
        raise ConfigError(
            f"cannot scaffold templates under {config.template_dir}: {e} "
            f"(already written before the failure: {landed})"
        ) from e
    return tuple(written)
