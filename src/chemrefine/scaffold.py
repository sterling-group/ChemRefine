"""Starter templates for the files a config names but the filesystem lacks.

A validated config is not yet a runnable one: every template-driven step needs its
``templates/stepN.<suffix>`` and every scheduler-run step a SLURM header, and a first
run's likeliest failure is exactly one of those files missing
(:func:`chemrefine.ids.require_template`'s reason to exist). :func:`plan_templates`
answers *which files this config expects and whether they exist* — the view the GUI's
template chips and an agent's next-action decision render from — and
:func:`scaffold_templates` writes a commented starter into each gap so the user edits a
working file instead of authoring one from a blank page.

Starters are deliberately minimal echoes of the shipped examples (``examples/first_run``
et al.). Each template-driven engine declares its own
(:attr:`~chemrefine.engines.api.TemplateDriven.template_starter`, which registration
requires), so the text travels with the engine and adding one edits nothing here. A
starter is honest about its limits: where a real template is load-bearing in ways a stub
cannot be, it points at a worked example rather than pretending.

Which header a step needs is :func:`chemrefine.validate.effective_header`'s answer —
the same resolution dispatch uses — so the plan and the run cannot disagree about the
file's name.
"""

from __future__ import annotations

import dataclasses
import textwrap
from pathlib import Path
from typing import Literal, cast

from chemrefine.config import Config
from chemrefine.engines.api import JobExecutable, TemplateDriven, get_engine
from chemrefine.errors import ConfigError
from chemrefine.ids import step_template_path
from chemrefine.io import user_write, write_user_file
from chemrefine.slurm import CUDA_HEADER
from chemrefine.validate import effective_header, shared_template_clash

_HEADER_STARTERS: dict[str, str] = {
    CUDA_HEADER: (
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
    starter would silently replace the first with ``overwrite`` still False. Refused here,
    at planning time, on :func:`chemrefine.validate.shared_template_clash`'s answer — the
    validator reports the same answer, so a config it calls OK is one this planner
    accepts — and every consumer of this seam (the CLI, the GUI's chips, the agent tools)
    inherits the refusal before a byte is written.
    """
    if (clash := shared_template_clash(config)) is not None:
        raise ConfigError(clash)
    plans: list[TemplatePlan] = []
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
            plan = TemplatePlan(
                path=path,
                exists=path.is_file(),
                kind="step",
                step=step.step,
                engine=step.engine,
            )
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
    """The starter body for one planned file: the engine's own, or the header's by name."""
    if plan.engine is None:  # a SLURM header, which any number of steps may share
        return _HEADER_STARTERS.get(plan.path.name, _HEADER_DEFAULT)
    # `plan_templates` plans a step only for a TemplateDriven engine, and registration holds
    # that capability whole, so the narrowing is for the type checker alone.
    engine = cast(TemplateDriven, get_engine(plan.engine))
    # Only this one placeholder is filled: a starter is a *template*, and its `$XYZ_PATH`,
    # `$CHARGE` and option placeholders belong to the renderer that runs per structure.
    return engine.template_starter.replace("$OUTPUT_CONTRACT", _output_contract_comment(engine))


def scaffold_templates(config: Config, *, overwrite: bool = False) -> tuple[Path, ...]:
    """Write a starter into every planned gap; return the paths written.

    Existing files are left alone unless ``overwrite`` — scaffolding is for the blank
    page, and a template the user has edited is exactly the file this must never touch
    by default. ``template_dir`` is created if missing.

    A destination the filesystem refuses — a read-only tree, an exhausted quota (ENOSPC on
    HPC scratch is the everyday case) — is :class:`~chemrefine.errors.WriteRefusedError`
    naming the starter it could not write, or ``template_dir`` when that could not be
    created. Starters are written in plan order, each whole or not at all, so the ones
    planned before the named file are on disk and the named one is not: scaffolding again
    keeps those and writes the rest.
    """
    written: list[Path] = []
    with user_write(config.template_dir):
        config.template_dir.mkdir(parents=True, exist_ok=True)
    for plan in plan_templates(config):
        # Steps sharing a template each carry a plan (per-step lookups need one), so
        # the file itself is written once: `exists` is snapshotted before any write, so
        # re-writing per sharing step would redo identical work at best.
        if plan.path in written or (plan.exists and not overwrite):
            continue
        # A `template:` override may name a subdirectory (or an absolute path elsewhere)
        # — the same shape `agent_tools.write_template` creates parents for. Whole or not
        # at all, like every other writer of a file a user edits: with `overwrite`, this
        # replaces a template the user may have worked on, and a write torn by a kill or a
        # full disk would leave a stub where it had been.
        write_user_file(plan.path, _starter_for(plan))
        written.append(plan.path)
    return tuple(written)
