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
et al.), keyed by engine name with a suffix-shaped fallback, so a drop-in engine still
scaffolds something useful. The trainer's starter is honest about its limits: a real
``mlip-train`` template is load-bearing in ways a stub cannot be, so it points at the
worked example rather than pretending.

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
from chemrefine.engines.api import JobExecutable, TemplateDriven, get_engine
from chemrefine.errors import ConfigError, EngineNotFoundError
from chemrefine.ids import step_template_path
from chemrefine.validate import effective_header

_STEP_STARTERS: dict[str, str] = {
    "orca": (
        "# ORCA starter — edit the keywords; ChemRefine appends each structure's geometry.\n"
        "! B3LYP D4 def2-SVP Opt\n"
        "%pal nprocs 4 end\n"
        "%maxcore 2000\n"
    ),
    "qchem": (
        # The comment deliberately never spells a section name: the input writer's block
        # regexes are line-anchored, but a starter that does not mention them is one whose
        # rendering can never depend on that anchoring.
        "$comment\n"
        "Q-Chem starter — ChemRefine swaps each structure's geometry into the first\n"
        "coordinate block below.\n"
        "$end\n"
        "\n"
        "$molecule\n"
        "0 1\n"
        "H 0.0 0.0 0.0\n"
        "$end\n"
        "\n"
        "$rem\n"
        "  jobtype     opt\n"
        "  method      b3lyp\n"
        "  basis       def2-svp\n"
        "$end\n"
    ),
    "mlip": (
        "# MLIP starter. Rendered per structure: $XYZ_PATH / $CHARGE / $MULTIPLICITY come\n"
        "# from the pipeline, $MODEL_NAME / $TASK_NAME / $DEVICE from the step options.\n"
        "$OUTPUT_CONTRACT"
        "from ase.io import read\n"
        "from ase.units import Hartree\n"
        "\n"
        "from chemrefine.engines.mlip.calculator import MlipCalculator\n"
        "\n"
        'mlip = MlipCalculator(model_name="$MODEL_NAME", task_name="$TASK_NAME", '
        'device="$DEVICE")\n'
        'atoms = mlip.optimize(read("$XYZ_PATH"), fmax=0.03)\n'
        "\n"
        "energy_hartree = atoms.get_potential_energy() / Hartree\n"
        "positions_angstrom = atoms.get_positions()\n"
    ),
    "pyscf": (
        "# PySCF starter. Rendered per structure: $XYZ_PATH / $CHARGE / $MULTIPLICITY come\n"
        "# from the pipeline, $METHOD / $XC / $BASIS / $DF from the step options.\n"
        "$OUTPUT_CONTRACT"
        "from pyscf import dft, gto, scf\n"
        "\n"
        "mol = gto.M(\n"
        '    atom="$XYZ_PATH",\n'
        '    basis="$BASIS",\n'
        "    charge=$CHARGE,\n"
        "    spin=$MULTIPLICITY - 1,\n"
        ")\n"
        "\n"
        'if "$METHOD" == "hf":\n'
        "    mf = scf.HF(mol)\n"
        "else:\n"
        '    mf = dft.KS(mol, xc="$XC")\n'
        "if $DF:\n"
        "    mf = mf.density_fit()\n"
        "\n"
        "energy_hartree = mf.kernel()\n"
    ),
    "mlip-train": (
        "# mlip-train starter — NOT runnable as written. A trainer template is the\n"
        "# backend's own config where one exists (mace/fairchem: every part of a working\n"
        "# one is load-bearing; sevenn: `sevenn preset fine_tune` writes one), and\n"
        "# chemrefine's own small schema where none does (chgnet/orb). Start from the\n"
        "# worked examples instead:\n"
        "#   examples/fairchem_finetune/templates/  (UMA fine-tune, commented line by line)\n"
        "#   docs -> Engines -> MLIP training templates  (one per trainable backend)\n"
    ),
}
_STEP_STARTERS["mlip-extopt"] = _STEP_STARTERS["orca"]
_STEP_STARTERS["pyscf-extopt"] = _STEP_STARTERS["orca"]

_SUFFIX_FALLBACKS: dict[str, str] = {
    "inp": _STEP_STARTERS["orca"],
    "py": (
        "# Script starter. Rendered per structure: $XYZ_PATH / $CHARGE / $MULTIPLICITY\n"
        "# come from the pipeline; step options render as $UPPERCASE placeholders.\n"
        "$OUTPUT_CONTRACT"
    ),
}
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
    """
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
            plans.append(
                TemplatePlan(
                    path=path,
                    exists=path.is_file(),
                    kind="step",
                    step=step.step,
                    engine=step.engine,
                )
            )
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
    """The starter body for one planned file — engine-keyed, suffix fallback, generic."""
    if plan.kind == "slurm-header":
        return _HEADER_STARTERS.get(plan.path.name, _HEADER_DEFAULT)
    starter = _STEP_STARTERS.get(plan.engine) if plan.engine is not None else None
    if starter is None:
        suffix = plan.path.suffix.lstrip(".")
        starter = _SUFFIX_FALLBACKS.get(suffix, _GENERIC_STARTER)
    # `plan_templates` resolves every engine before any starter is chosen, so a name that
    # does not resolve reaches here only from a direct caller — and an engine nobody can look
    # up has no contract to describe, which is what a non-script engine answers too.
    engine: object = None
    if plan.engine is not None:
        with suppress(EngineNotFoundError):
            engine = get_engine(plan.engine)
    # Only this one placeholder is filled: a starter is a *template*, and its `$XYZ_PATH`,
    # `$CHARGE` and option placeholders belong to the renderer that runs per structure.
    return starter.replace("$OUTPUT_CONTRACT", _output_contract_comment(engine))


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
            if plan.exists and not overwrite:
                continue
            # A `template:` override may name a subdirectory (or an absolute path elsewhere)
            # — the same shape `agent_tools.write_template` already creates parents for.
            plan.path.parent.mkdir(parents=True, exist_ok=True)
            plan.path.write_text(_starter_for(plan), encoding="utf-8")
            written.append(plan.path)
    except OSError as e:
        raise ConfigError(f"cannot scaffold templates under {config.template_dir}: {e}") from e
    return tuple(written)
