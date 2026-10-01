"""Direct template-driven PySCF engine; registered under the YAML name ``"pyscf"``.

All lifecycle logic lives on
:class:`chemrefine.engines._script.ScriptEngine`; this
module binds the backend identity (``name`` + ``label``), the registry
entry, and the option placeholders the template can use.

The ORCA-driven PySCF flavor (``engine: pyscf-extopt``) is unrelated;
see :mod:`chemrefine.engines.pyscf.extopt_engine`.
"""

from __future__ import annotations

from typing import ClassVar

from chemrefine.config import StepConfig
from chemrefine.engines._script import ScriptEngine
from chemrefine.engines.api import register
from chemrefine.engines.pyscf.backend import PyscfBackend
from chemrefine.engines.pyscf.options import PyscfOptions
from chemrefine.state import StepContext, StepInputs


@register("pyscf")
class PyscfEngine(PyscfBackend, ScriptEngine[PyscfOptions]):
    """Direct PySCF engine — runs the user's ``step{N}.py`` per structure."""

    name: ClassVar[str] = "pyscf"
    label: ClassVar[str] = "PySCF"
    options_cls: ClassVar[type[PyscfOptions]] = PyscfOptions
    template_starter: ClassVar[str] = (
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
        "# PySCF returns the last iterate rather than raising, so the verdict is reported here:\n"
        "# False is ledgered as a convergence failure instead of ranking as a result.\n"
        "converged = bool(mf.converged)\n"
    )
    """What ``chemrefine scaffold`` writes for a missing ``stepN.py`` — see
    :attr:`~chemrefine.engines.api.TemplateDriven.template_starter`; ``$OUTPUT_CONTRACT``
    becomes the comment naming this engine's output fields."""
    preflight_refuses: ClassVar[str] = (
        "a step naming no level of theory (`basis`, and `xc` under `method: dft`), the one "
        "engine that would otherwise compute at defaults nobody chose"
    )

    def check_step(self, step_cfg: StepConfig, *, charge: int, multiplicity: int) -> None:
        """Refuse a step that names no level of theory, before anything runs.

        A direct pyscf step names its level of theory in its options, so one that omits
        ``basis`` (or ``xc`` under ``method: dft``) is refused rather than rendered with
        defaults into ``$BASIS``/``$XC`` — a level the user never chose. The rule is
        :meth:`~chemrefine.engines.pyscf.options.PyscfOptions.require_level_of_theory`,
        shared with the ExtOpt engine's strict read; asked through the preflight hook
        rather than a strict validation because the script engines read leniently by
        documented design — a ``step{N}.py`` may carry knobs no model declares, and
        rendering must not fail over them. ``charge`` / ``multiplicity`` are unread —
        the signature is the capability's.
        """
        self.options_cls.require_level_of_theory(step_cfg.options)

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Re-make the preflight refusal, then render the inputs.

        :meth:`check_step` already ran at the run's preflight walk; repeating it here
        covers the recovery paths that reach ``prepare`` without one — the same
        discipline as the ExtOpt and training engines, for the same price.
        """
        self.check_step(ctx.step_cfg, charge=ctx.charge, multiplicity=ctx.multiplicity)
        return super().prepare(ctx)
