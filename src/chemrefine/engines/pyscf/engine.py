"""Direct template-driven PySCF engine; registered under the YAML name ``"pyscf"``.

All lifecycle logic lives on
:class:`chemrefine.engines._script.ScriptEngine`; this
module binds the backend identity (``name`` + ``label``), the registry
entry, and the option placeholders the template can use.

The ORCA-driven PySCF flavor (``engine: pyscf-extopt``) is unrelated;
see :mod:`chemrefine.engines.pyscf.extopt_engine`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from chemrefine.engines._script import ScriptEngine
from chemrefine.engines.api import BackendRequirement, register
from chemrefine.engines.pyscf.options import PyscfOptions
from chemrefine.state import StepContext


@register("pyscf")
class PyscfEngine(ScriptEngine):
    """Direct PySCF engine — runs the user's ``step{N}.py`` per structure."""

    name: ClassVar[str] = "pyscf"
    label: ClassVar[str] = "PySCF"
    options_cls: ClassVar[type[PyscfOptions]] = PyscfOptions

    def backend_requirement(self, options: dict[str, Any] | None) -> BackendRequirement:
        """The backend env this step needs — PySCF, whatever the options say."""
        return BackendRequirement(extra="pyscf", import_name="pyscf")

    def backend_extras(self) -> frozenset[str]:
        """The one extra this engine can require."""
        return frozenset({"pyscf"})

    def _template_vars(self, ctx: StepContext) -> dict[str, object]:
        """Expose the SCF knobs as template placeholders, for parity with direct MLIP.

        Lets a direct ``step{N}.py`` read ``$METHOD`` / ``$XC`` / ``$BASIS`` from the YAML
        ``step.options`` instead of hardcoding them. Read leniently, so a template's extra
        knobs never fail the render — the ``pyscf-extopt`` path is the one that validates
        strictly, since it also has to require ``basis`` / ``xc`` explicitly.

        Through :attr:`options_cls`, not off the raw dict. The defaults spelled here by
        hand happened to match the model's, but "happened to" is the whole problem: the
        same shape — a literal default beside a model that declares one — has already
        split twice in this codebase, and neither split was visible until it produced a
        wrong job.
        """
        opts = self.options_cls.from_raw_lenient(ctx.step_cfg.options)
        return {"METHOD": opts.method, "XC": opts.xc, "BASIS": opts.basis}
