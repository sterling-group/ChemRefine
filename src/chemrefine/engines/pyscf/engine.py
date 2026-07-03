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
from chemrefine.state import StepContext


@register("pyscf")
class PyscfEngine(ScriptEngine):
    """Direct PySCF engine — runs the user's ``step{N}.py`` per structure."""

    name: ClassVar[str] = "pyscf"
    label: ClassVar[str] = "PySCF"

    def backend_requirement(self, options: dict[str, Any] | None) -> BackendRequirement:
        """The backend env this step needs — PySCF, whatever the options say."""
        return BackendRequirement(extra="pyscf", import_name="pyscf")

    def backend_extras(self) -> frozenset[str]:
        """The one extra this engine can require."""
        return frozenset({"pyscf"})

    def _template_vars(self, ctx: StepContext) -> dict[str, object]:
        """Expose the SCF knobs as template placeholders, for parity with direct MLIP.

        Lets a direct ``step{N}.py`` read ``$METHOD`` / ``$XC`` / ``$BASIS`` from
        the YAML ``step.options`` instead of hardcoding them. Read tolerantly (with
        the standard fallbacks) so a template's extra knobs never fail the render;
        the ``pyscf-extopt`` path validates strictly via
        :class:`~chemrefine.engines.pyscf.options.PyscfOptions`.
        """
        raw = ctx.step_cfg.options or {}
        return {
            "METHOD": raw.get("method", "dft"),
            "XC": raw.get("xc", "pbe"),
            "BASIS": raw.get("basis", "def2-svp"),
        }
