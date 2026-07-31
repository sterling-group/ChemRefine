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

from chemrefine.engines._script import ScriptEngine
from chemrefine.engines.api import register
from chemrefine.engines.pyscf.backend import PyscfBackend
from chemrefine.engines.pyscf.options import PyscfOptions


@register("pyscf")
class PyscfEngine(PyscfBackend, ScriptEngine[PyscfOptions]):
    """Direct PySCF engine — runs the user's ``step{N}.py`` per structure."""

    name: ClassVar[str] = "pyscf"
    label: ClassVar[str] = "PySCF"
    options_cls: ClassVar[type[PyscfOptions]] = PyscfOptions

    def _vars_from(self, opts: PyscfOptions) -> dict[str, object]:
        """Expose the SCF knobs as template placeholders, for parity with direct MLIP.

        Lets a direct ``step{N}.py`` read ``$METHOD`` / ``$XC`` / ``$BASIS`` from the YAML
        ``step.options`` instead of hardcoding them. The base reads them leniently, so a
        template's extra knobs never fail the render — the ``pyscf-extopt`` path is the one
        that validates strictly, since it also has to require ``basis`` / ``xc`` explicitly.
        """
        return {"METHOD": opts.method, "XC": opts.xc, "BASIS": opts.basis}
