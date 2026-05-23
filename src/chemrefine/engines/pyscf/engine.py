"""Direct template-driven PySCF engine; registered under the YAML name ``"pyscf"``.

All lifecycle logic lives on
:class:`chemrefine.engines._template_engine.TemplateScriptEngine`; this
module just binds the backend identity (``name`` + ``label``) and the
registry entry.

The ORCA-driven PySCF flavor (``engine: pyscf-extopt``) is unrelated;
see :mod:`chemrefine.engines.pyscf.extopt_engine`.
"""

from __future__ import annotations

from chemrefine.engines._template_engine import TemplateScriptEngine
from chemrefine.engines.base import register


@register("pyscf")
class PyscfEngine(TemplateScriptEngine):
    """Direct PySCF engine — runs the user's ``step{N}.py`` per structure."""

    name = "pyscf"
    label = "PySCF"
