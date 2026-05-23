"""Direct template-driven MLFF engine; registered under the YAML name ``"mlff"``.

All lifecycle logic lives on
:class:`chemrefine.engines._template_engine.TemplateScriptEngine`; this
module just binds the backend identity (``name`` + ``label``) and the
registry entry. The user picks any ASE-compatible MLFF library by
importing it inside their ``step{N}.py`` template — ChemRefine does
not need to know about each backend. For the built-in factory used by
the ORCA-driven sibling, see
:mod:`chemrefine.engines.mlff.calculator`.

The ORCA-driven MLFF flavor (``engine: mlff-extopt``) is unrelated;
see :mod:`chemrefine.engines.mlff.extopt_engine`.
"""

from __future__ import annotations

from chemrefine.engines._template_engine import TemplateScriptEngine
from chemrefine.engines.base import register


@register("mlff")
class MlffEngine(TemplateScriptEngine):
    """Direct MLFF engine — runs the user's ``step{N}.py`` per structure."""

    name = "mlff"
    label = "MLFF"
