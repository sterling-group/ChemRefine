"""Direct template-driven MLIP engine, registered as ``"mlip"``.

Legacy ``mlff`` YAML is rewritten to ``mlip`` by the config normalizer.

All lifecycle logic lives on
:class:`chemrefine.engines._script.ScriptEngine`; this
module just binds the backend identity (``name`` + ``label``), the registry
entry, and the option placeholders the template can use. The user picks any
ASE-compatible MLIP library by importing it inside their ``step{N}.py`` template
(or by calling :class:`~chemrefine.engines.mlip.calculator.MlipCalculator` with
the injected ``$MODEL_NAME`` / ``$TASK_NAME`` / ``$DEVICE``), so the YAML
``step.options`` drive a direct run the same way they drive ``mlip-extopt``.

The ORCA-driven MLIP flavor (``engine: mlip-extopt``) is unrelated;
see :mod:`chemrefine.engines.mlip.extopt_engine`.
"""

from __future__ import annotations

from typing import ClassVar

from chemrefine.engines._script import ScriptEngine
from chemrefine.engines.api import register
from chemrefine.engines.mlip.options import MlipOptions
from chemrefine.state import StepContext


@register("mlip")
class MlipEngine(ScriptEngine):
    """Direct MLIP engine — runs the user's ``step{N}.py`` per structure."""

    name: ClassVar[str] = "mlip"
    label: ClassVar[str] = "MLIP"

    def _template_vars(self, ctx: StepContext) -> dict[str, object]:
        """Expose the MLIP options as template placeholders.

        Lets a direct ``step{N}.py`` read ``$MODEL_NAME`` / ``$TASK_NAME`` /
        ``$DEVICE`` from the YAML ``step.options`` instead of hardcoding them.
        Reads tolerantly (alias-aware, ignoring unknown keys) so a template's
        extra knobs never fail the render; the ExtOpt path validates strictly.
        """
        raw = ctx.step_cfg.options or {}
        defaults = MlipOptions()
        return {
            "MODEL_NAME": raw.get("model_name")
            or raw.get("model")
            or raw.get("size")
            or defaults.model_name,
            "TASK_NAME": raw.get("task_name") or raw.get("task") or defaults.task_name,
            "DEVICE": raw.get("device") or defaults.device,
        }
