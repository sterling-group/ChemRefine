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

from typing import Any, ClassVar

from chemrefine.engines._script import ScriptEngine
from chemrefine.engines.api import BackendRequirement, register
from chemrefine.engines.mlip.calculator import registered_extras, requirement_from_options
from chemrefine.engines.mlip.options import MlipOptions
from chemrefine.state import StepContext


@register("mlip")
class MlipEngine(ScriptEngine):
    """Direct MLIP engine — runs the user's ``step{N}.py`` per structure."""

    name: ClassVar[str] = "mlip"
    label: ClassVar[str] = "MLIP"

    def backend_requirement(self, options: dict[str, Any] | None) -> BackendRequirement:
        """The backend env this step needs — derived from its task/model selection."""
        return requirement_from_options(options)

    def backend_extras(self) -> frozenset[str]:
        """Every extra a registered MLIP backend declares."""
        return registered_extras()

    def _template_vars(self, ctx: StepContext) -> dict[str, object]:
        """Expose the MLIP options as template placeholders.

        Lets a direct ``step{N}.py`` read ``$MODEL_NAME`` / ``$TASK_NAME`` /
        ``$DEVICE`` from the YAML ``step.options`` instead of hardcoding them.
        Read leniently, through :class:`MlipOptions` itself: a template may carry
        knobs no engine model declares and rendering must not fail over them, but
        the alias rules (``model`` / ``size`` for ``model_name``) belong to the
        model rather than being spelled out a second time here.
        """
        opts = MlipOptions.from_raw_lenient(ctx.step_cfg.options)
        return {
            "MODEL_NAME": opts.model_name,
            "TASK_NAME": opts.task_name,
            "DEVICE": opts.device,
        }
