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
from chemrefine.engines.mlip.backend import MlipBackend
from chemrefine.engines.mlip.options import CALCULATOR_KNOBS, MlipOptions


@register("mlip")
class MlipEngine(MlipBackend, ScriptEngine[MlipOptions]):
    """Direct MLIP engine — runs the user's ``step{N}.py`` per structure."""

    name: ClassVar[str] = "mlip"
    label: ClassVar[str] = "MLIP"
    options_cls: ClassVar[type[MlipOptions]] = MlipOptions

    def _vars_from(self, opts: MlipOptions) -> dict[str, object]:
        """Expose the MLIP options as template placeholders.

        Lets a direct ``step{N}.py`` read ``$MODEL_NAME`` / ``$TASK_NAME`` / ``$DEVICE`` /
        ``$MODEL_PATH`` from the YAML ``step.options`` instead of hardcoding them. The base
        already read them through :class:`MlipOptions` — including its alias rules (``model``
        / ``size`` for ``model_name``), which belong to the model rather than being spelled
        out again here.

        ``MODEL_PATH`` is here because the knob exists: a step can select a local checkpoint,
        and without the placeholder its own template had no way to name the file it selected.
        Empty when unset, so a template that never uses it renders unchanged — which is why
        adding it cannot disturb an existing one.
        """
        return {name.upper(): getattr(opts, name) or "" for name in CALCULATOR_KNOBS}
