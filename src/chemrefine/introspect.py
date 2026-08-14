"""Machine-readable descriptions of the config schema and the engine registry.

The one *produced* view of "what can a config say?". The Pydantic models in
:mod:`chemrefine.config` and each engine's declared options model are the source; this
module only assembles them into a JSON-serializable document, so nothing here is
hand-maintained and nothing can drift from what :func:`chemrefine.config.load_config`
actually validates with. Built for consumers that cannot import chemrefine: the
YAML-builder GUI renders its forms from :func:`schema_document`, an agent driving the
MCP server reads the same document before writing a config, and ``chemrefine schema`` /
``chemrefine engines --json`` print it.

Engine facts are read the way the pipeline reads them — capability Protocols detected
via ``isinstance`` (:class:`~chemrefine.engines.api.TemplateDriven`,
:class:`~chemrefine.engines.api.OptionsDeclaring`, …), never a parallel table — so a
drop-in engine package is described the moment it registers. An engine that declares no
options model is reported with ``options_schema: None`` rather than a schema it does not
honour: ORCA's knobs live in its step template, and inventing a form for it would
advertise fields the engine never reads.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from chemrefine import __version__
from chemrefine.config import Config
from chemrefine.engines.api import (
    ENGINES,
    ArtifactEngine,
    CalculationEngine,
    NmsCapableEngine,
    OptionsDeclaring,
    ProvisionableEngine,
    StreamingSubmit,
    TemplateDriven,
    get_engine,
)
from chemrefine.nms import NmsOptions


@dataclasses.dataclass(frozen=True)
class EngineDescriptor:
    """One registered engine, flattened for a consumer that cannot import it.

    Every field is derived from the engine class itself (ClassVars and capability
    Protocols), so the descriptor is exactly as current as the registry. ``capabilities``
    uses a small stable vocabulary — ``artifact`` / ``nms`` / ``provisionable`` /
    ``streaming`` — naming the Protocols the pipeline itself detects.
    """

    name: str
    """Registry key — the YAML ``engine:`` spelling."""
    template_driven: bool
    """Whether the engine reads a per-step template (``templates/stepN.<suffix>``)."""
    template_suffix: str | None
    """The template's extension without the dot; ``None`` for a template-free engine."""
    label: str | None
    """Human name used in the engine's own error messages; ``None`` when template-free."""
    options_schema: dict[str, Any] | None
    """JSON Schema of the engine's declared ``step.options`` model.

    ``None`` when the engine declares none (see
    :class:`~chemrefine.engines.api.OptionsDeclaring`) — the honest answer for an engine
    configured entirely through its template."""
    capabilities: tuple[str, ...]
    """Detected capability Protocols, sorted, from the stable vocabulary above."""
    backend_extras: tuple[str, ...]
    """Pip extras this engine's backends install from (``chemrefine backends install``)."""


def _describe(name: str, engine: CalculationEngine) -> EngineDescriptor:
    """Flatten one engine instance into its descriptor — one ``isinstance`` per capability."""
    template_driven = False
    template_suffix: str | None = None
    label: str | None = None
    if isinstance(engine, TemplateDriven):
        template_driven = True
        template_suffix = engine.template_suffix
        label = engine.label
    capabilities: list[str] = []
    if isinstance(engine, ArtifactEngine):
        capabilities.append("artifact")
    if isinstance(engine, NmsCapableEngine):
        capabilities.append("nms")
    if isinstance(engine, ProvisionableEngine):
        capabilities.append("provisionable")
    if isinstance(engine, StreamingSubmit):
        capabilities.append("streaming")
    return EngineDescriptor(
        name=name,
        template_driven=template_driven,
        template_suffix=template_suffix,
        label=label,
        options_schema=(
            engine.options_cls.model_json_schema() if isinstance(engine, OptionsDeclaring) else None
        ),
        capabilities=tuple(capabilities),
        backend_extras=(
            tuple(sorted(engine.backend_extras()))
            if isinstance(engine, ProvisionableEngine)
            else ()
        ),
    )


def describe_engines() -> tuple[EngineDescriptor, ...]:
    """A descriptor for every registered engine, sorted by name.

    Fresh instances via :func:`~chemrefine.engines.api.get_engine`, exactly as the
    orchestrator obtains them — so what this reports is what a run would use.
    """
    return tuple(_describe(name, get_engine(name)) for name in sorted(ENGINES))


def schema_document() -> dict[str, Any]:
    """The complete machine-readable schema document, JSON-serializable.

    Four parts: the package version (a consumer caches against it), the config schema
    (:class:`~chemrefine.config.Config` — steps, sampling and top-level keys, with the
    nested models under ``$defs``), the NMS knob schema
    (:class:`~chemrefine.nms.NmsOptions` — read from ``step.options`` by the NMS
    coordinator, not by any engine, so it is schema'd once here rather than merged into
    every engine's model), and one :class:`EngineDescriptor` per registered engine.
    """
    return {
        "chemrefine_version": __version__,
        "config": Config.model_json_schema(),
        "nms": NmsOptions.model_json_schema(),
        "engines": {d.name: dataclasses.asdict(d) for d in describe_engines()},
    }
