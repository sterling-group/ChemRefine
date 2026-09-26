"""Typed-option registries for independently replaceable Qiskit components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.options import ComponentSelection, QiskitOptions
from chemrefine.errors import ConfigError

ComponentBuilder = Callable[..., Any]


class NoComponentOptions(BaseModel):
    """Strict empty options model for components that expose no knobs."""

    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass(frozen=True)
class ComponentSpec:
    """A component's schema, builder, capabilities, and optional provider environment."""

    options_cls: type[BaseModel]
    builder: ComponentBuilder
    capabilities: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()
    backend_requirement: BackendRequirement | None = None
    execution: Literal["nature", "native"] = "nature"


class ComponentRegistry:
    """Registry mapping stable YAML names to lazy component builders."""

    def __init__(self, category: str) -> None:
        self.category = category
        self._specs: dict[str, ComponentSpec] = {}

    def register(
        self,
        name: str,
        options_cls: type[BaseModel] = NoComponentOptions,
        *,
        capabilities: frozenset[str] = frozenset(),
        requires: frozenset[str] = frozenset(),
        backend_requirement: BackendRequirement | None = None,
        execution: Literal["nature", "native"] = "nature",
    ) -> Callable[[ComponentBuilder], ComponentBuilder]:
        """Decorate a builder and register its complete declaration exactly once."""
        key = name.strip().lower().replace("-", "_")

        def decorate(builder: ComponentBuilder) -> ComponentBuilder:
            if key in self._specs:
                raise ValueError(f"{self.category} component {key!r} is already registered")
            self._specs[key] = ComponentSpec(
                options_cls=options_cls,
                builder=builder,
                capabilities=capabilities,
                requires=requires,
                backend_requirement=backend_requirement,
                execution=execution,
            )
            return builder

        return decorate

    def spec(self, name: str) -> ComponentSpec:
        """Return a component declaration or raise an actionable config error."""
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ConfigError(
                f"unsupported qiskit {self.category} component {name!r} "
                f"(known: {sorted(self._specs)})"
            ) from exc

    def options_for(self, selection: ComponentSelection) -> BaseModel:
        """Validate and return one selection's component-specific options."""
        spec = self.spec(selection.name)
        try:
            return spec.options_cls(**selection.options)
        except ValidationError as exc:
            raise ConfigError(
                f"invalid qiskit {self.category} options for {selection.name!r}:\n{exc}"
            ) from exc

    def build(self, selection: ComponentSelection, **context: Any) -> Any:
        """Validate ``selection`` and invoke its lazy builder."""
        spec = self.spec(selection.name)
        return spec.builder(options=self.options_for(selection), **context)

    def names(self) -> frozenset[str]:
        """Return every registered key in this category."""
        return frozenset(self._specs)


MAPPERS = ComponentRegistry("mapper")
ALGORITHMS = ComponentRegistry("algorithm")
ANSATZE = ComponentRegistry("ansatz")
INITIAL_STATES = ComponentRegistry("initial_state")
ESTIMATORS = ComponentRegistry("estimator")
SAMPLERS = ComponentRegistry("sampler")
OPTIMIZERS = ComponentRegistry("optimizer")
INITIAL_POINTS = ComponentRegistry("initial_point")

REGISTRIES: dict[str, ComponentRegistry] = {
    "mapper": MAPPERS,
    "algorithm": ALGORITHMS,
    "ansatz": ANSATZE,
    "initial_state": INITIAL_STATES,
    "estimator": ESTIMATORS,
    "sampler": SAMPLERS,
    "optimizer": OPTIMIZERS,
    "initial_point": INITIAL_POINTS,
}


def validate_component_graph(
    options: QiskitOptions, *, operator_pool_supplied: bool = False
) -> None:
    """Validate names, per-component options, and algorithm/ansatz compatibility."""
    for category, selection in options.component_selections().items():
        REGISTRIES[category].options_for(selection)

    from chemrefine.engines.qiskit.components.fermionic_algorithms import validate_ffsim_options
    from chemrefine.engines.qiskit.components.subspace_algorithms import validate_subspace_options

    if options.algorithm.name == "ffsim_vqe":
        validate_ffsim_options(options)
    validate_subspace_options(options)

    algorithm = ALGORITHMS.spec(options.algorithm.name)
    ansatz = ANSATZE.spec(options.ansatz.name)
    estimator = ESTIMATORS.spec(options.estimator.name)
    component_requirements = {"estimator", "sampler", "optimizer", "initial_state", "initial_point"}
    capabilities = ansatz.capabilities | ({"operator_pool"} if operator_pool_supplied else set())
    missing = algorithm.requires - capabilities - component_requirements
    if missing:
        raise ConfigError(
            f"qiskit algorithm {options.algorithm.name!r} requires ansatz capabilities "
            f"{sorted(missing)}, but {options.ansatz.name!r} provides "
            f"{sorted(ansatz.capabilities)}"
        )
    if (
        options.algorithm.name == "adapt_vqe"
        and not operator_pool_supplied
        and options.ansatz.name in {"uccsd", "ucc", "ucc_ranks"}
        and getattr(ANSATZE.options_for(options.ansatz), "reps", 1) != 1
    ):
        raise ConfigError(
            "qiskit ADAPT-VQE consumes the UCC operator pool rather than its repeated "
            "fixed circuit; set ansatz.options.reps to 1"
        )
    if options.device == "cuda":
        execution_category = "sampler" if "sampler" in algorithm.requires else "estimator"
        if execution_category not in algorithm.requires:
            raise ConfigError(
                f"qiskit algorithm {options.algorithm.name!r} is CPU-only; set device: cpu"
            )
        executor = (
            SAMPLERS.spec(options.sampler.name) if execution_category == "sampler" else estimator
        )
        if "cuda" not in executor.capabilities:
            selected = options.component_selections()[execution_category]
            raise ConfigError(
                f"qiskit {execution_category} {selected.name!r} does not support device: cuda; "
                "choose a GPU-capable Aer component, or set device: cpu"
            )
    aer_selection = (
        SAMPLERS.options_for(options.sampler)
        if "sampler" in algorithm.requires and options.sampler.name == "aer"
        else ESTIMATORS.options_for(options.estimator)
        if options.estimator.name == "aer_shots"
        else None
    )
    if aer_selection is not None:
        method = aer_selection.model_dump()["method"]
        if method == "tensor_network" and options.device != "cuda":
            raise ConfigError("qiskit Aer method 'tensor_network' requires device: cuda")
        if options.device == "cuda" and method not in {
            "automatic",
            "statevector",
            "density_matrix",
            "tensor_network",
        }:
            raise ConfigError(
                f"qiskit Aer method {method!r} is not GPU-compatible; choose automatic, "
                "statevector, density_matrix, or tensor_network"
            )
