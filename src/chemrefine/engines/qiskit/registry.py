"""Typed-option registries for independently replaceable Qiskit components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from chemrefine.engines.api import BackendRequirement, ComponentCategory, ComponentDescriptor
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
    status: Literal["standard", "experimental"] = "standard"
    supported_domains: tuple[str, ...] = ()


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
        status: Literal["standard", "experimental"] = "standard",
        supported_domains: tuple[str, ...] = (),
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
                status=status,
                supported_domains=supported_domains,
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

    def describe(self, default: str) -> ComponentCategory:
        """Publish the registered schemas without constructing any provider object."""
        return ComponentCategory(
            default=default,
            components={
                name: ComponentDescriptor(
                    options_schema=spec.options_cls.model_json_schema(),
                    capabilities=tuple(sorted(spec.capabilities)),
                    requires=tuple(sorted(spec.requires)),
                    backend_extra=(
                        spec.backend_requirement.extra if spec.backend_requirement else None
                    ),
                    execution=spec.execution,
                    status=spec.status,
                    supported_domains=spec.supported_domains,
                )
                for name, spec in sorted(self._specs.items())
            },
        )


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


def consumed_component_categories(options: QiskitOptions) -> frozenset[str]:
    """Follow resource requirements through every consumed component, without building it.

    For example an optimizer can require a sampler even when the algorithm declares
    only an estimator. Circuit and pool requirements also consume their reference state.
    Dependency cycles terminate because each selected category is visited only once.
    """
    algorithm = ALGORITHMS.spec(options.algorithm.name)
    pending = {"algorithm"}
    if algorithm.execution == "nature":
        pending.add("mapper")
    supplied_counts = False
    if options.algorithm.name in {"sqd", "extended_sqd"}:
        supplied_counts = (
            ALGORITHMS.options_for(options.algorithm).model_dump()["counts"] is not None
        )
        if not supplied_counts:
            pending.update({"mapper", "ansatz", "initial_state", "initial_point"})
    selections = options.component_selections()
    consumed: set[str] = set()
    while pending:
        category = pending.pop()
        consumed.add(category)
        requirements = REGISTRIES[category].spec(selections[category].name).requires
        resources = set(requirements & REGISTRIES.keys())
        if category == "algorithm" and supplied_counts:
            resources.discard("sampler")
        if requirements & {"circuit", "operator_pool"}:
            resources.update({"ansatz", "initial_state"})
        pending.update(resources - consumed)
    return frozenset(consumed)


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
    capabilities = ansatz.capabilities | ({"operator_pool"} if operator_pool_supplied else set())
    selections = options.component_selections()
    consumed = consumed_component_categories(options)
    if (
        "optimizer" in consumed
        and "circuit" in OPTIMIZERS.spec(options.optimizer.name).requires
        and "circuit" not in algorithm.requires
    ):
        raise ConfigError(
            f"qiskit optimizer {options.optimizer.name!r} requires a fixed-circuit algorithm"
        )
    for category in sorted(consumed):
        selection = selections[category]
        required = REGISTRIES[category].spec(selection.name).requires
        missing = required - capabilities - REGISTRIES.keys()
        if missing:
            raise ConfigError(
                f"qiskit {category} {selection.name!r} requires ansatz capabilities "
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
    executors = consumed & {"estimator", "sampler"}
    if options.device == "cuda" and not executors:
        raise ConfigError(
            f"qiskit algorithm {options.algorithm.name!r} is CPU-only; set device: cpu"
        )
    for category in sorted(executors):
        selection = selections[category]
        executor = REGISTRIES[category].spec(selection.name)
        if options.device == "cuda" and "cuda" not in executor.capabilities:
            raise ConfigError(
                f"qiskit {category} {selection.name!r} does not support device: cuda; "
                "choose a GPU-capable Aer component, or set device: cpu"
            )
        if (category, selection.name) not in {("sampler", "aer"), ("estimator", "aer_shots")}:
            continue
        method = REGISTRIES[category].options_for(selection).model_dump()["method"]
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
