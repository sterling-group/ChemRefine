"""Provider profiles and registry schemas stay in step with component consumption."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest

from chemrefine.engines.api import (
    ENGINES,
    BackendRequirement,
    ComponentCatalogDeclaring,
    ComponentCategory,
    ComponentDescriptor,
    get_engine,
)
from chemrefine.engines.qiskit.backend import QiskitBackend
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.profiles import BACKEND_PROFILES, BackendProfileRegistry
from chemrefine.engines.qiskit.registry import (
    INITIAL_STATES,
    OPTIMIZERS,
    REGISTRIES,
    SAMPLERS,
    ComponentRegistry,
    ComponentSpec,
    NoComponentOptions,
    consumed_component_categories,
)
from chemrefine.errors import ConfigError
from chemrefine.introspect import schema_document


def test_catalog_matches_all_registered_models_and_defaults():
    """Every runtime selection is discoverable with precisely its validating schema."""
    engine = get_engine("qiskit")
    assert isinstance(engine, ComponentCatalogDeclaring)
    catalog = engine.component_catalog()
    defaults = QiskitOptions().component_selections()
    assert set(catalog) == set(REGISTRIES)
    for category, registry in REGISTRIES.items():
        assert catalog[category].default == defaults[category].name
        assert list(catalog[category].components) == sorted(registry.names())
        for name in registry.names():
            spec = registry.spec(name)
            described = catalog[category].components[name]
            assert described.options_schema == spec.options_cls.model_json_schema()
            assert described.requires == tuple(sorted(spec.requires))
            assert described.capabilities == tuple(sorted(spec.capabilities))
            assert described.backend_extra == (
                spec.backend_requirement.extra if spec.backend_requirement else None
            )
    document = schema_document()
    assert document["engines"]["qiskit"]["component_catalog"] == {
        name: dataclasses.asdict(category) for name, category in catalog.items()
    }
    assert document["engines"]["orca"]["component_catalog"] is None
    json.dumps(document)


def test_catalog_is_generic_and_reflects_registration(monkeypatch):
    """Introspection detects the capability rather than naming Qiskit specially."""
    category = ComponentCategory("demo", {"demo": ComponentDescriptor({"type": "object"})})

    class CatalogEngine(type(get_engine("fake"))):
        """A third-party engine with its own nested selection."""

        def component_catalog(self):
            """Supply an ordinary public catalog declaration."""
            return {"solver": category}

    monkeypatch.setitem(ENGINES, "catalog_test", CatalogEngine)
    assert schema_document()["engines"]["catalog_test"]["component_catalog"] == {
        "solver": dataclasses.asdict(category)
    }
    registry = ComponentRegistry("demo")
    registry.register("new", requires=frozenset({"sampler"}))(lambda: pytest.fail("not built"))
    assert registry.describe("new").components["new"].requires == ("sampler",)


def test_schema_discovery_never_imports_compute_providers():
    """Even an SDK-free wheel can publish complete component schemas."""
    script = """
import importlib.abc
import sys
class NoCompute(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'qiskit', 'qiskit_nature', 'qiskit_algorithms',
                                      'qiskit_aer', 'ffsim', 'qiskit_fermions', 'pyscf'}:
            raise AssertionError('optional import: ' + fullname)
sys.meta_path.insert(0, NoCompute())
from chemrefine.introspect import schema_document
assert schema_document()['engines']['qiskit']['component_catalog']['algorithm']['components']
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_profiles_resolve_transitive_combination_without_selection_of_the_profile():
    """Two required providers resolve through a separately registered compatible superset."""
    profiles = BackendProfileRegistry()
    base = BackendRequirement("qiskit", "qiskit_nature")
    aer = BackendRequirement("aer", "qiskit_aer")
    remote = BackendRequirement("remote", "remote")
    shared = BackendRequirement("shared", "shared")
    profiles.register(base)
    profiles.register(aer, includes=frozenset({"qiskit"}))
    profiles.register(remote, includes=frozenset({"qiskit"}))
    profiles.register(shared, includes=frozenset({"aer", "remote"}))
    assert profiles.provided_extras("leaf") == {"leaf"}
    assert profiles.provided_extras("shared") == {"shared", "aer", "remote", "qiskit"}
    assert profiles.resolve([]) == base
    assert profiles.resolve([aer]) == aer
    assert profiles.resolve([remote, aer]) == shared
    assert profiles.resolve([aer, remote]) == shared
    with pytest.raises(ValueError, match="already registered"):
        profiles.register(shared)


def test_profiles_reject_undeclared_combinations_and_cycles():
    """A fallback custom provider cannot silently absorb unrelated dependencies."""
    profiles = BackendProfileRegistry()
    left = BackendRequirement("left", "left")
    right = BackendRequirement("right", "right")
    assert profiles.resolve([left]) == left
    with pytest.raises(ConfigError, match="incompatible managed environments"):
        profiles.resolve([left, right])
    profiles.register(left, includes=frozenset({"right"}))
    profiles.register(right, includes=frozenset({"left"}))
    with pytest.raises(ConfigError, match="cyclic"):
        profiles.resolve([left])


def test_combined_profiles_are_discoverable(monkeypatch):
    """The provisioner's engine extra roster includes registered shared profiles."""
    from chemrefine.engines._provision import known_backend_extras

    monkeypatch.setattr(BACKEND_PROFILES, "_profiles", dict(BACKEND_PROFILES._profiles))
    BACKEND_PROFILES.register(
        BackendRequirement("qiskit-shared-test", "shared"),
        includes=frozenset({"qiskit-fermionic"}),
    )
    assert "qiskit-shared-test" in QiskitBackend().backend_extras()
    assert "qiskit-shared-test" in known_backend_extras()


def test_provider_dependencies_follow_optimizer_sampler_and_reference(monkeypatch):
    """Indirect resource requirements are charged to the same selected worker."""
    sampler_requirement = BackendRequirement("qiskit-test-sampler", "sampler_test")
    monkeypatch.setitem(
        OPTIMIZERS._specs,
        "indirect_test",
        ComponentSpec(NoComponentOptions, lambda: None, requires=frozenset({"sampler"})),
    )
    monkeypatch.setitem(
        SAMPLERS._specs,
        "indirect_test",
        ComponentSpec(
            NoComponentOptions,
            lambda: None,
            requires=frozenset({"optimizer"}),
            backend_requirement=sampler_requirement,
        ),
    )
    options = QiskitOptions(algorithm="vqe", optimizer="indirect_test", sampler="indirect_test")
    assert consumed_component_categories(options) == {
        "algorithm",
        "mapper",
        "estimator",
        "optimizer",
        "sampler",
        "initial_point",
        "initial_state",
        "ansatz",
    }
    assert QiskitBackend().backend_requirement(options.as_job_spec()) == sampler_requirement
    reference_requirement = BackendRequirement("qiskit-test-reference", "reference_test")
    monkeypatch.setitem(
        INITIAL_STATES._specs,
        "custom_test",
        ComponentSpec(NoComponentOptions, lambda: None, backend_requirement=reference_requirement),
    )
    assert (
        QiskitBackend().backend_requirement({"algorithm": "vqe", "initial_state": "custom_test"})
        == reference_requirement
    )


def test_unused_components_and_supplied_counts_add_no_resources():
    """Exact and supplied-count execution do not provision unused sampler components."""
    assert consumed_component_categories(QiskitOptions()) == {"algorithm", "mapper"}
    assert consumed_component_categories(QiskitOptions(algorithm="sqd")) == {
        "algorithm",
        "mapper",
        "sampler",
        "ansatz",
        "initial_state",
        "initial_point",
    }
    supplied = QiskitOptions(algorithm={"name": "sqd", "options": {"counts": {"11": 4}}})
    assert consumed_component_categories(supplied) == {"algorithm"}


def test_indirect_component_capabilities_fail_during_preflight(monkeypatch):
    """An optimizer's requirements cannot bypass the graph's capability checks."""
    monkeypatch.setitem(
        OPTIMIZERS._specs,
        "requires_pool",
        ComponentSpec(NoComponentOptions, lambda: None, requires=frozenset({"operator_pool"})),
    )
    with pytest.raises(ConfigError, match="optimizer 'requires_pool' requires ansatz"):
        QiskitBackend().backend_requirement(
            {"algorithm": "vqe", "ansatz": "efficient_su2", "optimizer": "requires_pool"}
        )
