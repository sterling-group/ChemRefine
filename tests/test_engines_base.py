"""Tests for the engine Protocol, the ENGINES registry, and registration."""

from __future__ import annotations

import pytest

# Importing from ``chemrefine.engines.base`` triggers the parent package's
# ``__init__``, which self-registers every bundled engine into ``ENGINES``.
from chemrefine.engines.base import (
    ENGINES,
    CalculationEngine,
    get_engine,
    register,
)
from chemrefine.errors import EngineNotFoundError


def test_engines_registry_is_populated_by_side_effect_import():
    """Importing chemrefine.engines must trigger every bundled engine's registration."""
    assert "fake" in ENGINES


def test_get_engine_returns_fresh_instance():
    engine_a = get_engine("fake")
    engine_b = get_engine("fake")
    # Same class, distinct instances.
    assert type(engine_a) is type(engine_b)
    assert engine_a is not engine_b


def test_get_engine_unknown_raises():
    with pytest.raises(EngineNotFoundError):
        get_engine("definitely-not-registered")


def test_every_registered_engine_satisfies_protocol():
    """Every entry in ENGINES must structurally conform to ``CalculationEngine``."""
    for name, engine_cls in ENGINES.items():
        instance = engine_cls()
        assert isinstance(instance, CalculationEngine), (
            f"engine {name!r} does not conform to CalculationEngine"
        )


def test_register_decorator_adds_entry_and_returns_class():
    @register("temp-test-engine")
    class _TempEngine:
        name = "temp-test-engine"
        supports_nms = False

        def prepare(self, ctx):
            return None

        def submit(self, inputs, ctx):
            return None

        def wait(self, batch):
            return None

        def parse(self, inputs, ctx):
            return None

        def normal_mode_sample(self, results, ctx):
            return None

    try:
        assert ENGINES["temp-test-engine"] is _TempEngine
    finally:
        ENGINES.pop("temp-test-engine", None)


def test_register_rejects_duplicate_with_different_class():
    @register("dup-test-engine")
    class _A:
        pass

    with pytest.raises(ValueError):

        @register("dup-test-engine")
        class _B:
            pass

    ENGINES.pop("dup-test-engine", None)
