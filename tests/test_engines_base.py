"""Tests for the engine Protocol, the ENGINES registry, and registration."""

from __future__ import annotations

import pytest

# Importing from ``chemrefine.engines.api`` triggers the parent package's
# ``__init__``, which self-registers every bundled engine into ``ENGINES``.
from chemrefine.engines.api import (
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

        def prepare(self, ctx):
            return None

        def submit(self, inputs, ctx):
            return None

        def parse(self, inputs, ctx):
            return None

    try:
        # ENGINES is typed to the protocol; the registry stores the concrete class.
        assert ENGINES["temp-test-engine"] is _TempEngine  # type: ignore[comparison-overlap]
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


def test_registry_holds_only_canonical_engine_names():
    """Legacy spellings (``mlff*``, ``dft``) are mapped by the config normalizer,
    not the registry — so the registry exposes only canonical names."""
    assert {"orca", "mlip", "mlip-extopt", "mlip-train"} <= set(ENGINES)
    for legacy in ("mlff", "mlff-extopt", "mlff-train", "dft"):
        assert legacy not in ENGINES


# --- base: the abstract SLURM hooks -----------------------------------------


def test_an_incomplete_job_engine_cannot_be_constructed():
    """A subclass missing a primitive fails at construction, not after submitting jobs.

    These four are what a ``JobEngine`` cannot supply for itself. While they raised
    ``NotImplementedError`` on call, a subclass that forgot one still satisfied ``isinstance``,
    still registered, and still prepared and submitted every job of a step — the failure
    surfaced in ``parse``, once the cluster time was already spent. Abstract, it surfaces in
    ``get_engine``.
    """
    from chemrefine.engines._job import JobEngine

    class _MissingParseOne(JobEngine):
        name = "incomplete"
        label = "Incomplete"
        template_suffix = "inp"
        output_suffix = "out"
        output_globs = ("*.out",)

        def build_input(self, *, xyz_path, template_path, input_path, output_path, ctx): ...

        def run_block(self, ctx, inp_path, out_path): ...

        def pal(self, ctx):
            return 1

    with pytest.raises(TypeError, match="parse_one"):
        _MissingParseOne()  # type: ignore[abstract]
