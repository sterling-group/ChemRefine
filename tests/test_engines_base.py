"""Tests for the engine Protocol, the ENGINES registry, and registration."""

from __future__ import annotations

import pytest

# Importing from ``chemrefine.engines.api`` triggers the parent package's
# ``__init__``, which self-registers every bundled engine into ``ENGINES``.
from chemrefine.engines._job import JobEngine
from chemrefine.engines.api import (
    ENGINES,
    CalculationEngine,
    contract_members,
    get_engine,
    register,
)
from chemrefine.errors import EngineNotFoundError
from chemrefine.state import RunBlock


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
        assert ENGINES["temp-test-engine"] is _TempEngine
    finally:
        ENGINES.pop("temp-test-engine", None)


def _conforming(engine_name: str) -> type:
    """The smallest class ``register`` accepts — the contract and nothing else.

    Both halves matter: it satisfies ``CalculationEngine`` structurally, and it does not
    inherit it. A test that needs *a registrable engine* and not a particular one uses this,
    so the gate is exercised rather than worked around.
    """

    class _Engine:
        # On the class, not bound in __init__: the gate runs without constructing anything,
        # which is what stops an engine's __init__ running at import of the package.
        name = engine_name

        def prepare(self, ctx: object) -> None:
            return None

        def submit(self, inputs: object, ctx: object) -> None:
            return None

        def parse(self, inputs: object, ctx: object) -> None:
            return None

    return _Engine


def test_register_rejects_duplicate_with_different_class():
    # try/finally like the temp-test-engine above: if the duplicate guard under test ever
    # regressed, a bare pop after the raises-block would leave a stray class in the global
    # registry, and every wholesale ENGINES read across the suite would fail alongside the
    # one real regression.
    try:
        register("dup-test-engine")(_conforming("dup-test-engine"))
        with pytest.raises(ValueError):
            register("dup-test-engine")(_conforming("dup-test-engine"))
    finally:
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

    These four are what a ``JobEngine`` cannot supply for itself. Raised as
    ``NotImplementedError`` on call, a subclass that forgets one still satisfies
    ``isinstance``, still registers, and still prepares and submits every job of a step — the
    failure surfacing in ``parse``, once the cluster time is spent. Abstract, it surfaces in
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


# ---------------------------------------------------------------------------
# The registration gate — the decorator refuses what it cannot use
# ---------------------------------------------------------------------------


def test_the_derived_contract_members_match_the_interpreters_own():
    """``contract_members`` stands in for ``__protocol_attrs__``, which is 3.12+.

    ``requires-python`` is ``>=3.11``, so the members are derived from the Protocol's methods
    and annotations instead. Where the interpreter offers the real thing, the two must agree —
    otherwise the gate enforces a contract subtly different from the declared one.
    """
    truth = getattr(CalculationEngine, "__protocol_attrs__", None)
    if truth is None:  # pragma: no cover - only on 3.11, which CI also runs
        pytest.skip("__protocol_attrs__ is a 3.12 addition")
    assert contract_members(CalculationEngine) == frozenset(truth)


def test_register_refuses_a_class_that_does_not_satisfy_the_contract():
    """``ENGINES`` is typed ``type[CalculationEngine]``; nothing used to check that.

    ``register`` was ``Callable[[type], type]`` and ``type`` is ``type[Any]``, so a class with
    only a ``name`` registered, `get_engine` handed it to the pipeline as a
    ``CalculationEngine``, and mypy said nothing at the decorator site.
    """
    with pytest.raises(TypeError, match=r"does not satisfy CalculationEngine — missing"):
        register("gate-probe")(type("Bare", (), {"name": "gate-probe"}))
    assert "gate-probe" not in ENGINES


def test_register_refuses_a_class_that_inherits_the_protocol():
    """Inheriting a ``runtime_checkable`` Protocol manufactures conformance.

    Every method arrives as an ellipsis body returning ``None``, so ``isinstance`` passes and
    ``prepare`` returns ``None`` — defeating the structural checks that stand in for this gate,
    including the one at the top of this module.
    """

    class _Hollow(CalculationEngine):
        name = "gate-probe"

    # mypy *does* catch this statically — a Protocol subclass leaves its stubs implicitly
    # abstract, so it refuses to construct one. The runtime does not: `__abstractmethods__`
    # is empty, the class instantiates, and every method returns None. The ignores below are
    # the gap between the two checkers, which is exactly what this gate closes.
    assert isinstance(_Hollow(), CalculationEngine), "the hazard this refuses"  # type: ignore[abstract]
    assert _Hollow().prepare(None) is None  # type: ignore[abstract]
    with pytest.raises(TypeError, match="inherits CalculationEngine"):
        register("gate-probe")(_Hollow)
    assert "gate-probe" not in ENGINES


def test_register_refuses_an_engine_that_leaves_a_declaration_unset():
    """The ClassVars no ``ABCMeta`` machinery watches — ``abstractmethod`` covers methods only.

    Unset, ``output_suffix`` surfaced as a bare ``AttributeError`` from inside ``prepare``,
    after ``run_step`` had built a context, derived a cache key and made a directory.
    """

    class _NoSuffix(JobEngine):
        name = "gate-probe"
        label = "Gate"
        template_suffix = "inp"
        output_globs = ()

        def build_input(self, **kwargs: object) -> None: ...

        def parse_one(self, output_path, structure_id, ctx):
            return []

        def run_block(self, ctx, inp_path, out_path):
            return RunBlock(body="true")

        def pal(self, ctx) -> int:
            return 1

    with pytest.raises(TypeError, match=r"missing declaration\(s\) \['output_suffix'\]"):
        register("gate-probe")(_NoSuffix)
    assert "gate-probe" not in ENGINES


def test_every_shipped_engine_declares_what_its_base_requires():
    """The gate's positive side, over the real registry.

    ``register`` proves this at import for every bundled engine, so this asserts the
    requirement itself is non-empty and reachable — a `required_declarations` that silently
    became `()` would make the gate vacuous without failing anything.
    """
    job_engines = [n for n, c in ENGINES.items() if issubclass(c, JobEngine)]
    assert job_engines, "no JobEngine-based engines registered — has discovery broken?"
    for name in job_engines:
        cls = ENGINES[name]
        required = cls.required_declarations
        assert required, f"{name}: required_declarations is empty, so the gate checks nothing"
        assert [d for d in required if not hasattr(cls, d)] == []
