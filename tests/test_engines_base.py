"""Tests for the engine Protocol, the ENGINES registry, and registration."""

from __future__ import annotations

import re
from typing import cast

import pytest

# Importing from ``chemrefine.engines.api`` triggers the parent package's
# ``__init__``, which self-registers every bundled engine into ``ENGINES``.
from chemrefine.config import StepConfig
from chemrefine.engines._job import JobEngine
from chemrefine.engines.api import (
    ENGINES,
    CalculationEngine,
    contract_members,
    get_engine,
    register,
    require_whitespace_free,
)
from chemrefine.errors import ConfigError, EngineNotFoundError
from chemrefine.state import RunBlock


def test_engines_registry_is_populated_by_side_effect_import():
    """Importing chemrefine.engines must trigger every bundled engine's registration.

    Asserted on a *bundled* engine: ``fake`` is test scaffolding conftest registers
    itself, so it stays in the registry with the package's ``_load_plugins()`` deleted —
    which is exactly the side effect this test is named for. (The full bundled roster is
    ``test_engines_discovery``'s; one shipped name is enough to prove the mechanism.)
    """
    assert "orca" in ENGINES
    assert "fake" in ENGINES  # and conftest's scaffolding engine is present for the suite


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
        # `is`-compared through `object`: ENGINES is typed `type[CalculationEngine]`,
        # and mypy calls the identity check non-overlapping against a concrete class.
        assert ENGINES["temp-test-engine"] is cast("object", _TempEngine)
    finally:
        ENGINES.pop("temp-test-engine", None)


def test_preflight_steps_asks_only_the_engines_that_declare_the_capability():
    """``preflight_steps`` is the run's t=0 walk: opt-in, with the effective species.

    Opt-in via ``PreflightChecking`` because a generic strict pass would refuse valid
    configs — the direct script engines read leniently by documented design. The hook
    receives the *effective* charge and multiplicity (config defaults with the step's
    own override applied), the same resolution ``build_context`` performs for the run
    itself — resolved in the walk so no hook re-spells the fallback.
    """
    from chemrefine.engines.api import preflight_steps

    calls: list[tuple[int, int, int]] = []

    class _Checked:
        name = "preflight-probe"
        preflight_refuses = "a probe step; the sentence the registration gate asks of any hook"

        def prepare(self, ctx: object) -> None: ...

        def submit(self, inputs: object, ctx: object) -> None: ...

        def parse(self, inputs: object, ctx: object) -> None: ...

        def check_step(self, step_cfg: StepConfig, *, charge: int, multiplicity: int) -> None:
            calls.append((step_cfg.step, charge, multiplicity))

    try:
        register("preflight-probe")(_Checked)
        steps = [
            StepConfig(step=1, engine="fake"),  # no capability — not asked
            StepConfig(step=2, engine="preflight-probe"),  # config defaults apply
            StepConfig(step=3, engine="preflight-probe", charge=-1, multiplicity=2),
        ]
        preflight_steps(steps, charge=0, multiplicity=1)
    finally:
        ENGINES.pop("preflight-probe", None)

    assert calls == [(2, 0, 1), (3, -1, 2)]


def test_a_declared_vocabulary_is_held_at_the_preflight_walk_with_no_hook():
    """Declaring ``operations`` is the whole rule: the walk refuses anything outside it.

    No ``check_step`` of the engine's own, so no hook to forget and no second copy of the
    refusal; an omitted operation is the engine's to infer, and an engine that declares no
    vocabulary reads the field as a label.
    """
    from chemrefine.engines.api import preflight_steps

    class _Declaring:
        name = "vocabulary-probe"
        operations = ("scan", "relax")

        def prepare(self, ctx: object) -> None: ...

        def submit(self, inputs: object, ctx: object) -> None: ...

        def parse(self, inputs: object, ctx: object) -> None: ...

    try:
        register("vocabulary-probe")(_Declaring)
        preflight_steps(
            [
                StepConfig(step=1, engine="vocabulary-probe", operation="scan"),
                StepConfig(step=2, engine="vocabulary-probe"),
                StepConfig(step=3, engine="fake", operation="Relax"),
            ],
            charge=0,
            multiplicity=1,
        )
        with pytest.raises(ConfigError) as excinfo:
            preflight_steps(
                [StepConfig(step=4, engine="vocabulary-probe", operation="md")],
                charge=0,
                multiplicity=1,
            )
    finally:
        ENGINES.pop("vocabulary-probe", None)
    assert str(excinfo.value) == (
        "step 4: unknown operation 'md' for engine 'vocabulary-probe', which reads relax, "
        "scan. Correct it, or omit `operation:`."
    )


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


def test_register_refuses_a_name_no_config_can_reach():
    """``engine:`` is folded to lower case as a step is read, so a capitalised registry key
    could never be looked up — refused at its decorator, naming the spelling to use."""
    try:
        with pytest.raises(ValueError, match="register it as 'gate-probe'"):
            register("Gate-Probe")(_conforming("Gate-Probe"))
        assert "Gate-Probe" not in ENGINES
    finally:
        ENGINES.pop("Gate-Probe", None)


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
    """v1.3.1's spellings (``mlff``, ``dft``) are mapped by the config normalizer, not
    the registry — so the registry exposes only canonical names."""
    assert {"orca", "mlip", "mlip-extopt", "mlip-train"} <= set(ENGINES)
    for legacy in ("mlff", "dft"):
        assert legacy not in ENGINES


# --- the whitespace rule, as the engines that declare it ask it -------------


@pytest.mark.parametrize("whitespace", [" ", "\t", "\v"])
def test_the_whitespace_rule_refuses_any_whitespace_not_just_the_space(tmp_path, whitespace: str):
    """A field split on whitespace splits on a tab too, so the rule does.

    A tab or vertical tab in a directory name is absurd and perfectly legal on POSIX, and an
    input that reads a path through a whitespace-delimited field truncates on each of them.
    """
    with pytest.raises(ConfigError, match="contains whitespace"):
        require_whitespace_free(tmp_path / f"my{whitespace}outputs", what="it", reason="why")


def test_the_whitespace_refusal_quotes_its_engines_reason_and_names_no_engine(tmp_path):
    """Actionable and engine-neutral: the engine's own reason, what to change, and who need not.

    Steps whose engine can take the path are named by that, not by engine — so no roster has
    to be kept true here, and an engine that shares the capability is never called unaffected.
    """
    path = tmp_path / "my outputs"
    reason = "this input reads a path through a whitespace-delimited field"
    with pytest.raises(ConfigError) as refused:
        require_whitespace_free(path, what="step 2's directory", reason=reason)
    message = str(refused.value)
    assert message.startswith(f"step 2's directory contains whitespace: {str(path)!r}. {reason}")
    assert "`output_dir`" in message
    assert message.endswith(
        "Steps whose engine can take such a path are unaffected and need no change."
    )
    named = [
        n for n in ENGINES if re.search(rf"\b{re.escape(n)}\b", message.replace(str(path), ""))
    ]
    assert not named, f"the core refusal names an engine: {named}"


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

        def build_input(self, *, xyz_path, template, input_path, output_path, ctx): ...

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
    if not hasattr(CalculationEngine, "__protocol_attrs__"):  # pragma: no cover - 3.11 only
        pytest.skip("__protocol_attrs__ is a 3.12 addition")
    from chemrefine.engines import api

    protocols = [
        value
        for value in vars(api).values()
        if isinstance(value, type)
        and getattr(value, "_is_protocol", False)
        and value.__module__ == api.__name__
    ]
    assert CalculationEngine in protocols and api.OperationInferring in protocols
    for protocol in protocols:
        truth = frozenset(getattr(protocol, "__protocol_attrs__", ()))
        assert contract_members(protocol) == truth, protocol.__name__


def test_register_refuses_a_class_that_does_not_satisfy_the_contract():
    """``ENGINES`` is typed ``type[CalculationEngine]``, and this is what makes that true.

    Without the gate a class carrying only a ``name`` registers, and :func:`get_engine` hands
    it to the pipeline as a ``CalculationEngine``.
    """
    with pytest.raises(TypeError, match=r"does not satisfy CalculationEngine — missing"):
        register("gate-probe")(type("Bare", (), {"name": "gate-probe"}))
    assert "gate-probe" not in ENGINES


def test_register_refuses_a_preflight_hook_without_its_claim():
    """``check_step`` without ``preflight_refuses`` would silently leave the preflight walk.

    A Protocol member left unset makes ``isinstance(engine, PreflightChecking)`` answer no,
    so the omission would not fail — the hook would simply never be called. Refused at the
    decorator line instead, naming the sentence the engine owes.
    """
    hooked = type(
        "Hooked",
        (),
        {
            "name": "gate-probe",
            "prepare": lambda self, ctx: None,
            "submit": lambda self, inputs, ctx: None,
            "parse": lambda self, inputs, ctx: None,
            "check_step": lambda self, step_cfg, *, charge, multiplicity: None,
        },
    )
    refusal = "declares ['check_step'] of PreflightChecking but not ['preflight_refuses']"
    with pytest.raises(TypeError, match=re.escape(refusal)):
        register("gate-probe")(hooked)
    assert "gate-probe" not in ENGINES


_RUNNABLE = {
    "prepare": lambda self, ctx: None,
    "submit": lambda self, inputs, ctx: None,
    "parse": lambda self, inputs, ctx: None,
}


def test_register_refuses_a_template_engine_without_its_starter():
    """Without ``template_starter`` the class would silently not be ``TemplateDriven``.

    ``isinstance`` answers no to a class that declares only some of a Protocol's members,
    so the engine would read no template — every consumer of the capability would pass it
    by — while its author believed it declared one.
    """
    partial = type(
        "Partial",
        (),
        {"name": "gate-probe", "label": "Gate", "template_suffix": "toml", **_RUNNABLE},
    )
    refusal = "declares ['label', 'template_suffix'] of TemplateDriven but not ['template_starter']"
    with pytest.raises(TypeError, match=re.escape(refusal)):
        register("gate-probe")(partial)
    assert "gate-probe" not in ENGINES


@pytest.mark.parametrize(
    ("member", "refusal"),
    [
        (
            "template_starter",
            "declares ['template_starter'] of TemplateDriven but not ['label', 'template_suffix']",
        ),
        (
            "preflight_refuses",
            "declares ['preflight_refuses'] of PreflightChecking but not ['check_step']",
        ),
        (
            "check_template",
            "declares ['check_template'] of TemplateChecking but not ['preflight_refuses']",
        ),
        (
            "infer_operation",
            "declares ['infer_operation'] of OperationInferring but not ['reads_alike']",
        ),
        (
            "nms_input_info",
            "declares ['nms_input_info'] of NmsCapableEngine but not ['frequency_request']",
        ),
    ],
)
def test_register_refuses_a_claim_without_its_capability(member: str, refusal: str):
    """A member alone claims a capability the class does not have."""
    claimed = type("Claimed", (), {"name": "gate-probe", member: "x", **_RUNNABLE})
    with pytest.raises(TypeError, match=re.escape(refusal)):
        register("gate-probe")(claimed)
    assert "gate-probe" not in ENGINES


def test_the_capabilities_register_holds_whole_have_these_members():
    """Literal pins of what each capability is taken by: a member added to one is a member
    every engine taking it must now declare, so it lands here as a decision. Each is the
    capability's own — what it adds to the Protocols it extends."""
    from chemrefine.engines.api import _CAPABILITIES

    assert {capability.__name__: members for capability, members in _CAPABILITIES} == {
        "TemplateDriven": {"label", "template_starter", "template_suffix"},
        "OperationInferring": {"infer_operation", "reads_alike"},
        "PreflightChecking": {"check_step", "preflight_refuses"},
        "TemplateChecking": {"check_template", "preflight_refuses"},
        "NmsCapableEngine": {"frequency_request", "nms_input_info"},
    }


def test_register_refuses_an_inference_hook_that_cannot_say_what_reads_alike():
    """Without ``reads_alike`` the class would silently not be ``OperationInferring``: its
    ``operation:`` keyed and read as written, its template never asked."""
    partial = type(
        "Partial",
        (),
        {
            "name": "gate-probe",
            "label": "Gate",
            "template_suffix": "toml",
            "template_starter": "",
            "infer_operation": lambda self, template: "sp",
            **_RUNNABLE,
        },
    )
    refusal = "declares ['infer_operation'] of OperationInferring but not ['reads_alike']"
    with pytest.raises(TypeError, match=re.escape(refusal)):
        register("gate-probe")(partial)
    assert "gate-probe" not in ENGINES


@pytest.mark.parametrize(
    ("members", "refusal"),
    [
        (
            {
                "infer_operation": lambda self, template: "sp",
                "reads_alike": lambda self, first, second: first == second,
            },
            "declares OperationInferring (['infer_operation', 'reads_alike']) but not "
            "['label', 'template_starter', 'template_suffix'], which it extends",
        ),
        (
            {
                "nms_input_info": lambda self, ctx: None,
                "frequency_request": "a frequency job",
            },
            "declares NmsCapableEngine (['frequency_request', 'nms_input_info']) but not "
            "['artifact_paths'], which it extends",
        ),
    ],
    ids=["inference-without-a-template", "sampling-without-artifact-paths"],
)
def test_register_refuses_a_capability_without_what_it_extends(
    members: dict[str, object], refusal: str
):
    """``isinstance`` asks a Protocol's inherited members too, so a capability whose own
    members are all declared would still go undetected without them."""
    hollow = type("Hollow", (), {"name": "gate-probe", **members, **_RUNNABLE})
    with pytest.raises(TypeError, match=re.escape(refusal)):
        register("gate-probe")(hollow)
    assert "gate-probe" not in ENGINES


def test_a_template_rule_alone_is_a_whole_capability():
    """The sentence two capabilities share is held by either one declared whole.

    An engine whose only refusal before the run is about its template takes
    ``check_template`` and the sentence, and owes no ``check_step``.
    """
    from chemrefine.engines.api import PreflightChecking, TemplateChecking

    ruled = type(
        "Ruled",
        (),
        {
            "name": "gate-probe",
            "preflight_refuses": "a template that names nothing",
            "check_template": lambda self, ctx, template: None,
            **_RUNNABLE,
        },
    )
    register("gate-probe")(ruled)
    engine = get_engine("gate-probe")
    assert isinstance(engine, TemplateChecking)
    assert not isinstance(engine, PreflightChecking)


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
    # mypy refuses this call outright now that `register` is typed over the contract, which
    # is the static half of the same gate; the ignore is what lets the runtime half be
    # exercised from a test.
    with pytest.raises(TypeError, match="inherits CalculationEngine"):
        register("gate-probe")(_Hollow)  # type: ignore[type-abstract]
    assert "gate-probe" not in ENGINES


def test_register_refuses_an_engine_that_leaves_a_declaration_unset():
    """The ClassVars no ``ABCMeta`` machinery watches — ``abstractmethod`` covers methods only.

    Unrefused here, an unset ``output_suffix`` surfaces as a bare ``AttributeError`` from
    inside ``prepare`` — after ``run_step`` has built a context, derived a cache key and made
    a directory.
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
