"""Shared pytest fixtures for the ChemRefine test suite."""

import asyncio
import builtins
import contextlib
import importlib
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import synthetic

# The in-memory "fake" engine is test scaffolding, not a shipped plugin — it lives here in
# tests/ and registers itself (via its `@register("fake")`) once for the whole suite.
importlib.import_module("fake_engine")


@pytest.fixture
def qiskit_spawn_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise serialization without forking a parent already running JAX threads."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    parallel = pytest.importorskip("qiskit.utils.parallel")
    monkeypatch.setattr(
        parallel,
        "ProcessPoolExecutor",
        partial(ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")),
    )


@pytest.fixture(scope="session")
def _scratch_chemrefine_home(tmp_path_factory: pytest.TempPathFactory) -> str:
    """One empty managed-backend root for the whole run, applied per test below."""
    return str(tmp_path_factory.mktemp("chemrefine-home"))


@pytest.fixture(autouse=True)
def _isolate_chemrefine_home(
    request: pytest.FixtureRequest, _scratch_chemrefine_home: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point ``$CHEMREFINE_HOME`` at a scratch dir — except for the tier-3 live tests.

    Without this the suite reads the *developer's* managed backend envs.
    :func:`chemrefine.engines._provision.chemrefine_home` falls back to
    ``<sys.prefix>/share/chemrefine``, so any test that reaches ``launcher_for`` resolves a
    real provisioned interpreter — and tests that assert on ``sys.executable`` then fail on
    a machine where the backend they name happens to be installed. Which of the five
    managed envs a developer has decided the outcome, and CI could never see it: runners
    are fresh and provision nothing, which is precisely the configuration the feature does
    not exist for.

    **The ``integration`` tier is exempt, and that exemption is the point of the tier.** Its
    cases exist to run against the real stacks, which live in exactly the managed envs this
    fixture hides; applied to them it guarantees the opposite of what they assert, since
    ``preflight_backends`` then finds an empty root and refuses to start. Marker-scoped
    rather than session-scoped for that reason — "which home" is a property of the tier, not
    of the run. The tests in ``test_provision.py`` that care about a *specific* home still
    set their own over the top.
    """
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setenv("CHEMREFINE_HOME", _scratch_chemrefine_home)


@pytest.fixture(autouse=True)
def _isolate_cuda_visible_devices(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset ``CUDA_VISIBLE_DEVICES`` — except for the tiers that want the real allocation.

    The local GPU budget is *derived* from this variable
    (:func:`chemrefine.slurm.resolve_gpu_budget`), so a developer running the suite inside
    an allocation would get a different budget from CI's bare runner, and every test that
    asserts on devices ``0``/``1`` would pass or fail by where it was run. That is the same
    class of environment leak as :func:`_isolate_chemrefine_home` above, and it is exempt
    for the same tiers and the same reason: ``integration`` and ``gpu`` exist to meet the
    real thing. Tests that care about a specific allocation set it themselves over the top.
    """
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("gpu"):
        return
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.fixture(autouse=True)
def _isolate_gpu_probe(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the ``nvidia-smi`` probe with one fixed device — the isolation's other half.

    Deleting ``CUDA_VISIBLE_DEVICES`` above is what *forces* the probe:
    :func:`~chemrefine.slurm.dispatch.resolve_gpu_budget` reads the variable first and
    falls through to :func:`~chemrefine.slurm.dispatch._detected_devices` exactly when it
    is absent — so the fixture meant to stop the budget varying by machine sent every
    test that reaches a local batch to fork ``nvidia-smi`` and memoize the *developer's*
    hardware for the session. One fixed device is CI's bare-runner answer everywhere.

    Exempt alongside its sibling for the tiers that exist to meet the real thing, and for
    any test that takes ``uncached_gpu_probe`` — the fixture whose whole purpose is to
    vary what the real probe sees. Tests that fake a specific device set still patch
    ``_detected_devices`` over the top, as they always have.
    """
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("gpu"):
        return
    if "uncached_gpu_probe" in request.fixturenames:
        return
    from chemrefine.slurm import dispatch

    monkeypatch.setattr(dispatch, "_detected_devices", lambda: ("0",))


@pytest.fixture(autouse=True)
def _restore_engine_registries() -> Iterator[None]:
    """Snapshot the engine (and MLIP backend) registries; restore them after every test.

    Both are process-global dicts a test may extend with a scratch plugin, and the suite
    holds *exact* membership invariants over them from another file
    (``test_engines_invariants``) — so one missed hand-written ``finally`` cascades into
    failures pointing at the wrong module, and re-registering a name raises from
    ``register`` itself. The hand-written cleanups in the individual tests stay (each is
    local documentation of what that test touches); this is the net under all of them,
    so a missed one can no longer reach a different file.
    """
    from chemrefine.engines.api import ENGINES
    from chemrefine.engines.mlip import registry

    engines_before = dict(ENGINES)
    backends_before = dict(registry._BACKENDS)
    yield
    ENGINES.clear()
    ENGINES.update(engines_before)
    registry._BACKENDS.clear()
    registry._BACKENDS.update(backends_before)


@pytest.fixture(autouse=True, scope="session")
def _keep_model_requests_offline() -> Iterator[None]:
    """Hold PydanticAI's ``ALLOW_MODEL_REQUESTS`` at ``False`` for the whole session.

    The suite drives real agents against ``TestModel`` / ``FunctionModel`` only, and the
    flag is the library's own fence against a test that reaches a real endpoint. It is a
    process-global, so it is owned here with the others (``CHEMREFINE_HOME``,
    ``CUDA_VISIBLE_DEVICES``, ``DISPLAY``, the engine registries) rather than assigned at
    import of one test module — a fence that stood only when collection happened to
    include that file. Without the ``agent`` extra there is nothing to fence.
    """
    try:
        from pydantic_ai import models
    except ImportError:
        yield
        return
    previous = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = False
    yield
    models.ALLOW_MODEL_REQUESTS = previous


@contextlib.contextmanager
def _owned_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Give the calling thread an event loop of its own for the block, and close it after.

    The one rule behind both fixtures below: a thread that runs a synchronous agent turn
    owns the loop that turn runs on. ``Agent.run_sync`` takes whatever loop the policy holds
    for the thread and, finding none, creates one and leaves it there — open, and owned by
    nobody. Owned here, it is closed on the way out and the policy is left as it was found.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@pytest.fixture(scope="session")
def _owned_event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """The one event loop the suite's synchronous agent paths run on, owned and closed here.

    ``Agent.run_sync`` — what ``chat.repl``, the GUI's chat endpoint and the harness tests
    drive — runs on whatever loop ``asyncio.get_event_loop()`` answers, and when the policy
    holds none it *creates* one and leaves it there: open, and owned by nobody. The anyio
    tests (``test_mcp_server``) run under anyio's own runner, which installs its loop on the
    policy for the test and unsets it afterwards — dropping the only reference to that
    implicit loop while it is still open. It sits in a reference cycle, so the collector
    reaches it whenever the object graph happens to trigger a pass, and its ``__del__`` then
    raises ``ResourceWarning`` (the loop and its self-pipe socket pair) into whichever test is
    running — which ``filterwarnings = error`` turns into a failure of an innocent test. The
    dev environment reached that point only at exit; a fresh dependency resolution reached
    it mid-run, and the sdist gate went red on ``test_package_boundaries``.

    Owned here, the loop is never garbage: ``_current_event_loop`` below puts it back on the
    policy before every test that finds none, so no path ever creates an implicit one, and
    the session closes it. The ``DeprecationWarning`` the stdlib raises on implicit creation
    is therefore no longer filtered in ``pyproject.toml`` — it would now mean a new path has
    started creating loops of its own.
    """
    with _owned_loop() as loop:
        yield loop


@pytest.fixture
def owned_event_loop() -> Callable[
    [], contextlib.AbstractContextManager[asyncio.AbstractEventLoop]
]:
    """:func:`_owned_loop` for a test that runs a synchronous agent turn on a thread of its own.

    The session loop above belongs to the main thread; ``run_sync`` on any other thread
    creates that thread's own implicit loop, which dies unowned with the thread — the same
    ``ResourceWarning`` by a shorter route, raised into whichever test the collector reaches
    it under. A thread that sends a turn wraps it in this, so the loop is closed with the
    thread. Waitress's workers never leave, which is why the GUI itself needs no such care.
    """
    return _owned_loop


@pytest.fixture(autouse=True)
def _current_event_loop(_owned_event_loop: asyncio.AbstractEventLoop) -> None:
    """Install the session's loop on the policy whenever a test would otherwise find none.

    Anyio's runner sets the policy's loop to ``None`` when it closes; the next synchronous
    ``run_sync`` would then create an implicit loop. Reinstalled per test rather than once,
    for that reason. A loop that is present and open is left alone — an anyio test's own
    runner installs its loop inside the test, after this fixture has run.
    """
    policy = asyncio.get_event_loop_policy()
    current = getattr(getattr(policy, "_local", None), "_loop", None)
    if current is None or current.is_closed():
        asyncio.set_event_loop(_owned_event_loop)


@pytest.fixture(autouse=True)
def _isolate_display_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin a non-headless display environment — the suite must not care where it runs.

    ``chemrefine.gui.serve`` decides between opening a browser and printing the SSH
    forwarding recipe from ``DISPLAY``/``WAYLAND_DISPLAY``, and reads ``SSH_CONNECTION``
    for the recipe's host — so a developer running pytest over ssh (DISPLAY unset) would
    take the recipe branch in tests that assert a browser opened, while CI's bare runner
    keeps them green. That is the same environment-leak class as
    :func:`_isolate_cuda_visible_devices` above. Tests that want the headless posture
    set their own values over the top.
    """
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def without_extra(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Simulate an environment that never installed one of the optional extras.

    Call it with the third-party module names to hide, plus the ChemRefine modules that
    import them::

        without_extra("flask", "waitress", purge=("chemrefine.gui.app", "chemrefine.gui.serve"))

    Both halves are load-bearing, and every hand-rolled copy of this simulation in the
    suite got at least one of them wrong — which is why it lives here now rather than
    three times over.

    *Hide the SDK, not our own module.* The three CLI guard tests used to raise on
    ``chemrefine.gui`` / ``chemrefine.agent`` / ``chemrefine.mcp_server``, none of which a
    user is ever missing. That proves only that ``except ImportError`` catches an
    ImportError; it cannot tell a live guard from a dead one, and two of the three guards
    were in fact dead.

    *Evict from `sys.modules` **and** the parent package.* This suite imports those modules
    at module scope, so ``from chemrefine.gui.serve import launch`` is otherwise a cache
    hit and the SDK import never re-runs. Deleting only the ``sys.modules`` entry is not
    enough for the ``from package import submodule`` spelling: the submodule survives as an
    attribute of the already-imported parent, so the import resolves off that instead — and
    a test that meant to assert a refusal quietly exercised the real command.
    """
    real_import = builtins.__import__

    def block(*sdks: str, purge: tuple[str, ...] = ()) -> None:
        for dotted in purge:
            monkeypatch.delitem(sys.modules, dotted, raising=False)
            parent, _, leaf = dotted.rpartition(".")
            if (owner := sys.modules.get(parent)) is not None:
                monkeypatch.delattr(owner, leaf, raising=False)

        def refuse(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".")[0] in sdks:
                raise ImportError(f"No module named {name.split('.')[0]!r}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)

    return block


@pytest.fixture
def orca_error_termination(tmp_path: Path) -> Path:
    """A captured ORCA abort on disk, with its ``.err`` sidecar beside it; returns the ``.out``.

    Written out rather than passed as text because both readers take a *path*: ``parse_dft``
    opens the output, and the termination message quotes the stderr it finds by swapping the
    suffix — so the pair has to exist as files sharing a stem for the test to prove anything
    about the message.

    In ``tmp_path``, so a test that also builds a step directory there gets both from one
    place. The bytes are in :mod:`synthetic`, marked captured.
    """
    out = tmp_path / f"{synthetic.ORCA_ERROR_TERMINATION_STEM}.out"
    out.write_text(synthetic.ORCA_ERROR_TERMINATION_OUT, encoding="utf-8")
    out.with_suffix(".err").write_text(synthetic.ORCA_ERROR_TERMINATION_ERR, encoding="utf-8")
    return out


def pytest_addoption(parser: pytest.Parser) -> None:
    """Suite-wide flags: golden regeneration and fixture re-recording."""
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="rewrite the per-engine contract goldens instead of asserting",
    )
    parser.addoption(
        "--record",
        action="store_true",
        default=False,
        help="re-pack the e2e recordings from passing live (-m integration) runs",
    )
    parser.addoption(
        "--update-recordings",
        action="store_true",
        default=False,
        help="re-pack the e2e recordings from a parse-only rebuild (no ORCA, no MLIP stack)",
    )
