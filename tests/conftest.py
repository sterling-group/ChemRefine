"""Shared pytest fixtures for the ChemRefine test suite."""

import builtins
import importlib
import os
import stat
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import synthetic

# The in-memory "fake" engine is test scaffolding, not a shipped plugin — it lives here in
# tests/ and registers itself (via its `@register("fake")`) once for the whole suite.
importlib.import_module("fake_engine")


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
def _isolate_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset every variable ChemRefine reads to configure the chat, so no test meets the
    developer's endpoint.

    The docs tell users to export ``CHEMREFINE_LLM_*``, and a test that resolves a provider
    from the environment then resolved the developer's: a real connection attempt from a
    test that meant to script its model, and a verdict that depended on whose shell ran it.
    The names are the product's own list (:data:`chemrefine.agent.providers.LLM_ENV`), so a
    variable the chat starts reading is cleared by being added there. No tier is exempt:
    every agent test scripts its model. A test that wants one sets it over the top.
    """
    from chemrefine.agent.providers import LLM_ENV

    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sha1_fingerprints_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host whose crypto policy allows SHA-1 only as a fingerprint.

    ``hashlib.new`` is refused outright — handed a name, :func:`hashlib.file_digest` builds
    its hash there with ``usedforsecurity`` at its default — and a SHA-1 not asked for as a
    fingerprint (``usedforsecurity=False``) is refused as such a host refuses it.
    """
    import hashlib

    sha1 = hashlib.sha1

    def refuse(name: str, *args: object, **kwargs: object) -> Any:
        raise AssertionError(f"hashlib.new({name!r}) is the path a crypto policy can refuse")

    def fingerprints_only(*args: Any, usedforsecurity: bool = True, **kwargs: Any) -> Any:
        if usedforsecurity:
            raise ValueError("[digital envelope routines] unsupported")  # the host's answer
        return sha1(*args, usedforsecurity=False, **kwargs)

    monkeypatch.setattr(hashlib, "new", refuse)
    monkeypatch.setattr(hashlib, "sha1", fingerprints_only)


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


@pytest.fixture(scope="session")
def shipped_engines() -> dict[str, type]:
    """The registered engines ChemRefine ships: each whose class lives in a plugin package.

    A plugin is a bare-named package under ``chemrefine/engines/``, discovered as the
    registry discovers them; a test double the suite registers (``fake``, a scratch plugin)
    lives elsewhere and is not one. Selected by where the class lives rather than by a list
    of names, so a new plugin is covered by existing and no test double passes for one.
    """
    import chemrefine.engines as engines_pkg
    from chemrefine.engines.api import ENGINES

    root = Path(engines_pkg.__file__).parent
    plugins = {
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and (p / "__init__.py").is_file()
    }
    return {
        name: cls
        for name, cls in ENGINES.items()
        if cls.__module__.split(".")[:2] == ["chemrefine", "engines"]
        and cls.__module__.split(".")[2:3] != []
        and cls.__module__.split(".")[2] in plugins
    }


@pytest.fixture(autouse=True, scope="session")
def _keep_model_requests_offline() -> Iterator[None]:
    """Hold PydanticAI's ``ALLOW_MODEL_REQUESTS`` at ``False`` for the whole session.

    The suite drives real agents against ``TestModel`` / ``FunctionModel`` only, and the
    flag is the library's own fence against a test that reaches a real endpoint. It is a
    process-global, so it is owned here with this file's other isolations rather than
    assigned at import of one test module — a fence that stood only when collection
    happened to include that file. Without the ``agent`` extra there is nothing to fence.
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

        without_extra("flask", "waitress", purge=("chemrefine.gui",))

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

    *A purged package goes with every module under it.* A submodule left cached is never
    imported again, so an eager SDK import added to it — or to the package's ``__init__`` —
    would not run under the block, and the guard it breaks would look alive.
    """
    real_import = builtins.__import__

    def block(*sdks: str, purge: tuple[str, ...] = ()) -> None:
        cached = [
            name
            for name in sys.modules
            if any(name == dotted or name.startswith(f"{dotted}.") for dotted in purge)
        ]
        for dotted in sorted(cached):
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


@pytest.fixture
def refusal_reader() -> Callable[[str, dict[str, Any]], Any]:
    """A scripted model that calls one tool once, then replies with the failure it was handed.

    What both embedded harnesses must show: a tool's refusal reaches the model as that
    tool's failed result, in :func:`~chemrefine.errors.refusal_text`'s words, and the turn
    goes on from it. One fixture, because the terminal and the web harness tests ask the
    same question of the same channel. PydanticAI is imported inside, so the suite still
    collects without the agent extra.
    """
    from pydantic_ai.messages import (
        ModelMessage,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    def build(tool: str, args: dict[str, Any]) -> FunctionModel:
        def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            failed = [
                part
                for part in messages[-1].parts
                if isinstance(part, ToolReturnPart) and part.outcome == "failed"
            ]
            if not failed:
                return ModelResponse(parts=[ToolCallPart(tool, args)])
            return ModelResponse(parts=[TextPart(str(failed[0].content))])

        return FunctionModel(script)

    return build


@pytest.fixture
def crashing_tool(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Swap one shared tool for a same-named one that crashes — a failure, not a refusal.

    Under its own name the swap keeps the tool's gate: a mutating one is still confirmed in
    the terminal and still waits for its card in the builder. Both harnesses read
    :data:`~chemrefine.agent_tools.TOOLS` as they build, so the swap reaches either.
    """
    from chemrefine import agent_tools

    def swap(name: str) -> None:
        def crash(config_path: str) -> dict[str, Any]:
            raise RuntimeError(f"{name} crashed")

        crash.__name__ = name
        tools = tuple(crash if tool.__name__ == name else tool for tool in agent_tools.TOOLS)
        monkeypatch.setattr(agent_tools, "TOOLS", tools)

    return swap


@pytest.fixture
def slow_config_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every atomic write takes 0.2 s — long enough to still be running beside a crash.

    What makes "a save and a crashing call in one response" deterministic: run side by
    side, the crash lands while the save is still writing.
    """
    import time

    from chemrefine import io

    write = io.atomic_write

    def slowly(*args: Any, **kwargs: Any) -> None:
        time.sleep(0.2)
        write(*args, **kwargs)

    monkeypatch.setattr(io, "atomic_write", slowly)


@pytest.fixture
def unsearchable() -> Iterator[Callable[[Path], Path]]:
    """Take this account's search permission off a directory; given back when the test ends.

    For a test of a path a person named that sits under a directory this account cannot
    search — a colleague's project folder, a scratch tree with another owner. The test
    builds the directory, fills it, and hands it here; what it then asks about any path
    inside must be the same on every supported Python, where ``Path.exists`` and
    ``Path.is_file`` raise on 3.11 to 3.13 and answer ``False`` on 3.14. Skipped for root,
    which searches through any mode.
    """
    if os.geteuid() == 0:
        pytest.skip("root searches through any mode; the wall cannot be built")
    locked: list[Path] = []

    def lock(directory: Path) -> Path:
        locked.append(directory)
        directory.chmod(0)
        return directory

    yield lock
    for directory in locked:
        directory.chmod(0o755)


_HOLD_SECONDS = 5.0
"""How long a mode gets to take hold before the test that set it fails."""


def _answers(ask: Callable[[Any], object], path: Any) -> bool:
    """Whether the filesystem still answers ``ask`` of ``path`` rather than refusing it."""
    try:
        ask(path)
    except OSError:
        return False  # refused, or gone: either way nothing is answered for it
    return True


def _first_entry(directory: Any) -> None:
    """Open ``directory``'s listing and read from it, which is where a refusal shows."""
    with os.scandir(directory) as scan:
        next(scan, None)


def _directory_mode(path: Any, how: dict[str, Any]) -> int | None:
    """The mode of the directory a ``chmod(path, …, **how)`` is about to change, if it is one.

    Asked as ``chmod`` itself resolves ``path``: a link is followed unless ``how`` says
    not to. ``None`` for what the wait has nothing to ask of — a file, an open file's
    number, a name relative to a directory's, a path out of reach.
    """
    if not isinstance(path, str | os.PathLike) or how.get("dir_fd") is not None:
        return None
    try:
        status = Path(os.fsdecode(path)).stat(follow_symlinks=how.get("follow_symlinks", True))
    except OSError:
        return None
    return stat.S_IMODE(status.st_mode) if stat.S_ISDIR(status.st_mode) else None


@pytest.fixture(autouse=True)
def _modes_take_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mode that walls a directory off is in force by the time ``chmod`` returns.

    A test that takes search or read permission off a directory and then asks for something
    inside assumes the filesystem answers by the new mode at once. A local disk does. A
    network filesystem may go on answering from what it remembers: the group share this was
    met on resolves the directories inside a walled-off one for about a second more, so the
    refusal such a test is about came or not by timing. The wait is made here, once for
    every test: after a ``chmod`` that takes search permission off a directory this returns
    when every entry it held is refused, and after one that takes read permission off, when
    its listing is. A wall that has not come within ``_HOLD_SECONDS`` is taken down again
    and fails the test, as it would have failed it unwaited-for.
    """
    if os.geteuid() == 0:
        return  # nothing is refused to root, and each test of a wall skips itself there
    chmod = os.chmod

    def taking_hold(path: Any, mode: int, **how: Any) -> None:
        searched, listed = bool(mode & 0o100), bool(mode & 0o400)
        was = None if searched and listed else _directory_mode(path, how)
        held: list[Path] = []
        if was is not None and not searched:
            try:
                held = list(Path(os.fsdecode(path)).iterdir())
            except OSError:
                held = []  # walled off already: it holds nothing known to be answered for
        chmod(path, mode, **how)
        if was is None:
            return
        deadline = time.monotonic() + _HOLD_SECONDS
        while any(_answers(os.lstat, entry) for entry in held) or (
            not listed and _answers(_first_entry, path)
        ):
            if time.monotonic() > deadline:
                chmod(path, was, **how)
                pytest.fail(
                    f"this filesystem still answers behind mode {mode:03o} on {path} "
                    f"{_HOLD_SECONDS:g} s after it was set"
                )
            time.sleep(0.05)

    monkeypatch.setattr(os, "chmod", taking_hold)


@pytest.fixture
def list_only() -> Iterator[Callable[[Path], Path]]:
    """Leave a directory listable and not searchable; given back when the test ends.

    Mode ``r--``: another account's directory that lets this one read its names. A local
    disk lists it and says of each entry whether it is a directory, with nothing searched
    for, and that listing is what the test asks about. A filesystem need not have one: a
    network share may refuse the listing outright, or name the entries without their kind
    and leave the kind to a search it refuses. There the directory is one that cannot be
    listed, which the tests of those hold, and this one is skipped.
    """
    left: list[Path] = []

    def leave(directory: Path) -> Path:
        left.append(directory)
        directory.chmod(0o444)
        try:
            with os.scandir(directory) as scan:
                for entry in scan:
                    entry.is_dir()
        except PermissionError:
            pytest.skip("this filesystem lists no directory it may not search")
        return directory

    yield leave
    for directory in left:
        directory.chmod(0o755)


_LONG_PAST_NS = 1_000_000_000 * 10**9
"""A modification time no write leaves on a file: September 2001, in whole seconds, so a
filesystem that keeps only those gives the same number back."""


def _identity(path: Path) -> tuple[int, int]:
    """What a file is the same file by: its inode and its modification time."""
    status = path.stat()
    return status.st_ino, status.st_mtime_ns


@pytest.fixture
def back_dated() -> Callable[[Path], Callable[[], bool]]:
    """Back-date a file and return the question of whether it has been written since.

    A document saved atomically comes back under a new inode. A file rewritten in place
    keeps its inode and takes the time of the write, and that time, read off the
    filesystem's clock before and after, proves nothing: a filesystem that keeps whole
    seconds gives a file rewritten a moment later the time it had. Set first to a time
    long past, it is told from any write, and only a write that never happened keeps both.
    """

    def back_date(path: Path) -> Callable[[], bool]:
        os.utime(path, ns=(_LONG_PAST_NS, _LONG_PAST_NS))
        before = _identity(path)
        return lambda: _identity(path) != before

    return back_date


@pytest.fixture
def text_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every path decoded through :meth:`pathlib.Path.read_text` while the test runs, in order.

    For a test that holds a read count: the one text reader
    (:func:`chemrefine.io.read_utf8`) and a direct ``read_text`` both land here, so a second
    decode cannot hide behind a route the test did not patch.
    """
    reads: list[Path] = []
    read_text = Path.read_text

    def recording(path: Path, *args: Any, **kwargs: Any) -> str:
        reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording)
    return reads


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
