"""Shared pytest fixtures for the ChemRefine test suite."""

import builtins
import importlib
import sys
from collections.abc import Callable
from pathlib import Path

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
