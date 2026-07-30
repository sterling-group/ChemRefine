"""Shared pytest fixtures for the ChemRefine test suite."""

import importlib

import pytest

# The in-memory "fake" engine is test scaffolding, not a shipped plugin — it lives here in
# tests/ and registers itself (via its `@register("fake")`) once for the whole suite.
importlib.import_module("fake_engine")


@pytest.fixture(autouse=True, scope="session")
def _isolate_chemrefine_home(tmp_path_factory: pytest.TempPathFactory):
    """Point ``$CHEMREFINE_HOME`` at a scratch dir for the whole run.

    Without this the suite reads the *developer's* managed backend envs.
    :func:`chemrefine.engines._provision.chemrefine_home` falls back to
    ``<sys.prefix>/share/chemrefine``, so any test that reaches ``launcher_for`` resolves a
    real provisioned interpreter — and tests that assert on ``sys.executable`` then fail on
    a machine where the backend they name happens to be installed. Which of the five
    managed envs a developer has decided the outcome, and CI could never see it: runners
    are fresh and provision nothing, which is precisely the configuration the feature does
    not exist for.

    Session-scoped and autouse because it is an environment property, not a per-test one;
    the tests in ``test_provision.py`` that care about a *specific* home still set their own
    over the top.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CHEMREFINE_HOME", str(tmp_path_factory.mktemp("chemrefine-home")))
        yield


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
