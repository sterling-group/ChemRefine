"""Shared pytest fixtures for the ChemRefine test suite."""

import importlib

import pytest

# The in-memory "fake" engine is test scaffolding, not a shipped plugin — it lives here in
# tests/ and registers itself (via its `@register("fake")`) once for the whole suite.
importlib.import_module("fake_engine")


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
        help="re-pack the tests/data/e2e archives from passing live (-m integration) runs",
    )
