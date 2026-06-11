"""Tests for ``chemrefine/__init__.py`` and ``chemrefine/__main__.py``."""

from __future__ import annotations

import importlib
import runpy
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest


def test_version_falls_back_when_uninstalled():
    """``__version__`` is set to a sentinel when the package isn't installed."""
    import chemrefine

    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        reloaded = importlib.reload(chemrefine)
        assert reloaded.__version__ == "0.0.0+unknown"

    # Restore the real metadata-driven version for any subsequent tests in
    # the same process; otherwise the sentinel sticks around.
    importlib.reload(chemrefine)


def test_version_reads_from_metadata_when_installed():
    """The happy path: ``importlib.metadata.version`` returns a real string."""
    import chemrefine

    assert isinstance(chemrefine.__version__, str)
    assert chemrefine.__version__


def test_python_dash_m_runs_the_cli():
    """``python -m chemrefine --help`` executes via ``__main__.py``.

    ``runpy.run_module`` runs the module exactly the way the python
    ``-m`` flag would, so ``__main__.py``'s import + ``main()`` call are
    real coverage hits. Typer exits with code 0 on ``--help``; we catch
    the SystemExit and assert it.
    """
    with patch("sys.argv", ["chemrefine", "--help"]), pytest.raises(SystemExit) as excinfo:
        runpy.run_module("chemrefine", run_name="__main__")
    assert excinfo.value.code == 0


def test_python_dash_m_translates_legacy_argv():
    """``python -m chemrefine CONFIG --skip`` gets the same legacy translation
    as the ``chemrefine`` console script (both route through ``cli.main``)."""
    import sys

    captured = {}

    def _fake_app():
        captured["argv"] = sys.argv[1:]

    with (
        patch("chemrefine.cli.app", _fake_app),
        patch("sys.argv", ["chemrefine", "input.yaml", "--skip"]),
    ):
        runpy.run_module("chemrefine", run_name="__main__")
    assert captured["argv"] == ["resume", "input.yaml"]
