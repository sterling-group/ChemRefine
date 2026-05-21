"""Tests for ``chemrefine/__init__.py`` — version resolution."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch


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
