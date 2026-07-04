"""Tests for plugin/backend auto-discovery — additions are drop-in self-contained.

The naming convention *is* the discovery rule: a bare-named subpackage under ``engines/`` is a
plugin (imported, self-registers); underscored packages are building blocks and plain modules
are never plugins. Same for ``mlip/backends``: bare modules register, underscored are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chemrefine.engines as engines_pkg
from chemrefine.engines.api import ENGINES, get_engine
from chemrefine.engines.mlip import backends as backends_pkg
from chemrefine.engines.mlip.calculator import _BACKENDS

_BUNDLED = {"orca", "mlip", "mlip-extopt", "mlip-train", "pyscf", "pyscf-extopt"}


def test_every_bundled_plugin_is_discovered():
    """Importing chemrefine.engines registers every bundled engine — no import list."""
    assert set(ENGINES) >= _BUNDLED


def test_every_bundled_backend_is_discovered():
    """Importing mlip.backends registers every bundled backend module's heads."""
    assert {"omol", "mace_off", "sevenn", "orb", "chgnet"} <= set(_BACKENDS)


def test_dropped_in_plugin_package_is_discovered(monkeypatch, tmp_path: Path):
    """A bare-named package dropped into engines/ registers itself; others are skipped."""
    (tmp_path / "goodplug").mkdir()
    (tmp_path / "goodplug" / "__init__.py").write_text(
        "from chemrefine.engines.api import register\n"
        "\n"
        '@register("goodplug-test")\n'
        "class GoodPlug:\n"
        '    name = "goodplug-test"\n',
        encoding="utf-8",
    )
    (tmp_path / "_hiddenplug").mkdir()
    (tmp_path / "_hiddenplug" / "__init__.py").write_text(
        'raise AssertionError("underscored packages must not be discovered")\n',
        encoding="utf-8",
    )
    (tmp_path / "straymodule.py").write_text(
        'raise AssertionError("plain modules must not be discovered")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(engines_pkg, "__path__", [*engines_pkg.__path__, str(tmp_path)])
    try:
        engines_pkg._load_plugins()
        assert get_engine("goodplug-test").name == "goodplug-test"
        assert "chemrefine.engines._hiddenplug" not in sys.modules
        assert "chemrefine.engines.straymodule" not in sys.modules
    finally:
        ENGINES.pop("goodplug-test", None)
        sys.modules.pop("chemrefine.engines.goodplug", None)


def test_dropped_in_backend_module_is_discovered(monkeypatch, tmp_path: Path):
    """A bare module dropped into mlip/backends registers itself; underscored is skipped."""
    (tmp_path / "dummybackend.py").write_text(
        "from chemrefine.engines.mlip.calculator import register_backend\n"
        "\n"
        "@register_backend(\n"
        '    "dummy_head", extra="mlip-dummy", package="dummy-pkg", import_name="dummy_mod"\n'
        ")\n"
        "def _build_dummy(**_kw):\n"
        '    """Test-only builder."""\n'
        '    return "DUMMY"\n',
        encoding="utf-8",
    )
    (tmp_path / "_helper.py").write_text(
        'raise AssertionError("underscored modules must not be discovered")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(backends_pkg, "__path__", [*backends_pkg.__path__, str(tmp_path)])
    try:
        backends_pkg._load_backends()
        assert _BACKENDS["dummy_head"].extra == "mlip-dummy"
        assert _BACKENDS["dummy_head"].builder() == "DUMMY"
        assert "chemrefine.engines.mlip.backends._helper" not in sys.modules
    finally:
        _BACKENDS.pop("dummy_head", None)
        sys.modules.pop("chemrefine.engines.mlip.backends.dummybackend", None)
