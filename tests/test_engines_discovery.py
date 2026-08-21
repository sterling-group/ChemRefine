"""Tests for plugin/backend auto-discovery — additions are drop-in self-contained.

The naming convention *is* the discovery rule: a bare-named subpackage under ``engines/`` is a
plugin (imported, self-registers); underscored packages are building blocks and plain modules
are never plugins. Same for ``mlip/backends``, which holds one module per MLIP library:
bare modules register their capabilities, underscored are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest

import chemrefine.engines as engines_pkg
from chemrefine.engines.api import ENGINES, get_engine
from chemrefine.engines.mlip import backends as backends_pkg
from chemrefine.engines.mlip.registry import _BACKENDS, CalculatorSpec

_BUNDLED = {"orca", "mlip", "mlip-extopt", "mlip-train", "pyscf", "pyscf-extopt", "qchem"}


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


def test_a_dropped_in_library_module_registers_both_its_capabilities(monkeypatch, tmp_path: Path):
    """One file per library is the extension point, and it declares its env exactly once.

    The promise `backends/` makes is that making an MLIP available — to run, to train, or
    both — is one new module and no edit to any existing one. Registering both capabilities
    from a single `MlipLibrary` is also what makes the environment they resolve identical by
    construction rather than by two modules agreeing; this drops in a module that does it and
    checks the entry carries both.

    Underscored modules stay helpers and are not discovered.
    """
    (tmp_path / "dummylib.py").write_text(
        "from chemrefine.engines.mlip.registry import MlipLibrary\n"
        "\n"
        "DUMMY = MlipLibrary(\n"
        '    extra="mlip-dummy", package="dummy-pkg", import_name="dummy_mod"\n'
        ")\n"
        "\n"
        '@DUMMY.calculator("dummy_head")\n'
        "def _build_dummy(spec):\n"
        '    """Test-only builder — one CalculatorSpec in, per the registered contract."""\n'
        '    return "DUMMY"\n'
        "\n"
        '@DUMMY.trainer("dummy_head")\n'
        "class DummyTrainer:\n"
        '    """Test-only trainer."""\n'
        "\n"
        '    required_placeholders = frozenset({"TRAIN_SET"})\n'
        "    output_globs = ()\n"
        "    output_dirs = ()\n",
        encoding="utf-8",
    )
    (tmp_path / "_helper.py").write_text(
        'raise AssertionError("underscored modules must not be discovered")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(backends_pkg, "__path__", [*backends_pkg.__path__, str(tmp_path)])
    try:
        backends_pkg._load_backends()
        spec = _BACKENDS["dummy_head"]
        assert spec.extra == "mlip-dummy"
        calc_spec = CalculatorSpec(
            task_name="dummy_head", model_name="", device="cpu", weights=None
        )
        assert spec.builder is not None and spec.builder(calc_spec) == "DUMMY"
        assert spec.trainer is not None and spec.trainer.__name__ == "DummyTrainer"
        assert "chemrefine.engines.mlip.backends._helper" not in sys.modules
    finally:
        _BACKENDS.pop("dummy_head", None)
        sys.modules.pop("chemrefine.engines.mlip.backends.dummylib", None)


def test_a_builder_that_is_not_a_single_spec_callable_is_refused_at_its_own_line():
    """The decorator is the gate: a malformed drop-in fails at import of its module.

    The old keyword-plus-catch-all shape was checked by nothing, and three shipped
    builders silently swallowed a knob the dispatch was passing. Arity is the property a
    signature can prove, so registration proves it — and names the rule, so the author of
    a new backend is told what a builder is rather than left to diff a sibling.
    """
    from chemrefine.engines.mlip.registry import CalculatorBuilder, MlipLibrary

    def _kwargs_only(**_kw: object) -> None:
        return None

    def _two_params(_spec: CalculatorSpec, _extra: str) -> None:
        return None

    lib = MlipLibrary(extra="mlip-bad", package="bad", import_name="bad")
    for malformed in ("not-a-function", _kwargs_only, _two_params):
        with pytest.raises(TypeError, match=r"must decorate a callable|exactly one positional"):
            lib.calculator("bad_head")(cast(CalculatorBuilder, malformed))
    assert "bad_head" not in _BACKENDS


def test_one_task_cannot_name_two_libraries(monkeypatch, tmp_path: Path):
    """The drift the single registry exists to make impossible, refused at registration.

    Two modules claiming the same `task_name` for different environments is exactly what the
    old split registries could not see: each was internally consistent, and the step was
    provisioned into one env and launched expecting the other.
    """
    from chemrefine.engines.mlip.registry import MlipLibrary

    first = MlipLibrary(extra="mlip-a", package="a", import_name="a")
    second = MlipLibrary(extra="mlip-b", package="b", import_name="b")
    first.calculator("contested")(lambda _spec: None)
    try:
        with pytest.raises(ValueError, match="one task names one library"):
            second.trainer("contested")(type("T", (), {}))
    finally:
        _BACKENDS.pop("contested", None)
