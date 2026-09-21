"""Independent mapper selection without requiring the optional Qiskit packages."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from chemrefine.engines.qiskit.options import ComponentSelection, QiskitOptions
from chemrefine.engines.qiskit.registry import MAPPERS, validate_component_graph


def test_bravyi_kitaev_is_a_lazy_replaceable_mapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new mapping uses the same registry as JW and parity, and no driver."""
    module = ModuleType("qiskit_nature.second_q.mappers")
    expected = object()
    monkeypatch.setattr(module, "BravyiKitaevMapper", lambda: expected, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    options = QiskitOptions(mapper="bravyi-kitaev")
    validate_component_graph(options)
    assert options.mapper.name == "bravyi_kitaev"
    assert MAPPERS.build(ComponentSelection.named("bravyi_kitaev"), problem=None) is expected
