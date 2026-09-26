"""Native solver energy conventions and provider dependency composition."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.backend import QiskitBackend
from chemrefine.engines.qiskit.native import NativeOutcome, NativeSolveRequest, summarize_native
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.registry import (
    ALGORITHMS,
    OPTIMIZERS,
    ComponentSpec,
    NoComponentOptions,
)
from chemrefine.engines.qiskit.workflow import run_problem
from chemrefine.errors import ConfigError


def _prepared(offsets):
    """A native result needs only prepared metadata and physical scalar constants."""
    return SimpleNamespace(
        energy_offsets=offsets,
        original_num_spatial_orbitals=5,
        active_orbitals=(1, 3),
        num_particles=(1, 1),
        num_spin_orbitals=4,
        metadata={"transformations": ["active_space"]},
        provenance={"source": "test"},
    )


@pytest.mark.parametrize("nuclear", [None, 0.0, 0.7])
def test_native_restores_each_energy_offset_once(nuclear):
    """Inactive energy is electronic; nuclear presence determines total convention."""
    offsets = {"ActiveSpaceTransformer": -4.0}
    if nuclear is not None:
        offsets["nuclear_repulsion_energy"] = nuclear
    request = NativeSolveRequest(_prepared(offsets), QiskitOptions(), reference_energy_hartree=-5)
    outcome = NativeOutcome(
        -1.25, converged=None, num_qubits=4, diagnostics={"tier": "experimental"}
    )
    result = summarize_native(outcome, request, runtime_seconds=0.1)
    assert result.electronic_energy_hartree == -5.25
    assert result.energy_hartree == -5.25 + (nuclear or 0)
    assert result.total_energy_hartree == (None if nuclear is None else result.energy_hartree)
    assert result.nuclear_repulsion_energy_hartree == nuclear
    assert result.energy_error_hartree == result.energy_hartree + 5
    assert result.converged is None
    assert result.active_space["active_orbitals"] == [1, 3]
    assert result.as_dict()["metadata"]["energy_convention"] == (
        "electronic" if nuclear is None else "total"
    )


@pytest.mark.parametrize(
    "outcome",
    [
        object(),
        NativeOutcome(float("nan")),
        NativeOutcome(1j),
        NativeOutcome(-1.0, diagnostics={"bad": object()}),
        NativeOutcome(-1.0, diagnostics={"bad": float("inf")}),
    ],
)
def test_native_rejects_invalid_outcomes(outcome):
    """Nonfinite values and opaque diagnostics cannot enter pipeline artifacts."""
    with pytest.raises(ConfigError):
        summarize_native(
            outcome, NativeSolveRequest(_prepared({}), QiskitOptions()), runtime_seconds=0
        )


def test_native_missing_optional_versions_are_omitted(monkeypatch):
    """Reporting works without assuming every optional package is installed."""

    def missing(name):
        """Simulate a separately distributed provider with no local metadata."""
        raise PackageNotFoundError(name)

    monkeypatch.setattr("chemrefine.engines.qiskit.native.version", missing)
    result = summarize_native(
        NativeOutcome(-1), NativeSolveRequest(_prepared({}), QiskitOptions()), runtime_seconds=0
    )
    assert result.metadata["provenance"]["package_versions"] == {}


def test_native_dispatch_does_not_construct_nature_solver(monkeypatch):
    """Native solvers consume the original prepared problem and explicit controls."""
    calls = []

    def solve(*, options, request):
        """Return a minimal algorithm-owned outcome and retain the request."""
        calls.append(request)
        return NativeOutcome(-2.0, termination_reason="done")

    monkeypatch.setitem(
        ALGORITHMS._specs,
        "native_test",
        ComponentSpec(NoComponentOptions, solve, execution="native"),
    )
    prepared = _prepared({"nuclear_repulsion_energy": 0.5})
    result = run_problem(
        prepared,
        options={"algorithm": "native_test"},
        initial_point=[0.2],
        reference_energy_hartree=-1.4,
    )
    assert calls[0].prepared is prepared
    assert calls[0].initial_point == [0.2]
    assert result.energy_hartree == -1.5
    assert result.solver == "native_test"


def test_optimizer_provider_is_discovered_when_consumed(monkeypatch):
    """Provider dependency declarations apply to any component category."""
    requirement = BackendRequirement(extra="custom_optimizer", import_name="custom_optimizer")
    monkeypatch.setitem(
        OPTIMIZERS._specs,
        "custom_test",
        ComponentSpec(NoComponentOptions, lambda: None, backend_requirement=requirement),
    )
    backend = QiskitBackend()
    assert (
        backend.backend_requirement({"algorithm": "vqe", "optimizer": "custom_test"}) == requirement
    )
    assert (
        backend.backend_requirement({"algorithm": "exact", "optimizer": "custom_test"}).extra
        == "qiskit"
    )
    assert "custom_optimizer" in backend.backend_extras()
