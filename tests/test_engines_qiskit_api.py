"""Public solver entry points preserve inputs and forward externally supplied artifacts."""

from types import SimpleNamespace

import pytest

from chemrefine.engines.qiskit import api
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.result import QiskitRunResult


@pytest.fixture
def recording_runner(monkeypatch):
    """Capture solver calls without importing optional quantum packages."""
    calls = []
    expected = QiskitRunResult(-1.0)

    def run(prepared, **kwargs):
        calls.append((prepared, kwargs))
        return expected

    monkeypatch.setattr(api, "run_problem", run)
    return calls, expected


def test_exact_entry_point_keeps_matching_options_and_resets_other_solver_options(recording_runner):
    calls, expected = recording_runner
    prepared = object()
    exact = QiskitOptions(mapper="bravyi_kitaev")
    assert api.solve_exact(prepared, options=exact) is expected
    assert calls[-1] == (prepared, {"options": exact})
    assert calls[-1][1]["options"] is exact
    source = QiskitOptions(algorithm={"name": "adapt_vqe", "options": {"max_iterations": 9}})
    assert api.solve_exact(prepared, options=source) is expected
    assert calls[-1][1]["options"].algorithm.name == "exact"
    assert calls[-1][1]["options"].algorithm.options == {}
    assert source.algorithm.options == {"max_iterations": 9}
    assert api.solve_exact(prepared) is expected


@pytest.mark.parametrize("ansatz", ["uccsd", "ucc", "efficient_su2"])
def test_custom_excitation_injection_retains_only_compatible_ansatz_options(
    recording_runner, ansatz
):
    calls, expected = recording_runner
    prepared = object()
    custom = [((0,), (1,)), ((0, 2), (1, 3))]
    selected_options = {"reps": 2} if ansatz in {"ucc", "uccsd"} else {"flatten": True}
    source = QiskitOptions(ansatz={"name": ansatz, "options": selected_options})
    initial_point = [0.1, 0.2]

    def callback(_record):
        return None

    assert (
        api.run_vqe(
            prepared,
            options=source,
            excitations=custom,
            initial_point=initial_point,
            callback=callback,
            reference_energy_hartree=-1.2,
        )
        is expected
    )
    forwarded = calls[-1][1]
    options = forwarded["options"]
    assert options.algorithm.name == "vqe"
    assert options.ansatz.name == "ucc"
    assert options.ansatz.options["excitations"] == custom
    assert "flatten" not in options.ansatz.options
    assert (options.ansatz.options.get("reps") == 2) == (ansatz in {"ucc", "uccsd"})
    assert forwarded["initial_point"] is initial_point
    assert forwarded["callback"] is callback
    assert forwarded["reference_energy_hartree"] == -1.2
    assert source.ansatz.name == ansatz
    assert source.ansatz.options == selected_options
    custom.append(((2,), (3,)))
    assert len(options.ansatz.options["excitations"]) == 2


def test_default_vqe_and_external_adapt_pool_forward_without_reconstruction(recording_runner):
    calls, expected = recording_runner
    prepared = object()
    assert api.run_vqe(prepared, options={"mapper": "parity"}) is expected
    assert calls[-1][1]["options"].ansatz.name == "uccsd"
    assert calls[-1][1]["options"].mapper.name == "parity"
    pool = SimpleNamespace(operators=(object(),), metadata=({"label": "external"},))

    def callback(_record):
        return None

    assert (
        api.run_adapt_vqe(
            prepared,
            options={"algorithm": {"name": "adapt_vqe", "options": {"max_iterations": 3}}},
            operator_pool=pool,
            callback=callback,
            reference_energy_hartree=-1.2,
        )
        is expected
    )
    forwarded = calls[-1][1]
    assert forwarded["operator_pool"] is pool
    assert forwarded["callback"] is callback
    assert forwarded["reference_energy_hartree"] == -1.2
    assert forwarded["options"].algorithm.options == {"max_iterations": 3}
    assert api.run_adapt_vqe(prepared) is expected
    assert calls[-1][1]["options"].algorithm.name == "adapt_vqe"
