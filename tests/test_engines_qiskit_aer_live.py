"""Real numerical execution must confirm the requested method and physical device."""

import pytest

from chemrefine.engines.qiskit.components.estimators import build_aer_backend


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
@pytest.mark.parametrize("precision", ["single", "double"])
def test_aer_method_executes_on_requested_device(device, method, precision):
    """Bell-state counts and provider metadata certify execution rather than discovery."""
    aer = pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit

    requested = "GPU" if device == "cuda" else "CPU"
    if requested not in aer.AerSimulator().available_devices():
        pytest.skip(f"{requested} unavailable in this Aer build")
    backend = build_aer_backend(
        method=method, device=device, cores=1, simulation_precision=precision
    )
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    result = backend.run(circuit, shots=256, seed_simulator=19).result()
    assert result.success
    assert set(result.get_counts()) == {"00", "11"}
    assert sum(result.get_counts().values()) == 256
    assert result.get_counts()["00"] / 256 == pytest.approx(0.5, abs=0.15)
    assert result.results[0].metadata["device"] == requested
    assert result.results[0].metadata["method"] == method
