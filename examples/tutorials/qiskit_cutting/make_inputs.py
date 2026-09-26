"""Write the bound GHZ preparation consumed by the cutting tutorials."""

from pathlib import Path

from qiskit import QuantumCircuit, qpy

directory = Path(__file__).resolve().parent / "inputs"
directory.mkdir(exist_ok=True)
circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
with (directory / "ghz.qpy").open("wb") as stream:
    qpy.dump(circuit, stream)
