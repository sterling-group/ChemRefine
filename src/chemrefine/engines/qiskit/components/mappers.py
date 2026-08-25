"""Built-in fermion-to-qubit mapper factories."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from chemrefine.engines.qiskit.registry import MAPPERS, NoComponentOptions


class ParityMapperOptions(BaseModel):
    """Options controlling ParityMapper's optional two-qubit reduction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    two_qubit_reduction: bool = True


@MAPPERS.register("jordan_wigner", NoComponentOptions)
def build_jordan_wigner_mapper(*, options: BaseModel, problem: Any) -> object:
    """Build the occupation-preserving Jordan-Wigner mapper."""
    del options, problem
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    return JordanWignerMapper()


@MAPPERS.register("parity", ParityMapperOptions)
def build_parity_mapper(*, options: ParityMapperOptions, problem: Any) -> object:
    """Build a parity mapper, optionally reducing two qubits by particle parity."""
    from qiskit_nature.second_q.mappers import ParityMapper

    num_particles = problem.num_particles if options.two_qubit_reduction else None
    return ParityMapper(num_particles=num_particles)
