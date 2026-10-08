"""Built-in fermion-to-qubit mapper factories."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from chemrefine.engines.qiskit.registry import MAPPERS, NoComponentOptions
from chemrefine.engines.qiskit.tapering import Z2TaperingOptions


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


@MAPPERS.register("bravyi_kitaev", NoComponentOptions)
def build_bravyi_kitaev_mapper(*, options: BaseModel, problem: Any) -> object:
    """Build Bravyi-Kitaev mapping without imposing a symmetry reduction."""
    del options, problem
    from qiskit_nature.second_q.mappers import BravyiKitaevMapper

    return BravyiKitaevMapper()


@MAPPERS.register("parity", ParityMapperOptions)
def build_parity_mapper(*, options: ParityMapperOptions, problem: Any) -> object:
    """Build a parity mapper, optionally reducing two qubits by particle parity."""
    from qiskit_nature.second_q.mappers import ParityMapper

    num_particles = problem.num_particles if options.two_qubit_reduction else None
    return ParityMapper(num_particles=num_particles)


@MAPPERS.register(
    "z2_tapered",
    Z2TaperingOptions,
    capabilities=frozenset({"reference_aware"}),
    status="experimental",
    supported_domains=(
        "reference-compatible molecular Pauli symmetries",
        "automatic Clifford reference or explicitly verified general reference",
    ),
)
def build_z2_tapered_mapper(
    *,
    options: Z2TaperingOptions,
    problem: Any,
    reference_selection: Any = None,
    prepared: Any = None,
) -> object:
    """Taper only symmetries with verified eigenvalues in the actual selected reference."""
    from chemrefine.engines.qiskit.tapering import build_reference_tapered_mapper

    return build_reference_tapered_mapper(
        problem, options=options, reference_selection=reference_selection, prepared=prepared
    )
