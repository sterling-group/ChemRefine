"""Bind a portable electronic calculation to one molecular pipeline structure."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chemrefine.engines.qiskit.integral_io import load_integrals
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.problem import PreparedProblem, prepare_problem
from chemrefine.errors import ConfigError
from chemrefine.io import read_xyz_frames


def prepare_integral_job(
    xyz_path: Path, *, charge: int, multiplicity: int, options: QiskitOptions
) -> PreparedProblem:
    """Validate geometry/state identity before using supplied MO integrals as an energy.

    Standalone integral APIs may describe models without nuclei. A molecular pipeline
    input must supply its molecular metadata and nuclear constant so its canonical
    energy cannot silently become an electronic-only or different-geometry value.
    """
    source = options.integral_source
    if source is None:
        raise ConfigError("integral preparation requires integral_source")
    data = load_integrals(Path(source.bundle_path), max_bytes=source.max_input_bytes)
    molecule = data.molecular_metadata
    if molecule is None or data.nuclear_repulsion_energy is None:
        raise ConfigError("molecular integral input requires molecular_metadata and nuclear energy")
    frames = read_xyz_frames(xyz_path)
    if len(frames) != 1:
        raise ConfigError("molecular integral input requires exactly one XYZ geometry")
    atoms = frames[0]
    if (
        tuple(atoms.get_chemical_symbols()) != molecule.symbols
        or not np.allclose(
            atoms.positions,
            molecule.coordinates,
            atol=source.geometry_tolerance_angstrom,
            rtol=0,
        )
        or molecule.charge != charge
        or data.multiplicity != multiplicity
    ):
        raise ConfigError("integral bundle geometry, charge or multiplicity differs from the job")
    return prepare_problem(data, active_space=options.active_space, freeze_core=options.freeze_core)
