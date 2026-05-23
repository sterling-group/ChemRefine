"""MLFF Python-script template renderer (thin wrapper around the shared template).

Delegates to :mod:`chemrefine.engines._template` so MLFF and PySCF
share one renderer. See that module for placeholder + output-contract
semantics.
"""

from __future__ import annotations

from pathlib import Path

from chemrefine.engines import _template


def build_input(
    *,
    xyz_path: Path,
    template_path: Path,
    output_path: Path,
    output_json_path: Path,
    charge: int,
    multiplicity: int,
) -> Path:
    """Render an MLFF ``step{N}.py`` template into ``output_path``.

    See :func:`chemrefine.engines._template.build_input` for the
    placeholder + output-contract semantics. This wrapper just sets
    the backend-specific "not found" error message.
    """
    return _template.build_input(
        xyz_path=xyz_path,
        template_path=template_path,
        output_path=output_path,
        output_json_path=output_json_path,
        charge=charge,
        multiplicity=multiplicity,
        not_found_message="MLFF template not found",
    )
