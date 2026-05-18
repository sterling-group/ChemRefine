"""HTTP client invoked by the PySCF ExtOpt wrapper script.

TODO: port from ``origin/pyscf`` (``src/chemrefine/pyscf_client.py``).
Mirrors :mod:`chemrefine.engines.mlff.client` — reads ORCA's
``.extinp.tmp``, POSTs the geometry to ``http://<bind>/calculate``,
writes the returned energy + gradient to ``.engrad``.
"""

from __future__ import annotations


def submit_calculation(
    *,
    server_url: str,
    atom_types: list[str],
    coordinates: list[list[float]],
    charge: int,
    mult: int,
    dograd: bool,
    nthreads: int,
    settings: dict,
    tag: str | None = None,
) -> tuple[float, list[list[float]]]:
    """Send one geometry to the PySCF server; return ``(energy, gradient)``.

    Placeholder — see module docstring for the port plan.
    """
    _ = server_url, atom_types, coordinates, charge, mult, dograd, nthreads, settings, tag
    raise NotImplementedError(
        "PySCF client not yet ported — see TODO in engines/pyscf/client.py"
    )


def main() -> int:
    """CLI entry point (placeholder)."""
    raise NotImplementedError(
        "PySCF client not yet ported — see TODO in engines/pyscf/client.py"
    )
