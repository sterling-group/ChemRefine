"""HTTP client invoked by ORCA's ExtOpt wrapper.

TODO: port the v3 client from :file:`src/chemrefine/client.py` on
``main``. The flow is:

1. ORCA writes the current geometry to ``{base}.extinp.tmp`` and
   invokes the wrapper script.
2. The wrapper reads ``.extinp.tmp`` (atom types, coordinates, charge,
   multiplicity, ``dograd`` flag, ``ncores``).
3. POSTs the payload to ``http://<bind>/calculate``.
4. Writes the returned energy + gradient as ``{base}.engrad`` for ORCA
   to pick up.

Verifying needs a running MLFF server fixture; placeholder for now.
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
) -> tuple[float, list[list[float]]]:
    """Send one geometry to the MLFF server; return ``(energy, gradient)``.

    Placeholder — see module docstring for the port plan.
    """
    _ = server_url, atom_types, coordinates, charge, mult, dograd, nthreads
    raise NotImplementedError(
        "MLFF client not yet ported — see TODO in engines/mlff/client.py"
    )


def main() -> int:
    """CLI entry point (placeholder)."""
    raise NotImplementedError(
        "MLFF client not yet ported — see TODO in engines/mlff/client.py"
    )
