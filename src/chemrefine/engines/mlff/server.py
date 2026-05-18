"""Flask/Waitress HTTP server that serves MLFF energies + gradients.

TODO: port the v3 server from :file:`src/chemrefine/server.py` on
``main``. The route is:

    POST /calculate
    body: {"atom_types": [...], "coordinates": [...],
           "charge": int, "mult": int, "nthreads": int}
    returns: {"energy": float, "gradient": [[fx, fy, fz], ...]}

A single :class:`MlffCalculator` is constructed per worker thread (key:
``threading.get_ident()``) so model weights are loaded once and shared
across optimisation steps that hit the same thread.

The CLI parses ``--model``, ``--task-name``, ``--device``, ``--bind``,
``--nthreads``. Tests can exercise the route via ``pytest-flask`` or by
calling :func:`run` with a no-op calculator mock; verifying real
inference needs torch + a model checkpoint on disk.
"""

from __future__ import annotations


def main() -> int:
    """Server entry point (placeholder)."""
    raise NotImplementedError(
        "MLFF server not yet ported — see TODO in engines/mlff/server.py"
    )
