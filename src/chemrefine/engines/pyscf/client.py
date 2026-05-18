"""HTTP client + ExtOpt wrapper entry point for PySCF-driven ORCA runs.

Mirrors :mod:`chemrefine.engines.mlff.client` but adds a ``settings``
block in the payload (method / xc / basis / df / gpu) so the server
can override its per-process defaults on a request-by-request basis.

The verifiable parts (CLI parsing + HTTP RPC) ship in full; the
ExtOpt wrapper :func:`main` stays a placeholder.
"""

from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from chemrefine.errors import JobFailureError

DEFAULT_BIND = "127.0.0.1:8889"
DEFAULT_TIMEOUT = 600.0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Return the wrapper script's parsed CLI namespace.

    Flags match the v3 ``pyscf_client.py`` from ``origin/pyscf`` so the
    SLURM ``run_block`` produced by
    :class:`~chemrefine.engines.pyscf.engine.PyscfEngine` keeps working.
    """
    parser = argparse.ArgumentParser(prog="chemrefine-pyscf-client")
    parser.add_argument("--bind", default=DEFAULT_BIND, help="host:port of the PySCF server")
    parser.add_argument("--method", default="dft", choices=["dft", "hf"])
    parser.add_argument("--xc", default="pbe", help="exchange-correlation functional (DFT)")
    parser.add_argument("--basis", default="def2-svp", help="orbital basis set")
    parser.add_argument("--df", action="store_true", help="enable density fitting / RI")
    parser.add_argument("--gpu", action="store_true", help="attempt gpu4pyscf if installed")
    parser.add_argument("--tag", default=None, help="optional tag for server log")
    parser.add_argument("inputfile", help="ORCA-written ``.extinp.tmp`` to relay")
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Pack the method/basis/etc fields into the payload's ``settings`` block."""
    return {
        "method": args.method,
        "xc": args.xc,
        "basis": args.basis,
        "df": bool(args.df),
        "gpu": bool(args.gpu),
    }


def submit_calculation(
    *,
    server_url: str,
    atom_types: list[str],
    coordinates: list[list[float]],
    charge: int,
    mult: int,
    dograd: bool,
    nthreads: int,
    settings: dict[str, Any],
    tag: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[float, list[list[float]]]:
    """Send one geometry to the PySCF server; return ``(energy, gradient)``.

    Raises :class:`JobFailureError` on any HTTP / connection / JSON
    error so the calling step records a clean failure.
    """
    payload = {
        "atom_types": atom_types,
        "coordinates": coordinates,
        "charge": charge,
        "mult": mult,
        "dograd": dograd,
        "nthreads": nthreads,
        "settings": settings,
        "tag": tag,
    }
    request = Request(
        f"http://{server_url}/calculate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as e:
        raise JobFailureError(f"PySCF server returned HTTP {e.code}: {e.reason}") from e
    except URLError as e:
        raise JobFailureError(f"PySCF server unreachable at {server_url}: {e.reason}") from e
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise JobFailureError(f"PySCF server returned non-JSON body: {body!r}") from e
    if "error" in data:
        raise JobFailureError(f"PySCF server error: {data['error']}")
    try:
        return float(data["energy"]), list(data["gradient"])
    except (KeyError, TypeError, ValueError) as e:
        raise JobFailureError(f"PySCF server response missing fields: {data!r}") from e


def main() -> int:
    """ORCA-invoked entry point — glue between ``.extinp.tmp`` and ``.engrad``.

    TODO: wire :mod:`chemrefine.engines.orca.extopt` around
    :func:`submit_calculation` once a real ExtOpt fixture is captured.
    The shape will mirror :func:`chemrefine.engines.mlff.client.main`
    but call :func:`settings_from_args` to pack the PySCF-specific
    knobs into the payload.
    """
    raise NotImplementedError(
        "PySCF client main not yet ported — see TODO in engines/pyscf/client.py"
    )
