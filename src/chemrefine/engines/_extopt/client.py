"""ExtOpt wrapper-script client.

Invoked by ORCA via the ``ProgExt`` wrapper once per optimization step.
Reads the ``.extinp.tmp`` ORCA wrote, POSTs it to the shared ExtOpt
server, and writes the returned energy + gradient back as ``.engrad``
so ORCA can take its next step.

Backend-specific knobs (PySCF's ``--method``/``--xc``/``--basis``,
etc.) are passed as a ``settings`` block in the JSON payload — the
shared server forwards them unchanged to the backend's ``calc()``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from chemrefine.engines._extopt import protocol
from chemrefine.engines._extopt.registry import CALCULATORS
from chemrefine.errors import JobFailureError

DEFAULT_TIMEOUT: float = 600.0
"""Seconds before a single ``/calculate`` request times out."""

logger = logging.getLogger(__name__)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Return the wrapper script's parsed CLI namespace."""
    parser = argparse.ArgumentParser(prog="chemrefine-extopt-client")
    parser.add_argument(
        "--backend", required=True, choices=sorted(CALCULATORS),
        help="which BaseExtOptCalculator the server is running",
    )
    parser.add_argument(
        "--bind", default=None,
        help="explicit host:port (overrides --url-file)",
    )
    parser.add_argument(
        "--url-file", default=None,
        help="sidecar URL file (default: $WORK_DIR/server.url)",
    )
    parser.add_argument("--tag", default=None, help="optional correlation tag for server log")
    # PySCF settings (no-op for MLFF — server ignores unknown settings keys)
    parser.add_argument("--method", default="dft", choices=["dft", "hf"])
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--df", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("inputfile", help="ORCA-written ``.extinp.tmp`` to relay")
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Pack backend-specific knobs into the payload's ``settings`` block."""
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
    data: protocol.CalculationData,
    tag: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[float, list[list[float]]]:
    """Send one geometry to the ExtOpt server; return ``(energy, gradient)``.

    Raises :class:`JobFailureError` on any HTTP / connection / JSON
    error so the calling step records a clean failure.
    """
    payload = {
        "atom_types": list(data.symbols),
        "coordinates": data.positions_angstrom.tolist(),
        "charge": data.charge,
        "mult": data.multiplicity,
        "nthreads": data.nthreads,
        "dograd": data.dograd,
        "settings": data.settings,
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
        raise JobFailureError(f"ExtOpt server returned HTTP {e.code}: {e.reason}") from e
    except URLError as e:
        raise JobFailureError(f"ExtOpt server unreachable at {server_url}: {e.reason}") from e
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        raise JobFailureError(f"ExtOpt server returned non-JSON body: {body!r}") from e
    if "error" in parsed:
        raise JobFailureError(f"ExtOpt server error: {parsed['error']}")
    try:
        return float(parsed["energy"]), list(parsed["gradient"])
    except (KeyError, TypeError, ValueError) as e:
        raise JobFailureError(f"ExtOpt server response missing fields: {parsed!r}") from e


def _engrad_path_for(inputfile: str) -> Path:
    """ORCA writes ``X.extinp.tmp`` and reads ``X.engrad`` back — rewrite the suffix."""
    if inputfile.endswith(".extinp.tmp"):
        return Path(inputfile[: -len(".extinp.tmp")] + ".engrad")
    return Path(inputfile).with_suffix(".engrad")


def resolve_server_url(args: argparse.Namespace) -> str:
    """Determine the server URL from ``--bind`` (explicit) or ``--url-file``."""
    if args.bind:
        return args.bind
    import os

    url_file = args.url_file or os.path.join(
        os.environ.get("WORK_DIR", "."), "server.url"
    )
    return protocol.read_server_url(url_file)


def main() -> int:
    """ORCA-invoked entry point: relay one ``.extinp.tmp`` → ``.engrad`` step."""
    import sys

    args = parse_args(sys.argv[1:])
    server_url = resolve_server_url(args)
    data = protocol.read_extinp(args.inputfile, settings=settings_from_args(args))
    energy, gradient = submit_calculation(
        server_url=server_url, data=data, tag=args.tag
    )
    engrad_path = _engrad_path_for(args.inputfile)
    protocol.write_engrad(
        path=engrad_path,
        n_atoms=len(data.symbols),
        energy_hartree=energy,
        gradients_hartree_per_bohr=gradient if data.dograd else None,
        dograd=data.dograd,
    )
    return 0
