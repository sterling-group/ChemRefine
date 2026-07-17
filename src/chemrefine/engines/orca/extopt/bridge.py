"""ExtOpt wrapper-script client.

Invoked by ORCA via the ``ProgExt`` wrapper once per optimization step.
Reads the ``.extinp.tmp`` ORCA wrote, POSTs it to the shared ExtOpt
server, and writes the returned energy + gradient back as ``.engrad``
so ORCA can take its next step.

Backend-specific knobs are contributed by each backend's
:meth:`ComputeBackend.add_cli_args` and packed into the JSON
payload's ``settings`` block by its :meth:`settings_from_args` — the
shared layer here forwards them unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from chemrefine.engines._backend_server import sidecar
from chemrefine.engines._backend_server.base import SERVER_TOKEN_FILENAME, SERVER_URL_FILENAME
from chemrefine.engines._backend_server.registry import known_backends, load_calculator
from chemrefine.engines.orca.extopt import protocol
from chemrefine.errors import JobFailureError

DEFAULT_TIMEOUT: float = 600.0
"""Seconds before a single ``/calculate`` request times out."""

logger = logging.getLogger(__name__)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Return the wrapper script's parsed CLI namespace.

    Each registered backend contributes its own flags via
    :meth:`ComputeBackend.add_cli_args`; the shared layer owns
    only the generic skeleton (``--backend``, ``--bind``,
    ``--url-file``, ``--tag``, ``inputfile``).
    """
    parser = argparse.ArgumentParser(prog="chemrefine-extopt-bridge")
    parser.add_argument(
        "--backend",
        required=True,
        choices=known_backends(),
        help="which ComputeBackend the server is running",
    )
    parser.add_argument(
        "--bind",
        default=None,
        help="explicit host:port (overrides --url-file)",
    )
    parser.add_argument(
        "--url-file",
        default=None,
        help=f"sidecar URL file (default: $WORK_DIR/{SERVER_URL_FILENAME})",
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help=f"sidecar auth-token file (default: $WORK_DIR/{SERVER_TOKEN_FILENAME})",
    )
    parser.add_argument("--tag", default=None, help="optional correlation tag for server log")
    for backend_name in known_backends():
        load_calculator(backend_name).add_cli_args(parser)
    # ``inputfile`` is the positional; register it last so backend
    # contributions don't interleave with it.
    parser.add_argument("inputfile", help="ORCA-written ``.extinp.tmp`` to relay")
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch to the selected backend's ``settings_from_args``."""
    return load_calculator(args.backend).settings_from_args(args)


def submit_calculation(
    *,
    server_url: str,
    data: protocol.CalculationData,
    tag: str | None = None,
    token: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[float, list[list[float]]]:
    """Send one geometry to the ExtOpt server; return ``(energy, gradient)``.

    ``token`` (the per-run secret from the server's token sidecar) rides as
    a bearer ``Authorization`` header; the server rejects requests without
    it. Raises :class:`JobFailureError` on any HTTP / connection / JSON
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
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        f"http://{server_url}/calculate",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
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
    if inputfile.endswith(protocol.EXTINP_SUFFIX):
        return Path(inputfile[: -len(protocol.EXTINP_SUFFIX)] + protocol.ENGRAD_SUFFIX)
    return Path(inputfile).with_suffix(protocol.ENGRAD_SUFFIX)


def _tag_for(inputfile: str) -> str:
    """Per-call correlation tag = the ORCA jobname (``.extinp.tmp`` stripped).

    ORCA reuses the same ``<job>.extinp.tmp`` for every geometry step of one
    optimisation, so this tags all calls of one structure with that structure's
    name. Backends that dump per-call artefacts (e.g. PySCF active-space tensors)
    then write one file per structure, holding the converged-geometry result
    (last write wins) instead of overwriting a single shared file across
    structures.
    """
    name = Path(inputfile).name
    if name.endswith(protocol.EXTINP_SUFFIX):
        return name[: -len(protocol.EXTINP_SUFFIX)]
    return Path(inputfile).stem


def resolve_server_url(args: argparse.Namespace) -> str:
    """Determine the server URL from ``--bind`` (explicit) or ``--url-file``."""
    if args.bind:
        return cast(str, args.bind)
    import os

    default_dir = Path(os.environ.get("WORK_DIR", "."))
    url_file = args.url_file or str(default_dir / SERVER_URL_FILENAME)
    return sidecar.read_server_url(url_file)


def resolve_server_token(args: argparse.Namespace) -> str | None:
    """Return the per-run bearer token, or ``None`` when no token sidecar exists.

    Resolution mirrors :func:`resolve_server_url`: an explicit ``--token-file``
    wins, else ``$WORK_DIR/server.token`` (the wrapper runs with ``WORK_DIR``
    exported, so the ``--bind`` shortcut still finds the token). A missing
    file degrades to ``None`` — the header is omitted and the server decides.
    """
    import os

    default_dir = Path(os.environ.get("WORK_DIR", "."))
    token_file = Path(args.token_file) if args.token_file else default_dir / SERVER_TOKEN_FILENAME
    if not token_file.is_file():
        return None
    return sidecar.read_server_token(token_file)


def main() -> int:
    """ORCA-invoked entry point: relay one ``.extinp.tmp`` → ``.engrad`` step."""
    import sys

    args = parse_args(sys.argv[1:])
    server_url = resolve_server_url(args)
    data = protocol.read_extinp(args.inputfile, settings=settings_from_args(args))
    energy, gradient = submit_calculation(
        server_url=server_url,
        data=data,
        tag=args.tag or _tag_for(args.inputfile),
        token=resolve_server_token(args),
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


if __name__ == "__main__":
    raise SystemExit(main())
