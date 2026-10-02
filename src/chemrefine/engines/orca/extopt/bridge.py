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
import math
from http.client import HTTPException
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request

from chemrefine.engines._backend_server import sidecar
from chemrefine.engines._backend_server.base import (
    SERVER_TOKEN_FILENAME,
    SERVER_URL_FILENAME,
    CalculationData,
)
from chemrefine.engines._backend_server.registry import known_backends, load_calculator
from chemrefine.engines.orca.extopt import protocol
from chemrefine.engines.orca.extopt.engine import ExtOptOptions
from chemrefine.errors import (
    ENDPOINT_UNREACHABLE,
    ChemRefineError,
    JobFailureError,
    fetch,
    refusal_text,
)

DEFAULT_TIMEOUT: float = float(ExtOptOptions.model_fields["gradient_timeout_seconds"].default)
"""Seconds before a single ``/calculate`` request times out.

The options model's default, read off the field that declares ``gradient_timeout_seconds``
so the flag and the knob cannot name two numbers — the engine renders the flag from the
model on every job, and a hand-run bridge without it answers with the same value.
"""

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
        help=f"sidecar auth-token file (default: {SERVER_TOKEN_FILENAME} beside the URL file)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="seconds to wait on one /calculate call before giving the geometry up",
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


def _timed_out(server_url: str, timeout: float) -> str:
    """The one sentence for a gradient that did not come back in time."""
    return (
        f"ExtOpt server at {server_url} gave no reply within {timeout:g}s — a gradient "
        "slower than the step's gradient_timeout_seconds, or a hung server"
    )


def submit_calculation(
    *,
    server_url: str,
    data: CalculationData,
    token: str,
    tag: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[float, list[list[float]]]:
    """Send one geometry to the ExtOpt server; return ``(energy, gradient)``.

    ``token`` (the per-run secret from the server's token sidecar) rides as
    a bearer ``Authorization`` header; the server rejects requests without
    it. Raises :class:`JobFailureError` on any HTTP / connection / JSON
    error so the calling step records a clean failure.

    ``timeout`` bounds the one call. Its expiry arrives two ways — a ``TimeoutError``
    straight from the socket while the server is still computing, or wrapped in a
    ``URLError`` when the connection itself timed out — and both are named as what they
    are: a gradient slower than the step's ``gradient_timeout_seconds``, or a hung
    server. Unnamed, the bare form would escape every handler here and crash the wrapper
    with a traceback, and the wrapped one would read as an unreachable server.
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
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    request = Request(
        f"http://{server_url}/calculate",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        # The server wrote its own sidecar; no user-supplied scheme can reach it.
        body = fetch(request, timeout=timeout)
    except HTTPError as e:
        raise JobFailureError(f"ExtOpt server returned HTTP {e.code}: {e.reason}") from e
    except URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise JobFailureError(_timed_out(server_url, timeout)) from e
        raise JobFailureError(f"ExtOpt server unreachable at {server_url}: {e.reason}") from e
    except TimeoutError as e:
        raise JobFailureError(_timed_out(server_url, timeout)) from e
    except HTTPException as e:
        # A server that answers and then breaks the protocol mid-body (IncompleteRead on a
        # dying worker) — neither an HTTPError nor a URLError, so without this clause it
        # would crash the wrapper with a traceback in the runlog instead of the classified
        # failure ORCA's step records like every other server fault.
        raise JobFailureError(f"ExtOpt server sent a broken response: {e!r}") from e
    except ENDPOINT_UNREACHABLE as e:
        # Every other way the endpoint gives no usable answer — a socket reset after the
        # server took the request — is the same classified failure, not a traceback.
        raise JobFailureError(f"ExtOpt server connection to {server_url} failed: {e!r}") from e
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        raise JobFailureError(f"ExtOpt server returned non-JSON body: {body!r}") from e
    if "error" in parsed:
        raise JobFailureError(f"ExtOpt server error: {parsed['error']}")
    try:
        energy = float(parsed["energy"])
        gradient = [[float(component) for component in row] for row in parsed["gradient"]]
    except (KeyError, TypeError, ValueError) as e:
        raise JobFailureError(f"ExtOpt server response missing fields: {parsed!r}") from e
    # `json.loads` accepts the bare `NaN` / `Infinity` Python's encoder emits, and
    # `f"{x:.12e}"` writes them into the .engrad as text ORCA then reads. The server
    # refuses them on its side too; this is the classified failure for a server that
    # does not — the same JobFailureError every other unusable answer becomes.
    if not math.isfinite(energy) or any(
        not math.isfinite(component) for row in gradient for component in row
    ):
        raise JobFailureError(
            "ExtOpt server returned a non-finite energy or gradient (nan/inf): the backend "
            "could not evaluate this geometry"
        )
    return energy, gradient


def _engrad_path_for(inputfile: str) -> Path:
    """ORCA writes ``X.extinp.tmp`` and reads ``X.engrad`` back — rewrite the suffix."""
    if inputfile.endswith(protocol.EXTINP_SUFFIX):
        return Path(inputfile[: -len(protocol.EXTINP_SUFFIX)] + protocol.ENGRAD_SUFFIX)
    return Path(inputfile).with_suffix(protocol.ENGRAD_SUFFIX)


def _tag_for(inputfile: str) -> str:
    """Per-call correlation tag = the ORCA jobname (``.extinp.tmp`` stripped).

    ORCA reuses the same ``<job>.extinp.tmp`` for every geometry step of one
    optimisation, so this tags all calls of one structure with that structure's
    name. Backends that dump per-call artefacts (a tensor dump, say)
    then write one file per structure, holding the converged-geometry result
    (last write wins) instead of overwriting a single shared file across
    structures.
    """
    name = Path(inputfile).name
    if name.endswith(protocol.EXTINP_SUFFIX):
        return name[: -len(protocol.EXTINP_SUFFIX)]
    return Path(inputfile).stem


def _url_file(args: argparse.Namespace) -> Path:
    """The URL sidecar: ``--url-file``, else ``$WORK_DIR/server.url`` (the wrapper exports it)."""
    import os

    if args.url_file:
        return Path(args.url_file)
    return Path(os.environ.get("WORK_DIR", ".")) / SERVER_URL_FILENAME


def resolve_server_url(args: argparse.Namespace) -> str:
    """Determine the server URL from ``--bind`` (explicit) or the URL sidecar."""
    if args.bind:
        return cast(str, args.bind)
    return sidecar.read_server_url(_url_file(args))


def resolve_server_token(args: argparse.Namespace) -> str:
    """Return the per-run bearer token the server wrote beside its URL sidecar.

    An explicit ``--token-file`` wins, else the token sidecar beside the URL sidecar
    (:func:`~chemrefine.engines._backend_server.sidecar.token_file_for`) — where the server
    writes it, and where the wrapper's ``--bind`` shortcut finds it through ``$WORK_DIR``.
    The server refuses every request without it, so a missing file is refused here, by
    name, rather than sent on as a request bound to be turned away.
    """
    token_file = (
        Path(args.token_file) if args.token_file else sidecar.token_file_for(_url_file(args))
    )
    if not token_file.is_file():
        raise JobFailureError(
            f"no ExtOpt server token at {token_file}: the server writes it beside its URL when "
            f"it starts and refuses a request without it"
        )
    return sidecar.read_server_token(token_file)


def main() -> int:
    """ORCA-invoked entry point: relay one ``.extinp.tmp`` → ``.engrad`` step.

    A failure ChemRefine classifies — no reply within the timeout, a server that cannot be
    reached or refuses the request, a geometry it cannot read — exits with its one-line
    refusal on stderr and its code, as the CLI does. ORCA stops on the non-zero exit either
    way, and it passes the wrapper's stderr to the job's ``.err``, which the step's ledgered
    reason quotes: raised, that quote would end in traceback frames naming install paths
    rather than the reason. Anything else is a defect and keeps its traceback.
    """
    import sys

    args = parse_args(sys.argv[1:])
    try:
        server_url = resolve_server_url(args)
        data = protocol.read_extinp(args.inputfile, settings=settings_from_args(args))
        energy, gradient = submit_calculation(
            server_url=server_url,
            data=data,
            tag=args.tag or _tag_for(args.inputfile),
            token=resolve_server_token(args),
            timeout=args.timeout,
        )
    except ChemRefineError as e:
        print(refusal_text(e), file=sys.stderr)
        return e.exit_code
    protocol.write_engrad(
        path=_engrad_path_for(args.inputfile),
        n_atoms=len(data.symbols),
        energy_hartree=energy,
        gradients_hartree_per_bohr=gradient if data.dograd else None,
        dograd=data.dograd,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
