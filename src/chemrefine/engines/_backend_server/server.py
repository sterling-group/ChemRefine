"""Shared Flask + waitress backend compute server.

Dispatches to one backend per process. The :func:`parse_args` CLI is
**backend-agnostic** at the shared layer; each registered backend
contributes its own flags via
:meth:`ComputeBackend.add_cli_args`. Kernel-assigned ports
(``--bind 127.0.0.1:0``) are the default so multiple SLURM jobs on
the same node never collide on a hardcoded port; the actual
``host:port`` is recorded to a sidecar URL file so the wrapper
script can find it.

The :func:`main` glue (binding waitress, writing the sidecar, blocking
on requests) is the only path not exercised by tests — it lives on
the production side of the import boundary.
"""

from __future__ import annotations

import argparse
import logging
import secrets
import socket
from typing import TYPE_CHECKING, Any

from chemrefine.engines._backend_server.base import (
    DEFAULT_BIND_HOST,
    DEFAULT_BIND_PORT,
    SERVER_TOKEN_FILENAME,
    SERVER_URL_FILENAME,
    CalculationData,
    ComputeBackend,
)
from chemrefine.engines._backend_server.registry import known_backends, load_calculator

if TYPE_CHECKING:
    from flask import Flask

logger = logging.getLogger(__name__)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Return the shared server's parsed CLI namespace.

    Backend-specific flags come from each registered backend's
    :meth:`ComputeBackend.add_cli_args`; the shared layer here
    owns only the generic flags every backend shares.
    """
    parser = argparse.ArgumentParser(prog="chemrefine-backend-server")
    parser.add_argument(
        "--backend",
        required=True,
        choices=known_backends(),
        help="which ComputeBackend to load",
    )
    parser.add_argument(
        "--bind",
        default=f"{DEFAULT_BIND_HOST}:{DEFAULT_BIND_PORT}",
        help="host:port to bind (port 0 = kernel-assigned)",
    )
    parser.add_argument(
        "--url-file",
        default=None,
        help=f"path of the sidecar URL file (default: $WORK_DIR/{SERVER_URL_FILENAME})",
    )
    parser.add_argument("--nthreads", type=int, default=4, help="waitress worker threads")
    parser.add_argument("--log-file", default=None, help="logging destination (default: stderr)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    for backend_name in known_backends():
        load_calculator(backend_name).add_cli_args(parser)
    return parser.parse_args(argv)


def create_app(calculator: ComputeBackend, *, token: str | None = None) -> Flask:
    """Build a Flask app with the ``/healthz`` + ``/calculate`` routes.

    ``token`` is the per-run bearer secret: when set, ``/calculate`` rejects
    any request whose ``Authorization`` header doesn't carry it (``/healthz``
    stays open for the run_block's readiness curl). ``None`` disables the
    check — a unit-test affordance; :func:`main` always passes a token.

    The return annotation resolves under ``TYPE_CHECKING`` only — ``flask``
    is still imported lazily inside, not at module load, because the server
    deps are an optional extra.
    """
    from flask import Flask, jsonify, request

    app = Flask("chemrefine-backend-server")
    app.config["PROPAGATE_EXCEPTIONS"] = True

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify({"status": "ok", "backend": calculator.name})

    @app.post("/calculate")
    def calculate() -> Any:
        # Compared as bytes: Werkzeug decodes headers as latin-1, and `compare_digest`
        # raises TypeError on a str holding a non-ASCII character — so a header with any
        # byte above 0x7F turned a plain 401 into a 500 and a logged traceback. Encoding
        # both sides answers every input in constant time instead of the one class of
        # wrong token we happened to think of.
        if token is not None and not secrets.compare_digest(
            request.headers.get("Authorization", "").encode(), f"Bearer {token}".encode()
        ):
            return jsonify({"error": "unauthorized"}), 401
        try:
            payload = request.get_json(force=True)
            data = _payload_to_data(payload)
            tag = payload.get("tag")
            if tag:
                logger.info(
                    "[req=%s] calc backend=%s n_atoms=%d nthreads=%d dograd=%s",
                    tag,
                    calculator.name,
                    len(data.symbols),
                    data.nthreads,
                    data.dograd,
                )
            energy, gradient = calculator.calc(data)
        except Exception as e:
            # The full exception — message, paths, backend internals — goes to the
            # server log, which only the job owner can read. The response carries a
            # correlation id instead: the client is ORCA's wrapper script, which does
            # nothing with the text beyond surfacing it, and the server is reachable
            # by any same-host user who gets hold of the token.
            request_id = secrets.token_hex(8)
            logger.exception("[req=%s] calculate failed: %s", request_id, e)
            return jsonify(
                {
                    "error": f"backend calculation failed (request {request_id}); "
                    f"see the ExtOpt server log for details"
                }
            ), 500
        return jsonify({"energy": float(energy), "gradient": gradient})

    return app


def _payload_to_data(payload: dict[str, Any]) -> CalculationData:
    """Build a :class:`CalculationData` from a wrapper-script POST payload.

    The top-level ``tag`` (the bridge's per-call correlation id, derived
    from the ``.extinp.tmp`` stem) is folded into ``settings['tag']`` so
    backends that key per-call artefacts on it — e.g. PySCF active-space
    tensor dumps — can read it through the same ``settings`` channel as
    every other knob.
    """
    import numpy as np

    settings = dict(payload.get("settings", {}))
    tag = payload.get("tag")
    if tag is not None:
        settings.setdefault("tag", tag)

    return CalculationData(
        symbols=tuple(payload["atom_types"]),
        positions_angstrom=np.asarray(payload["coordinates"], dtype=float),
        charge=int(payload["charge"]),
        multiplicity=int(payload["mult"]),
        nthreads=int(payload.get("nthreads", 1)),
        dograd=bool(payload.get("dograd", True)),
        settings=settings,
    )


def main() -> int:
    """Server entry point: bind, write sidecars, serve.

    Binds a real socket on the requested (possibly kernel-assigned) port,
    resolves it via ``socket.getsockname()``, writes the actual URL plus a
    freshly generated per-run bearer token to the sidecars, then builds a
    waitress server **on that pre-bound socket** and blocks in
    ``server.run()``. The token gates ``/calculate`` so other users on the
    same node can't drive this server.
    """
    import importlib.util
    import os
    import sys
    from pathlib import Path

    from chemrefine.engines._backend_server import sidecar

    args = parse_args(sys.argv[1:])
    logging.basicConfig(
        level=args.log_level,
        filename=args.log_file,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )

    # Probed after logging is configured and before anything imports them, so a missing
    # server stack is an actionable line in the ``--log-file`` the run block tails on
    # failure. Importing first crashed before the log file existed, and the job reported
    # ``cat: …server_….log: No such file or directory`` instead of the reason — with the
    # traceback stranded in the job's ``.err``, which nothing pointed at.
    #
    # A module already in ``sys.modules`` counts as importable — ``import`` would return
    # it — and must not reach ``find_spec``, which raises on one whose ``__spec__`` is
    # ``None`` (exactly what a test injecting a fake module leaves there).
    missing = [
        name
        for name in ("flask", "waitress")
        if name not in sys.modules and importlib.util.find_spec(name) is None
    ]
    if missing:
        logger.error(
            "cannot start: %s not importable in %s. Install `chemrefine[server]` into "
            "this environment, or provision the backend in its own managed env with "
            "`chemrefine backends install <extra>`.",
            " and ".join(missing),
            sys.executable,
        )
        return 1

    from waitress.server import create_server

    backend_cls = load_calculator(args.backend)
    calculator = backend_cls.from_args(args)
    token = secrets.token_hex(32)
    app = create_app(calculator, token=token)

    host, port_str = args.bind.rsplit(":", 1)
    # Bind a real OS socket first so getsockname() reveals the
    # kernel-assigned ephemeral port (when port=0). waitress's
    # documented ``sockets=`` parameter then accepts the pre-bound
    # socket directly.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, int(port_str)))
    actual_host, actual_port = sock.getsockname()
    actual_url = f"{actual_host}:{actual_port}"

    default_dir = Path(os.environ.get("WORK_DIR", "."))
    url_file = args.url_file or str(default_dir / SERVER_URL_FILENAME)
    sidecar.write_server_url(url_file, actual_url)
    sidecar.write_server_token(Path(url_file).with_name(SERVER_TOKEN_FILENAME), token)
    logger.info("ExtOpt server (%s) bound at %s, sidecar=%s", args.backend, actual_url, url_file)
    # Serve on the pre-bound socket. Passing the server to ``waitress.serve``
    # would ignore this socket and start a *second* server on the default
    # 0.0.0.0:8080, so the advertised kernel port would have nothing listening.
    server = create_server(app, sockets=[sock], threads=args.nthreads)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
