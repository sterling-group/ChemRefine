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
import socket
from typing import TYPE_CHECKING, Any

from chemrefine.engines._backend_server.base import (
    DEFAULT_BIND_HOST,
    DEFAULT_BIND_PORT,
    SERVER_URL_FILENAME,
    CalculationData,
    ComputeBackend,
)
from chemrefine.engines._backend_server.registry import CALCULATORS, load_calculator

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
        "--backend", required=True, choices=sorted(CALCULATORS),
        help="which ComputeBackend to load",
    )
    parser.add_argument(
        "--bind",
        default=f"{DEFAULT_BIND_HOST}:{DEFAULT_BIND_PORT}",
        help="host:port to bind (port 0 = kernel-assigned)",
    )
    parser.add_argument(
        "--url-file", default=None,
        help=f"path of the sidecar URL file (default: $WORK_DIR/{SERVER_URL_FILENAME})",
    )
    parser.add_argument("--nthreads", type=int, default=4, help="waitress worker threads")
    parser.add_argument("--log-file", default=None, help="logging destination (default: stderr)")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    for backend_name in sorted(CALCULATORS):
        load_calculator(backend_name).add_cli_args(parser)
    return parser.parse_args(argv)


def create_app(calculator: ComputeBackend) -> Flask:
    """Build a Flask app with the ``/healthz`` + ``/calculate`` routes."""
    from flask import Flask, jsonify, request

    app = Flask("chemrefine-backend-server")
    app.config["PROPAGATE_EXCEPTIONS"] = True

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify({"status": "ok", "backend": calculator.name})

    @app.post("/calculate")
    def calculate() -> Any:
        try:
            payload = request.get_json(force=True)
            data = _payload_to_data(payload)
            tag = payload.get("tag")
            if tag:
                logger.info(
                    "[req=%s] calc backend=%s n_atoms=%d nthreads=%d dograd=%s",
                    tag, calculator.name, len(data.symbols), data.nthreads, data.dograd,
                )
            energy, gradient = calculator.calc(data)
        except Exception as e:
            logger.exception("calculate failed: %s", e)
            return jsonify({"error": str(e)}), 500
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
    """Server entry point: bind, write sidecar, serve.

    Binds a real socket on the requested (possibly kernel-assigned) port,
    resolves it via ``socket.getsockname()``, writes the actual URL to the
    sidecar, then builds a waitress server **on that pre-bound socket** and
    blocks in ``server.run()``.
    """
    import os
    import sys
    from pathlib import Path

    from waitress.server import create_server

    from chemrefine.engines._backend_server import sidecar

    args = parse_args(sys.argv[1:])
    logging.basicConfig(
        level=args.log_level,
        filename=args.log_file,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )

    backend_cls = load_calculator(args.backend)
    calculator = backend_cls.from_args(args)
    app = create_app(calculator)

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
    logger.info("ExtOpt server (%s) bound at %s, sidecar=%s", args.backend, actual_url, url_file)
    # Serve on the pre-bound socket. Passing the server to ``waitress.serve``
    # would ignore this socket and start a *second* server on the default
    # 0.0.0.0:8080, so the advertised kernel port would have nothing listening.
    server = create_server(app, sockets=[sock], threads=args.nthreads)
    server.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
