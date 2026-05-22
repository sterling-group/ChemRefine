"""Shared Flask + waitress ExtOpt server.

Dispatches to one backend per process. The :func:`parse_args` CLI is
the union of every backend's options; each backend's
:meth:`BaseExtOptCalculator.from_args` picks only the fields it cares
about. Kernel-assigned ports (``--bind 127.0.0.1:0``) are the default
so multiple SLURM jobs on the same node never collide on a hardcoded
port; the actual ``host:port`` is recorded to a sidecar URL file so
the wrapper script can find it.

The :func:`main` glue (binding waitress, writing the sidecar, blocking
on requests) is the only path not exercised by tests — it lives on
the production side of the import boundary.
"""

from __future__ import annotations

import argparse
import logging
import socket
from typing import TYPE_CHECKING, Any

from chemrefine.engines._extopt.base import (
    DEFAULT_BIND_HOST,
    SERVER_URL_FILENAME,
    BaseExtOptCalculator,
    CalculationData,
)
from chemrefine.engines._extopt.registry import CALCULATORS, load_calculator

if TYPE_CHECKING:
    from flask import Flask

logger = logging.getLogger(__name__)

DEFAULT_BIND_PORT: int = 0
"""``0`` asks the kernel for any free ephemeral port (collision-safe)."""


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Return the shared server's parsed CLI namespace.

    Backend-specific flags (``--model``, ``--xc``, ...) are all defined
    here; each backend's :meth:`from_args` reads only what it needs.
    """
    parser = argparse.ArgumentParser(prog="chemrefine-extopt-server")
    parser.add_argument(
        "--backend", required=True, choices=sorted(CALCULATORS),
        help="which BaseExtOptCalculator to load",
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
    # MLFF-specific
    parser.add_argument("--model", default=None, help="MLFF pretrained model name")
    parser.add_argument("--task-name", default="omol", help="MLFF task name")
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    parser.add_argument("--model-path", default=None, help="custom MACE model checkpoint")
    # PySCF-specific
    parser.add_argument("--method", default="dft", choices=["dft", "hf"])
    parser.add_argument("--xc", default="pbe", help="DFT exchange-correlation functional")
    parser.add_argument("--basis", default="def2-svp", help="orbital basis set")
    parser.add_argument("--df", action="store_true", help="enable density fitting / RI")
    parser.add_argument("--gpu", action="store_true", help="attempt gpu4pyscf if installed")
    return parser.parse_args(argv)


def create_app(calculator: BaseExtOptCalculator) -> Flask:
    """Build a Flask app with the ``/healthz`` + ``/calculate`` routes."""
    from flask import Flask, jsonify, request

    app = Flask("chemrefine-extopt-server")
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
    """Build a :class:`CalculationData` from a wrapper-script POST payload."""
    import numpy as np

    return CalculationData(
        symbols=tuple(payload["atom_types"]),
        positions_angstrom=np.asarray(payload["coordinates"], dtype=float),
        charge=int(payload["charge"]),
        multiplicity=int(payload["mult"]),
        nthreads=int(payload.get("nthreads", 1)),
        dograd=bool(payload.get("dograd", True)),
        settings=dict(payload.get("settings", {})),
    )


def main() -> int:
    """Server entry point: bind, write sidecar, serve.

    Resolves the kernel-assigned port via ``socket.getsockname()`` after
    waitress creates its server object, writes the actual URL to the
    sidecar, then blocks in ``waitress.serve``.
    """
    import os
    import sys
    from pathlib import Path

    import waitress
    from waitress.server import create_server

    from chemrefine.engines._extopt import protocol

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
    protocol.write_server_url(url_file, actual_url)
    logger.info("ExtOpt server (%s) bound at %s, sidecar=%s", args.backend, actual_url, url_file)
    server = create_server(app, sockets=[sock], threads=args.nthreads)
    waitress.serve(server)
    return 0
