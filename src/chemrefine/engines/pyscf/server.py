"""Flask/Waitress HTTP server exposing PySCF energies + gradients.

Ported in shape from the unmerged ``origin/pyscf`` PR — the verifiable
parts (CLI parsing + Flask app factory) ship in full; the model-loading
:func:`main` wrapper that calls ``waitress.serve`` stays a placeholder
because exercising a real PySCF (or gpu4pyscf) run in CI is heavy and
backend-specific.
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flask import Flask

DEFAULT_BIND = "127.0.0.1:8889"

logger = logging.getLogger(__name__)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Return the server's parsed CLI namespace.

    Flags match the v3 ``pyscf_server.py`` from ``origin/pyscf``.
    """
    parser = argparse.ArgumentParser(prog="chemrefine-pyscf-server")
    parser.add_argument("--bind", default=DEFAULT_BIND, help="bind address (host:port)")
    parser.add_argument("--nthreads", type=int, default=4, help="waitress worker threads")
    parser.add_argument("--log-file", default="pyscf_server.log", help="server log path")
    parser.add_argument(
        "--default-method", default="dft", choices=["dft", "hf"], dest="method"
    )
    parser.add_argument("--default-xc", default="pbe", dest="xc")
    parser.add_argument("--default-basis", default="def2-svp", dest="basis")
    parser.add_argument("--default-df", action="store_true", dest="df")
    parser.add_argument("--default-gpu", action="store_true", dest="gpu")
    return parser.parse_args(argv)


def defaults_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Pack the per-process method defaults that a payload may override."""
    return {
        "method": args.method,
        "xc": args.xc,
        "basis": args.basis,
        "df": bool(args.df),
        "gpu": bool(args.gpu),
    }


def create_app(defaults: dict[str, Any]) -> Flask:
    """Build a Flask app with the ``/calculate`` route installed.

    The route is wired but the actual ``run_calc`` placeholder must be
    ported before the server can serve real requests — see
    :func:`run_calc`.
    """
    from flask import Flask, jsonify, request

    app = Flask("chemrefine-pyscf-server")
    app.config["PROPAGATE_EXCEPTIONS"] = True

    @app.post("/calculate")
    def calculate() -> Any:
        try:
            payload = request.get_json(force=True)
            energy, gradient, meta = run_calc(payload, defaults)
        except NotImplementedError as e:
            logger.warning("calculate: %s", e)
            return jsonify({"error": str(e)}), 501
        except Exception as e:  # pragma: no cover - generic 500 path unreachable until run_calc is ported in B6
            logger.exception("calculate failed: %s", e)
            return jsonify({"error": str(e)}), 500
        return jsonify({"energy": energy, "gradient": gradient, "meta": meta})  # pragma: no cover - reached after B6 ports run_calc

    return app


def run_calc(
    payload: dict[str, Any], defaults: dict[str, Any]
) -> tuple[float, list[list[float]], dict[str, Any]]:
    """Build a PySCF :class:`gto.Mole`, run RKS/UKS, return ``(E, grad, meta)``.

    TODO: port from ``origin/pyscf/src/chemrefine/pyscf_server.py``.
    The reference implements:

    * Å → Bohr conversion and ``mol.spin = mult - 1`` for UKS dispatch.
    * Defaults overridable via ``payload['settings']``.
    * GPU dispatch via ``gpu4pyscf.dft.{RKS,UKS}`` when ``settings['gpu']``.

    Verifying needs PySCF + (optionally) gpu4pyscf installed; left as
    a placeholder so the rest of the module can ship verified.
    """
    raise NotImplementedError(
        "PySCF server run_calc not yet ported — see TODO in engines/pyscf/server.py"
    )


def main() -> int:  # pragma: no cover - placeholder until B1 wires waitress
    """Server entry point — placeholder.

    TODO: parse argv, build the app via :func:`create_app`, and serve
    with waitress on ``args.bind`` / ``args.nthreads``.
    """
    raise NotImplementedError(
        "PySCF server main not yet ported — see TODO in engines/pyscf/server.py"
    )
