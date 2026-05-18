"""Flask/Waitress HTTP server that serves MLFF energies + gradients.

A single :class:`~chemrefine.engines.mlff.calculator.MlffCalculator` is
constructed per worker thread (key: ``threading.get_ident()``) so model
weights are loaded once and shared across optimisation steps that hit
the same thread.

The verifiable parts ship in full:

* :func:`parse_args` — server CLI surface.
* :func:`create_app` — builds the Flask app with the ``/calculate``
  route installed. Tests can hit the route with a mocked calculator.

The :func:`main` wrapper that actually loads the model and calls
``waitress.serve`` is a placeholder until we have a CI-friendly way to
exercise a real MLFF backend.
"""

from __future__ import annotations

import argparse
import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flask import Flask

DEFAULT_BIND = "127.0.0.1:8888"

logger = logging.getLogger(__name__)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Return the server's parsed CLI namespace.

    Flags match the v3 ``server.py`` from ``main``.
    """
    parser = argparse.ArgumentParser(prog="chemrefine-mlff-server")
    parser.add_argument("--model", help="pretrained MLFF model name (e.g. uma-s-1 or a MACE preset)")
    parser.add_argument("--task-name", default="omol", help="task name for pretrained models")
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    parser.add_argument("--bind", default=DEFAULT_BIND, help="bind address (host:port)")
    parser.add_argument("--nthreads", type=int, default=4, help="waitress worker threads")
    parser.add_argument("--model-path", default=None, help="custom MACE model checkpoint path")
    args = parser.parse_args(argv)
    if not args.model and not args.model_path:
        parser.error("either --model or --model-path is required")
    return args


def create_app(
    *,
    model_name: str | None,
    task_name: str,
    device: str,
    model_path: str | None = None,
) -> Flask:
    """Build a Flask app with the ``/calculate`` route installed.

    Calculators are cached per OS thread so concurrent requests reuse
    loaded model weights instead of re-initialising them.
    """
    from flask import Flask, jsonify, request

    from chemrefine.engines.mlff.calculator import MlffCalculator

    app = Flask("chemrefine-mlff-server")
    app.config["PROPAGATE_EXCEPTIONS"] = True
    calculators: dict[int, MlffCalculator] = {}

    @app.post("/calculate")
    def calculate() -> Any:
        try:
            payload = request.get_json(force=True)
            from ase import Atoms

            atoms = Atoms(
                symbols=payload["atom_types"], positions=payload["coordinates"]
            )
            atoms.info = {"charge": payload["charge"], "spin": payload["mult"]}

            tid = threading.get_ident()
            if tid not in calculators:
                calculators[tid] = MlffCalculator(
                    model_name=model_name or "",
                    task_name=task_name,
                    device=device,
                    model_path=model_path,
                )
            energy, gradient = calculators[tid].single_point(atoms)
            return jsonify({"energy": float(energy), "gradient": gradient})
        except Exception as e:
            logger.exception("calculate failed: %s", e)
            return jsonify({"error": str(e)}), 500

    return app


def main() -> int:
    """Server entry point — loads the model and starts waitress.

    TODO: wire :func:`create_app` to :mod:`waitress.serve` once we have
    a CI-friendly way to exercise a real model. The shape will be::

        args = parse_args(sys.argv[1:])
        app = create_app(
            model_name=args.model, task_name=args.task_name,
            device=args.device, model_path=args.model_path,
        )
        import waitress
        waitress.serve(app, listen=args.bind, threads=args.nthreads)
        return 0
    """
    raise NotImplementedError(
        "MLFF server main not yet ported — see TODO in engines/mlff/server.py"
    )
