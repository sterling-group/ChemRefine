#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import threading
from typing import Any

import waitress
from flask import Flask, request, jsonify

from pyscf import gto, scf, dft, lib

BOHR_PER_ANG = 1.0 / 0.529177210903  # Å -> Bohr


def parse_server_args(arglist):
    p = argparse.ArgumentParser(description="Start PySCF external-method server")
    p.add_argument("--bind", default="127.0.0.1:8888", help="Bind address (default: 127.0.0.1:8888)")
    p.add_argument("--nthreads", type=int, default=4, help="Waitress threads (default: 4)")
    # Optional defaults (can be overridden per request by client settings)
    p.add_argument("--default-method", default="dft", choices=["dft", "hf"])
    p.add_argument("--default-xc", default="pbe")
    p.add_argument("--default-basis", default="def2-svp")
    p.add_argument("--default-df", action="store_true")
    p.add_argument("--default-gpu", action="store_true")
    return p.parse_args(arglist)


app = Flask("pyscfserver")
app.config["PROPAGATE_EXCEPTIONS"] = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyscfserver")


# Store per-thread “GPU enabled?” cached decision, etc. (optional)
thread_state: dict[int, dict[str, Any]] = {}


def _try_enable_gpu():
    """
    Best-effort GPU enablement using gpu4pyscf, if installed.
    This is intentionally conservative; if not available, we fall back to CPU.
    """
    try:
        import gpu4pyscf  # noqa: F401
        return True
    except Exception:
        return False


def build_mol(atom_types, coords_ang, charge, mult, basis) -> gto.Mole:
    # Convert to Bohr to make gradient units unambiguous (Eh/Bohr)
    coords_bohr = [[x * BOHR_PER_ANG, y * BOHR_PER_ANG, z * BOHR_PER_ANG] for (x, y, z) in coords_ang]
    atom = [(sym, tuple(r)) for sym, r in zip(atom_types, coords_bohr)]

    spin = int(mult) - 1  # mult = 2S+1 -> spin = 2S
    mol = gto.Mole()
    mol.atom = atom
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.basis = basis
    mol.unit = "Bohr"
    mol.build()
    return mol


def run_calc(payload: dict) -> tuple[float, list[float]]:
    atom_types = payload["atom_types"]
    coords = payload["coordinates"]  # Angstrom
    charge = payload["charge"]
    mult = payload["mult"]
    dograd = bool(payload.get("dograd", True))
    nthreads = int(payload.get("nthreads", 1))

    settings = payload.get("settings") or {}
    method = (settings.get("method") or DEFAULTS["method"]).lower()
    xc = settings.get("xc") or DEFAULTS["xc"]
    basis = settings.get("basis") or DEFAULTS["basis"]
    use_df = bool(settings.get("df", DEFAULTS["df"]))
    use_gpu = bool(settings.get("gpu", DEFAULTS["gpu"]))

    # Thread control inside PySCF (separate from waitress threads)
    lib.num_threads(nthreads)

    mol = build_mol(atom_types, coords, charge, mult, basis)

    if method == "hf":
        mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
    else:
        mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
        mf.xc = xc

    if use_df:
        mf = mf.density_fit()

    # GPU best-effort
    if use_gpu:
        if _try_enable_gpu():
            try:
                # gpu4pyscf provides .to_gpu() on many mf objects
                mf = mf.to_gpu()
            except Exception:
                logger.warning("gpu4pyscf installed but .to_gpu() failed; falling back to CPU.")
        else:
            logger.warning("Requested --gpu but gpu4pyscf is not installed; running on CPU.")

    energy = float(mf.kernel())
    if not mf.converged:
        logger.warning("SCF not converged (continuing anyway).")

    if not dograd:
        return energy, []

    # Gradients in Eh/Bohr (since mol.unit='Bohr')
    g = mf.nuc_grad_method().kernel()  # shape (natoms, 3)
    grad_flat = [float(x) for x in g.reshape(-1)]
    return energy, grad_flat


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        payload = request.get_json()
        energy, gradient = run_calc(payload)
        return jsonify({"energy": energy, "gradient": gradient})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


DEFAULTS = {"method": "dft", "xc": "pbe", "basis": "def2-svp", "df": False, "gpu": False}


def run(arglist: list[str]):
    args = parse_server_args(arglist)

    DEFAULTS["method"] = args.default_method
    DEFAULTS["xc"] = args.default_xc
    DEFAULTS["basis"] = args.default_basis
    DEFAULTS["df"] = bool(args.default_df)
    DEFAULTS["gpu"] = bool(args.default_gpu)

    logger.info(
        "Starting PySCF server on %s (waitress: %d threads). Defaults: method=%s xc=%s basis=%s df=%s gpu=%s",
        args.bind, args.nthreads, DEFAULTS["method"], DEFAULTS["xc"], DEFAULTS["basis"], DEFAULTS["df"], DEFAULTS["gpu"]
    )
    waitress.serve(app, listen=args.bind, threads=args.nthreads)


def main():
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
