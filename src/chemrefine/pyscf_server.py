#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, Dict, Tuple

import waitress
from flask import Flask, request, jsonify

from pyscf import gto, scf, dft, lib

BOHR_PER_ANG = 1.0 / 0.529177210903  # Å -> Bohr


def parse_server_args(arglist):
    p = argparse.ArgumentParser(description="Start PySCF external-method server (ORCA ProgExt)")
    p.add_argument("--bind", default="127.0.0.1:8888", help="Bind address (default: 127.0.0.1:8888)")
    p.add_argument("--nthreads", type=int, default=4, help="Waitress threads (default: 4)")
    p.add_argument("--log-file", default="pyscf_server.log", help="Write server log to this file")

    # Defaults (can be overridden per-request via payload['settings'])
    p.add_argument("--default-method", default="dft", choices=["dft", "hf"])
    p.add_argument("--default-xc", default="pbe")
    p.add_argument("--default-basis", default="def2-svp")
    p.add_argument("--default-df", action="store_true")
    p.add_argument("--default-gpu", action="store_true")

    return p.parse_args(arglist)


app = Flask("pyscfserver")
app.config["PROPAGATE_EXCEPTIONS"] = True

DEFAULTS: Dict[str, Any] = {
    "method": "dft",
    "xc": "pbe",
    "basis": "def2-svp",
    "df": False,
    "gpu": False,
}

logger = logging.getLogger("pyscfserver")


def _gpu_dft_classes():
    """
    Import GPU4PySCF DFT classes (the same API that worked for you).
    """
    from gpu4pyscf.dft import RKS as GPU_RKS  # type: ignore
    from gpu4pyscf.dft import UKS as GPU_UKS  # type: ignore
    return GPU_RKS, GPU_UKS


def _build_mol(atom_types, coords_ang, charge: int, mult: int, basis: str) -> gto.Mole:
    # Convert Å -> Bohr and set unit=Bohr so gradients are Eh/Bohr
    coords_bohr = [[x * BOHR_PER_ANG, y * BOHR_PER_ANG, z * BOHR_PER_ANG] for (x, y, z) in coords_ang]
    atom = [(sym, tuple(r)) for sym, r in zip(atom_types, coords_bohr)]

    spin = int(mult) - 1  # mult=2S+1 -> spin=2S

    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "Bohr"
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.basis = basis
    mol.build()
    return mol


def run_calc(payload: Dict[str, Any]) -> Tuple[float, list[float], Dict[str, Any]]:
    t0 = time.perf_counter()

    tag = payload.get("tag", "untagged")
    atom_types = payload["atom_types"]
    coords = payload["coordinates"]  # Angstrom
    charge = int(payload["charge"])
    mult = int(payload["mult"])
    dograd = bool(payload.get("dograd", True))
    nthreads = int(payload.get("nthreads", 1))

    settings = payload.get("settings") or {}
    method = str(settings.get("method", DEFAULTS["method"])).lower()
    xc = str(settings.get("xc", DEFAULTS["xc"]))
    basis = str(settings.get("basis", DEFAULTS["basis"]))
    use_df = bool(settings.get("df", DEFAULTS["df"]))
    want_gpu = bool(settings.get("gpu", DEFAULTS["gpu"]))

    # PySCF threading
    lib.num_threads(nthreads)

    mol = _build_mol(atom_types, coords, charge, mult, basis)
    closed_shell = (mol.spin == 0)

    gpu_used = False
    gpu_msg = ""

    # Build MF object
    if method == "hf":
        # GPU HF is not handled here (unreliable); always CPU
        mf = scf.RHF(mol) if closed_shell else scf.UHF(mol)
        if want_gpu:
            gpu_msg = "GPU requested but HF GPU path not enabled; using CPU HF"
    else:
        # DFT: can use GPU4PySCF by constructing GPU_RKS/UKS directly
        if want_gpu:
            try:
                GPU_RKS, GPU_UKS = _gpu_dft_classes()
                mf = GPU_RKS(mol) if closed_shell else GPU_UKS(mol)
                gpu_used = True
                gpu_msg = "GPU4PySCF DFT backend (gpu4pyscf.dft.RKS/UKS)"
            except Exception as e:
                mf = dft.RKS(mol) if closed_shell else dft.UKS(mol)
                gpu_used = False
                gpu_msg = f"Failed to init GPU4PySCF classes; fell back to CPU DFT ({e})"
        else:
            mf = dft.RKS(mol) if closed_shell else dft.UKS(mol)

        mf.xc = xc

    # Density fitting
    if use_df:
        try:
            mf = mf.density_fit()
        except Exception as e:
            # DF not always supported on GPU objects depending on version; fall back gracefully
            logger.warning("tag=%s density_fit() failed (%s). Continuing without DF.", tag, e)

    # Solve
    energy = float(mf.kernel())
    converged = bool(getattr(mf, "converged", False))

    grad_flat: list[float] = []
    grad_norm = 0.0
    if dograd:
        g = mf.nuc_grad_method().kernel()  # Eh/Bohr (mol.unit=Bohr)
        grad_flat = [float(x) for x in g.reshape(-1)]
        grad_norm = float((g * g).sum() ** 0.5)

    dt = time.perf_counter() - t0

    meta = {
        "tag": tag,
        "method": method,
        "xc": xc,
        "basis": basis,
        "df": use_df,
        "gpu_requested": want_gpu,
        "gpu_used": gpu_used,
        "gpu_msg": gpu_msg,
        "converged": converged,
        "energy_eh": energy,
        "grad_len": len(grad_flat),
        "grad_norm": grad_norm,
        "time_s": dt,
        "nthreads": nthreads,
    }
    return energy, grad_flat, meta


@app.route("/calculate", methods=["POST"])
def calculate():
    payload = request.get_json() or {}
    tag = payload.get("tag", "untagged")
    settings = payload.get("settings", {}) or {}

    logger.info("REQ tag=%s settings=%s", tag, settings)

    try:
        energy, gradient, meta = run_calc(payload)

        logger.info(
            "DONE tag=%s E(Eh)=%.12f conv=%s df=%s gpu_req=%s gpu_used=%s |grad|=%g t=%.3fs (%s)",
            meta["tag"],
            meta["energy_eh"],
            meta["converged"],
            meta["df"],
            meta["gpu_requested"],
            meta["gpu_used"],
            meta["grad_norm"],
            meta["time_s"],
            meta["gpu_msg"],
        )

        if meta["gpu_requested"] and not meta["gpu_used"]:
            logger.warning(
                "GPU requested but not used for tag=%s (%s)",
                meta["tag"],
                meta["gpu_msg"],
            )

        return jsonify({"energy": energy, "gradient": gradient})

    except Exception:
        import traceback
        tb = traceback.format_exc()

        logger.error("CALC FAILED tag=%s\n%s", tag, tb)
        print(f"[SERVER] CALC FAILED tag={tag}\n{tb}", file=sys.stderr, flush=True)

        # Returning tb is useful for debugging; you can remove it once stable
        return jsonify({"error": "calculation failed", "traceback": tb}), 500



def run(arglist: list[str]):
    args = parse_server_args(arglist)

    # Logging to stdout + file
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, mode="a"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )

    DEFAULTS["method"] = args.default_method
    DEFAULTS["xc"] = args.default_xc
    DEFAULTS["basis"] = args.default_basis
    DEFAULTS["df"] = bool(args.default_df)
    DEFAULTS["gpu"] = bool(args.default_gpu)

    logger.info("Starting PySCF server on %s | waitress_threads=%d | defaults=%s", args.bind, args.nthreads, DEFAULTS)

    waitress.serve(app, listen=args.bind, threads=args.nthreads)


def main():
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
