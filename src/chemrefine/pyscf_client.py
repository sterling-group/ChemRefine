#!/usr/bin/env python3
from __future__ import annotations

import time
start_time = time.perf_counter()

import argparse
import requests
import sys
import traceback

from chemrefine import utils_extopt as common


def parse_extended_args(arglist):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=str, default="127.0.0.1:8888")

    # level of theory / basis
    parser.add_argument("--method", type=str, default="dft", choices=["dft", "hf"])
    parser.add_argument("--xc", type=str, default="pbe")
    parser.add_argument("--basis", type=str, default="def2-svp")
    parser.add_argument("--df", action="store_true", help="Use density fitting / RI")

    # GPU toggle
    parser.add_argument("--gpu", action="store_true", help="Attempt gpu4pyscf (if installed)")

    # optional tag to make server logs readable (otherwise basename is used)
    parser.add_argument("--tag", type=str, default=None, help="Tag for server log (default: xyz basename)")

    parser.add_argument("inputfile")
    return parser.parse_args(arglist)




def submit_pyscf(
    server_url: str,
    atom_types: list[str],
    coordinates: list[tuple[float, float, float]],  # Angstrom
    charge: int,
    mult: int,   # ORCA multiplicity (2S+1)
    dograd: bool,
    nthreads: int,
    settings: dict,
    tag: str | None = None,
) -> tuple[float, list[float]]:
    payload = {
        "atom_types": atom_types,
        "coordinates": coordinates,
        "charge": charge,
        "mult": mult,
        "dograd": dograd,
        "nthreads": nthreads,
        "settings": settings,
        "tag": tag,
    }

    try:
        r = requests.post(f"http://{server_url}/calculate", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.HTTPError as http_err:
        print("HTTP error occurred:", http_err)
        print("Server response:", getattr(r, "text", "<no body>"))
        print("The server is probably not running or crashed.")
        traceback.print_exc()
        sys.exit(1)
    except requests.exceptions.ConnectionError as conn_err:
        print("Connection error: could not reach the server.")
        print("Details:", conn_err)
        traceback.print_exc()
        sys.exit(1)
    except requests.exceptions.Timeout as timeout_err:
        print("Request to PySCF server timed out:", timeout_err)
        traceback.print_exc()
        sys.exit(1)
    except requests.exceptions.RequestException as req_err:
        print("General request error:", req_err)
        traceback.print_exc()
        sys.exit(1)
    except Exception as err:
        print("Unexpected error occurred:", err)
        traceback.print_exc()
        sys.exit(1)

    if "error" in data:
        print("Server returned an error:", data["error"])
        sys.exit(1)

    return float(data["energy"]), list(data["gradient"])


def run(arglist: list[str]):
    args = parse_extended_args(arglist)

    # ORCA-generated external-tool input
    xyzname, charge, mult, ncores, dograd = common.read_input(args.inputfile)

    basename = xyzname.rstrip(".xyz")
    orca_engrad = basename + ".engrad"

    atom_types, coordinates = common.read_xyzfile(xyzname)
    natoms = len(atom_types)

    settings = {
    "method": args.method,
    "xc": args.xc,
    "basis": args.basis,
    "df": bool(args.df),
    "gpu": bool(args.gpu),
}
    tag = args.tag or basename
    energy, gradient = submit_pyscf(
        server_url=args.bind,
        atom_types=atom_types,
        coordinates=coordinates,
        charge=charge,
        mult=mult,
        dograd=dograd,
        nthreads=ncores,
        settings=settings,
        tag=tag,
    )

    common.write_engrad(orca_engrad, natoms, energy, dograd, gradient)
    print(f"Total time: {time.perf_counter() - start_time:.3f} seconds")


def main():
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
