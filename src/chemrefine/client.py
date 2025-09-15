#!/usr/bin/env python3

from __future__ import annotations

# Timings including imports
import time
start_time = time.perf_counter()

import requests
import sys
import argparse
import traceback
from chemrefine import utils_extopt as common

end_import = time.perf_counter()


def parse_extended_args(arglist):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=str, default="127.0.0.1:8888")
    parser.add_argument("inputfile")
    return parser.parse_args(arglist)


def submit_uma(server_url: str,
               atom_types: list[str],
               coordinates: list[tuple[float, float, float]],
               charge: int,
               mult: int,
               dograd: bool,
               nthreads: int
               ) -> tuple[float, list[float]]:
    """
    Sends an UMA calculation to the server and returns the result.
    """
    payload = {
        "atom_types": atom_types,
        "coordinates": coordinates,
        "mult": mult,
        "charge": charge,
        "dograd": dograd,
        "nthreads": nthreads
    }

    try:
        response = requests.post(f'http://{server_url}/calculate', json=payload)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as http_err:
        print("HTTP error occurred:", http_err)
        print("The server is probably not running.")
        print("Please start the server with the umaserver.sh script.")
        traceback.print_exc()
        sys.exit(1)
    except requests.exceptions.ConnectionError as conn_err:
        print("Connection error: could not reach the server.")
        print("Details:", conn_err)
        traceback.print_exc()
        sys.exit(1)
    except requests.exceptions.Timeout as timeout_err:
        print("Request to UMA server timed out:", timeout_err)
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

    energy = data["energy"]
    gradient = data["gradient"]
    return energy,gradient


def run(arglist: list[str]):
    """Run a calculation on a given structure using the UMA server."""
    args = parse_extended_args(arglist)

    # Read the ORCA-generated input
    xyzname, charge, mult, ncores, dograd = common.read_input(args.inputfile)

    # Prepare filenames
    basename = xyzname.rstrip(".xyz")
    orca_engrad = basename + ".engrad"

    # Load geometry
    atom_types, coordinates = common.read_xyzfile(xyzname)
    natoms = len(atom_types)

    # Submit job to the UMA server
    energy, gradient = submit_uma(
        server_url=args.bind,
        atom_types=atom_types,
        coordinates=coordinates,
        charge=charge,
        mult=mult,
        dograd=dograd,
        nthreads=ncores
    )

    # Save result
    common.write_engrad(orca_engrad, natoms, energy, dograd, gradient)

    print(f"Total time: {time.perf_counter() - start_time:.3f} seconds")


def main():
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
