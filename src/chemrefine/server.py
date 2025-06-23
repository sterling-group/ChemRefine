from __future__ import annotations

import logging
import sys
import threading
from typing import Callable
from ase import Atoms

import torch
import waitress
from flask import Flask, request, jsonify

from chemrefine import utils_extopt as common
from chemrefine.mlff import MLFFCalculator
import argparse

def parse_server_args(arglist):
    parser = argparse.ArgumentParser(description="Start UMA MLFF server")
    parser.add_argument("--model", required=True, help="Model name to use (e.g., uma-s-1)")
    parser.add_argument("--task-name", default="omol", help="Task name (default: omol)")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--bind", default="127.0.0.1:8888", help="Bind address (default: 127.0.0.1:8888)")
    parser.add_argument("--nthreads", type=int, default=4, help="Number of threads (default: 4)")
    return parser.parse_args(arglist)

app = Flask('umaserver')
app.config["PROPAGATE_EXCEPTIONS"] = True
app.debug = True  # Optional: Enables debug mode for Flask

# Add logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("waitress")
logger.setLevel(logging.DEBUG)


model: str = ''  # will hold the selected model

calculators: dict[int | Callable] = {}  # will hold one UMACalculator per server thread


@app.route('/calculate', methods=['POST'])
def run_uma():
    try:
        print("[SERVER DEBUG] /calculate called")
        input = request.get_json()
        print("[SERVER DEBUG] Received JSON:", input)

        atoms = Atoms(symbols=input["atom_types"], positions=input["coordinates"])
        atoms.info = {"charge": input["charge"], "spin": input["mult"]}

        nthreads = input.get('nthreads', 1)
        torch.set_num_threads(nthreads)

        thread_id = threading.get_ident()
        global calculators
        if thread_id not in calculators:
            # FIX THIS LINE IF NEEDED
            calculators[thread_id] = MLFFCalculator(
                model_name=model, task_name="omol", device="cuda"
            )

        calc = calculators[thread_id]
        atoms.calc = calc.calculator
        energy, gradient = common.process_output(atoms)

        return jsonify({'energy': energy, 'gradient': gradient})

    except Exception as e:
        import traceback
        print("[SERVER ERROR] Exception occurred:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



def run(arglist: list[str]):
    """Start the UMA calculation server using a specified model file."""
    args = parse_server_args(arglist)

    # get the absolute path of the model file as a plain string
    global model
    model = str(args.model)

    # set up logging
    logger = logging.getLogger('waitress')
    logger.setLevel(logging.DEBUG)

    # start the server
    logging.info(f'Starting UMA server with model: {model}')
    logging.info(f'Listening on {args.bind} with {args.nthreads} threads')
    waitress.serve(app, listen=args.bind, threads=args.nthreads)


def main():
    """Entry point for CLI execution"""
    run(sys.argv[1:])


if __name__ == '__main__':
    main()