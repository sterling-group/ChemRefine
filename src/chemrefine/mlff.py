import logging
from typing import List, Tuple, Optional
from ase import Atoms
from .utils import Utility
from pathlib import Path
import time
import socket
import pickle
import torch
from multiprocessing import Process

class MLFFCalculator:
    """Flexible MLFF calculator supporting MACE, UMA (FairChem), CHGNet, etc."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        model_path: Optional[str] = None,
        task_name: str = "mace_off"
    ):
        self.model_name = model_name
        self.device = device
        self.task_name = task_name
        self.model_path = model_path
        self.calculator = self._setup_calculator()

    def _setup_calculator(self):
        """Initialize and return the correct ASE calculator based on task_name."""
        if self.task_name.startswith("mace"):
            return self._setup_mace()
        elif self.task_name.startswith(("omol", "omat", "odac", "uma", "fairchem")):
            return self._setup_fairchem()
        elif self.task_name.startswith("chgnet"):
            return self._setup_chgnet()
        else:
            raise ValueError(f"Unsupported task name: {self.task_name}")

    def _setup_mace(self):
        from mace.calculators import mace_off, mace_mp
        if self.task_name == "mace_off":
            logging.info(f"Using MACE OFF model: {self.model_name} on device: {self.device}")   
            return mace_off(model=self.model_name, device=self.device)
        elif self.task_name == "mace_mp":
            logging.info(f"Using MACE MP model: {self.model_name} on device: {self.device}")
            return mace_mp(model=self.model_name, device=self.device)
        else:
            raise ValueError(f"Unsupported MACE task name: {self.task_name}")

    def _setup_fairchem(self):
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        predictor = pretrained_mlip.get_predict_unit(
            model_name=self.model_name,
            device=self.device,
        )
        logging.info(f"Using FAIRChem model: {self.model_name} on device: {self.device}")
        return FAIRChemCalculator(predictor, task_name=self.task_name)

    def _setup_chgnet(self):
        from chgnet.model import CHGNet
        from chgnet.calculators import CHGNetCalculator
        model = CHGNet.load(self.model_path) if self.model_path else CHGNet.load()
        return CHGNetCalculator(model=model)

    def get_single_point(self, atoms: Atoms) -> Tuple[float, List[float]]:
        """Compute energy and gradient using the configured MLFF."""
        atoms.calc = self.calculator
        from chemrefine.utils_extopt import process_output
        return process_output(atoms)

    def calculate(self, atoms: Atoms, fmax: float = 0.03, steps: int = 200) -> Atoms:
        """Optional: geometry optimization using LBFGS."""
        atoms.calc = self.calculator
        from ase.optimize import LBFGS
        optimizer = LBFGS(atoms, logfile=None)
        optimizer.run(fmax=fmax, steps=steps)
        return atoms

def serve_mlff(model_name, task_name, device="cuda", port=50051):
    """
    Run a socket server exposing MLFF energy and gradient calculation.
    This mimics ORCA's external server mode.

    Args:
        model_name (str): Name of the MLFF model.
        task_name (str): Task type (e.g., 'mace_off', 'uma-s-1', etc.)
        device (str): 'cuda' or 'cpu'.
        port (int): Port to listen on.
    """
    from ase import Atoms
    from chemrefine.mlff import MLFFCalculator

    calc = MLFFCalculator(model_name=model_name, task_name=task_name, device=device)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", port))
        s.listen()
        print(f"[MLFF Server] Listening on port {port}...")
        while True:
            conn, _ = s.accept()
            with conn:
                data = b""
                while True:
                    packet = conn.recv(4096)
                    if not packet:
                        break
                    data += packet
                atoms = pickle.loads(data)
                energy, forces = calc.get_single_point(atoms)
                result = pickle.dumps((energy, forces))
                conn.sendall(result)    

def query_mlff_server(atoms, host='127.0.0.1', port=8888):
    """
    Queries the MLFF server for energy and forces for the given Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Atomic structure for inference.
    host : str
        Server host address.
    port : int
        Port number the server is bound to.

    Returns
    -------
    Tuple[float, ndarray]
        Energy and forces from the server.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        payload = pickle.dumps(atoms)
        s.sendall(struct.pack(">I", len(payload)) + payload)
        length_data = s.recv(4)
        result_len = struct.unpack(">I", length_data)[0]
        result = b""
        while len(result) < result_len:
            chunk = s.recv(result_len - len(result))
            if not chunk:
                raise RuntimeError("Connection lost.")
            result += chunk
        energy, forces = pickle.loads(result)
        return energy, forces