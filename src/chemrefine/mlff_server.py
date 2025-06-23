import argparse
import socket
import struct
import pickle
from chemrefine.mlff import MLFFCalculator
import logging

def handle_connection(conn, calc):
    while True:
        length_data = conn.recv(4)
        if not length_data:
            break
        msg_len = struct.unpack(">I", length_data)[0]
        data = b""
        while len(data) < msg_len:
            chunk = conn.recv(msg_len - len(data))
            if not chunk:
                break
            data += chunk
        atoms = pickle.loads(data)
        atoms.calc = calc.calculator
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        result = pickle.dumps((energy, forces))
        conn.sendall(struct.pack(">I", len(result)) + result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="uma-s-1")
    parser.add_argument("--task-name", default="omol")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting MLFF server with model={args.model} task={args.task_name}")

    calc = MLFFCalculator(
        model_name=args.model,
        task_name=args.task_name,
        device=args.device
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", args.port))
        s.listen(1)
        conn, _ = s.accept()
        with conn:
            handle_connection(conn, calc)

if __name__ == "__main__":
    main()
