import argparse
from .mlff import run_mlff_calculation


def main():
    parser = argparse.ArgumentParser(description="Run MLFF optimization on an XYZ file")
    parser.add_argument("xyz", help="Path to the XYZ file")
    parser.add_argument("--model", default="mol", help="Model name")
    parser.add_argument("--device", default=None, help="Computation device")
    parser.add_argument("--fmax", type=float, default=0.03, help="LBFGS force convergence")
    parser.add_argument("--steps", type=int, default=200, help="Maximum optimisation steps")
    args = parser.parse_args()

    coords, energy = run_mlff_calculation(
        args.xyz,
        model_name=args.model,
        device=args.device,
        fmax=args.fmax,
        steps=args.steps,
    )

    base = args.xyz.rsplit(".", 1)[0]
    with open(f"{base}.opt.xyz", "w") as f:
        f.write(f"{len(coords)}\n\n")
        for elem, x, y, z in coords:
            f.write(f"{elem} {x} {y} {z}\n")
    with open(f"{base}.energy", "w") as f:
        f.write(str(energy))


if __name__ == "__main__":
    main()
