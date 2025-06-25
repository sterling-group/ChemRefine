# chemrefine/utils_extopt.py

from pathlib import Path

ENERGY_CONVERSION = 27.21138625
LENGTH_CONVERSION = 0.529177210903

def read_input(inpfile):
    lines = Path(inpfile).read_text().splitlines()
    xyzname = lines[0].split("#")[0].strip()
    charge = int(lines[1].split("#")[0].strip())
    mult = int(lines[2].split("#")[0].strip())
    ncores = int(lines[3].split("#")[0].strip())
    dograd = bool(int(lines[4].split("#")[0].strip()))
    return xyzname, charge, mult, ncores, dograd

def read_xyzfile(xyzname):
    atom_types = []
    coordinates = []
    with open(xyzname) as f:
        natoms = int(f.readline())
        f.readline()
        for _ in range(natoms):
            parts = f.readline().split()
            atom_types.append(parts[0])
            coordinates.append(tuple(float(x) for x in parts[1:4]))
    return atom_types, coordinates

def process_output(atoms):
    e_ev = atoms.get_potential_energy()
    forces = atoms.get_forces()
    energy = e_ev / ENERGY_CONVERSION
    grad = (-forces * LENGTH_CONVERSION / ENERGY_CONVERSION).flatten().tolist()
    return energy, grad

def write_engrad(outfile, natoms, energy, dograd, gradient):
    with open(outfile, "w") as f:
        f.write("#\n# Number of atoms\n#\n")
        f.write(f"{natoms}\n")
        f.write("#\n# Total energy [Eh]\n#\n")
        f.write(f"{energy:.12e}\n")
        if dograd:
            f.write("#\n# Gradient [Eh/Bohr] A1X, A1Y, A1Z, ...\n#\n")
            for g in gradient:
                f.write(f"{g:.12e}\n")
