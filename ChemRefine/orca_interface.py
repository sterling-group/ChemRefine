import os
import re
from .utils import Utility

class OrcaInterface:
    def __init__(self):
        self.utility = Utility()

    def create_input(self, xyz_files, template, charge, multiplicity, output_dir='./'):
        input_files, output_files = [], []

        os.makedirs(output_dir, exist_ok=True)

        for xyz in xyz_files:
            base = os.path.splitext(os.path.basename(xyz))[0]
            inp = os.path.join(output_dir, f"{base}.inp")
            out = os.path.join(output_dir, f"{base}.out")
            input_files.append(inp)
            output_files.append(out)

            with open(template, "r") as tmpl:
                content = tmpl.read()

            # Strip existing xyzfile lines and clean formatting
            content = re.sub(r'^\s*\*\s+xyzfile.*$', '', content, flags=re.MULTILINE)
            content = content.rstrip() + '\n\n'
            content += f"* xyzfile {charge} {multiplicity} {xyz}\n\n"

            with open(inp, "w") as f:
                f.write(content)

        return input_files, output_files

    def parse_output(self, file_paths, calculation_type, dir='./'):
        coordinates, energies = [], []
        for out_file in file_paths:
            path = os.path.join(dir, os.path.basename(out_file))
            if not os.path.exists(path):
                continue
            with open(path) as f:
                content = f.read()

            if calculation_type.lower() == 'goat':
                final_xyz = path.replace('.out', '.finalensemble.xyz')
                if os.path.exists(final_xyz):
                    with open(final_xyz) as fxyz:
                        lines = fxyz.readlines()
                    current_structure = []
                    for line in lines:
                        if len(line.strip().split()) == 4:
                            current_structure.append(tuple(line.strip().split()))
                    coordinates.append(current_structure)
                    energy_match = re.search(r"^\s*[-]?\d+\.\d+", lines[1])
                    energies.append(float(energy_match.group()) if energy_match else None)
            else:
                coord_block = re.findall(
                    r"CARTESIAN COORDINATES \\(ANGSTROEM\\)\n-+\n((?:.*?\n)+?)-+\n",
                    content,
                    re.DOTALL
                )
                coords = [line.split() for line in coord_block[-1].strip().splitlines()] if coord_block else []
                energy_match = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", content)
                energy = float(energy_match[-1]) if energy_match else None
                coordinates.append(coords)
                energies.append(energy)

        return coordinates, energies
