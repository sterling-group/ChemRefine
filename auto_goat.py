import numpy as np
import re
from pathlib import Path

def read_input_file(input_path):
    """Read the ORCA input file."""
    if not input_path.is_file():
        print(
            f"Error: Input file '{input_path}' does not exist. "
            "Please check the file path and try again.",
            file=sys.stderr
        )
        sys.exit(1)
    with input_path.open('r') as f:
        lines = f.readlines()
    return lines

def parse_pal_from_input(lines):
    """
    Parse PAL value from ORCA input file lines.

    Args:
        lines (list): List of strings representing lines in the ORCA input file.

    Returns:
        int: The PAL value found in the input file, or None if not found.
    """
    input_text = ''.join(lines)
    # Use regex to find %pal ... end blocks, regardless of formatting
    pal_blocks = re.findall(r'%pal\s*(.*?)\s*end', input_text, re.IGNORECASE | re.DOTALL)
    for pal_block in pal_blocks:
        # Search for nprocs value within pal_block
        match = re.search(r'nprocs\s+(\d+)', pal_block, re.IGNORECASE)
        if match:
            pal_value = int(match.group(1))
            return pal_value
    # Also check for pal in '!' line
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('!'):
            tokens = stripped_line[1:].strip().split()
            for token in tokens:
                match = re.match(r'pal(\d+)', token.lower())
                if match:
                    pal_value = int(match.group(1))
                    return pal_value
    return None
def parse_coordinates(data):
    # If data is already a list of lines, no need to split
    if isinstance(data, str):
        lines = data.strip().split('\n')
    else:
        lines = data  # Assume it's already a list
    
    result = []
    recording = False

    for line in lines:
        line = line.strip()
        if line.startswith('* xyz'):
            # Parse charge and multiplicity
            _, _, charge, multiplicity = line.split()
            charge, multiplicity = int(charge), int(multiplicity)
            recording = True
            current_structure = {
                "header": f"* xyz {charge} {multiplicity}",
                "atoms": []
            }
        elif line.startswith('*'):
            if recording:
                result.append(current_structure)
                recording = False
        elif recording:
            # Parse atom data
            parts = line.split()
            atom = f"{parts[0]:<2} {float(parts[1]):>12.6f} {float(parts[2]):>12.6f} {float(parts[3]):>12.6f}"
            current_structure["atoms"].append(atom)
    
    return result

def write_preopt_coordinates(structures, output_file,cores):
    with open(output_file, "w") as f:
        f.write('%pal nprocs' + str(cores) +'\n')
        f.write('\n!opt pbe def2svp \n')
        f.write('%geom\nMaxIter 50 \nConvergence loose \nend \n')
        for structure in structures:
            f.write("\n" + structure["header"] + "\n")
            f.write("\n".join(structure["atoms"]) + "\n")
            f.write("*\n")





