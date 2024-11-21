import numpy as np

def parse_orca_input(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize placeholders
    nprocs = None
    input_line = None
    charge = None
    multiplicity = None
    coordinates = []

    for line in lines:
        line = line.strip()

        # Extract number of cores from %pal
        if line.startswith('%pal'):
            for pal_line in lines[lines.index(line):]:
                if 'nprocs' in pal_line:
                    nprocs = int(pal_line.split()[1])
                    print(nprocs)
                    break
                if pal_line.strip() == 'end':
                    break
                    
      # Extract input line starting with "!"
        elif line.startswith('!'):
            input_line = line.split()

        # Extract charge and multiplicity from "* xyz"
        elif line.startswith('* xyz'):
            _, _, charge, multiplicity = line.split()
            charge = int(charge)
            multiplicity = int(multiplicity)

        # Extract atomic coordinates
        elif line and not line.startswith(('*', '%', '!')):
            parts = line.split()
            if len(parts) >= 4:  # Ensure it's a valid coordinate line
                print(line)
                #coordinates.append([parts[0]] + list(map(float, parts[1:4])))

    # Convert coordinates to NumPy array
    coordinates = np.array(coordinates, dtype=object)

    # Output results
    return {
        "nprocs": nprocs,
        "input_line": input_line,
        "charge": charge,
        "multiplicity": multiplicity,
        "coordinates": coordinates
    }

# Parse the uploaded ORCA input file
file_path = "/mnt/e/Documents/Postdoc/Gevorgyan/Q1/input/Pd_BINAP_guess.inp"
parsed_data = parse_orca_input(file_path)

# Print the parsed data
print("Number of cores (nprocs):", parsed_data["nprocs"])
print("Input line:", parsed_data["input_line"])
print("Charge:", parsed_data["charge"])
print("Multiplicity:", parsed_data["multiplicity"])
print("Coordinates:\n", parsed_data["coordinates"])
