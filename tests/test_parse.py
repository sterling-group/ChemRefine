"""
Test script to parse an ORCA output file using ChemRefine's OrcaInterface class.

Usage:
    python test_orca_parse.py
"""

from chemrefine.orca_interface import OrcaInterface


def main():
    # Path to the ORCA output file
    filepath = "./step2/step2_structure_0.out"
    file_dir = "./step2"
    # logging.basicConfig(level=logging.DEBUG)

    # Instantiate the OrcaInterface
    orca_parser = OrcaInterface()

    # Parse the file
    coordinates, energy = orca_parser.parse_output(
        [filepath], calculation_type="dft", dir=file_dir
    )

    # Print the parsing results
    print("Parsed Results:")
    print(energy)
    print(coordinates)


if __name__ == "__main__":
    main()
