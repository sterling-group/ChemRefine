#!/usr/bin/env python3

"""
ChemRefine: Main entry point for conformer sampling and refinement using ORCA.

Usage:
    python main.py input.yaml [--maxcores N] [--skip]
"""

import logging
from chemrefine import ChemRefiner

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Run ChemRefiner pipeline
    ChemRefiner().run()

if __name__ == "__main__":
    main()
