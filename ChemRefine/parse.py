import argparse
from .constants import DEFAULT_CORES

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Automated conformer searching with ORCA and refinement."
    )
    parser.add_argument("input_file", help="YAML input for workflow.")
    parser.add_argument("--maxcores", type=int, default=DEFAULT_CORES,
                        help="Max cores for conformer search.")
    parser.add_argument("--skip", action="store_true", default=False,
                        help="Skip first step if already run.")
    return parser.parse_known_args()
