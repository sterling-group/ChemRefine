import argparse
from .constants import DEFAULT_CORES


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Automated conformer searching with ORCA and refinement."
        )
        self.parser.add_argument("input_yaml", help="YAML input for workflow.")
        self.parser.add_argument(
            "--maxcores",
            type=int,
            default=DEFAULT_CORES,
            help="Max cores for conformer search.",
        )
        self.parser.add_argument(
            "--skip",
            action="store_true",
            default=False,
            help="Skip first step if already run.",
        )
        self.parser.add_argument(
            "--rebuild_cache",
            action="store_true",
            default=False,
            help="Rebuild cache from a failed run.",
        )

    def parse(self):
        return self.parser.parse_known_args()
