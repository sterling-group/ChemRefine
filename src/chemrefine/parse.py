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
            nargs="?",
            const=True,
            type=int,
            default=False,
            help="Rebuild cache from a failed run. Optionally specify a step number (e.g. --rebuild_cache 3).",
        )

        self.parser.add_argument(
            "--rebuild_nms",
            nargs="?",
            const=True,
            type=int,
            default=False,
            help="Rebuild normal mode sampling (NMS) displacements and cache. Optionally specify a step number (e.g. --rebuild_nms 2).",
        )

        self.parser.add_argument(
            "--rerun_errors",
            nargs="?",
            const=True,
            type=int,
            default=False,
            help="Rerun failed jobs from a specific step (e.g., --rerun_errors 3). "
            "If no step is given, reruns the most recent step.",
        )

    def parse(self):
        return self.parser.parse_known_args()
