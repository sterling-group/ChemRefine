"""Choose ``solvation_model: auto`` in the step's own environment, before the run.

The weights ``auto`` chooses by live where the step runs: in its MLIP environment's
packages, its Hugging Face cache, or a file it names. So the orchestrator, which may have
none of them, asks that environment's interpreter, the one the step's jobs launch with:

    <step's python> -m chemrefine.engines.mlip.solvation._choose <library> [<weights>]

which prints one JSON line, ``{"model": …, "skipped": […]}`` or ``{"error": …}``. Underscored,
so the library discovery in :mod:`chemrefine.engines.mlip.solvation` does not import it.
"""

from __future__ import annotations

import json
import subprocess
import sys

from chemrefine.errors import ConfigError

_TIMEOUT_SECONDS = 300
"""Long enough for a cold import of the library on a shared filesystem, short enough that a
hung interpreter is reported before anything is submitted."""


def choose_in_environment(
    launcher: str, solvation: str, weights: str | None
) -> tuple[str, tuple[str, ...]]:
    """The model ``auto`` chooses in ``launcher``'s environment, and why it passed over others.

    A :class:`ConfigError` when the environment cannot answer: the library is not
    installed there, no model of it is, or the interpreter does not run.
    """
    command = [launcher, "-m", __name__, solvation, weights or ""]
    try:
        # No shell; the interpreter is the step's resolved launcher, the module is this one,
        # and the two arguments are validated option values passed as argv, never as code.
        done = subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfigError(
            f"solvation: could not ask {launcher} which {solvation} model is installed: {exc}"
        ) from exc
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    try:
        answer = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        raise ConfigError(
            f"solvation: {launcher} gave no answer for which {solvation} model is installed "
            f"(exit {done.returncode}): {done.stderr.strip()[-500:]}"
        ) from None
    if "error" in answer:
        raise ConfigError(f"solvation: `solvation_model: auto` in {launcher}: {answer['error']}")
    return str(answer["model"]), tuple(str(why) for why in answer["skipped"])


def main(argv: list[str]) -> int:
    """Print this environment's ``auto`` choice for ``argv = [library, weights]`` as JSON."""
    from chemrefine.engines.mlip.solvation import solvation_library

    solvation, weights = argv[0], (argv[1] if len(argv) > 1 and argv[1] else None)
    try:
        model, skipped = solvation_library(solvation).choose(weights)
    except ImportError as exc:
        print(json.dumps({"error": f"{solvation} is not installed in this environment: {exc}"}))
    except ConfigError as exc:
        print(json.dumps({"error": str(exc)}))
    else:
        print(json.dumps({"model": model, "skipped": list(skipped)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
