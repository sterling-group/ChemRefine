"""Translate a v1.3.1 flag-style command line into the current subcommand argv.

The CLI half of the compatibility layer — :mod:`chemrefine.config_legacy` is the YAML half.
``chemrefine --input x.yaml --resume`` becomes ``chemrefine resume x.yaml`` here, before
Typer ever sees it, so every other line of :mod:`chemrefine.cli` describes one grammar.

**Removal horizon: 3.0**, together with its YAML counterpart. See
``docs/migrating-v1-to-v2.md``.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


_SUBCOMMANDS = frozenset(
    {
        "run",
        "resume",
        "rerun",
        "rerun-errors",
        "rebuild-cache",
        "rebuild-nms",
        "backends",
        "engines",
        "mcp",
        "scaffold",
        "schema",
        "validate",
    }
)
"""Every current subcommand — the translator's pass-through list.

A name missing here is treated as a v1.3.1 positional CONFIG and rewritten to
``chemrefine run <name>``, which is how ``chemrefine mcp`` once became ``run mcp``
with a "File 'mcp' does not exist" error. ``test_the_legacy_translator_knows_every_
subcommand`` pins this set against the Typer app, so adding a command without
extending it fails CI instead of failing users.
"""


def translate_argv(argv: list[str]) -> list[str]:
    """Map a v1.3.1 flag-style invocation to the new subcommand argv.

    The single home for the old flags. New-style argv (first positional is a
    known subcommand, or ``--version`` / ``--help`` / no positional) is returned
    unchanged. Otherwise the old flags are parsed and rewritten:
    ``CONFIG``→``run``; ``--skip``→``resume``; ``--rebuild_cache [N]``→
    ``rebuild-cache``; ``--rebuild_nms [N]``→``rebuild-nms``; ``--rerun_errors
    [N]``→``rerun-errors``; ``--maxcores`` carried through.
    """
    if any(a in ("--version", "--help", "-h") for a in argv):
        return argv
    first_positional = next((a for a in argv if not a.startswith("-")), None)
    if first_positional is None or first_positional in _SUBCOMMANDS:
        return argv

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("input_yaml")
    parser.add_argument("--maxcores", type=int)
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--rebuild_cache", nargs="?", const=True, type=int, default=False)
    parser.add_argument("--rebuild_nms", nargs="?", const=True, type=int, default=False)
    parser.add_argument("--rerun_errors", nargs="?", const=True, type=int, default=False)
    parser.add_argument("-v", "--verbose", action="store_true")
    try:
        ns, _unknown = parser.parse_known_args(argv)
    except SystemExit:
        return argv  # malformed legacy args — let Typer surface the error

    def _step(value: object) -> list[str]:
        return [str(value)] if isinstance(value, int) and not isinstance(value, bool) else []

    if ns.rebuild_cache is not False:
        command, step = "rebuild-cache", _step(ns.rebuild_cache)
    elif ns.rebuild_nms is not False:
        command, step = "rebuild-nms", _step(ns.rebuild_nms)
    elif ns.rerun_errors is not False:
        command, step = "rerun-errors", _step(ns.rerun_errors)
    elif ns.skip:
        command, step = "resume", []
    else:
        command, step = "run", []

    new_argv = (["-v"] if ns.verbose else []) + [command, ns.input_yaml, *step]
    if ns.maxcores is not None:
        new_argv += ["--maxcores", str(ns.maxcores)]
    logger.warning("legacy CLI flags detected; mapped to `chemrefine %s`", " ".join(new_argv))
    return new_argv
