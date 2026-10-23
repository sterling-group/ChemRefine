"""The xtb step template: an xcontrol file, held to what the command line does not set.

The template (``step{N}.xcontrol``) is an ordinary xtb xcontrol file (see ``man 7
xcontrol``), handed to xtb as ``--input``. It carries what xtb takes only there: constraints
(``$constrain``, ``$fix``), a confining wall (``$wall``), a scan (``$scan``), and the
optimiser's and the SCC's settings (``$opt``, ``$scc``). Each structure's rendered input is
the template unchanged. The geometry is not in it: xtb reads it from the structure's
``_inp.xyz``, named on the command line (:data:`GEOMETRY_SUFFIX`).

Everything that decides what is computed — run type, method, solvation, charge, spin — is
on the command line (:mod:`chemrefine.engines.xtb.command`), and xtb's command line wins
over its xcontrol input. A template that sets one of those as well would be silently
overridden, so :func:`require_fillable` refuses it, naming where the setting belongs.
"""

from __future__ import annotations

import re
from pathlib import Path

from chemrefine.errors import ConfigError
from chemrefine.ids import StepTemplate, input_geometry_path, structure_stem
from chemrefine.io import write_tree_file

GEOMETRY_SUFFIX = input_geometry_path(Path(), 1, "0").name.removeprefix(structure_stem(1, "0"))
"""What a structure's input geometry adds to its stem (``_inp.xyz``): the run block names
``${stem}`` plus this as xtb's geometry. Derived from :func:`chemrefine.ids.
input_geometry_path`, which names the file, so the two cannot come apart."""

_GROUP_RE = re.compile(r"^[ \t]*\$(\w+)")
_KEY_RE = re.compile(r"^[ \t]*(\w+)[ \t]*[=:]")

_SET_ON_THE_COMMAND_LINE: dict[tuple[str, str | None], str] = {
    ("chrg", None): "the config's `charge` (the step's own, or the workflow's)",
    ("spin", None): "the config's `multiplicity` (the step's own, or the workflow's)",
    ("gfn", "method"): "the step's `options.method`",
    ("gbsa", "solvent"): "the step's `options.solvation` and `options.solvent`",
    ("solvation", "solvent"): "the step's `options.solvation` and `options.solvent`",
}
"""Each xcontrol setting ChemRefine writes on the command line instead, and where a step
sets it. A ``None`` key is the whole group (``$chrg 1``, ``$spin 2``)."""


def _settings(text: str) -> list[tuple[str, str | None]]:
    """Every ``(group, key)`` the xcontrol text sets, ``(group, None)`` for the group line.

    ``#`` starts a comment, and ``$end`` ends the input, as xtb reads it: nothing after
    it is an instruction, so nothing after it is refused.
    """
    found: list[tuple[str, str | None]] = []
    group: str | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0]
        opened = _GROUP_RE.match(line)
        if opened:
            group = opened.group(1).lower()
            if group == "end":
                break
            found.append((group, None))
            continue
        key = _KEY_RE.match(line)
        if key and group is not None:
            found.append((group, key.group(1).lower()))
    return found


def require_fillable(template: StepTemplate) -> None:
    """Refuse a template that sets what ChemRefine sets on xtb's command line.

    Asked by the engine's template rule
    (:meth:`~chemrefine.engines.xtb.engine.XtbEngine.check_template`) when the run starts
    and again when the step prepares, before anything is written.
    """
    for group, key in _settings(template.text):
        where = _SET_ON_THE_COMMAND_LINE.get((group, key))
        if where is not None:
            what = f"${group}" if key is None else f"`{key}` in ${group}"
            raise ConfigError(
                f"xtb template {template.path} sets {what}, which ChemRefine passes on xtb's "
                f"command line from {where}. The command line overrides the template, so the "
                f"template's value would be ignored. Remove it from the template."
            )


def build_input(*, template: StepTemplate, output_path: Path) -> Path:
    """Write one structure's xcontrol input to ``output_path``; return that path.

    The template unchanged: nothing in an xcontrol file is per structure. It has been
    held to :func:`require_fillable` before it reaches here, so the one refusal left is
    the output tree's, a write it will not take (:func:`chemrefine.io.write_tree_file`).
    """
    text = template.text
    write_tree_file(output_path, text if text.endswith("\n") or not text else text + "\n")
    return output_path
