"""Q-Chem input generation from a template + XYZ geometry (the writer half).

The template is a user-provided Q-Chem input (e.g. ``step1.in``) declaring the ``$rem``
settings; this module puts the per-structure geometry into it. Q-Chem takes its geometry
**inline**, in the first job's ``$molecule`` block, and a multi-job ``@@@`` template's later
jobs read it back with ``$molecule read $end``. So the generated block must land in job 1
and nowhere else:

* the template is split at its first ``@@@`` and only job 1 is edited — its ``$molecule``
  block, if it has one, is replaced in place whatever its body (a job-1 ``read`` has nothing
  to read from in the fresh scratch each per-structure job runs in), and a job 1 without one
  gets the block prepended; every later job passes through byte-for-byte, the ``read`` idiom
  included. Splitting first is what keeps a chain whose job 1 leaves the block to ChemRefine
  from having the geometry spliced over job 2's ``read`` while job 1 runs with no molecule;
* a job-1 block partitioned into fragments (separator lines that start with ``--``, the
  EDA / SCFMI idiom) is refused by name (:func:`require_fillable`, asked when the run
  starts): the block is regenerated from one whole-molecule geometry, and nothing can say
  which of the new atoms belong to which fragment, so a flattened block would run a
  different calculation than the template describes.

Q-Chem reads its sections in any order within a job, so the replacement never has to move a
block — position is preserved, and the diff between template and rendered input is exactly
the geometry.
"""

from __future__ import annotations

import re
from pathlib import Path

from chemrefine.errors import ConfigError
from chemrefine.ids import StepTemplate
from chemrefine.io import read_xyz_frames, write_tree_file

# Both markers are anchored to their own lines, which is where Q-Chem reads them. Matched
# anywhere, the literal text ``$molecule`` inside a ``$comment`` block starts the match and
# the *comment's* ``$end`` closes it — so a template whose comment merely mentions the
# block name would have the generated geometry spliced into its comment, and Q-Chem would
# run job 1 on the template's own placeholder geometry instead. The shipped starter's
# comment is such a mention.
_MOLECULE_BLOCK_RE = re.compile(
    r"^[ \t]*\$molecule\b.*?^[ \t]*\$end[ \t]*$", re.IGNORECASE | re.DOTALL | re.MULTILINE
)

# `INPUT_BOHR true` in a $rem section, in Q-Chem's accepted spellings (`=` optional,
# case-free, `1` for true). Anchored to its own line so a comment merely mentioning the
# rem cannot trip it.
_INPUT_BOHR_RE = re.compile(r"^[ \t]*input_bohr[ \t=]+(?:true|1)\b", re.IGNORECASE | re.MULTILINE)

# The multi-job separator, on a line of its own — where Q-Chem reads it.
_JOB_SEPARATOR_RE = re.compile(r"^[ \t]*@@@[ \t]*$", re.MULTILINE)

# A fragment separator is a line that *starts* with two dashes: Q-Chem reads ``---`` and
# longer rules as it reads ``--``. No other line of a ``$molecule`` block starts that
# way — a negative charge is one dash and a digit — so nothing else is caught.
_FRAGMENT_SEPARATOR_RE = re.compile(r"^[ \t]*--", re.MULTILINE)


def _molecule_block(xyz_path: Path, charge: int, multiplicity: int) -> str:
    """Render the ``$molecule`` block for one structure's ``_inp.xyz`` geometry.

    Rows are formatted like every geometry this package writes
    (:func:`chemrefine.io.write_single_xyz`'s six decimals) — the ``_inp.xyz`` this reads is
    already written at that precision, so nothing is lost restating it.
    """
    atoms = read_xyz_frames(xyz_path)[0]
    rows = [
        f"{symbol:2s} {x:.6f} {y:.6f} {z:.6f}"
        for symbol, (x, y, z) in zip(
            atoms.get_chemical_symbols(), atoms.get_positions(), strict=True
        )
    ]
    return "\n".join([r"$molecule", f"{charge} {multiplicity}", *rows, r"$end"])


def _split_job1(text: str) -> tuple[str, str]:
    """``text`` split at its first ``@@@``: job 1, and every later job (``""`` for none)."""
    separator = _JOB_SEPARATOR_RE.search(text)
    if separator is None:
        return text, ""
    return text[: separator.start()], text[separator.start() :]


def require_fillable(template: StepTemplate) -> None:
    """Refuse a template a structure's geometry cannot be put into faithfully.

    Both cases are decided by the template alone, so the engine's template rule
    (:meth:`~chemrefine.engines.qchem.engine.QchemEngine.check_template`) refuses them
    when the run starts, before :func:`build_input` is reached:

    * ``input_bohr`` — the block is written in Å (every geometry this package writes is), so
      a template declaring Bohr input would have Q-Chem compute on a molecule scaled by
      1/0.529. The output parser holds the other half of the same rule: its banner match
      pins ``(Angstroms)``, so a Bohr run would parse to nothing rather than to wrong
      coordinates.
    * a job-1 ``$molecule`` block split into fragments — see the module docstring.
    """
    if _INPUT_BOHR_RE.search(template.text):
        raise ConfigError(
            f"Q-Chem template {template.path} sets `input_bohr`, but ChemRefine writes the "
            f"$molecule geometry in Ångström — the job would run on a molecule scaled by "
            f"1/0.529. Remove the rem; coordinates are supplied in Å."
        )
    job1, _ = _split_job1(template.text)
    existing = _MOLECULE_BLOCK_RE.search(job1)
    if existing and _FRAGMENT_SEPARATOR_RE.search(existing.group(0)):
        raise ConfigError(
            f"Q-Chem template {template.path} partitions its $molecule block into fragments "
            f"(`--`), which ChemRefine cannot preserve: the block is regenerated per structure "
            f"from one whole-molecule geometry, and nothing can say which of its atoms belong "
            f"to which fragment. Drop the fragment lines (and the fragment rems) for this step."
        )


def build_input(
    *,
    xyz_path: Path,
    template: StepTemplate,
    output_path: Path,
    charge: int,
    multiplicity: int,
) -> Path:
    """Write a Q-Chem input to ``output_path``; return that path.

    The geometry replaces job 1's ``$molecule … $end`` block in place, or is prepended to
    job 1 when it declares none — see the module docstring for why job 1 and only job 1.
    Everything else in the template, every later job of a ``@@@`` chain and its
    ``$molecule read $end`` included, passes through byte-for-byte. The template has been
    held to :func:`require_fillable` before it reaches here, so the one refusal left is the
    output tree's, a write it will not take (:func:`chemrefine.io.write_tree_file`).
    """
    text = template.text
    block = _molecule_block(xyz_path, charge, multiplicity)
    job1, later_jobs = _split_job1(text)
    existing = _MOLECULE_BLOCK_RE.search(job1)
    if existing:
        rendered_job1 = job1[: existing.start()] + block + job1[existing.end() :]
    else:
        rendered_job1 = f"{block}\n\n{job1.lstrip()}"
    rendered = rendered_job1 + later_jobs
    write_tree_file(output_path, rendered.rstrip() + "\n")
    return output_path
