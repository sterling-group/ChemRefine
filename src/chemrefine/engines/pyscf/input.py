"""PySCF Python-script template rendering.

Mirrors :func:`chemrefine.engines.orca.input.build_input` — the user
supplies a ``step{N}.py`` template that's a working PySCF script with
``string.Template`` placeholders; ChemRefine renders one rendered
``.py`` per structure with the placeholders substituted.

Placeholders recognised in the template:

* ``$XYZ_PATH`` — absolute path to the per-structure ``.xyz`` file.
* ``$CHARGE`` — integer total charge (from ``ctx.charge``).
* ``$MULTIPLICITY`` — integer spin multiplicity (``= 2S + 1``).
* ``$OUTPUT_JSON`` — absolute path the script MUST write to.

The script must produce a JSON document at ``$OUTPUT_JSON`` with at
least an ``"energy_hartree"`` field; optional ``positions_angstrom``
and ``gradient_hartree_per_bohr`` are picked up by
:meth:`PyscfDirectEngine.parse` when present.

``string.Template`` (``$VAR`` syntax) is used in preference to
``str.format`` because real PySCF scripts contain Python ``{`` /
``}`` brackets in dicts, f-strings, and call sites; the dollar
syntax doesn't collide.
"""

from __future__ import annotations

from pathlib import Path
from string import Template


def build_input(
    *,
    xyz_path: Path,
    template_path: Path,
    output_path: Path,
    output_json_path: Path,
    charge: int,
    multiplicity: int,
) -> Path:
    """Render ``template_path`` into ``output_path`` and return the rendered path.

    ``$XYZ_PATH`` / ``$CHARGE`` / ``$MULTIPLICITY`` / ``$OUTPUT_JSON``
    are substituted via :class:`string.Template.safe_substitute` —
    unknown ``$NAME`` references in the template are left alone so
    the user can use other shell-style ``$VAR`` lookups inside their
    PySCF script without collision.
    """
    if not template_path.is_file():
        raise FileNotFoundError(f"PySCF template not found: {template_path}")
    text = template_path.read_text(encoding="utf-8")
    rendered = Template(text).safe_substitute(
        XYZ_PATH=str(xyz_path),
        CHARGE=charge,
        MULTIPLICITY=multiplicity,
        OUTPUT_JSON=str(output_json_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path
