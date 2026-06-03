"""Shared Python-script template renderer for engines that take a user template.

Both `engines.pyscf.engine.PyscfEngine` and `engines.mlff.engine.MlffEngine`
take a user-supplied ``step{N}.py`` template, substitute geometry
placeholders, and append a footer that harvests well-known variable
names from the template's locals and writes a canonical JSON output.
This module owns that renderer so adding a third template-driven
backend is "import the renderer + call build_input".

Placeholders the renderer substitutes (``string.Template`` ``$VAR``
syntax — collision-free with Python's ``{`` / ``}`` brackets):

* ``$XYZ_PATH`` — absolute path to the per-structure ``.xyz`` file.
* ``$CHARGE`` — integer total charge (from ``ctx.charge``).
* ``$MULTIPLICITY`` — integer spin multiplicity (``= 2S + 1``).

Output contract (the appended footer harvests these names if present):

==============================  ========================================
Name (assign in template)        Footer behaviour
==============================  ========================================
``energy_hartree``               REQUIRED. ``NameError`` if missing.
``gradient_hartree_per_bohr``    Optional list / numpy array.
``positions_angstrom``           Optional list / numpy array.
==============================  ========================================

The footer writes to a *basename* (relative path) so the file lands
in ``cwd = $WORK_DIR`` (scratch). The surrounding SLURM machinery
copies it back to step_dir at exit.
"""

from __future__ import annotations

from pathlib import Path
from string import Template


def _build_output_footer(output_basename: str) -> str:
    """Return the appended footer that harvests result vars and writes the JSON.

    The output filename is a basename — the script runs with
    ``cwd = $WORK_DIR`` (scratch), so a relative write goes into
    scratch and the SLURM script's ``*.json`` glob copies it back
    to step_dir.
    """
    return (
        "\n"
        "# --- ChemRefine output footer (generated; do not edit) ---\n"
        "import json as _chemrefine_json\n"
        "\n"
        "\n"
        "class _ChemRefineEncoder(_chemrefine_json.JSONEncoder):\n"
        '    """JSON encoder that calls ``.tolist()`` on numpy-like arrays."""\n'
        "\n"
        "    def default(self, o):\n"
        '        if hasattr(o, "tolist"):\n'
        "            return o.tolist()\n"
        "        return super().default(o)\n"
        "\n"
        "\n"
        '_chemrefine_result = {"energy_hartree": float(energy_hartree)}\n'
        '_chemrefine_optional = ("gradient_hartree_per_bohr", "positions_angstrom")\n'
        "for _chemrefine_name in _chemrefine_optional:\n"
        "    if _chemrefine_name in dir():\n"
        "        _chemrefine_result[_chemrefine_name] = locals()[_chemrefine_name]\n"
        f"with open({output_basename!r}, \"w\") as _chemrefine_fh:\n"
        "    _chemrefine_json.dump(\n"
        "        _chemrefine_result, _chemrefine_fh, cls=_ChemRefineEncoder\n"
        "    )\n"
    )


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

    ``$XYZ_PATH`` / ``$CHARGE`` / ``$MULTIPLICITY`` are substituted via
    :class:`string.Template.safe_substitute` so unknown ``$NAME``
    references in the template are left alone — users can keep
    shell-style ``$VAR`` lookups inside their script without
    collision.

    The appended footer reads the well-known variable names
    ``energy_hartree`` (required), ``gradient_hartree_per_bohr``, and
    ``positions_angstrom`` out of the template's locals and writes
    them to ``output_json_path.name`` (a *basename*, so the file
    lands in ``$WORK_DIR`` / scratch).

    Engines call this through
    :class:`chemrefine.engines._template_engine.TemplateScriptEngine`,
    which already raises a backend-specific ``FileNotFoundError`` for
    a missing template; the same check is kept here as a defensive
    guard for any direct caller.
    """
    if not template_path.is_file():
        raise FileNotFoundError(f"template not found: {template_path}")
    text = template_path.read_text(encoding="utf-8")
    rendered = Template(text).safe_substitute(
        XYZ_PATH=str(xyz_path),
        CHARGE=charge,
        MULTIPLICITY=multiplicity,
    )
    rendered = rendered.rstrip() + _build_output_footer(output_json_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path
