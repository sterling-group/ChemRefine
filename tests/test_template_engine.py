"""Tests for the shared template renderer + ``ScriptEngine`` helpers.

The per-engine ``test_engines_pyscf.py`` and ``test_engines_mlip.py``
exercise the lifecycle end-to-end with their backend labels. The
tests here exercise the *shared* surface area — the renderer
(``_template_render.build_input``) and the output-parsing helpers
(``_template_output._atoms_from_output`` / ``_forces_from_gradient``) — once,
not twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine.engines._script import render as _template_render
from chemrefine.engines._script.engine import ScriptEngine
from chemrefine.engines._script.output import _atoms_from_output, _forces_from_gradient
from chemrefine.errors import OutputParseError


def test_base_template_vars_default_is_empty():
    """The base ``_template_vars`` injects nothing; subclasses (mlip/pyscf) override it."""
    assert ScriptEngine()._template_vars(None) == {}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _template_render.build_input — renderer
# ---------------------------------------------------------------------------


def test_build_input_substitutes_geometry_placeholders(tmp_path: Path):
    template = tmp_path / "step1.py"
    template.write_text(
        'mol = gto.M(atom="$XYZ_PATH", charge=$CHARGE, spin=$MULTIPLICITY - 1)\n',
        encoding="utf-8",
    )
    out = tmp_path / "rendered.py"
    _template_render.build_input(
        xyz_path=tmp_path / "frame.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "frame.json",
        charge=-1,
        multiplicity=2,
    )
    text = out.read_text(encoding="utf-8")
    assert f'atom="{tmp_path / "frame.xyz"}"' in text
    assert "charge=-1" in text
    assert "spin=2 - 1" in text


def test_build_input_appends_output_footer(tmp_path: Path):
    """The rendered file must end with the canonical JSON-writing footer."""
    template = tmp_path / "step1.py"
    template.write_text("energy_hartree = -1.0\n", encoding="utf-8")
    out = tmp_path / "rendered.py"
    _template_render.build_input(
        xyz_path=tmp_path / "frame.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "step1_structure_0.json",
        charge=0,
        multiplicity=1,
    )
    text = out.read_text(encoding="utf-8")
    assert "energy_hartree = -1.0" in text
    assert "# --- ChemRefine output footer (generated; do not edit) ---" in text
    assert "_chemrefine_result" in text
    # Writes to the BASENAME so it lands in $WORK_DIR.
    assert "with open('step1_structure_0.json', \"w\")" in text
    assert "$OUTPUT_JSON" not in text


def test_build_input_leaves_legacy_output_json_placeholder_alone(tmp_path: Path):
    """``$OUTPUT_JSON`` is not a known placeholder; ``safe_substitute`` leaves it intact."""
    template = tmp_path / "step1.py"
    template.write_text(
        "energy_hartree = -1.0\n# legacy: $OUTPUT_JSON\n",
        encoding="utf-8",
    )
    out = tmp_path / "rendered.py"
    _template_render.build_input(
        xyz_path=tmp_path / "frame.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "step1_structure_0.json",
        charge=0,
        multiplicity=1,
    )
    text = out.read_text(encoding="utf-8")
    assert "# legacy: $OUTPUT_JSON" in text
    assert "with open('step1_structure_0.json', \"w\")" in text


def test_build_input_missing_template_raises_generic(tmp_path: Path):
    """Direct call (no engine layer above it) surfaces the generic message."""
    with pytest.raises(FileNotFoundError, match="template not found"):
        _template_render.build_input(
            xyz_path=tmp_path / "x.xyz",
            template_path=tmp_path / "missing.py",
            output_path=tmp_path / "out.py",
            output_json_path=tmp_path / "out.json",
            charge=0,
            multiplicity=1,
        )


def test_build_input_preserves_python_braces(tmp_path: Path):
    """A real script uses Python ``{ ... }`` everywhere; those must survive intact."""
    template = tmp_path / "step1.py"
    template.write_text(
        'payload = {"key": float(mf.e_tot)}\n'
        "gradient = [(i, x) for i, x in enumerate(grad)]\n"
        'f = f"step{step}_done"\n'
        "energy_hartree = -1.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "rendered.py"
    _template_render.build_input(
        xyz_path=tmp_path / "x.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "y.json",
        charge=0,
        multiplicity=1,
    )
    text = out.read_text()
    assert 'payload = {"key": float(mf.e_tot)}' in text
    assert "gradient = [(i, x) for i, x in enumerate(grad)]" in text
    assert 'f = f"step{step}_done"' in text


def test_build_input_leaves_unknown_placeholders_intact(tmp_path: Path):
    """``safe_substitute`` should leave ``$UNKNOWN`` references alone."""
    template = tmp_path / "step1.py"
    template.write_text(
        'pyenv = "$VIRTUAL_ENV"\nenergy_hartree = -1.0\n',
        encoding="utf-8",
    )
    out = tmp_path / "rendered.py"
    _template_render.build_input(
        xyz_path=tmp_path / "x.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "y.json",
        charge=0,
        multiplicity=1,
    )
    assert "$VIRTUAL_ENV" in out.read_text()


# ---------------------------------------------------------------------------
# _atoms_from_output / _forces_from_gradient — shared helpers
# ---------------------------------------------------------------------------


def test_atoms_from_output_falls_back_to_seed_atoms():
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms = _atoms_from_output({"energy_hartree": -1.0}, fallback=seed)
    np.testing.assert_allclose(atoms.get_positions(), seed.get_positions())


def test_atoms_from_output_uses_positions_when_present():
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms = _atoms_from_output(
        {"energy_hartree": -1.0, "positions_angstrom": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]},
        fallback=seed,
    )
    np.testing.assert_allclose(atoms.get_positions(), [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])


def test_atoms_from_output_raises_without_fallback_and_no_positions():
    with pytest.raises(OutputParseError, match="positions_angstrom"):
        _atoms_from_output({"energy_hartree": -1.0}, fallback=None)


def test_forces_from_gradient_converts_units():
    from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A

    forces = _forces_from_gradient([[1.0, 0.0, 0.0]])
    assert forces is not None
    np.testing.assert_allclose(forces[0], [-HARTREE_PER_BOHR_TO_EV_PER_A, 0.0, 0.0])


def test_forces_from_gradient_handles_none():
    assert _forces_from_gradient(None) is None


def test_forces_from_gradient_handles_empty():
    assert _forces_from_gradient([]) is None
