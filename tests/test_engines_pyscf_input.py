"""Tests for the PySCF Python-script template renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemrefine.engines.pyscf.input import build_input


def test_build_input_substitutes_geometry_placeholders(tmp_path: Path):
    template = tmp_path / "step1.py"
    template.write_text(
        'mol = gto.M(atom="$XYZ_PATH", charge=$CHARGE, spin=$MULTIPLICITY - 1)\n',
        encoding="utf-8",
    )
    out = tmp_path / "rendered.py"
    build_input(
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
    build_input(
        xyz_path=tmp_path / "frame.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "step1_structure_0.json",
        charge=0,
        multiplicity=1,
    )
    text = out.read_text(encoding="utf-8")
    # The user's body survives.
    assert "energy_hartree = -1.0" in text
    # Footer markers.
    assert "# --- ChemRefine output footer (generated; do not edit) ---" in text
    assert "_chemrefine_result" in text
    # Writes to the BASENAME (relative), so the file lands in $WORK_DIR.
    assert "with open('step1_structure_0.json', \"w\")" in text
    # No leftover $OUTPUT_JSON placeholder anywhere.
    assert "$OUTPUT_JSON" not in text


def test_build_input_no_longer_substitutes_output_json(tmp_path: Path):
    """``$OUTPUT_JSON`` is no longer documented; if the user types it, it stays.

    safe_substitute leaves unknown placeholders alone, so a legacy
    template that still references ``$OUTPUT_JSON`` ships its
    placeholder through to runtime where it becomes an obvious shell
    string. The user gets the canonical JSON via the appended footer
    regardless.
    """
    template = tmp_path / "step1.py"
    template.write_text(
        'energy_hartree = -1.0\n'
        "# legacy: $OUTPUT_JSON\n",
        encoding="utf-8",
    )
    out = tmp_path / "rendered.py"
    build_input(
        xyz_path=tmp_path / "frame.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "step1_structure_0.json",
        charge=0,
        multiplicity=1,
    )
    text = out.read_text(encoding="utf-8")
    # Legacy reference unchanged (safe_substitute).
    assert "# legacy: $OUTPUT_JSON" in text
    # Canonical write still happens via the appended footer.
    assert "with open('step1_structure_0.json', \"w\")" in text


def test_build_input_missing_template_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="PySCF template not found"):
        build_input(
            xyz_path=tmp_path / "x.xyz",
            template_path=tmp_path / "missing.py",
            output_path=tmp_path / "out.py",
            output_json_path=tmp_path / "out.json",
            charge=0,
            multiplicity=1,
        )


def test_build_input_preserves_python_braces(tmp_path: Path):
    """A real PySCF script uses Python ``{ ... }`` everywhere; those must survive intact."""
    template = tmp_path / "step1.py"
    template.write_text(
        'payload = {"key": float(mf.e_tot)}\n'
        'gradient = [(i, x) for i, x in enumerate(grad)]\n'
        'f = f"step{step}_done"\n'
        'energy_hartree = -1.0\n',
        encoding="utf-8",
    )
    out = tmp_path / "rendered.py"
    build_input(
        xyz_path=tmp_path / "x.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "y.json",
        charge=0,
        multiplicity=1,
    )
    text = out.read_text()
    # Python dict / list / f-string braces survive unchanged.
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
    build_input(
        xyz_path=tmp_path / "x.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "y.json",
        charge=0,
        multiplicity=1,
    )
    text = out.read_text()
    assert "$VIRTUAL_ENV" in text  # unknown placeholder preserved
