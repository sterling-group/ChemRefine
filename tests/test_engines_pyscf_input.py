"""Tests for the PySCF Python-script template renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemrefine.engines.pyscf.input import build_input


def test_build_input_substitutes_placeholders(tmp_path: Path):
    template = tmp_path / "step1.py"
    template.write_text(
        'mol = gto.M(atom="$XYZ_PATH", charge=$CHARGE, spin=$MULTIPLICITY - 1)\n'
        'with open("$OUTPUT_JSON", "w") as fh:\n'
        "    fh.write('{}')\n",
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
    assert f'open("{tmp_path / "frame.json"}"' in text


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
        'payload = {"energy_hartree": float(mf.e_tot)}\n'
        'gradient = [(i, x) for i, x in enumerate(grad)]\n'
        'f = f"step{step}_done"\n'
        "# placeholder: $OUTPUT_JSON\n",
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
    assert 'payload = {"energy_hartree": float(mf.e_tot)}' in text
    assert "gradient = [(i, x) for i, x in enumerate(grad)]" in text
    assert 'f = f"step{step}_done"' in text
    # The known placeholder still gets substituted.
    assert str(tmp_path / "y.json") in text


def test_build_input_leaves_unknown_placeholders_intact(tmp_path: Path):
    """``safe_substitute`` should leave ``$UNKNOWN`` references alone."""
    template = tmp_path / "step1.py"
    template.write_text(
        'pyenv = "$VIRTUAL_ENV"\n# $OUTPUT_JSON gets substituted\n',
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
    assert str(tmp_path / "y.json") in text
