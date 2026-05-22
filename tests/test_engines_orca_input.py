"""Tests for ORCA input file generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemrefine.engines.orca.input import build_input, parse_pal


def _template(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "template.inp"
    path.write_text(body, encoding="utf-8")
    return path


def test_build_input_appends_xyzfile_directive(tmp_path: Path):
    template = _template(tmp_path, "! B3LYP def2-SVP\n%pal\n  nprocs 4\nend\n")
    out = tmp_path / "step1_structure_0.inp"
    xyz = tmp_path / "step1_structure_0.xyz"
    build_input(
        xyz_path=xyz,
        template_path=template,
        output_path=out,
        charge=0,
        multiplicity=1,
    )
    text = out.read_text()
    assert "! B3LYP def2-SVP" in text
    assert "%pal" in text
    assert '%base "step1_structure_0"' in text
    assert f"* xyzfile 0 1 {xyz}" in text


def test_build_input_strips_existing_xyzfile_line(tmp_path: Path):
    template = _template(
        tmp_path,
        "! B3LYP def2-SVP\n* xyzfile 0 1 stale.xyz\n",
    )
    out = tmp_path / "step1.inp"
    xyz = tmp_path / "step1.xyz"
    build_input(
        xyz_path=xyz,
        template_path=template,
        output_path=out,
        charge=-1,
        multiplicity=2,
    )
    text = out.read_text()
    assert "stale.xyz" not in text
    assert f"* xyzfile -1 2 {xyz}" in text


def test_build_input_inserts_extra_blocks_before_xyzfile(tmp_path: Path):
    template = _template(tmp_path, "! B3LYP\n")
    out = tmp_path / "step1.inp"
    xyz = tmp_path / "step1.xyz"
    extra = '%method\n  ProgExt "/path/to/wrapper.sh"\nend'
    build_input(
        xyz_path=xyz,
        template_path=template,
        output_path=out,
        charge=0,
        multiplicity=1,
        extra_blocks=extra,
    )
    text = out.read_text()
    assert "ProgExt" in text
    # extra blocks must appear before the %base + xyzfile directives
    assert text.index("ProgExt") < text.index("%base")
    assert text.index("%base") < text.index("* xyzfile")


def test_build_input_uses_output_stem_for_base_directive(tmp_path: Path):
    template = _template(tmp_path, "! B3LYP\n")
    out = tmp_path / "step3_structure_0-1.inp"
    xyz = tmp_path / "step3_structure_0-1.xyz"
    build_input(
        xyz_path=xyz,
        template_path=template,
        output_path=out,
        charge=0,
        multiplicity=1,
    )
    assert '%base "step3_structure_0-1"' in out.read_text()


def test_build_input_missing_template_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_input(
            xyz_path=tmp_path / "x.xyz",
            template_path=tmp_path / "missing.inp",
            output_path=tmp_path / "step1.inp",
            charge=0,
            multiplicity=1,
        )


# ---------------------------------------------------------------------------
# parse_pal
# ---------------------------------------------------------------------------


def test_parse_pal_reads_nprocs(tmp_path: Path):
    inp = tmp_path / "step1.inp"
    inp.write_text("! B3LYP def2-SVP\n%pal\n  nprocs 8\nend\n", encoding="utf-8")
    assert parse_pal(inp) == 8


def test_parse_pal_reads_inline_directive(tmp_path: Path):
    inp = tmp_path / "step1.inp"
    inp.write_text("! B3LYP PAL4\n", encoding="utf-8")
    assert parse_pal(inp) == 4


def test_parse_pal_defaults_to_one(tmp_path: Path):
    inp = tmp_path / "step1.inp"
    inp.write_text("! B3LYP def2-SVP\n", encoding="utf-8")
    assert parse_pal(inp) == 1
