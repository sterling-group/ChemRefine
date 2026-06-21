"""Tests for ORCA input file generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemrefine.engines.orca.input import build_input, clamp_pal, parse_pal


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
    assert "%base" not in text  # ORCA defaults base to the .inp stem
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
    # extra blocks must appear before the xyzfile directive
    assert text.index("ProgExt") < text.index("* xyzfile")


def test_build_input_omits_base_directive(tmp_path: Path):
    """No explicit %base — ORCA defaults the base to the .inp stem."""
    template = _template(tmp_path, "! B3LYP\n")
    out = tmp_path / "step3_0-1.inp"
    xyz = tmp_path / "step3_0-1_inp.xyz"
    build_input(
        xyz_path=xyz,
        template_path=template,
        output_path=out,
        charge=0,
        multiplicity=1,
    )
    assert "%base" not in out.read_text()


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


# ---------------------------------------------------------------------------
# clamp_pal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("%pal\n  nprocs 16\nend\n", "nprocs 8"),
        ("! B3LYP PAL16\n", "PAL8"),
        ("! b3lyp pal16\n", "pal8"),  # keyword case is preserved
        ("%pal\nPAL 16\nend\n", "PAL 8"),
    ],
)
def test_clamp_pal_rewrites_every_declaration_shape(body: str, expected: str):
    """All three PAL spellings are clamped down to the budget."""
    assert expected in clamp_pal(body, 8)


def test_clamp_pal_keeps_declarations_at_or_below_budget():
    assert "nprocs 4" in clamp_pal("%pal\n  nprocs 4\nend\n", 8)
    assert "PAL8" in clamp_pal("! B3LYP PAL8\n", 8)


def test_build_input_clamps_template_pal_to_max_pal(tmp_path: Path):
    """A template asking for more ranks than ``max_cores`` is clamped in the ``.inp``.

    Regression: the SLURM allocation was clamped but the ``%pal`` block was
    copied verbatim, so ORCA launched more MPI ranks than the job owned.
    """
    template = _template(tmp_path, "! B3LYP def2-SVP\n%pal\n  nprocs 16\nend\n")
    out = tmp_path / "step1_structure_0.inp"
    build_input(
        xyz_path=tmp_path / "step1_structure_0.xyz",
        template_path=template,
        output_path=out,
        charge=0,
        multiplicity=1,
        max_pal=8,
    )
    text = out.read_text()
    assert "nprocs 8" in text
    assert "nprocs 16" not in text
    assert parse_pal(out) == 8
