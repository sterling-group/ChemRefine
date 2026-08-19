"""Tests for ORCA input file generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemrefine.engines.orca.input import build_input, clamp_pal
from chemrefine.engines.orca.inspect import inspect_template
from chemrefine.errors import ConfigError


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


def test_the_xyzfile_path_is_emitted_bare_which_is_why_output_dir_forbids_spaces(tmp_path: Path):
    """The geometry path is the last, unquoted, whitespace-delimited token of the directive.

    This is the *reason* `Config._reject_space_in_output_dir` exists, pinned where the
    emission happens: ORCA reads `* xyzfile` by splitting on whitespace and does not treat
    the filename as quotable, so a path with a space in it is truncated at the space
    (verified against ORCA 6.1.1 — `CANNOT OPEN FILE`, naming the prefix — with and
    without quotes around the value).

    So the config rule and this line are one decision. If anyone ever changes the emission
    — quotes it, or moves to a directive that *is* quotable — this test fails and the
    config rule should be re-examined rather than left standing for a format that no longer
    needs it.
    """
    template = _template(tmp_path, "! B3LYP\n")
    out = tmp_path / "step1_0.inp"
    xyz = tmp_path / "step1_0_inp.xyz"
    build_input(xyz_path=xyz, template_path=template, output_path=out, charge=0, multiplicity=1)
    directive = next(line for line in out.read_text().splitlines() if line.startswith("* xyzfile"))
    assert directive == f"* xyzfile 0 1 {xyz}"
    assert '"' not in directive


def test_build_input_missing_template_raises(tmp_path: Path):
    with pytest.raises(ConfigError):
        build_input(
            xyz_path=tmp_path / "x.xyz",
            template_path=tmp_path / "missing.inp",
            output_path=tmp_path / "step1.inp",
            charge=0,
            multiplicity=1,
        )


# ---------------------------------------------------------------------------
# PAL parsing (via inspect_template — the one reader)
# ---------------------------------------------------------------------------


def test_inspect_template_reads_nprocs(tmp_path: Path):
    inp = tmp_path / "step1.inp"
    inp.write_text("! B3LYP def2-SVP\n%pal\n  nprocs 8\nend\n", encoding="utf-8")
    assert inspect_template(inp).pal == 8


def test_inspect_template_reads_inline_pal_directive(tmp_path: Path):
    inp = tmp_path / "step1.inp"
    inp.write_text("! B3LYP PAL4\n", encoding="utf-8")
    assert inspect_template(inp).pal == 4


def test_inspect_template_pal_defaults_to_one(tmp_path: Path):
    inp = tmp_path / "step1.inp"
    inp.write_text("! B3LYP def2-SVP\n", encoding="utf-8")
    assert inspect_template(inp).pal == 1


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

    The SLURM allocation and the ``%pal`` block must be clamped together; if only the first is,
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
    assert inspect_template(out).pal == 8


def test_build_input_absolutizes_relative_template_paths(tmp_path: Path):
    """A quoted path that exists relative to the template dir is pinned absolute.

    A ``%DOCKER GUEST "../templates/cl.xyz"`` path resolves against the
    scratch work dir at run time, so ORCA died with CANNOT OPEN FILE.
    """
    templates = tmp_path / "templates"
    templates.mkdir()
    guest = templates / "cl.xyz"
    guest.write_text("1\nchloride\nCl 0.0 0.0 0.0\n", encoding="utf-8")
    template = templates / "step1.inp"
    template.write_text(
        '! XTB\n%DOCKER\n\tGUEST "../templates/cl.xyz"\n\tGuestCharge -1\nEND\n',
        encoding="utf-8",
    )
    out = tmp_path / "step1_0.inp"
    build_input(
        xyz_path=tmp_path / "step1_0.xyz",
        template_path=template,
        output_path=out,
        charge=0,
        multiplicity=1,
    )
    text = out.read_text()
    assert f'GUEST "{guest.resolve()}"' in text
    assert '"../templates/cl.xyz"' not in text


def test_build_input_leaves_absolute_and_unresolvable_paths_alone(tmp_path: Path):
    """Absolute paths and quoted strings that match no file pass through untouched."""
    template = _template(
        tmp_path,
        '! XTB\n%DOCKER\n\tGUEST "/abs/cl.xyz"\nEND\n%foo BAR "not-a-file.xyz" end\n',
    )
    out = tmp_path / "step1_0.inp"
    build_input(
        xyz_path=tmp_path / "step1_0.xyz",
        template_path=template,
        output_path=out,
        charge=0,
        multiplicity=1,
    )
    text = out.read_text()
    assert 'GUEST "/abs/cl.xyz"' in text
    assert '"not-a-file.xyz"' in text


def test_build_input_requests_orca_property_json(tmp_path: Path):
    """Generated inputs ask ORCA (>= 6) for its native property.json artifact."""
    template = _template(tmp_path, "! B3LYP def2-SVP\n")
    out = tmp_path / "step1_0.inp"
    build_input(
        xyz_path=tmp_path / "step1_0.xyz",
        template_path=template,
        output_path=out,
        charge=0,
        multiplicity=1,
    )
    assert "JSONPropFile True" in out.read_text()


def test_build_input_respects_template_jsonpropfile_override(tmp_path: Path):
    """A template that already sets JSONPropFile wins — no second %output block."""
    template = _template(tmp_path, "! B3LYP def2-SVP\n%output\n  JSONPropFile False\nend\n")
    out = tmp_path / "step1_0.inp"
    build_input(
        xyz_path=tmp_path / "step1_0.xyz",
        template_path=template,
        output_path=out,
        charge=0,
        multiplicity=1,
    )
    text = out.read_text()
    assert "JSONPropFile False" in text
    assert "JSONPropFile True" not in text
