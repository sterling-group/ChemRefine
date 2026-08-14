"""Tests for the Q-Chem input writer (``engines/qchem/input.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemrefine.engines.qchem.input import build_input
from chemrefine.errors import ConfigError
from chemrefine.io import write_single_xyz


def _seed_xyz(tmp_path: Path) -> Path:
    """A two-atom seed geometry, written the way the engine's prepare does."""
    return write_single_xyz(
        [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)], tmp_path / "step1_0_inp.xyz"
    )


def _render(tmp_path: Path, template_text: str, *, charge: int = 0, multiplicity: int = 1) -> str:
    """Render ``template_text`` against the seed geometry and return the result."""
    template = tmp_path / "step1.in"
    template.write_text(template_text, encoding="utf-8")
    out = build_input(
        xyz_path=_seed_xyz(tmp_path),
        template_path=template,
        output_path=tmp_path / "0" / "step1_0.in",
        charge=charge,
        multiplicity=multiplicity,
    )
    return out.read_text(encoding="utf-8")


def test_the_first_molecule_block_is_replaced_in_place(tmp_path: Path):
    """The template's geometry gives way to the structure's, in the block's own position."""
    text = _render(
        tmp_path,
        "$molecule\n0 1\nHe 0.0 0.0 0.0\n$end\n\n$rem\n  jobtype sp\n$end\n",
    )
    assert "He" not in text
    assert "H  0.000000 0.000000 0.740000" in text
    assert text.index("$molecule") < text.index("$rem"), "the block must not move"
    assert text.count("$molecule") == 1


def test_a_template_without_a_molecule_block_gets_one_prepended(tmp_path: Path):
    """No ``$molecule`` in the template → the generated block becomes job 1's first section."""
    text = _render(tmp_path, "$rem\n  jobtype sp\n$end\n")
    assert text.startswith("$molecule\n0 1\n")
    assert "$rem" in text


def test_later_jobs_read_directive_survives(tmp_path: Path):
    """Only job 1's block is generated; a ``@@@`` chain's ``$molecule read $end`` stands."""
    text = _render(
        tmp_path,
        "$molecule\n0 1\nHe 0.0 0.0 0.0\n$end\n$rem\n  jobtype opt\n$end\n"
        "\n@@@\n\n$molecule\nread\n$end\n$rem\n  jobtype freq\n$end\n",
    )
    assert "H  0.000000 0.000000 0.740000" in text
    assert "$molecule\nread\n$end" in text
    assert "He" not in text


def test_a_job1_read_is_overwritten(tmp_path: Path):
    """A job-1 ``read`` has nothing to read from in a fresh per-structure scratch."""
    text = _render(tmp_path, "$molecule\nread\n$end\n$rem\n  jobtype sp\n$end\n")
    assert "H  0.000000 0.000000 0.740000" in text
    assert "read" not in text.split("$rem")[0]


def test_charge_and_multiplicity_render_into_the_block(tmp_path: Path):
    """The pipeline's charge/multiplicity land on the block's first line."""
    text = _render(tmp_path, "$rem\n  jobtype sp\n$end\n", charge=-1, multiplicity=3)
    assert "$molecule\n-1 3\n" in text


def test_a_missing_template_is_a_config_error(tmp_path: Path):
    """The defensive guard mirrors ORCA's: a named-but-absent template names itself."""
    with pytest.raises(ConfigError, match="Q-Chem template not found"):
        build_input(
            xyz_path=_seed_xyz(tmp_path),
            template_path=tmp_path / "missing.in",
            output_path=tmp_path / "out.in",
            charge=0,
            multiplicity=1,
        )
