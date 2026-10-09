"""Tests for the Q-Chem input writer (``engines/qchem/input.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemrefine.engines.qchem.input import build_input, require_fillable
from chemrefine.errors import ConfigError
from chemrefine.ids import StepTemplate, require_template
from chemrefine.io import write_single_xyz


def _seed_xyz(tmp_path: Path) -> Path:
    """A two-atom seed geometry, written the way the engine's prepare does."""
    return write_single_xyz(
        [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)], tmp_path / "step1_0_inp.xyz"
    )


def _template(tmp_path: Path, template_text: str) -> StepTemplate:
    """``template_text`` as the engine reads a step's template."""
    template = tmp_path / "step1.in"
    template.write_text(template_text, encoding="utf-8")
    return require_template(template, label="Q-Chem")


def _render(tmp_path: Path, template_text: str, *, charge: int = 0, multiplicity: int = 1) -> str:
    """Render ``template_text`` against the seed geometry and return the result."""
    out = build_input(
        xyz_path=_seed_xyz(tmp_path),
        template=_template(tmp_path, template_text),
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


def test_a_chain_whose_first_job_has_no_block_gets_one_in_job_1(tmp_path: Path):
    """The generated block belongs to job 1 even when only a later job names one.

    The first ``$molecule`` in the *file* used to be the target, so a chain that left
    job 1's block to ChemRefine had the geometry spliced over job 2's ``read`` — and job 1
    ran with no molecule at all.
    """
    text = _render(
        tmp_path,
        "$rem\n  jobtype opt\n$end\n\n@@@\n\n$molecule\nread\n$end\n$rem\n  jobtype freq\n$end\n",
    )
    job1, _, job2 = text.partition("@@@")
    assert job1.startswith("$molecule\n0 1\nH  0.000000 0.000000 0.000000\n")
    assert "$molecule\nread\n$end" in job2 and "0.740000" not in job2


@pytest.mark.parametrize("separator", ["--", "---", "----", "  ---  ", "-- fragment 1"])
def test_a_fragment_partitioned_block_is_refused_by_name(tmp_path: Path, separator: str):
    """Fragment separators (EDA / SCFMI) cannot survive a regenerated block.

    The writer builds the block from one whole-molecule geometry and cannot assign the
    new atoms to fragments, so a flattened block would run a different calculation than
    the template describes — refused up front, like ``input_bohr``.

    A separator is a line that starts with two dashes, however it goes on: Q-Chem runs a
    block ruled with ``---`` as the fragment job a ``--`` one is, and a pattern that knew
    only the two-dash spelling let such a template through to be flattened.
    """
    with pytest.raises(ConfigError, match="fragments"):
        require_fillable(
            _template(
                tmp_path,
                f"$molecule\n0 1\n{separator}\n0 1\nHe 0.0 0.0 0.0\n{separator}\n0 1\n"
                "Ne 3.0 0.0 0.0\n$end\n$rem\n  jobtype sp\n  frgm_method stoll\n$end\n",
            )
        )


def test_a_negative_charge_is_not_a_fragment_separator(tmp_path: Path):
    """One dash and a digit is the block's charge line, and the template stays fillable."""
    text = "$molecule\n-1 2\nHe 0.0 0.0 0.0\n$end\n$rem\n  jobtype sp\n$end\n"
    require_fillable(_template(tmp_path, text))
    assert "$molecule\n-1 2\nH  0.000000 0.000000 0.000000\n" in _render(
        tmp_path, text, charge=-1, multiplicity=2
    )


def test_a_later_jobs_fragments_are_its_own(tmp_path: Path):
    """Only job 1's block is regenerated, so a fragmented block in a later job of the chain
    passes through as written."""
    text = (
        "$rem\n  jobtype sp\n$end\n@@@\n"
        "$molecule\n0 1\n--\n0 1\nHe 0.0 0.0 0.0\n--\n0 1\nNe 3.0 0.0 0.0\n$end\n"
        "$rem\n  jobtype sp\n  frgm_method stoll\n$end\n"
    )
    require_fillable(_template(tmp_path, text))
    assert "--\n0 1\nHe 0.0 0.0 0.0" in _render(tmp_path, text)


def test_charge_and_multiplicity_render_into_the_block(tmp_path: Path):
    """The pipeline's charge/multiplicity land on the block's first line."""
    text = _render(tmp_path, "$rem\n  jobtype sp\n$end\n", charge=-1, multiplicity=3)
    assert "$molecule\n-1 3\n" in text


@pytest.mark.parametrize("rem", ["input_bohr true", "INPUT_BOHR  TRUE", "input_bohr = 1"])
def test_an_input_bohr_template_is_refused_by_name(tmp_path: Path, rem: str):
    """``input_bohr`` would read the Å geometry this writer emits as Bohr — refused up front.

    The rendered ``$molecule`` is Ångström (every geometry this package writes is), so a
    template declaring Bohr input has Q-Chem compute on a molecule scaled by 1/0.529 —
    silently wrong science, not a crash. The output parser holds the other half of the
    same rule by pinning ``(Angstroms)`` in its banner match; this is the point of use,
    where the two units would actually meet.
    """
    with pytest.raises(ConfigError, match="input_bohr"):
        require_fillable(_template(tmp_path, f"$rem\n  jobtype sp\n  {rem}\n$end\n"))


def test_a_comment_mentioning_input_bohr_is_not_the_rem(tmp_path: Path):
    """The refusal is line-anchored, so prose about the rem does not trip it."""
    text = (
        "$comment\nnever set input_bohr true here — geometry arrives in Angstrom\n$end\n\n"
        "$rem\n  jobtype sp\n$end\n"
    )
    require_fillable(_template(tmp_path, text))
    assert "jobtype sp" in _render(tmp_path, text)


def test_a_comment_mentioning_the_block_name_is_not_the_block(tmp_path: Path):
    """``$molecule`` in a ``$comment``'s prose must not become the replacement target.

    The block regex once matched the literal text anywhere, so a comment that merely
    *named* the block had the geometry spliced into it — and job 1 then ran the
    template's own placeholder geometry. The shipped starter's comment was exactly such
    a mention (see test_scaffold's cross-render test for that half).
    """
    text = _render(
        tmp_path,
        "$comment\nthe first $molecule block is replaced per structure\n$end\n\n"
        "$molecule\n0 1\nHe 0.0 0.0 0.0\n$end\n\n$rem\n  jobtype sp\n$end\n",
    )
    assert "the first $molecule block is replaced per structure\n$end" in text
    assert "H  0.000000 0.000000 0.740000" in text
    assert "He" not in text
