"""Tests for the ORCA template inspector (``engines/orca/inspect.py``).

The inspector lets a step omit ``operation``: ChemRefine reads the template's
keywords to pick the parser and flag TS / frequency runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chemrefine.engines.orca.inspect import inspect_template


def _write(tmp_path: Path, body: str) -> Path:
    template = tmp_path / "step1.inp"
    template.write_text(body, encoding="utf-8")
    return template


@pytest.mark.parametrize(
    ("body", "operation"),
    [
        ("! GOAT XTB\n* xyzfile 0 1 geom.xyz\n", "goat"),
        ("! DOCKER\n", "docker"),
        ("! SOLVATOR\n", "solvator"),
        ("! B3LYP def2-SVP Opt\n%geom Scan B 0 1 = 1.0, 2.0, 10 end\nend\n", "pes"),
        ("! B3LYP def2-SVP Opt Freq\n", "opt_sp"),
        ("! B3LYP def2-SVP OptTS\n", "opt_sp"),
        ("! PBE def2-SVP Freq\n", "sp"),  # frequency-only: a single point + Hessian
        ("! HF def2-SVP\n", "sp"),  # no run-type keyword → ORCA's SP fallback
    ],
)
def test_inspect_picks_parser_from_keywords(tmp_path: Path, body: str, operation: str):
    assert inspect_template(_write(tmp_path, body)).operation == operation


def test_inspect_is_case_insensitive(tmp_path: Path):
    assert inspect_template(_write(tmp_path, "! goat xtb\n")).operation == "goat"
    assert inspect_template(_write(tmp_path, "! GoAt XtB\n")).operation == "goat"


def test_inspect_empty_template_is_single_point(tmp_path: Path):
    assert inspect_template(_write(tmp_path, "* xyzfile 0 1 geom.xyz\n")).operation == "sp"


@pytest.mark.parametrize(
    "keyword",
    [
        "Opt",
        "OptTS",
        "COpt",
        "ExtOpt",
        "SloppyOpt",
        "LooseOpt",
        "NormalOpt",
        "TightOpt",
        "VeryTightOpt",
    ],
)
def test_every_optimisation_keyword_orca_accepts_reads_as_an_optimisation(
    tmp_path: Path, keyword: str
):
    """The list is what ORCA 6.1.1 actually takes, checked against the binary.

    Guessing here fails silently in both directions: an accepted keyword this does not
    recognise picks the wrong parser and, through the NMS frequency gate, admits or refuses
    a step for a reason that is not true.
    """
    assert inspect_template(_write(tmp_path, f"! HF STO-3G {keyword}\n")).operation == "opt_sp"


@pytest.mark.parametrize(
    "keyword", ["TightOptTS", "VeryTightOptTS", "LooseOptTS", "COptTS", "ExtOptTS"]
)
def test_a_ts_search_is_only_ever_the_bare_keyword(tmp_path: Path, keyword: str):
    """ORCA rejects every prefixed `…OptTS` spelling, so none can reach a real run.

    This is what lets `is_ts` test for the exact token: a tightness level for a saddle-point
    search is written as its own keyword (`! OptTS TightOpt`), read here as two tokens.
    """
    assert inspect_template(_write(tmp_path, f"! HF STO-3G {keyword}\n")).is_ts is False
    assert inspect_template(_write(tmp_path, f"! HF STO-3G OptTS {keyword}\n")).is_ts is True


def test_inspect_flags_optts_as_ts(tmp_path: Path):
    run = inspect_template(_write(tmp_path, "! B3LYP def2-SVP OptTS Freq\n"))
    # OptTS still parses like a normal opt (single optimized structure)…
    assert run.operation == "opt_sp"
    # …but is flagged as a TS search and a frequency run.
    assert run.is_ts is True
    assert run.has_freq is True


def test_inspect_plain_opt_is_not_ts_and_not_freq(tmp_path: Path):
    run = inspect_template(_write(tmp_path, "! B3LYP def2-SVP Opt\n"))
    assert run.is_ts is False
    assert run.has_freq is False


def test_inspect_detects_numfreq(tmp_path: Path):
    assert inspect_template(_write(tmp_path, "! PBE def2-SVP NumFreq\n")).has_freq is True


def test_inspect_ignores_keywords_in_non_bang_lines(tmp_path: Path):
    # 'goat' only counts on a ``!`` keyword line, not in a coordinate comment.
    body = "! B3LYP def2-SVP Opt\n# a goat wandered through the docker yard\n"
    assert inspect_template(_write(tmp_path, body)).operation == "opt_sp"


def test_inspect_ignores_inline_comment_keywords(tmp_path: Path):
    """An inline ``#`` comment on a ``!`` line is stripped before keyword scanning."""
    run = inspect_template(_write(tmp_path, "! B3LYP def2-SVP Opt  # goat solvator freq later\n"))
    assert run.operation == "opt_sp"  # 'goat'/'solvator' in the comment are ignored
    assert run.has_freq is False  # so is the commented-out 'freq'


def test_inspect_ignores_commented_out_scan_block(tmp_path: Path):
    """A ``%geom Scan`` hidden behind comments is not treated as a PES scan."""
    body = "! B3LYP def2-SVP Opt\n# %geom Scan B 0 1 = 1.0, 2.0, 10 end end\n"
    assert inspect_template(_write(tmp_path, body)).operation == "opt_sp"


def test_inspect_keeps_a_hash_inside_a_quoted_filename(tmp_path: Path):
    """A ``#`` in a quoted path is part of the filename, not the start of a comment.

    Cutting the line there truncated whatever followed, so a template naming
    ``"lig#3.xyz"`` lost the rest of its own line — here the `Freq` that gates NMS and the
    `end` that closes the scan block. The step was then classified with the wrong parser, or
    refused NMS for a reason that was not true: a chemistry-shaped failure caused by a
    character in a filename.
    """
    run = inspect_template(_write(tmp_path, '! B3LYP def2-SVP Opt "lig#3.xyz" Freq\n'))
    assert run.has_freq is True, "the keyword after the quoted '#' was swallowed"
    assert run.operation == "opt_sp"


def test_inspect_still_strips_a_real_comment_after_a_quoted_string(tmp_path: Path):
    """Quote tracking must not cost us actual comment stripping."""
    run = inspect_template(_write(tmp_path, '! B3LYP def2-SVP Opt "lig#3.xyz"  # Freq later\n'))
    assert run.has_freq is False


# ---------------------------------------------------------------------------
# Keyword matching is by whole token, checked against the real templates
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_extopt_is_an_optimisation():
    """ORCA's ``ExtOpt`` runs its optimiser over an external program's gradients.

    Read from a real tutorial template rather than a fabricated one: this is the
    keyword every ``mlip-extopt`` / ``pyscf-extopt`` step uses, and tokenising the
    keyword line without allowing for it reclassified all of them as single points.
    """
    template = REPO_ROOT / "examples/tutorials/redox/amines/templates/step2.inp"
    assert "!ExtOpt" in template.read_text()
    assert inspect_template(template).operation == "opt_sp"


def test_every_shipped_extopt_template_is_an_optimisation():
    """Whatever the tutorials use, the classification has to hold for all of it."""
    templates = sorted(
        p
        for root in ("tests/data/e2e/cases", "examples")
        for p in (REPO_ROOT / root).rglob("templates/*.inp")
        if any(
            line.lstrip().lstrip("!").split()[:1] == ["ExtOpt"]
            for line in p.read_text(errors="replace").splitlines()
            if line.lstrip().startswith("!")
        )
    )
    assert templates, "no ExtOpt templates found — has the layout moved?"
    assert {inspect_template(p).operation for p in templates} == {"opt_sp"}


def test_a_keyword_merely_containing_opt_is_not_an_optimisation(tmp_path: Path):
    """The point of matching whole tokens: substrings must not decide the run type."""
    template = tmp_path / "step1.inp"
    template.write_text("! B3LYP def2-SVP Optimizer-Is-Not-A-Keyword\n", encoding="utf-8")
    assert inspect_template(template).operation == "sp"
