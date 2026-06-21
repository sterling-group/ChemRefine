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
