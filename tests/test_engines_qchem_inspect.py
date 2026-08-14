"""Tests for the Q-Chem template inspector (``engines/qchem/inspect.py``)."""

from __future__ import annotations

from pathlib import Path

from chemrefine.engines.qchem.inspect import inspect_template


def _write(tmp_path: Path, body: str) -> Path:
    template = tmp_path / "step1.in"
    template.write_text(body, encoding="utf-8")
    return template


def test_a_freq_job_anywhere_in_the_chain_computes_frequencies(tmp_path: Path):
    """opt→freq is a multi-job ``@@@`` input whose *second* job carries the freq."""
    info = inspect_template(
        _write(
            tmp_path,
            "$rem\n  jobtype opt\n$end\n\n@@@\n\n$molecule\nread\n$end\n"
            "$rem\n  JOBTYPE = freq\n$end\n",
        )
    )
    assert info.has_freq
    assert not info.is_ts


def test_a_ts_search_is_flagged(tmp_path: Path):
    assert inspect_template(_write(tmp_path, "$rem\n  jobtype ts\n$end\n")).is_ts


def test_a_commented_jobtype_declares_nothing(tmp_path: Path):
    """``!`` comments are stripped first — a commented-out freq must not gate NMS open."""
    info = inspect_template(_write(tmp_path, "$rem\n  ! jobtype freq\n  jobtype sp\n$end\n"))
    assert not info.has_freq


def test_mem_total_is_the_peak_across_the_chain(tmp_path: Path):
    """``@@@`` jobs run one after another, so the max declaration is the run's requirement."""
    info = inspect_template(
        _write(
            tmp_path,
            "$rem\n  jobtype opt\n  mem_total 8000\n$end\n\n@@@\n\n"
            "$rem\n  jobtype freq\n  MEM_TOTAL = 16000\n$end\n",
        )
    )
    assert info.mem_total_mb == 16000


def test_no_mem_total_declares_no_memory(tmp_path: Path):
    """Absence is ``None``, not a default: the header's memory policy stands."""
    assert inspect_template(_write(tmp_path, "$rem\n  jobtype sp\n$end\n")).mem_total_mb is None
