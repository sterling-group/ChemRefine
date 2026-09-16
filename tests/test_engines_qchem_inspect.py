"""Tests for the Q-Chem template inspector (``engines/qchem/inspect.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_a_final_vibrational_analysis_in_geom_opt_computes_frequencies(tmp_path: Path):
    """Q-Chem 6's one-job opt→freq: ``$geom_opt final_vibrational_analysis true``.

    Recent real inputs spell the workflow this way rather than as an ``@@@`` chain, and
    the output then carries the full VIBRATIONAL ANALYSIS — so the NMS gate has to open
    for it exactly as it does for ``jobtype freq``.
    """
    info = inspect_template(
        _write(
            tmp_path,
            "$rem\n  JOBTYPE  opt\n  METHOD   wB97M-V\n$end\n\n"
            "$geom_opt\n  INITIAL_HESSIAN  exact\n  FINAL_VIBRATIONAL_ANALYSIS  true\n$end\n",
        )
    )
    assert info.has_freq and info.operation == "opt_sp" and not info.is_ts


@pytest.mark.parametrize(
    "block",
    [
        "$geom_opt\n  final_vibrational_analysis = false\n$end\n",
        "$geom_opt\n  ! final_vibrational_analysis true\n$end\n",
        "$comment\nthe $geom_opt block's final_vibrational_analysis true is the idiom\n$end\n",
    ],
)
def test_a_final_vibrational_analysis_off_commented_or_in_prose_declares_nothing(
    tmp_path: Path, block: str
):
    """``false``, a ``!`` comment and a ``$comment`` mention all leave the gate shut."""
    info = inspect_template(_write(tmp_path, "$rem\n  jobtype opt\n$end\n\n" + block))
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


def test_run_type_inference_is_opt_sp_or_sp(tmp_path: Path):
    """opt/ts anywhere in the chain -> ``opt_sp``; anything else -> ``sp``.

    The parser-dispatch key when a step omits ``operation:``. ``freq`` is deliberately
    never inferred — frequencies parse off the output unconditionally and ``has_freq``
    carries the fact.
    """
    assert inspect_template(_write(tmp_path, "$rem\n  jobtype opt\n$end\n")).operation == "opt_sp"
    assert inspect_template(_write(tmp_path, "$rem\n  jobtype ts\n$end\n")).operation == "opt_sp"
    assert inspect_template(_write(tmp_path, "$rem\n  jobtype freq\n$end\n")).operation == "sp"
    assert inspect_template(_write(tmp_path, "$rem\n  jobtype sp\n$end\n")).operation == "sp"
    chain = "$rem\n  jobtype opt\n$end\n\n@@@\n\n$rem\n  jobtype freq\n$end\n"
    assert inspect_template(_write(tmp_path, chain)).operation == "opt_sp"


def test_every_jobtype_in_the_chain_is_reported_lower_cased(tmp_path: Path):
    """The raw vocabulary travels on the info, so a refusal can say what inferred what."""
    chain = "$rem\n  JOBTYPE = OPT\n$end\n\n@@@\n\n$rem\n  jobtype Freq\n$end\n"
    assert inspect_template(_write(tmp_path, chain)).jobtypes == frozenset({"opt", "freq"})


@pytest.mark.parametrize(
    ("jobtype", "operation"),
    [
        ("PES_SCAN", "pes"),  # a relaxed surface scan, in ChemRefine's own word
        ("rpath", "irc"),  # an intrinsic reaction coordinate
        ("aimd", "md"),  # a trajectory
        ("fsm", "fsm"),  # the string methods keep their names
        ("gsm", "gsm"),
        ("pimc", "sp"),  # unlisted: runs as a single point, never refused
    ],
)
def test_a_multi_geometry_jobtype_infers_an_engine_neutral_operation(
    tmp_path: Path, jobtype: str, operation: str
):
    """A scan, path, string or trajectory names its run kind in ChemRefine's own words.

    Inferring ``sp`` for them filed the last frame as the structure's result with nothing
    said; naming the operation lets the output fan out into its geometries once the
    dispatch has a reader for the word. A job type not listed still infers ``sp`` and
    runs — the inspector never refuses a template for naming a feature the program has.
    """
    info = inspect_template(_write(tmp_path, f"$rem\n  jobtype {jobtype}\n$end\n"))
    assert info.operation == operation and info.jobtypes == frozenset({jobtype.lower()})


def test_a_scan_after_an_optimisation_is_still_a_scan(tmp_path: Path):
    """The output's shape is the multi-geometry job's, wherever it sits in the chain."""
    chain = "$rem\n  jobtype opt\n$end\n\n@@@\n\n$rem\n  jobtype pes_scan\n$end\n"
    assert inspect_template(_write(tmp_path, chain)).operation == "pes"


def test_a_comment_block_naming_rem_facts_declares_nothing(tmp_path: Path):
    """``$rem`` and ``jobtype ts`` in a ``$comment``'s prose are words, not directives.

    Only ``!`` comments are stripped before the scan, so the ``$rem`` regex has to be
    line-anchored or a comment mentioning it opens a phantom block that ends at the
    comment's own ``$end`` — and its prose becomes the step's JOBTYPE facts.
    """
    info = inspect_template(
        _write(
            tmp_path,
            "$comment\nthis template's $rem sets jobtype ts eventually\n$end\n\n"
            "$rem\n  jobtype sp\n$end\n",
        )
    )
    assert not info.is_ts
    assert not info.has_freq
