"""Tests for what an xtb step computes: the options model, the command line, the template rule.

The spellings pinned here are xtb 6.7.1's own: the run types, the opt levels it knows (an
unknown one it runs as ``normal`` without a word), and the reference states it accepts after
a solvent (``bar1M`` and ``reference``; ``gsolv`` is its default and no argument at all).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chemrefine.engines.xtb import command
from chemrefine.engines.xtb import input as xtb_input
from chemrefine.engines.xtb.options import XtbOptions
from chemrefine.errors import ConfigError
from chemrefine.ids import StepTemplate


def _args(operation: str = "sp", charge: int = 0, multiplicity: int = 1, **options: object):
    return command.arguments(
        XtbOptions.from_raw(options), operation=operation, charge=charge, multiplicity=multiplicity
    )


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operation", "run_type"),
    [
        ("sp", ["--grad"]),
        ("opt_sp", ["--opt", "--grad"]),
        ("freq", ["--hess", "--grad"]),
        ("opt_freq", ["--ohess", "--grad"]),
    ],
)
def test_each_operation_runs_its_run_type_with_a_gradient(operation: str, run_type: list[str]):
    """Every run asks for the gradient, so every structure carries forces."""
    assert list(_args(operation)) == [*run_type, "--gfn", "2", "--chrg", "0", "--uhf", "0"]


@pytest.mark.parametrize(
    ("method", "flags"),
    [
        ("gfn2", ["--gfn", "2"]),
        ("gfn1", ["--gfn", "1"]),
        ("gfn0", ["--gfn", "0"]),
        ("gfnff", ["--gfnff"]),
    ],
)
def test_the_method_is_always_written(method: str, flags: list[str]):
    """Never left to the binary's default."""
    assert list(_args(method=method)[1:-4]) == flags


def test_multiplicity_is_given_as_unpaired_electrons():
    """``--uhf`` counts unpaired electrons: a doublet has one, a triplet two."""
    assert _args(charge=1, multiplicity=2)[-4:] == ("--chrg", "1", "--uhf", "1")
    assert command.unpaired_electrons(3) == 2


def test_solvation_follows_the_method_with_its_reference_state():
    """``--alpb SOLVENT [STATE]``; ``gsolv`` is xtb's default and adds no word."""
    assert _args(solvation="alpb", solvent="water")[3:6] == ("--alpb", "water", "--chrg")
    assert _args(solvation="gbsa", solvent="thf", solvation_reference="bar1M")[3:6] == (
        "--gbsa",
        "thf",
        "bar1M",
    )
    gsolv = _args(solvation="alpb", solvent="water", solvation_reference="gsolv")
    assert gsolv == _args(solvation="alpb", solvent="water")


def test_cpcmx_runs_a_single_point_without_a_gradient():
    """xtb's CPCM-X has no gradient: ``--scc`` instead, and only as ``sp``."""
    assert _args(solvation="cpcmx", solvent="1,2-dichloroethane")[:5] == (
        "--scc",
        "--gfn",
        "2",
        "--cpcmx",
        "1,2-dichloroethane",
    )
    with pytest.raises(ValueError, match="runs only as `operation: sp`"):
        _args("opt_sp", solvation="cpcmx", solvent="water")


def test_the_opt_level_follows_the_optimising_run_type_only():
    """``--opt LEVEL`` / ``--ohess LEVEL``; a run that does not optimise never carries one."""
    assert _args("opt_sp", opt_level="tight")[:3] == ("--opt", "tight", "--grad")
    assert _args("opt_freq", opt_level="crude")[:3] == ("--ohess", "crude", "--grad")
    assert "tight" not in _args("freq", opt_level="tight")


def test_accuracy_and_electronic_temperature_are_passed_when_set():
    """Unset, xtb's own defaults stand."""
    words = _args(accuracy=0.01, etemp=1000.0)
    assert words[3:7] == ("--acc", "0.01", "--etemp", "1000.0")
    assert "--acc" not in _args() and "--etemp" not in _args()


def test_an_operation_without_a_run_type_is_refused():
    """A caller that skipped the preflight learns it here, not from xtb."""
    with pytest.raises(ValueError, match="no run type for operation 'md'"):
        _args("md")


def test_the_vocabulary_is_the_run_types():
    """The engine declares exactly what the command line can run."""
    from chemrefine.engines.xtb.engine import XtbEngine

    assert set(XtbEngine.operations) == set(command.RUN_TYPES)
    assert set(command.RUN_TYPES) >= command.OPTIMISING | command.FREQUENCY


# ---------------------------------------------------------------------------
# The options model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"solvent": "water"}, "set together"),
        ({"solvation": "alpb"}, "set together"),
        ({"solvation_reference": "bar1M"}, "for `solvation: alpb` or `gbsa`"),
        (
            {"solvation": "cpcmx", "solvent": "water", "solvation_reference": "bar1M"},
            "for `solvation: alpb` or `gbsa`",
        ),
        ({"solvation": "alpb", "solvent": "water; rm -rf ~"}, "not a solvent name"),
        ({"solvation": "alpb", "solvent": "two words"}, "not a solvent name"),
        ({"solvation": "alpb", "solvent": "water", "solvation_reference": "bar1m"}, "bar1M"),
        ({"opt_level": "bogus"}, "opt_level"),
        ({"opt_level": "verytight"}, "opt_level"),
        ({"method": "gfn3"}, "method"),
        ({"accuracy": 0}, "accuracy"),
        ({"etemp": -1}, "etemp"),
    ],
)
def test_a_value_xtb_would_misread_is_refused(raw: dict[str, object], match: str):
    """Each one is a job xtb would run as something else, or one that could not run at all."""
    with pytest.raises(ConfigError, match=match):
        XtbOptions.from_raw(raw)


@pytest.mark.parametrize(
    "solvent", ["water", "CH2Cl2", "1,2-dichloroethane", "N,N-dimethylformamide", "o-xylene"]
)
def test_solvent_names_from_both_tables_are_accepted(solvent: str):
    """The ALPB/GBSA names and the Minnesota-database names CPCM-X reads."""
    assert XtbOptions.from_raw({"solvation": "alpb", "solvent": solvent}).solvent == solvent


def test_the_cores_are_not_part_of_what_a_job_computes():
    """Raising the thread count over a finished step stays a cache hit."""
    assert "cores" in XtbOptions.unkeyed()
    assert {"method", "solvation", "solvent"}.isdisjoint(XtbOptions.unkeyed())


# ---------------------------------------------------------------------------
# The template rule
# ---------------------------------------------------------------------------


def _template(text: str) -> StepTemplate:
    return StepTemplate(path=Path("step1.xcontrol"), text=text)


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("$chrg 1\n$end\n", r"sets \$chrg"),
        ("$spin 2\n$end\n", r"sets \$spin"),
        ("$gfn\n   method=1\n$end\n", r"sets `method` in \$gfn"),
        ("$gbsa\n   solvent=water\n$end\n", r"sets `solvent` in \$gbsa"),
        ("$solvation\n  solvent: thf\n$end\n", r"sets `solvent` in \$solvation"),
        ("$CHRG 1\n", r"sets \$chrg"),
    ],
)
def test_a_template_that_sets_what_the_command_line_sets_is_refused(text: str, said: str):
    """xtb's command line wins over its xcontrol input, so the template's value would be lost."""
    with pytest.raises(ConfigError, match=said):
        xtb_input.require_fillable(_template(text))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "$opt\n   maxcycle=5\n$end\n",
        "$constrain\n   force constant=0.5\n   distance: 1, 2, auto\n$end\n",
        "$gfn\n   scc=true\n$end\n",
        "$gbsa\n   gbsagrid=tight\n$end\n",
        "# $chrg 1 in a comment\n$opt\n  maxcycle=3 # $spin 2\n$end\n",
        "$end\n$chrg 1\n",
        "   method=1\n",
    ],
)
def test_what_only_xcontrol_can_say_is_accepted(text: str):
    """Constraints, optimiser and SCC settings, other keys of the same groups; comments and
    whatever follows ``$end`` are not instructions; a key outside any group sets nothing."""
    xtb_input.require_fillable(_template(text))


def test_each_structures_input_is_the_template(tmp_path: Path):
    """Nothing in an xcontrol file is per structure; a missing final newline is added."""
    out = xtb_input.build_input(
        template=_template("$opt\n$end"), output_path=tmp_path / "a.xcontrol"
    )
    assert out.read_text(encoding="utf-8") == "$opt\n$end\n"
    empty = xtb_input.build_input(template=_template(""), output_path=tmp_path / "b.xcontrol")
    assert empty.read_text(encoding="utf-8") == ""


def test_the_geometry_suffix_is_the_input_geometrys():
    """The run block names ``${stem}_inp.xyz``, the file ``prepare`` writes."""
    from chemrefine.ids import input_geometry_path

    assert xtb_input.GEOMETRY_SUFFIX == "_inp.xyz"
    assert input_geometry_path(Path("d"), 3, "7-1").name == f"step3_7-1{xtb_input.GEOMETRY_SUFFIX}"
