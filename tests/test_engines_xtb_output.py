"""Tests for the xtb output package (``engines/xtb/output/``): each reader, then the coordinator.

The snippets are cut from real xtb 6.7.1 outputs, the same runs the contract fixtures under
``tests/data/engines/xtb/`` were trimmed from; the whole-run records are the contract test's.
These hold the edges: a run cut off, xtb's fatal-error block, a file stdout names that is not
there, a file left by another calculation, and rows that do not read.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from chemrefine.engines.xtb.output import frequencies, geometry, gradient, parse_xtb, stdout
from chemrefine.errors import OutputParseError, OutputTerminationError
from chemrefine.quantities import BOHR_TO_ANGSTROM, HARTREE_PER_BOHR_TO_EV_PER_A

DATA = Path(__file__).resolve().parent / "data" / "engines" / "xtb"

_FATAL_CPCMX = """\
########################################################################
[ERROR] Program stopped due to fatal error
-2- Command line argument parsing failed
-1- prog_main_parseArguments: The CPCM-X library was not included in this version of xTB.
########################################################################
"""
"""The whole stdout of ``--cpcmx water`` on a binary built without CPCM-X."""

_ENERGY_BOX = """\
           -------------------------------------------------
          | TOTAL ENERGY               -5.085021243116 Eh   |
          | HOMO-LUMO GAP              14.762819677288 eV   |
           -------------------------------------------------
"""

_FINISHED = """\
------------------------------------------------------------------------
 * finished run on 2026/10/23 at 16:44:40.659
------------------------------------------------------------------------
"""

_THERMO = """\
       T/K    H(0)-H(T)+PV         H(T)/Eh          T*S/Eh         G(T)/Eh
   ------------------------------------------------------------------------
    100.00    0.100000E-02    0.200000E-01    0.300000E-02    0.170000E-01
    298.15    0.378305E-02    0.234562E-01    0.214115E-01    0.204479E-02
   ------------------------------------------------------------------------

         :::::::::::::::::::::::::::::::::::::::::::::::::::::
         ::                  THERMODYNAMIC                  ::
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
         :: total free energy          -5.082976455523 Eh   ::
         ::.................................................::
         :: total energy               -5.085021243116 Eh   ::
         :: zero point energy           0.019673198040 Eh   ::
         :: G(RRHO) w/o ZPVE           -0.017628410448 Eh   ::
         :: G(RRHO) contrib.            0.002044787593 Eh   ::
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
"""
"""Water's summary at 298.15 K, with a 100 K row added above it as ``$thermo`` would print."""


def _stage(tmp_path: Path, case: str) -> Path:
    """A contract case's files in a fresh directory; the stdout's path."""
    target = tmp_path / case
    shutil.copytree(DATA / case, target)
    return target / "step1_0.out"


def _parse(out: Path) -> list:
    """The coordinator over ``out`` and the input geometry beside it."""
    return parse_xtb(out, input_geometry=out.with_name("step1_0_inp.xyz"))


# ---------------------------------------------------------------------------
# stdout
# ---------------------------------------------------------------------------


def test_the_last_energy_box_is_the_energy():
    """An optimisation prints the box again after its last cycle; the last one is final."""
    earlier = _ENERGY_BOX.replace("-5.085021243116", "-5.000000000000")
    assert stdout.parse_total_energy(earlier + _ENERGY_BOX) == -5.085021243116
    assert stdout.parse_total_energy("no energy here") is None


def test_termination_is_read_from_stdouts_own_endings():
    """The closing line, or xtb's fatal block, or neither: a run cut off is not a clean end."""
    assert stdout.parse_terminated_normally(_ENERGY_BOX + _FINISHED) is True
    assert stdout.parse_terminated_normally(_ENERGY_BOX) is False
    assert stdout.parse_terminated_normally(_FATAL_CPCMX) is False
    assert stdout.parse_terminated_normally(_FINISHED + _FATAL_CPCMX) is False


def test_xtbs_reasons_are_read_outermost_first():
    """Every ``-N-`` line under the banner, in xtb's order, and nothing past the rule."""
    assert stdout.fatal_reasons(_FATAL_CPCMX + "-9- not a reason\n") == (
        "Command line argument parsing failed",
        "prog_main_parseArguments: The CPCM-X library was not included in this version of xTB.",
    )
    assert stdout.fatal_reasons(_ENERGY_BOX) == ()
    # A warning prints its reason the same way, but without the banner it is no fatal error.
    assert stdout.fatal_reasons("-1- gfnff_setup: Could not read topology file.\n") == ()


def test_the_last_optimisation_verdict_wins():
    """``--ohess`` and a restarted optimisation print more than one; the last is the run's."""
    converged = "   *** GEOMETRY OPTIMIZATION CONVERGED AFTER 4 ITERATIONS ***\n"
    failed = "   *** FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 2 ITERATIONS ***\n"
    assert stdout.parse_converged(converged) is True
    assert stdout.parse_converged(failed) is False
    assert stdout.parse_converged(failed + converged) is True
    assert stdout.parse_converged(converged + failed) is False
    assert stdout.parse_converged(_ENERGY_BOX) is None


def test_the_files_stdout_says_xtb_wrote_are_named():
    """The evidence the coordinator reads a sibling file on."""
    text = "optimized geometry written to: xtbopt.xyz\n writing <g98.out> molden fake output.\n"
    assert stdout.written_geometry(text) == "xtbopt.xyz"
    assert stdout.written_frequencies(text) == "g98.out"
    assert stdout.written_geometry(_ENERGY_BOX) is None
    assert stdout.written_frequencies(_ENERGY_BOX) is None


def test_thermochemistry_is_taken_at_the_summarys_temperature():
    """G and E+ZPE from the summary; H from the table row whose G(T) is the summary's."""
    e = -5.085021243116
    thermo = stdout.parse_thermochemistry(_THERMO, electronic_hartree=e)
    assert thermo is not None
    assert thermo.gibbs_hartree == -5.082976455523
    assert thermo.energy_zpe_hartree == pytest.approx(e + 0.019673198040, abs=1e-15)
    assert thermo.enthalpy_hartree == pytest.approx(e + 0.0234562, abs=1e-15)
    assert thermo.enthalpy_hartree is not None
    assert thermo.enthalpy_hartree > thermo.gibbs_hartree, "H - TS = G, with S > 0"


def test_an_enthalpy_no_row_matches_is_left_unknown():
    """A table that cannot be tied to the summary, or none at all, gives no enthalpy."""
    e = -5.0
    unmatched = _THERMO.replace("0.204479E-02", "0.999999E-02")
    assert stdout.parse_thermochemistry(unmatched, electronic_hartree=e).enthalpy_hartree is None
    summary = _THERMO[_THERMO.index("         ::::") :]
    assert stdout.parse_thermochemistry(summary, electronic_hartree=e).enthalpy_hartree is None
    assert stdout.parse_thermochemistry(_ENERGY_BOX, electronic_hartree=e) is None


# ---------------------------------------------------------------------------
# gradient, geometry, g98.out
# ---------------------------------------------------------------------------

_GRADIENT = """\
$grad
  cycle =      1    SCF energy =    -5.08487182319   |dE/dxyz| =  0.006329
    0.00000000000000      0.00000000000000      0.22676713495445      O
    0.00000000000000      1.43619185471153     -0.88817127857161      H
    0.00000000000000     -1.43619185471153     -0.88817127857161      H
   1.3174938250541E-17  -1.8094238989079E-17   3.5163861354515E-03
  -6.1882567773108E-18  -3.2789187073597D-03  -1.7581930677257E-03
  -6.9880224962838E-18   3.2789187073597E-03  -1.7581930677258E-03
$end
"""


def test_the_last_gradient_cycle_is_read_in_its_own_units():
    """xtb appends a cycle to a file already there; Fortran ``D`` exponents read too."""
    second = _GRADIENT.replace("cycle =      1", "cycle =      2").replace("-5.084871", "-5.090000")
    cycle = gradient.parse_gradient_text(_GRADIENT.replace("$end\n", "") + second)
    assert cycle.energy_hartree == -5.09000082319
    assert cycle.symbols == ("O", "H", "H")
    assert cycle.positions_bohr[1, 1] == 1.43619185471153
    assert cycle.gradient_hartree_per_bohr[1, 1] == -3.2789187073597e-03


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("$grad\n$end\n", "no `cycle =` block"),
        (
            _GRADIENT.replace(
                "  -6.9880224962838E-18   3.2789187073597E-03  -1.7581930677258E-03\n", ""
            ),
            "2 gradient rows for 3 atoms",
        ),
        (_GRADIENT.replace("3.5163861354515E-03", "nan"), "non-finite"),
        (_GRADIENT.replace("3.5163861354515E-03", "*****"), "could not convert"),
        (
            _GRADIENT.replace("-1.7581930677258E-03\n", "-1.7581930677258E-03 1.0\n"),
            "unreadable row",
        ),
    ],
)
def test_a_gradient_that_cannot_be_the_molecules_is_refused(text: str, match: str):
    """Cut off, non-finite, overflowed or misshapen: never a smaller or NaN gradient."""
    with pytest.raises(ValueError, match=match):
        gradient.parse_gradient_text(text)


_XTBOPT = """\
3
 energy: -5.085021243116 gnorm: 0.000112069562 xtb: 6.7.1 (26b2801)
O            0.00000000000000        0.00000000018710        0.10755539192635
H            0.00000000000000        0.77460455837201       -0.46377769589053
H           -0.00000000000000       -0.77460455855911       -0.46377769603582
"""


def test_an_xyz_geometry_is_read_in_angstrom():
    """xtb's extended-XYZ comment is not read; the rows are."""
    symbols, positions = geometry.parse_xyz_text(_XTBOPT)
    assert symbols == ("O", "H", "H")
    assert positions[0, 2] == 0.10755539192635


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "no atom count"),
        ("three\n\n", "no atom count"),
        (_XTBOPT.rsplit("H", 1)[0], "2 coordinate rows for 3 atoms"),
        (_XTBOPT.replace("0.10755539192635", "inf"), "non-finite"),
        (_XTBOPT.replace("0.10755539192635", ""), "unreadable row"),
        ("0\n\n", "0 coordinate rows for 0 atoms"),
    ],
)
def test_a_geometry_that_cannot_be_the_molecules_is_refused(text: str, match: str):
    """Cut off, empty, or non-finite: never a smaller molecule or a NaN position."""
    with pytest.raises(ValueError, match=match):
        geometry.parse_xyz_text(text)


def _g98(case: str) -> str:
    return (DATA / case / "step1_0.g98.out").read_text(encoding="utf-8")


def test_the_mode_index_puts_the_vibrations_after_the_trivial_modes():
    """Six translations and rotations for ammonia, five for linear CO2."""
    imaginary, table, tensor = frequencies.parse_g98_text(_g98("freq"), n_atoms=4)
    assert sorted(table) == list(range(6, 12))
    assert list(imaginary) == [6] and imaginary[6] < 0, "the umbrella mode of planar NH3"
    assert tensor.shape == (4, 3, 12) and not tensor[:, :, :6].any()
    _imag, linear, _tensor = frequencies.parse_g98_text(_g98("linear"), n_atoms=3)
    assert sorted(linear) == [5, 6, 7, 8]


def test_each_mode_is_scaled_back_to_unit_length():
    """Two printed decimals leave a vector a little short; NMS displaces by its length."""
    _imag, table, tensor = frequencies.parse_g98_text(_g98("opt_freq"), n_atoms=6)
    norms = np.linalg.norm(tensor.reshape(-1, tensor.shape[2]), axis=0)
    assert np.allclose(norms[sorted(table)], 1.0) and not norms[:6].any()


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda t: t.replace("Frequencies --", "Frequencies:"), "no `Frequencies --` row"),
        (lambda t: t.replace(" Atom AN", " Atom number"), "no `Atom AN`"),
        (
            lambda t: t.replace("Frequencies --  -870.2191", "Frequencies --  nan"),
            "could not|not finite",
        ),
        (lambda t: t.rsplit("\n   4   1", 1)[0] + "\n", "3 displacement rows for 4 atoms"),
    ],
)
def test_a_frequency_file_that_cannot_be_the_molecules_is_refused(edit, match: str):
    """A block that does not read is the file's failure, not a shorter table."""
    with pytest.raises(ValueError, match=match):
        frequencies.parse_g98_text(edit(_g98("freq")), n_atoms=4)


def test_a_frequency_file_with_too_many_modes_or_a_bad_row_is_refused():
    """More modes than 3N - 5, a row of the wrong width, a mode of zero length."""
    with pytest.raises(ValueError, match="vibrational modes for 1 atoms"):
        frequencies.parse_g98_text(_g98("freq").replace("   2   1", "   9   1"), n_atoms=1)
    text = _g98("linear")
    lines = text.splitlines()
    row = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("Atom AN")) + 1
    widened = "\n".join([*lines[:row], lines[row] + " 0.00", *lines[row + 1 :]])
    with pytest.raises(ValueError, match="unreadable displacement row"):
        frequencies.parse_g98_text(widened, n_atoms=3)
    zeroed = "\n".join(
        [
            *lines[:row],
            *(
                " ".join(ln.split()[:2] + ["0.00"] * (len(ln.split()) - 2))
                for ln in lines[row : row + 3]
            ),
            *lines[row + 3 :],
        ]
    )
    with pytest.raises(ValueError, match="has no displacement"):
        frequencies.parse_g98_text(zeroed, n_atoms=3)
    infinite = text.replace(lines[row].split()[2], "inf", 1)
    with pytest.raises(ValueError, match="not finite"):
        frequencies.parse_g98_text(infinite, n_atoms=3)


# ---------------------------------------------------------------------------
# The coordinator
# ---------------------------------------------------------------------------


def test_forces_are_the_negative_gradient_in_ev_per_angstrom(tmp_path: Path):
    """ASE's sign and units, from the gradient file of the structure reported."""
    out = _stage(tmp_path, "sp")
    (parsed,) = _parse(out)
    cycle = gradient.parse_gradient_text(
        (out.parent / "step1_0.gradient").read_text(encoding="utf-8")
    )
    assert np.array_equal(
        parsed.forces_ev_per_a, -cycle.gradient_hartree_per_bohr * HARTREE_PER_BOHR_TO_EV_PER_A
    )
    assert np.allclose(cycle.positions_bohr * BOHR_TO_ANGSTROM, parsed.positions, atol=1e-5)


def test_a_run_with_no_gradient_file_has_no_forces(tmp_path: Path):
    """A CPCM-X single point writes no gradient; that is not an error."""
    out = _stage(tmp_path, "sp")
    (out.parent / "step1_0.gradient").unlink()
    (parsed,) = _parse(out)
    assert parsed.forces_ev_per_a is None and parsed.energy_hartree is not None


def test_a_gradient_of_another_geometry_is_refused(tmp_path: Path):
    """A file left by an earlier run beside the output is not attached as this run's forces."""
    out = _stage(tmp_path, "sp")
    shutil.copy(DATA / "opt" / "step1_0.gradient", out.parent / "step1_0.gradient")
    with pytest.raises(OutputParseError, match="gradient of another geometry"):
        _parse(out)


def test_a_gradient_of_another_calculation_is_refused(tmp_path: Path):
    """Same atoms, same geometry, another energy: another method or solvent computed it."""
    out = _stage(tmp_path, "sp")
    path = out.parent / "step1_0.gradient"
    text = path.read_text(encoding="utf-8")
    energy = text.split("SCF energy =")[1].split()[0]
    path.write_text(text.replace(energy, f"{float(energy) + 0.01:.11f}"), encoding="utf-8")
    with pytest.raises(OutputParseError, match="gradient of another calculation"):
        _parse(out)


def test_a_malformed_gradient_file_is_the_outputs_failure(tmp_path: Path):
    """The reader's ``ValueError`` reaches the lifecycle as the structure's parse failure."""
    out = _stage(tmp_path, "sp")
    (out.parent / "step1_0.gradient").write_text("$grad\n$end\n", encoding="utf-8")
    with pytest.raises(OutputParseError, match=r"malformed step1_0\.gradient"):
        _parse(out)


def test_a_stale_optimised_geometry_is_not_read_for_a_single_point(tmp_path: Path):
    """``xtbopt.xyz`` is read only when stdout says this run wrote it."""
    out = _stage(tmp_path, "sp")
    (out.parent / "step1_0.xtbopt.xyz").write_text(_XTBOPT, encoding="utf-8")
    (parsed,) = _parse(out)
    assert len(parsed.symbols) == 6, "the input geometry, not the stale water"


def test_the_xtboptok_file_is_no_verdict(tmp_path: Path):
    """xtb 6.7.1 leaves ``.xtboptok`` behind an optimisation that did not converge."""
    out = _stage(tmp_path, "opt_unconverged")
    (out.parent / ".xtboptok").touch()
    (parsed,) = _parse(out)
    assert parsed.converged is False and parsed.terminated_normally is True


@pytest.mark.parametrize(
    ("case", "missing", "match"),
    [
        ("opt", "step1_0.xtbopt.xyz", r"no optimised geometry step1_0\.xtbopt\.xyz"),
        ("sp", "step1_0_inp.xyz", r"no input geometry step1_0_inp\.xyz"),
        ("freq", "step1_0.g98.out", r"no frequency file step1_0\.g98\.out"),
    ],
)
def test_a_file_stdout_depends_on_that_is_missing_is_refused(
    tmp_path: Path, case: str, missing: str, match: str
):
    """Said as the output's failure, naming the file, never a structure without it."""
    out = _stage(tmp_path, case)
    (out.parent / missing).unlink()
    with pytest.raises(OutputParseError, match=match):
        _parse(out)


@pytest.mark.parametrize(
    ("case", "name", "edit", "match"),
    [
        ("opt", "step1_0.xtbopt.xyz", lambda t: "x" + t, "malformed optimised geometry"),
        (
            "freq",
            "step1_0.g98.out",
            lambda t: t.replace("Frequencies --", "F:"),
            "malformed step1_0.g98.out",
        ),
    ],
)
def test_a_file_stdout_depends_on_that_does_not_read_is_refused(
    tmp_path: Path, case: str, name: str, edit, match: str
):
    """The reader's ``ValueError``, filed as the output's with the file named."""
    out = _stage(tmp_path, case)
    path = out.parent / name
    path.write_text(edit(path.read_text(encoding="utf-8")), encoding="utf-8")
    with pytest.raises(OutputParseError, match=match):
        _parse(out)


def test_a_fatal_error_is_the_runs_failure_with_xtbs_reasons(tmp_path: Path):
    """The CPCM-X-missing stdout: a termination failure carrying xtb's own words."""
    out = tmp_path / "step1_0.out"
    out.write_text(_FATAL_CPCMX, encoding="utf-8")
    with pytest.raises(
        OutputTerminationError,
        match=r"xtb stopped: Command line argument parsing failed; .*CPCM-X library was not "
        r"included.* — no TOTAL ENERGY in .*step1_0\.out",
    ):
        _parse(out)


def test_a_run_cut_off_before_its_energy_is_the_runs_failure(tmp_path: Path):
    """No banner, no closing line: xtb did not finish, which is what the reason says."""
    out = tmp_path / "step1_0.out"
    out.write_text("   * xtb version 6.7.1 (26b2801)\n", encoding="utf-8")
    with pytest.raises(OutputTerminationError, match="xtb did not finish — no TOTAL ENERGY"):
        _parse(out)


def test_a_finished_run_with_no_energy_is_a_parse_failure(tmp_path: Path):
    """A clean ending with nothing to read is the parser's problem to report, not the run's."""
    out = tmp_path / "step1_0.out"
    out.write_text(_FINISHED, encoding="utf-8")
    with pytest.raises(OutputParseError, match="no TOTAL ENERGY") as raised:
        _parse(out)
    assert not isinstance(raised.value, OutputTerminationError)


def test_an_unreadable_sibling_leaves_as_the_filesystems_error(tmp_path: Path):
    """Only a missing file is the output's; any other refusal is the ``OSError`` it is."""
    out = _stage(tmp_path, "sp")
    (out.parent / "step1_0.gradient").unlink()
    (out.parent / "step1_0.gradient").mkdir()
    with pytest.raises(IsADirectoryError):
        _parse(out)
    shutil.rmtree(out.parent / "step1_0.gradient")
    (out.parent / "step1_0_inp.xyz").unlink()
    (out.parent / "step1_0_inp.xyz").mkdir()
    with pytest.raises(IsADirectoryError):
        _parse(out)


def test_src_names_the_output_in_every_reason(tmp_path: Path):
    """The coordinator's ``src`` is what a reason calls the output."""
    out = _stage(tmp_path, "opt")
    (out.parent / "step1_0.xtbopt.xyz").unlink()
    with pytest.raises(OutputParseError, match=r"in 0/step1_0\.out$"):
        parse_xtb(out, input_geometry=out.with_name("step1_0_inp.xyz"), src="0/step1_0.out")


def test_files_cut_off_at_their_last_row_still_read_to_the_end():
    """A gradient with no ``$end`` and a fatal block with no closing rule are read whole."""
    cycle = gradient.parse_gradient_text(_GRADIENT.replace("$end\n", ""))
    assert cycle.gradient_hartree_per_bohr.shape == (3, 3)
    cut = _FATAL_CPCMX.rsplit("####", 1)[0].rstrip("#\n") + "\n"
    assert len(stdout.fatal_reasons(cut)) == 2


def test_a_frequency_row_with_no_values_is_refused():
    """A block whose row reads as nothing would shift every later mode's index."""
    with pytest.raises(ValueError, match="empty frequency row"):
        frequencies.parse_g98_text(
            re.sub(r"(?m)^ Frequencies --.*$", " Frequencies --", _g98("freq"), count=1),
            n_atoms=4,
        )
