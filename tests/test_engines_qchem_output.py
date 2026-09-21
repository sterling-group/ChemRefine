"""Tests for the Q-Chem output package (``engines/qchem/output/``).

The synthetic snippets restate the exact layout of a real Q-Chem output
(``IQmol3/samples/Acetaldehyde-Freq.out`` is the reference); the trimmed real file itself
arrives with the engine's contract fixture. The package holds full readers beside
sections that answer "not reported" until they parse — the tests below pin *today's*
``None`` answers and the seams a full reader lands in, so implementing one flips
documented tests rather than silence.
"""

from __future__ import annotations

import numpy as np
import pytest

from chemrefine.engines.qchem.output import energy as qchem_energy
from chemrefine.engines.qchem.output import forces as qchem_forces
from chemrefine.engines.qchem.output import known_operations, parse_output, parse_qchem_text
from chemrefine.engines.qchem.output import status as qchem_status
from chemrefine.errors import OutputParseError, OutputTerminationError

_ORIENTATION = """\
       Standard Nuclear Orientation (Angstroms)
    I     Atom         X            Y            Z
 ----------------------------------------------------
    1      H      -1.713730     0.219516     0.880805
    2      O      -1.171140    -0.148534     0.000000
 ----------------------------------------------------
"""

_ENERGY = " Total energy in the final basis set = -153.8301110890\n"

_FREQ_BLOCK = """\
 **                       VIBRATIONAL ANALYSIS                       **

 Mode:                 1                      2                      3
 Frequency:      -151.64                 505.89                 778.03
 Force Cnst:      0.0168                 0.3835                 0.4013
 Red. Mass:       1.2403                 2.5435                 1.1251
 IR Active:          YES                    YES                    YES
 IR Intens:        0.278                 12.882                  0.658
 Raman Active:       YES                    YES                    YES
               X      Y      Z        X      Y      Z        X      Y      Z
 H         -0.279  0.388 -0.343    0.013  0.333 -0.015    0.453  0.202  0.137
 O          0.000  0.000 -0.004   -0.166  0.024  0.000   -0.000  0.000 -0.070
 TransDip  -0.000  0.000  0.017   -0.088 -0.074  0.000    0.000  0.000  0.026
"""


def test_last_energy_and_last_geometry_win():
    """Optimisation cycles re-print both; the final pair is the result — ORCA's discipline."""
    early = _ORIENTATION.replace("-1.713730", "99.000000")
    text = early + " Total energy in the final basis set = -1.0\n" + _ORIENTATION + _ENERGY
    parsed = parse_qchem_text(text)
    assert len(parsed) == 1
    assert parsed[0].energy_hartree == -153.8301110890
    assert parsed[0].symbols == ("H", "O")
    assert parsed[0].positions[0][0] == -1.713730
    assert parsed[0].forces_ev_per_a is None
    assert parsed[0].converged is None and parsed[0].terminated_normally is None


def test_a_missing_energy_is_unparseable():
    """No final energy, no result — the step's failure machinery takes it from there."""
    with pytest.raises(OutputParseError, match="Total energy in the final basis set"):
        parse_qchem_text(_ORIENTATION)


def test_a_missing_geometry_is_unparseable():
    with pytest.raises(OutputParseError, match="Standard Nuclear Orientation"):
        parse_qchem_text(_ENERGY)


def test_a_bohr_orientation_refuses_rather_than_misreads():
    """The banner match pins ``(Angstroms)`` — a Bohr block is *no* block, never Å.

    Q-Chem prints ``Standard Nuclear Orientation (Bohr)`` under ``input_bohr``, and a
    match without the unit consumed those values straight into ``ParsedResult.positions``
    (contract: Å) — a geometry silently wrong by 0.529 everywhere downstream, with no
    later check able to see it. Refusing makes it an ordinary parse failure; the input
    writer refuses the rem itself one layer earlier. Same rule as ORCA's ``(ANGSTROEM)``.
    """
    bohr = _ORIENTATION.replace("(Angstroms)", "(Bohr)")
    with pytest.raises(OutputParseError, match=r"Standard Nuclear Orientation \(Angstroms\)"):
        parse_qchem_text(bohr + _ENERGY)


def test_a_corrupt_coordinate_is_unparseable():
    """A ``*****`` overflow token becomes a per-file parse failure, not a bare ValueError."""
    with pytest.raises(OutputParseError, match="malformed coordinate row"):
        parse_qchem_text(_ORIENTATION.replace("-1.713730", "*********") + _ENERGY)


@pytest.mark.parametrize("literal", ["NaN", "inf", "-inf"])
def test_a_non_finite_coordinate_is_unparseable(literal: str):
    """A diverged geometry is the same failure as the overflow above.

    ``float()`` rejects ``*****`` but accepts these, so they parsed into a ``Structure``
    and reached the cache sidecar, which stores coordinates without inspecting them.
    """
    with pytest.raises(OutputParseError, match="non-finite coordinate"):
        parse_qchem_text(_ORIENTATION.replace("-1.713730", literal) + _ENERGY)


def test_no_frequency_section_reads_as_none():
    """``None`` = no section at all; distinct from a section with zero imaginary modes."""
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY)[0]
    assert parsed.imaginary_freqs is None
    assert parsed.normal_modes is None


def test_a_negative_frequency_is_imaginary_at_the_shifted_index():
    """Q-Chem's 1-based vibrational mode k lands at tensor index k+5 — the NMS contract.

    Q-Chem prints only the 3N-6 vibrational modes; ChemRefine's tensor carries the six
    trivial modes first, so mode 1's -151.64 cm**-1 is imaginary_freqs[6].
    """
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY + _FREQ_BLOCK)[0]
    assert parsed.imaginary_freqs == {6: -151.64}


def test_the_tensor_is_zero_padded_and_indexed_by_the_shift():
    """Six inert zero columns lead; each printed column lands at its shifted index."""
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY + _FREQ_BLOCK)[0]
    modes = parsed.normal_modes
    assert modes is not None and modes.shape == (2, 3, 9)  # 6 trivial + modes 1..3
    assert not modes[:, :, :6].any(), "the trivial-mode padding must be zero"
    assert modes[0, :, 6] == pytest.approx([-0.279, 0.388, -0.343])  # mode 1, atom 1
    assert modes[1, :, 8] == pytest.approx([-0.000, 0.000, -0.070])  # mode 3, atom 2


def test_a_second_mode_block_extends_the_tensor():
    """Blocks of three accumulate: mode 4 sits at tensor index 9."""
    second = (
        " Mode:                 4\n"
        " Frequency:       892.35\n"
        " IR Active:          YES\n"
        " Raman Active:       YES\n"
        "               X      Y      Z\n"
        " H          0.440  0.379 -0.045\n"
        " O          0.253 -0.014 -0.000\n"
        " TransDip   0.001  0.002  0.003\n"
    )
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY + _FREQ_BLOCK + second)[0]
    assert parsed.imaginary_freqs == {6: -151.64}
    modes = parsed.normal_modes
    assert modes is not None and modes.shape == (2, 3, 10)
    assert modes[0, :, 9] == pytest.approx([0.440, 0.379, -0.045])


def test_the_whole_table_lands_in_the_same_shifted_index_space():
    """The real modes are kept too, shifted exactly like the imaginary ones.

    Q-Chem builds the full table and used to throw away everything that was not negative,
    which is why ``analyze_mode`` could name a mode's frequency for ORCA and not here. Both
    now come from one shift, so the subset cannot drift out of the table.
    """
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY + _FREQ_BLOCK)[0]
    assert parsed.frequencies is not None
    assert parsed.frequencies == {6: -151.64, 7: 505.89, 8: 778.03}
    assert parsed.imaginary_freqs is not None
    assert parsed.imaginary_freqs.items() <= parsed.frequencies.items()


def test_no_vibrational_section_means_no_table_either():
    """``None`` for all three, so "not computed" never reads as "nothing found"."""
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY)[0]
    assert (parsed.imaginary_freqs, parsed.frequencies, parsed.normal_modes) == (None, None, None)


def test_an_all_real_spectrum_is_a_verified_minimum():
    """A frequency section with no negative values yields ``{}`` — counted and zero."""
    text = _ORIENTATION + _ENERGY + _FREQ_BLOCK.replace("-151.64", " 151.64")
    parsed = parse_qchem_text(text)[0]
    assert parsed.imaginary_freqs == {}
    assert parsed.normal_modes is not None


def test_the_transdip_row_is_not_a_displacement_row():
    """``TransDip`` matches the row width exactly; the name test is what excludes it."""
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY + _FREQ_BLOCK)[0]
    modes = parsed.normal_modes
    assert modes is not None
    assert modes[1, :, 6] == pytest.approx([0.000, 0.000, -0.004])  # atom 2, not TransDip
    assert not np.isclose(modes[1, 2, 6], 0.017)  # TransDip's z never enters


def test_the_transdip_exclusion_is_decisive_on_a_short_table():
    """A table missing one atom row must not absorb ``TransDip`` as the missing atom.

    On the complete table above the name test is never decisive: the row loop stops at
    ``n_atoms`` rows before TransDip is ever evaluated, so deleting the exclusion clause
    passed that test — and every other, since the truncation cases all cut *before*
    TransDip. Here one atom row is gone and TransDip has exactly the right width
    (1 + 3·n_modes tokens): without the name test it becomes the final "atom" and the
    NMS tensor silently carries a transition-dipole vector as an atomic displacement.
    """
    o_row = " O          0.000  0.000 -0.004   -0.166  0.024  0.000   -0.000  0.000 -0.070\n"
    short = _FREQ_BLOCK.replace(o_row, "")  # atom 2 lost; TransDip still follows
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY + short)[0]
    assert parsed.imaginary_freqs == {6: -151.64}  # the spectrum survives
    assert parsed.normal_modes is None, (
        "one real row + TransDip is not two atoms — the tensor must be withheld"
    )


# ---------------------------------------------------------------------------
# Failure paths — the coverage gate's findings, each pinned
# ---------------------------------------------------------------------------


def test_an_orientation_block_at_end_of_text_still_parses():
    """A file cut right after the atom rows (no closing separator) keeps its geometry."""
    no_separator = _ORIENTATION.rstrip().rsplit("\n", 1)[0]
    parsed = parse_qchem_text(_ENERGY + no_separator)
    assert parsed[0].symbols == ("H", "O")


def test_an_orientation_block_with_no_atoms_is_unparseable():
    """The marker alone proves nothing; zero rows must fail, not return an empty molecule."""
    with pytest.raises(OutputParseError, match="has no atoms"):
        parse_qchem_text(_ENERGY + "       Standard Nuclear Orientation (Angstroms)\n done\n")


def test_a_mode_block_without_its_frequency_row_is_skipped():
    """A ``Mode:`` line with no ``Frequency:`` beneath it contributes nothing, quietly.

    And a section where *no* block contributed reads ``None``, not ``{}``: this test once
    pinned ``{}`` here, which is the "verified minimum" verdict — handed out for a section
    that parsed no data at all.
    """
    text = (
        _ORIENTATION + _ENERGY + " **  VIBRATIONAL ANALYSIS  **\n Mode:                 1\n done\n"
    )
    parsed = parse_qchem_text(text)[0]
    assert parsed.imaginary_freqs is None
    assert parsed.normal_modes is None


def test_a_frequency_row_of_the_wrong_width_skips_the_block():
    """Two mode indices with one value cannot be zipped honestly — the block is dropped."""
    text = (
        _ORIENTATION
        + _ENERGY
        + " **  VIBRATIONAL ANALYSIS  **\n"
        + " Mode:                 1                      2\n"
        + " Frequency:      -151.64\n"
    )
    parsed = parse_qchem_text(text)[0]
    assert parsed.imaginary_freqs is None  # nothing parsed = no data, not a minimum
    assert parsed.normal_modes is None


def test_a_truncated_section_is_no_data_not_a_minimum():
    """Killed right after the header, the section must not read as 0 imaginary modes.

    Q-Chem can exit 0 on an internal error and the status banners are parsed elsewhere,
    so ``{}`` here would flow through ``get_frequencies`` as ``imaginary_count: 0`` — a
    verified minimum, from a job that never printed a single mode.
    """
    text = _ORIENTATION + _ENERGY + " **  VIBRATIONAL ANALYSIS  **\n"
    parsed = parse_qchem_text(text)[0]
    assert parsed.imaginary_freqs is None
    assert parsed.normal_modes is None


def test_a_block_without_a_displacement_table_keeps_its_frequencies():
    """A block cut short by the next ``Mode:`` line still contributes its frequencies.

    Its tensor column stays zero-padded; the block that does carry a table fills its own.
    """
    text = (
        _ORIENTATION
        + _ENERGY
        + " **  VIBRATIONAL ANALYSIS  **\n"
        + " Mode:                 1\n"
        + " Frequency:      -100.00\n"
        + " Mode:                 2\n"
        + " Frequency:       500.00\n"
        + " Raman Active:       YES\n"
        + "               X      Y      Z\n"
        + " H          0.100  0.200  0.300\n"
        + " O          0.400  0.500  0.600\n"
    )
    parsed = parse_qchem_text(text)[0]
    assert parsed.imaginary_freqs == {6: -100.00}
    modes = parsed.normal_modes
    assert modes is not None and modes.shape == (2, 3, 8)
    assert not modes[:, :, 6].any(), "the table-less mode's column stays zero"
    assert modes[0, :, 7] == pytest.approx([0.100, 0.200, 0.300])


def test_a_corrupt_displacement_token_drops_the_tensor_not_the_frequencies():
    """A non-float token of the right row width loses the tensor; the spectrum survives."""
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY + _FREQ_BLOCK.replace("-0.279", "*.279*"))[0]
    assert parsed.imaginary_freqs == {6: -151.64}
    assert parsed.normal_modes is None


def test_a_non_finite_displacement_token_drops_the_tensor_too():
    """A diverged displacement is withheld like a corrupt one, and for a sharper reason.

    This tensor is what ``nms.displace_along_mode`` adds to a parent's coordinates, so a
    non-finite component would not merely be stored — it would put NaN into the geometry of
    every displaced child the NMS round builds.
    """
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY + _FREQ_BLOCK.replace("-0.279", "nan"))[0]
    assert parsed.imaginary_freqs == {6: -151.64}
    assert parsed.normal_modes is None


def test_a_truncated_displacement_table_drops_the_tensor():
    """Fewer rows than atoms at end-of-file cannot be reshaped — the tensor is withheld."""
    lines = _FREQ_BLOCK.splitlines()
    truncated = "\n".join(
        lines[
            : lines.index(
                " H         -0.279  0.388 -0.343    0.013  0.333 -0.015    0.453  0.202  0.137"
            )
            + 1
        ]
    )
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY + truncated)[0]
    assert parsed.imaginary_freqs == {6: -151.64}
    assert parsed.normal_modes is None


def test_a_blank_line_inside_the_displacement_table_is_skipped():
    """Formatting noise between rows (a blank line) is passed over, not read as a row."""
    header_row = "               X      Y      Z        X      Y      Z        X      Y      Z\n"
    spaced = _FREQ_BLOCK.replace(header_row, header_row + "\n")
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY + spaced)[0]
    modes = parsed.normal_modes
    assert modes is not None and modes.shape == (2, 3, 9)
    assert modes[0, :, 6] == pytest.approx([-0.279, 0.388, -0.343])


# ---------------------------------------------------------------------------
# The operation dispatch — one table, every known key on the one assembler
# ---------------------------------------------------------------------------


def test_every_known_operation_routes_to_the_assembler(tmp_path):
    """``sp`` / ``opt_sp`` / ``freq`` all parse the same output the same way, today.

    The uniformity is deliberate (the coordinator's docstring says why); this pins that
    naming any of them — in either spelling — is never a behavior fork until a parser
    genuinely diverges.
    """
    out = tmp_path / "step1_0.out"
    out.write_text(_ORIENTATION + _ENERGY, encoding="utf-8")
    assert known_operations() == {"sp", "opt_sp", "freq"}
    for operation in (*known_operations(), "OPT+SP"):
        [parsed] = parse_output(out, operation)
        assert parsed.energy_hartree == -153.8301110890


def test_an_unknown_operation_is_refused_by_name(tmp_path):
    """A vocabulary miss is a config mistake, raised before any behavior is guessed at.

    The preflight (`QchemEngine.check_step`) refuses it at t=0; this is the same refusal
    for a caller that skipped the preflight.
    """
    out = tmp_path / "step1_0.out"
    out.write_text(_ORIENTATION + _ENERGY, encoding="utf-8")
    with pytest.raises(OutputParseError, match="unknown Q-Chem operation"):
        parse_output(out, "goat")


# ---------------------------------------------------------------------------
# The unparsed sections — today's "not reported" answers, pinned
# ---------------------------------------------------------------------------
#
# Each of these fails the moment its section starts answering, which is the point:
# implementing a reader flips a documented test naming the contract, never silence.


_CLEAN_EXIT = "        Thank you very much for using Q-Chem.  Have a nice day.\n"
_FATAL = " Q-Chem fatal error occurred in module x\n"


def test_status_answers_not_reported_even_on_banner_text():
    """``None`` whatever the text says — the banners are the *contract*, not yet the code.

    A full status reader takes the clean-exit and fatal banners (pinned against full
    captured output, never guessed) and makes these three answers True/False/False.
    """
    assert qchem_status.parse_terminated_normally(_ENERGY + _CLEAN_EXIT) is None
    assert qchem_status.parse_terminated_normally(_ENERGY + _FATAL) is None
    assert qchem_status.parse_converged(_ENERGY) is None


def test_thermochemistry_answers_not_reported():
    """``None`` even over a thermodynamics-shaped section — the reader does not parse yet."""
    text = _ENERGY + " Zero point vibrational energy:      34.675 kcal/mol\n"
    assert qchem_energy.parse_thermochemistry_from_text(text, electronic_hartree=-1.0) is None


def test_forces_answer_not_reported():
    """``None`` even over a gradient-shaped section — the reader does not parse yet."""
    text = _ENERGY + " Gradient of SCF Energy\n  1  0.001  0.002  0.003\n"
    assert qchem_forces.parse_forces_from_text(text, n_atoms=1) is None


# ---------------------------------------------------------------------------
# The seams — the coordinator threads each section's answer where it belongs,
# proven by substituting a real answer, so a full reader lands already wired.
# ---------------------------------------------------------------------------


def test_a_dead_run_upgrades_a_missing_section_once_status_can_say_so(monkeypatch):
    """The unreadable-vs-dead distinction is wired; only the verdict is missing.

    A died run's unusable output is filed as NOT_TERMINATED_NORMALLY so the reader
    looks at the job, not the parser. The moment `parse_terminated_normally` can answer
    False, the coordinator makes that upgrade — with no edit to it.
    """
    monkeypatch.setattr(qchem_status, "parse_terminated_normally", lambda text: False)
    with pytest.raises(OutputTerminationError, match="did not terminate normally"):
        parse_qchem_text(_ORIENTATION)  # killed before any energy line


def test_the_status_verdicts_land_on_the_parsed_structure(monkeypatch):
    """Both flags flow to the fields `lifecycle.succeeded` reads."""
    monkeypatch.setattr(qchem_status, "parse_terminated_normally", lambda text: True)
    monkeypatch.setattr(qchem_status, "parse_converged", lambda text: False)
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY)[0]
    assert parsed.terminated_normally is True
    assert parsed.converged is False


def test_thermochemistry_lands_on_the_parsed_structure(monkeypatch):
    """The three energies flow to the fields the filters and `steps.csv` read."""

    def fake(text: str, *, electronic_hartree: float) -> qchem_energy.Thermochemistry:
        return qchem_energy.Thermochemistry(
            gibbs_hartree=-153.70,
            enthalpy_hartree=-153.65,
            energy_zpe_hartree=electronic_hartree + 0.03,
        )

    monkeypatch.setattr(qchem_energy, "parse_thermochemistry_from_text", fake)
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY)[0]
    assert parsed.gibbs_hartree == -153.70
    assert parsed.enthalpy_hartree == -153.65
    assert parsed.energy_zpe_hartree == pytest.approx(-153.8301110890 + 0.03)


def test_forces_land_on_the_parsed_structure(monkeypatch):
    """The gradient's forces flow to `forces_ev_per_a`, atom count threaded through."""
    seen: dict[str, int] = {}

    def fake(text: str, *, n_atoms: int) -> np.ndarray:
        seen["n_atoms"] = n_atoms
        return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

    monkeypatch.setattr(qchem_forces, "parse_forces_from_text", fake)
    parsed = parse_qchem_text(_ORIENTATION + _ENERGY)[0]
    assert seen["n_atoms"] == 2  # the geometry block's own count, threaded through
    assert parsed.forces_ev_per_a is not None
    assert parsed.forces_ev_per_a[1] == pytest.approx([0.0, 0.0, -1.0])


def test_a_malformed_gradient_becomes_this_structures_failure(monkeypatch):
    """A ValueError from the forces reader is wrapped, naming the gradient — not a crash.

    The rule the real parser inherits: a bad row must become this structure's ledgered
    failure, and `lifecycle._parse_job` catches only OutputParseError.
    """

    def explode(text: str, *, n_atoms: int) -> np.ndarray:
        raise ValueError("read 1 gradient row(s) for a 2-atom structure")

    monkeypatch.setattr(qchem_forces, "parse_forces_from_text", explode)
    with pytest.raises(OutputParseError, match="malformed gradient row"):
        parse_qchem_text(_ORIENTATION + _ENERGY)
