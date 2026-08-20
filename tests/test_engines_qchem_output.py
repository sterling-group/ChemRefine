"""Tests for the Q-Chem output reader (``engines/qchem/output.py``).

The synthetic snippets restate the exact layout of a real Q-Chem output
(``IQmol3/samples/Acetaldehyde-Freq.out`` is the reference); the trimmed real file itself
arrives with the engine's contract fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from chemrefine.engines.qchem.output import parse_qchem_text
from chemrefine.errors import OutputParseError

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
