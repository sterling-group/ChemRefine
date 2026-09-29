"""Tests for the ORCA frequency table parser + normal-mode tensor (text-based)."""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import (
    FREQUENCY_BLOCK as _SYNTH_BLOCK,
)
from synthetic import (
    NORMAL_MODES_BLOCK_2_ATOMS as _SYNTH_MODES,
)

from chemrefine.engines.orca.output.frequencies import (
    parse_frequencies_from_text,
    parse_imaginary_frequencies_from_text,
    parse_mode_table_from_text,
    parse_normal_modes_tensor_from_text,
)

# A transition state as ORCA 6.1.1 printed one (the HCN/HNC saddle, at GFN2-xTB):
# the translations and rotations first at 0.00, then the vibrations ascending, so the
# imaginary mode leads them at index 6.
_TS_BLOCK = """
-----------------------
VIBRATIONAL FREQUENCIES
-----------------------

Scaling factor for frequencies =  1.000000000  (already applied!)

     0:       0.00 cm**-1
     1:       0.00 cm**-1
     2:       0.00 cm**-1
     3:       0.00 cm**-1
     4:       0.00 cm**-1
     5:       0.00 cm**-1
     6:   -1425.82 cm**-1  ***imaginary mode***
     7:    2000.59 cm**-1
     8:    2386.04 cm**-1


------------
NORMAL MODES
------------
"""

# ---------------------------------------------------------------------------
# parse_frequencies_from_text
# ---------------------------------------------------------------------------


def test_parse_frequencies_skips_first_five_real_modes_by_default():
    """Indices 0..5 are translational/rotational; the parser drops them by default."""
    freqs = parse_frequencies_from_text(_SYNTH_BLOCK)
    assert 6 in freqs
    assert 7 in freqs
    assert 0 not in freqs
    assert 5 not in freqs


def test_parse_frequencies_includes_imaginary_in_full_pass():
    """The full (non-imag-only) pass still includes imaginary modes."""
    freqs = parse_frequencies_from_text(_SYNTH_BLOCK)
    assert freqs[37] == -118.27
    assert freqs[38] == -42.10
    assert freqs[39] == 45.50


def test_parse_frequencies_keeps_actual_values():
    assert parse_frequencies_from_text(_SYNTH_BLOCK)[6] == 15.11


def test_parse_imaginary_frequencies_filters_to_imaginary_only():
    imag = parse_imaginary_frequencies_from_text(_SYNTH_BLOCK)
    assert set(imag) == {37, 38}
    assert imag[37] == -118.27
    assert imag[38] == -42.10


def test_parse_frequencies_stops_at_blank_line():
    """A blank line ends the frequency block — nothing past it should match."""
    text = _SYNTH_BLOCK + "\n     6:    9999.99 cm**-1  ***imaginary mode***\n"
    imag = parse_imaginary_frequencies_from_text(text)
    # The poisoned line is after the trailing blank line, so it should not appear.
    assert 6 not in imag


def test_parse_frequencies_empty_when_no_freq_block():
    assert parse_frequencies_from_text("no frequency block here\n") == {}


def test_a_transition_states_table_holds_its_imaginary_mode_and_its_vibrations():
    """The mode a TS is looked at for, and the real vibrations, under ORCA's own indices."""
    table = parse_mode_table_from_text(_TS_BLOCK)
    assert table == {6: -1425.82, 7: 2000.59, 8: 2386.04}
    assert parse_imaginary_frequencies_from_text(_TS_BLOCK) == {6: -1425.82}


def test_the_mode_table_leaves_the_translations_and_rotations_out():
    """Six modes at 0.00 cm⁻¹ that nobody animates, and that are not the molecule moving."""
    assert not set(parse_mode_table_from_text(_TS_BLOCK)) & {0, 1, 2, 3, 4, 5}


def test_the_imaginary_modes_are_a_subset_of_the_table():
    """The invariant the two callers rely on, on both block shapes.

    ``analyze_mode`` reads ``frequency_cm1`` from the table and ``is_imaginary`` from the
    subset. If a mode could be in the subset and not the table, it would be reported as
    imaginary with no frequency — the null it was changed to stop returning.
    """
    # And by construction, not by layout: a rotation ORCA printed imaginary — a geometry
    # short of a stationary point — sits inside the window the table otherwise skips.
    rotation = _TS_BLOCK.replace(
        "     5:       0.00 cm**-1", "     5:     -35.10 cm**-1  ***imaginary mode***"
    )
    for block in (_SYNTH_BLOCK, _TS_BLOCK, rotation):
        assert parse_imaginary_frequencies_from_text(block).items() <= (
            parse_mode_table_from_text(block).items()
        )


def test_every_selector_reads_the_same_block_when_a_ts_reprints_the_table():
    """One scan, so the three cannot end up describing different Hessians.

    A TS search recomputes the Hessian as it goes and prints a table per recompute; each
    selector takes the last. Splitting the scan per rule is how one of them would come to
    answer from an earlier one — a converged structure reported with the imaginary mode it
    had on the way there.
    """
    stale = _TS_BLOCK.replace("-1425.82", "-1999.99").replace("2000.59", "111.11")
    both = stale + _TS_BLOCK
    assert parse_mode_table_from_text(both) == parse_mode_table_from_text(_TS_BLOCK)
    assert parse_imaginary_frequencies_from_text(both) == {6: -1425.82}
    assert parse_frequencies_from_text(both) == {6: -1425.82, 7: 2000.59, 8: 2386.04}


# ---------------------------------------------------------------------------
# parse_normal_modes_tensor_from_text
# ---------------------------------------------------------------------------


def test_parse_normal_modes_tensor_shape_and_values():
    """The tensor should have shape (n_atoms, 3, n_modes) and pick up mode 5."""
    tensor = parse_normal_modes_tensor_from_text(_SYNTH_MODES, num_atoms=2)
    assert tensor.shape == (2, 3, 6)
    # mode 5 carries (0.1, 0.2, 0.3) on atom 0 and -(0.1, 0.2, 0.3) on atom 1
    np.testing.assert_allclose(tensor[0, :, 5], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(tensor[1, :, 5], [-0.1, -0.2, -0.3])


def test_parse_normal_modes_tensor_raises_when_no_block():
    """The banner is present but carries no column blocks under it."""
    with pytest.raises(ValueError, match="no normal-mode blocks"):
        parse_normal_modes_tensor_from_text("NORMAL MODES\nnothing tabular here\n", num_atoms=3)


def test_parse_normal_modes_tensor_refuses_a_non_finite_component():
    """An overflowing exponent parses to ``inf`` and would ride into every child geometry."""
    text = _SYNTH_MODES.replace("0.100000", "1.0E999", 1)
    with pytest.raises(ValueError, match="non-finite component"):
        parse_normal_modes_tensor_from_text(text, num_atoms=2)


def test_parse_normal_modes_tensor_raises_on_wrong_atom_count():
    """The 2-atom synthetic block should mismatch a 3-atom expectation."""
    with pytest.raises(ValueError, match="expected 9"):
        parse_normal_modes_tensor_from_text(_SYNTH_MODES, num_atoms=3)


def test_parse_normal_modes_tensor_concatenates_multiple_column_blocks():
    """ORCA prints modes in 6-column blocks; the parser must hstack them."""
    text = (
        "NORMAL MODES\n"
        "-----\n"
        "                  0          1          2          3          4          5\n"
        "      0       0.100000   0.000000   0.000000   0.000000   0.000000   0.000000\n"
        "      1       0.000000   0.200000   0.000000   0.000000   0.000000   0.000000\n"
        "      2       0.000000   0.000000   0.300000   0.000000   0.000000   0.000000\n"
        "                  6          7\n"
        "      0       0.900000   0.500000\n"
        "      1       0.000000   0.000000\n"
        "      2       0.000000   0.000000\n"
        "-----\n"
    )
    tensor = parse_normal_modes_tensor_from_text(text, num_atoms=1)
    assert tensor.shape == (1, 3, 8)
    assert tensor[0, 0, 6] == 0.9
    assert tensor[0, 0, 7] == 0.5


def test_parse_normal_modes_tensor_tolerates_blank_line_inside_block():
    """A blank line between the column header and the rows is skipped, not a terminator."""
    text = (
        "NORMAL MODES\n"
        "                  0          1          2          3          4          5\n"
        "\n"
        "      0       0.100000   0.000000   0.000000   0.000000   0.000000   0.000000\n"
        "      1       0.000000   0.200000   0.000000   0.000000   0.000000   0.000000\n"
        "      2       0.000000   0.000000   0.300000   0.000000   0.000000   0.000000\n"
        "-----\n"
    )
    tensor = parse_normal_modes_tensor_from_text(text, num_atoms=1)
    assert tensor.shape == (1, 3, 6)
    assert tensor[0, 0, 0] == 0.1


def test_parse_normal_modes_tensor_raises_when_header_has_no_rows():
    """A column header followed directly by a separator carries no mode data."""
    with pytest.raises(ValueError, match="no normal-mode blocks"):
        parse_normal_modes_tensor_from_text(
            "NORMAL MODES\n                  0          1\n-----\n", num_atoms=1
        )


def test_parse_normal_modes_tensor_ignores_numeric_tables_before_the_banner():
    """The scan is anchored on ``NORMAL MODES``, not on the first table of integers.

    Unanchored, the column-header pattern (two or more integers on a line) matched
    plenty of earlier ORCA tables — symmetry, basis-set summaries, internal
    coordinates. Latching onto one and stopping at its separator gave a
    wrongly-shaped array, which the coordinator swallows into ``modes=None``; every
    structure then failed NMS with "no normal-mode tensor", blaming the frequency job
    instead of the parser.
    """
    decoy = (
        "INTERNAL COORDINATES\n"
        "                  1          2          3\n"
        "      0       9.900000   9.900000   9.900000\n"
        "      1       9.900000   9.900000   9.900000\n"
        "-----\n"
    )
    real = (
        "NORMAL MODES\n"
        "                  0          1          2          3          4          5\n"
        "      0       0.100000   0.000000   0.000000   0.000000   0.000000   0.000000\n"
        "      1       0.000000   0.200000   0.000000   0.000000   0.000000   0.000000\n"
        "      2       0.000000   0.000000   0.300000   0.000000   0.000000   0.000000\n"
        "-----\n"
    )
    tensor = parse_normal_modes_tensor_from_text(decoy + real, num_atoms=1)
    assert tensor.shape == (1, 3, 6)
    assert tensor[0, 0, 0] == 0.1  # the real block, not the decoy's 9.9


def test_parse_normal_modes_tensor_raises_without_the_banner():
    """No ``NORMAL MODES`` section at all is a clear error, not a silent misparse."""
    with pytest.raises(ValueError, match="NORMAL MODES"):
        parse_normal_modes_tensor_from_text(
            "VIBRATIONAL FREQUENCIES\n  0: 1.0 cm**-1\n", num_atoms=1
        )


_LINEAR_BLOCK = """\
-----------------------
VIBRATIONAL FREQUENCIES
-----------------------

Scaling factor for frequencies =  1.000000000  (already applied!)

     0:       0.00 cm**-1
     1:       0.00 cm**-1
     2:       0.00 cm**-1
     3:       0.00 cm**-1
     4:       0.00 cm**-1
     5:    1871.38 cm**-1


Temperature         ...   298.15 K
Total Mass          ...     7.95 AMU
{statement}
"""
"""LiH's frequency table and the thermochemistry after it, as ORCA 6.1.1 printed them."""


def test_a_linear_molecule_keeps_the_vibration_at_index_five():
    """ORCA says when a molecule is linear, and then only indices 0-4 are trivial.

    Read as six trivial modes, a diatomic's only vibration was dropped: its frequency job
    read as "no frequency data", and NMS ledgered a converged minimum unresolved.
    """
    linear = _LINEAR_BLOCK.format(statement="The molecule is recognized as being linear")
    assert parse_mode_table_from_text(linear) == {5: 1871.38}
    assert parse_frequencies_from_text(linear) == {5: 1871.38}


def test_a_molecule_orca_does_not_call_linear_keeps_six_trivial_modes():
    assert parse_mode_table_from_text(_LINEAR_BLOCK.format(statement="")) == {}
