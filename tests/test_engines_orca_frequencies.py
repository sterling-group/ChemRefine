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

from chemrefine.engines.orca.frequencies import (
    parse_frequencies_from_text,
    parse_imaginary_frequencies_from_text,
    parse_normal_modes_tensor_from_text,
)

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
    with pytest.raises(ValueError, match="no normal-mode blocks"):
        parse_normal_modes_tensor_from_text("no normal modes here\n", num_atoms=3)


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
        parse_normal_modes_tensor_from_text("                  0          1\n-----\n", num_atoms=1)
