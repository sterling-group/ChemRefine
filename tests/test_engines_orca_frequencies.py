"""Tests for the ORCA frequency parser + normal-mode tensor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synthetic import (
    FREQUENCY_BLOCK as _SYNTH_BLOCK,
)
from synthetic import (
    NORMAL_MODES_BLOCK_2_ATOMS as _SYNTH_MODES,
)

from chemrefine.engines.orca.frequencies import (
    parse_frequencies,
    parse_imaginary_frequencies,
    parse_normal_modes_tensor,
)


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "orca.out"
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_frequencies_skips_first_five_real_modes_by_default(tmp_path: Path):
    """Indices 0..5 are translational/rotational; v3 default drops them."""
    freqs = parse_frequencies(_write(tmp_path, _SYNTH_BLOCK))
    assert 6 in freqs
    assert 7 in freqs
    assert 0 not in freqs
    assert 5 not in freqs


def test_parse_frequencies_includes_imaginary_in_full_pass(tmp_path: Path):
    """The full (non-imag-only) pass still includes imaginary modes."""
    freqs = parse_frequencies(_write(tmp_path, _SYNTH_BLOCK))
    assert freqs[37] == -118.27
    assert freqs[38] == -42.10
    assert freqs[39] == 45.50


def test_parse_frequencies_keeps_actual_values(tmp_path: Path):
    freqs = parse_frequencies(_write(tmp_path, _SYNTH_BLOCK))
    assert freqs[6] == 15.11


def test_parse_imaginary_frequencies_filters_to_imaginary_only(tmp_path: Path):
    imag = parse_imaginary_frequencies(_write(tmp_path, _SYNTH_BLOCK))
    assert set(imag) == {37, 38}
    assert imag[37] == -118.27
    assert imag[38] == -42.10


def test_parse_frequencies_stops_at_blank_line(tmp_path: Path):
    """A blank line ends the frequency block — nothing past it should match."""
    text = _SYNTH_BLOCK + "\n     6:    9999.99 cm**-1  ***imaginary mode***\n"
    imag = parse_imaginary_frequencies(_write(tmp_path, text))
    # The poisoned line is after the trailing blank line, so it should not appear.
    assert 6 not in imag


def test_parse_frequencies_empty_when_no_freq_block(tmp_path: Path):
    text = "no frequency block here\n"
    assert parse_frequencies(_write(tmp_path, text)) == {}


def test_parse_frequencies_missing_file_returns_empty(tmp_path: Path):
    """Skip the freq parser cleanly on outputs that ran but never got to FREQ."""
    p = tmp_path / "no-freq.out"
    p.write_text("FINAL SINGLE POINT ENERGY     -1.0\n", encoding="utf-8")
    assert parse_frequencies(p) == {}


# ---------------------------------------------------------------------------
# parse_normal_modes_tensor
# ---------------------------------------------------------------------------


def test_parse_normal_modes_tensor_shape_and_values(tmp_path: Path):
    """The tensor should have shape (n_atoms, 3, n_modes) and pick up mode 5."""
    out = _write(tmp_path, _SYNTH_MODES)
    tensor = parse_normal_modes_tensor(out, num_atoms=2)
    assert tensor.shape == (2, 3, 6)
    # mode 5 carries (0.1, 0.2, 0.3) on atom 0 and -(0.1, 0.2, 0.3) on atom 1
    np.testing.assert_allclose(tensor[0, :, 5], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(tensor[1, :, 5], [-0.1, -0.2, -0.3])


def test_parse_normal_modes_tensor_raises_when_no_block(tmp_path: Path):
    out = _write(tmp_path, "no normal modes here\n")
    with pytest.raises(ValueError, match="no normal-mode blocks"):
        parse_normal_modes_tensor(out, num_atoms=3)


def test_parse_normal_modes_tensor_raises_on_wrong_atom_count(tmp_path: Path):
    """The 2-atom synthetic block should mismatch a 3-atom expectation."""
    out = _write(tmp_path, _SYNTH_MODES)
    with pytest.raises(ValueError, match="expected 9"):
        parse_normal_modes_tensor(out, num_atoms=3)


def test_parse_normal_modes_tensor_concatenates_multiple_column_blocks(tmp_path: Path):
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
    tensor = parse_normal_modes_tensor(_write(tmp_path, text), num_atoms=1)
    assert tensor.shape == (1, 3, 8)
    assert tensor[0, 0, 6] == 0.9
    assert tensor[0, 0, 7] == 0.5
