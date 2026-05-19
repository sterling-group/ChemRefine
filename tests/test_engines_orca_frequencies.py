"""Tests for the ORCA frequency parser."""

from __future__ import annotations

from pathlib import Path

from synthetic import FREQUENCY_BLOCK as _SYNTH_BLOCK

from chemrefine.engines.orca.frequencies import (
    parse_frequencies,
    parse_imaginary_frequencies,
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
