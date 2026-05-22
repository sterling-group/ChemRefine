"""Tests for the ORCA output parser, anchored on the real fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemrefine.engines.orca.output import (
    ParsedStructure,
    parse_dft,
    parse_docker,
    parse_forces,
    parse_goat_ensemble,
    parse_output,
    parse_pes,
    parse_solvator,
)
from chemrefine.errors import OutputParseError

DATA = Path(__file__).parent / "data"
FIXTURE = DATA / "orca.out"
GOAT_FIXTURE = DATA / "goat_finalensemble.xyz"
DOCKER_FIXTURE = DATA / "docker_allopt.xyz"
SOLVATOR_FIXTURE = DATA / "solvator_solventbuild.xyz"


# ---------------------------------------------------------------------------
# Fixture sanity
# ---------------------------------------------------------------------------


def test_fixture_present():
    assert FIXTURE.is_file(), f"missing fixture: {FIXTURE}"


# ---------------------------------------------------------------------------
# parse_dft — verified against the real fixture
# ---------------------------------------------------------------------------


def test_parse_dft_returns_single_structure():
    parsed = parse_dft(FIXTURE)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert isinstance(parsed[0], ParsedStructure)


def test_parse_dft_final_energy_matches_fixture():
    """The last FINAL SINGLE POINT ENERGY in the fixture is -6044.555726221861 Hartree.

    The fixture deliberately contains TWO FINAL SP ENERGY lines with different
    values; this test locks in the 'last instance wins' contract by asserting
    the *exact* last value (not the first).
    """
    parsed = parse_dft(FIXTURE)
    assert abs(parsed[0].energy_hartree - (-6044.555726221861)) < 1e-9


def test_parse_dft_picks_last_coord_block_not_first():
    """The fixture's two coord blocks have different positions for atom 0;
    the parser must return the LAST one.
    """
    parsed = parse_dft(FIXTURE)
    # The cycle-1 first-atom x-coord differs from the stationary first-atom x-coord.
    # Cycle-1 block has C at x ≈ -0.1539; the stationary block at x ≈ -0.1556.
    first_atom_x = parsed[0].positions[0][0]
    assert abs(first_atom_x - (-0.155603)) < 1e-4
    assert abs(first_atom_x - (-0.153892)) > 1e-4


def test_parse_dft_coordinates_have_expected_atom_count():
    parsed = parse_dft(FIXTURE)
    # The molecule in the fixture has > 50 atoms (final geometry block).
    assert parsed[0].positions.shape[0] > 50
    assert parsed[0].positions.shape[1] == 3
    assert len(parsed[0].symbols) == parsed[0].positions.shape[0]


def test_parse_dft_symbols_are_strings():
    parsed = parse_dft(FIXTURE)
    assert all(isinstance(s, str) for s in parsed[0].symbols)
    # Fixture contains C, Fe, N, Br at minimum.
    assert {"C", "N"} <= set(parsed[0].symbols)


def test_parse_dft_forces_optional_for_opt_outputs():
    """The fixture is a geometry-opt converged run; gradient may or may not be present."""
    parsed = parse_dft(FIXTURE)
    # Force shape, if present, matches the coord block.
    if parsed[0].forces_ev_per_a is not None:
        assert parsed[0].forces_ev_per_a.shape == parsed[0].positions.shape


# ---------------------------------------------------------------------------
# parse_dft — synthetic minimal cases
# ---------------------------------------------------------------------------


def test_parse_dft_picks_last_energy_when_multiple_appear(tmp_path: Path):
    from synthetic import synthetic_dft_output

    text = synthetic_dft_output(
        [-1.0, -2.0, -3.0],
        [("H", 0.0, 0.0, 0.0), ("H", 0.74, 0.0, 0.0)],
    )
    path = tmp_path / "synth.out"
    path.write_text(text, encoding="utf-8")
    parsed = parse_dft(path)
    assert parsed[0].energy_hartree == -3.0


def test_parse_dft_skips_short_lines_in_coord_block(tmp_path: Path):
    """Lines inside a coord block that don't have at least 4 whitespace-separated
    tokens (e.g. continuation markers, blank lines that survive ``strip``) are skipped."""
    text = (
        "CARTESIAN COORDINATES (ANGSTROEM)\n"
        "---------------------------------\n"
        "  H   0.000000   0.000000   0.000000\n"
        "  ...continuation\n"               # only 1 token, len(parts) < 4
        "  H   0.740000   0.000000   0.000000\n"
        "---------------------------------\n"
        "FINAL SINGLE POINT ENERGY     -1.10\n"
    )
    path = tmp_path / "short.out"
    path.write_text(text, encoding="utf-8")
    parsed = parse_dft(path)
    # Two valid atom rows survive; the short continuation row is dropped.
    assert parsed[0].symbols == ("H", "H")


def test_parse_dft_missing_energy_raises(tmp_path: Path):
    path = tmp_path / "no-energy.out"
    path.write_text(
        "CARTESIAN COORDINATES (ANGSTROEM)\n"
        "---------------------------------\n"
        "  H  0.0 0.0 0.0\n"
        "---------------------------------\n",
        encoding="utf-8",
    )
    with pytest.raises(OutputParseError):
        parse_dft(path)


def test_parse_dft_missing_coords_raises(tmp_path: Path):
    path = tmp_path / "no-coords.out"
    path.write_text("FINAL SINGLE POINT ENERGY     -1.0\n", encoding="utf-8")
    with pytest.raises(OutputParseError):
        parse_dft(path)


# ---------------------------------------------------------------------------
# parse_forces
# ---------------------------------------------------------------------------


def test_parse_forces_returns_none_when_absent():
    assert parse_forces("no gradient here") is None


def test_parse_forces_handles_synthetic_block():
    text = (
        "CARTESIAN GRADIENT\n"
        "------------------\n"
        "   0  H :    0.001000   -0.002000    0.003000\n"
        "   1  H :   -0.004000    0.005000   -0.006000\n"
        "------------------\n"
    )
    forces = parse_forces(text, to_ev_per_A=False)
    assert forces is not None
    assert forces.shape == (2, 3)
    # F = -dE/dx, so first atom's fx should be -0.001
    assert abs(forces[0][0] - (-0.001)) < 1e-9


# ---------------------------------------------------------------------------
# GOAT ensemble (verified against the real fixture)
# ---------------------------------------------------------------------------


def test_parse_goat_ensemble_returns_one_per_frame():
    parsed = parse_goat_ensemble(GOAT_FIXTURE)
    # Trimmed fixture: 3 frames.
    assert len(parsed) == 3


def test_parse_goat_ensemble_first_energy_matches_fixture():
    parsed = parse_goat_ensemble(GOAT_FIXTURE)
    # The fixture's first frame header is ``-199.4369175187 converged=true``.
    assert abs(parsed[0].energy_hartree - (-199.4369175187)) < 1e-9


def test_parse_goat_ensemble_atom_count_consistent():
    parsed = parse_goat_ensemble(GOAT_FIXTURE)
    expected = parsed[0].positions.shape[0]
    assert expected == 137  # from the fixture header
    assert all(p.positions.shape == (expected, 3) for p in parsed)


def test_parse_goat_ensemble_symbols_include_pd_and_p():
    parsed = parse_goat_ensemble(GOAT_FIXTURE)
    # Pd/P-containing complex; both atoms appear in every frame's symbol set.
    assert "Pd" in parsed[0].symbols
    assert "P" in parsed[0].symbols


def test_parse_goat_ensemble_forces_are_none():
    """GOAT ensemble files don't carry gradient info."""
    parsed = parse_goat_ensemble(GOAT_FIXTURE)
    assert all(p.forces_ev_per_a is None for p in parsed)


def test_parse_goat_ensemble_missing_raises(tmp_path: Path):
    empty = tmp_path / "empty.xyz"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(OutputParseError):
        parse_goat_ensemble(empty)


# ---------------------------------------------------------------------------
# Docker ensemble (verified against the real fixture)
# ---------------------------------------------------------------------------


def test_parse_docker_drops_last_frame():
    """The trailing structure is dropped as non-sensible."""
    parsed = parse_docker(DOCKER_FIXTURE)
    # Trimmed fixture: 4 frames, parser drops the last → 3.
    assert len(parsed) == 3


def test_parse_docker_first_energy_matches_fixture():
    parsed = parse_docker(DOCKER_FIXTURE)
    # First header in the fixture: ``3 Eopt=-137.1176174850 (Eh) Einter=...``
    assert abs(parsed[0].energy_hartree - (-137.1176174850)) < 1e-9


def test_parse_docker_atom_count_is_46():
    parsed = parse_docker(DOCKER_FIXTURE)
    for p in parsed:
        assert p.positions.shape == (46, 3)


def test_parse_docker_too_few_frames_raises(tmp_path: Path):
    """A single-frame docker output is unusable (last is always dropped)."""
    f = tmp_path / "tiny.xyz"
    f.write_text("1\n0 Eopt=-1.0 (Eh)\nH 0.0 0.0 0.0\n", encoding="utf-8")
    with pytest.raises(OutputParseError):
        parse_docker(f)


# ---------------------------------------------------------------------------
# Solvator ensemble (verified against the real fixture)
# ---------------------------------------------------------------------------


def test_parse_solvator_returns_all_frames():
    parsed = parse_solvator(SOLVATOR_FIXTURE)
    # Trimmed fixture: 3 frames.
    assert len(parsed) == 3


def test_parse_solvator_first_energy_matches_fixture():
    parsed = parse_solvator(SOLVATOR_FIXTURE)
    # First header in the fixture: ``Energy -141.854985``.
    assert abs(parsed[0].energy_hartree - (-141.854985)) < 1e-6


def test_parse_solvator_atom_count_consistent():
    parsed = parse_solvator(SOLVATOR_FIXTURE)
    # Solvent box grows by frame so atom counts vary; just confirm each is sane.
    for p in parsed:
        assert p.positions.shape[0] >= 49
        assert p.positions.shape[1] == 3


# ---------------------------------------------------------------------------
# Dispatcher + remaining placeholder
# ---------------------------------------------------------------------------


def test_parse_output_dispatches_dft():
    parsed = parse_output(FIXTURE, "opt_sp")
    assert len(parsed) == 1


def test_parse_output_accepts_opt_plus_sp_alias():
    parsed = parse_output(FIXTURE, "OPT+SP")
    assert len(parsed) == 1


def test_parse_output_dispatches_goat():
    parsed = parse_output(GOAT_FIXTURE, "goat")
    assert len(parsed) == 3


def test_parse_output_dispatches_docker():
    parsed = parse_output(DOCKER_FIXTURE, "docker")
    assert len(parsed) == 3   # 4 frames - 1 (last dropped)


def test_parse_output_dispatches_solvator():
    parsed = parse_output(SOLVATOR_FIXTURE, "solvator")
    assert len(parsed) == 3


def test_parse_output_unknown_operation_raises():
    with pytest.raises(OutputParseError):
        parse_output(FIXTURE, "weather_forecast")


# ---------------------------------------------------------------------------
# parse_pes — synthetic PES scan segments
# ---------------------------------------------------------------------------


def test_parse_pes_returns_one_structure_per_completed_segment(tmp_path: Path):
    from synthetic import synthetic_pes_segment

    text = (
        synthetic_pes_segment(
            coords=[("H", 0.0, 0.0, 0.0), ("H", 0.74, 0.0, 0.0)],
            energy=-1.10,
        )
        + synthetic_pes_segment(
            coords=[("H", 0.0, 0.0, 0.0), ("H", 0.80, 0.0, 0.0)],
            energy=-1.05,
        )
    )
    out = tmp_path / "pes.out"
    out.write_text(text, encoding="utf-8")
    parsed = parse_pes(out)
    assert len(parsed) == 2
    assert parsed[0].energy_hartree == -1.10
    assert parsed[1].energy_hartree == -1.05
    assert parsed[0].symbols == ("H", "H")
    assert parsed[0].positions.shape == (2, 3)


def test_parse_pes_takes_last_coord_block_and_energy_per_segment(tmp_path: Path):
    """Each segment's *last* coord block and *last* FINAL SP ENERGY must win."""
    from synthetic import synthetic_pes_segment

    text = synthetic_pes_segment(
        coords=[("H", 0.0, 0.0, 0.0), ("H", 0.74, 0.0, 0.0)],
        energy=-1.10,
        intermediate_energies=[-1.50, -1.30],  # earlier optimisation cycles
    )
    out = tmp_path / "pes.out"
    out.write_text(text, encoding="utf-8")
    parsed = parse_pes(out)
    assert len(parsed) == 1
    assert parsed[0].energy_hartree == -1.10
    # The intermediates shifted positions by +99 Å so we'd see them if the parser
    # picked the wrong block; assert the final positions match the converged frame.
    assert parsed[0].positions[1, 0] == 0.74


def test_parse_pes_skips_incomplete_trailing_segment(tmp_path: Path):
    """A trailing segment without the DONE marker is discarded."""
    from synthetic import synthetic_pes_segment

    text = synthetic_pes_segment(
        coords=[("H", 0.0, 0.0, 0.0), ("H", 0.74, 0.0, 0.0)],
        energy=-1.10,
    )
    # Append an unfinished segment.
    text += (
        "CARTESIAN COORDINATES (ANGSTROEM)\n"
        "---------------------------------\n"
        "H 9.999 9.999 9.999\nH 9.999 9.999 9.999\n\n"
        "FINAL SINGLE POINT ENERGY     -9.99\n"
    )
    out = tmp_path / "pes.out"
    out.write_text(text, encoding="utf-8")
    parsed = parse_pes(out)
    assert len(parsed) == 1
    assert parsed[0].energy_hartree == -1.10


def test_parse_pes_handles_indexed_coord_format(tmp_path: Path):
    """ORCA also prints coords as '1 C x y z' (5 tokens). Parse that too."""
    body = (
        "CARTESIAN COORDINATES (ANGSTROEM)\n"
        "  1   H   0.000000  0.000000  0.000000\n"
        "  2   H   0.740000  0.000000  0.000000\n"
        "\n"
        "FINAL SINGLE POINT ENERGY     -1.10\n"
        "*** OPTIMIZATION RUN DONE ***\n"
    )
    out = tmp_path / "pes.out"
    out.write_text(body, encoding="utf-8")
    parsed = parse_pes(out)
    assert len(parsed) == 1
    assert parsed[0].symbols == ("H", "H")
    assert parsed[0].positions[1, 0] == 0.74


def test_parse_pes_raises_when_no_segments(tmp_path: Path):
    out = tmp_path / "empty.out"
    out.write_text("no PES content here\n", encoding="utf-8")
    with pytest.raises(OutputParseError, match="no PES scan frames"):
        parse_pes(out)


def test_parse_pes_skips_segment_without_energy(tmp_path: Path):
    """A segment with coords but no FINAL SP ENERGY is silently dropped."""
    text = (
        "CARTESIAN COORDINATES (ANGSTROEM)\n"
        "  H   0.0  0.0  0.0\n  H   0.74 0.0 0.0\n\n"
        "*** OPTIMIZATION RUN DONE ***\n"
        + "CARTESIAN COORDINATES (ANGSTROEM)\n"
        "  H   0.0  0.0  0.0\n  H   0.80 0.0 0.0\n\n"
        "FINAL SINGLE POINT ENERGY     -1.05\n"
        "*** OPTIMIZATION RUN DONE ***\n"
    )
    out = tmp_path / "pes.out"
    out.write_text(text, encoding="utf-8")
    parsed = parse_pes(out)
    assert len(parsed) == 1
    assert parsed[0].energy_hartree == -1.05


def test_parse_pes_skips_non_atom_lines_inside_coord_block(tmp_path: Path):
    """A 4-token line that isn't ``sym x y z`` (e.g. has non-numeric tokens) is skipped."""
    body = (
        "CARTESIAN COORDINATES (ANGSTROEM)\n"
        "  H   0.0    0.0    0.0\n"
        "  C   not    a      number\n"        # 4 tokens but two are non-numeric
        "  H   0.74   0.0    0.0\n"
        "\n"
        "FINAL SINGLE POINT ENERGY     -1.10\n"
        "*** OPTIMIZATION RUN DONE ***\n"
    )
    out = tmp_path / "pes.out"
    out.write_text(body, encoding="utf-8")
    parsed = parse_pes(out)
    assert parsed[0].symbols == ("H", "H")


def test_parse_pes_skips_segment_without_coords(tmp_path: Path):
    """A segment with FINAL SP ENERGY but no coord block is silently dropped."""
    text = (
        "FINAL SINGLE POINT ENERGY     -1.10\n"
        "*** OPTIMIZATION RUN DONE ***\n"
        + "CARTESIAN COORDINATES (ANGSTROEM)\n"
        "  H 0.0 0.0 0.0\n  H 0.80 0.0 0.0\n\n"
        "FINAL SINGLE POINT ENERGY     -1.05\n"
        "*** OPTIMIZATION RUN DONE ***\n"
    )
    out = tmp_path / "pes.out"
    out.write_text(text, encoding="utf-8")
    parsed = parse_pes(out)
    assert len(parsed) == 1
    assert parsed[0].energy_hartree == -1.05


def test_parse_output_dispatches_pes(tmp_path: Path):
    """``parse_output(..., 'pes')`` should reach :func:`parse_pes`."""
    from synthetic import synthetic_pes_segment

    text = synthetic_pes_segment(
        coords=[("H", 0.0, 0.0, 0.0), ("H", 0.74, 0.0, 0.0)],
        energy=-1.10,
    )
    out = tmp_path / "pes.out"
    out.write_text(text, encoding="utf-8")
    parsed = parse_output(out, "pes")
    assert len(parsed) == 1


# ---------------------------------------------------------------------------
# Ensemble parser skip branches (synthetic malformed frames)
# ---------------------------------------------------------------------------


def test_xyz_ensemble_skips_frames_with_unparseable_headers(tmp_path: Path):
    """A frame whose header doesn't match the regex must be silently skipped."""
    p = tmp_path / "mixed.xyz"
    p.write_text(
        "2\nnot a header at all\nH 0 0 0\nH 0 0 1\n"
        "2\n-1.5 converged=true\nH 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )
    parsed = parse_goat_ensemble(p)
    assert len(parsed) == 1
    assert parsed[0].energy_hartree == -1.5


def test_xyz_ensemble_skips_frames_with_malformed_atom_rows(tmp_path: Path):
    """An atom line with fewer than 4 whitespace-separated parts kills the frame."""
    p = tmp_path / "badrow.xyz"
    p.write_text(
        "2\n-1.0 converged=true\nH 0 0\nH 0 0 1\n"
        "2\n-2.0 converged=true\nH 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )
    parsed = parse_goat_ensemble(p)
    assert len(parsed) == 1
    assert parsed[0].energy_hartree == -2.0


def test_xyz_ensemble_skips_non_digit_lines(tmp_path: Path):
    """Stray text between frames advances past the line without consuming a frame."""
    p = tmp_path / "stray.xyz"
    p.write_text(
        "garbage line at the top\n"
        "more garbage\n"
        "2\n-2.0 converged=true\nH 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )
    parsed = parse_goat_ensemble(p)
    assert len(parsed) == 1


def test_parse_forces_returns_none_when_block_has_no_valid_rows(tmp_path: Path):
    """Gradient block with no parseable rows must yield None."""
    text = (
        "CARTESIAN GRADIENT\n"
        "------------------\n"
        "nothing parseable here at all\n"
        "------------------\n"
    )
    assert parse_forces(text) is None


def test_xyz_ensemble_breaks_on_truncated_file(tmp_path: Path):
    """A frame whose header claims more atoms than the file provides triggers break."""
    p = tmp_path / "truncated.xyz"
    # Says 100 atoms but only 1 line follows — should break out cleanly with no frames.
    p.write_text("100\n-1.0 converged=true\nH 0 0 0\n", encoding="utf-8")
    with pytest.raises(OutputParseError):
        parse_goat_ensemble(p)
