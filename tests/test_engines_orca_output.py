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

FIXTURE = Path(__file__).parent / "data" / "orca.out"


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
    """The last FINAL SINGLE POINT ENERGY in the fixture is -6044.555... Hartree."""
    parsed = parse_dft(FIXTURE)
    assert parsed[0].energy_hartree < 0
    # Lock in the magnitude order; small drift between ORCA versions is OK.
    assert abs(parsed[0].energy_hartree + 6044.555) < 1e-2


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
    if parsed[0].forces_eV_per_A is not None:
        assert parsed[0].forces_eV_per_A.shape == parsed[0].positions.shape


# ---------------------------------------------------------------------------
# parse_dft — synthetic minimal cases
# ---------------------------------------------------------------------------


def _synth_dft_output(energies: list[float], coords: list[tuple[str, float, float, float]]) -> str:
    """Build a minimal ORCA-shaped output snippet for unit testing."""
    coord_lines = "\n".join(
        f"  {sym:2s}  {x:.6f}  {y:.6f}  {z:.6f}" for sym, x, y, z in coords
    )
    head = "CARTESIAN COORDINATES (ANGSTROEM)\n---------------------------------\n"
    tail = "\n---------------------------------\n"
    body = head + coord_lines + tail
    body += "\n".join(f"FINAL SINGLE POINT ENERGY     {e}" for e in energies) + "\n"
    return body


def test_parse_dft_picks_last_energy_when_multiple_appear(tmp_path: Path):
    text = _synth_dft_output(
        [-1.0, -2.0, -3.0],
        [("H", 0.0, 0.0, 0.0), ("H", 0.74, 0.0, 0.0)],
    )
    path = tmp_path / "synth.out"
    path.write_text(text, encoding="utf-8")
    parsed = parse_dft(path)
    assert parsed[0].energy_hartree == -3.0


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
# Dispatcher + placeholders
# ---------------------------------------------------------------------------


def test_parse_output_dispatches_dft():
    parsed = parse_output(FIXTURE, "opt_sp")
    assert len(parsed) == 1


def test_parse_output_accepts_opt_plus_sp_alias():
    parsed = parse_output(FIXTURE, "OPT+SP")
    assert len(parsed) == 1


def test_parse_output_unknown_operation_raises():
    with pytest.raises(OutputParseError):
        parse_output(FIXTURE, "weather_forecast")


@pytest.mark.parametrize(
    "func",
    [parse_goat_ensemble, parse_pes, parse_docker, parse_solvator],
)
def test_placeholder_parsers_raise_with_todo(tmp_path: Path, func):
    """Placeholders raise NotImplementedError until a real fixture exists."""
    with pytest.raises(NotImplementedError):
        func(tmp_path / "missing.out")
