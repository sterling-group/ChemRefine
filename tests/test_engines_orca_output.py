"""Tests for the ORCA output parser, anchored on the real fixture."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from synthetic import THERMOCHEMISTRY_BLOCK, synthetic_dft_output

from chemrefine.engines.api import ParsedResult
from chemrefine.engines.orca.output import parse_dft, parse_dft_from_text, parse_output, status
from chemrefine.engines.orca.output.ensembles import (
    parse_docker,
    parse_goat_ensemble,
    parse_pes,
    parse_solvator,
)
from chemrefine.engines.orca.output.forces import parse_forces_from_text as parse_forces
from chemrefine.errors import OutputParseError

_WATER = [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.0), ("H", 0.0, 1.0, 0.0)]


def test_parse_dft_attaches_thermochemistry_when_present():
    text = synthetic_dft_output([-76.40], _WATER) + "\n" + THERMOCHEMISTRY_BLOCK
    parsed = parse_dft_from_text(text)
    assert parsed[0].gibbs_hartree == -76.41
    assert parsed[0].enthalpy_hartree == -76.38
    assert parsed[0].energy_zpe_hartree == pytest.approx(-76.38)


def test_parse_dft_without_thermochemistry_leaves_none():
    parsed = parse_dft_from_text(synthetic_dft_output([-76.40], _WATER))
    assert parsed[0].gibbs_hartree is None
    assert parsed[0].enthalpy_hartree is None
    assert parsed[0].energy_zpe_hartree is None


DATA = Path(__file__).parent / "data"
FIXTURE = DATA / "engines" / "orca" / "dft" / "step1_0.out"
GOAT_FIXTURE = DATA / "engines" / "orca" / "goat" / "step1_0.finalensemble.xyz"
# Named exactly like ORCA's sidecar so the legacy drop-last-frame rule applies.
DOCKER_FIXTURE = DATA / "engines" / "orca" / "docker" / "step1_0.docker.struc1.allopt.xyz"
DOCKER_611_FIXTURE = (
    DATA / "engines" / "orca" / "docker_611" / "step1_0.docker.struc1.all.optimized.xyz"
)
SOLVATOR_FIXTURE = DATA / "engines" / "orca" / "solvator" / "step1_0.solventbuild.xyz"


# ---------------------------------------------------------------------------
# Fixture sanity
# ---------------------------------------------------------------------------


def test_fixture_present():
    assert FIXTURE.is_file(), f"missing fixture: {FIXTURE}"


# ---------------------------------------------------------------------------
# Run-status flags (terminated / converged) read in the single parse pass
# ---------------------------------------------------------------------------


def _minimal_out(*, terminated_normally: bool, not_converged: bool) -> str:
    """Smallest .out parse_dft accepts, with optional status markers."""
    body = (
        "FINAL SINGLE POINT ENERGY     -1.500000\n\n"
        "CARTESIAN COORDINATES (ANGSTROEM)\n"
        "----------------\n"
        "  H   0.000000   0.000000   0.000000\n"
        "----------------\n"
    )
    if not_converged:
        # ORCA's real wording for an optimisation that has not finished. The previous
        # fixture said "The optimization HAS NOT CONVERGED", which ORCA never prints —
        # a fabricated fixture agreeing with a fabricated regex is why the gap survived.
        body += "\nThe optimization has not yet converged - more geometry cycles are needed\n"
    if terminated_normally:
        body += "\n                  ****ORCA TERMINATED NORMALLY****\n"
    return body


def test_parse_dft_marks_success(tmp_path: Path):
    out = tmp_path / "ok.out"
    out.write_text(_minimal_out(terminated_normally=True, not_converged=False), encoding="utf-8")
    ps = parse_dft(out)[0]
    assert ps.terminated_normally is True
    assert ps.converged is True


def test_parse_dft_marks_not_terminated(tmp_path: Path):
    out = tmp_path / "crash.out"
    out.write_text(_minimal_out(terminated_normally=False, not_converged=False), encoding="utf-8")
    assert parse_dft(out)[0].terminated_normally is False


def test_parse_dft_marks_not_converged(tmp_path: Path):
    out = tmp_path / "maxiter.out"
    out.write_text(_minimal_out(terminated_normally=True, not_converged=True), encoding="utf-8")
    assert parse_dft(out)[0].converged is False


# ---------------------------------------------------------------------------
# Convergence is the LAST verdict, not "did the file ever say NOT CONVERGED"
# ---------------------------------------------------------------------------

# Verbatim ORCA 6.1.1 wording — taken from its binaries and the recorded outputs, not
# invented. `_GEOM_PENDING` is what a geometry optimisation prints after every non-final
# cycle; it is also all it leaves behind when it runs out of them.
_SCF_FAIL = "               *        SCF NOT CONVERGED AFTER 125 CYCLES         *"
_SCF_OK = "               *           SCF CONVERGED AFTER   6 CYCLES          *"
_GEOM_PENDING = "The optimization has not yet converged - more geometry cycles are needed"
_GEOM_OK = "                    ***        THE OPTIMIZATION HAS CONVERGED     ***"
# Real ORCA strings that must NOT be read as run failures.
_LOCALIZATION_FAIL = "LOCALIZATION HAS NOT CONVERGED."
_OPT_RECOVERY = "Not converged! We will rebuild the internal coordinates and try again."


def test_converged_true_when_an_early_scf_failure_recovers():
    """A recovered SCF must not fail the run.

    ORCA retries a failed SCF with a different guess; the optimisation then
    converges. Scanning the whole file for ``NOT CONVERGED`` called that run
    failed, which halted an ``on_failure: stop`` pipeline that had actually
    succeeded and resubmitted a converged structure.
    """
    assert status.parse_converged("\n".join([_SCF_FAIL, _SCF_OK, _GEOM_OK])) is True


def test_converged_false_when_the_optimisation_runs_out_of_cycles():
    """A geometry optimisation that exhausts MaxIter is a failure.

    ORCA has no "optimisation failed" banner: it prints "has not yet converged" after
    every non-final cycle and, when the cycles run out, simply stops — so that line is
    the last word. Nothing matched it before, which meant a cycle-exhausted optimisation
    was reported converged and its unfinished geometry flowed downstream.
    """
    assert status.parse_converged("\n".join([_SCF_OK, _GEOM_PENDING])) is False


def test_converged_true_when_the_optimisation_finishes_after_pending_cycles():
    """The common case: 72 of the 108 recorded outputs contain the pending line, and in
    every one of them the success banner follows it."""
    assert (
        status.parse_converged("\n".join([_GEOM_PENDING, _GEOM_PENDING, _GEOM_OK, _SCF_OK])) is True
    )


def test_converged_true_when_only_orbital_localisation_failed():
    """Localisation is post-processing for printing — it says nothing about the result.

    A bare "NOT CONVERGED" catch-all matched ORCA's real LOCALIZATION HAS NOT CONVERGED
    and failed an otherwise perfect run. Localisation also runs near the end of a job, so
    it would usually be the *last* verdict and win.
    """
    assert status.parse_converged("\n".join([_SCF_OK, _GEOM_OK, _LOCALIZATION_FAIL])) is True


def test_converged_true_when_the_optimiser_recovers_its_coordinates():
    """ "Not converged! We will rebuild the internal coordinates and try again." is the
    optimiser recovering, not failing."""
    assert status.parse_converged("\n".join([_OPT_RECOVERY, _SCF_OK, _GEOM_OK])) is True


def test_converged_true_when_a_final_single_point_follows_a_converged_opt():
    """The post-optimisation single point is the last verdict and it passed."""
    assert status.parse_converged("\n".join([_GEOM_OK, _SCF_OK])) is True


def test_converged_false_when_the_last_scf_fails():
    assert status.parse_converged("\n".join([_SCF_OK, _GEOM_OK, _SCF_FAIL])) is False


def test_converged_true_when_no_verdict_is_printed():
    """No signal is not a failure signal — matches ``lifecycle.succeeded``."""
    assert status.parse_converged("FINAL SINGLE POINT ENERGY  -1.5\n") is True


def test_parse_dft_recovered_scf_is_a_success(tmp_path: Path):
    """The recovery reaches the parsed structure, not just the status helper."""
    out = tmp_path / "recovered.out"
    out.write_text(
        _minimal_out(terminated_normally=True, not_converged=False).replace(
            "FINAL SINGLE POINT ENERGY",
            f"{_SCF_FAIL}\n{_SCF_OK}\n{_GEOM_OK}\nFINAL SINGLE POINT ENERGY",
        ),
        encoding="utf-8",
    )
    parsed = parse_dft(out)[0]
    assert parsed.converged is True
    assert parsed.terminated_normally is True


# ---------------------------------------------------------------------------
# Ensemble frames take their run status from the .out beside the sidecar
# ---------------------------------------------------------------------------


def _goat_case(tmp_path: Path, *, terminated_normally: bool) -> Path:
    """A minimal GOAT job on disk: the ``.out`` plus its ensemble sidecar."""
    out = tmp_path / "step1_0.out"
    body = "GOAT                             ...       75.347 sec\n"
    if terminated_normally:
        body += "                             ****ORCA TERMINATED NORMALLY****\n"
    out.write_text(body, encoding="utf-8")
    (tmp_path / "step1_0.finalensemble.xyz").write_text(
        "1\n  -40.123456   converged=true\n  C   0.000000   0.000000   0.000000\n"
        "1\n  -40.100000   converged=true\n  C   0.100000   0.000000   0.000000\n",
        encoding="utf-8",
    )
    return out


def test_goat_frames_inherit_normal_termination(tmp_path: Path):
    frames = parse_output(_goat_case(tmp_path, terminated_normally=True), "goat")
    assert len(frames) == 2
    assert all(f.terminated_normally is True for f in frames)


def test_goat_frames_are_flagged_when_the_job_never_terminated(tmp_path: Path):
    """A crashed GOAT run must not read as success.

    The sidecar carries no run status, so frames parsed straight out of it default
    to ``terminated_normally=None`` — which ``succeeded()`` reads as "not a failure signal".
    A job killed after writing a partial ensemble was therefore an unconditional
    success with an empty ledger.
    """
    frames = parse_output(_goat_case(tmp_path, terminated_normally=False), "goat")
    assert len(frames) == 2
    assert all(f.terminated_normally is False for f in frames)


def test_ensemble_frames_leave_converged_unreported(tmp_path: Path):
    """An ensemble is a *set* of stationary points — one stubborn pose must not
    condemn the rest, so the whole-run convergence verdict is not stamped."""
    frames = parse_output(_goat_case(tmp_path, terminated_normally=True), "goat")
    assert all(f.converged is None for f in frames)


# ---------------------------------------------------------------------------
# parse_dft — verified against the real fixture
# ---------------------------------------------------------------------------


def test_parse_dft_returns_single_structure():
    parsed = parse_dft(FIXTURE)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert isinstance(parsed[0], ParsedResult)


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
        "  ...continuation\n"  # only 1 token, len(parts) < 4
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


def test_parse_dft_empty_coord_block_raises(tmp_path: Path):
    """A coord block present but with no parseable atom rows is a corrupt output —
    raise rather than emit a 0-atom structure that would be cached/filtered silently."""
    path = tmp_path / "empty-coords.out"
    path.write_text(
        "CARTESIAN COORDINATES (ANGSTROEM)\n"
        "---------------------------------\n"
        "  ...truncated\n"  # no 4-token atom rows survive
        "---------------------------------\n"
        "FINAL SINGLE POINT ENERGY     -1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(OutputParseError, match="no atoms"):
        parse_dft(path)


def test_parse_dft_corrupt_coordinate_token_raises_parse_error(tmp_path: Path):
    """A non-numeric coordinate token (e.g. a ``*****`` overflow placeholder)
    is a per-file parse failure for the ledger, never a raw ``ValueError``
    that would crash the whole step."""
    path = tmp_path / "corrupt-coords.out"
    path.write_text(
        "CARTESIAN COORDINATES (ANGSTROEM)\n"
        "---------------------------------\n"
        "  H   0.000000   *********   0.000000\n"
        "---------------------------------\n"
        "FINAL SINGLE POINT ENERGY     -1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(OutputParseError, match="malformed coordinate row"):
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


def test_parse_output_goat_reads_finalensemble_sidecar(tmp_path: Path):
    """``parse_output(.out, 'goat')`` must read ``<base>.finalensemble.xyz``.

    ORCA writes the ensemble to a sidecar next to the ``.out`` (the engine only
    knows the ``.out`` path), so the dispatcher must resolve the sidecar.
    """
    base = tmp_path / "step1_structure_0"
    base.with_suffix(".out").write_text("ORCA log, not an ensemble\n", encoding="utf-8")
    (tmp_path / "step1_structure_0.finalensemble.xyz").write_text(
        GOAT_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    parsed = parse_output(base.with_suffix(".out"), "goat")
    assert len(parsed) == 3  # same as the fixture frames


def test_parse_output_goat_missing_sidecar_raises(tmp_path: Path):
    out = tmp_path / "step1_structure_0.out"
    out.write_text("log only\n", encoding="utf-8")
    with pytest.raises(OutputParseError, match="finalensemble"):
        parse_output(out, "goat")


def test_parse_goat_ensemble_missing_raises(tmp_path: Path):
    empty = tmp_path / "empty.xyz"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(OutputParseError):
        parse_goat_ensemble(empty)


def test_parse_goat_ensemble_skips_frame_with_corrupt_coordinate(tmp_path: Path):
    """A frame whose coordinate token isn't numeric is skipped like any other
    malformed frame — it must not crash the walk with a raw ``ValueError``."""
    ensemble = tmp_path / "corrupt.finalensemble.xyz"
    ensemble.write_text(
        "1\n"
        "-1.0\n"
        "H 0.0 0.0 0.0\n"
        "1\n"
        "-2.0\n"
        "H 0.0 ******** 0.0\n"  # corrupt middle frame
        "1\n"
        "-3.0\n"
        "H 0.0 0.0 1.0\n",
        encoding="utf-8",
    )
    parsed = parse_goat_ensemble(ensemble)
    assert [p.energy_hartree for p in parsed] == [-1.0, -3.0]


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
    """A single-frame legacy docker output is unusable (its last is always dropped)."""
    f = tmp_path / "tiny.docker.struc1.allopt.xyz"
    f.write_text("1\n0 Eopt=-1.0 (Eh)\nH 0.0 0.0 0.0\n", encoding="utf-8")
    with pytest.raises(OutputParseError):
        parse_docker(f)


def test_parse_docker_611_layout_keeps_every_frame():
    """ORCA 6.1.1's ``all.optimized.xyz`` holds only real poses — nothing is dropped."""
    parsed = parse_docker(DOCKER_611_FIXTURE)
    assert len(parsed) == 3
    assert abs(parsed[0].energy_hartree - (-13.0314929160)) < 1e-9
    for p in parsed:
        assert p.positions.shape == (7, 3)


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


def _with_sidecar(tmp_path: Path, suffix: str, fixture: Path) -> Path:
    """Lay out a dummy ``.out`` plus its ``<base>.<suffix>`` ensemble sidecar."""
    out = tmp_path / "step1_structure_0.out"
    out.write_text("ORCA log\n", encoding="utf-8")
    (tmp_path / f"step1_structure_0.{suffix}").write_text(
        fixture.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return out


def test_parse_output_dispatches_goat(tmp_path: Path):
    out = _with_sidecar(tmp_path, "finalensemble.xyz", GOAT_FIXTURE)
    assert len(parse_output(out, "goat")) == 3


def test_parse_output_dispatches_docker(tmp_path: Path):
    out = _with_sidecar(tmp_path, "docker.struc1.allopt.xyz", DOCKER_FIXTURE)
    assert len(parse_output(out, "docker")) == 3  # 4 frames - 1 (last dropped)


def test_parse_output_falls_back_to_the_611_docker_sidecar(tmp_path: Path):
    """With no legacy ``allopt`` file, dispatch finds 6.1.1's renamed pose file."""
    out = _with_sidecar(tmp_path, "docker.struc1.all.optimized.xyz", DOCKER_611_FIXTURE)
    assert len(parse_output(out, "docker")) == 3  # nothing dropped


def test_missing_ensemble_sidecar_error_names_every_candidate(tmp_path: Path):
    """The error for a missing docker sidecar lists both accepted filenames."""
    out = tmp_path / "step1_0.out"
    out.write_text("ORCA log\n", encoding="utf-8")
    with pytest.raises(OutputParseError, match=r"allopt\.xyz or .*all\.optimized\.xyz"):
        parse_output(out, "docker")


def test_parse_output_dispatches_solvator(tmp_path: Path):
    out = _with_sidecar(tmp_path, "solventbuild.xyz", SOLVATOR_FIXTURE)
    assert len(parse_output(out, "solvator")) == 3


def test_parse_output_falls_back_to_the_611_solvator_sidecar(tmp_path: Path):
    """With no legacy file, dispatch finds 6.1.1's ``solvator.solventbuild.xyz``."""
    out = _with_sidecar(tmp_path, "solvator.solventbuild.xyz", SOLVATOR_FIXTURE)
    assert len(parse_output(out, "solvator")) == 3


def test_parse_output_unknown_operation_raises():
    with pytest.raises(OutputParseError):
        parse_output(FIXTURE, "weather_forecast")


# ---------------------------------------------------------------------------
# parse_pes — synthetic PES scan segments
# ---------------------------------------------------------------------------


def test_parse_pes_returns_one_structure_per_completed_segment(tmp_path: Path):
    from synthetic import synthetic_pes_segment

    text = synthetic_pes_segment(
        coords=[("H", 0.0, 0.0, 0.0), ("H", 0.74, 0.0, 0.0)],
        energy=-1.10,
    ) + synthetic_pes_segment(
        coords=[("H", 0.0, 0.0, 0.0), ("H", 0.80, 0.0, 0.0)],
        energy=-1.05,
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
        "*** OPTIMIZATION RUN DONE ***\n" + "CARTESIAN COORDINATES (ANGSTROEM)\n"
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
        "  C   not    a      number\n"  # 4 tokens but two are non-numeric
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
        "*** OPTIMIZATION RUN DONE ***\n" + "CARTESIAN COORDINATES (ANGSTROEM)\n"
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
        "2\nnot a header at all\nH 0 0 0\nH 0 0 1\n2\n-1.5 converged=true\nH 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )
    parsed = parse_goat_ensemble(p)
    assert len(parsed) == 1
    assert parsed[0].energy_hartree == -1.5


def test_xyz_ensemble_skips_frames_with_malformed_atom_rows(tmp_path: Path):
    """An atom line with fewer than 4 whitespace-separated parts kills the frame."""
    p = tmp_path / "badrow.xyz"
    p.write_text(
        "2\n-1.0 converged=true\nH 0 0\nH 0 0 1\n2\n-2.0 converged=true\nH 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )
    parsed = parse_goat_ensemble(p)
    assert len(parsed) == 1
    assert parsed[0].energy_hartree == -2.0


def test_xyz_ensemble_skips_non_digit_lines(tmp_path: Path):
    """Stray text between frames advances past the line without consuming a frame."""
    p = tmp_path / "stray.xyz"
    p.write_text(
        "garbage line at the top\nmore garbage\n2\n-2.0 converged=true\nH 0 0 0\nH 0 0 1\n",
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


def test_pes_coord_block_at_end_of_segment_is_parsed():
    """A coordinate block that runs to the very end of its segment (no trailing
    blank line) still yields its atoms — the row loop must exit cleanly at EOF."""
    from chemrefine.engines.orca.output.ensembles import _parse_last_pes_coord_block

    segment = (
        "CARTESIAN COORDINATES (ANGSTROEM)\n"
        "---------------------------------\n"
        "H 0.0 0.0 0.0\n"
        "H 0.74 0.0 0.0"
    )
    atoms = _parse_last_pes_coord_block(segment)
    assert [a[0] for a in atoms] == ["H", "H"]
    assert atoms[1][1] == 0.74


# --- orca output text dispatcher --------------------------------------------


def test_parse_text_handles_pes():
    from synthetic import synthetic_pes_segment

    from chemrefine.engines.orca import output

    text = synthetic_pes_segment(coords=[("H", 0.0, 0.0, 0.0)], energy=-1.0)
    parsed = output.parse_text(text, "pes", src="x")
    assert parsed and parsed[-1].energy_hartree == -1.0


def test_parse_text_rejects_non_text_operation():
    from chemrefine.engines.orca import output

    with pytest.raises(OutputParseError):
        output.parse_text("", "goat", src="x")


# ---------------------------------------------------------------------------
# Multi-Hessian outputs: every parser must describe the same (final) geometry
# ---------------------------------------------------------------------------

_MULTI_HESSIAN = (
    Path(__file__).resolve().parent
    / "data"
    / "engines"
    / "orca"
    / "freq_multi_hessian"
    / "step1_0.out"
)


def _blocks(text: str) -> list[str]:
    """The text of each ``VIBRATIONAL FREQUENCIES`` group, in file order."""
    return text.split("VIBRATIONAL FREQUENCIES")[1:]


def test_the_fixture_really_has_disagreeing_hessians():
    """Guard the guard: if the fixture stops having two differing blocks, the tests below pass
    vacuously and the defect they exist for could come back unnoticed."""
    blocks = _blocks(_MULTI_HESSIAN.read_text(encoding="utf-8"))
    counts = [len(re.findall("imaginary mode", b)) for b in blocks]
    assert len(counts) >= 2, counts
    assert counts[0] != counts[-1], f"first and last Hessian agree ({counts}) — fixture is useless"


def test_frequencies_come_from_the_final_hessian():
    """A TS search recomputes the Hessian; only the last one describes the converged geometry.

    Reading the first meant a structure that had converged to a clean transition state was
    reported with the imaginary modes it had *before* converging — so `nms` re-optimised it
    along modes that no longer existed. On the run this fixture is trimmed from, 66 of 72
    round-2 optimisations were spawned for structures already at the target.
    """
    text = _MULTI_HESSIAN.read_text(encoding="utf-8")
    expected = len(re.findall("imaginary mode", _blocks(text)[-1]))

    parsed = parse_dft_from_text(text)[0]

    assert len(parsed.imaginary_freqs or {}) == expected


def test_every_parser_reads_the_same_hessian():
    """Energy, geometry, thermochemistry and frequencies must describe one geometry.

    They drifted because nothing said they had to agree: four parsers took the *last* match and
    the two frequency parsers took the *first*. This asserts the property rather than the
    implementation, so a future parser that reaches for a different block fails here.
    """
    text = _MULTI_HESSIAN.read_text(encoding="utf-8")
    # Everything from the converged geometry onward: its coordinates, its energy and its
    # Hessian, with the earlier cycles removed entirely. Parsing the whole file must agree
    # with parsing this — that is what "one geometry" means.
    final_only = text[text.rindex("CARTESIAN COORDINATES (ANGSTROEM)") :]
    assert _blocks(final_only) and len(_blocks(final_only)) == 1, "slice must hold one Hessian"

    whole = parse_dft_from_text(text)[0]
    converged = parse_dft_from_text(final_only)[0]

    assert whole.imaginary_freqs == converged.imaginary_freqs
    assert whole.gibbs_hartree == converged.gibbs_hartree
    assert whole.energy_hartree == converged.energy_hartree
    np.testing.assert_array_equal(whole.normal_modes, converged.normal_modes)
