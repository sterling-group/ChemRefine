"""Tests for the ORCA energy parser: electronic energy + thermochemistry."""

from __future__ import annotations

import pytest
from synthetic import (
    THERMOCHEMISTRY_BLOCK as _SYNTH_THERMO,
)

from chemrefine.engines.orca.output.energy import (
    parse_final_energy_from_text,
    parse_thermochemistry_from_text,
)

# ---------------------------------------------------------------------------
# electronic energy
# ---------------------------------------------------------------------------


def test_final_energy_none_when_absent():
    assert parse_final_energy_from_text("no energy here\n") is None


def test_final_energy_takes_the_last_value():
    """Geometry optimisation re-prints the energy; the converged (last) one wins."""
    text = "FINAL SINGLE POINT ENERGY     -1.0\nFINAL SINGLE POINT ENERGY     -2.5\n"
    assert parse_final_energy_from_text(text) == -2.5


def test_final_energy_reads_external_program_variant():
    """The ExtOpt path prints '(From external program)'."""
    text = "FINAL SINGLE POINT ENERGY (From external program)     -3.25\n"
    assert parse_final_energy_from_text(text) == -3.25


# ---------------------------------------------------------------------------
# thermochemistry
# ---------------------------------------------------------------------------


def test_thermochemistry_none_without_block():
    assert parse_thermochemistry_from_text("no thermo here", electronic_hartree=-76.4) is None


def test_thermochemistry_parses_absolute_values_and_zpe():
    thermo = parse_thermochemistry_from_text(_SYNTH_THERMO, electronic_hartree=-76.40)
    assert thermo is not None
    assert thermo.gibbs_hartree == -76.41
    assert thermo.enthalpy_hartree == -76.38
    # electronic + ZPE correction (-76.40 + 0.03) — distinct from the enthalpy on
    # purpose, so computing this *as* the enthalpy cannot pass.
    assert thermo.energy_zpe_hartree == pytest.approx(-76.37)


def test_thermochemistry_missing_lines_yield_none():
    """A block present but without the individual lines yields None per quantity."""
    thermo = parse_thermochemistry_from_text(
        "THERMOCHEMISTRY AT 298.15K\n(no value lines)\n", electronic_hartree=-76.40
    )
    assert thermo is not None
    assert thermo.gibbs_hartree is None
    assert thermo.enthalpy_hartree is None
    assert thermo.energy_zpe_hartree is None
