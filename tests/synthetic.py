"""Centralised synthetic ORCA-shaped data for tests.

The tests/data/ directory holds real (trimmed) ORCA fixtures for the
happy-path parsers; this module holds **minimal synthetic snippets**
for fast unit tests and known-failure-mode regression coverage.

Following the pytest community convention (and PyA3EDA's
``tests/synthetic_outputs.py``): minimal, self-documenting, no
external file I/O for the basic edge cases.
"""

from __future__ import annotations


def synthetic_dft_output(
    energies: list[float],
    coords: list[tuple[str, float, float, float]],
) -> str:
    """Build a minimal ORCA-shaped DFT output snippet.

    Parameters
    ----------
    energies
        Each value becomes a separate ``FINAL SINGLE POINT ENERGY`` line.
        Order matters — the parser keeps the last one.
    coords
        ``(symbol, x, y, z)`` tuples for a single ``CARTESIAN
        COORDINATES (ANGSTROEM)`` block.
    """
    coord_lines = "\n".join(
        f"  {sym:2s}  {x:.6f}  {y:.6f}  {z:.6f}" for sym, x, y, z in coords
    )
    head = "CARTESIAN COORDINATES (ANGSTROEM)\n---------------------------------\n"
    tail = "\n---------------------------------\n"
    body = head + coord_lines + tail
    body += "\n".join(f"FINAL SINGLE POINT ENERGY     {e}" for e in energies) + "\n"
    return body


# A minimal vibrational-frequencies block. Mirrors the layout in the real
# ORCA ``.out`` (banner + scaling-factor line + per-mode index/value lines).
# Two of the entries carry the ``***imaginary mode***`` marker so parsers
# that filter to imaginary modes can be exercised against synthetic data.
FREQUENCY_BLOCK = """
some preamble
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
     6:      15.11 cm**-1
     7:      17.30 cm**-1
    37:   -118.27 cm**-1  ***imaginary mode***
    38:    -42.10 cm**-1  ***imaginary mode***
    39:     45.50 cm**-1

trailing text
"""


def synthetic_gradient_block(
    rows: list[tuple[int, str, float, float, float]],
) -> str:
    """Build a synthetic ``CARTESIAN GRADIENT`` block."""
    lines = "\n".join(
        f"   {idx}  {sym}  :    {dx:.6f}   {dy:.6f}   {dz:.6f}"
        for idx, sym, dx, dy, dz in rows
    )
    return f"CARTESIAN GRADIENT\n------------------\n{lines}\n------------------\n"


# Synthetic ``NORMAL MODES`` block for a 2-atom system.
# 2 atoms by 3 axes = 6 displacement rows per column block. ORCA prints
# 6 modes per column block; for 6 total modes a single block suffices.
# Modes 0-4 are zero (translational/rotational); mode 5 carries
# distinct ``0.1, 0.2, 0.3`` on atom 0 and ``-0.1, -0.2, -0.3`` on
# atom 1 so the parser's reshape can be verified by index.
NORMAL_MODES_BLOCK_2_ATOMS = """\
NORMAL MODES
-----------------------------------------
                  0          1          2          3          4          5
      0       0.000000   0.000000   0.000000   0.000000   0.000000   0.100000
      1       0.000000   0.000000   0.000000   0.000000   0.000000   0.200000
      2       0.000000   0.000000   0.000000   0.000000   0.000000   0.300000
      3       0.000000   0.000000   0.000000   0.000000   0.000000  -0.100000
      4       0.000000   0.000000   0.000000   0.000000   0.000000  -0.200000
      5       0.000000   0.000000   0.000000   0.000000   0.000000  -0.300000
-----------------------------------------
"""
