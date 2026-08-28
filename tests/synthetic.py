"""Centralised ORCA-shaped text for tests — the snippets and the captured failures.

``tests/data/engines/`` holds the outputs that *parse into a golden record*: a real trimmed
output per engine, checked against ``expected.json``. This module holds the text that has no
golden record — the **minimal synthetic snippets** the per-section unit tests build on, and
the **captured failure outputs** that exist to be refused rather than parsed.

Following the pytest community convention (and PyA3EDA's ``tests/synthetic_outputs.py``):
self-documenting, no external file I/O for an edge case.

The two kinds are not interchangeable and the constants say which they are. A synthetic
snippet may be edited to sharpen a case; **a captured one may not** — its value is that ORCA
really printed it, and a fixture edited to agree with a parser is how a wrong regex looks like
a passing test. Where a captured constant is used, the test asserts against ORCA's own
wording, so changing the text changes what is being proven.
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
    coord_lines = "\n".join(f"  {sym:2s}  {x:.6f}  {y:.6f}  {z:.6f}" for sym, x, y, z in coords)
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
        f"   {idx}  {sym}  :    {dx:.6f}   {dy:.6f}   {dz:.6f}" for idx, sym, dx, dy, dz in rows
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


# A minimal ORCA THERMOCHEMISTRY block. ``Zero point energy`` is the (positive)
# ZPE correction; ``Total Enthalpy`` / ``Final Gibbs free energy`` are absolute Eh.
# With an electronic energy of -76.40, electronic+ZPE = -76.40 + 0.03 = -76.37.
#
# The correction is 0.03 so that electronic+ZPE (-76.37) differs from the enthalpy
# (-76.38): at the old 0.02 the two coincided, and the assertion meant to pin the ZPE
# arithmetic could not tell it from "return the enthalpy" — a degenerate fixture is an
# assertion that cannot fail.
THERMOCHEMISTRY_BLOCK = """\
-------------------------
THERMOCHEMISTRY AT 298.15K
-------------------------

Zero point energy                ...      0.03000000 Eh      18.83 kcal/mol
Total Enthalpy                   ...    -76.38000000 Eh
Final Gibbs free energy          ...    -76.41000000 Eh
"""


def synthetic_pes_segment(
    *,
    coords: list[tuple[str, float, float, float]],
    energy: float,
    intermediate_energies: list[float] | None = None,
) -> str:
    """Build one PES segment.

    A PES segment contains potentially multiple coordinate blocks (the
    optimization-cycle intermediates) plus one or more
    ``FINAL SINGLE POINT ENERGY`` lines. The last of each is the
    converged geometry / energy. The segment ends with the
    ``*** OPTIMIZATION RUN DONE ***`` marker the splitter looks for.
    """
    parts: list[str] = []
    intermediates = intermediate_energies or []
    # An optional intermediate cycle so parse_pes can prove "last wins"
    for inter_energy in intermediates:
        parts.append("CARTESIAN COORDINATES (ANGSTROEM)")
        parts.append("---------------------------------")
        for sym, x, y, z in coords:
            parts.append(f"  {sym:2s}  {x + 99:.6f}  {y + 99:.6f}  {z + 99:.6f}")
        parts.append("")
        parts.append(f"FINAL SINGLE POINT ENERGY     {inter_energy}")
    parts.append("CARTESIAN COORDINATES (ANGSTROEM)")
    parts.append("---------------------------------")
    for sym, x, y, z in coords:
        parts.append(f"  {sym:2s}  {x:.6f}  {y:.6f}  {z:.6f}")
    parts.append("")
    parts.append(f"FINAL SINGLE POINT ENERGY     {energy}")
    parts.append("*** OPTIMIZATION RUN DONE ***")
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Captured, not synthesised — ORCA 6.1.1, verbatim. Do not edit to suit a parser.
# ---------------------------------------------------------------------------

ORCA_ERROR_TERMINATION_STEM = "step2_5-54"
"""Basename the captured abort was written under, kept so the ``.err`` sidecar pairs with it.

:func:`chemrefine.engines.orca.output.coordinator._stderr_tail` finds the stderr by swapping
the output's suffix, so the two files have to share a stem for the quoted tail to be found."""

ORCA_ERROR_TERMINATION_OUT = """\
 Group   1 Type H   : 3s contracted to 1s pattern {3}

Atom   0H    basis set group =>   1
Atom   1H    basis set group =>   1
sh: 1: /opt/orca/orca_startup: not found

ORCA finished by error termination in Startup
Calling Command: /opt/orca/orca_startup step2_5-54.int.tmp 
[file orca_tools/qcmsg.cpp, line 394]: 
  .... aborting the run
"""
"""A real ORCA 6.1.1 run that died in ``Startup`` — its whole ``.out``.

It carries the abort banner and no ``FINAL SINGLE POINT ENERGY``, which is the pair the
termination path turns into an :class:`~chemrefine.errors.OutputTerminationError` rather than
an "unparseable" ledger entry: the run is what failed, not the reader."""

ORCA_ERROR_TERMINATION_ERR = """\
sh: 1: /opt/orca/orca_startup: not found
"""
"""The job's stderr, holding the cause ORCA's own banner does not name.

The banner says which module aborted; this says why. Quoting it is what stops the failure
reading as "error termination in Startup" with the actual reason in a file nothing points at."""
