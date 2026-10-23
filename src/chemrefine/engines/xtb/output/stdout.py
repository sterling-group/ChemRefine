"""Read xtb's standard output: the energy, the verdicts, the thermochemistry, the files written.

xtb's stdout (the job's ``.out``) is the account of the run. Every reader here works on its
text, so the coordinator reads the file once. Each spelling matched is one xtb 6.7.1 prints,
held to the real outputs the contract fixtures were trimmed from. Every reader takes the
**last** match, because an optimisation reprints what it reports cycle by cycle.

Two things are read here rather than from the files xtb leaves beside its output:

* **Convergence.** xtb 6.7.1 touches ``.xtboptok`` whether or not the optimisation
  converged (``$opt maxcycle=2`` on an unconverged start leaves it, and exits 0), so the
  only verdict is the banner stdout prints.
* **Termination.** xtb's "normal termination of xtb" goes to stderr, so stdout is read
  for its own two endings: the closing ``* finished run on`` line, or the
  ``[ERROR] Program stopped due to fatal error`` block, which carries xtb's reasons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOTAL_ENERGY_RE = re.compile(r"^\s*\|\s*TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh\s*\|", re.MULTILINE)
_FINISHED_RE = re.compile(r"^\s*\* finished run on ", re.MULTILINE)
_FATAL_RE = re.compile(r"^\[ERROR\] Program stopped due to fatal error\s*$", re.MULTILINE)
_REASON_RE = re.compile(r"^-\d+- (.*?)\s*$")
_OPT_VERDICT_RE = re.compile(
    r"\*\*\* (?:(GEOMETRY OPTIMIZATION CONVERGED)|FAILED TO CONVERGE GEOMETRY OPTIMIZATION)"
    r" (?:AFTER|IN) \d+ ITERATIONS \*\*\*"
)
_WROTE_GEOMETRY_RE = re.compile(r"^\s*optimized geometry written to:\s*(\S+)\s*$", re.MULTILINE)
_WROTE_G98_RE = re.compile(r"^\s*writing <(\S+)> molden fake output\.\s*$", re.MULTILINE)

_THERMO_LINE_RE = r"^\s*::\s*{label}\s+(-?\d+\.\d+)\s+Eh\s*::\s*$"
_FREE_ENERGY_RE = re.compile(_THERMO_LINE_RE.format(label=r"total free energy"), re.MULTILINE)
_ZPE_RE = re.compile(_THERMO_LINE_RE.format(label=r"zero point energy"), re.MULTILINE)
_G_RRHO_RE = re.compile(_THERMO_LINE_RE.format(label=r"G\(RRHO\) contrib\."), re.MULTILINE)
_THERMO_TABLE_HEADER = "H(T)/Eh"
_THERMO_ROW_RE = re.compile(r"^\s*(\d+\.\d+)((?:\s+-?\d\.\d+E[-+]\d+){4})\s*$")

_G_MATCH_TOLERANCE = 1e-6
"""How closely a table row's ``G(T)`` must equal the summary's ``G(RRHO) contrib.`` (Eh)
to be its row: the table prints six significant figures, the summary twelve decimals."""


def parse_total_energy(text: str) -> float | None:
    """The last ``| TOTAL ENERGY … Eh |`` box (Hartree), or ``None`` when there is none.

    xtb prints the box once per evaluation it reports (once for a single point, again after
    an optimisation, again after a Hessian), and the last describes the final structure.
    """
    matches = _TOTAL_ENERGY_RE.findall(text)
    return float(matches[-1]) if matches else None


def parse_terminated_normally(text: str) -> bool:
    """Whether xtb finished, from stdout alone.

    ``False`` for xtb's fatal-error block, and for an output cut off before xtb's closing
    ``* finished run on`` line — a walltime kill leaves exactly that file. ``True`` when the
    closing line is there.
    """
    return _FATAL_RE.search(text) is None and _FINISHED_RE.search(text) is not None


def fatal_reasons(text: str) -> tuple[str, ...]:
    """xtb's own reasons for a fatal error — the ``-N-`` lines under its ``[ERROR]`` banner.

    ``()`` for an output with no fatal error. xtb prints the outermost reason first
    (``-3- Some atoms in the start geometry are *very* close``) down to the innermost
    (``-1- Found *very* short distance …``); all of them are kept, in that order.
    """
    found = _FATAL_RE.search(text)
    if found is None:
        return ()
    reasons: list[str] = []
    for line in text[found.end() :].splitlines()[1:]:
        reason = _REASON_RE.match(line)
        if reason is None:
            break
        reasons.append(reason.group(1))
    return tuple(reasons)


def parse_converged(text: str) -> bool | None:
    """The optimisation's verdict, the last one printed; ``None`` for a run that optimises nothing.

    An SCC that does not converge is not a verdict here: xtb stops with a fatal error, which
    :func:`parse_terminated_normally` reads.
    """
    verdicts = _OPT_VERDICT_RE.findall(text)
    return bool(verdicts[-1]) if verdicts else None


def written_geometry(text: str) -> str | None:
    """The optimised-geometry file stdout says xtb wrote (``xtbopt.xyz``), or ``None``.

    The name as stdout prints it, which leaves out the run's ``--namespace``: the file is
    ``<namespace>.xtbopt.xyz``."""
    matches = _WROTE_GEOMETRY_RE.findall(text)
    return matches[-1] if matches else None


def written_frequencies(text: str) -> str | None:
    """The Gaussian-style frequency file stdout says xtb wrote (``g98.out``), or ``None``.

    As stdout prints it, without the run's ``--namespace`` (see :func:`written_geometry`)."""
    matches = _WROTE_G98_RE.findall(text)
    return matches[-1] if matches else None


@dataclass(frozen=True)
class Thermochemistry:
    """Absolute thermochemistry (Hartree) from xtb's THERMODYNAMIC summary.

    ``enthalpy_hartree`` is ``None`` when no row of the temperature table can be tied to
    the summary (see :func:`parse_thermochemistry`).
    """

    gibbs_hartree: float
    enthalpy_hartree: float | None
    energy_zpe_hartree: float


def parse_thermochemistry(text: str, *, electronic_hartree: float) -> Thermochemistry | None:
    """G, H and E + ZPE from a Hessian run's output, or ``None`` when it has no summary.

    * ``gibbs`` is the summary's ``total free energy``: E + G(RRHO), the zero-point energy
      included.
    * ``energy_zpe`` is E + the summary's ``zero point energy``.
    * ``enthalpy`` is E + H(T), from the temperature table above the summary. The summary
      prints no enthalpy, and the table has a row per temperature when ``$thermo`` names
      several, so the row taken is the one whose ``G(T)`` is the summary's
      ``G(RRHO) contrib.``: the temperature the summary was computed at. H(T) includes the
      zero-point energy, as G(T) does (G(T) = H(T) - T·S).

    The last summary is read, the one for the final geometry.
    """
    free = _FREE_ENERGY_RE.findall(text)
    zpe = _ZPE_RE.findall(text)
    if not free or not zpe:
        return None
    return Thermochemistry(
        gibbs_hartree=float(free[-1]),
        enthalpy_hartree=_enthalpy(text, electronic_hartree),
        energy_zpe_hartree=electronic_hartree + float(zpe[-1]),
    )


def _enthalpy(text: str, electronic_hartree: float) -> float | None:
    """E + H(T) at the summary's temperature — see :func:`parse_thermochemistry`."""
    g_rrho = _G_RRHO_RE.findall(text)
    start = text.rfind(_THERMO_TABLE_HEADER)
    if not g_rrho or start < 0:
        return None
    target = float(g_rrho[-1])
    for line in text[start:].splitlines()[1:]:
        row = _THERMO_ROW_RE.match(line)
        if row is None:
            continue
        _h0, h_t, _ts, g_t = (float(token) for token in row.group(2).split())
        if abs(g_t - target) <= _G_MATCH_TOLERANCE:
            return electronic_hartree + h_t
    return None
