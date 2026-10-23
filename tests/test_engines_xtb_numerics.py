"""Live numerical checks of an xtb binary: its gradients against its energies, its Hessian
against its gradients.

A recorded golden is only as good as the build that recorded it. These run the binary at
hand and hold its numbers to themselves, so a miscompiled build fails here rather than
handing wrong forces to an optimiser or an MLIP fit. This has happened: conda-forge's
xtb 6.7.1 aarch64 gfortran build (``edcfbbe``) gives GFN1 and GFN2 gradients wrong by
about their own size, while its GFN-FF is right. Point the tests at any build with
``CHEMREFINE_XTB=/path/to/xtb pytest -m integration tests/test_engines_xtb_numerics.py``;
without it they use the ``xtb`` on ``PATH``.

xtb has no analytic Hessian for any method: ``--hess`` is a central difference of its
analytic gradients. So the checks are:

* **Gradient vs. energies.** The forces the engine parses (through the command line it
  builds) are held to central differences of the energy, step 0.001 bohr, on a distorted
  molecule, for every method in gas and in both Born solvation models, and for a charged
  doublet. A good build agrees to ~2e-7 Eh/bohr, the step's own truncation error (it
  scales as h²); the bound is 1e-6.
* **Hessian vs. gradients.** At an optimised geometry, xtb's ``hessian`` file is held to
  central differences of its gradients at xtb's own step (0.005 bohr), and the
  frequencies the engine parses from ``g98.out`` to that Hessian's. In gas phase this is
  tight: 3e-5 Eh/bohr² and 0.3 cm⁻¹ on a good build, bounds 1e-4 and 1 cm⁻¹.
  **In implicit solvent only the stiff modes are held.** With ALPB or GBSA, xtb's
  ``--hess`` differs from differences of the same binary's solvated gradients by 4-27
  cm⁻¹ on a soft torsion, for every method, and the solvated surface is not rotationally
  invariant at that level either (a 0.5 rad rotation of the molecule moves the same mode
  by 6-35 cm⁻¹, against nothing in gas phase). Modes above 1000 cm⁻¹ agree to within
  3.6 cm⁻¹ (0.3%) on a good build, and are held to 5 cm⁻¹. The solvated *gradients* are
  held as tightly as the gas-phase ones, above, so this bounds only how far xtb's own
  ``--hess`` may stray from them in solvent.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from ase.data import atomic_masses_iupac2016, atomic_numbers
from numpy.typing import NDArray
from test_e2e_live import real_xtb, skip_unless_required

from chemrefine.engines.api import ParsedResult
from chemrefine.engines.xtb import command
from chemrefine.engines.xtb.engine import XtbEngine
from chemrefine.engines.xtb.options import XtbOptions
from chemrefine.engines.xtb.output import parse_xtb
from chemrefine.quantities import BOHR_TO_ANGSTROM, HARTREE_PER_BOHR_TO_EV_PER_A

pytestmark = pytest.mark.integration

_SYMBOLS = ("C", "O", "H", "H", "H", "H")
_DISTORTED_METHANOL = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.45, 0.1, -0.05],
        [-0.4, 1.05, 0.1],
        [-0.35, -0.5, 0.95],
        [-0.42, -0.45, -0.88],
        [1.8, -0.7, 0.3],
    ]
)
"""Methanol pulled off its minimum and every symmetry, so no gradient component is zero by
symmetry and every one is tested; the O-H donor gives the solvation models something to do."""

_ACCURACY = 1e-4
"""``--acc`` for every run here: an SCC converged far enough that differences of energies
are differences of the surface, not of the iteration."""

_GRADIENT_STEP_BOHR = 0.001
_GRADIENT_BOUND = 1e-6
"""Eh/bohr. The step's truncation error is ~2e-7 on a good build; a bad one is off by 1e-2."""

_HESSIAN_STEP_BOHR = 0.005
"""xtb's own ``--hess`` step, so the two Hessians share their truncation error."""
_HESSIAN_BOUND = 1e-4
"""Eh/bohr², gas phase."""
_FREQUENCY_BOUND = 1.0
"""cm⁻¹, every mode, gas phase."""
_STIFF_CM = 1000.0
_SOLVENT_FREQUENCY_BOUND = 5.0
"""cm⁻¹, the modes above :data:`_STIFF_CM`, implicit solvent (see the docstring): the most
a good build was seen off by is 3.6 (GFN2/ALPB at 1131 cm⁻¹, GFN1/ALPB's O-H stretch)."""

_AMU_IN_ELECTRON_MASSES = 1822.888486209
_HARTREE_IN_WAVENUMBERS = 219474.6313632

_GAS: dict[str, object] = {}
_ALPB: dict[str, object] = {"solvation": "alpb", "solvent": "water"}
_GBSA: dict[str, object] = {"solvation": "gbsa", "solvent": "water"}


def _xtb() -> str:
    """The binary under test, by the live tier's rule (:func:`test_e2e_live.real_xtb`)."""
    found = real_xtb()
    if not found:
        skip_unless_required(
            "the xtb numerics need a binary: set CHEMREFINE_XTB or put xtb on PATH"
        )
    return found


@dataclass
class _Runner:
    """Runs one xtb calculation per call, each in a directory of its own."""

    xtb: str
    root: Path
    count: int = 0

    def __call__(
        self,
        positions: NDArray[np.float64],
        operation: str,
        options: dict[str, object],
        *,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> tuple[ParsedResult, Path]:
        """The engine's parse of one run at ``positions`` (Å), and the run's directory."""
        self.count += 1
        work = self.root / str(self.count)
        work.mkdir()
        geometry = work / "step1_0_inp.xyz"
        rows = "".join(
            f"{s} {x:.12f} {y:.12f} {z:.12f}\n"
            for s, (x, y, z) in zip(_SYMBOLS, positions, strict=True)
        )
        geometry.write_text(f"{len(_SYMBOLS)}\n\n{rows}", encoding="utf-8")
        words = command.arguments(
            XtbOptions.from_raw({**options, "accuracy": _ACCURACY}),
            operation=operation,
            charge=charge,
            multiplicity=multiplicity,
        )
        env = {"PATH": os.environ.get("PATH", ""), "OMP_NUM_THREADS": "1,1"}
        parameters = XtbEngine.parameter_dir(str(Path(self.xtb).resolve()))
        if parameters is not None:
            env["XTBPATH"] = parameters
        out = work / "step1_0.out"
        with out.open("w", encoding="utf-8") as stdout:
            done = subprocess.run(
                # The namespace is the stdout's stem, as the engine runs it: the parse looks
                # for `step1_0.gradient` and the rest.
                [self.xtb, geometry.name, "--namespace", out.stem, *words],
                cwd=work,
                stdout=stdout,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=300,
            )
        assert done.returncode == 0, f"xtb failed in {work}:\n{done.stderr[-2000:]}"
        (parsed,) = parse_xtb(out, input_geometry=geometry)
        return parsed, work


@pytest.fixture
def run(tmp_path: Path) -> _Runner:
    """A runner over the binary under test."""
    return _Runner(_xtb(), tmp_path)


def _gradient(parsed: ParsedResult) -> NDArray[np.float64]:
    """The parsed forces as a gradient in Eh/bohr: undoes the engine's sign and units."""
    assert parsed.forces_ev_per_a is not None, "the run reported no gradient"
    return -parsed.forces_ev_per_a / HARTREE_PER_BOHR_TO_EV_PER_A


# ---------------------------------------------------------------------------
# Gradient vs. energies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solvation", [_GAS, _ALPB, _GBSA], ids=["gas", "alpb", "gbsa"])
@pytest.mark.parametrize("method", ["gfn2", "gfn1", "gfnff"])
def test_the_gradient_is_the_derivative_of_the_energy(
    run: _Runner, method: str, solvation: dict[str, object]
):
    """Every component of the analytic gradient, held to a central difference of energies."""
    _assert_gradient_matches_energies(run, {"method": method, **solvation})


@pytest.mark.parametrize("solvation", [_GAS, _ALPB], ids=["gas", "alpb"])
def test_the_gradient_of_a_charged_doublet_is_the_derivative_of_its_energy(
    run: _Runner, solvation: dict[str, object]
):
    """``--chrg`` / ``--uhf`` reach the run, and the open-shell gradient is as right."""
    _assert_gradient_matches_energies(run, solvation, charge=1, multiplicity=2)


def _assert_gradient_matches_energies(
    run: _Runner, options: dict[str, object], *, charge: int = 0, multiplicity: int = 1
) -> None:
    analytic = _gradient(
        run(_DISTORTED_METHANOL, "sp", options, charge=charge, multiplicity=multiplicity)[0]
    )
    step = _GRADIENT_STEP_BOHR * BOHR_TO_ANGSTROM
    numeric = np.zeros_like(analytic)
    for atom in range(len(_SYMBOLS)):
        for axis in range(3):
            energies = []
            for sign in (1.0, -1.0):
                moved = _DISTORTED_METHANOL.copy()
                moved[atom, axis] += sign * step
                parsed, _ = run(moved, "sp", options, charge=charge, multiplicity=multiplicity)
                energies.append(parsed.energy_hartree)
            numeric[atom, axis] = (energies[0] - energies[1]) / (2 * _GRADIENT_STEP_BOHR)
    error = np.abs(analytic - numeric).max()
    assert np.abs(analytic).max() > 1e-2, "the geometry must be off its minimum to test anything"
    assert error < _GRADIENT_BOUND, (
        f"{options}: the analytic gradient differs from the energies' by {error:.2e} Eh/bohr "
        f"(largest component {np.abs(analytic).max():.2e}) — is this build miscompiled?"
    )


# ---------------------------------------------------------------------------
# Hessian vs. gradients
# ---------------------------------------------------------------------------


def _read_hessian(path: Path) -> NDArray[np.float64]:
    """xtb's ``hessian`` file (Turbomole ``$hessian``, Eh/bohr²) as a 3N x 3N matrix."""
    values = [float(token) for token in path.read_text(encoding="utf-8").split()[1:]]
    n = round(len(values) ** 0.5)
    assert n * n == len(values), f"{path} is not a square matrix"
    return np.array(values).reshape(n, n)


def _frequencies(hessian: NDArray[np.float64]) -> NDArray[np.float64]:
    """The vibrational frequencies (cm⁻¹) of a Cartesian Hessian at a minimum, ascending.

    Mass-weighted and diagonalised; the six smallest eigenvalues — translations and
    rotations, near zero at a minimum — are dropped.
    """
    masses = np.repeat([atomic_masses_iupac2016[atomic_numbers[s]] for s in _SYMBOLS], 3)
    weighted = hessian / np.sqrt(np.outer(masses, masses) * _AMU_IN_ELECTRON_MASSES**2)
    eigenvalues = np.linalg.eigvalsh(weighted)
    order = np.argsort(np.abs(eigenvalues))
    vibrations = eigenvalues[np.sort(order[6:])]
    return np.sign(vibrations) * np.sqrt(np.abs(vibrations)) * _HARTREE_IN_WAVENUMBERS


@pytest.mark.parametrize(
    ("method", "solvation"),
    [
        ("gfn2", _GAS),
        ("gfn1", _GAS),
        ("gfnff", _GAS),
        ("gfn2", _ALPB),
        ("gfn2", _GBSA),
        ("gfn1", _ALPB),
        ("gfnff", _ALPB),
    ],
    ids=["gfn2-gas", "gfn1-gas", "gfnff-gas", "gfn2-alpb", "gfn2-gbsa", "gfn1-alpb", "gfnff-alpb"],
)
def test_the_hessian_is_the_derivative_of_the_gradient(
    run: _Runner, method: str, solvation: dict[str, object]
):
    """xtb's Hessian and the parsed frequencies, held to differences of xtb's gradients."""
    options = {"method": method, **solvation}
    minimum, _ = run(_DISTORTED_METHANOL, "opt_sp", {**options, "opt_level": "extreme"})
    assert minimum.converged is True
    positions = minimum.positions
    frequency_run, work = run(positions, "freq", options)
    xtb_hessian = _read_hessian(work / "step1_0.hessian")

    step = _HESSIAN_STEP_BOHR * BOHR_TO_ANGSTROM
    n = 3 * len(_SYMBOLS)
    numeric = np.zeros((n, n))
    for column in range(n):
        atom, axis = divmod(column, 3)
        gradients = []
        for sign in (1.0, -1.0):
            moved = positions.copy()
            moved[atom, axis] += sign * step
            gradients.append(_gradient(run(moved, "sp", options)[0]))
        numeric[:, column] = ((gradients[0] - gradients[1]) / (2 * _HESSIAN_STEP_BOHR)).ravel()
    numeric = 0.5 * (numeric + numeric.T)

    assert frequency_run.frequencies is not None, "a Hessian run the engine read no modes from"
    parsed = np.array([frequency_run.frequencies[k] for k in sorted(frequency_run.frequencies)])
    differenced = _frequencies(numeric)
    assert len(parsed) == len(differenced) == n - 6
    deviation = np.abs(parsed - differenced)

    if not solvation:
        worst = np.abs(xtb_hessian - numeric).max()
        assert worst < _HESSIAN_BOUND, (
            f"{options}: xtb's Hessian differs from differences of its gradients by "
            f"{worst:.2e} Eh/bohr² — is this build miscompiled?"
        )
        held, bound = np.ones_like(deviation, dtype=bool), _FREQUENCY_BOUND
    else:
        held, bound = differenced > _STIFF_CM, _SOLVENT_FREQUENCY_BOUND
        assert held.sum() >= 8, "methanol has eight modes above 1000 cm⁻¹ at every level here"
    assert deviation[held].max() < bound, (
        f"{options}: parsed frequencies {np.round(parsed[held], 2)} vs differenced "
        f"{np.round(differenced[held], 2)} cm⁻¹"
    )
