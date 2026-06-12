"""Shared PySCF compute helpers used by both extopt and direct engines.

Exposes :func:`build_mol`, :func:`run_dft`,
:func:`get_active_space_tensors`, :func:`save_tensors`, and
:func:`print_tensors_file`. All PySCF imports are lazy (inside the
helper bodies) so this module imports cleanly when ``pyscf`` isn't
installed — the integration suite patches them out under mocked CPU
backends.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemrefine.quantities import BOHR_TO_ANGSTROM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Molecule + SCF
# ---------------------------------------------------------------------------


def build_mol(
    *,
    symbols: tuple[str, ...] | list[str],
    positions_angstrom: NDArray[np.float64],
    charge: int,
    multiplicity: int,
    basis: str,
) -> Any:
    """Return a ``pyscf.gto.Mole`` built from Cartesian coordinates.

    Coordinates are converted to Bohr internally so ``mol.unit = "Bohr"``
    and downstream gradients come out in Hartree/Bohr (matching ORCA's
    ``.engrad`` convention).
    """
    from pyscf import gto

    bohr_per_ang = 1.0 / BOHR_TO_ANGSTROM
    coords_bohr = [
        (x * bohr_per_ang, y * bohr_per_ang, z * bohr_per_ang)
        for x, y, z in positions_angstrom.tolist()
    ]
    atom = [(sym, coord) for sym, coord in zip(symbols, coords_bohr, strict=True)]

    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "Bohr"
    mol.charge = int(charge)
    mol.spin = int(multiplicity) - 1  # mult = 2S + 1 → spin = 2S
    mol.basis = basis
    mol.build()
    return mol


def _build_scf(
    mol: Any, *, method: str, xc: str, want_gpu: bool, closed_shell: bool
) -> tuple[Any, bool, str]:
    """Construct the (un-run) SCF object; return ``(mf, gpu_used, gpu_msg)``.

    ``method`` is ``"hf"`` (RHF/UHF) or DFT (RKS/UKS). For DFT, ``want_gpu`` tries
    the gpu4pyscf RKS/UKS and falls back to CPU pyscf on any import/init failure;
    HF stays CPU-only.
    """
    from pyscf import dft, scf

    if method == "hf":
        mf = scf.RHF(mol) if closed_shell else scf.UHF(mol)
        hf_msg = "GPU requested but HF GPU path not enabled; using CPU HF" if want_gpu else ""
        return mf, False, hf_msg

    if want_gpu:
        try:
            from gpu4pyscf.dft import RKS as GPU_RKS
            from gpu4pyscf.dft import UKS as GPU_UKS

            mf = GPU_RKS(mol) if closed_shell else GPU_UKS(mol)
            mf.xc = xc
            return mf, True, "GPU4PySCF DFT backend (gpu4pyscf.dft.RKS/UKS)"
        except Exception as e:
            mf = dft.RKS(mol) if closed_shell else dft.UKS(mol)
            mf.xc = xc
            return mf, False, f"Failed to init gpu4pyscf; fell back to CPU DFT ({e})"

    mf = dft.RKS(mol) if closed_shell else dft.UKS(mol)
    mf.xc = xc
    return mf, False, ""


def run_dft(
    mol: Any,
    *,
    method: str = "dft",
    xc: str = "pbe",
    use_df: bool = False,
    want_gpu: bool = False,
    nthreads: int = 1,
    dograd: bool = True,
) -> tuple[float, list[list[float]], dict[str, Any], Any]:
    """Solve the SCF and return ``(energy, gradient, meta, mf)``.

    ``method`` is ``"dft"`` (RKS/UKS) or ``"hf"`` (RHF/UHF). The
    GPU path uses :mod:`gpu4pyscf.dft` when ``want_gpu`` is true and the
    import succeeds; otherwise the calculation runs on CPU.

    ``gradient`` is returned as a list of ``[gx, gy, gz]`` rows when
    ``dograd`` is true, otherwise as an empty list. ``mf`` is returned
    so the caller can run downstream tensor extraction.
    """
    from pyscf import lib

    lib.num_threads(nthreads)
    t0 = time.perf_counter()
    closed_shell = mol.spin == 0
    mf, gpu_used, gpu_msg = _build_scf(
        mol, method=method, xc=xc, want_gpu=want_gpu, closed_shell=closed_shell
    )

    if use_df:
        try:
            mf = mf.density_fit()
        except Exception as e:
            logger.warning("density_fit() failed (%s); continuing without DF", e)

    energy = float(mf.kernel())
    converged = bool(getattr(mf, "converged", False))

    gradient_rows: list[list[float]] = []
    grad_norm = 0.0
    if dograd:
        g = mf.nuc_grad_method().kernel()
        gradient_rows = [[float(c) for c in row] for row in np.asarray(g).reshape(-1, 3)]
        grad_norm = float(np.linalg.norm(g))

    meta = {
        "method": method,
        "xc": xc,
        "use_df": use_df,
        "gpu_requested": want_gpu,
        "gpu_used": gpu_used,
        "gpu_msg": gpu_msg,
        "converged": converged,
        "energy_hartree": energy,
        "grad_norm": grad_norm,
        "elapsed_seconds": time.perf_counter() - t0,
        "nthreads": nthreads,
    }
    return energy, gradient_rows, meta, mf


# ---------------------------------------------------------------------------
# Active-space tensor extraction
# ---------------------------------------------------------------------------


def get_active_space_tensors(
    mol: Any, mf: Any, *, localized: bool = False
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(nuc, h1_mo, h2_mo)`` — the one- and two-electron MO-basis tensors.

    When ``localized`` is true, Boys-localize the occupied and virtual
    blocks separately before transforming. The two-electron tensor uses
    :func:`pyscf.ao2mo.incore.full`.
    """
    from pyscf import ao2mo

    nuc = float(mol.energy_nuc())
    ao_kin = mol.intor("int1e_kin")
    ao_nuc = mol.intor("int1e_nuc")
    ao_obi = ao_kin + ao_nuc
    ao_eri = mol.intor("int2e")
    coeff = mf.mo_coeff

    if localized:
        from pyscf import lo

        nocc = int((mf.mo_occ > 0).sum())
        coeff_occ = coeff[:, :nocc]
        coeff_vir = coeff[:, nocc:]
        loc_occ = lo.Boys(mol, coeff_occ).kernel(verbose=0)
        loc_vir = lo.Boys(mol, coeff_vir).kernel(verbose=0)
        mo_final = np.column_stack((loc_occ, loc_vir))
    else:
        mo_final = coeff

    h1 = np.asarray(mo_final.T @ ao_obi @ mo_final, dtype=float)
    h2 = np.asarray(ao2mo.incore.full(ao_eri, mo_final), dtype=float)
    return nuc, h1, h2


def save_tensors(
    *,
    path: str | Path,
    nuc: float,
    h1: NDArray[np.float64],
    h2: NDArray[np.float64],
) -> Path:
    """Write ``nuc / h1 / h2`` to a single compressed ``.npz`` and return its path.

    Keys (``hc`` for the nuclear-repulsion scalar, ``h1e`` for the
    one-electron tensor, ``h2e`` for the two-electron tensor) match
    the format the downstream inspection helpers expect.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, hc=nuc, h1e=h1, h2e=h2)
    return target


def print_tensors_file(npz_file: str | Path) -> None:
    """Pretty-print the tensors stored in a :func:`save_tensors` output file."""
    data = np.load(str(npz_file))
    print(f"File: {npz_file}")
    print("hc (scalar):")
    print(data["hc"])
    print("h1e (one-electron tensor):")
    print(data["h1e"])
    print("h2e (two-electron tensor):")
    print(data["h2e"])
