# PySCF starter. Rendered per structure: $XYZ_PATH / $CHARGE / $MULTIPLICITY come
# from the pipeline, $METHOD / $XC / $BASIS / $DF from the step options.
# Assign `energy_hartree` (optionally `positions_angstrom`, `gradient_hartree_per_bohr`,
# `converged`) — the appended output footer harvests them.
from pyscf import dft, gto, scf

mol = gto.M(
    atom="$XYZ_PATH",
    basis="$BASIS",
    charge=$CHARGE,
    spin=$MULTIPLICITY - 1,
)

if "$METHOD" == "hf":
    mf = scf.HF(mol)
else:
    mf = dft.KS(mol, xc="$XC")
if $DF:
    mf = mf.density_fit()

energy_hartree = mf.kernel()
# PySCF returns the last iterate rather than raising, so the verdict is reported here:
# False is ledgered as a convergence failure instead of ranking as a result.
converged = bool(mf.converged)
