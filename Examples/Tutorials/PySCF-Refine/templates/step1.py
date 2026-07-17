# Direct-PySCF refinement step. ChemRefine renders this per structure:
# $XYZ_PATH / $CHARGE / $MULTIPLICITY come from the pipeline, and
# $METHOD / $XC / $BASIS from the YAML step options.
# Assign `energy_hartree` — the appended output footer harvests it.
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

energy_hartree = mf.kernel()
