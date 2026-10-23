"""AnySolv (Ni & Lee, arXiv:2610.17226): an ML correction trained on DFT-SMD solvation energies.

AnySolv's ``predict_solvation_energy`` takes a geometry, its total charge and **multiplicity**
(its ``spin`` argument is the multiplicity, the value ChemRefine stamps as
``atoms.info["spin"]``) and an SMD solvent name, and returns ΔE (eV) and ΔF (eV/Å). Added to
a gas-phase MLIP, the sum is the MLIP analogue of a DFT-SMD energy.

Two models. ``compact`` ships in the wheel (MIT, no login) and is the default. ``moe`` is
the paper's model; its weights (about 1.1 GB) are gated on Hugging Face under the FAIR
Chemistry License and are not in the wheel. A step names the model it wants, or asks for
``auto``: ``moe`` where its weights are found, else ``compact``, chosen once per step before
the run and keyed by the model chosen. AnySolv's own choice, made per process from what is
on disk, is never used.

The MoE weights are looked for, in order: the step's ``solvation_weights`` file, the
Hugging Face cache (``hf download antonknee/anysolv model_moe.pt``, which needs
``huggingface_hub`` in the env), and AnySolv's own ``models/`` directory, where an editable
checkout keeps them.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from chemrefine.engines.mlip.solvation._registry import (
    Correction,
    PythonRequirement,
    SolvationLibrary,
    SolvationSpec,
    WeightsMissing,
)
from chemrefine.errors import ConfigError

if TYPE_CHECKING:
    from ase import Atoms

_HF_REPO = "antonknee/anysolv"
"""The Hugging Face repository the gated MoE weights are downloaded from."""

_MOE = "model_moe"


def _hf_cached(filename: str) -> tuple[str | None, str]:
    """``filename`` from the Hugging Face cache, or ``None`` and why; never the network."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None, "huggingface_hub is not installed, so its cache was not searched"
    found = try_to_load_from_cache(repo_id=_HF_REPO, filename=filename)
    if isinstance(found, str):
        return found, ""
    return None, f"not in the Hugging Face cache for {_HF_REPO}"


def _locate(checkpoint: str, weights: str | None) -> str:
    """The weights file of ``checkpoint`` (``model_moe`` / ``model_compact``), by path.

    For ``model_moe``: the step's ``solvation_weights`` when it names one, then the Hugging
    Face cache, then the package's ``models/``. A ``solvation_weights`` that is not a file is
    a config error and is never passed over, so a mistyped path cannot quietly become
    ``compact`` under ``auto``. ``model_compact`` ships in the package.
    """
    looked: list[str] = []
    if checkpoint == _MOE:
        if weights:
            if not Path(weights).is_file():
                raise ConfigError(f"solvation_weights: {weights} is not a file")
            return weights
        looked.append("no `solvation_weights` named")
        cached, why = _hf_cached(f"{_MOE}.pt")
        if cached:
            return cached
        looked.append(why)
    packaged = resources.files("anysolv.models").joinpath(f"{checkpoint}.pt")
    if packaged.is_file():
        return str(packaged)
    looked.append(f"not in {packaged}")
    hint = (
        f"; accept the FAIR Chemistry License at huggingface.co/{_HF_REPO} and run "
        f"`hf download {_HF_REPO} {_MOE}.pt`, or name the file as `solvation_weights`"
        if checkpoint == _MOE
        else ""
    )
    raise WeightsMissing(f"anysolv's {checkpoint} weights: {', '.join(looked)}{hint}")


ANYSOLV = SolvationLibrary(
    name="anysolv",
    requirement=PythonRequirement(package="anysolv", import_name="anysolv"),
    models={"compact": "model_compact", "moe": _MOE},
    default_model="compact",
    pairs_with=frozenset({"mlip-fairchem"}),
    auto=("moe", "compact"),
    weighted=frozenset({"moe"}),
    locate=_locate,
)
"""The one declaration of AnySolv. FAIRChem only, for now: the MACE env caps torch below
2.15, and AnySolv has not yet been run under that cap."""


@ANYSOLV.correction
def _build_anysolv(spec: SolvationSpec) -> Correction:
    """Resolve the checkpoint and the solvent once; return the per-geometry correction.

    Both are checked here, when the calculator is built, rather than on the first
    evaluation: an unknown solvent or a missing checkpoint is then reported before an
    optimiser or an ExtOpt server has started. ``"gas"``, ``"vacuum"`` and the like are
    refused along with unknown names: AnySolv reads them as no solvent, which would make a
    solvated step a gas-phase one with nothing said.

    ``inference_settings="default"`` is named rather than left to AnySolv, because it keeps
    the MoE model **unmerged**. The MoE checkpoint mixes its experts by each structure's
    composition, charge, multiplicity and solvent on every evaluation, so one loaded model
    serves every species a step holds. AnySolv's ``fast_gpu`` preset merges the experts for
    the first structure it sees and locks the model to that one species: every other
    species in the step, or in an ExtOpt server's lifetime, would need a model of its own.
    """
    from anysolv import predict_solvation_energy
    from anysolv.solvent import resolve_solvent_name

    checkpoint = _locate(spec.model, spec.weights)
    solvent = resolve_solvent_name(spec.solvent)
    if not solvent:
        raise ConfigError(
            f"solvation: anysolv has no solvent {spec.solvent!r} — name one in its SMD table "
            f"(`anysolv.solvent.list_solvents()`)"
        )

    def correction(atoms: Atoms, charge: int, multiplicity: int) -> tuple[float, np.ndarray]:
        delta_e, delta_f = predict_solvation_energy(
            atoms,
            charge=charge,
            spin=multiplicity,
            solvent=solvent,
            checkpoint=checkpoint,
            device=spec.device,
            inference_settings="default",
        )
        return float(delta_e), np.asarray(delta_f, dtype=float)

    return correction
