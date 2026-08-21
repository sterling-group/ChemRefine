"""ORB (Orbital Materials): the calculator chemrefine runs, and the trainer that fine-tunes it.

``model_name`` names a loader in :mod:`orb_models.forcefield.pretrained`
(e.g. ``orb_v3_conservative_inf_omat``, ``orb_v2``). The loader returns either
a model or a ``(model, atoms_adapter)`` tuple depending on the orb-models
version; both are handled. Requires ``pip install orb-models``.

One module for one library, so the environment is declared once (:data:`ORB`) and both
capabilities hang off it. Training takes the CHGNet route — orb-models publishes its
fine-tune entry point as an unpackaged repo-root script — so the trainer's
:meth:`OrbTrainer.run_training` hook rebuilds that script's slim loop from the packaged
utilities, driven by the shared :mod:`chemrefine.engines.mlip.train.driver`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from chemrefine.engines.mlip.registry import CalculatorSpec, MlipLibrary
from chemrefine.engines.mlip.train.base import ApiTrainerBase, TrainingPlan
from chemrefine.errors import ConfigError
from chemrefine.state import Structure

ORB = MlipLibrary(extra="mlip-orb", package="orb-models", import_name="orb_models")
"""The one declaration of what provides this library."""


@ORB.calculator("orb")
def _build_orb(spec: CalculatorSpec) -> Any:
    """ORB potential; ``model_name`` picks a loader from ``orb_models...pretrained``.

    The loader is the *architecture*, so it is selected by ``model_name`` even when the
    weights come from a file: a checkpoint carries weights, not an architecture, and the
    pretrained loaders take a ``weights_path`` (defaulting to the release URL, accepting
    a local file). An older loader without the keyword is reported as the version
    limitation it is (:class:`~chemrefine.errors.ConfigError`), not left as a
    ``TypeError`` naming neither the step nor the option.
    """
    from orb_models.forcefield import pretrained

    try:  # v3 layout
        from orb_models.forcefield.inference.calculator import ORBCalculator
    except ImportError:  # older layout
        from orb_models.forcefield.calculator import ORBCalculator

    loader = getattr(pretrained, spec.model_name, None)
    if loader is None:
        raise ValueError(
            f"unknown ORB model {spec.model_name!r}; pick a loader from "
            "orb_models.forcefield.pretrained (e.g. 'orb_v3_conservative_inf_omat')"
        )
    if spec.weights is None:
        loaded = loader(device=spec.device)
    else:
        try:
            loaded = loader(weights_path=str(spec.weights), device=spec.device)
        except TypeError as e:
            raise ConfigError(
                f"this orb-models version's {spec.model_name!r} loader takes no local "
                f"weights_path, so model_path cannot be honoured; upgrade orb-models "
                f"or drop model_path to run the named release"
            ) from e
    orbff = loaded[0] if isinstance(loaded, tuple) else loaded
    return ORBCalculator(orbff, device=spec.device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@ORB.trainer("orb")
class OrbTrainer(ApiTrainerBase):
    """Fine-tune an ORB model on a step's labelled structures.

    orb-models publishes its fine-tuning entry point as a repo-root script that is **not
    in the wheel** (no console script, ``packages = ["orb_models*"]``), so the trainer
    takes the same route as CHGNet's: a chemrefine-defined YAML rendered from the step
    template, and :meth:`run_training` — called by the shared
    :mod:`~chemrefine.engines.mlip.train.driver` in the backend env — re-implementing the
    script's slim loop from the utilities that *are* packaged
    (``orb_models.common.dataset`` / ``common.training`` / ``forcefield.pretrained``,
    the same import layout from the 0.6 floor through main). The schema:

    .. code-block:: yaml

        train_set: $TRAIN_SET          # required — ASE sqlite db written by the trainer
        run_name:  $RUN_NAME           # required — names the fixed final checkpoint
        base_model: orb_v3_conservative_inf_omat   # the pretrained loader = architecture
        start_from: $FOUNDATION_MODEL  # optional: a local checkpoint for the weights
        device:    $DEVICE
        seed:      $SEED
        epochs: 50
        learning_rate: 3e-4            # the script's own defaults
        batch_size: 100
        gradient_clip: 0.5
    """

    label = "ORB"
    driver_task = "orb"
    required_config_keys = ("train_set", "run_name", "base_model")
    missing_config_hint = (
        "the template must reference $TRAIN_SET and $RUN_NAME, and name a base_model "
        "(the pretrained loader)"
    )
    artifact_filename = "{run_name}.ckpt"
    """orb's own per-epoch names embed the epoch (``checkpoint_epoch{n}.ckpt``), which
    no later step could name before the run; the hook re-saves the final state under
    this fixed name, loadable through the pretrained loaders' ``weights_path`` — the
    same door the calculator builder opens for ``model_path``."""

    required_placeholders: ClassVar[frozenset[str]] = frozenset({"TRAIN_SET", "RUN_NAME"})
    """No ``VALID_SET``: orb's fine-tune loop is train-only — it has no evaluation pass, so
    requiring a validation file would demand data nothing reads (and ``needs_validation``
    stays False for the same reason). ``RUN_NAME`` is required because :meth:`artifact`
    derives from it, the MACE/CHGNet reasoning."""

    output_globs: ClassVar[tuple[str, ...]] = ("*.ckpt",)
    """The fixed final save and orb's own per-epoch ``checkpoint_epoch{n}.ckpt`` files —
    all written into the working directory, carried home from ``$WORK_DIR`` by this."""

    def write_split(self, plan: TrainingPlan, name: str, structures: tuple[Structure, ...]) -> Path:
        """One split as an ASE sqlite database — the one format orb's dataset reads.

        ``AseSqliteDataset`` is constructed on a database path ("you must convert your
        data into this format" — the script's own words), and the labels ride on each
        row's calculator exactly as :func:`~chemrefine.engines.mlip.train.base.
        labelled_atoms` attaches them — the FAIRChem trainer's sqlite precedent, one
        library over. The valid/test splits are written when present so the data a user
        held out stays visible on disk, but orb's loop reads only the training set.
        """
        from ase.db import connect

        from chemrefine.engines.mlip.train.base import labelled_atoms

        path = plan.run_dir / f"{name}.db"
        path.unlink(missing_ok=True)  # ase.db appends; a rerun must not stack rows
        with connect(str(path)) as db:
            for struct in structures:
                db.write(labelled_atoms(struct))
        return path

    # -- the backend-side half (runs under the mlip-orb env's interpreter) -------------

    def train_with_library(self, config: dict[str, Any]) -> int:
        """orb's fine-tune loop, rebuilt from the packaged utilities.

        The published loop lives in an unpackaged script, so this is a re-implementation,
        deliberately slim and pinned line-for-line to that script's shape: the pretrained
        loader with ``train=True`` (``weights_path`` when ``start_from`` names a local
        checkpoint), ``build`` of the sqlite dataset with the adapter's own ``batch``
        collate, ``get_optim`` for optimizer + scheduler, and per-epoch
        ``checkpoint_epoch{n}.ckpt`` saves — plus the one thing the script never had, a
        fixed-name final save for :meth:`artifact`. One full pass per epoch, rather than
        the script's fixed ``num_steps``: an epoch that sees every structure is the
        behaviour a training step's re-run can reproduce.
        """
        import torch
        from orb_models.common.dataset import property_definitions
        from orb_models.common.dataset.ase_sqlite_dataset import AseSqliteDataset
        from orb_models.common.dataset.loaders import worker_init_fn
        from orb_models.common.training.util import get_optim, init_device
        from orb_models.common.utils import seed_everything
        from orb_models.forcefield import pretrained
        from torch.utils.data import BatchSampler, DataLoader, RandomSampler

        device = init_device()
        seed_everything(int(config.get("seed", 42)))
        base_model = str(config["base_model"])
        loader_fn = getattr(pretrained, base_model, None)
        if loader_fn is None:
            raise SystemExit(
                f"orb training: unknown base_model {base_model!r}; pick a loader from "
                f"orb_models.forcefield.pretrained"
            )
        start_from = str(config.get("start_from") or "")
        loader_kwargs: dict[str, Any] = {"device": device, "train": True}
        if start_from:
            loader_kwargs["weights_path"] = start_from
        model, atoms_adapter = loader_fn(**loader_kwargs)

        dataset = AseSqliteDataset(
            str(config["run_name"]),
            str(config["train_set"]),
            atoms_adapter=atoms_adapter,
            target_config=property_definitions.instantiate_property_config(None),
            augmentations=[],
        )
        batch_size = int(config.get("batch_size", 100))
        train_loader = DataLoader(
            dataset,
            num_workers=0,
            worker_init_fn=worker_init_fn,
            collate_fn=atoms_adapter.batch,
            batch_sampler=BatchSampler(
                RandomSampler(dataset), batch_size=batch_size, drop_last=False
            ),
        )

        epochs = int(config.get("epochs", 50))
        steps_per_epoch = len(train_loader)
        optimizer, lr_scheduler = get_optim(
            float(config.get("learning_rate", 3e-4)), epochs * steps_per_epoch, model
        )
        clip = float(config.get("gradient_clip", 0.5))

        model.to(device=device)
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                out = model.loss(batch.to(device))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
            torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.ckpt")

        target = Path(self.artifact_filename.format(run_name=config["run_name"]))
        torch.save(model.state_dict(), target)
        print(f"orb training: saved {target}")
        return 0
