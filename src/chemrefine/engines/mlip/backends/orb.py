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
    orbff, _adapter = _model_and_adapter(loaded)
    return ORBCalculator(orbff, device=spec.device)


def _model_and_adapter(loaded: Any) -> tuple[Any, Any | None]:
    """Both loader return shapes, one spelling: ``(model, atoms_adapter-or-None)``.

    The pretrained loaders return a bare model or a ``(model, atoms_adapter)`` tuple
    depending on the orb-models version. Normalized here for the builder *and* the
    trainer, because two spellings of the shape is how the trainer came to unpack
    unconditionally — a bare-model version crashed fine-tuning with a naked
    ``TypeError: cannot unpack`` while the builder handled it a screen away.
    """
    return loaded if isinstance(loaded, tuple) else (loaded, None)


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
    the same import layout from the 0.6 floor through main). The schema — backend knobs
    only; device, seed and the foundation weights are plan facts and arrive on the
    driver's own command line, never through the template:

    .. code-block:: yaml

        train_set: $TRAIN_SET          # required — ASE sqlite db written by the trainer
        run_name:  $RUN_NAME           # required — names the fixed final checkpoint
        base_model: orb_v3_conservative_inf_omat   # the pretrained loader = architecture
        epochs: 50
        learning_rate: 3e-4            # the script's own defaults
        batch_size: 100
        gradient_clip: 0.5
    """

    label = "ORB"
    driver_task = "orb"
    required_config_keys = ("train_set", "run_name", "device", "seed")
    """The base's set as is. ``base_model`` is deliberately not on it: the architecture
    may arrive as the template's ``base_model`` *or* as the step's ``model_name`` (the
    driver's ``foundation`` fact), and only the hook can weigh the pair — a flat
    presence check would refuse a step that named the loader the second way."""
    missing_config_hint = (
        "the template must reference $TRAIN_SET and $RUN_NAME (device and seed arrive on "
        "the driver's own command line)"
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
        loader with ``train=True`` (``weights_path`` when the step's ``model_path`` names a local
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
        from orb_models.common.training.util import get_optim
        from orb_models.common.utils import seed_everything
        from orb_models.forcefield import pretrained
        from torch.utils.data import BatchSampler, DataLoader, RandomSampler

        # The step's device, with no fallback: orb's own `init_device()` takes `cuda:0`
        # whenever torch sees a GPU, unconditionally — the silent wrong-hardware failure
        # the preflight exists to prevent. A named `cuda` that is not there fails loudly
        # in `.to` rather than quietly running on CPU for days against a GPU booking.
        # The driver overlaid the value, so absence here is a broken invocation, not a
        # choice to fall back on.
        device = torch.device(str(config["device"]))
        seed_everything(int(config["seed"]))
        # The architecture: the template's `base_model`, or the step's own `model_name`
        # (the driver's `foundation` fact) — for orb the two words name the same thing, a
        # pretrained loader. Silently preferring either is how a declared model_name went
        # ignored while the sidecar recorded it, so a disagreement refuses by name.
        base_model = str(config.get("base_model") or "")
        foundation = str(config.get("foundation") or "")
        if base_model and foundation and base_model != foundation:
            raise SystemExit(
                f"orb training: the template names base_model {base_model!r} but the "
                f"step's model_name says {foundation!r} — keep one (the template row, or "
                f"the option)"
            )
        base_model = base_model or foundation
        if not base_model:
            raise SystemExit(
                "orb training: no architecture named — set base_model in the template "
                "(a loader from orb_models.forcefield.pretrained) or model_name on the step"
            )
        loader_fn = getattr(pretrained, base_model, None)
        if loader_fn is None:
            raise SystemExit(
                f"orb training: unknown base_model {base_model!r}; pick a loader from "
                f"orb_models.forcefield.pretrained"
            )
        weights = str(config.get("weights_path") or "")
        loader_kwargs: dict[str, Any] = {"device": device, "train": True}
        if weights:
            loader_kwargs["weights_path"] = weights
        try:
            loaded = loader_fn(**loader_kwargs)
        except TypeError as e:
            # The builder's guard, on the training side: an older loader without the
            # keyword must say so, not die as a naked TypeError in a submitted job.
            raise SystemExit(
                f"orb training: this orb-models version's {base_model!r} loader takes no "
                f"local weights_path, so model_path cannot be honoured; upgrade "
                f"orb-models or drop model_path to fine-tune the named release"
            ) from e
        model, atoms_adapter = _model_and_adapter(loaded)
        if atoms_adapter is None:
            raise SystemExit(
                f"orb training: {base_model!r} returned no atoms adapter on this "
                f"orb-models version, and fine-tuning needs one to batch the dataset — "
                f"upgrade orb-models"
            )

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
