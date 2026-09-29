"""ORB (Orbital Materials): the calculator chemrefine runs, and the trainer that fine-tunes it.

``model_name`` names a loader in :mod:`orb_models.forcefield.pretrained`
(e.g. ``orb_v3_conservative_inf_omat``, ``orb_v2``) — one of the functions orb-models
registers in its ``ORB_PRETRAINED_MODELS``, by its key there or its function name. From
the 0.6.2 floor on, every such loader but the deprecated v1 stubs (which only raise)
returns ``(model, atoms_adapter)`` and takes ``weights_path``, ``device`` and ``compile``.
Requires ``pip install orb-models``.

One module for one library, so the environment is declared once (:data:`ORB`) and both
capabilities hang off it. Training takes the CHGNet route — orb-models publishes its
fine-tune entry point as an unpackaged repo-root script — so the trainer's
:meth:`OrbTrainer.run_training` hook rebuilds that script's slim loop from the packaged
utilities, driven by the shared :mod:`chemrefine.engines.mlip.train.driver`.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, ClassVar

from chemrefine.engines.mlip.registry import CalculatorSpec, MlipLibrary
from chemrefine.engines.mlip.train.base import ApiTrainerBase, TrainingPlan
from chemrefine.errors import ConfigError
from chemrefine.state import Structure

logger = logging.getLogger(__name__)

ORB = MlipLibrary(extra="mlip-orb", package="orb-models", import_name="orb_models")
"""The one declaration of what provides this library."""

#: The loader an unset ``model_name`` runs and trains from — the conservative v3 model
#: trained on OMat24, the release ORB's own README leads with. ORB has no notion of a
#: default loader (every pretrained function is named), so the builder and the trainer
#: supply this one: the ``model_name`` contract is that unset means the library's own
#: default, and it must hold for every library.
_DEFAULT_MODEL = "orb_v3_conservative_inf_omat"


def _loader(pretrained: Any, name: str) -> Any:
    """The pretrained loader ``name`` names; a :class:`ConfigError` if none can run.

    Resolved through orb-models' own registry (``ORB_PRETRAINED_MODELS``) rather than as any
    attribute of the module: an architecture builder or a helper in the same module is not
    a loader, and called with a loader's arguments it fails as a ``TypeError`` that names
    neither the step nor the option. A name is a registry key, in orb's spelling
    (``orbmol-v1-direct``) or with underscores, or a loader's function name, which the two
    differ from where orb registers an alias (``orbmol-v1-direct`` loads
    ``orb_v3_direct_omol``) or names a function otherwise (``separate_d3_direct_3layer``).

    A ``ConfigError``, since the name came from the YAML and every failure a config can
    cause carries the exit code the CLI maps. A loader that cannot take ``compile`` is one
    of orb-models' deprecated v1 stubs, which only raise: refused by name, not left to the
    ``TypeError`` the call would raise.
    """
    loaders: dict[str, Any] = {}
    for key, fn in pretrained.ORB_PRETRAINED_MODELS.items():
        for spelling in (key, key.replace("-", "_"), fn.__name__):
            loaders.setdefault(spelling, fn)
    loader = loaders.get(name)
    if loader is None:
        raise ConfigError(
            f"unknown ORB model {name!r}; pick a loader from "
            f"orb_models.forcefield.pretrained (e.g. {_DEFAULT_MODEL!r})"
        )
    try:
        inspect.signature(loader).bind_partial(compile=False)
    except TypeError:
        raise ConfigError(
            f"ORB model {name!r} is one orb-models has deprecated: its loader only raises; "
            f"pick a current loader from orb_models.forcefield.pretrained "
            f"(e.g. {_DEFAULT_MODEL!r})"
        ) from None
    return loader


def _same_loader(pretrained: Any, name: str, other: str) -> bool:
    """Whether two names are one pretrained loader: by what each resolves to, else as written.

    orb registers a loader under its key, the key with underscores and the function's own name
    (:func:`_loader`), so two spellings can name one model. A name that resolves to no loader
    is compared as written, so the disagreement is reported before the unknown name is.
    """
    if name == other:
        return True
    try:
        return _loader(pretrained, name) is _loader(pretrained, other)
    except ConfigError:
        return False


@ORB.calculator("orb")
def _build_orb(spec: CalculatorSpec) -> Any:
    """ORB potential; ``model_name`` picks a loader from ``orb_models...pretrained``.

    The loader is the *architecture*, so it is selected by ``model_name`` even when the
    weights come from a file: a checkpoint carries weights, not an architecture, and the
    pretrained loaders take a ``weights_path`` (defaulting to the release URL, accepting
    a local file). Unset, the loader is :data:`_DEFAULT_MODEL` — with a ``model_path``
    too, so a checkpoint an ``mlip-train`` step produced runs without naming the
    architecture its own training defaulted to.

    The calculator takes the atoms adapter the loader returns beside the model, and the
    model is loaded uncompiled. orb compiles by default, and a compile is paid once per
    process, while every calculator ChemRefine builds serves one job in one process — a
    single point, an optimisation, an ExtOpt server. On CPU the compile was 70 s of an
    80 s single point of water.
    """
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    loader = _loader(pretrained, spec.model_name or _DEFAULT_MODEL)
    weights = {} if spec.weights is None else {"weights_path": str(spec.weights)}
    model, adapter = loader(**weights, device=spec.device, compile=False)
    return ORBCalculator(model, adapter, device=spec.device)


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
    the same import layout from the 0.6.2 floor through main). The schema — backend knobs
    only; device, seed and the foundation weights are plan facts and arrive on the
    driver's own command line, never through the template:

    .. code-block:: yaml

        train_set: $TRAIN_SET          # required — ASE sqlite db written by the trainer
        run_name:  $RUN_NAME           # required — names the fixed final checkpoint
        epochs: 50
        learning_rate: 3e-4            # the script's own defaults
        batch_size: 100
        gradient_clip: 0.5

    The architecture is a pretrained loader, and the step names it: its ``model_name``, or
    :data:`_DEFAULT_MODEL` unset, arriving as the driver's ``foundation`` fact. A template
    ``base_model:`` (the reference script's name for it) may repeat that loader, and one
    that names another is refused.

    A model that carries a stress output trains with it switched off, and the job warns:
    ChemRefine's labels are molecular (energy and forces, no cell), and orb's loss reads a
    stress target from every batch while the output is on.
    """

    label = "ORB"
    driver_task = "orb"
    default_foundation = _DEFAULT_MODEL
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
        # The architecture: the step's `model_name`, which the plan hands over as the
        # `foundation` fact, the default loader when the step names none. A template
        # `base_model` is the reference script's word for the same loader. Silently
        # preferring either is how a declared model_name went ignored while the sidecar
        # recorded it, so a template that names another loader is refused by name — another
        # loader, not another of orb's spellings of the same one.
        foundation = str(config.get("foundation") or "") or _DEFAULT_MODEL
        base_model = str(config.get("base_model") or "")
        if base_model and not _same_loader(pretrained, base_model, foundation):
            raise SystemExit(
                f"orb training: the template names base_model {base_model!r} but the step "
                f"starts from {foundation!r} (its model_name, or ORB's default when unset) "
                f"— name the loader once, as the step's model_name, and drop base_model "
                f"from the template"
            )
        try:
            loader_fn = _loader(pretrained, foundation)
        except ConfigError as e:
            raise SystemExit(f"orb training: {e}") from e
        weights = str(config.get("weights_path") or "")
        loader_kwargs: dict[str, Any] = {"device": device, "train": True}
        if weights:
            loader_kwargs["weights_path"] = weights
        # Uncompiled for the builder's reason: one fit is one process, and on CPU the
        # compile made a one-epoch fit take 67 s.
        model, atoms_adapter = loader_fn(**loader_kwargs, compile=False)
        if model.has_stress:
            # orb's loss reads `batch.system_targets["stress"]` from every batch while a
            # model's stress output is on, and ChemRefine's labels are molecular — energy
            # and forces off a DFT single point, no cell, no stress. Switched off, the
            # model fits the targets it was given, as orb-models itself provides for.
            model.disable_stress()
            logger.warning(
                "orb training: %r carries a stress output; ChemRefine's labels are "
                "molecular (energy and forces), so it trains with stress off",
                foundation,
            )

        # The targets the reference script trains on, by their orb spellings: energy is
        # a graph property, forces a node one. `instantiate_property_config(None)` is
        # the *feature* default (no targets at all), and a dataset built with it yields
        # batches whose first `model.loss` call dies on `system_targets["energy"]`.
        dataset = AseSqliteDataset(
            str(config["run_name"]),
            str(config["train_set"]),
            atoms_adapter=atoms_adapter,
            target_config=property_definitions.instantiate_property_config(
                {"graph": ["energy"], "node": ["forces"]}
            ),
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
