"""Backend-agnostic MLIP *training*: the split, the plan, and the config renderer.

Everything about a training run that does **not** depend on which library trains. What a
:class:`Trainer` supplies is only what genuinely differs between libraries, and each one lives
in its library's own module under :mod:`chemrefine.engines.mlip.backends` — beside that
library's calculator, sharing its one environment declaration. Which library a step selects is
:mod:`chemrefine.engines.mlip.registry`.

Generic and never written twice: the train/valid/test split (:func:`split_structures`), the
placeholders every config may use (:func:`base_placeholders`), and the rendering with its
validation (:func:`render_config`). Library-specific, and so a :class:`Trainer` method:

===================  ==========================================================
``write_dataset``    MACE wants extxyz with ``REF_*`` keys; FAIRChem wants an
                     ASE db whose labels ride on a ``SinglePointCalculator``;
                     AIMNet2 wants HDF5. There is no common file.
``placeholders``     what its config calls those files.
``command``          ``mace_run_train`` vs ``fairchem -c`` vs ``aimnet train``,
                     and each library's own multi-GPU idiom.
``artifact``         where the trained model lands.
===================  ==========================================================

Rendering, not patching
-----------------------

A trainer config is **rendered** through :class:`string.Template`, never parsed and edited.
The templates are the libraries' own native configs with ``$TRAIN_SET``-style placeholders
where paths go, and chemrefine only substitutes text.

This is not a stylistic choice. The previous trainer loaded the YAML and inserted
``train_file`` / ``test_file`` / ``log_dir`` / ``checkpoints_dir`` / ``results_dir`` at the top
level, which is meaningful to MACE and *rejected outright* by FAIRChem — its config allows
only ``job`` / ``runner`` / ``reducer`` there and raises ``Found unused keys in the config``
for anything else. No schema-aware patcher can serve both. Substitution serves any of them,
including formats nobody has written a trainer for yet, and it leaves unknown ``$NAME``
untouched so FAIRChem's own ``${…}`` interpolations survive it (:func:`render_config`).

The price is that a template must *say* where its dataset goes, and the price is paid
up front: each trainer declares ``required_placeholders`` and a template that references none
of them fails in ``prepare``, before a job is submitted, naming the placeholder to add.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from chemrefine import ids
from chemrefine.errors import ConfigError
from chemrefine.state import Structure

# ---------------------------------------------------------------------------
# What a training run is
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSplit:
    """One deterministic train / validation / test partition of a step's structures.

    Split **here, once**, rather than by each library's own knob. MACE's ``--valid_fraction``
    and FAIRChem's separate dataset stanzas would otherwise partition the same data
    differently, so a run's reported errors would depend on which library trained it — and
    ``resume`` would not even reproduce its own split. Doing it once also means the split is
    covered by the step's cache key, because it is a pure function of the parents and the
    seed, both of which the fingerprint already sees.
    """

    train: tuple[Structure, ...]
    valid: tuple[Structure, ...]
    test: tuple[Structure, ...]

    def counts(self) -> str:
        """``"45 train / 5 valid / 0 test"`` — for log lines and error messages."""
        return f"{len(self.train)} train / {len(self.valid)} valid / {len(self.test)} test"


@dataclass(frozen=True)
class DatasetFiles:
    """The files a trainer wrote, as its config's placeholders will name them.

    ``valid`` and ``test`` are ``None`` when that split is empty — a distinction the template
    needs, because pointing a library at an empty dataset file is not the same as not naming
    one, and several of them refuse to read one.
    """

    train: Path
    valid: Path | None = None
    test: Path | None = None


@dataclass(frozen=True)
class TrainingPlan:
    """*Where and how* one training run executes — resolved once, before anything is written.

    A single value threaded through the :class:`Trainer` methods, so ``command`` and
    ``placeholders`` cannot end up describing different runs: a config naming one directory
    and an argv naming another produces a step that trains correctly and is then reported as
    having produced nothing.

    Deliberately **not** the data. The split is passed to :meth:`Trainer.write_dataset`
    separately because it is the only thing here that depends on the step's structures, and
    folding it in would mean a trainer could not say what command it runs without first being
    handed an ensemble — which is exactly what the scheduler asks it, before there is one.
    """

    run_dir: Path
    """Where the run's files live — the config, the dataset, the logs, the product."""

    run_name: str
    """The run's name inside the backend, which several of them build filenames from."""

    device: str
    gpus: int
    cores: int
    seed: int
    charge: int
    multiplicity: int

    start_from: str | None
    """The foundation model or checkpoint to fine-tune; ``None`` trains from scratch."""

    launcher: Path
    """The interpreter of the backend's environment, from ``_provision.launcher_for``."""

    @property
    def bindir(self) -> Path:
        """The backend environment's ``bin/``, where its console scripts live.

        Derived from the launcher rather than resolved separately, so it is right in all three
        cases the provisioner can return — a managed env, an explicit ``backend_python``, or
        this interpreter when the backend is importable alongside the orchestrator.
        """
        return self.launcher.parent


@runtime_checkable
class Trainer(Protocol):
    """What one MLIP library needs in order to be trained by chemrefine.

    Four methods and three declarations, and that is the whole of it. A class rather than the
    bare function a *calculator* backend registers, because these decisions are not
    independent: the dataset format implies what the placeholders mean, which implies the
    argv, which implies where the product lands. Grouped in one class, a combination that does
    not agree with itself cannot be written by accident.
    """

    required_placeholders: ClassVar[frozenset[str]]
    """Placeholders this backend's config must reference, or the step is misconfigured.

    Checked before submission (:func:`render_config`). A config that never names the dataset
    chemrefine just wrote is not a config that trains on it — most libraries would either read
    some previous run's data or fail an hour later with a message about a path the user never
    typed."""

    output_globs: ClassVar[tuple[str, ...]]
    """Loose files to copy back out of the job's scratch directory."""

    output_dirs: ClassVar[tuple[str, ...]]
    """Whole directories to copy back (``checkpoints``, ``logs``, …)."""

    def write_dataset(self, plan: TrainingPlan, split: DatasetSplit) -> DatasetFiles:
        """Write the split to disk in this backend's format; return the paths.

        Runs in the **orchestrator's** process, not the backend's, so it may use only
        chemrefine's own dependencies — ``ase`` and ``numpy``. That is a real constraint and a
        deliberate one: it keeps every trainer module importable with no MLIP library
        installed, which is what lets the registry be built at import time. A format that
        genuinely needs the backend (SevenNet's graph pre-build, AIMNet2's HDF5 packer) is
        expressed as an extra line in :meth:`command`, where the library is importable.
        """
        ...

    def placeholders(self, plan: TrainingPlan, data: DatasetFiles) -> dict[str, str]:
        """This backend's ``$VAR`` substitutions, merged over :func:`base_placeholders`."""
        ...

    def command(self, plan: TrainingPlan, config: Path) -> str:
        """The bash that runs the training inside ``$WORK_DIR``.

        One command, no ``trap`` and no teardown: the job script owns the single exit handler
        that copies results back, and an engine that emits its own would replace it.
        """
        ...

    def artifact(self, run_dir: Path, run_name: str) -> Path:
        """The trained model — the file whose existence is this step's success test.

        Takes the run's directory and name rather than a whole :class:`TrainingPlan`, because
        it must be answerable *without* one: ``rebuild-cache`` adopts a product having
        submitted nothing, and ``chemrefine.step`` asks after a failure, when there is no
        dataset to split and no launcher to resolve. Both of those are the moments the answer
        matters most.
        """
        ...


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def split_structures(
    structures: Sequence[Structure],
    *,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> DatasetSplit:
    """Partition labelled structures into train / validation / test, reproducibly.

    Every structure must carry both an energy and forces: an MLIP is fitted to both, and a
    structure missing either contributes nothing while silently shrinking the set. Raised as
    a :class:`~chemrefine.errors.ConfigError` because the cause is upstream configuration —
    a training step pointed at a step that computed no gradients — and it names the structure
    so the user can find it.

    The permutation is seeded, so a re-run partitions identically and the numbers a step
    reports are comparable across runs. ``valid_fraction`` is rounded **down** but floored at
    one structure whenever it is non-zero: asking for validation and silently receiving none
    is worse than a small set, and a library given an empty validation file usually fails to
    read it rather than proceeding without.
    """
    for struct in structures:
        if struct.energy_hartree is None:
            raise ConfigError(
                f"structure {struct.id} has no energy, so it cannot be trained on. "
                f"An MLIP is fitted to energies and forces — check that the step feeding "
                f"this one computes both."
            )
        if struct.forces_ev_per_a is None:
            raise ConfigError(
                f"structure {struct.id} has no forces, so it cannot be trained on. "
                f"An MLIP is fitted to energies and forces — check that the step feeding "
                f"this one computes gradients (ORCA: `! EnGrad`, or an optimisation)."
            )

    n = len(structures)
    if n == 0:
        raise ConfigError("no structures to train on — the previous step produced none")

    # ``int(np.floor(...))``, not a bare cast: these index the permutation below, so they
    # have to be ints, and truncation would round a negative fraction the wrong way.
    n_test = int(np.floor(test_fraction * n))
    n_valid = int(np.floor(valid_fraction * n))
    if valid_fraction > 0:
        n_valid = max(1, n_valid)
    n_train = n - n_valid - n_test
    if n_train < 1:
        raise ConfigError(
            f"a {valid_fraction:g} validation / {test_fraction:g} test split of {n} "
            f"structure(s) leaves {n_train} to train on. Lower the fractions, or train on "
            f"more structures."
        )

    order = np.random.default_rng(seed).permutation(n)
    pick = [structures[i] for i in order]
    return DatasetSplit(
        train=tuple(pick[:n_train]),
        valid=tuple(pick[n_train : n_train + n_valid]),
        test=tuple(pick[n_train + n_valid :]),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def base_placeholders(plan: TrainingPlan) -> dict[str, str]:
    """The substitutions every trainer config may use, whatever library it drives.

    Provided centrally so a template reads the same across backends: ``$RUN_DIR`` means the
    run's directory whether the trainer is MACE or FAIRChem, and a user moving between them
    re-learns only what genuinely differs. A trainer's own
    :meth:`Trainer.placeholders` is merged **over** these, so it can specialise one.
    """
    return {
        "RUN_DIR": str(plan.run_dir),
        "RUN_NAME": plan.run_name,
        "DEVICE": plan.device,
        "SEED": str(plan.seed),
        "NGPUS": str(plan.gpus),
        "CORES": str(plan.cores),
        "CHARGE": str(plan.charge),
        "MULTIPLICITY": str(plan.multiplicity),
        "FOUNDATION_MODEL": plan.start_from or "",
    }


def placeholders_for(trainer: Trainer, plan: TrainingPlan, data: DatasetFiles) -> dict[str, str]:
    """:func:`base_placeholders` with the trainer's own merged over them.

    One function so the precedence is decided once. A trainer overriding a shared name is
    legitimate — it knows its library — but two call sites merging in opposite orders would
    make which one wins depend on where the render happened.
    """
    return {**base_placeholders(plan), **trainer.placeholders(plan, data)}


def render_config(
    template: Path | None,
    placeholders: Mapping[str, str],
    *,
    required: frozenset[str],
    dest: Path,
) -> Path:
    """Substitute ``placeholders`` into the trainer template; write it to ``dest``.

    ``safe_substitute`` rather than ``substitute``: a placeholder chemrefine does not supply
    must survive untouched, because FAIRChem's configs are full of OmegaConf ``${…}``
    interpolations that its own loader resolves later. Substituting strictly would fail on
    every one of them.

    A template that references none of ``required`` is refused **here**, before the step
    submits anything, rather than an hour later inside a library complaining about a path the
    user never wrote. The message names the placeholder to add.
    """
    source = ids.require_template(template, label="MLIP training")
    text = source.read_text(encoding="utf-8")
    missing = required - set(Template(text).get_identifiers())
    if missing:
        wanted = ", ".join(f"${name}" for name in sorted(missing))
        raise ConfigError(
            f"MLIP training template {source} never references {wanted}, so the rendered "
            f"config would not point at the dataset chemrefine writes for it. Add it where "
            f"your trainer expects the dataset — e.g. `train_file: $TRAIN_SET`."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(Template(text).safe_substitute(placeholders), encoding="utf-8")
    return dest


def digest_of(path: Path) -> str:
    """A short content digest of a produced model, for the run's sidecar record.

    Streamed rather than ``read_bytes()``: a UMA inference checkpoint is 1-2 GB, and this runs
    in the driver process — which on a cluster is a login node with a memory cap that a
    fine-tuned foundation model can genuinely exceed.
    """
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha1").hexdigest()[:16]
