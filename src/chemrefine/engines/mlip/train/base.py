"""Backend-agnostic MLIP *training*: the split, the plan, and the config renderer.

Everything about a training run that does **not** depend on which library trains. What a
:class:`TrainerBase` subclass supplies is only what genuinely differs between libraries, and
each one lives in its library's own module under :mod:`chemrefine.engines.mlip.backends` —
beside that library's calculator, sharing its one environment declaration. Which library a
step selects is :mod:`chemrefine.engines.mlip.registry`.

Generic and never written twice: the train/valid/test split (:func:`split_structures`), the
placeholders every config may use (:func:`base_placeholders`), the rendering with its
validation (:func:`render_config`), and the calculator-labelled extxyz writer
(:func:`write_labelled_extxyz`) for the libraries that read labels off ``atoms.calc``.
Library-specific, and so a :class:`TrainerBase` hook:

===================  ==========================================================
``write_split``      MACE wants extxyz with ``REF_*`` keys; FAIRChem wants an
                     ASE db whose labels ride on a ``SinglePointCalculator``;
                     SevenNet and the CHGNet driver read the shared
                     calculator-labelled extxyz. There is no common file.
``command``          ``mace_run_train`` vs ``fairchem -c`` vs ``sevenn train``,
                     and each library's own multi-GPU idiom.
``artifact``         where the trained model lands.
===================  ==========================================================

A library whose training is a pure Python API — CHGNet has no CLI and no config format —
subclasses :class:`ApiTrainerBase` instead: ``command`` and ``artifact`` derive from its
declarations, and the one hook it adds, ``train_with_library(config)``, is what the shared
:mod:`~chemrefine.engines.mlip.train.driver` reaches through the registry in the *backend*
environment. The driver dispatches nominally — ``isinstance(trainer, ApiTrainerBase)`` — so
being drivable is a fact of inheritance, not of resemblance.

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
import logging
import shlex
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from chemrefine import ids
from chemrefine.errors import ConfigError
from chemrefine.quantities import HARTREE_TO_EV
from chemrefine.state import Structure

if TYPE_CHECKING:
    from ase import Atoms

logger = logging.getLogger(__name__)

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

    def items(self) -> tuple[tuple[str, tuple[Structure, ...]], ...]:
        """The three splits in writing order, under the names their files take.

        The triple every dataset writer iterates — owned here so no trainer spells the
        ``("train", …), ("valid", …), ("test", …)`` sequence for itself and drifts.
        """
        return (("train", self.train), ("valid", self.valid), ("test", self.test))


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

    def as_placeholders(self) -> dict[str, str]:
        """The ``TRAIN_SET`` / ``VALID_SET`` / ``TEST_SET`` triple, ``""`` for absent.

        The empty string rather than a missing key, so a template referencing a split
        that was not written renders an empty value its library can refuse by name —
        instead of a ``$VALID_SET`` surviving substitution as a literal."""
        return {
            "TRAIN_SET": str(self.train),
            "VALID_SET": str(self.valid) if self.valid else "",
            "TEST_SET": str(self.test) if self.test else "",
        }


@dataclass(frozen=True)
class TrainingPlan:
    """*Where and how* one training run executes — resolved once, before anything is written.

    A single value threaded through the :class:`TrainerBase` hooks, so ``command`` and
    ``placeholders`` cannot end up describing different runs: a config naming one directory
    and an argv naming another produces a step that trains correctly and is then reported as
    having produced nothing.

    Deliberately **not** the data. The split is passed to :meth:`TrainerBase.write_dataset`
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

    weights: Path | None
    """A local checkpoint to fine-tune from; ``None`` means no file was named.

    Two typed fields rather than the folded ``start_from`` string they replace, mirroring
    :class:`~chemrefine.engines.mlip.registry.CalculatorSpec` — the same subsystem's
    statement that a name and a path are different facts a consumer must not have to
    guess apart. Folded, CHGNet's hook sent a release *name* through ``from_file`` (a
    path-only door) while its own builder twenty lines up dispatches the two correctly."""

    foundation: str | None
    """A released foundation model to start from, in the library's own spelling; ``None``
    with :attr:`weights` also ``None`` means training from scratch."""

    @property
    def start_from(self) -> str | None:
        """The one-string display of what training starts from; ``None`` = from scratch.

        For the consumers that genuinely want one string — ``$FOUNDATION_MODEL`` for the
        CLI-trainer templates (MACE and SevenNet take either spelling in the same config
        key), and the runlog/sidecar rows. Never for dispatch: a hook that needs to *act*
        on the value reads the typed fields.
        """
        return str(self.weights) if self.weights is not None else (self.foundation or None)

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

    @property
    def is_neutral_singlet(self) -> bool:
        """Whether this run's species is the neutral singlet every chargeless library assumes.

        The predicate behind the charge/spin warning, owned once, so no trainer spells
        ``charge != 0 or multiplicity != 1`` for itself and drifts in how it describes the
        consequence.
        """
        return self.charge == 0 and self.multiplicity == 1


class TrainerBase(ABC):
    """The trainer contract: declarations up top, hooks abstract, machinery concrete.

    A backend declares what the parent needs — a label for messages, whether its library
    demands a validation set (and why), whether its dataset carries charge and spin, the
    placeholders its template must reference, what to copy home — and writes only the
    three hooks that are genuinely a library fact: how one split is written
    (:meth:`write_split`), what command trains (:meth:`command`), and where the product
    lands (:meth:`artifact`). Everything a trainer would otherwise write for itself — the
    validation refusal, the charge/spin warning, the split loop, the placeholder triple,
    the launcher quoting — is concrete here, written once.

    An ``ABC``, not a ``Protocol``: a registry — an explicit declaration channel —
    already exists here, so a missing abstract hook fails at instantiation naming the
    member and :meth:`MlipLibrary.trainer` verifies the declarations at import, which is
    earlier and more specific than any structural check at the point of use. Protocols
    stay the tool across package boundaries, where no shared base can exist (the
    :mod:`chemrefine.engines.api` doctrine, unchanged).

    A class rather than the bare function a *calculator* backend registers, because
    these decisions are not independent: the dataset format implies what the
    placeholders mean, which implies the argv, which implies where the product lands.
    Grouped in one class, a combination that does not agree with itself cannot be
    written by accident.
    """

    label: ClassVar[str]
    """The library's name as messages spell it — ``"SevenNet"``, ``"CHGNet"``."""

    needs_validation: ClassVar[bool] = False
    """Whether this library refuses to train without a validation set."""

    validation_reason: ClassVar[str] = ""
    """The why-clause of that refusal — the one library fact in it. No trailing period."""

    charge_spin_aware: ClassVar[bool] = False
    """True when the dataset carries charge and spin; False warns on non-neutral data."""

    required_placeholders: ClassVar[frozenset[str]] = frozenset({"TRAIN_SET"})
    """Placeholders this backend's template must reference, or the step is misconfigured.

    Checked before submission (:func:`render_config`). A config that never names the
    dataset chemrefine just wrote is not a config that trains on it — most libraries
    would either read some previous run's data or fail an hour later with a message
    about a path the user never typed."""

    output_globs: ClassVar[tuple[str, ...]]
    """Loose files to copy back out of the job's scratch directory."""

    output_dirs: ClassVar[tuple[str, ...]] = ()
    """Whole directories to copy back; most libraries write flat files."""

    # -- the hooks: what genuinely differs per library ---------------------------------

    @abstractmethod
    def write_split(self, plan: TrainingPlan, name: str, structures: tuple[Structure, ...]) -> Path:
        """Write one non-empty split (``"train"``/``"valid"``/``"test"``) in this
        library's format; return the file its template will name.

        Runs in the **orchestrator's** process, not the backend's, so only chemrefine's
        own dependencies — ``ase`` and ``numpy`` — are importable. That is a real
        constraint and a deliberate one: it keeps every trainer module importable with no
        MLIP library installed, which is what lets the registry be built at import time.
        A format that genuinely needs the backend (SevenNet's graph pre-build, AIMNet2's
        HDF5 packer) is expressed as an extra line in :meth:`command`, where the library
        is importable.
        """

    @abstractmethod
    def command(self, plan: TrainingPlan, config: Path) -> str:
        """The bash that runs the training inside ``$WORK_DIR``.

        One command, no ``trap`` and no teardown: the job script owns the single exit
        handler that copies results back, and an engine that emits its own would replace
        it.
        """

    @abstractmethod
    def artifact(self, run_dir: Path, run_name: str) -> Path:
        """The trained model — the file whose existence is this step's success test.

        Takes the run's directory and name rather than a whole :class:`TrainingPlan`,
        because it must be answerable *without* one: ``rebuild-cache`` adopts a product
        having submitted nothing, and ``chemrefine.step`` asks after a failure, when
        there is no dataset to split and no launcher to resolve. Both of those are the
        moments the answer matters most.
        """

    # -- the machinery: written once, driven by the declarations -----------------------

    def write_dataset(self, plan: TrainingPlan, split: DatasetSplit) -> DatasetFiles:
        """Refuse, warn, and write each non-empty split through :meth:`write_split`.

        The refusal fires only when :attr:`needs_validation` says the library cannot
        train without one, and says why in that library's own words
        (:attr:`validation_reason`). The warning fires for non-neutral data only when
        the dataset format carries no charge/spin channel — silence is the only wrong
        answer, and it is not per-library prose any more. An empty split gets no file at
        all: ase refuses a zero-byte extxyz, so naming one would turn "no test set" into
        a crash.
        """
        if self.needs_validation and not split.valid:
            raise ConfigError(
                f"{self.label} training needs a validation set — "
                f"{self.validation_reason}. Raise `valid_fraction` above 0."
            )
        if not self.charge_spin_aware and not plan.is_neutral_singlet:
            logger.warning(
                "%s has no charge/spin channel: charge %d, multiplicity %d will be "
                "fitted as if neutral singlet — the labels carry no trace of either",
                self.label,
                plan.charge,
                plan.multiplicity,
            )
        plan.run_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path | None] = {"train": None, "valid": None, "test": None}
        for name, structures in split.items():
            if not structures:
                continue
            written[name] = self.write_split(plan, name, structures)
        assert written["train"] is not None  # noqa: S101 - split_structures guarantees one
        return DatasetFiles(train=written["train"], valid=written["valid"], test=written["test"])

    def placeholders(self, plan: TrainingPlan, data: DatasetFiles) -> dict[str, str]:
        """The dataset triple, by default — a backend extends or overrides it."""
        return data.as_placeholders()

    # -- command helpers: the one spelling of each ------------------------------------

    def quoted_launcher(self, plan: TrainingPlan) -> str:
        """The backend interpreter, quoted for bash — a real path, unlike the basename."""
        return shlex.quote(str(plan.launcher))

    def console_script(self, plan: TrainingPlan, name: str) -> str:
        """A console script from the backend env's own ``bin/``, quoted for bash."""
        return shlex.quote(str(plan.bindir / name))

    def torchrun(self, plan: TrainingPlan, *argv: str) -> str:
        """The DDP launcher line, spelled once: ``python -m torch.distributed.run …``.

        Module form under the backend's interpreter rather than a ``torchrun`` script,
        for the reason every command here uses the launcher: the script is on nobody's
        ``PATH`` once the library lives somewhere the orchestrator does not.
        """
        return (
            f"{self.quoted_launcher(plan)} -m torch.distributed.run --standalone "
            f"--nnodes 1 --nproc_per_node {plan.gpus} " + " ".join(argv)
        )


class ApiTrainerBase(TrainerBase):
    """A trainer driven through the shared train driver — for libraries with no CLI.

    CHGNet and ORB train through a pure Python API: no console script, no native config
    format. Their route is the shared :mod:`~chemrefine.engines.mlip.train.driver`,
    running under the backend environment's interpreter (a managed env is a
    ``pip install "chemrefine[<extra>]"``, so chemrefine is importable there), which
    re-resolves the same class through the registry and calls :meth:`run_training`.

    Being *this kind* of trainer is a fact of inheritance, not of resemblance: the driver
    dispatches on ``isinstance(trainer, ApiTrainerBase)``. What such a trainer writes is
    only :meth:`write_split` and :meth:`train_with_library` — the driver line, the
    config-key check and the artifact path all derive from the declarations, so the
    artifact's filename cannot drift from the name the backend-side save uses: both read
    :attr:`artifact_filename`.
    """

    driver_task: ClassVar[str]
    """The word the driver re-resolves in the backend env — the library's task key."""

    required_config_keys: ClassVar[tuple[str, ...]] = ("train_set", "run_name", "device", "seed")
    """Config keys :meth:`run_training` refuses to proceed without (present and non-null).

    ``device`` and ``seed`` are here because the driver **guarantees** them: they are plan
    facts, and plan facts ride the driver's own command line rather than the user's
    template (:meth:`command`). Read from the template with a silent per-backend fallback,
    each one was lost whenever the template did not happen to reference its placeholder —
    a ``device: cuda`` step trained on CPU with the GPU booked, a declared foundation
    model trained from scratch while the sidecar recorded it, a ``seed`` reached the
    split but not torch. The orchestrator charges, keys the cache and logs by the
    declared values; the hook must run by the same ones."""

    missing_config_hint: ClassVar[str]
    """One sentence naming the fix when a required key is missing — which placeholders
    the template must reference. No trailing period."""

    artifact_filename: ClassVar[str]
    """The fixed-name final save, as a ``{run_name}`` format — ``"{run_name}.pth.tar"``.

    The **one** source both :meth:`artifact` and the backend-side hook's save read.
    Fixed deliberately: the libraries' own per-epoch checkpoints embed the epoch (and
    CHGNet's the error) in their names, which no later step could name before the run.
    """

    def command(self, plan: TrainingPlan, config: Path) -> str:
        """The shared train driver, under the backend env's interpreter.

        ``-m chemrefine.engines.mlip.train.driver <task> <config> --device … --seed …`` —
        the driver is chemrefine's, present in the managed env because the env is a
        ``chemrefine[<extra>]`` install. The config rides by basename for the
        array-sentinel reason MACE's command spells out. Single-process on any device:
        DDP is not a mode these libraries' APIs document.

        **The plan facts ride this argv, not the template.** chemrefine owns both ends of
        this channel — this method writes the command, the driver parses it — so device,
        seed and the foundation weights go straight across as authoritative values
        (:func:`~chemrefine.engines.mlip.train.driver.main` overlays them onto the
        rendered config and refuses a template value that disagrees). Routed through the
        user's template as ``$DEVICE``-style placeholders, each fact was silently lost the
        moment a template did not reference it, while the budget, the cache key and the
        sidecar all asserted the declared value. A template channel is the CLI trainers'
        necessity — their libraries read the config file themselves — not this route's.

        ``shlex.quote`` on the two free-text values, the same protection every other
        engine-emitted command applies to a value it did not mint (``orca_command``, the
        ``console_script`` helper above): ``device`` is a validated Literal and ``seed``
        an int, but a checkpoint path and a model name are the user's own text.
        """
        python = self.quoted_launcher(plan)
        parts = [
            f"{python} -m chemrefine.engines.mlip.train.driver {self.driver_task} {config.name}",
            f"--device {plan.device} --seed {plan.seed}",
        ]
        if plan.weights is not None:
            parts.append(f"--weights-path {shlex.quote(str(plan.weights))}")
        if plan.foundation:
            parts.append(f"--foundation {shlex.quote(plan.foundation)}")
        return " ".join(parts)

    def artifact(self, run_dir: Path, run_name: str) -> Path:
        """:attr:`artifact_filename` under the run directory — derived, never restated."""
        return run_dir / self.artifact_filename.format(run_name=run_name)

    def run_training(self, config: dict[str, Any]) -> int:
        """Check the declared keys, then hand the config to the library. The driver's hook.

        Present-and-non-null, not truthy: ``seed: 0`` is an ordinary value a truthiness
        test would refuse as missing — the same required-means-non-null semantics the
        script engines' output boundary holds.
        """
        missing = [key for key in self.required_config_keys if config.get(key) is None]
        if missing:
            raise SystemExit(
                f"{self.driver_task} training config is missing {missing} — "
                f"{self.missing_config_hint}"
            )
        return self.train_with_library(config)

    @abstractmethod
    def train_with_library(self, config: dict[str, Any]) -> int:
        """Train with the library's own API — runs in the *backend* environment.

        The heavy imports live here and nowhere the orchestrator reaches; the final
        model must be saved under :meth:`artifact`'s name (format
        :attr:`artifact_filename` with the config's ``run_name``).
        """


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
    :meth:`TrainerBase.placeholders` is merged **over** these, so it can specialise one.
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


def placeholders_for(
    trainer: TrainerBase, plan: TrainingPlan, data: DatasetFiles
) -> dict[str, str]:
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


def labelled_atoms(struct: Structure) -> Atoms:
    """One :class:`~ase.Atoms` whose labels ride on a ``SinglePointCalculator``.

    The *plain* labelled-extxyz form — ase's writer emits the standard ``energy=`` /
    ``free_energy=`` comment fields plus a ``forces`` column, and its reader reconstructs
    the calculator — shared by the trainers whose libraries read labels off ``atoms.calc``
    (SevenNet's loader, and the CHGNet driver's own conversion). MACE deliberately does
    not use it: its dataset speaks ``REF_*`` info keys, written in its own module.
    ``free_energy`` is set alongside ``energy`` because SevenNet asks
    ``get_potential_energy(force_consistent=True)`` first.

    Energies are converted Hartree → eV; forces are already eV/Å. The copy keeps the
    pipeline's own structure untouched — its force arrays are deliberately read-only, and
    an attached calculator would otherwise ride the shared reference.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms: Atoms = struct.atoms.copy()
    if struct.energy_hartree is None or struct.forces_ev_per_a is None:
        raise ConfigError(
            f"structure {struct.id} is missing an energy or forces — "
            f"split_structures should have refused it before any writer ran"
        )
    energy_ev = struct.energy_hartree * HARTREE_TO_EV
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=energy_ev,
        free_energy=energy_ev,
        forces=np.asarray(struct.forces_ev_per_a, dtype=float),
    )
    return atoms


def write_labelled_extxyz(path: Path, structures: Sequence[Structure]) -> Path:
    """Write ``structures`` as calculator-labelled extxyz frames at ``path``."""
    from ase.io import write as ase_write

    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(path), [labelled_atoms(s) for s in structures], format="extxyz")
    return path


def _fingerprint_sha1() -> hashlib._Hash:
    """A SHA-1 marked as a content fingerprint — the constructor form for a streamed hash.

    Named rather than written inline at the one call site, because it is the same concept
    :func:`chemrefine.cache._fingerprint_sha1` spells for the cache keys, and one of the two
    reading as a named idea while the other reads as an anonymous lambda is how the next
    person comes to think they differ.

    Deliberately *not* imported from there: :mod:`chemrefine.engines` must not import
    :mod:`chemrefine.cache` — a boundary :meth:`chemrefine.cache.StepKey.of` documents and
    ``tests/test_engines_invariants.py`` enforces. Each module names the concept; three
    duplicated lines are the price of a separation worth more than they cost.
    """
    return hashlib.sha1(usedforsecurity=False)


def digest_of(path: Path) -> str:
    """A short content digest of a produced model, for the run's sidecar record.

    Streamed rather than ``read_bytes()``: a UMA inference checkpoint is 1-2 GB, and this runs
    in the driver process — which on a cluster is a login node with a memory cap that a
    fine-tuned foundation model can genuinely exceed.

    The constructor is passed rather than the name ``"sha1"`` so the hash can be marked as a
    content fingerprint. Handed a name, :func:`hashlib.file_digest` builds the object through
    ``hashlib.new(...)`` with ``usedforsecurity`` at its default, and a host whose crypto
    policy forbids SHA-1 *as a digest* then refuses to make one at all — which here would
    fail a training step for recording what it had already produced. The value is unchanged
    either way; the flag is a policy hint, not an input.
    """
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, _fingerprint_sha1)
    return digest.hexdigest()[:16]
