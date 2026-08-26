"""Tests for the backend-agnostic half of MLIP training (``engines/mlip/training.py``).

The split, the config renderer and the trainer registry — everything that is the same
whichever library trains. Per-library behaviour is ``test_engines_mlip_trainers.py``; the
engine that drives both is ``test_engines_mlip_train_engine.py``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine.engines.mlip.registry import (
    backend_spec,
    registered_backends,
    registered_extras,
    registered_trainers,
    requirement_from_options,
    trainer_for,
)
from chemrefine.engines.mlip.train import base as training
from chemrefine.engines.mlip.train.base import (
    DatasetFiles,
    DatasetSplit,
    TrainingPlan,
    base_placeholders,
    digest_of,
    placeholders_for,
    render_config,
    split_structures,
)
from chemrefine.errors import ConfigError
from chemrefine.state import Structure


def _labelled(sid: str, *, energy: float = -1.0) -> Structure:
    """A structure carrying both the labels an MLIP is fitted to."""
    return Structure(
        id=sid,
        atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
        energy_hartree=energy,
        forces_ev_per_a=np.zeros((2, 3)),
    )


def _plan(tmp_path: Path, **overrides: object) -> TrainingPlan:
    base: dict[str, object] = {
        "run_dir": tmp_path / "train",
        "run_name": "train",
        "device": "cpu",
        "gpus": 0,
        "cores": 4,
        "seed": 42,
        "charge": 0,
        "multiplicity": 1,
        "start_from": None,
        "launcher": Path("/envs/mlip-mace/bin/python"),
    }
    base.update(overrides)
    return TrainingPlan(**base)


# ---------------------------------------------------------------------------
# split_structures
# ---------------------------------------------------------------------------


def test_the_split_partitions_every_structure_exactly_once():
    structures = [_labelled(str(i), energy=-1.0 - i) for i in range(20)]
    split = split_structures(structures, valid_fraction=0.1, test_fraction=0.1, seed=42)

    assert (len(split.train), len(split.valid), len(split.test)) == (16, 2, 2)
    ids = [s.id for s in (*split.train, *split.valid, *split.test)]
    assert sorted(ids, key=int) == [s.id for s in structures], "no structure lost or duplicated"


def test_the_split_is_reproducible_from_its_seed():
    """A re-run must partition identically, or a step's reported errors are not comparable."""
    structures = [_labelled(str(i)) for i in range(20)]
    first = split_structures(structures, valid_fraction=0.2, test_fraction=0.0, seed=7)
    again = split_structures(structures, valid_fraction=0.2, test_fraction=0.0, seed=7)
    other = split_structures(structures, valid_fraction=0.2, test_fraction=0.0, seed=8)

    assert [s.id for s in first.train] == [s.id for s in again.train]
    assert [s.id for s in first.train] != [s.id for s in other.train]


def test_asking_for_validation_always_yields_some():
    """A fraction that rounds to zero must not silently produce no validation set.

    Several libraries fail to read an empty validation file rather than training without one,
    so rounding down to nothing turns a small dataset into a crash an hour in.
    """
    split = split_structures(
        [_labelled(str(i)) for i in range(5)], valid_fraction=0.1, test_fraction=0.0, seed=42
    )
    assert len(split.valid) == 1
    assert len(split.train) == 4


def test_no_validation_is_expressible():
    """`valid_fraction: 0` means none — distinct from a fraction that rounds to none."""
    split = split_structures(
        [_labelled(str(i)) for i in range(5)], valid_fraction=0.0, test_fraction=0.0, seed=42
    )
    assert split.valid == ()
    assert len(split.train) == 5


def test_a_structure_without_energy_is_refused_by_name():
    bad = Structure(id="7", atoms=Atoms("H"), energy_hartree=None, forces_ev_per_a=np.zeros((1, 3)))
    with pytest.raises(ConfigError, match="structure 7 has no energy"):
        split_structures([bad], valid_fraction=0.1, test_fraction=0.0, seed=1)


def test_a_structure_without_forces_is_refused_by_name():
    """The likeliest real misconfiguration: a training step fed by a step computing no gradients."""
    bad = Structure(id="3", atoms=Atoms("H"), energy_hartree=-1.0, forces_ev_per_a=None)
    with pytest.raises(ConfigError, match="structure 3 has no forces"):
        split_structures([bad], valid_fraction=0.1, test_fraction=0.0, seed=1)


def test_an_empty_ensemble_is_refused():
    with pytest.raises(ConfigError, match="no structures to train on"):
        split_structures([], valid_fraction=0.1, test_fraction=0.0, seed=1)


def test_a_split_that_leaves_nothing_to_train_on_is_refused_with_its_counts():
    """The case only the structure count can decide, so the field bounds cannot catch it."""
    with pytest.raises(ConfigError, match="leaves 0 to train on"):
        split_structures(
            [_labelled("0"), _labelled("1")], valid_fraction=0.5, test_fraction=0.5, seed=1
        )


def test_counts_reads_as_a_sentence():
    split = split_structures(
        [_labelled(str(i)) for i in range(10)], valid_fraction=0.1, test_fraction=0.0, seed=1
    )
    assert split.counts() == "9 train / 1 valid / 0 test"


# ---------------------------------------------------------------------------
# render_config
# ---------------------------------------------------------------------------


def test_rendering_substitutes_the_placeholders(tmp_path: Path):
    template = tmp_path / "step1.yaml"
    template.write_text("train_file: $TRAIN_SET\nname: $RUN_NAME\n", encoding="utf-8")
    dest = tmp_path / "out" / "rendered.yaml"

    render_config(
        template,
        {"TRAIN_SET": "/data/train.xyz", "RUN_NAME": "train"},
        required=frozenset({"TRAIN_SET"}),
        dest=dest,
    )

    assert dest.read_text() == "train_file: /data/train.xyz\nname: train\n"


def test_rendering_leaves_a_placeholder_it_does_not_supply(tmp_path: Path):
    """FAIRChem configs are full of OmegaConf `${…}` its own loader resolves later.

    Substituting strictly would fail on every one of them, so an unknown name survives
    untouched rather than becoming an error or an empty string.
    """
    template = tmp_path / "step1.yaml"
    template.write_text("src: $TRAIN_SET\nlr: ${data.lr}\nbatch: $UNKNOWN\n", encoding="utf-8")
    dest = tmp_path / "rendered.yaml"

    render_config(template, {"TRAIN_SET": "/d/t.db"}, required=frozenset({"TRAIN_SET"}), dest=dest)

    text = dest.read_text()
    assert "${data.lr}" in text
    assert "$UNKNOWN" in text


def test_a_template_that_never_names_the_dataset_is_refused_before_submitting(tmp_path: Path):
    """The price of rendering instead of patching, charged up front rather than an hour in."""
    template = tmp_path / "step1.yaml"
    template.write_text("max_num_epochs: 20\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"never references \$RUN_NAME, \$TRAIN_SET"):
        render_config(
            template,
            {},
            required=frozenset({"TRAIN_SET", "RUN_NAME"}),
            dest=tmp_path / "rendered.yaml",
        )


def test_a_missing_template_reports_itself_as_a_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="MLIP training template not found"):
        render_config(
            tmp_path / "absent.yaml", {}, required=frozenset(), dest=tmp_path / "rendered.yaml"
        )


# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------


def test_the_shared_placeholders_describe_the_run(tmp_path: Path):
    values = base_placeholders(_plan(tmp_path, gpus=2, start_from="medium"))
    assert values["RUN_DIR"] == str(tmp_path / "train")
    assert values["NGPUS"] == "2"
    assert values["FOUNDATION_MODEL"] == "medium"


def test_training_from_scratch_renders_an_empty_foundation_model(tmp_path: Path):
    assert base_placeholders(_plan(tmp_path, start_from=None))["FOUNDATION_MODEL"] == ""


def test_a_trainer_can_specialise_a_shared_placeholder(tmp_path: Path):
    """Precedence is decided once, so it cannot depend on where the render happened."""

    class _Trainer:
        def placeholders(self, plan: TrainingPlan, data: DatasetFiles) -> dict[str, str]:
            return {"RUN_NAME": "overridden", "TRAIN_SET": str(data.train)}

    merged = placeholders_for(
        _Trainer(),
        _plan(tmp_path),
        DatasetFiles(train=Path("/d/train.xyz")),
    )
    assert merged["RUN_NAME"] == "overridden"
    assert merged["RUN_DIR"] == str(tmp_path / "train"), "the rest still comes from the base"


def test_bindir_is_the_launchers_own(tmp_path: Path):
    """Where a backend's console scripts live, right in all three provisioner outcomes."""
    assert _plan(tmp_path).bindir == Path("/envs/mlip-mace/bin")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_every_mace_task_name_dispatches_to_the_mace_trainer():
    for task in ("mace_off", "mace_mp", "mace_omol"):
        assert trainer_for(task).__name__ == "MaceTrainer"


def test_the_trainer_is_selected_by_task_name_alone():
    """``model_path`` selects nothing — it says where the named library's weights are.

    It used to select: a ``model_path`` with no ``task_name`` dispatched to MACE. That read
    well while MACE was the only library that could produce a checkpoint, and became a way to
    hand a FAIRChem model to a MACE loader as soon as a second one could. Continuing a local
    checkpoint now means naming its library, which is the same word used to run it.
    """
    assert trainer_for("mace_off").__name__ == "MaceTrainer"
    assert trainer_for("omol").__name__ == "FairchemTrainer"
    assert inspect.signature(trainer_for).parameters.keys() == {"task_name"}


def _untrainable_task():
    """A synthetic runnable-but-untrainable registry entry, patched in for one test.

    Synthetic rather than a shipped task: every shipped backend is on its way to a
    trainer, and a fixture reading "X happens to lack one today" breaks the day that
    stops being true — a fact about the roster, not about the rule under test.
    """
    from unittest.mock import patch

    from chemrefine.engines.mlip import registry as mlip_registry
    from chemrefine.engines.mlip.registry import BackendSpec, MlipLibrary

    lib = MlipLibrary(extra="mlip-untrainable", package="untrainable-lib", import_name="untl")
    spec = BackendSpec(lib, builder=lambda **_: None, trainer=None)
    return patch.dict(mlip_registry._BACKENDS, {"untrainable": spec}, clear=False)


def test_a_runnable_but_untrainable_task_says_which_it_is():
    """ "Unknown task" and "known task, no trainer" are different mistakes.

    The first is a typo; the second is a library chemrefine can run but not yet train, which
    is a fact about the backend rather than about the config. One registry is what lets the
    message tell them apart at all.
    """
    with _untrainable_task(), pytest.raises(ConfigError, match="can be run but not trained"):
        trainer_for("untrainable")
    with pytest.raises(ConfigError, match="unsupported MLIP backend"):
        trainer_for("not_a_task_at_all")


def test_the_registry_reports_its_own_membership():
    assert "mace_off" in registered_trainers()
    # ``<=``, not ``<``: every shipped backend now trains, so the two sets are equal —
    # the doctrine is that trainable can never *exceed* runnable, not that some backend
    # must be left behind.
    assert registered_trainers() <= registered_backends(), "trainable implies runnable"
    assert registered_extras() >= {"mlip-mace"}


def test_the_backend_requirement_follows_the_selection():
    requirement = requirement_from_options(
        {"task_name": "mace_off", "device": "cpu"}, require_trainer=True
    )
    assert (requirement.extra, requirement.import_name) == ("mlip-mace", "mace")


def test_a_training_step_naming_an_untrainable_task_is_refused_by_the_preflight():
    """Before any step submits, rather than after the upstream steps computed a dataset."""
    with _untrainable_task():
        with pytest.raises(ConfigError, match="can be run but not trained"):
            requirement_from_options({"task_name": "untrainable"}, require_trainer=True)
        # ...while the same selection is a perfectly good *inference* backend.
        assert requirement_from_options({"task_name": "untrainable"}).extra == "mlip-untrainable"


def test_the_requirement_reads_the_selection_through_the_model():
    """`task` is an accepted spelling of `task_name`; the preflight must honour it too.

    A second reader of the same knobs is free to disagree with the first about aliases, and
    the disagreement shows up as a preflight that passes a step the engine then rejects.
    """
    assert requirement_from_options({"task": "mace_off"}).extra == "mlip-mace"


def test_digest_of_reads_the_files_contents(tmp_path: Path):
    """The sidecar records which weights a run produced, so it must follow the bytes."""
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    before = digest_of(model)
    model.write_bytes(b"other weights")

    assert len(before) == 16
    assert digest_of(model) != before


def test_digest_of_is_computable_where_sha1_is_policy_restricted(tmp_path: Path, monkeypatch):
    """Recording what a training step produced must not need a restricted code path.

    `file_digest` handed the name `"sha1"` goes through `hashlib.new(...)` with
    `usedforsecurity` at its default, which a host allowing SHA-1 only as a fingerprint
    refuses — failing a step for describing work it had already finished. Passing the
    constructor keeps `new` out of it, which is what this pins.
    """
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")

    def refuse(name: str, *args: object, **kwargs: object):
        raise AssertionError(f"hashlib.new({name!r}) is the path a crypto policy can refuse")

    monkeypatch.setattr(training.hashlib, "new", refuse)
    assert len(digest_of(model)) == 16


def test_an_empty_split_is_still_a_valid_dataset_split():
    """`DatasetSplit` is a value; the emptiness rules live in `split_structures`."""
    assert DatasetSplit(train=(), valid=(), test=()).counts() == "0 train / 0 valid / 0 test"


# ---------------------------------------------------------------------------
# The invariants one registry buys
# ---------------------------------------------------------------------------


def test_every_trainable_task_can_also_be_run():
    """The property the merged registry exists to guarantee: trainable implies runnable.

    A library's training CLI and its ASE calculator come out of one `chemrefine[<extra>]`
    install, so a task that can be trained and not run would be a step that fine-tunes a model
    it cannot then evaluate. When this was two registries each declared its own task list and
    its own packaging triple, and nothing compared them.

    (What this replaced compared `requirement_from_options` to itself under a flag whose raise
    could not fire, because the loop iterated exactly the trainable tasks. It would have passed
    with `require_trainer` deleted.)
    """
    for task in sorted(registered_trainers()):
        spec = backend_spec(task)
        assert spec.trainer is not None
        assert spec.builder is not None, f"{task} can be trained but not run"


def test_a_task_that_cannot_train_is_refused_by_the_preflight():
    """`require_trainer` is what turns a typo into a message before any step submits.

    On a pipeline that spends days computing labels before it trains, this is the difference
    between a mistake caught in seconds and one caught on Thursday. Asserted on the
    synthetic runnable-but-untrainable task (:func:`_untrainable_task`), so the fixture
    stays true however the shipped roster evolves.
    """
    with _untrainable_task():
        assert requirement_from_options({"task_name": "untrainable"}).extra == "mlip-untrainable"
        with pytest.raises(ConfigError, match="can be run but not trained"):
            requirement_from_options({"task_name": "untrainable"}, require_trainer=True)


def test_every_registered_extra_is_declared_in_pyproject():
    """A backend naming an extra that does not exist provisions an empty environment.

    `pip install "chemrefine[typo]"` warns and exits 0, so `chemrefine backends install`
    succeeds, `<env>/bin/python` exists, the preflight is satisfied — and the step then dies
    on the backend import. Nothing tied the code's extras to the metadata that declares them.
    """
    import tomllib

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = set(tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"])

    assert registered_extras() <= declared, (
        f"extras declared by a backend but not by pyproject: "
        f"{sorted(registered_extras() - declared)}"
    )


def test_every_backend_module_imports_with_no_mlip_library_installed():
    """The registry is built at import time, so every library module must import cheaply.

    A top-level `import mace` in one of them would make the whole package unimportable on a
    machine that has FAIRChem instead. Merging each trainer in beside its calculator puts
    more code behind that promise, which is why it is asserted rather than trusted: the heavy
    imports must stay inside the builder and inside the generated job.
    """
    import ast

    backends = Path(__file__).resolve().parent.parent / "src/chemrefine/engines/mlip/backends"
    heavy = {"mace", "fairchem", "sevenn", "orb_models", "chgnet", "torch", "aimnet"}
    for module in sorted(backends.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in tree.body:  # top level only — inside a function is the whole point
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            assert not (set(names) & heavy), f"{module.name} imports {names} at module scope"


# ---------------------------------------------------------------------------
# The trainer contract — invariants every registered trainer inherits by existing
# ---------------------------------------------------------------------------
#
# Parametrised over ``registered_trainers()`` at collection time, so a backend that gains
# a trainer gains this floor with no edit here — the same property the calculator
# invariants already have. Each invariant is one of the contract's stated rules
# (``training.Trainer``'s docstrings); a trainer that cannot pass them cannot be driven
# by the engine, whatever its own unit tests say.


def _full_split() -> DatasetSplit:
    """Every split populated — the baseline shape a trainer must accept."""
    structures = [_labelled(str(i), energy=-1.0 - 0.1 * i) for i in range(8)]
    return split_structures(structures, valid_fraction=0.25, test_fraction=0.125, seed=1)


@pytest.mark.parametrize("task", sorted(registered_trainers()))
def test_every_trainer_writes_its_dataset_in_the_orchestrators_process(tmp_path: Path, task: str):
    """``write_dataset`` may use only ase + numpy — proven by execution, not by review.

    No MLIP backend is installed in this suite's environment, so a writer that imported
    its library would fail right here. That constraint is what keeps every trainer module
    importable at registry-build time (``training.Trainer.write_dataset``'s stated rule);
    a format that genuinely needs the backend belongs in ``command``, where the backend
    environment is live.
    """
    trainer = trainer_for(task)()
    files = trainer.write_dataset(_plan(tmp_path), _full_split())
    assert files.train.is_file()
    assert files.valid is not None and files.valid.is_file()


@pytest.mark.parametrize("task", sorted(registered_trainers()))
def test_every_trainer_covers_its_required_placeholders(tmp_path: Path, task: str):
    """What a trainer *requires* of a template, it must also *supply* to the render.

    ``render_config`` refuses a template that references none of ``required_placeholders``;
    a required name the merged mapping never provides would make every template for that
    trainer unrenderable — a combination no caller can fix from the YAML.
    """
    trainer = trainer_for(task)()
    plan = _plan(tmp_path)
    files = trainer.write_dataset(plan, _full_split())
    provided = set(placeholders_for(trainer, plan, files))
    missing = trainer.required_placeholders - provided
    assert not missing, f"{task} requires placeholders it never supplies: {sorted(missing)}"


@pytest.mark.parametrize("task", sorted(registered_trainers()))
def test_every_trainer_names_its_artifact_without_a_plan(task: str):
    """``artifact(run_dir, run_name)`` must answer from those two values alone.

    ``rebuild-cache`` adopts a finished product having submitted nothing, and the step
    asks after a failure — both moments with no dataset to split and no launcher to
    resolve (the contract's own words). The product also has to live under the run
    directory, or the scheduler's copy-back and the manifest describe different trees.
    """
    trainer = trainer_for(task)()
    run_dir = Path("/proj/outputs/step3/train")
    artifact = trainer.artifact(run_dir, "train")
    assert run_dir in artifact.parents


@pytest.mark.parametrize("task", sorted(registered_trainers()))
def test_every_trainer_command_names_the_config_by_basename(tmp_path: Path, task: str):
    """The rendered command carries the config's basename, never its absolute path.

    The job script copies the config into ``$WORK_DIR`` and runs there, and under
    ``slurm_array: true`` the "config" is a ``$INP_NAME`` sentinel the array script
    expands per task — an absolute path in the command breaks both. And no trainer may
    emit a ``trap``: bash keeps one handler per signal, so it would displace the job
    script's single EXIT handler and take the copy-back with it.
    """
    trainer = trainer_for(task)()
    plan = _plan(tmp_path)
    cmd = trainer.command(plan, tmp_path / "step3_train.yaml")
    assert "step3_train.yaml" in cmd
    assert str(tmp_path / "step3_train.yaml") not in cmd
    assert "trap " not in cmd


@pytest.mark.parametrize("task", sorted(registered_trainers()))
def test_every_trainers_artifact_basename_is_copy_back_eligible(task: str):
    """Some declared ``output_globs`` pattern matches the artifact's own basename.

    The globs are the net that keeps a model out of the scratch cleanup when a template
    leaves the library writing into ``$WORK_DIR``. A trainer whose artifact no glob
    matches has declared a net with a hole exactly where its product lands. (FAIRChem's
    nested product is the documented exception the net cannot reach — its basename still
    matches, which is what this pins.)
    """
    from fnmatch import fnmatch

    trainer = trainer_for(task)()
    artifact = trainer.artifact(Path("/r"), "train")
    assert any(fnmatch(artifact.name, glob) for glob in trainer.output_globs), (
        f"{task}: no output_globs entry matches {artifact.name!r}"
    )


@pytest.mark.parametrize("task", sorted(registered_trainers()))
def test_a_missing_validation_set_is_refused_exactly_when_the_library_needs_one(
    tmp_path: Path, task: str
):
    """The refusal fires iff ``needs_validation`` — and says why in the library's words.

    The refusal is one base branch driven by two declarations rather than a copy per
    trainer, so the invariant to hold is the *iff*: a library that cannot train
    without validation refuses with its own reason, and one that can trains on.
    """
    trainer = trainer_for(task)()
    split = split_structures(
        [_labelled(str(i)) for i in range(4)], valid_fraction=0, test_fraction=0, seed=1
    )
    if trainer.needs_validation:
        assert trainer.validation_reason, f"{task}: a refusal with no why-clause"
        with pytest.raises(ConfigError, match=f"{trainer.label} training needs a validation set"):
            trainer.write_dataset(_plan(tmp_path), split)
    else:
        files = trainer.write_dataset(_plan(tmp_path), split)
        assert files.train.is_file() and files.valid is None


@pytest.mark.parametrize("task", sorted(registered_trainers()))
def test_non_neutral_data_warns_exactly_when_the_format_cannot_carry_it(
    tmp_path: Path, task: str, caplog
):
    """The charge/spin warning fires iff the dataset has no channel for either.

    A format that carries charge and spin (MACE's ``total_*`` keys, FAIRChem's row data)
    fits an ion as itself and must not cry wolf; one that cannot must never be silent —
    and neutral-singlet data warns nowhere.
    """
    import logging

    trainer = trainer_for(task)()
    with caplog.at_level(logging.WARNING):
        trainer.write_dataset(_plan(tmp_path, charge=-1, multiplicity=2), _full_split())
    warned = "no charge/spin channel" in caplog.text
    assert warned == (not trainer.charge_spin_aware), f"{task}: warning iff unaware"

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        trainer.write_dataset(_plan(tmp_path / "neutral"), _full_split())
    assert "no charge/spin channel" not in caplog.text


@pytest.mark.parametrize("task", sorted(registered_trainers()))
def test_an_empty_split_gets_no_file_from_any_trainer(tmp_path: Path, task: str):
    """An empty split yields ``None``, never a zero-byte file.

    ase raises ``Empty file`` on a zero-byte extxyz, so naming one turns "no test set"
    into a crash inside the library — the base's loop skips it for every trainer.
    """
    trainer = trainer_for(task)()
    split = split_structures(
        [_labelled(str(i)) for i in range(8)], valid_fraction=0.25, test_fraction=0, seed=1
    )
    plan = _plan(tmp_path)
    files = trainer.write_dataset(plan, split)
    assert files.test is None
    assert not list(plan.run_dir.glob("test.*")), f"{task}: an empty split left a file"


@pytest.mark.parametrize("task", sorted(registered_trainers()))
def test_every_trainer_supplies_the_dataset_placeholder(tmp_path: Path, task: str):
    """``$TRAIN_SET`` always renders as the file the trainer just wrote.

    The one placeholder every template must reference (the contract's floor), so the one
    value every trainer must supply — whether it inherits the base triple or overrides
    wholesale, as FAIRChem does.
    """
    trainer = trainer_for(task)()
    plan = _plan(tmp_path)
    files = trainer.write_dataset(plan, _full_split())
    assert placeholders_for(trainer, plan, files)["TRAIN_SET"] == str(files.train)


@pytest.mark.parametrize("task", sorted(registered_trainers()))
def test_no_trainer_mutates_the_pipelines_structures(tmp_path: Path, task: str):
    """The dataset lands on copies — the structures go on through the pipeline.

    A calculator or label key attached in place would ride the shared reference into
    every later step, and the positions buffer is the one ``structure_digest`` hashes into
    downstream cache keys.
    """
    structures = [_labelled(str(i)) for i in range(8)]
    split = split_structures(structures, valid_fraction=0.25, test_fraction=0.125, seed=1)
    trainer_for(task)().write_dataset(_plan(tmp_path), split)
    for struct in structures:
        assert struct.atoms.calc is None
        assert not struct.atoms.info, f"{task}: the seed's info dict gained training keys"
        assert struct.forces_ev_per_a is not None and not struct.forces_ev_per_a.flags.writeable


def test_the_shared_writer_refuses_an_unlabelled_structure(tmp_path: Path):
    """The backstop behind ``split_structures``' refusal, named for the caller that skipped it.

    ``labelled_atoms`` is public machinery: a trainer (or a future caller) could hand it
    structures that never went through the split's label check, and an unlabelled frame
    written silently would surface as a library-side KeyError an hour into the job.
    """
    from chemrefine.engines.mlip.train.base import write_labelled_extxyz

    bare = Structure(id="9", atoms=Atoms("H"))
    with pytest.raises(ConfigError, match="structure 9 is missing"):
        write_labelled_extxyz(tmp_path / "train.xyz", [bare])
