"""Tests for the MLIP backend registry in :mod:`chemrefine.engines.mlip`.

``task_name`` is the registry key (the method/head); ``model_name`` is the
weights handed to the builder. The real ML backends aren't installed in the
dev / CI env, so each test installs a *fake* module under the right
``sys.modules`` key before instantiating :class:`MlipCalculator`, so the real
built-in builder bodies (in ``chemrefine.engines.mlip.backends.*``) run without
a multi-GB ML stack.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, CalculatorSetupError, all_changes

from chemrefine.engines.mlip.calculator import MlipCalculator, build_calculator
from chemrefine.engines.mlip.registry import (
    VACUUM_PADDING_ANGSTROM,
    MlipLibrary,
    backend_spec,
    registered_backends,
)
from chemrefine.errors import ConfigError


def _fake_module(name: str, **attrs: object) -> types.ModuleType:
    """Build a ``types.ModuleType`` with ``attrs`` set (dynamic, no type: ignore)."""
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


# ---------------------------------------------------------------------------
# MACE — task_name selects the variant; model_name is the size
# ---------------------------------------------------------------------------


def _install_fake_mace(monkeypatch) -> dict[str, MagicMock]:
    """Insert a fake ``mace.calculators`` exposing the three factories."""
    factories = {
        "mace_off": MagicMock(return_value="OFF_CALC"),
        "mace_mp": MagicMock(return_value="MP_CALC"),
        "mace_omol": MagicMock(return_value="OMOL_CALC"),
        "MACECalculator": MagicMock(return_value="CUSTOM_CALC"),
    }
    mod = _fake_module("mace.calculators", **factories)
    monkeypatch.setitem(sys.modules, "mace", _fake_module("mace", calculators=mod))
    monkeypatch.setitem(sys.modules, "mace.calculators", mod)
    return factories


def test_build_mace_off_uses_model_name_as_size(monkeypatch):
    factories = _install_fake_mace(monkeypatch)
    calc = MlipCalculator(task_name="mace_off", model_name="medium", device="cpu")
    factories["mace_off"].assert_called_once_with(model="medium", device="cpu")
    assert calc.calculator == "OFF_CALC"


def test_build_mace_mp_uses_model_name_as_size(monkeypatch):
    factories = _install_fake_mace(monkeypatch)
    calc = MlipCalculator(task_name="mace_mp", model_name="large", device="cpu")
    factories["mace_mp"].assert_called_once_with(model="large", device="cpu")
    assert calc.calculator == "MP_CALC"


def test_build_mace_omol_uses_model_name_as_size(monkeypatch):
    factories = _install_fake_mace(monkeypatch)
    calc = MlipCalculator(task_name="mace_omol", model_name="extra_large", device="cpu")
    factories["mace_omol"].assert_called_once_with(model="extra_large", device="cpu")
    assert calc.calculator == "OMOL_CALC"


def test_build_mace_default_size_is_none(monkeypatch):
    """An empty ``model_name`` falls back to the library default (``model=None``)."""
    factories = _install_fake_mace(monkeypatch)
    MlipCalculator(task_name="mace_off", model_name="", device="cpu")
    factories["mace_off"].assert_called_once_with(model=None, device="cpu")


def test_build_calculator_unknown_task_raises(monkeypatch):
    """A `ConfigError` — an unknown task is a config mistake and carries the CLI's exit code."""
    _install_fake_mace(monkeypatch)
    with pytest.raises(ConfigError, match="unsupported MLIP backend"):
        build_calculator(task_name="mace_weird", model_name="x", device="cpu")


@pytest.mark.parametrize("task", ["mace_off", "mace_mp", "mace_omol"])
def test_a_local_checkpoint_goes_through_the_named_familys_own_loader(
    monkeypatch, tmp_path: Path, task: str
):
    """MACE's foundation loaders take a path, so a checkpoint needs no separate task key.

    ``mace_off``'s ``model`` is typed ``str | Path`` and documented as "path to the model";
    only a known name or an ``https:`` URL is treated as a download. That is what lets one
    builder serve released weights and fine-tuned ones alike — and why there is no
    ``custom_fairchem`` to mirror the old ``custom_mace``.
    """
    factories = _install_fake_mace(monkeypatch)
    model_file = tmp_path / "fake.model"
    model_file.touch()
    MlipCalculator(task_name=task, model_name="ignored", device="cuda", model_path=str(model_file))
    factories[task].assert_called_once_with(model=model_file, device="cuda")


def _registered_tasks() -> list[str]:
    from chemrefine.engines.mlip.registry import registered_backends

    return sorted(registered_backends())


@pytest.mark.parametrize("task", _registered_tasks())
def test_a_checkpoint_that_is_not_there_names_the_task(tmp_path: Path, task: str):
    """The dispatch vets the checkpoint once, for every backend, before any library imports.

    The refusal names the ``task_name`` the user wrote — the YAML value they can act on.
    A library's own failure is a ``torch.load`` traceback naming neither the step nor the
    option, which on a pipeline whose training step ran overnight is the difference
    between a typo and a hunt. One parametrised test rather than a copy per backend,
    because the check itself lives in the dispatch rather than in each builder.
    """
    missing = tmp_path / "does_not_exist.model"
    with pytest.raises(FileNotFoundError, match=f"{task} checkpoint not found: {missing}"):
        MlipCalculator(task_name=task, model_name="x", model_path=str(missing))


def test_the_legacy_alias_without_a_checkpoint_says_what_to_write_instead(monkeypatch):
    """``custom_mace`` is kept for old configs, and can only mean "load this file".

    Naming it with nothing to load is the one case it cannot serve, so it says which family to
    name rather than silently picking one — the alias predates the families and has no way to
    know which was meant.
    """
    _install_fake_mace(monkeypatch)
    with pytest.raises(ConfigError, match="mace_off, mace_mp, mace_omol"):
        MlipCalculator(task_name="custom_mace", model_name="medium")


def test_the_legacy_alias_still_loads_a_checkpoint(monkeypatch, tmp_path: Path):
    """A v1 config naming ``custom_mace`` + ``model_path`` keeps working unchanged."""
    factories = _install_fake_mace(monkeypatch)
    model_file = tmp_path / "fake.model"
    model_file.touch()
    MlipCalculator(task_name="custom_mace", model_path=str(model_file), device="cpu")
    factories["mace_off"].assert_called_once_with(model=model_file, device="cpu")


# ---------------------------------------------------------------------------
# FAIRChem (UMA / eSEN) — task_name is the head, model_name the checkpoint
# ---------------------------------------------------------------------------


def _install_fake_fairchem(monkeypatch) -> tuple[MagicMock, MagicMock]:
    """Install a fake ``fairchem.core``; return ``(get_predict_unit, calc_class)``."""
    predict = MagicMock()
    predict.get_predict_unit = MagicMock(return_value="PREDICTOR")
    fairchem_calc = MagicMock(return_value="FAIRCHEM_CALC")
    core = _fake_module("fairchem.core", FAIRChemCalculator=fairchem_calc, pretrained_mlip=predict)
    monkeypatch.setitem(sys.modules, "fairchem", _fake_module("fairchem", core=core))
    monkeypatch.setitem(sys.modules, "fairchem.core", core)
    return predict.get_predict_unit, fairchem_calc


def _install_fake_load_predict_unit(monkeypatch) -> MagicMock:
    """Add the *other* loader — the one that takes a path — to the fake fairchem.

    It lives in a different submodule from `get_predict_unit`, which is why the builder
    imports it inside the branch that uses it: a caller running a named release must not need
    this one to be importable.
    """
    loader = MagicMock(return_value="PATH_PREDICTOR")
    unit = _fake_module("fairchem.core.units.mlip_unit", load_predict_unit=loader)
    monkeypatch.setitem(sys.modules, "fairchem.core.units", _fake_module("fairchem.core.units"))
    monkeypatch.setitem(sys.modules, "fairchem.core.units.mlip_unit", unit)
    return loader


def test_build_fairchem_loads_a_local_checkpoint_through_the_path_api(monkeypatch, tmp_path):
    """A model a training step produced can only be run through `load_predict_unit`.

    `get_predict_unit` resolves a *registry name* and raises `KeyError` for a path — there is
    no path branch in it — so without this the FAIRChem trainer could produce a checkpoint
    that nothing in the pipeline could then load.
    """
    _get_predict_unit, fairchem_calc = _install_fake_fairchem(monkeypatch)
    loader = _install_fake_load_predict_unit(monkeypatch)
    ckpt = tmp_path / "inference_ckpt.pt"
    ckpt.touch()

    calc = MlipCalculator(task_name="omol", model_path=str(ckpt), device="cpu")

    loader.assert_called_once_with(str(ckpt), device="cpu")
    _get_predict_unit.assert_not_called()
    fairchem_calc.assert_called_once_with("PATH_PREDICTOR", task_name="omol")
    assert calc.calculator == "FAIRCHEM_CALC"


def test_build_fairchem_routes_through_predictor(monkeypatch):
    """The builder loads the checkpoint then constructs a ``FAIRChemCalculator``."""
    get_predict_unit, fairchem_calc = _install_fake_fairchem(monkeypatch)
    calc = MlipCalculator(task_name="omol", model_name="uma-s-1p2", device="cuda")
    get_predict_unit.assert_called_once_with(model_name="uma-s-1p2", device="cuda")
    fairchem_calc.assert_called_once_with("PREDICTOR", task_name="omol")
    assert calc.calculator == "FAIRCHEM_CALC"


@pytest.mark.parametrize("head", ["omat", "odac", "oc20", "oc22", "oc25", "omc"])
def test_build_fairchem_passes_non_omol_head_through(monkeypatch, head):
    """Every non-``omol`` head passes through: ``task_name: omat`` builds the omat head."""
    _get_predict_unit, fairchem_calc = _install_fake_fairchem(monkeypatch)
    MlipCalculator(task_name=head, model_name="uma-s-1p2", device="cpu")
    fairchem_calc.assert_called_once_with("PREDICTOR", task_name=head)


def test_build_fairchem_defaults_model_name(monkeypatch):
    """An empty ``model_name`` falls back to the default UMA checkpoint."""
    get_predict_unit, _calc = _install_fake_fairchem(monkeypatch)
    MlipCalculator(task_name="omol", model_name="", device="cpu")
    get_predict_unit.assert_called_once_with(model_name="uma-s-1p2", device="cpu")


# ---------------------------------------------------------------------------
# CHGNet
# ---------------------------------------------------------------------------


def _install_fake_chgnet(monkeypatch) -> tuple[MagicMock, MagicMock]:
    """Install a fake ``chgnet.model`` (canonical import); return ``(CHGNet cls, calc)``.

    ``load`` and ``from_file`` accept only what the real classmethods accept — ``load``
    is keyword-only with a ``model_name`` — so a builder calling either with a stray
    positional fails here the way it fails in production. The old bare ``MagicMock``
    accepted anything, which is how ``CHGNet.load(path)`` — a ``TypeError`` against the
    real keyword-only signature — sat pinned as the checkpoint path for two releases.
    """

    def _load(*, model_name: str = "0.3.0", **_kwargs: object) -> str:
        return "CHGNET_MODEL"

    def _from_file(path: str, **_kwargs: object) -> str:
        return f"CHGNET_FROM_FILE:{path}"

    chgnet_cls = MagicMock()
    chgnet_cls.load = MagicMock(side_effect=_load)
    chgnet_cls.from_file = MagicMock(side_effect=_from_file)
    calc_class = MagicMock(return_value=MagicMock(implemented_properties=["energy", "forces"]))
    model_mod = _fake_module("chgnet.model", CHGNet=chgnet_cls, CHGNetCalculator=calc_class)
    monkeypatch.setitem(sys.modules, "chgnet", _fake_module("chgnet", model=model_mod))
    monkeypatch.setitem(sys.modules, "chgnet.model", model_mod)
    return chgnet_cls, calc_class


def test_build_chgnet_uses_the_released_default_when_nothing_is_named(monkeypatch):
    chgnet_cls, calc_class = _install_fake_chgnet(monkeypatch)
    calc = MlipCalculator(task_name="chgnet", device="cpu")
    chgnet_cls.load.assert_called_once_with()
    chgnet_cls.from_file.assert_not_called()
    calc_class.assert_called_once_with(model="CHGNET_MODEL", use_device="cpu")
    assert calc.calculator.inner is calc_class.return_value  # boxed: CHGNet is periodic-only


def test_build_chgnet_passes_a_release_name_through(monkeypatch):
    """`model_name` is a CHGNet release for keyword-only `load` — the knob the old
    kwargs builder silently dropped into its catch-all, which the spec makes impossible."""
    chgnet_cls, _calc_class = _install_fake_chgnet(monkeypatch)
    MlipCalculator(task_name="chgnet", model_name="0.3.0", device="cpu")
    chgnet_cls.load.assert_called_once_with(model_name="0.3.0")
    chgnet_cls.from_file.assert_not_called()


def test_a_chgnet_checkpoint_goes_through_from_file(monkeypatch, tmp_path: Path):
    """A local checkpoint is ``CHGNet.from_file``'s job — ``load`` takes release names.

    ``CHGNet.load`` is keyword-only (``*, model_name="0.3.0"``) and resolves names, never
    paths; ``from_file`` reads the ``{"model": as_dict()}`` file the training driver
    saves. This is what closes the train→run round trip for a fine-tuned CHGNet.
    """
    chgnet_cls, calc_class = _install_fake_chgnet(monkeypatch)
    model_file = tmp_path / "finetuned.pth.tar"
    model_file.touch()
    MlipCalculator(task_name="chgnet", device="cpu", model_path=str(model_file))
    chgnet_cls.from_file.assert_called_once_with(str(model_file))
    chgnet_cls.load.assert_not_called()
    calc_class.assert_called_once_with(model=f"CHGNET_FROM_FILE:{model_file}", use_device="cpu")


# ---------------------------------------------------------------------------
# SevenNet — task_name selects it, the weights come from name or path
# ---------------------------------------------------------------------------


def _install_fake_sevenn(monkeypatch) -> MagicMock:
    factory = MagicMock(return_value="SEVENN_CALC")
    mod = _fake_module("sevenn.calculator", SevenNetCalculator=factory)
    monkeypatch.setitem(sys.modules, "sevenn", _fake_module("sevenn", calculator=mod))
    monkeypatch.setitem(sys.modules, "sevenn.calculator", mod)
    return factory


def test_build_sevenn_constructs_calculator(monkeypatch):
    factory = _install_fake_sevenn(monkeypatch)
    calc = MlipCalculator(task_name="sevenn", model_name="7net-0", device="cpu")
    factory.assert_called_once_with(model="7net-0", device="cpu")
    assert calc.calculator == "SEVENN_CALC"


def test_a_sevenn_checkpoint_reaches_sevenns_own_loader(monkeypatch, tmp_path: Path):
    """``model_path`` is honoured with SevenNet's library, like every builder's.

    ``SevenNetCalculator``'s ``model`` is typed ``str | Path`` — "or path to the checkpoint"
    — and its resolution checks the filesystem before trying release names. Dropping the
    option instead ran the *named release* against a config that pinned a file, silently:
    the fingerprint had digested the checkpoint, so the run even looked pinned to it.
    """
    factory = _install_fake_sevenn(monkeypatch)
    model_file = tmp_path / "finetuned.pth"
    model_file.touch()
    MlipCalculator(
        task_name="sevenn", model_name="7net-0", device="cuda", model_path=str(model_file)
    )
    factory.assert_called_once_with(model=model_file, device="cuda")


def test_build_sevenn_falls_back_to_the_librarys_own_default(monkeypatch):
    """Neither name nor path: SevenNet's own default release loads — the library owns its
    default, exactly as FAIRChem's builder owns `uma-s-1p2`."""
    factory = _install_fake_sevenn(monkeypatch)
    MlipCalculator(task_name="sevenn", device="cpu")
    factory.assert_called_once_with(device="cpu")


# ---------------------------------------------------------------------------
# ORB — a separate library, model_name names the loader
# ---------------------------------------------------------------------------


def _install_fake_orb(monkeypatch) -> tuple[MagicMock, MagicMock, tuple[object, object]]:
    """Install a fake ``orb_models`` tree; return ``(loader, calc_class, (model, adapter))``.

    Shaped as orb-models is from the 0.6 floor on: the loader is registered in
    ``ORB_PRETRAINED_MODELS`` and returns ``(model, atoms_adapter)``, and the calculator
    takes the adapter positionally, so a builder that drops it fails here as it does
    against the real library. The module also holds an architecture builder, which is not
    a loader.
    """
    loaded = (object(), object())
    loader = MagicMock(return_value=loaded)
    loader.__name__ = "orb_v3_conservative_inf_omat"
    pretrained = _fake_module(
        "orb_models.forcefield.pretrained",
        orb_v3_conservative_inf_omat=loader,
        orb_v2_architecture=MagicMock(),
        ORB_PRETRAINED_MODELS={"orb-v3-conservative-inf-omat": loader},
    )
    forcefield = _fake_module("orb_models.forcefield", pretrained=pretrained)

    def _calculator(model, atoms_adapter, *, device=None):
        return "ORB_CALC"

    calc_class = MagicMock(side_effect=_calculator)
    calc_mod = _fake_module("orb_models.forcefield.inference.calculator", ORBCalculator=calc_class)
    inference = _fake_module("orb_models.forcefield.inference")
    for name, mod in {
        "orb_models": _fake_module("orb_models", forcefield=forcefield),
        "orb_models.forcefield": forcefield,
        "orb_models.forcefield.pretrained": pretrained,
        "orb_models.forcefield.inference": inference,
        "orb_models.forcefield.inference.calculator": calc_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return loader, calc_class, loaded


def test_build_orb_constructs_real_orb_calculator(monkeypatch):
    """The calculator gets the model and the atoms adapter the loader returned beside it.

    orb-models' calculator takes the adapter positionally; dropped, ``task_name: orb`` failed
    before its first structure. The model loads uncompiled: a compile is paid once per
    process, and a ChemRefine process serves one job.
    """
    loader, calc_class, (model, adapter) = _install_fake_orb(monkeypatch)
    calc = MlipCalculator(task_name="orb", model_name="orb_v3_conservative_inf_omat", device="cpu")
    loader.assert_called_once_with(device="cpu", compile=False)
    calc_class.assert_called_once_with(model, adapter, device="cpu")
    assert calc.calculator == "ORB_CALC"


def test_an_unset_orb_model_name_runs_the_default_loader(monkeypatch, tmp_path: Path):
    """``task_name: orb`` alone runs a model, as every other library's builder does.

    The ``model_name`` contract is that unset means the library's own default. Every other
    builder honoured it while ORB's asked ``pretrained`` for a loader named ``""`` and
    reported "unknown ORB model ''" — inside the job, after the step was scheduled. A
    checkpoint from an ``mlip-train`` step is the same case: it carries weights, not an
    architecture, and the run that produced it never had to name one either.
    """
    loader, calc_class, (model, adapter) = _install_fake_orb(monkeypatch)
    MlipCalculator(task_name="orb", device="cpu")
    loader.assert_called_once_with(device="cpu", compile=False)
    calc_class.assert_called_once_with(model, adapter, device="cpu")

    loader.reset_mock()
    model_file = tmp_path / "finetuned.ckpt"
    model_file.touch()
    MlipCalculator(task_name="orb", device="cpu", model_path=str(model_file))
    loader.assert_called_once_with(weights_path=str(model_file), device="cpu", compile=False)


_FAKE_LIBRARIES = {
    "mace": _install_fake_mace,
    "fairchem": _install_fake_fairchem,
    "chgnet": _install_fake_chgnet,
    "sevenn": _install_fake_sevenn,
    "orb_models": _install_fake_orb,
}
"""One fake per library the registry knows, keyed by the module the provisioner probes."""


@pytest.mark.parametrize("task_name", sorted(registered_backends()))
def test_every_backend_builds_a_calculator_with_model_name_unset(monkeypatch, task_name: str):
    """Unset ``model_name`` means the library's own default, for every registered backend.

    The contract is stated on the options model and in the docs table, and it was held by
    hand: one builder asked its library for a loader named ``""`` and reported an unknown
    model — inside the job, after the step was scheduled. Held here over the registry
    rather than per backend, a backend added later is covered by existing, and a library
    with no fake in the table above fails loudly instead of going quietly untested.

    The one task that names no library default — the legacy ``custom_mace`` alias, whose
    meaning is "your own checkpoint" — keeps the other half of the contract: a
    :class:`ConfigError` before the library is touched, naming ``model_path`` as the
    knob to write. Anything else escaping a builder is the failure this test refuses.
    """
    library = backend_spec(task_name).library.import_name
    _FAKE_LIBRARIES[library](monkeypatch)
    try:
        calc = MlipCalculator(task_name=task_name, device="cpu")
    except ConfigError as e:
        assert "model_path" in str(e), f"{task_name}: a refusal that does not name the fix"
    else:
        assert calc.calculator is not None


def test_build_orb_unknown_loader_raises(monkeypatch):
    """A model that isn't a loader in orb_models.forcefield.pretrained is a config error.

    ``ConfigError`` rather than ``ValueError``: the name came from the YAML, and the
    exit-code contract every config mistake honours must hold here too — bare, it
    reached the ExtOpt server as "crashed during startup" and the CLI as a traceback.
    """
    _install_fake_orb(monkeypatch)
    with pytest.raises(ConfigError, match="unknown ORB model"):
        MlipCalculator(task_name="orb", model_name="orb_not_a_loader", device="cpu")


def test_an_orb_module_attribute_that_is_not_a_loader_is_an_unknown_model(monkeypatch):
    """A name is resolved through orb's registry of loaders, not as any module attribute.

    An architecture builder lives in the same module; called with a loader's arguments it
    failed as a ``TypeError`` naming neither the step nor the option.
    """
    _install_fake_orb(monkeypatch)
    with pytest.raises(ConfigError, match="unknown ORB model 'orb_v2_architecture'"):
        MlipCalculator(task_name="orb", model_name="orb_v2_architecture", device="cpu")


def test_an_orb_checkpoint_reaches_the_named_loaders_weights_path(monkeypatch, tmp_path: Path):
    """``model_path`` is honoured with ORB's library, like every builder's.

    The pretrained loaders take ``weights_path`` (defaulting to the release URL) and accept
    a local file. ``model_name`` still names the loader: a checkpoint carries weights, not
    an architecture, so the loader that built it is named alongside it — the same doctrine
    as naming the library that trained it.
    """
    loader, calc_class, (model, adapter) = _install_fake_orb(monkeypatch)
    model_file = tmp_path / "finetuned.ckpt"
    model_file.touch()
    MlipCalculator(
        task_name="orb",
        model_name="orb_v3_conservative_inf_omat",
        device="cpu",
        model_path=str(model_file),
    )
    loader.assert_called_once_with(weights_path=str(model_file), device="cpu", compile=False)
    calc_class.assert_called_once_with(model, adapter, device="cpu")


# ---------------------------------------------------------------------------
# single_point + optimize — duck-typed fake ASE calculator
# ---------------------------------------------------------------------------


def _calc_with_fake(monkeypatch) -> MlipCalculator:
    """Construct an ``MlipCalculator`` with a fake backend already installed."""
    _install_fake_mace(monkeypatch)
    return MlipCalculator(task_name="mace_off", model_name="small", device="cpu")


def test_single_point_returns_energy_and_negative_gradient(monkeypatch):
    """``single_point`` returns ``(energy, [-fx, -fy, -fz] per atom)``."""
    calc = _calc_with_fake(monkeypatch)
    fake_forces = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    class _StubAtoms:
        def __init__(self):
            self.calc = None

        def get_potential_energy(self):
            return -42.0

        def get_forces(self):
            return fake_forces

    atoms = _StubAtoms()
    energy, gradient = calc.single_point(atoms)
    assert energy == -42.0
    assert gradient == [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert atoms.calc is calc.calculator


class _StubAtomsWithInfo:
    """The duck-typed stub above, plus the ``info`` dict the species stamping writes."""

    def __init__(self, info: dict | None = None):
        self.calc = None
        self.info: dict = dict(info or {})

    def get_potential_energy(self):
        return -1.0

    def get_forces(self):
        return np.zeros((1, 3))


def test_single_point_stamps_charge_and_spin_where_the_libraries_read_them(monkeypatch):
    """Constructor charge/multiplicity land in ``atoms.info`` — the per-geometry channel.

    Both shipped charge-aware backends read ``atoms.info["charge"]`` / ``["spin"]``
    (FAIRChem's a2g args name exactly those keys; MACE's calculator maps them onto its
    ``total_charge``/``total_spin`` inputs) and silently assume a neutral singlet when
    they are absent. Before the wrapper stamped them, the constructor arguments
    selected nothing on the direct path — an anion scored as a neutral molecule with
    nothing said — while the ExtOpt adapter stamped the very same keys from ORCA's
    per-call values (its own test sits in ``test_engines_mlip.py``).
    """
    _install_fake_mace(monkeypatch)
    calc = MlipCalculator(task_name="mace_omol", device="cpu", charge=-1, multiplicity=2)

    atoms = _StubAtomsWithInfo()
    calc.single_point(atoms)

    assert atoms.info == {"charge": -1, "spin": 2}


def test_a_value_the_template_set_on_the_atoms_itself_wins(monkeypatch):
    """``setdefault``: ``atoms.info`` is per-structure state, more specific than the step's.

    A template that sets its own per-geometry charge (a scan over charge states, say)
    must not have it overwritten by the step-wide constructor value.
    """
    _install_fake_mace(monkeypatch)
    calc = MlipCalculator(task_name="mace_omol", device="cpu", charge=-1, multiplicity=2)

    atoms = _StubAtomsWithInfo({"charge": 0})
    calc.single_point(atoms)

    assert atoms.info == {"charge": 0, "spin": 2}


def test_unset_charge_and_multiplicity_stamp_nothing(monkeypatch):
    """The defaults invent no keys — absent stays absent, and the library's own
    neutral-singlet assumption applies exactly as it would to a bare calculator."""
    calc = _calc_with_fake(monkeypatch)

    atoms = _StubAtomsWithInfo()
    calc.single_point(atoms)

    assert atoms.info == {}


def test_optimize_stamps_the_same_keys(monkeypatch):
    """The optimisation path shares the stamping — LBFGS calls the calculator per step,
    and every one of those calls reads the same ``atoms.info``."""
    _install_fake_mace(monkeypatch)
    calc = MlipCalculator(task_name="mace_omol", device="cpu", charge=1, multiplicity=1)

    lbfgs_instance = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "ase.optimize",
        _fake_module("ase.optimize", LBFGS=MagicMock(return_value=lbfgs_instance)),
    )
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    calc.optimize(atoms, fmax=0.05, steps=10)

    assert atoms.info["charge"] == 1
    assert atoms.info["spin"] == 1


def test_optimize_invokes_lbfgs_with_fmax_and_steps(monkeypatch):
    """``optimize`` runs ``ase.optimize.LBFGS(...).run(fmax=…, steps=…)``."""
    calc = _calc_with_fake(monkeypatch)

    lbfgs_instance = MagicMock()
    lbfgs_class = MagicMock(return_value=lbfgs_instance)
    monkeypatch.setitem(
        sys.modules, "ase.optimize", _fake_module("ase.optimize", LBFGS=lbfgs_class)
    )

    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    result = calc.optimize(atoms, fmax=0.05, steps=10)
    lbfgs_class.assert_called_once_with(atoms, logfile=os.devnull)
    lbfgs_instance.run.assert_called_once_with(fmax=0.05, steps=10)
    assert result is atoms
    assert atoms.calc is calc.calculator


def test_optimize_keeps_the_optimisers_verdict(monkeypatch, caplog):
    """An optimiser that ran out of steps says so; the helper no longer discards it.

    ``LBFGS.run`` returns whether ``fmax`` was reached within ``steps``. With that boolean
    dropped, an exhausted optimisation was indistinguishable from a converged one: the
    template wrote the last geometry's energy, the parser read no verdict, and the structure
    ranked as a survivor. The verdict now lands on the atoms and on the wrapper, so a
    template assigns ``converged = mlip.last_converged`` and the output contract carries it.
    """
    calc = _calc_with_fake(monkeypatch)
    lbfgs_instance = MagicMock()
    lbfgs_instance.run.return_value = False
    monkeypatch.setitem(
        sys.modules,
        "ase.optimize",
        _fake_module("ase.optimize", LBFGS=MagicMock(return_value=lbfgs_instance)),
    )
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    assert calc.last_converged is None, "no verdict before the first optimisation"

    with caplog.at_level("WARNING", logger="chemrefine.engines.mlip.calculator"):
        calc.optimize(atoms, fmax=0.05, steps=10)

    assert calc.last_converged is False
    assert atoms.info["converged"] is False
    assert "10 steps" in caplog.text and "0.05" in caplog.text

    lbfgs_instance.run.return_value = True
    calc.optimize(atoms, fmax=0.05, steps=10)
    assert calc.last_converged is True
    assert atoms.info["converged"] is True


# ---------------------------------------------------------------------------
# A periodic-only library evaluates a molecule in a vacuum box
# ---------------------------------------------------------------------------

_BOND = 0.74

_BOXING = MlipLibrary(
    extra="mlip-box",
    package="box-lib",
    import_name="box_lib",
    periodic_only=True,
    periodic_reason="it reads every geometry through its lattice",
)
"""A periodic-only library, for its :meth:`~MlipLibrary.boxed` alone."""


class _NeedsALattice(Calculator):
    """A harmonic H2 bond that, like CHGNet, cannot evaluate a structure with no cell."""

    def __init__(self):
        Calculator.__init__(self)
        self.implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if not self.atoms.cell.rank:
            raise np.linalg.LinAlgError("Singular matrix")
        bond = self.atoms.positions[1] - self.atoms.positions[0]
        length = float(np.linalg.norm(bond))
        pull = 2 * (length - _BOND) * bond / length
        self.results = {"energy": (length - _BOND) ** 2, "forces": np.array([pull, -pull])}


def _periodic_only(task: str = "needs_a_lattice") -> MlipLibrary:
    """Register a periodic-only library whose builder returns :class:`_NeedsALattice`."""
    library = MlipLibrary(
        extra="mlip-lattice",
        package="lattice-lib",
        import_name="lattice_lib",
        periodic_only=True,
        periodic_reason="it reads every geometry through its lattice",
    )
    library.calculator(task)(lambda _spec: _NeedsALattice())
    return library


def _h2(length: float) -> Atoms:
    return Atoms("H2", positions=[[1.0, 2.0, 3.0], [1.0 + length, 2.0, 3.0]])


def test_a_periodic_only_library_evaluates_a_molecule_in_a_vacuum_box(caplog):
    """The molecule is computed boxed, the caller's atoms stay as they were, and it is said once.

    Unboxed, CHGNet fails on a structure with no cell as a singular-matrix error naming
    neither the structure nor the library, in a single point, an optimisation and an
    ExtOpt server alike.
    """
    _periodic_only()
    mlip = MlipCalculator(task_name="needs_a_lattice", device="cpu")
    atoms = _h2(0.80)
    before = atoms.positions.copy()

    with caplog.at_level("WARNING", logger="chemrefine.engines.mlip.calculator"):
        energy, gradient = mlip.single_point(atoms)
        mlip.single_point(_h2(0.90))

    assert energy == pytest.approx((0.80 - _BOND) ** 2)
    assert gradient[0][0] == pytest.approx(-2 * (0.80 - _BOND))
    np.testing.assert_array_equal(atoms.positions, before)
    assert atoms.cell.rank == 0 and not atoms.pbc.any()
    assert caplog.text.count("lattice-lib is periodic-only") == 1
    assert "it reads every geometry through its lattice" in caplog.text
    assert "stress" not in mlip.calculator.implemented_properties  # the box's, not the molecule's


def test_an_optimisation_on_a_periodic_only_library_moves_only_the_callers_atoms():
    """LBFGS walks the caller's own atoms; each step is evaluated on a boxed copy.

    The path is the one LBFGS takes over a hand-boxed copy, shifted by the centring, so the
    box changes where the energy is computed and nothing about where the atoms go.
    """
    from ase.optimize import LBFGS

    _periodic_only()
    mlip = MlipCalculator(task_name="needs_a_lattice", device="cpu")
    atoms = _h2(0.90)
    relaxed = mlip.optimize(atoms, fmax=1e-4, steps=100)

    by_hand = _BOXING.boxed(_h2(0.90))
    shift = by_hand.positions[0] - _h2(0.90).positions[0]
    by_hand.calc = _NeedsALattice()
    LBFGS(by_hand, logfile=None).run(fmax=1e-4, steps=100)

    assert relaxed is atoms and mlip.last_converged is True
    assert atoms.cell.rank == 0
    assert atoms.get_distance(0, 1) == pytest.approx(_BOND, abs=1e-4)
    np.testing.assert_allclose(atoms.positions, by_hand.positions - shift, atol=1e-9)


def test_a_box_leaves_the_padding_on_every_side_and_the_caller_untouched():
    """A centred cube, the structure's extent plus the padding each side; a copy, not a move."""
    atoms = _h2(0.74)
    box = _BOXING.boxed(atoms)

    assert box is not atoms and atoms.cell.rank == 0
    side = 0.74 + 2 * VACUUM_PADDING_ANGSTROM
    np.testing.assert_allclose(box.cell.lengths(), [side] * 3)
    np.testing.assert_allclose(box.positions.mean(axis=0), [side / 2] * 3)
    assert box.pbc.all()
    np.testing.assert_allclose(box.get_distance(0, 1), 0.74)


def test_a_structure_with_a_cell_or_a_molecular_library_is_not_boxed():
    """Only a periodic-only library boxes, and only a structure with no cell of its own."""
    periodic = _h2(0.74)
    periodic.set_cell([7.0, 7.0, 7.0])
    assert _BOXING.boxed(periodic) is periodic

    molecular = MlipLibrary(extra="mlip-mol", package="mol", import_name="mol")
    atoms = _h2(0.74)
    assert molecular.boxed(atoms) is atoms


def test_a_molecular_librarys_calculator_is_returned_as_built():
    """The wrapper exists for periodic-only libraries; any other calculator is untouched."""
    built = _NeedsALattice()
    MlipLibrary(extra="mlip-mol", package="mol", import_name="mol").calculator("molecular")(
        lambda _spec: built
    )
    assert build_calculator(task_name="molecular", device="cpu") is built


def test_a_periodic_only_calculator_asked_about_no_atoms_says_so():
    """With nothing ever attached, there is nothing to box — ASE's own setup error, named."""
    _periodic_only()
    calc = build_calculator(task_name="needs_a_lattice", device="cpu")
    with pytest.raises(CalculatorSetupError, match="no atoms to evaluate"):
        calc.get_potential_energy()


# ---------------------------------------------------------------------------
# The ExtOpt adapter hands the library the species the direct path does
# ---------------------------------------------------------------------------


class _SeesTheSpecies(Calculator):
    """Records the ``atoms.info`` each evaluation is handed; a flat, force-free surface."""

    def __init__(self, seen: list[dict]):
        Calculator.__init__(self)
        self.implemented_properties = ["energy", "forces"]
        self.seen = seen

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.seen.append(dict(self.atoms.info))
        self.results = {"energy": 0.0, "forces": np.zeros((len(self.atoms), 3))}


def test_the_extopt_adapter_stamps_the_species_the_direct_path_does():
    """Both paths reach a charge-aware library through the same ``atoms.info`` keys.

    The direct engine stamps the step's charge and multiplicity; the ExtOpt adapter stamps
    ORCA's per-call values. Keys spelled apart could drift, and a library reading the other
    spelling would fit an ion as a neutral singlet without a word.
    """
    from chemrefine.engines._backend_server.base import CalculationData
    from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator

    seen: list[dict] = []
    MlipLibrary(extra="mlip-species", package="species-lib", import_name="species_lib").calculator(
        "species"
    )(lambda _spec: _SeesTheSpecies(seen))

    MlipCalculator(task_name="species", device="cpu", charge=1, multiplicity=2).single_point(
        _h2(0.74)
    )
    MlipExtOptCalculator(model_name=None, task_name="species", device="cpu").calc(
        CalculationData(
            symbols=("H", "H"),
            positions_angstrom=_h2(0.74).positions,
            charge=1,
            multiplicity=2,
            nthreads=1,
            dograd=True,
            settings={},
        )
    )

    direct, extopt = seen
    assert direct == extopt == {"charge": 1, "spin": 2}
