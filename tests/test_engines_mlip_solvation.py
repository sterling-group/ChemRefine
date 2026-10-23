"""Tests for the implicit-solvation correction on MLIP steps.

See :mod:`chemrefine.engines.mlip.solvation`.

The solvated surface is the base potential plus the library's ΔE / ΔF, so every test here
runs a fake base calculator and either a fake solvation library or a stand-in ``anysolv``
package written to disk: no MLIP, no torch, no weights.
"""

from __future__ import annotations

import argparse
import ast
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from string import Template

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, CalculatorSetupError, all_changes

from chemrefine.engines._backend_server.base import CalculationData
from chemrefine.engines.mlip.calculator import MlipCalculator, build_calculator
from chemrefine.engines.mlip.engine import MlipEngine
from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator
from chemrefine.engines.mlip.options import MlipExtOptOptions, MlipOptions, MlipTrainOptions
from chemrefine.engines.mlip.registry import MlipLibrary, requirement_from_options
from chemrefine.engines.mlip.solvation import (
    AUTO,
    Correction,
    PythonRequirement,
    SolvationLibrary,
    SolvationSpec,
    WeightsMissing,
    build_correction,
    registered_solvation,
    solvation_library,
)
from chemrefine.engines.mlip.solvation._choose import choose_in_environment
from chemrefine.errors import ConfigError

_BOND = 0.74


class _Harmonic(Calculator):
    """A harmonic H2 bond at :data:`_BOND`, standing in for the gas-phase MLIP."""

    def __init__(self, periodic_only: bool = False):
        Calculator.__init__(self)
        self.implemented_properties = ["energy", "forces", "stress"]
        self.periodic_only = periodic_only

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if self.periodic_only and not self.atoms.cell.rank:
            raise np.linalg.LinAlgError("Singular matrix")
        bond = self.atoms.positions[1] - self.atoms.positions[0]
        length = float(np.linalg.norm(bond))
        pull = 2 * (length - _BOND) * bond / length
        self.results = {"energy": (length - _BOND) ** 2, "forces": np.array([pull, -pull])}


def _base(task: str = "harmonic", *, periodic_only: bool = False) -> str:
    """Register a base library under ``task`` whose builder returns :class:`_Harmonic`."""
    library = MlipLibrary(
        extra="mlip-harmonic",
        package="harmonic-lib",
        import_name="harmonic_lib",
        periodic_only=periodic_only,
        periodic_reason="it reads every geometry through its lattice" if periodic_only else "",
    )
    library.calculator(task)(lambda _spec: _Harmonic(periodic_only))
    return task


class _Seen:
    """What a fake correction was asked: one ``(atoms, charge, multiplicity)`` per call."""

    def __init__(self) -> None:
        self.calls: list[tuple[Atoms, int, int]] = []
        self.specs: list[SolvationSpec] = []


def _fake_library(seen: _Seen, *, name: str = "fake_solv") -> SolvationLibrary:
    """Register a correction: ΔE = -0.1·charge - 0.05·multiplicity, ΔF = 0.01 per Å of position."""
    library = SolvationLibrary(
        name=name,
        requirement=PythonRequirement(package="fake-solv", import_name="fake_solv"),
        models={"small": "fake_small", "big": "fake_big"},
        default_model="small",
        pairs_with=frozenset({"mlip-harmonic"}),
    )

    @library.correction
    def _build(spec: SolvationSpec):
        seen.specs.append(spec)

        def correction(atoms: Atoms, charge: int, multiplicity: int):
            seen.calls.append((atoms.copy(), charge, multiplicity))
            return -0.1 * charge - 0.05 * multiplicity, 0.01 * atoms.positions

        return correction

    return library


def _h2(length: float = 0.80) -> Atoms:
    return Atoms("H2", positions=[[1.0, 2.0, 3.0], [1.0 + length, 2.0, 3.0]])


# ---------------------------------------------------------------------------
# The solvated surface is the sum
# ---------------------------------------------------------------------------


def test_the_solvated_energy_and_forces_are_the_base_plus_the_correction():
    seen = _Seen()
    _fake_library(seen)
    task = _base()
    gas = MlipCalculator(task_name=task, charge=-1, multiplicity=2)
    solvated = MlipCalculator(
        task_name=task, charge=-1, multiplicity=2, solvation="fake_solv", solvent="water"
    )

    gas_energy, gas_gradient = gas.single_point(_h2())
    energy, gradient = solvated.single_point(_h2())

    assert energy == pytest.approx(gas_energy + 0.1 - 0.1)
    np.testing.assert_allclose(gradient, np.array(gas_gradient) - 0.01 * _h2().positions)
    ((_atoms, charge, multiplicity),) = seen.calls
    assert (charge, multiplicity) == (-1, 2)
    assert seen.specs == [SolvationSpec("fake_solv", "fake_small", "water", "cpu")]


def test_an_optimisation_walks_the_solvated_surface():
    """LBFGS lands where the sum is stationary, not where the base potential is."""
    seen = _Seen()
    library = SolvationLibrary(
        name="pull",
        requirement=PythonRequirement(package="pull", import_name="pull"),
        models={"m": "m"},
        default_model="m",
        pairs_with=frozenset({"mlip-harmonic"}),
    )
    # ΔE = s·r along the bond: the sum (r - r0)² + s·r is stationary at r0 - s/2.
    slope = 0.2

    @library.correction
    def _build(spec: SolvationSpec):
        def correction(atoms: Atoms, charge: int, multiplicity: int):
            seen.calls.append((atoms.copy(), charge, multiplicity))
            bond = atoms.positions[1] - atoms.positions[0]
            unit = bond / np.linalg.norm(bond)
            return slope * float(np.linalg.norm(bond)), np.array([slope * unit, -slope * unit])

        return correction

    mlip = MlipCalculator(
        task_name=_base(), charge=0, multiplicity=1, solvation="pull", solvent="water"
    )
    atoms = mlip.optimize(_h2(0.90), fmax=1e-5, steps=200)

    assert mlip.last_converged is True
    assert atoms.get_distance(0, 1) == pytest.approx(_BOND - slope / 2, abs=1e-4)


def test_the_correction_sees_the_molecule_not_the_vacuum_box():
    """Outside the box a periodic-only library computes in, so a cell never reaches it."""
    seen = _Seen()
    _fake_library(seen)
    calc = build_calculator(
        task_name=_base("lattice", periodic_only=True), solvation="fake_solv", solvent="water"
    )
    atoms = _h2()
    atoms.info.update(charge=0, spin=1)
    atoms.calc = calc

    atoms.get_potential_energy()

    ((seen_atoms, _charge, _multiplicity),) = seen.calls
    assert seen_atoms.cell.rank == 0 and not seen_atoms.pbc.any()
    np.testing.assert_array_equal(seen_atoms.positions, atoms.positions)
    assert "stress" not in calc.implemented_properties


def test_a_structure_with_no_charge_or_multiplicity_is_refused_not_assumed_neutral():
    _fake_library(_Seen())
    atoms = _h2()
    atoms.calc = build_calculator(task_name=_base(), solvation="fake_solv", solvent="water")
    with pytest.raises(CalculatorSetupError, match="charge and multiplicity"):
        atoms.get_potential_energy()


def test_the_extopt_adapter_hands_the_correction_orcas_species():
    """Each call's charge and multiplicity are ORCA's, read off ``atoms.info`` like the base's."""
    seen = _Seen()
    _fake_library(seen)
    calc = MlipExtOptCalculator(
        model_name=None,
        task_name=_base(),
        device="cpu",
        solvation="fake_solv",
        solvent="water",
    )
    calc.calc(
        CalculationData(
            symbols=("H", "H"),
            positions_angstrom=_h2().positions,
            charge=1,
            multiplicity=2,
            nthreads=1,
            dograd=True,
            settings={},
        )
    )
    ((_atoms, charge, multiplicity),) = seen.calls
    assert (charge, multiplicity) == (1, 2)


def test_unset_knobs_and_empty_placeholders_build_the_gas_phase_calculator():
    """An unset knob renders as an empty string in a template; that is gas phase."""
    seen = _Seen()
    _fake_library(seen)
    calc = build_calculator(task_name=_base(), solvation="", solvation_model="", solvent="")
    assert isinstance(calc, _Harmonic)
    assert seen.specs == []


# ---------------------------------------------------------------------------
# The knobs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"solvent": "water"}, "needs `solvation`"),
        ({"solvation_model": "small"}, "needs `solvation`"),
        ({"solvation": "fake_solv"}, "needs a `solvent`"),
        ({"solvation": "nope", "solvent": "water"}, "unknown solvation library 'nope'"),
        (
            {"solvation": "fake_solv", "solvation_model": "huge", "solvent": "water"},
            "has no model 'huge'",
        ),
    ],
)
def test_an_inconsistent_solvation_is_refused_before_the_run(options, message):
    _fake_library(_Seen())
    with pytest.raises(ConfigError, match=message):
        MlipOptions.from_raw(options)
    with pytest.raises(ConfigError, match=message):
        build_correction(
            solvation=options.get("solvation"),
            solvation_model=options.get("solvation_model"),
            solvent=options.get("solvent"),
            device="cpu",
        )


def test_the_librarys_default_model_is_written_into_the_options():
    """So the cache key names the model whether or not the YAML did."""
    _fake_library(_Seen())
    opts = MlipOptions.from_raw({"solvation": "fake_solv", "solvent": "water"})
    assert opts.solvation_model == "small"
    assert MlipOptions.from_raw({}).solvation_model is None


def test_unset_solvation_leaves_the_cache_key_as_it_was():
    """Gas-phase steps keep their keys; a solvated step keys all three knobs."""
    _fake_library(_Seen())
    gas = MlipOptions.from_raw({})
    assert {"solvation", "solvation_model", "solvent"} <= gas.key_exclusions()
    assert "solvation" not in gas.model_dump(exclude=gas.key_exclusions())

    solvated = MlipOptions.from_raw({"solvation": "fake_solv", "solvent": "water"})
    keyed = solvated.model_dump(exclude=solvated.key_exclusions())
    assert keyed["solvation"] == "fake_solv"
    assert keyed["solvation_model"] == "small"
    assert keyed["solvent"] == "water"


def test_a_training_step_refuses_the_solvation_knobs():
    _fake_library(_Seen())
    with pytest.raises(ConfigError, match="mlip-train fits a model"):
        MlipTrainOptions.from_raw(
            {"task_name": "omol", "device": "cpu", "solvation": "fake_solv", "solvent": "water"}
        )


def test_a_library_is_refused_with_a_backend_it_has_not_been_run_with():
    _fake_library(_Seen())
    MlipLibrary(extra="mlip-other", package="other", import_name="other").calculator("other")(
        lambda _spec: _Harmonic()
    )
    requirement_from_options({"task_name": _base(), "solvation": "fake_solv", "solvent": "water"})
    with pytest.raises(ConfigError, match="has not been run with the mlip-other backend"):
        requirement_from_options(
            {"task_name": "other", "solvation": "fake_solv", "solvent": "water"}
        )


def test_the_extopt_server_reads_back_the_solvation_it_was_given():
    """Options → server flags → parsed args → a solvated calculator, with nothing dropped."""
    seen = _Seen()
    _fake_library(seen)
    opts = MlipExtOptOptions.from_raw(
        {"task_name": _base(), "solvation": "fake_solv", "solvation_model": "big", "solvent": "thf"}
    )
    tokens = MlipExtOptCalculator.server_cli_from_options(opts.model_dump())
    assert tokens[tokens.index("--solvation") + 1] == "fake_solv"

    parser = argparse.ArgumentParser()
    MlipExtOptCalculator.add_cli_args(parser)
    MlipExtOptCalculator.from_args(parser.parse_args(tokens))

    assert seen.specs == [SolvationSpec("fake_solv", "fake_big", "thf", "cpu")]


def test_the_mlip_starter_hands_the_solvation_knobs_to_its_calculator():
    """A starter that dropped them would run a solvated step in the gas phase, unsaid."""
    values = {"SOLVATION": "anysolv", "SOLVATION_MODEL": "compact", "SOLVENT": "water"}
    # The contract comment is spliced in by text, directly before the first import.
    script = Template(MlipEngine.template_starter.replace("$OUTPUT_CONTRACT", ""))
    rendered = script.safe_substitute({n: values.get(n, 0) for n in script.get_identifiers()})
    (call,) = [
        node
        for node in ast.walk(ast.parse(rendered))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "MlipCalculator"
    ]
    passed = {kw.arg: kw.value.value for kw in call.keywords if isinstance(kw.value, ast.Constant)}
    assert {k: passed.get(k.lower()) for k in values} == values


# ---------------------------------------------------------------------------
# The AnySolv entry, against a stand-in package on disk
# ---------------------------------------------------------------------------

_FAKE_ANYSOLV = """
calls = []

def predict_solvation_energy(
    atoms,
    charge=0,
    spin=1,
    solvent="water",
    checkpoint=None,
    device="cpu",
    inference_settings="default",
):
    import numpy as np
    calls.append(
        dict(
            charge=charge,
            spin=spin,
            solvent=solvent,
            checkpoint=checkpoint,
            device=device,
            inference_settings=inference_settings,
        )
    )
    return -0.25, np.full((len(atoms), 3), 0.5, dtype=np.float32)
"""

_FAKE_SOLVENT = """
def resolve_solvent_name(name):
    key = name.strip().lower()
    if key in {"gas", "vacuum", "none", ""}:
        return ""
    return {"water": "water", "h2o": "water", "thf": "tetrahydrofuran"}.get(key)
"""


@pytest.fixture
def fake_anysolv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An importable ``anysolv`` with the API the entry uses; returns its package directory.

    It ships the compact weights only, and ``huggingface_hub`` is made unimportable, so a
    test that wants the MoE weights puts them in one place on purpose.
    """
    package = tmp_path / "site" / "anysolv"
    (package / "models").mkdir(parents=True)
    (package / "__init__.py").write_text(_FAKE_ANYSOLV, encoding="utf-8")
    (package / "solvent.py").write_text(_FAKE_SOLVENT, encoding="utf-8")
    (package / "models" / "__init__.py").write_text("", encoding="utf-8")
    (package / "models" / "model_compact.pt").write_bytes(b"")
    monkeypatch.syspath_prepend(str(package.parent))
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    for name in [m for m in sys.modules if m == "anysolv" or m.startswith("anysolv.")]:
        monkeypatch.delitem(sys.modules, name)
    yield package
    for name in [m for m in sys.modules if m == "anysolv" or m.startswith("anysolv.")]:
        del sys.modules[name]


def _hf_cache_holds(monkeypatch: pytest.MonkeyPatch, path: Path | None) -> list[tuple[str, str]]:
    """Stand in for ``huggingface_hub``'s cache lookup, holding ``path`` (or nothing)."""
    asked: list[tuple[str, str]] = []

    def try_to_load_from_cache(repo_id: str, filename: str) -> str | None:
        asked.append((repo_id, filename))
        return str(path) if path else None

    module = types.ModuleType("huggingface_hub")
    setattr(module, "try_to_load_from_cache", try_to_load_from_cache)  # noqa: B010
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return asked


def _anysolv(solvent: str, model: str = "compact", weights: str | None = None) -> Correction:
    correction = build_correction(
        solvation="anysolv",
        solvation_model=model,
        solvent=solvent,
        device="cuda",
        solvation_weights=weights,
    )
    assert correction is not None
    return correction


def _checkpoint_used(model: str, weights: str | None = None) -> str:
    """The checkpoint AnySolv is handed for ``model``, by one evaluation."""
    import anysolv

    _anysolv("water", model, weights)(_h2(), 0, 1)
    return str(anysolv.calls[-1]["checkpoint"])


def test_anysolv_is_registered_with_compact_as_its_default():
    assert "anysolv" in registered_solvation()
    opts = MlipOptions.from_raw({"solvation": "anysolv", "solvent": "water"})
    assert opts.solvation_model == "compact"


def test_anysolv_is_asked_with_the_multiplicity_as_its_spin(fake_anysolv: Path):
    import anysolv

    correction = _anysolv("H2O")
    delta_e, delta_f = correction(_h2(), -1, 2)

    assert delta_e == -0.25
    assert delta_f.dtype == np.float64 and delta_f.shape == (2, 3)
    compact = str(fake_anysolv / "models" / "model_compact.pt")
    asked = {"charge": -1, "spin": 2, "solvent": "water", "checkpoint": compact}
    # Unmerged: the MoE model mixes its experts per structure, so it serves every species.
    assert anysolv.calls == [{**asked, "device": "cuda", "inference_settings": "default"}]


@pytest.mark.parametrize("solvent", ["benzene-ish", "vacuum", "gas"])
def test_anysolv_refuses_a_solvent_it_has_no_descriptors_for(fake_anysolv: Path, solvent: str):
    """A gas-phase name too: AnySolv would read it as no solvent, and the step as solvated."""
    with pytest.raises(ConfigError, match=f"no solvent '{solvent}'"):
        _anysolv(solvent)


def test_a_missing_anysolv_names_the_library_to_install(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "anysolv", None)
    with pytest.raises(ImportError, match="needs 'anysolv' in the environment this step runs in"):
        _anysolv("water")


# ---------------------------------------------------------------------------
# Where the MoE weights are found
# ---------------------------------------------------------------------------


def test_moe_weights_named_by_the_step_are_used_first(
    fake_anysolv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Ahead of the Hugging Face cache and the package, both of which also hold a copy."""
    named = tmp_path / "weights" / "model_moe.pt"
    named.parent.mkdir()
    named.write_bytes(b"")
    (fake_anysolv / "models" / "model_moe.pt").write_bytes(b"")
    asked = _hf_cache_holds(monkeypatch, tmp_path / "hf" / "model_moe.pt")

    assert _checkpoint_used("moe", str(named)) == str(named)
    assert asked == []


def test_moe_weights_come_from_the_hugging_face_cache_next(
    fake_anysolv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (fake_anysolv / "models" / "model_moe.pt").write_bytes(b"")
    cached = tmp_path / "hf" / "model_moe.pt"
    asked = _hf_cache_holds(monkeypatch, cached)

    assert _checkpoint_used("moe") == str(cached)
    assert asked == [("antonknee/anysolv", "model_moe.pt")]


def test_moe_weights_in_the_package_are_used_last(
    fake_anysolv: Path, monkeypatch: pytest.MonkeyPatch
):
    _hf_cache_holds(monkeypatch, None)
    packaged = fake_anysolv / "models" / "model_moe.pt"
    packaged.write_bytes(b"")

    assert _checkpoint_used("moe") == str(packaged)


def test_moe_weights_in_none_of_the_places_say_where_they_were_looked_for(fake_anysolv: Path):
    with pytest.raises(WeightsMissing) as refused:
        _anysolv("water", "moe")
    said = str(refused.value)
    assert "no `solvation_weights` named" in said
    assert "huggingface_hub is not installed" in said
    assert "hf download antonknee/anysolv model_moe.pt" in said


def test_a_weights_file_that_is_not_there_is_a_mistake_not_a_missing_model(
    fake_anysolv: Path, tmp_path: Path
):
    """Under ``auto`` too: a mistyped path must not quietly become ``compact``."""
    nowhere = str(tmp_path / "typo" / "model_moe.pt")
    for model in ("moe", AUTO):
        with pytest.raises(ConfigError, match="is not a file") as refused:
            _anysolv("water", model, nowhere)
        assert not isinstance(refused.value, WeightsMissing)


# ---------------------------------------------------------------------------
# solvation_model: auto
# ---------------------------------------------------------------------------


def test_auto_chooses_moe_where_its_weights_are(fake_anysolv: Path):
    (fake_anysolv / "models" / "model_moe.pt").write_bytes(b"")
    assert solvation_library("anysolv").choose(None) == ("moe", ())


def test_auto_falls_back_to_compact_and_says_why(fake_anysolv: Path):
    chosen, skipped = solvation_library("anysolv").choose(None)
    assert chosen == "compact"
    (why,) = skipped
    assert why.startswith("moe: anysolv's model_moe weights:")


def test_auto_built_outside_a_run_chooses_in_its_own_environment(fake_anysolv: Path):
    """A ``step{N}.py`` run by hand: no run chose for it, so the calculator does."""
    assert _checkpoint_used(AUTO) == str(fake_anysolv / "models" / "model_compact.pt")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"solvation_model": "auto"}, "needs `solvation`"),
        ({"solvation_weights": "/w.pt"}, "needs `solvation`"),
        (
            {"solvation": "anysolv", "solvation_model": "compact", "solvation_weights": "/w.pt"},
            "reads none",
        ),
        ({"solvation": "fake_solv", "solvation_model": "auto"}, "offers no `solvation_model"),
    ],
)
def test_auto_and_weights_are_refused_where_they_mean_nothing(options, message):
    _fake_library(_Seen())
    with pytest.raises(ConfigError, match=message):
        MlipOptions.from_raw({"solvent": "water", **options})


@pytest.mark.parametrize("model", ["moe", "auto"])
def test_weights_are_accepted_for_a_model_that_reads_them(model: str):
    opts = MlipOptions.from_raw(
        {
            "solvation": "anysolv",
            "solvation_model": model,
            "solvent": "water",
            "solvation_weights": "/w.pt",
        }
    )
    assert opts.solvation_model == model and opts.solvation_weights == "/w.pt"


def test_the_step_environment_is_asked_through_its_own_interpreter(
    fake_anysolv: Path, monkeypatch: pytest.MonkeyPatch
):
    """A real subprocess, as the run asks before submitting: the answer is that env's."""
    monkeypatch.setenv("PYTHONPATH", str(fake_anysolv.parent))
    chosen, skipped = choose_in_environment(sys.executable, "anysolv", None)
    assert chosen == "compact"
    assert len(skipped) == 1 and skipped[0].startswith("moe:")

    (fake_anysolv / "models" / "model_moe.pt").write_bytes(b"")
    assert choose_in_environment(sys.executable, "anysolv", None) == ("moe", ())


def test_an_environment_without_the_library_is_refused_before_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))  # nothing named anysolv here
    with pytest.raises(ConfigError, match="anysolv is not installed in this environment"):
        choose_in_environment(sys.executable, "anysolv", None)


def test_the_mlip_engines_choose_auto_and_nothing_else(monkeypatch: pytest.MonkeyPatch, caplog):
    asked: list[tuple[str, str, str | None]] = []

    def choose(launcher: str, solvation: str, weights: str | None):
        asked.append((launcher, solvation, weights))
        return "compact", ("moe: not here",)

    monkeypatch.setattr("chemrefine.engines.mlip.solvation._choose.choose_in_environment", choose)
    engine = MlipEngine()
    solvated = {"task_name": "omol", "solvation": "anysolv", "solvent": "water"}

    assert engine.resolve_options(solvated, "/env/bin/python", None) == {}
    moe = {**solvated, "solvation_model": "moe"}
    assert engine.resolve_options(moe, "/env/bin/python", None) == {}
    with caplog.at_level("WARNING"):
        chosen = engine.resolve_options(
            {**solvated, "solvation_model": "auto", "solvation_weights": "/w.pt"},
            "/env/bin/python",
            None,
        )
    assert chosen == {"solvation_model": "compact"}
    assert asked == [("/env/bin/python", "anysolv", "/w.pt")]
    assert "passed over moe: not here" in caplog.text


def _environment_has(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    """The step's environment, asked before the run, answers ``model``."""
    skipped = () if model == "moe" else ("moe: anysolv's model_moe weights: not found",)
    monkeypatch.setattr(
        "chemrefine.engines.mlip.solvation._choose.choose_in_environment",
        lambda launcher, solvation, weights: (model, skipped),
    )


_AUTO = {"task_name": "omol", "solvation": "anysolv", "solvation_model": "auto", "solvent": "water"}


def test_a_step_that_ran_on_moe_is_refused_rather_than_recomputed_on_compact(
    monkeypatch: pytest.MonkeyPatch,
):
    """The weights went missing since: falling back would replace moe results with compact's."""
    _environment_has(monkeypatch, "compact")
    with pytest.raises(ConfigError) as refused:
        MlipEngine().resolve_options(_AUTO, "/env/bin/python", {"solvation_model": "moe"})
    said = str(refused.value)
    assert "chose moe when this step last ran" in said
    assert "model_moe weights: not found" in said
    assert "`solvation_model: compact`" in said and "`chemrefine rerun`" in said


@pytest.mark.parametrize(
    ("before", "now"),
    [(None, "compact"), ("compact", "compact"), ("moe", "moe"), ("compact", "moe")],
)
def test_a_first_run_the_same_choice_or_a_better_one_goes_ahead(
    monkeypatch: pytest.MonkeyPatch, before: str | None, now: str
):
    """Weights that have appeared since a compact run recompute the step on moe."""
    _environment_has(monkeypatch, now)
    previous = {"solvation_model": before} if before else None
    assert MlipEngine().resolve_options(_AUTO, "/env/bin/python", previous) == {
        "solvation_model": now
    }
