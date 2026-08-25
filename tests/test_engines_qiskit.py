"""Tests for the modular Qiskit engine without requiring the optional Qiskit stack.

The base CI environment installs ``chemrefine[test]``, not ``chemrefine[qiskit]``.
Consequently these tests exercise orchestration with registered fakes and exercise the
built-in lazy factories with a temporary in-memory module tree.  A real Qiskit run belongs
to the opt-in live tier; the parser contract below it is pinned separately under
``tests/data/engines/qiskit``.
"""

from __future__ import annotations

import ast
import json
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from ase import Atoms
from pydantic import BaseModel, ConfigDict, Field

from chemrefine.config import StepConfig, load_config
from chemrefine.engines import known_backend_extras, preflight_backends
from chemrefine.engines.api import BackendRequirement, ProvisionableEngine, get_engine
from chemrefine.engines.qiskit import workflow
from chemrefine.engines.qiskit.components.algorithms import (
    AdaptVQEOptions,
    _require,
    build_adapt_vqe,
    build_exact,
    build_vqe,
)
from chemrefine.engines.qiskit.components.ansatze import (
    EfficientSU2Options,
    UCCSDOptions,
    build_efficient_su2,
    build_uccsd,
)
from chemrefine.engines.qiskit.components.estimators import (
    AerShotsEstimatorOptions,
    AerStatevectorEstimatorOptions,
    BasicBackendEstimatorOptions,
    StatevectorEstimatorOptions,
    build_aer_shots_estimator,
    build_aer_statevector_estimator,
    build_basic_backend_estimator,
    build_statevector_estimator,
)
from chemrefine.engines.qiskit.components.initial_points import (
    RandomInitialPointOptions,
    _parameter_count,
    build_random_initial_point,
    build_zero_initial_point,
)
from chemrefine.engines.qiskit.components.initial_states import (
    build_hartree_fock,
    build_zero_state,
)
from chemrefine.engines.qiskit.components.mappers import (
    ParityMapperOptions,
    build_jordan_wigner_mapper,
    build_parity_mapper,
)
from chemrefine.engines.qiskit.components.optimizers import (
    COBYLAOptions,
    SLSQPOptions,
    SPSAOptions,
    build_cobyla,
    build_slsqp,
    build_spsa,
)
from chemrefine.engines.qiskit.context import (
    AlgorithmArtifacts,
    AnsatzArtifacts,
    ElectronicStructureContext,
    EstimatorResource,
    SolverComponents,
)
from chemrefine.engines.qiskit.options import (
    ActiveSpaceOptions,
    ComponentSelection,
    QiskitOptions,
)
from chemrefine.engines.qiskit.registry import (
    ALGORITHMS,
    ANSATZE,
    ESTIMATORS,
    INITIAL_POINTS,
    INITIAL_STATES,
    OPTIMIZERS,
    ComponentRegistry,
    ComponentSpec,
    NoComponentOptions,
    validate_component_graph,
)
from chemrefine.errors import ConfigError
from chemrefine.state import PipelineState, StepContext, Structure


class _ValueOptions(BaseModel):
    """Small strict component-options model used by registry tests."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: int = Field(1, ge=1)


def _seed() -> Structure:
    """Return the H2 seed used by engine rendering tests."""
    return Structure(
        id="0",
        atoms=Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.735]]),
    )


def _ctx(tmp_path: Path, *, options: dict[str, Any] | None = None) -> StepContext:
    """Build a minimal Qiskit context with the real thin-runner template shape."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    template = template_dir / "step1.py"
    template.write_text(
        "import json\n"
        "from chemrefine.engines.qiskit.workflow import run_job\n"
        "result = run_job(\n"
        '    "$XYZ_PATH", charge=int("$CHARGE"),\n'
        '    multiplicity=int("$MULTIPLICITY"),\n'
        '    options=json.loads("$QISKIT_OPTIONS_JSON"),\n'
        ")\n"
        "energy_hartree = result.energy_hartree\n"
        "engine_metadata = result.metadata\n",
        encoding="utf-8",
    )
    (template_dir / "cpu.slurm.header").write_text("#!/bin/bash\n", encoding="utf-8")
    step_cfg = StepConfig(
        step=1,
        engine="qiskit",
        operation="sp",
        options=options or {},
    )
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=template_dir,
        template=template,
        scratch_dir=None,
        prev_state=PipelineState(structures=(_seed(),)),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        executables={},
    )


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attributes: object) -> ModuleType:
    """Install a temporary importable module and all of its parent packages."""
    parts = name.split(".")
    parent: ModuleType | None = None
    for index in range(1, len(parts) + 1):
        module_name = ".".join(parts[:index])
        module = sys.modules.get(module_name)
        if not isinstance(module, ModuleType):
            module = ModuleType(module_name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, module_name, module)
        if parent is not None:
            monkeypatch.setattr(parent, parts[index - 1], module, raising=False)
        parent = module
    assert parent is not None
    for key, value in attributes.items():
        monkeypatch.setattr(parent, key, value, raising=False)
    return parent


def _recording_class(name: str, *, num_parameters: int = 3) -> type:
    """Return a constructor fake that records all calls on its class."""

    class Recording:
        calls: ClassVar[list[tuple[tuple[object, ...], dict[str, object]]]] = []

        def __init__(self, *args: object, **kwargs: object) -> None:
            type(self).calls.append((args, kwargs))
            self.args = args
            self.kwargs = kwargs
            self.num_parameters = num_parameters

    Recording.__name__ = name
    return Recording


def _fake_qiskit_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Install the Qiskit names imported lazily by every built-in factory."""
    quantum_circuit = _recording_class("QuantumCircuit", num_parameters=0)
    efficient_su2 = _recording_class("EfficientSU2", num_parameters=5)
    statevector_estimator = _recording_class("StatevectorEstimator")
    backend_estimator = _recording_class("BackendEstimatorV2")
    numpy_solver = _recording_class("NumPyMinimumEigensolver")
    vqe = _recording_class("VQE")
    adapt_vqe = _recording_class("AdaptVQE")
    slsqp = _recording_class("SLSQP")
    cobyla = _recording_class("COBYLA")
    spsa = _recording_class("SPSA")
    hartree_fock = _recording_class("HartreeFock")
    jordan_wigner = _recording_class("JordanWignerMapper")
    parity = _recording_class("ParityMapper")
    algorithm_globals = SimpleNamespace(random_seed=None)

    class AerError(Exception):
        """Fake of the public Aer configuration exception."""

    class AerSimulator:
        calls: ClassVar[list[dict[str, object]]] = []
        available: ClassVar[tuple[str, ...]] = ("CPU", "GPU")
        methods: ClassVar[tuple[str, ...]] = (
            "automatic",
            "statevector",
            "density_matrix",
            "matrix_product_state",
            "tensor_network",
        )
        fail: ClassVar[bool] = False

        def __init__(self, **kwargs: object) -> None:
            if type(self).fail:
                raise AerError("simulator rejected its configuration")
            type(self).calls.append(kwargs)
            self.kwargs = kwargs

        def available_devices(self) -> tuple[str, ...]:
            return type(self).available

        def available_methods(self) -> tuple[str, ...]:
            return type(self).methods

    class AerEstimatorV2:
        calls: ClassVar[list[tuple[object, dict[str, object]]]] = []

        @classmethod
        def from_backend(cls, backend: object, **kwargs: object) -> AerEstimatorV2:
            cls.calls.append((backend, kwargs))
            instance = cls()
            instance.backend = backend
            instance.kwargs = kwargs
            return instance

    class NoiseModel:
        calls: ClassVar[list[dict[str, object]]] = []

        @classmethod
        def from_dict(cls, value: dict[str, object]) -> object:
            warnings.warn("from_dict has been deprecated by Aer", DeprecationWarning, stacklevel=2)
            cls.calls.append(value)
            if value.get("invalid"):
                raise AerError("bad serialized noise")
            return SimpleNamespace(serialized=value)

    class UCCSD:
        calls: ClassVar[list[tuple[tuple[object, ...], dict[str, object]]]] = []

        def __init__(self, *args: object, **kwargs: object) -> None:
            type(self).calls.append((args, kwargs))
            self.args = args
            self.kwargs = kwargs
            self.num_parameters = 4
            self.operators = ["mapped:excitation-a", "mapped:excitation-b"]

    class BasicProvider:
        instances: ClassVar[list[BasicProvider]] = []

        def __init__(self) -> None:
            self.requested: list[str] = []
            type(self).instances.append(self)

        def get_backend(self, name: str) -> str:
            self.requested.append(name)
            return f"backend:{name}"

    pass_manager_calls: list[dict[str, object]] = []

    def generate_preset_pass_manager(**kwargs: object) -> str:
        pass_manager_calls.append(kwargs)
        return "preset-pass-manager"

    _install_module(monkeypatch, "qiskit", QuantumCircuit=quantum_circuit)
    _install_module(monkeypatch, "qiskit.circuit.library", EfficientSU2=efficient_su2)
    _install_module(
        monkeypatch,
        "qiskit.primitives",
        StatevectorEstimator=statevector_estimator,
        BackendEstimatorV2=backend_estimator,
    )
    _install_module(monkeypatch, "qiskit.providers.basic_provider", BasicProvider=BasicProvider)
    _install_module(
        monkeypatch,
        "qiskit.transpiler.preset_passmanagers",
        generate_preset_pass_manager=generate_preset_pass_manager,
    )
    _install_module(
        monkeypatch,
        "qiskit_aer",
        AerError=AerError,
        AerSimulator=AerSimulator,
    )
    _install_module(monkeypatch, "qiskit_aer.primitives", EstimatorV2=AerEstimatorV2)
    _install_module(monkeypatch, "qiskit_aer.noise", NoiseModel=NoiseModel)
    _install_module(
        monkeypatch,
        "qiskit_algorithms",
        NumPyMinimumEigensolver=numpy_solver,
        VQE=vqe,
        AdaptVQE=adapt_vqe,
    )
    _install_module(
        monkeypatch,
        "qiskit_algorithms.optimizers",
        SLSQP=slsqp,
        COBYLA=cobyla,
        SPSA=spsa,
    )
    _install_module(monkeypatch, "qiskit_algorithms.utils", algorithm_globals=algorithm_globals)
    _install_module(
        monkeypatch,
        "qiskit_nature.second_q.circuit.library",
        HartreeFock=hartree_fock,
        UCCSD=UCCSD,
    )
    _install_module(
        monkeypatch,
        "qiskit_nature.second_q.mappers",
        JordanWignerMapper=jordan_wigner,
        ParityMapper=parity,
    )
    return SimpleNamespace(
        QuantumCircuit=quantum_circuit,
        EfficientSU2=efficient_su2,
        StatevectorEstimator=statevector_estimator,
        BackendEstimatorV2=backend_estimator,
        AerError=AerError,
        AerSimulator=AerSimulator,
        AerEstimatorV2=AerEstimatorV2,
        NoiseModel=NoiseModel,
        NumPyMinimumEigensolver=numpy_solver,
        VQE=vqe,
        AdaptVQE=adapt_vqe,
        SLSQP=slsqp,
        COBYLA=cobyla,
        SPSA=spsa,
        algorithm_globals=algorithm_globals,
        HartreeFock=hartree_fock,
        UCCSD=UCCSD,
        BasicProvider=BasicProvider,
        pass_manager_calls=pass_manager_calls,
        JordanWignerMapper=jordan_wigner,
        ParityMapper=parity,
    )


def _context() -> ElectronicStructureContext:
    """Return a dependency-free electronic-structure context for factory tests."""

    class Mapper:
        def map(self, operator: object) -> str:
            return f"mapped:{operator}"

    problem = SimpleNamespace(get_default_filter_criterion=lambda: "filter")
    return ElectronicStructureContext(
        problem=problem,
        mapper=Mapper(),
        qubit_hamiltonian=SimpleNamespace(num_qubits=4),
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        num_qubits=4,
        multiplicity=1,
    )


# ---------------------------------------------------------------------------
# Options and component graph validation
# ---------------------------------------------------------------------------


def test_component_selection_normalizes_names_and_supports_shorthand() -> None:
    selection = ComponentSelection(name="  Efficient-SU2 ", options={"reps": 2})
    assert selection.name == "efficient_su2"
    assert ComponentSelection.named("VQE").name == "vqe"

    options = QiskitOptions(
        mapper="parity",
        algorithm="vqe",
        ansatz="efficient-su2",
        initial_state="zero",
        estimator="basic-backend",
        optimizer="cobyla",
        initial_point="random",
    )
    assert {name: selected.name for name, selected in options.component_selections().items()} == {
        "mapper": "parity",
        "algorithm": "vqe",
        "ansatz": "efficient_su2",
        "initial_state": "zero",
        "estimator": "basic_backend",
        "optimizer": "cobyla",
        "initial_point": "random",
    }
    with pytest.raises(ValueError, match="valid string"):
        ComponentSelection.model_validate({"name": 1})


@pytest.mark.parametrize(
    "electrons",
    [0, (-1, 2), (0, 0)],
)
def test_active_space_rejects_invalid_electron_populations(electrons: object) -> None:
    with pytest.raises(ValueError, match="electrons"):
        ActiveSpaceOptions(electrons=electrons, orbitals=2)


def test_active_space_accepts_total_and_spin_resolved_counts() -> None:
    assert ActiveSpaceOptions(electrons=2).electrons == 2
    assert ActiveSpaceOptions(electrons=(1, 1)).electrons == (1, 1)
    with pytest.raises(ValueError, match="do not fit"):
        ActiveSpaceOptions(electrons=3, orbitals=1)
    with pytest.raises(ValueError, match="do not fit"):
        ActiveSpaceOptions(electrons=(2, 0), orbitals=1)


def test_qiskit_options_are_strict_at_the_engine_boundary() -> None:
    with pytest.raises(ConfigError, match="unknown_knob"):
        QiskitOptions.from_raw({"unknown_knob": True})
    with pytest.raises(ValueError, match="valid dictionary"):
        QiskitOptions.model_validate("not-a-mapping")


def test_registry_validates_builds_and_lists_components() -> None:
    registry = ComponentRegistry("demo")

    @registry.register(" Thing-ONE ", _ValueOptions, capabilities=frozenset({"circuit"}))
    def build(*, options: _ValueOptions, marker: str) -> tuple[int, str]:
        return options.value, marker

    selection = ComponentSelection(name="thing_one", options={"value": 4})
    assert registry.names() == frozenset({"thing_one"})
    assert registry.spec("thing_one").capabilities == frozenset({"circuit"})
    assert registry.options_for(selection) == _ValueOptions(value=4)
    assert registry.build(selection, marker="seen") == (4, "seen")

    # Duplicate keys are rejected even when the original builder is supplied again.
    with pytest.raises(ValueError, match="already registered"):
        registry.register("thing_one", _ValueOptions)(build)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("thing_one")(lambda **_kwargs: None)


def test_registry_reports_unknown_names_and_bad_component_options() -> None:
    registry = ComponentRegistry("demo")
    registry.register("known", _ValueOptions)(lambda **kwargs: kwargs)

    with pytest.raises(ConfigError, match=r"unsupported qiskit demo.*known"):
        registry.spec("missing")
    with pytest.raises(ConfigError, match=r"(?s)invalid qiskit demo options.*value"):
        registry.options_for(ComponentSelection(name="known", options={"value": 0}))


def test_component_graph_rejects_an_ansatz_without_the_required_capability() -> None:
    options = QiskitOptions(algorithm="adapt_vqe", ansatz="efficient_su2")
    with pytest.raises(ConfigError, match=r"operator_pool.*efficient_su2"):
        validate_component_graph(options)


def test_component_graph_rejects_repeated_uccsd_for_adapt_vqe() -> None:
    options = QiskitOptions(
        algorithm="adapt_vqe",
        ansatz={"name": "uccsd", "options": {"reps": 2}},
    )
    with pytest.raises(ConfigError, match=r"set ansatz\.options\.reps to 1"):
        validate_component_graph(options)


def test_component_graph_rejects_unknown_components_and_accepts_defaults() -> None:
    validate_component_graph(QiskitOptions())
    assert ESTIMATORS.names() >= {
        "statevector",
        "basic_backend",
        "aer_statevector",
        "aer_shots",
    }
    with pytest.raises(ConfigError, match="unsupported qiskit estimator"):
        validate_component_graph(QiskitOptions(estimator="not_registered"))


@pytest.mark.parametrize(
    "estimator, match",
    [
        ({"name": "aer_shots", "options": {"default_precision": 0}}, "greater than 0"),
        ({"name": "aer_shots", "options": {"method": "unitary"}}, "method"),
        (
            {"name": "aer_statevector", "options": {"optimization_level": 4}},
            "less than or equal to 3",
        ),
        ({"name": "aer_statevector", "options": {"unknown": True}}, "unknown"),
    ],
)
def test_aer_estimator_options_are_strict(estimator: dict[str, object], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        validate_component_graph(QiskitOptions(algorithm="vqe", estimator=estimator))


def test_component_graph_enforces_device_and_aer_method_compatibility() -> None:
    with pytest.raises(ConfigError, match="algorithm 'exact' is CPU-only"):
        validate_component_graph(QiskitOptions(device="cuda"))
    with pytest.raises(ConfigError, match=r"estimator 'statevector'.*device: cuda"):
        validate_component_graph(
            QiskitOptions(device="cuda", algorithm="vqe", estimator="statevector")
        )
    with pytest.raises(ConfigError, match=r"tensor_network.*device: cuda"):
        validate_component_graph(
            QiskitOptions(
                algorithm="vqe",
                estimator={"name": "aer_shots", "options": {"method": "tensor_network"}},
            )
        )
    with pytest.raises(ConfigError, match=r"matrix_product_state.*not GPU-compatible"):
        validate_component_graph(
            QiskitOptions(
                device="cuda",
                algorithm="vqe",
                estimator={
                    "name": "aer_shots",
                    "options": {"method": "matrix_product_state"},
                },
            )
        )

    validate_component_graph(
        QiskitOptions(device="cuda", algorithm="vqe", estimator="aer_statevector")
    )
    validate_component_graph(QiskitOptions(device="cuda", algorithm="vqe", estimator="aer_shots"))


# ---------------------------------------------------------------------------
# Engine contract, rendering, and provisioning
# ---------------------------------------------------------------------------


def test_qiskit_engine_is_registered_and_provisionable() -> None:
    import tomllib

    engine = get_engine("qiskit")
    assert engine.name == "qiskit"
    assert isinstance(engine, ProvisionableEngine)
    assert engine.backend_requirement(None).extra == "qiskit"
    assert engine.backend_requirement({"algorithm": "vqe"}).import_name == "qiskit_nature"
    aer = {
        "algorithm": "vqe",
        "estimator": {"name": "aer_shots", "options": {"default_precision": 0.1}},
    }
    assert engine.backend_requirement(aer).extra == "qiskit-aer"
    assert engine.backend_requirement(aer).import_name == "qiskit_aer"
    assert engine.backend_requirement({**aer, "algorithm": "exact"}).extra == "qiskit"
    assert engine.backend_extras() == frozenset({"qiskit", "qiskit-aer"})
    assert {"qiskit", "qiskit-aer"} <= known_backend_extras()
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert "qiskit-aer>=0.17,<0.18" in extras["qiskit-aer"]
    assert "chemrefine[qiskit]" in extras["qiskit-aer"]


def test_shipped_qiskit_template_is_valid_python() -> None:
    template = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "tutorials"
        / "qiskit_sp"
        / "templates"
        / "step1.py"
    )
    source = template.read_text(encoding="utf-8")

    assert "$QISKIT_OPTIONS_JSON" in source
    ast.parse(source)


def test_prepare_renders_a_complete_json_job_spec(tmp_path: Path) -> None:
    options = {
        "basis": "sto-3g",
        "active_space": {"electrons": [1, 1], "orbitals": 2},
        "algorithm": "vqe",
        "ansatz": {"name": "uccsd", "options": {"reps": 2}},
        "estimator": {"name": "statevector", "options": {"seed": 7}},
        "optimizer": "slsqp",
        "initial_point": "zeros",
    }
    ctx = _ctx(tmp_path, options=options)
    engine = get_engine("qiskit")

    ((script, output, sid),) = engine.prepare(ctx).files

    assert sid == "0"
    assert output.name == "step1_0.json"
    rendered = script.read_text(encoding="utf-8")
    assert "$QISKIT_OPTIONS_JSON" not in rendered
    assert "$XYZ_PATH" not in rendered
    assert "engine_metadata" in rendered
    tree = ast.parse(rendered)
    loads_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "json"
        and node.func.attr == "loads"
    ]
    assert len(loads_calls) == 1
    rendered_options = json.loads(ast.literal_eval(loads_calls[0].args[0]))
    assert rendered_options["algorithm"] == {"name": "vqe", "options": {}}
    assert rendered_options["estimator"] == {
        "name": "statevector",
        "options": {"seed": 7},
    }


def test_engine_vars_escape_json_so_component_text_cannot_break_python() -> None:
    options = QiskitOptions(
        basis='sto-"quotes"\\backslash\n$XYZ_PATH',
        ansatz={"name": "uccsd", "options": {"generalized": True}},
        estimator={"name": "statevector", "options": {"seed": 3}},
    )
    value = get_engine("qiskit")._vars_from(options)["QISKIT_OPTIONS_JSON"]
    assignment = ast.parse(f'payload = "{value}"\n').body[0]
    assert isinstance(assignment, ast.Assign)
    payload = ast.literal_eval(assignment.value)
    assert json.loads(payload) == options.as_job_spec()


def test_prepare_fails_before_submission_for_an_unknown_component(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, options={"algorithm": "not_registered"})
    with pytest.raises(ConfigError, match="unsupported qiskit algorithm"):
        get_engine("qiskit").prepare(ctx)


def test_qiskit_run_block_uses_its_managed_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path / "home"))
    python = tmp_path / "home" / "backends" / "qiskit" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    ctx = _ctx(tmp_path)

    body = get_engine("qiskit").run_block(ctx, Path("step1_0.py"), Path("step1_0.json")).body

    assert body.splitlines()[-1] == f"{python} step1_0.py"


def test_aer_run_block_uses_its_provider_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path / "home"))
    python = tmp_path / "home" / "backends" / "qiskit-aer" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    ctx = _ctx(tmp_path, options={"algorithm": "vqe", "estimator": "aer_shots"})

    body = get_engine("qiskit").run_block(ctx, Path("step1_0.py"), Path("step1_0.json")).body

    assert body.splitlines()[-1] == f"{python} step1_0.py"


def test_qiskit_preflight_fails_actionably_when_the_backend_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "chemrefine.engines._provision.importlib.util.find_spec", lambda _name: None
    )
    step = StepConfig(step=1, engine="qiskit", operation="sp")
    with pytest.raises(ConfigError, match="backends install qiskit"):
        preflight_backends([step])


def test_aer_preflight_names_the_provider_environment_when_aer_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CHEMREFINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "chemrefine.engines._provision.importlib.util.find_spec", lambda _name: None
    )
    step = StepConfig(
        step=1,
        engine="qiskit",
        operation="sp",
        options={"algorithm": "vqe", "estimator": "aer_statevector"},
    )
    with pytest.raises(ConfigError, match="backends install qiskit-aer"):
        preflight_backends([step])


@pytest.mark.parametrize("filename", ["input.yaml", "aer_statevector.yaml", "aer_shots.yaml"])
def test_shipped_qiskit_examples_load_through_the_real_config_schema(filename: str) -> None:
    repo = Path(__file__).resolve().parent.parent
    cfg = load_config(repo / "examples" / "tutorials" / "qiskit_sp" / filename)
    assert cfg.steps[0].engine == "qiskit"
    resolved = QiskitOptions.from_raw(cfg.steps[0].options)
    assert resolved.algorithm.name == "vqe"
    validate_component_graph(resolved)


# ---------------------------------------------------------------------------
# Lazy built-in factories, using an in-memory fake Qiskit module tree
# ---------------------------------------------------------------------------


def test_lazy_mapper_and_initial_state_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_qiskit_modules(monkeypatch)
    context = _context()
    problem = SimpleNamespace(num_particles=(1, 1))

    jw = build_jordan_wigner_mapper(options=NoComponentOptions(), problem=problem)
    reduced = build_parity_mapper(options=ParityMapperOptions(), problem=problem)
    unreduced = build_parity_mapper(
        options=ParityMapperOptions(two_qubit_reduction=False), problem=problem
    )
    hf = build_hartree_fock(options=NoComponentOptions(), context=context)
    zero = build_zero_state(options=NoComponentOptions(), context=context)

    assert isinstance(jw, fake.JordanWignerMapper)
    assert reduced.kwargs["num_particles"] == (1, 1)
    assert unreduced.kwargs["num_particles"] is None
    assert hf.args == (2, (1, 1), context.mapper)
    assert zero.args == (4,)


def test_lazy_ansatz_and_initial_point_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_qiskit_modules(monkeypatch)
    context = _context()
    initial_state = object()

    ucc = build_uccsd(
        options=UCCSDOptions(reps=2, generalized=True, preserve_spin=False, include_imaginary=True),
        context=context,
        initial_state=initial_state,
    )
    efficient = build_efficient_su2(
        options=EfficientSU2Options(
            reps=3,
            entanglement="linear",
            su2_gates=["rx"],
            skip_final_rotation_layer=True,
            flatten=False,
        ),
        context=context,
        initial_state=initial_state,
    )

    assert isinstance(ucc.circuit, fake.UCCSD)
    assert ucc.operator_pool == (
        "mapped:excitation-a",
        "mapped:excitation-b",
    )
    assert isinstance(efficient.circuit, fake.EfficientSU2)
    assert efficient.operator_pool is None
    assert efficient.circuit.kwargs["initial_state"] is initial_state

    zeros = build_zero_initial_point(options=NoComponentOptions(), ansatz=ucc)
    random_a = build_random_initial_point(
        options=RandomInitialPointOptions(seed=11, scale=0.2), ansatz=ucc
    )
    random_b = build_random_initial_point(
        options=RandomInitialPointOptions(seed=11, scale=0.2), ansatz=ucc
    )
    assert np.array_equal(zeros, np.zeros(4))
    assert np.array_equal(random_a, random_b)
    assert np.all(np.abs(random_a) <= 0.2)
    assert _parameter_count(ucc) == 4
    with pytest.raises(ConfigError, match="fixed ansatz circuit"):
        _parameter_count(AnsatzArtifacts())


def test_lazy_estimator_and_optimizer_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_qiskit_modules(monkeypatch)

    statevector = build_statevector_estimator(
        options=StatevectorEstimatorOptions(default_precision=0.0, seed=5)
    )
    backend = build_basic_backend_estimator(
        options=BasicBackendEstimatorOptions(
            backend_name="basic_simulator",
            default_precision=0.125,
            abelian_grouping=False,
            seed_simulator=9,
        )
    )
    aer_statevector = build_aer_statevector_estimator(
        options=AerStatevectorEstimatorOptions(
            default_precision=0.0,
            seed_simulator=21,
            simulation_precision="single",
            optimization_level=2,
            seed_transpiler=22,
        ),
        device="cuda",
        cores=6,
    )
    noise_dict = {"errors": [{"type": "qerror"}]}
    aer_shots = build_aer_shots_estimator(
        options=AerShotsEstimatorOptions(
            method="density_matrix",
            default_precision=0.0625,
            abelian_grouping=False,
            seed_simulator=23,
            simulation_precision="double",
            optimization_level=3,
            noise_model=noise_dict,
        ),
        cores=4,
    )
    slsqp = build_slsqp(options=SLSQPOptions(maxiter=12, ftol=1e-8, disp=True))
    cobyla = build_cobyla(options=COBYLAOptions(maxiter=13, rhobeg=0.5, tol=1e-7, disp=True))
    spsa = build_spsa(
        options=SPSAOptions(
            maxiter=14,
            blocking=True,
            trust_region=True,
            learning_rate=0.2,
            perturbation=0.1,
            second_order=True,
            seed=17,
        )
    )
    unseeded_spsa = build_spsa(options=SPSAOptions())

    assert isinstance(statevector.estimator, fake.StatevectorEstimator)
    assert statevector.estimator.kwargs == {"default_precision": 0.0, "seed": 5}
    assert fake.BasicProvider.instances[-1].requested == ["basic_simulator"]
    assert isinstance(backend.estimator, fake.BackendEstimatorV2)
    assert backend.estimator.kwargs["backend"] == "backend:basic_simulator"
    assert backend.estimator.kwargs["options"]["seed_simulator"] == 9
    assert backend.transpiler == "preset-pass-manager"
    assert isinstance(aer_statevector.estimator, fake.AerEstimatorV2)
    aer_statevector_backend, aer_statevector_options = fake.AerEstimatorV2.calls[-1]
    assert aer_statevector_backend.kwargs == {
        "method": "statevector",
        "device": "GPU",
        "precision": "single",
        "max_parallel_threads": 6,
    }
    assert aer_statevector_options == {
        "options": {
            "default_precision": 0.0,
            "run_options": {"seed_simulator": 21},
        }
    }
    assert isinstance(aer_shots.estimator, fake.BackendEstimatorV2)
    assert aer_shots.estimator.kwargs["backend"].kwargs["method"] == "density_matrix"
    assert aer_shots.estimator.kwargs["backend"].kwargs["device"] == "CPU"
    assert aer_shots.estimator.kwargs["backend"].kwargs["max_parallel_threads"] == 4
    assert aer_shots.estimator.kwargs["backend"].kwargs["noise_model"].serialized == noise_dict
    assert aer_shots.estimator.kwargs["options"] == {
        "default_precision": 0.0625,
        "abelian_grouping": False,
        "seed_simulator": 23,
    }
    assert fake.NoiseModel.calls == [noise_dict]
    assert fake.pass_manager_calls == [
        {"backend": "backend:basic_simulator", "optimization_level": 1},
        {
            "backend": aer_statevector_backend,
            "optimization_level": 2,
            "seed_transpiler": 22,
        },
        {"backend": aer_shots.estimator.kwargs["backend"], "optimization_level": 3},
    ]
    assert isinstance(slsqp, fake.SLSQP)
    assert slsqp.kwargs["maxiter"] == 12
    assert isinstance(cobyla, fake.COBYLA)
    assert cobyla.kwargs["tol"] == 1e-7
    assert isinstance(spsa, fake.SPSA)
    assert spsa.kwargs["second_order"] is True
    assert fake.algorithm_globals.random_seed == 17
    assert isinstance(unseeded_spsa, fake.SPSA)
    with pytest.raises(ValueError, match="must be set together"):
        SPSAOptions(learning_rate=0.2)


def test_estimator_factories_reject_unsupported_devices_and_aer_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_qiskit_modules(monkeypatch)

    with pytest.raises(ConfigError, match=r"statevector.*cpu only"):
        build_statevector_estimator(options=StatevectorEstimatorOptions(), device="cuda")
    with pytest.raises(ConfigError, match=r"basic_backend.*cpu only"):
        build_basic_backend_estimator(options=BasicBackendEstimatorOptions(), device="cuda")

    unseeded = build_aer_statevector_estimator(options=AerStatevectorEstimatorOptions())
    assert unseeded.estimator.kwargs["options"]["run_options"] == {}

    fake.AerSimulator.available = ("CPU",)
    with pytest.raises(ConfigError, match="device 'GPU' is unavailable"):
        build_aer_statevector_estimator(options=AerStatevectorEstimatorOptions(), device="cuda")

    fake.AerSimulator.available = ("CPU", "GPU")
    fake.AerSimulator.methods = ("automatic", "density_matrix")
    with pytest.raises(ConfigError, match="method 'statevector' is unavailable"):
        build_aer_statevector_estimator(options=AerStatevectorEstimatorOptions())

    fake.AerSimulator.methods = (
        "automatic",
        "statevector",
        "density_matrix",
        "matrix_product_state",
        "tensor_network",
    )
    fake.AerSimulator.fail = True
    with pytest.raises(ConfigError, match=r"cannot configure.*simulator rejected"):
        build_aer_shots_estimator(options=AerShotsEstimatorOptions())

    fake.AerSimulator.fail = False
    with pytest.raises(ConfigError, match=r"invalid qiskit Aer noise_model.*bad serialized"):
        build_aer_shots_estimator(options=AerShotsEstimatorOptions(noise_model={"invalid": True}))


def test_lazy_algorithm_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_qiskit_modules(monkeypatch)
    context = _context()
    circuit = SimpleNamespace(num_parameters=2)
    pool = ("pool-a", "pool-b")
    components = SolverComponents(
        estimator="estimator",
        optimizer="optimizer",
        initial_state="reference",
        ansatz=AnsatzArtifacts(circuit=circuit, operator_pool=pool),
        initial_point=np.zeros(2),
        callback=lambda *_args: None,
        transpiler="transpiler",
        transpiler_options={"target": "basic"},
    )

    exact = build_exact(
        options=NoComponentOptions(), context=context, components=SolverComponents()
    )
    vqe = build_vqe(options=NoComponentOptions(), context=context, components=components)
    adapt = build_adapt_vqe(
        options=AdaptVQEOptions(
            gradient_threshold=1e-4,
            eigenvalue_threshold=2e-4,
            max_iterations=8,
            reps=1,
        ),
        context=context,
        components=components,
    )

    assert isinstance(exact.solver, fake.NumPyMinimumEigensolver)
    criterion = exact.solver.kwargs["filter_criterion"]
    assert criterion(None, 0.0, None) is True
    assert criterion(None, 0.0, {"ParticleNumber": (1.0, None)}) is False
    assert criterion(None, 0.0, {"ParticleNumber": (2.0, None)}) is True
    assert criterion(None, 0.0, {"AngularMomentum": (1.0, None)}) is False
    assert criterion(None, 0.0, {"AngularMomentum": (0.0, None)}) is True
    assert isinstance(vqe.solver, fake.VQE)
    assert vqe.solver.args == ("estimator", circuit, "optimizer")
    assert np.array_equal(vqe.solver.kwargs["initial_point"], np.zeros(2))
    assert vqe.solver.kwargs["transpiler"] == "transpiler"
    assert vqe.solver.kwargs["transpiler_options"] == {"target": "basic"}
    assert isinstance(adapt.solver, fake.AdaptVQE)
    assert adapt.solver.kwargs["operators"] == pool
    assert adapt.solver.kwargs["initial_state"] == "reference"
    assert isinstance(adapt.solver.args[0], fake.VQE)
    assert adapt.solver.args[0].kwargs["transpiler"] == "transpiler"

    assert _require("present", "a thing") == "present"
    with pytest.raises(ConfigError, match="requires a thing"):
        _require(None, "a thing")
    with pytest.raises(ConfigError, match="an estimator"):
        build_vqe(
            options=NoComponentOptions(),
            context=context,
            components=SolverComponents(
                optimizer="optimizer",
                ansatz=AnsatzArtifacts(circuit=circuit),
                initial_point=np.zeros(2),
            ),
        )
    with pytest.raises(ConfigError, match="at least one parameter"):
        build_vqe(
            options=NoComponentOptions(),
            context=context,
            components=SolverComponents(
                estimator="estimator",
                optimizer="optimizer",
                ansatz=AnsatzArtifacts(circuit=SimpleNamespace(num_parameters=0)),
                initial_point=np.zeros(0),
            ),
        )
    with pytest.raises(ConfigError, match="non-empty operator pool"):
        build_adapt_vqe(
            options=AdaptVQEOptions(),
            context=context,
            components=SolverComponents(ansatz=AnsatzArtifacts(operator_pool=())),
        )


# ---------------------------------------------------------------------------
# Workflow composition with registered fakes
# ---------------------------------------------------------------------------


def _put_spec(
    monkeypatch: pytest.MonkeyPatch,
    registry: ComponentRegistry,
    name: str,
    builder,
    *,
    capabilities: frozenset[str] = frozenset(),
    requires: frozenset[str] = frozenset(),
    backend_requirement: BackendRequirement | None = None,
) -> None:
    """Install one temporary spec without leaving global registry state behind."""
    monkeypatch.setitem(
        registry._specs,
        name,
        ComponentSpec(
            options_cls=NoComponentOptions,
            builder=builder,
            capabilities=capabilities,
            requires=requires,
            backend_requirement=backend_requirement,
        ),
    )


def test_estimator_provider_requirements_are_registry_driven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirement = BackendRequirement(extra="qiskit-provider", import_name="provider_module")
    _put_spec(
        monkeypatch,
        ESTIMATORS,
        "provider_estimator",
        lambda **_kwargs: EstimatorResource("provider"),
        backend_requirement=requirement,
    )
    engine = get_engine("qiskit")

    assert (
        engine.backend_requirement({"algorithm": "vqe", "estimator": "provider_estimator"})
        == requirement
    )
    assert "qiskit-provider" in engine.backend_extras()


def _install_ground_state_solver(monkeypatch: pytest.MonkeyPatch, result: object) -> type:
    """Install a GroundStateEigensolver fake returning ``result``."""

    class GroundStateEigensolver:
        calls: ClassVar[list[tuple[object, object, object]]] = []

        def __init__(self, mapper: object, solver: object) -> None:
            self.mapper = mapper
            self.solver = solver

        def solve(self, problem: object) -> object:
            type(self).calls.append((self.mapper, self.solver, problem))
            return result

    _install_module(
        monkeypatch,
        "qiskit_nature.second_q.algorithms",
        GroundStateEigensolver=GroundStateEigensolver,
    )
    return GroundStateEigensolver


def _patch_problem_context(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, ElectronicStructureContext]:
    """Bypass the real driver while retaining the workflow's assembly logic."""
    problem = SimpleNamespace(name="problem")
    context = _context()
    monkeypatch.setattr(workflow, "_build_problem", lambda *_args, **_kwargs: problem)
    monkeypatch.setattr(workflow, "_electronic_context", lambda *_args, **_kwargs: context)
    return problem, context


def test_run_job_exact_path_skips_variational_components(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    problem, context = _patch_problem_context(monkeypatch)
    solver = object()

    def algorithm_builder(**_kwargs):
        return AlgorithmArtifacts(solver=solver)

    _put_spec(monkeypatch, ALGORITHMS, "fake_exact", algorithm_builder)
    raw = SimpleNamespace(cost_function_evals=0)
    solved = SimpleNamespace(total_energies=[-1.25 + 0.0j], raw_result=raw)
    ground = _install_ground_state_solver(monkeypatch, solved)
    xyz = tmp_path / "unused.xyz"

    result = workflow.run_job(
        xyz,
        charge=0,
        multiplicity=1,
        options=QiskitOptions(algorithm="fake_exact").as_job_spec(),
    )

    assert result.energy_hartree == -1.25
    assert result.metadata["num_qubits"] == context.num_qubits
    assert result.metadata["device"] == "cpu"
    assert result.metadata["cores"] == 1
    assert result.metadata["solver"]["cost_function_evals"] == 0
    assert result.metadata["components"]["estimator"] == {
        "name": "statevector",
        "options": {"default_precision": 0.0, "seed": None},
    }
    assert result.metadata["components"]["optimizer"] == {
        "name": "slsqp",
        "options": {"maxiter": 100, "ftol": 1e-6, "disp": False},
    }
    assert ground.calls == [(context.mapper, solver, problem)]
    assert result.metadata["evaluations"] == []


def test_run_job_builds_and_closes_a_full_variational_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    problem, context = _patch_problem_context(monkeypatch)
    events: list[str] = []
    circuit = SimpleNamespace(num_parameters=2)
    solver = object()

    def state_builder(**_kwargs):
        events.append("state")
        return "state"

    def ansatz_builder(**kwargs):
        events.append(f"ansatz:{kwargs['initial_state']}")
        return AnsatzArtifacts(circuit=circuit)

    def point_builder(**kwargs):
        events.append(f"point:{kwargs['ansatz'].circuit is circuit}")
        return np.array([0.1, -0.1])

    def optimizer_builder(**_kwargs):
        events.append("optimizer")
        return "optimizer"

    def estimator_builder(**kwargs):
        events.append("estimator")
        assert kwargs["device"] == "cpu"
        assert kwargs["cores"] == 3
        return EstimatorResource(
            "estimator",
            close=lambda: events.append("closed"),
            transpiler="pass-manager",
            transpiler_options={"callback": "enabled"},
        )

    def algorithm_builder(**kwargs):
        events.append("algorithm")
        assembled = kwargs["components"]
        assert assembled.estimator == "estimator"
        assert assembled.optimizer == "optimizer"
        assert np.array_equal(assembled.initial_point, [0.1, -0.1])
        assert assembled.transpiler == "pass-manager"
        assert assembled.transpiler_options == {"callback": "enabled"}
        assert assembled.callback is not None
        assembled.callback(3, np.array([0.1, -0.1]), -1.2, {"variance": 0.0})
        assembled.callback(
            1,
            np.array([0.2, -0.2]),
            -1.1,
            {"variance": np.float64(0.1)},
        )
        assembled.callback(2, np.array([0.3, -0.3]), -1.0, {})
        return AlgorithmArtifacts(solver=solver)

    _put_spec(
        monkeypatch,
        ALGORITHMS,
        "fake_vqe",
        algorithm_builder,
        requires=frozenset({"estimator", "optimizer", "initial_state", "circuit", "initial_point"}),
    )
    _put_spec(monkeypatch, INITIAL_STATES, "fake_state", state_builder)
    _put_spec(
        monkeypatch,
        ANSATZE,
        "fake_ansatz",
        ansatz_builder,
        capabilities=frozenset({"circuit"}),
    )
    _put_spec(monkeypatch, INITIAL_POINTS, "fake_point", point_builder)
    _put_spec(monkeypatch, OPTIMIZERS, "fake_optimizer", optimizer_builder)
    _put_spec(monkeypatch, ESTIMATORS, "fake_estimator", estimator_builder)
    raw = SimpleNamespace(
        cost_function_evals=3,
        num_iterations=2,
        optimal_point=np.array([0.1, -0.1]),
        optimal_value=-1.2 + 0.0j,
        termination_criterion="done",
    )
    solved = SimpleNamespace(total_energies=[-1.2], raw_result=raw)
    ground = _install_ground_state_solver(monkeypatch, solved)
    options = QiskitOptions(
        algorithm="fake_vqe",
        ansatz="fake_ansatz",
        initial_state="fake_state",
        estimator="fake_estimator",
        optimizer="fake_optimizer",
        initial_point="fake_point",
        active_space={"electrons": 2, "orbitals": 2},
        cores=3,
    )

    result = workflow.run_job(
        tmp_path / "unused.xyz",
        charge=0,
        multiplicity=1,
        options=options,
    )

    assert events == [
        "state",
        "ansatz:state",
        "point:True",
        "optimizer",
        "estimator",
        "algorithm",
        "closed",
    ]
    assert ground.calls == [(context.mapper, solver, problem)]
    assert result.metadata["active_space"] == {"electrons": 2, "orbitals": 2}
    assert result.metadata["evaluations"] == [
        {
            "evaluation": 1,
            "inner_run": 1,
            "algorithm_evaluation": 3,
            "objective_value_hartree": -1.2,
            "metadata": {"variance": 0.0},
        },
        {
            "evaluation": 2,
            "inner_run": 2,
            "algorithm_evaluation": 1,
            "objective_value_hartree": -1.1,
            "metadata": {"variance": 0.1},
        },
        {
            "evaluation": 3,
            "inner_run": 2,
            "algorithm_evaluation": 2,
            "objective_value_hartree": -1.0,
            "metadata": {},
        },
    ]
    assert result.metadata["solver"]["optimal_value"] == {"real": -1.2, "imag": 0.0}


@pytest.mark.parametrize(
    "energies, match",
    [
        ([], "no total ground-state energy"),
        ([complex(-1.0, 1e-4)], "complex total energy"),
        ([float("nan")], "non-finite total energy"),
    ],
)
def test_run_job_rejects_missing_complex_or_nonfinite_energies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    energies: list[complex | float],
    match: str,
) -> None:
    _patch_problem_context(monkeypatch)
    _put_spec(
        monkeypatch,
        ALGORITHMS,
        "fake_bad_energy",
        lambda **_kwargs: AlgorithmArtifacts(solver=object()),
    )
    _install_ground_state_solver(
        monkeypatch,
        SimpleNamespace(total_energies=energies, raw_result=None),
    )
    with pytest.raises(ConfigError, match=match):
        workflow.run_job(
            tmp_path / "unused.xyz",
            charge=0,
            multiplicity=1,
            options=QiskitOptions(algorithm="fake_bad_energy"),
        )


# ---------------------------------------------------------------------------
# Driver/context helpers
# ---------------------------------------------------------------------------


def test_atom_spec_reads_one_xyz_frame_and_rejects_bad_inputs(tmp_path: Path) -> None:
    xyz = tmp_path / "h2.xyz"
    xyz.write_text(
        "2\nhydrogen\nH 0 0 0 extra\nH 0 0 0.735\n",
        encoding="utf-8",
    )
    assert workflow._atom_spec(xyz) == "H 0 0 0; H 0 0 0.735"

    with pytest.raises(ConfigError, match="cannot read Qiskit XYZ"):
        workflow._atom_spec(tmp_path / "missing.xyz")
    xyz.write_text("not-an-int\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot read Qiskit XYZ"):
        workflow._atom_spec(xyz)
    xyz.write_text("2\ncomment\nH 0 0 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="declares 2 atoms"):
        workflow._atom_spec(xyz)
    xyz.write_text("1\ncomment\nH 0 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="malformed atom row 1"):
        workflow._atom_spec(xyz)


def test_build_problem_is_lazy_and_applies_active_space(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nh2\nH 0 0 0\nH 0 0 0.735\n", encoding="utf-8")
    base_problem = SimpleNamespace(name="base")
    transformed_problem = SimpleNamespace(name="transformed")

    class PySCFDriver:
        calls: ClassVar[list[dict[str, object]]] = []

        def __init__(self, **kwargs: object) -> None:
            type(self).calls.append(kwargs)

        def run(self) -> object:
            return base_problem

    class ActiveSpaceTransformer:
        calls: ClassVar[list[dict[str, object]]] = []

        def __init__(self, **kwargs: object) -> None:
            type(self).calls.append(kwargs)

        def transform(self, problem: object) -> object:
            assert problem is base_problem
            return transformed_problem

    class DistanceUnit:
        ANGSTROM = "angstrom"

    _install_module(monkeypatch, "qiskit_nature.second_q.drivers", PySCFDriver=PySCFDriver)
    _install_module(
        monkeypatch,
        "qiskit_nature.second_q.transformers",
        ActiveSpaceTransformer=ActiveSpaceTransformer,
    )
    _install_module(monkeypatch, "qiskit_nature.units", DistanceUnit=DistanceUnit)

    plain = workflow._build_problem(
        xyz, charge=-1, multiplicity=2, options=QiskitOptions(active_space=None)
    )
    transformed = workflow._build_problem(
        xyz,
        charge=0,
        multiplicity=1,
        options=QiskitOptions(active_space={"electrons": (1, 1), "orbitals": 2}),
    )

    assert plain is base_problem
    assert transformed is transformed_problem
    assert PySCFDriver.calls[0]["spin"] == 1
    assert PySCFDriver.calls[0]["charge"] == -1
    assert ActiveSpaceTransformer.calls == [{"num_electrons": (1, 1), "num_spatial_orbitals": 2}]
    with pytest.raises(ConfigError, match="multiplicity must be at least 1"):
        workflow._build_problem(xyz, charge=0, multiplicity=0, options=QiskitOptions())


def test_electronic_context_builds_mapper_and_checks_particles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapped = SimpleNamespace(num_qubits=6)

    class Mapper:
        def map(self, operator: object) -> object:
            assert operator == "hamiltonian"
            return mapped

    problem = SimpleNamespace(
        hamiltonian=SimpleNamespace(second_q_op=lambda: "hamiltonian"),
        num_particles=(2, 1),
        num_spatial_orbitals=3,
    )
    monkeypatch.setattr(workflow.MAPPERS, "build", lambda *_args, **_kwargs: Mapper())
    context = workflow._electronic_context(problem, QiskitOptions(), multiplicity=2)
    assert context.num_particles == (2, 1)
    assert context.num_spatial_orbitals == 3
    assert context.num_qubits == 6
    assert context.multiplicity == 2
    assert context.qubit_hamiltonian is mapped

    bad = SimpleNamespace(
        hamiltonian=problem.hamiltonian,
        num_particles=(1, 1, 0),
        num_spatial_orbitals=3,
    )
    with pytest.raises(ConfigError, match="invalid particles"):
        workflow._electronic_context(bad, QiskitOptions(), multiplicity=1)


def test_jsonable_handles_qiskit_and_numpy_diagnostic_shapes() -> None:
    class ArrayLike:
        def tolist(self) -> list[int]:
            return [7, 8]

    class Other:
        def __str__(self) -> str:
            return "other"

    assert workflow._jsonable(None) is None
    assert workflow._jsonable(np.float64(1.5)) == 1.5
    assert workflow._jsonable("x") == "x"
    assert workflow._jsonable(2 + 3j) == {"real": 2.0, "imag": 3.0}
    assert workflow._jsonable({1: (np.array([1, 2]), ArrayLike())}) == {"1": [[1, 2], [7, 8]]}
    assert workflow._jsonable(Other()) == "other"


def test_estimator_resource_always_runs_its_cleanup() -> None:
    events: list[str] = []
    resource = EstimatorResource("estimator", close=lambda: events.append("closed"))
    with resource as estimator:
        assert estimator == "estimator"
    assert events == ["closed"]
