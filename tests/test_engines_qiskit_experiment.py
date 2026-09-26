"""Artifact quantum steps run through ordinary scheduling and preserve their products."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine.config import Config, StepConfig
from chemrefine.engines.api import ArtifactEngine, JobExecutable, TemplateDriven
from chemrefine.engines.qiskit.bundles import bundle_dependencies, read_bundle
from chemrefine.engines.qiskit.experiment import (
    EXPERIMENTS,
    ExperimentResult,
    QiskitExperimentEngine,
    QiskitExperimentOptions,
    run_experiment,
    validate_experiment,
)
from chemrefine.errors import ConfigError, NoUsableCacheError, OutputParseError
from chemrefine.state import PipelineState, StepContext, Structure
from chemrefine.step import StepMode, rebuild_cache_step, run_step


def _options():
    """A two-site hopping experiment with a known analytic trajectory."""
    return {
        "backend_python": sys.executable,
        "cores": 8,
        "experiment": {
            "name": "lattice_dynamics",
            "options": {
                "model": {
                    "num_sites": 2,
                    "spinful": False,
                    "edges": [{"source": 0, "target": 1}],
                },
                "times": [0, 0.37],
                "dynamics": {"exact_reference": True},
            },
        },
    }


def _ctx(tmp_path, **updates):
    """One artifact step, with inherited structures that must not acquire new energies."""
    templates = tmp_path / "templates"
    templates.mkdir(exist_ok=True)
    (templates / "cpu.slurm.header").write_text("#!/bin/bash\n")
    return StepContext(
        step_cfg=StepConfig(step=1, engine="qiskit-experiment", options={**_options(), **updates}),
        step_dir=tmp_path / "step1",
        template_dir=templates,
        template=None,
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(structures=(Structure(id="0", atoms=Atoms("H")),)),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        dispatch="local",
    )


def test_artifact_engine_schedules_real_worker_and_preserves_scratch_payloads(tmp_path):
    """The shared local job script copies a complete real numerical bundle back."""
    engine = QiskitExperimentEngine()
    assert engine.component_catalog()["experiment"].default == "lattice_dynamics"
    ctx = _ctx(tmp_path)
    assert isinstance(engine, ArtifactEngine)
    assert isinstance(engine, JobExecutable)
    assert not isinstance(engine, TemplateDriven)
    inputs = engine.prepare(ctx)
    assert engine.slurm_layout(ctx) == (1, 2)
    assert engine.gpus(ctx) == 0
    assert not engine.single_node(ctx)
    assert engine.memory_mb(ctx) is None
    engine.submit(inputs, ctx)
    result = engine.parse(inputs, ctx)
    assert result.structures == ctx.prev_state.structures
    bundle = read_bundle(engine.artifact(ctx))
    assert bundle.metadata["resolved_options"]["cores"] == 2
    np.testing.assert_allclose(
        bundle.arrays["occupations"][1], [np.cos(0.37) ** 2, np.sin(0.37) ** 2], atol=1e-12
    )
    assert bundle_dependencies(engine.artifact(ctx))["payload"].exists()
    assert not list(ctx.scratch_dir.glob("_work_*"))


def test_artifact_recovery_validates_payload_and_never_executes_during_rebuild(
    tmp_path, monkeypatch
):
    """A manifest alone cannot serve a cache after its NPZ payload disappears."""
    engine = QiskitExperimentEngine()
    ctx = _ctx(tmp_path)
    config = Config(
        template_dir=ctx.template_dir,
        output_dir=tmp_path,
        charge=0,
        multiplicity=1,
        max_cores=2,
        steps=[ctx.step_cfg],
        dispatch="local",
    )
    first = run_step(config, ctx.step_cfg, ctx.prev_state, engine=engine)
    assert not first.cache_hit
    assert run_step(config, ctx.step_cfg, ctx.prev_state, engine=engine).cache_hit

    def refuse_submit(*args, **kwargs):
        """Detect any accidental scheduling from non-executing recovery paths."""
        pytest.fail("rebuild/cache-only must not submit")

    monkeypatch.setattr(engine, "submit", refuse_submit)
    monkeypatch.setattr("chemrefine.step.get_engine", lambda name: engine)
    rebuild_cache_step(config, ctx.step_cfg, ctx.prev_state)
    bundle_dependencies(engine.artifact(ctx))["payload"].unlink()
    with pytest.raises(NoUsableCacheError):
        run_step(config, ctx.step_cfg, ctx.prev_state, engine=engine, mode=StepMode.CACHE_ONLY)
    with pytest.raises(OutputParseError):
        rebuild_cache_step(config, ctx.step_cfg, ctx.prev_state)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"experiment": "missing"}, "unsupported"),
        ({"experiment": {"name": "lattice_dynamics", "options": {"typo": 1}}}, "typo"),
        ({"device": "cuda"}, "cpu"),
        ({"max_output_bytes": 1}, "max_output_bytes"),
        (
            {"experiment": {"name": "lattice_dynamics", "options": {"model": {"num_sites": 99}}}},
            "max_qubits",
        ),
        (
            {"experiment": {"name": "lattice_dynamics", "options": {"occupied_modes": [0, 0]}}},
            "distinct",
        ),
        (
            {"experiment": {"name": "lattice_dynamics", "options": {"occupied_modes": [-1]}}},
            "distinct",
        ),
    ],
)
def test_invalid_experiments_fail_before_publication(tmp_path, options, match):
    """Unsupported configurations cannot produce a successful artifact descriptor."""
    with pytest.raises(ConfigError, match=match):
        run_experiment(options, tmp_path / "artifact.json")
    assert not (tmp_path / "artifact.json").exists()


def test_declared_provider_and_single_artifact_failure_policy(tmp_path):
    """Preflight, backend discovery and runtime share one options interpretation."""
    engine = QiskitExperimentEngine()
    assert engine.backend_requirement({}).extra == "qiskit-fermionic"
    assert {"qiskit", "qiskit-aer", "qiskit-fermionic"} <= engine.backend_extras()
    ctx = _ctx(tmp_path)
    with pytest.raises(ConfigError, match="on_failure"):
        engine.prepare(
            replace(ctx, step_cfg=ctx.step_cfg.model_copy(update={"on_failure": "skip"}))
        )
    assert engine.output_dirs(ctx) == ("checkpoints", "provider_jobs")
    assert '"$INP_NAME"' in engine.run_block(ctx, Path("$INP_NAME"), Path("out")).body


def test_registered_python_experiment_uses_same_bundle_contract(tmp_path, monkeypatch):
    """Independent builders can publish reports without fabricating a molecular energy."""
    monkeypatch.setattr(EXPERIMENTS, "_specs", dict(EXPERIMENTS._specs))

    @EXPERIMENTS.register("report", capabilities=frozenset({"cuda"}))
    def report(**context):
        """A provider-free report exercises registry extensibility."""
        return ExperimentResult(kind="report", arrays={}, metadata={"cores": context["cores"]})

    options = QiskitExperimentOptions.from_raw({"experiment": "report", "device": "cuda"})
    validate_experiment(options)
    engine = QiskitExperimentEngine()
    assert engine.backend_requirement(options.model_dump()).extra == "qiskit"
    path = run_experiment(options.model_dump(), tmp_path / "artifact.json")
    assert read_bundle(path).metadata["cores"] == 1


def test_spinful_trajectory_records_mode_order(tmp_path):
    """Spinful occupations retain the explicit alpha-then-beta convention."""
    options = {"experiment": {"name": "lattice_dynamics", "options": {"model": {"num_sites": 1}}}}
    path = run_experiment(options, tmp_path / "artifact.json")
    assert read_bundle(path).metadata["mode_order"] == "alpha_then_beta"


def test_recorded_lattice_artifact_contract():
    """A recorded real lattice result remains readable without executing an SDK."""
    root = Path(__file__).parent / "data" / "engines" / "qiskit-experiment"
    contract = json.loads((root / "artifact_contract.json").read_text())
    result = read_bundle(root / contract["fixtures"][0])
    assert result.description.kind == "lattice_trajectory"
    np.testing.assert_allclose(result.arrays["occupations"][0], [1, 0], atol=1e-12)
