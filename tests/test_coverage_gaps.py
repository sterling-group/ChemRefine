"""Branch-completion tests — the edge cases that take coverage to 100%.

These exercise error paths, optional-dependency fallbacks, and rarely-hit
branches that the per-module test files don't reach on their happy paths.
"""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from chemrefine.config import Config, StepConfig
from chemrefine.errors import CacheError, ChemRefineError, OutputParseError
from chemrefine.state import (
    PipelineState,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)
from chemrefine.step import StepMode
from chemrefine.step_failures import FailureKind, FailureRecord


def _ctx(tmp_path: Path, *, options=None, nms: bool = False, engine: str = "fake") -> StepContext:
    """A minimal StepContext for unit-level branch tests (no templates needed)."""
    step_cfg = StepConfig(step=1, engine=engine, operation="opt_sp", options=options or {}, nms=nms)
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=tmp_path / "templates",
        scratch_dir=None,
        prev_state=PipelineState(structures=()),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        executables={},
    )


def _cfg(tmp_path: Path, **step_over) -> Config:
    return Config(
        output_dir=tmp_path / "outputs",
        steps=[StepConfig(step=1, engine="fake", operation="opt_sp", **step_over)],
    )


# --- module __main__ entry points -------------------------------------------


@pytest.mark.filterwarnings("ignore:.*found in sys.modules.*:RuntimeWarning")
@pytest.mark.parametrize(
    "module",
    [
        "chemrefine.engines.orca.extopt.bridge",
        "chemrefine.engines._backend_server.server",
    ],
)
def test_module_entrypoint_runs_main(module, monkeypatch):
    """`python -m <module>` dispatches through the ``if __name__ == "__main__"`` guard.

    Driven via ``--help`` so ``main()`` exits cleanly (argparse SystemExit) without
    binding a socket or contacting a backend. ``runpy`` executes the module as
    ``__main__`` in-process so the guard line runs under coverage — a spawned
    subprocess isn't viable here (the server blocks; the bridge needs a live backend).
    """
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit):
        runpy.run_module(module, run_name="__main__")


# --- throttle ---------------------------------------------------------------


def test_throttler_register_rejects_negative_gpus():
    from chemrefine.throttle import Throttler

    with pytest.raises(ValueError, match="gpus must be >= 0"):
        Throttler(max_cores=8, max_gpus=1).register("g", 1, gpus=-1)


def test_throttler_assign_device_raises_when_all_taken():
    from chemrefine.throttle import Throttler

    # Unreachable on the real call path (wait_for_room admits first), so assign_device
    # fails loud rather than silently colliding two GPU jobs on device 0.
    t = Throttler(max_cores=8, max_gpus=1)
    t.register("g", 1, gpus=1, device=0)
    with pytest.raises(RuntimeError, match="no free GPU device"):
        t.assign_device()


# --- base: the abstract SLURM hooks -----------------------------------------


def test_job_engine_primitive_hooks_are_abstract():
    from chemrefine.engines._job import JobEngine

    eng = JobEngine()
    with pytest.raises(NotImplementedError):
        eng.pal(None)
    with pytest.raises(NotImplementedError):
        eng.run_block(None, Path("i"), Path("o"))
    with pytest.raises(NotImplementedError):
        eng.build_input(
            xyz_path=Path("x"),
            template_path=Path("t"),
            input_path=Path("i"),
            output_path=Path("o"),
            ctx=None,
        )
    with pytest.raises(NotImplementedError):
        eng.parse_one(Path("o"), "0", None)


# --- cache: corrupt failed-jobs ledger --------------------------------------


def test_load_failed_jobs_raises_on_corrupt_ledger(tmp_path: Path):
    from chemrefine import cache

    path = cache.failed_jobs_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CacheError, match="corrupt failed-jobs ledger"):
        cache.load_failed_jobs(tmp_path)


# --- cli: malformed legacy argv passes through ------------------------------


def test_translate_legacy_argv_passes_through_on_argparse_error():
    from chemrefine.cli import _translate_legacy_argv

    argv = ["c.yaml", "--maxcores", "not-an-int"]  # argparse SystemExit
    assert _translate_legacy_argv(argv) == argv


# --- config: legacy sample normalizer edges ---------------------------------


def test_normalize_sample_helpers_pass_through_non_dict():
    from chemrefine.config import _flatten_sample_type, _normalize_sample_block

    assert _flatten_sample_type("nope") == "nope"
    assert _normalize_sample_block("nope") == "nope"


def test_flatten_then_normalize_carries_through_extra_top_level_keys():
    from chemrefine.config import _flatten_sample_type, _normalize_sample_block

    flat = _flatten_sample_type(
        {"method": "boltzmann", "parameters": {"weight": 95}, "by_parent": True}
    )
    out = _normalize_sample_block(flat)
    assert out == {"method": "boltzmann", "percent_cumulative": 95, "by_parent": True}


def test_normalize_sample_block_without_method_passes_keys_through():
    """A block with no ``method`` key leaves it out (validation rejects it later)."""
    from chemrefine.config import _normalize_sample_block

    assert _normalize_sample_block({"count": 5}) == {"count": 5}


# --- orca output text dispatcher --------------------------------------------


def test_parse_text_handles_pes():
    from synthetic import synthetic_pes_segment

    from chemrefine.engines.orca import output

    text = synthetic_pes_segment(coords=[("H", 0.0, 0.0, 0.0)], energy=-1.0)
    parsed = output.parse_text(text, "pes", src="x")
    assert parsed and parsed[-1].energy_hartree == -1.0


def test_parse_text_rejects_non_text_operation():
    from chemrefine.engines.orca import output

    with pytest.raises(OutputParseError):
        output.parse_text("", "goat", src="x")


# --- mlip-train engine no-op / unsupported ----------------------------------


def test_mlip_train_engine_not_nms_capable(tmp_path: Path):
    from chemrefine.engines.api import NmsCapableEngine, get_engine

    eng = get_engine("mlip-train")
    # mlip-train is a pass-through: it doesn't satisfy the NMS hook contract.
    assert not isinstance(eng, NmsCapableEngine)


def test_trainer_rejects_valid_fraction_leaving_no_training(tmp_path: Path):
    from chemrefine.engines.mlip import trainer

    seeds = tuple(
        Structure(
            id=str(i),
            atoms=Atoms("H", positions=[[0, 0, 0]]),
            energy_hartree=-1.0,
            forces_ev_per_a=np.zeros((1, 3)),
        )
        for i in range(2)
    )
    ctx = _ctx(tmp_path, options={"valid_fraction": 1.0})
    with pytest.raises(ValueError, match="leaves no training"):
        trainer.prepare_inputs(StepResults(structures=seeds), ctx)


# --- orb backend success path (mock the optional library) -------------------


def test_build_orb_success_path(monkeypatch):
    pretrained = types.SimpleNamespace(orb_v2=MagicMock(return_value="ORBFF"))
    forcefield = types.ModuleType("orb_models.forcefield")
    forcefield.pretrained = pretrained
    calc_mod = types.ModuleType("orb_models.forcefield.inference.calculator")
    calc_mod.ORBCalculator = MagicMock(return_value="ORB_CALC")
    for name, mod in {
        "orb_models": types.ModuleType("orb_models"),
        "orb_models.forcefield": forcefield,
        "orb_models.forcefield.inference": types.ModuleType("orb_models.forcefield.inference"),
        "orb_models.forcefield.inference.calculator": calc_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    from chemrefine.engines.mlip.backends.orb import _build_orb

    assert _build_orb(model_name="orb_v2", device="cpu") == "ORB_CALC"


# --- step.py small branches -------------------------------------------------


def _struct(**kw) -> Structure:
    return Structure(id="0", atoms=Atoms("H", positions=[[0, 0, 0]]), **kw)


def test_failure_kind_branches():
    from chemrefine.step_failures import FailureKind, failure_kind

    assert failure_kind(_struct(terminated_normally=False)) is FailureKind.NOT_TERMINATED
    assert failure_kind(_struct(converged=False)) is FailureKind.NOT_CONVERGED
    assert failure_kind(_struct()) is FailureKind.FAILED


def test_failure_kind_values_are_the_human_wording():
    """The enum values *are* the messages, so the ledger stays readable and the
    recovery paths still branch on a name rather than on that wording."""
    from chemrefine.step_failures import Failure, FailureKind

    assert FailureKind.NOT_CONVERGED.value == "did not converge"
    assert Failure("0", FailureKind.MISSING_OUTPUT, None).reason == "output missing"
    # A kind that carries detail appends it rather than replacing the name.
    detailed = Failure("0", FailureKind.UNPARSEABLE, None, detail="bad token at line 3")
    assert detailed.reason == "unparseable: bad token at line 3"


def test_failure_record_round_trips_through_the_ledger():
    from chemrefine.step_failures import Failure, FailureKind, FailureRecord

    record = FailureRecord.of(Failure("7", FailureKind.NOT_CONVERGED, None))
    assert FailureRecord.from_json(record.to_json()) == record
    assert record.to_json()["kind"] == "did not converge"


def test_parse_with_failures_records_unparseable(tmp_path: Path):
    from chemrefine import step_failures

    out = tmp_path / "s.out"
    out.write_text("garbage", encoding="utf-8")

    class _Engine:
        def parse(self, inputs, ctx):
            raise OutputParseError("boom")

    inputs = StepInputs(files=((tmp_path / "s.inp", out, "0"),))
    successes, failures = step_failures.parse_with_failures(_Engine(), inputs, _ctx(tmp_path))
    assert successes == []
    assert failures[0].reason.startswith("unparseable")


def test_halt_if_pending_skips_a_cache_only_step(tmp_path: Path):
    from chemrefine import step

    cfg = _cfg(tmp_path, on_failure="stop")
    # CACHE_ONLY is the mode every step a scoped action isn't targeting runs in;
    # halting there would stop `rerun-errors N` before it ever reached step N.
    step.halt_if_pending(cfg, cfg.steps[0], StepMode.CACHE_ONLY)


def test_halt_if_pending_raises_when_stop_step_has_pending(tmp_path: Path):
    from chemrefine import cache, step

    cfg = _cfg(tmp_path, on_failure="stop")
    step_dir = step.step_dir_for(cfg, cfg.steps[0])
    cache.save_failed_jobs(step_dir, [{"structure_id": "1", "kind": "failed", "reason": "x"}])
    with pytest.raises(ChemRefineError, match="halted"):
        step.halt_if_pending(cfg, cfg.steps[0], StepMode.RESUME)


def test_halt_if_pending_no_pending_returns(tmp_path: Path):
    from chemrefine import step

    cfg = _cfg(tmp_path, on_failure="stop")
    step.step_dir_for(cfg, cfg.steps[0]).mkdir(parents=True, exist_ok=True)
    step.halt_if_pending(cfg, cfg.steps[0], StepMode.RESUME)  # no ledger → no raise


# --- recovery: rerun-errors with nothing pending ----------------------------


def test_rerun_errors_logs_when_no_failures(tmp_path: Path, monkeypatch):
    from chemrefine import recovery

    cfg = _cfg(tmp_path)
    monkeypatch.setattr(recovery.pipeline, "run", lambda *a, **k: [])
    recovery._action_rerun_errors(cfg, None)  # last step, no ledger → "no failures" branch


# --- orb older-layout fallback ----------------------------------------------


@pytest.mark.parametrize(
    "task, lib, package, extra",
    [
        ("mace_off", "mace", "mace-torch", "mlip-mace"),
        ("omol", "fairchem", "fairchem-core", "mlip-fairchem"),
        ("sevenn", "sevenn", "sevenn", "mlip-sevenn"),
        ("chgnet", "chgnet", "chgnet", "mlip-chgnet"),
        ("orb", "orb_models", "orb-models", "mlip-orb"),
    ],
)
def test_backend_missing_dependency_names_the_extra(task, lib, package, extra, monkeypatch):
    """A missing backend lib → a helpful ImportError naming the package + extra."""
    monkeypatch.setitem(sys.modules, lib, None)  # force the lazy import to fail
    from chemrefine.engines.mlip.calculator import build_calculator

    with pytest.raises(ImportError, match=f"{package}.*{extra}"):
        build_calculator(task_name=task, model_name="x")


def test_build_orb_older_layout(monkeypatch):
    """When the v3 ``inference.calculator`` import fails, fall back to the older path."""
    pretrained = types.SimpleNamespace(orb_v2=MagicMock(return_value="ORBFF"))
    forcefield = types.ModuleType("orb_models.forcefield")
    forcefield.pretrained = pretrained
    older_calc = types.ModuleType("orb_models.forcefield.calculator")
    older_calc.ORBCalculator = MagicMock(return_value="OLD_CALC")
    for name, mod in {
        "orb_models": types.ModuleType("orb_models"),
        "orb_models.forcefield": forcefield,
        # v3 layout absent (None ⇒ ImportError) → exercises the older-layout branch
        "orb_models.forcefield.inference.calculator": None,
        "orb_models.forcefield.calculator": older_calc,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    from chemrefine.engines.mlip.backends.orb import _build_orb

    assert _build_orb(model_name="orb_v2", device="cpu") == "OLD_CALC"


# --- no-manifest guards (step / nms) ----------------------------------------


def test_rebuild_cache_step_raises_without_manifest(tmp_path: Path):
    from chemrefine import step

    cfg = _cfg(tmp_path)
    with pytest.raises(CacheError, match="cannot rebuild-cache"):
        step.rebuild_cache_step(cfg, cfg.steps[0], PipelineState(structures=()))


def test_resubmit_failed_raises_without_manifest(tmp_path: Path):
    from chemrefine import step
    from chemrefine.engines.api import get_engine

    ctx = _ctx(tmp_path)
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(CacheError, match="no manifest to rehydrate"):
        step._resubmit_failed(
            get_engine("fake"),
            ctx,
            ctx.step_cfg,
            [FailureRecord("0", FailureKind.MISSING_OUTPUT, "output missing")],
            (),
        )


def test_reattempt_nms_raises_without_manifest(tmp_path: Path):
    from chemrefine import nms
    from chemrefine.engines.api import get_engine

    ctx = _ctx(tmp_path, nms=True, engine="orca")
    ctx.step_dir.mkdir(parents=True, exist_ok=True)  # no manifest written
    with pytest.raises(CacheError, match="no manifest"):
        nms.reattempt_nms(get_engine("orca"), ctx, ctx.step_cfg, None, ())


def test_rebuild_cache_step_nms_branch(tmp_path: Path):
    """rebuild-cache routes an NMS step through nms.rebuild_nms (here: an already-resolved
    round-1, so the survivor passes through at its canonical id)."""
    from synthetic import synthetic_dft_output

    from chemrefine import cache, step
    from chemrefine.ids import structure_artifact_path

    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        output_dir=tmp_path / "outputs",
        template_dir=template_dir,
        steps=[
            StepConfig(
                step=1, engine="orca", operation="freq", nms=True, options={"target": "minimum"}
            ),
        ],
    )
    step_cfg = cfg.steps[0]
    step_dir = (cfg.output_dir / step_cfg.dir_name()).resolve()
    out = structure_artifact_path(step_dir, 1, "0", "out")
    inp = structure_artifact_path(step_dir, 1, "0", "inp")
    out.parent.mkdir(parents=True, exist_ok=True)
    # A frequency table with NO imaginary modes ⇒ already at the minimum ⇒ resolved.
    out.write_text(
        synthetic_dft_output([-1.0], [("H", 0, 0, 0), ("H", 0.74, 0, 0)])
        + "VIBRATIONAL FREQUENCIES\n-----------------------\n     6:    100.00 cm**-1\n"
        + "\n****ORCA TERMINATED NORMALLY****\n",
        encoding="utf-8",
    )
    inp.write_text("! Opt Freq\n", encoding="utf-8")
    cache.save_manifest(
        StepInputs(files=((inp, out, "0"),)), step_dir, operation="freq", engine="orca"
    )
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    outcome = step.rebuild_cache_step(cfg, step_cfg, PipelineState(structures=(seed,)))
    assert outcome.cache_hit is False
    assert {s.id for s in outcome.state.structures} == {"0"}  # resolved, id kept


# --- _nms_reuse_outcome (NMS reuse-fingerprint path) ------------------------


def _pin_nms(monkeypatch, fp: str = "FP") -> None:
    """Pin the NMS reuse fingerprint + re-attempt result for these tests.

    ``step.py`` calls through the :mod:`chemrefine.cache` and :mod:`chemrefine.nms`
    module objects, so patching the module attributes redirects the orchestrator
    without touching its code.
    """
    from chemrefine import cache, nms

    monkeypatch.setattr(
        cache,
        "reuse_fingerprint",
        lambda step_cfg, parent_ids, *, parents_digest="", template_digest="": fp,
    )
    monkeypatch.setattr(
        nms,
        "reattempt_nms",
        lambda engine, ctx, step_cfg, cached, parent_ids: StepResults(
            structures=(
                Structure(id="re", atoms=Atoms("H", positions=[[0, 0, 0]]), energy_hartree=-1.0),
            )
        ),
    )


def _save_reuse_cache(ctx, reuse_fp: str):
    from chemrefine import cache

    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    cache.save(
        step_cfg=ctx.step_cfg,
        parent_ids=(),
        results=StepResults(
            structures=(Structure(id="c", atoms=Atoms("H", positions=[[0, 0, 0]])),)
        ),
        step_dir=ctx.step_dir,
        chemrefine_version="v",
        reuse_fingerprint=reuse_fp,
    )


def test_nms_reuse_outcome_none_without_cache(tmp_path: Path, monkeypatch):
    from chemrefine import step
    from chemrefine.engines.api import get_engine

    _pin_nms(monkeypatch)
    ctx = _ctx(tmp_path, nms=True, engine="orca")
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    assert step._nms_reuse_outcome(ctx, ctx.step_cfg, (), get_engine("orca")) is None


def test_nms_reuse_outcome_none_on_corrupt_cache(tmp_path: Path, monkeypatch):
    from chemrefine import cache, step
    from chemrefine.engines.api import get_engine

    _pin_nms(monkeypatch)
    ctx = _ctx(tmp_path, nms=True, engine="orca")
    cache_path = cache._cache_path(ctx.step_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"not json")
    assert step._nms_reuse_outcome(ctx, ctx.step_cfg, (), get_engine("orca")) is None


def test_nms_reuse_outcome_restamps_when_all_resolved(tmp_path: Path, monkeypatch):
    from chemrefine import step
    from chemrefine.engines.api import get_engine

    _pin_nms(monkeypatch, "FP")
    ctx = _ctx(tmp_path, nms=True, engine="orca")
    _save_reuse_cache(ctx, "FP")  # matching reuse fingerprint, no failed ledger
    out = step._nms_reuse_outcome(ctx, ctx.step_cfg, (), get_engine("orca"))
    assert out is not None and out.cache_hit is False


def test_nms_reuse_outcome_reattempts_when_ledger_present(tmp_path: Path, monkeypatch):
    from chemrefine import cache, step
    from chemrefine.engines.api import get_engine

    _pin_nms(monkeypatch, "FP")
    ctx = _ctx(tmp_path, nms=True, engine="orca")
    _save_reuse_cache(ctx, "FP")
    cache.save_failed_jobs(ctx.step_dir, [{"structure_id": "0", "kind": "failed", "reason": "x"}])
    out = step._nms_reuse_outcome(ctx, ctx.step_cfg, (), get_engine("orca"))
    assert out is not None and any(s.id == "re" for s in out.state.structures)
