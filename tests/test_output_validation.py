"""Native-product validation participates in completion, resume and parse-only recovery."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from ase import Atoms
from fake_engine import FakeEngine

from chemrefine import cache
from chemrefine.config import Config, StepConfig
from chemrefine.errors import NoUsableCacheError, OutputParseError
from chemrefine.state import JobBatch, PipelineState, StepInputs, StepResults, Structure
from chemrefine.step import (
    StepMode,
    _finish_artifact_step,
    _validate_cached_outputs,
    build_context,
    derive_step_key,
    rebuild_cache_step,
    run_step,
)


class _BundleEngine(FakeEngine):
    """A structure engine whose native result requires a separate numeric payload."""

    def __init__(self, *, broken=()):
        """Choose which simulated jobs produce a bad payload."""
        self.broken = set(broken)
        self.submitted = []
        self.validated = []

    def submit(self, inputs, ctx):
        """Produce ordinary structure outputs and mandatory external payloads."""
        batch = super().submit(inputs, ctx)
        for _, out, sid in inputs.files:
            self.submitted.append(sid)
            out.with_suffix(".bin").write_bytes(b"bad" if sid in self.broken else b"valid")
        return batch

    def validate_outputs(self, inputs, ctx):
        """Reject missing or mismatched payloads without modifying any on-disk state."""
        for _, out, sid in inputs.files:
            self.validated.append(sid)
            try:
                valid = out.with_suffix(".bin").read_bytes() == b"valid"
            except OSError as e:
                raise OutputParseError(f"payload missing for {sid}") from e
            if not valid:
                raise OutputParseError(f"payload mismatch for {sid}")


class _ArtifactEngine(_BundleEngine):
    """One non-structure product whose completed ensemble passes through unchanged."""

    def run_dir(self, ctx):
        """The artifact job's archiveable run directory."""
        return ctx.step_dir / "experiment"

    def artifact(self, ctx):
        """The deterministic product location, independent of a provider response."""
        return self.run_dir(ctx) / "result.json"

    def prepare(self, ctx):
        """Create a single non-structure job input."""
        self.run_dir(ctx).mkdir(parents=True, exist_ok=True)
        inp = self.run_dir(ctx) / "experiment.inp"
        inp.write_text("experiment")
        return StepInputs(files=((inp, self.artifact(ctx), "experiment"),))

    def submit(self, inputs, ctx):
        """Write the standalone descriptor and payload."""
        self.submitted.append("experiment")
        self.artifact(ctx).write_text("{}")
        self.artifact(ctx).with_suffix(".bin").write_bytes(
            b"bad" if "experiment" in self.broken else b"valid"
        )
        return JobBatch(jobs={})

    def parse(self, inputs, ctx):
        """Do not assign artificial energies or lineage to experiment products."""
        return StepResults(structures=ctx.prev_state.structures)


def _config(base):
    """An ordinary one-step local configuration."""
    return Config(output_dir=base / "outputs", steps=[StepConfig(step=1, engine="fake")])


def _seeds():
    """Two distinguishable structures expose accidental whole-step recomputation."""
    return PipelineState(structures=tuple(Structure(id=str(i), atoms=Atoms("H")) for i in range(2)))


def _inputs(config):
    """Read exactly the native files the production lifecycle recorded."""
    return cache.load_manifest(config.step_dir(config.steps[0]))


def test_invalid_payload_at_completion_becomes_a_real_job_failure(tmp_path):
    """A readable scalar output cannot hide a corrupt required RDM/circuit payload."""
    config = _config(tmp_path)
    engine = _BundleEngine(broken=("0",))
    result = run_step(config, config.steps[0], _seeds(), engine=engine)
    assert [s.id for s in result.state.structures] == ["1"]
    failures = cache.load_failure_records(config.step_dir(config.steps[0]))
    assert [f.structure_id for f in failures] == ["0"]
    assert "payload mismatch" in failures[0].reason


@pytest.mark.parametrize("remove", [False, True])
def test_resume_repairs_only_the_structure_with_an_invalid_payload(tmp_path, remove):
    """Payload corruption invalidates reuse without recomputing unaffected structures."""
    config = _config(tmp_path)
    engine = _BundleEngine()
    run_step(config, config.steps[0], _seeds(), engine=engine)
    payload = _inputs(config).files[0][1].with_suffix(".bin")
    if remove:
        payload.unlink()
    else:
        payload.write_bytes(b"corrupt")
    result = run_step(config, config.steps[0], _seeds(), engine=engine)
    assert not result.cache_hit
    assert engine.submitted == ["0", "1", "0"]
    assert payload.read_bytes() == b"valid"
    assert run_step(config, config.steps[0], _seeds(), engine=engine).cache_hit


def test_cache_only_refuses_corrupt_payload_without_submitting_or_rewriting(tmp_path):
    """A read-only recovery action cannot repair an invalid result by contacting a backend."""
    config = _config(tmp_path)
    engine = _BundleEngine()
    run_step(config, config.steps[0], _seeds(), engine=engine)
    output = _inputs(config).files[0][1]
    output.with_suffix(".bin").write_bytes(b"corrupt")
    original = (config.step_dir(config.steps[0]) / "_cache" / "step.json").read_bytes()
    with pytest.raises(NoUsableCacheError, match=r"payload mismatch.*cannot submit"):
        run_step(config, config.steps[0], _seeds(), engine=engine, mode=StepMode.CACHE_ONLY)
    assert engine.submitted == ["0", "1"]
    assert original == (config.step_dir(config.steps[0]) / "_cache" / "step.json").read_bytes()


def test_cached_known_failures_do_not_promise_missing_products(tmp_path):
    """CACHE_ONLY preserves the existing failed-row policy instead of failing on that row."""
    config = _config(tmp_path)
    engine = _BundleEngine(broken=("0",))
    run_step(config, config.steps[0], _seeds(), engine=engine)
    engine.validated.clear()
    outcome = run_step(config, config.steps[0], _seeds(), engine=engine, mode=StepMode.CACHE_ONLY)
    assert outcome.cache_hit
    assert engine.validated == ["1"]
    assert engine.submitted == ["0", "1"]


def test_rebuild_classifies_invalid_per_structure_payloads_without_submission(tmp_path):
    """Rebuild reparses local files, marks bad rows failed, and never computes replacements."""
    config = _config(tmp_path)
    engine = _BundleEngine()
    run_step(config, config.steps[0], _seeds(), engine=engine)
    _inputs(config).files[0][1].with_suffix(".bin").unlink()
    with patch("chemrefine.step.get_engine", return_value=engine):
        rebuilt = rebuild_cache_step(config, config.steps[0], _seeds())
    assert [s.id for s in rebuilt.state.structures] == ["1"]
    assert engine.submitted == ["0", "1"]
    assert cache.load_failure_records(config.step_dir(config.steps[0]))[0].structure_id == "0"


def test_artifact_validation_failure_does_not_create_success_cache(tmp_path):
    """One failed ensemble artifact is not a zero-structure successful calculation."""
    config = _config(tmp_path)
    engine = _ArtifactEngine(broken=("experiment",))
    with pytest.raises(OutputParseError, match="payload mismatch"):
        run_step(config, config.steps[0], _seeds(), engine=engine)
    assert cache.load(config.step_dir(config.steps[0])) is None


@pytest.mark.parametrize("missing_descriptor", [False, True])
def test_resume_recreates_missing_or_corrupt_artifact_product(tmp_path, missing_descriptor):
    """Artifact products are checked even when their pass-through structure cache is valid."""
    config = _config(tmp_path)
    engine = _ArtifactEngine()
    run_step(config, config.steps[0], _seeds(), engine=engine)
    ctx = build_context(config, config.steps[0], _seeds(), engine)
    if missing_descriptor:
        engine.artifact(ctx).unlink()
    else:
        engine.artifact(ctx).with_suffix(".bin").write_bytes(b"corrupt")
    result = run_step(config, config.steps[0], _seeds(), engine=engine)
    assert not result.cache_hit
    assert engine.submitted == ["experiment", "experiment"]
    assert result.state.structures == _seeds().structures


def test_rebuild_refuses_corrupt_artifact_and_accepts_repaired_local_product(tmp_path):
    """Rebuild may adopt a valid product after driver interruption but never manufacture it."""
    config = _config(tmp_path)
    engine = _ArtifactEngine()
    run_step(config, config.steps[0], _seeds(), engine=engine)
    payload = _inputs(config).files[0][1].with_suffix(".bin")
    payload.write_bytes(b"corrupt")
    cache.invalidate(config.step_dir(config.steps[0]))
    with patch("chemrefine.step.get_engine", return_value=engine):
        with pytest.raises(OutputParseError, match="payload mismatch"):
            rebuild_cache_step(config, config.steps[0], _seeds())
        assert cache.load(config.step_dir(config.steps[0])) is None
        payload.write_bytes(b"valid")
        rebuilt = rebuild_cache_step(config, config.steps[0], _seeds())
    assert rebuilt.state.structures == _seeds().structures
    assert engine.submitted == ["experiment"]


@pytest.mark.parametrize("empty", [False, True])
def test_validating_cache_requires_a_manifest_identifying_the_native_products(tmp_path, empty):
    """A valid structure cache cannot identify external payloads after losing its manifest."""
    config = _config(tmp_path)
    engine = _BundleEngine()
    run_step(config, config.steps[0], _seeds(), engine=engine)
    ctx = build_context(config, config.steps[0], _seeds(), engine)
    if empty:
        cache.save_manifest(StepInputs(files=()), ctx.step_dir, operation=None, engine="fake")
    else:
        cache.manifest_path(ctx.step_dir).unlink()
    with pytest.raises(OutputParseError, match="no job manifest"):
        _validate_cached_outputs(engine, ctx, set())


@pytest.mark.parametrize("empty", [False, True])
def test_artifact_completion_requires_its_native_job_manifest(tmp_path, empty):
    """Completion cannot validate a product using an absent or empty job list."""
    config = _config(tmp_path)
    engine = _ArtifactEngine()
    ctx = build_context(config, config.steps[0], _seeds(), engine)
    engine.submit(engine.prepare(ctx), ctx)
    if empty:
        cache.save_manifest(StepInputs(files=()), ctx.step_dir, operation=None, engine="fake")
    with pytest.raises(OutputParseError, match="no job manifest"):
        _finish_artifact_step(
            ctx, config.steps[0], derive_step_key(ctx, config.steps[0], engine), engine
        )
