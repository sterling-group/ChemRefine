"""Full-pipeline replays of captured real runs (see ``tests/replay.py``).

Every test drives ``chemrefine.pipeline.run`` with the shipped engines doing
their real prepare / parse / NMS / filter / cache work; only the compute is
satisfied from a ``tests/data/e2e`` archive. Assertions are data-driven —
recomputed from the archive with the real parsers — so re-capturing the
fixtures never stales them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from replay import ReplayCase, ReplaySubmitter, extract_case, replay_run_batch

from chemrefine import pipeline
from chemrefine.config import Config, load_config
from chemrefine.engines.orca.output import parse_output
from chemrefine.io import read_xyz_frames

RUN_BATCH = "chemrefine.engines._execution.run_batch"


def _with_step_update(config: Config, step_number: int, **updates: Any) -> Config:
    """A config copy with one step's fields replaced."""
    steps = [
        step.model_copy(update=updates) if step.step == step_number else step
        for step in config.steps
    ]
    return config.model_copy(update={"steps": steps})


def _with_backend_python(config: Config, step_number: int) -> Config:
    """Point a backend step at a dummy interpreter so preflight passes without torch."""
    step = config.find_step(step_number)
    options = dict(step.options or {})
    options["backend_python"] = "python"
    return _with_step_update(config, step_number, options=options)


def _replay(
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ReplayCase, ReplaySubmitter, Config]:
    case = extract_case(name, tmp_path)
    submitter = replay_run_batch(case)
    monkeypatch.setattr(RUN_BATCH, submitter)
    return case, submitter, load_config(case.config_path)


# ---------------------------------------------------------------------------
# conformers: goat ensemble -> opt+freq with Gibbs sorting -> sp
# ---------------------------------------------------------------------------


def test_conformers_full_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, submitter, config = _replay("conformers", tmp_path, monkeypatch)
    ensemble = read_xyz_frames(next(case.captured.glob("step1/*/step1_*.finalensemble.xyz")))
    assert len(ensemble) >= 3, "the GOAT ensemble must feed the downstream count: 3"

    outcomes = pipeline.run(config)

    assert len(outcomes) == 3
    fan_out = len(submitter.calls[1].files)
    assert 1 <= fan_out <= len(ensemble), "step 2 runs the boltzmann survivors"
    assert len(outcomes[1].state.structures) == min(3, fan_out)
    assert len(outcomes[2].state.structures) == 1
    csv_text = (case.output_dir / "steps.csv").read_text()
    assert csv_text.count("\n") >= 4, "steps.csv gains rows for all three steps"


def test_conformers_sorts_by_gibbs_not_electronic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, _submitter, config = _replay("conformers", tmp_path, monkeypatch)
    outcomes = pipeline.run(config)

    parsed: dict[str, tuple[float, float]] = {}
    for out in case.captured.glob("step2/*/step2_*.out"):
        result = parse_output(out, "opt_sp")[0]
        assert result.gibbs_hartree is not None
        assert result.energy_hartree is not None
        parsed[out.parent.name] = (
            result.gibbs_hartree,
            result.energy_hartree,
        )
    ran = set(parsed)
    survivors = [s.id for s in outcomes[1].state.structures]
    expected = sorted(ran, key=lambda sid: parsed[sid][0])[: len(survivors)]
    assert survivors == expected, "step-2 survivors must be the lowest-Gibbs structures"
    for structure in outcomes[1].state.structures:
        assert structure.gibbs_hartree is not None


def test_conformers_inputs_carry_charge_and_clamped_pal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, _submitter, config = _replay("conformers", tmp_path, monkeypatch)
    pipeline.run(config)

    goat_inp = (case.output_dir / "step1" / "0" / "step1_0.inp").read_text()
    assert f"nprocs {config.max_cores}" in goat_inp, "the template's 16 cores are clamped"
    assert "nprocs 16" not in goat_inp
    assert "* xyzfile 0 1 " in goat_inp
    step2_inp = next(case.output_dir.glob("step2/*/step2_*.inp")).read_text()
    assert "* xyzfile 0 1 " in step2_inp


def test_conformers_second_run_hits_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _case, submitter, config = _replay("conformers", tmp_path, monkeypatch)
    first = pipeline.run(config)
    calls_after_first = len(submitter.calls)

    second = pipeline.run(config)

    assert all(outcome.cache_hit for outcome in second)
    assert len(submitter.calls) == calls_after_first, "a cache hit submits nothing"
    assert [s.id for s in second[-1].state.structures] == [s.id for s in first[-1].state.structures]


# ---------------------------------------------------------------------------
# nms_minimum: a saddle-point seed resolved to a true minimum by NMS round 2
# ---------------------------------------------------------------------------


def test_nms_resolves_saddle_via_round_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, submitter, config = _replay("nms_minimum", tmp_path, monkeypatch)
    round_one = parse_output(case.captured / "step1" / "0" / "step1_0.out", "opt_sp")[0]
    assert round_one.imaginary_freqs, "the captured seed must be a saddle"

    outcomes = pipeline.run(config)

    children = sorted(p.name for p in (case.output_dir / "step1" / "0" / "attempt1").iterdir())
    assert children and all(name.startswith("0_m") for name in children)
    assert {sid for c in submitter.calls for _i, _o, sid in c.files} >= set(children)
    child_records = list((case.output_dir / "step1" / "0" / "attempt1").glob("*/*.result.json"))
    assert child_records, "round-2 children leave canonical result records too"
    (survivor,) = outcomes[0].state.structures
    assert survivor.id == "0", "the resolved child is written back under the parent id"
    assert survivor.converged
    assert not survivor.imaginary_freqs
    assert len(outcomes[1].state.structures) == 1


# ---------------------------------------------------------------------------
# ts_pes: max-sampling a scan, then an OptTS gated by the inferred ts target
# ---------------------------------------------------------------------------


def test_ts_pes_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, submitter, config = _replay("ts_pes", tmp_path, monkeypatch)
    scan_points = parse_output(case.captured / "step1" / "0" / "step1_0.out", "pes")
    assert len(scan_points) >= 3, "the scan must resolve several points"

    outcomes = pipeline.run(config)

    assert len(submitter.calls[1].files) == 1, "max count: 1 keeps one TS candidate"
    energies = [p.energy_hartree for p in scan_points]
    (ts_seed,) = outcomes[0].state.structures
    assert ts_seed.energy_hartree == max(e for e in energies if e is not None)
    (ts,) = outcomes[1].state.structures
    assert len(ts.imaginary_freqs) == 1, "a first-order saddle passes the ts target"
    ts_dir = case.output_dir / "step2" / ts.id
    assert not list(ts_dir.glob("attempt*")), "a clean TS needs no NMS round 2"


# ---------------------------------------------------------------------------
# host_guest: docker fan-out, per-step charge override, solvator parse
# ---------------------------------------------------------------------------


def test_host_guest_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case, submitter, config = _replay("host_guest", tmp_path, monkeypatch)
    poses = parse_output(case.captured / "step1" / "0" / "step1_0.out", "docker")
    assert len(poses) >= 2, "docking must produce poses to filter"

    outcomes = pipeline.run(config)

    kept = len(outcomes[0].state.structures)
    assert kept == min(3, len(poses))
    if kept > 1:
        assert {s.id for s in outcomes[0].state.structures} <= {
            f"0-{i}" for i in range(len(poses))
        }, "docker children carry the seed's lineage"
    step2_inp = next(case.output_dir.glob("step2/*/step2_*.inp")).read_text()
    assert "* xyzfile -1 1 " in step2_inp, "the per-step charge override reaches the input"
    assert len(submitter.calls[1].files) == kept
    assert outcomes[1].state.structures, "solvator output parses into survivors"


# ---------------------------------------------------------------------------
# mlip_screen / mlip_extopt: option carry-through into rendered inputs
# ---------------------------------------------------------------------------


def test_mlip_screen_renders_options_and_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, submitter, config = _replay("mlip_screen", tmp_path, monkeypatch)
    config = _with_backend_python(config, 1)
    seed_frames = read_xyz_frames(case.root / "glycol2.xyz")
    assert len(seed_frames) == 2, "the multi-frame seed drives two jobs"

    outcomes = pipeline.run(config)

    assert len(submitter.calls[0].files) == 2
    rendered = (case.output_dir / "step1" / "0" / "step1_0.py").read_text()
    assert '"small"' in rendered, "the model alias reaches the script"
    assert '"mace_off"' in rendered, "the task alias reaches the script"
    assert '"cpu"' in rendered, "device: cpu reaches the script"
    assert len(outcomes[0].state.structures) == 1
    assert outcomes[0].state.structures[0].energy_hartree is not None


def test_mlip_extopt_renders_server_block_and_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case, _submitter, config = _replay("mlip_extopt", tmp_path, monkeypatch)
    config = _with_backend_python(config, 1)

    outcomes = pipeline.run(config)

    rendered = (case.output_dir / "step1" / "0" / "step1_0.inp").read_text()
    assert "ProgExt" in rendered, "ExtOpt inputs must point ORCA at the gradient wrapper"
    (survivor,) = outcomes[0].state.structures
    assert survivor.converged
    assert survivor.energy_hartree is not None
