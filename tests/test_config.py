"""Tests for the Pydantic v4 ChemRefine config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from chemrefine.config import (
    BoltzmannSample,
    Config,
    EnergyWindowSample,
    IntegerSample,
    StepConfig,
    load_config,
)
from chemrefine.errors import ConfigError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config(**overrides) -> dict:
    base = {
        "steps": [
            {"step": 1, "engine": "fake", "operation": "opt_sp"},
        ],
    }
    base.update(overrides)
    return base


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "input.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Top-level config validation
# ---------------------------------------------------------------------------


def test_minimal_config_loads(tmp_path: Path):
    cfg = load_config(_write_yaml(tmp_path, _minimal_config()))
    assert cfg.charge == 0
    assert cfg.multiplicity == 1
    assert cfg.max_cores == 32
    assert len(cfg.steps) == 1
    assert cfg.steps[0].step == 1
    assert cfg.steps[0].engine == "fake"


def test_unknown_top_level_field_rejected(tmp_path: Path):
    data = _minimal_config(orca_excutable="orca")  # historic typo
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does-not-exist.yaml")


def test_non_mapping_yaml_rejected(tmp_path: Path):
    p = tmp_path / "input.yaml"
    p.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_malformed_yaml_rejected(tmp_path: Path):
    p = tmp_path / "input.yaml"
    p.write_text("steps: [\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


# ---------------------------------------------------------------------------
# Step ordering + uniqueness
# ---------------------------------------------------------------------------


def test_steps_must_be_contiguous(tmp_path: Path):
    data = _minimal_config(
        steps=[
            {"step": 1, "engine": "fake", "operation": "opt_sp"},
            {"step": 3, "engine": "fake", "operation": "opt_sp"},
        ]
    )
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))


def test_steps_must_be_one_indexed(tmp_path: Path):
    data = _minimal_config(
        steps=[{"step": 0, "engine": "fake", "operation": "opt_sp"}]
    )
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))


def test_step_names_must_be_unique(tmp_path: Path):
    data = _minimal_config(
        steps=[
            {"step": 1, "name": "refine", "engine": "fake", "operation": "opt_sp"},
            {"step": 2, "name": "refine", "engine": "fake", "operation": "opt_sp"},
        ]
    )
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))


def test_step_name_rejects_path_separator(tmp_path: Path):
    data = _minimal_config(
        steps=[{"step": 1, "name": "bad/name", "engine": "fake", "operation": "opt_sp"}]
    )
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))


def test_step_name_validator_accepts_explicit_none():
    """Explicit `name=None` must pass the validator's early-return branch."""
    sc = StepConfig(step=1, name=None, engine="fake", operation="opt_sp")
    assert sc.name is None


def test_empty_steps_list_rejected(tmp_path: Path):
    data = _minimal_config(steps=[])
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))


# ---------------------------------------------------------------------------
# Directory naming + step lookup
# ---------------------------------------------------------------------------


def test_dir_name_without_label():
    sc = StepConfig(step=2, engine="fake", operation="opt_sp")
    assert sc.dir_name() == "step2"


def test_dir_name_with_label():
    sc = StepConfig(step=2, name="refine", engine="fake", operation="opt_sp")
    assert sc.dir_name() == "step2_refine"


def test_config_step_dir_joins_output_dir():
    cfg = Config(
        output_dir=Path("/tmp/out"),
        steps=[StepConfig(step=1, name="screen", engine="fake", operation="opt_sp")],
    )
    assert cfg.step_dir(cfg.steps[0]) == Path("/tmp/out/step1_screen")


def test_find_step_by_number_and_name():
    cfg = Config(
        steps=[
            StepConfig(step=1, name="screen", engine="fake", operation="opt_sp"),
            StepConfig(step=2, name="refine", engine="fake", operation="opt_sp"),
        ]
    )
    assert cfg.find_step(2) is cfg.steps[1]
    assert cfg.find_step("2") is cfg.steps[1]
    assert cfg.find_step("refine") is cfg.steps[1]
    assert cfg.find_step("missing") is None


# ---------------------------------------------------------------------------
# Sample discriminated union
# ---------------------------------------------------------------------------


def test_boltzmann_sample_parses(tmp_path: Path):
    data = _minimal_config(
        steps=[
            {
                "step": 1,
                "engine": "fake",
                "operation": "opt_sp",
                "sample": {"method": "boltzmann", "percent_cumulative": 95},
            }
        ]
    )
    cfg = load_config(_write_yaml(tmp_path, data))
    sample = cfg.steps[0].sample
    assert isinstance(sample, BoltzmannSample)
    assert sample.percent_cumulative == 95


def test_energy_window_sample_requires_window(tmp_path: Path):
    data = _minimal_config(
        steps=[
            {
                "step": 1,
                "engine": "fake",
                "operation": "opt_sp",
                "sample": {"method": "energy_window"},
            }
        ]
    )
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))


def test_integer_sample_parses(tmp_path: Path):
    data = _minimal_config(
        steps=[
            {
                "step": 1,
                "engine": "fake",
                "operation": "opt_sp",
                "sample": {"method": "integer", "count": 5},
            }
        ]
    )
    cfg = load_config(_write_yaml(tmp_path, data))
    assert isinstance(cfg.steps[0].sample, IntegerSample)
    assert cfg.steps[0].sample.count == 5


def test_energy_window_sample_parses(tmp_path: Path):
    data = _minimal_config(
        steps=[
            {
                "step": 1,
                "engine": "fake",
                "operation": "opt_sp",
                "sample": {"method": "energy_window", "window_kcal": 3.0},
            }
        ]
    )
    cfg = load_config(_write_yaml(tmp_path, data))
    assert isinstance(cfg.steps[0].sample, EnergyWindowSample)
    assert cfg.steps[0].sample.window_kcal == 3.0


def test_unknown_sample_method_rejected(tmp_path: Path):
    data = _minimal_config(
        steps=[
            {
                "step": 1,
                "engine": "fake",
                "operation": "opt_sp",
                "sample": {"method": "random_walk"},
            }
        ]
    )
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))


# ---------------------------------------------------------------------------
# Per-step options + overrides
# ---------------------------------------------------------------------------


def test_step_options_round_trip(tmp_path: Path):
    data = _minimal_config(
        steps=[
            {
                "step": 1,
                "engine": "mlff",
                "operation": "opt_sp",
                "options": {"model": "mace_off23", "device": "cuda"},
            }
        ]
    )
    cfg = load_config(_write_yaml(tmp_path, data))
    assert cfg.steps[0].options == {"model": "mace_off23", "device": "cuda"}


def test_step_charge_override(tmp_path: Path):
    data = _minimal_config(
        charge=0,
        steps=[
            {"step": 1, "engine": "fake", "operation": "opt_sp", "charge": -1},
        ],
    )
    cfg = load_config(_write_yaml(tmp_path, data))
    assert cfg.charge == 0
    assert cfg.steps[0].charge == -1


def test_step_unknown_field_rejected(tmp_path: Path):
    data = _minimal_config(
        steps=[
            {"step": 1, "engine": "fake", "operation": "opt_sp", "wat": "huh"},
        ],
    )
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))
