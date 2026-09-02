"""Tests for the Pydantic v4 ChemRefine config loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from chemrefine.config import (
    BoltzmannSample,
    Config,
    MaxSample,
    MinSample,
    StepConfig,
    load_config,
)
from chemrefine.errors import ConfigError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _silent(loc: tuple[str | int, ...], message: str) -> None:
    """A deprecation sink that swallows — for helpers called directly, off the report path."""


def _collect() -> tuple[list[tuple[tuple[str | int, ...], str]], object]:
    """A sink plus the list it fills, for asserting what a rewrite announced and where."""
    seen: list[tuple[tuple[str | int, ...], str]] = []
    return seen, lambda loc, message: seen.append((loc, message))


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
    assert cfg.max_cores == 4
    assert cfg.max_gpus is None  # auto-resolved at submit time
    assert len(cfg.steps) == 1
    assert cfg.steps[0].step == 1
    assert cfg.steps[0].engine == "fake"
    assert cfg.steps[0].slurm_template is None  # falls back to the global header


def test_operation_is_optional(tmp_path: Path):
    """A step may omit ``operation`` — engines that inspect their input fill it in."""
    data = _minimal_config(steps=[{"step": 1, "engine": "fake"}])
    cfg = load_config(_write_yaml(tmp_path, data))
    assert cfg.steps[0].operation is None


def test_max_gpus_and_per_step_slurm_template_accepted(tmp_path: Path):
    data = _minimal_config(max_gpus=2)
    data["steps"][0]["slurm_template"] = "cuda.slurm.header"
    cfg = load_config(_write_yaml(tmp_path, data))
    assert cfg.max_gpus == 2
    assert cfg.steps[0].slurm_template == "cuda.slurm.header"


def test_unknown_top_level_field_rejected(tmp_path: Path):
    data = _minimal_config(orca_excutable="orca")  # a plausible misspelling
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


def test_non_string_keys_are_a_config_error(tmp_path: Path):
    """An unquoted ``on:`` key (a YAML 1.1 boolean) gets the documented error, not a TypeError."""
    p = tmp_path / "input.yaml"
    p.write_text("on: true\nsteps: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid config"):
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
    data = _minimal_config(steps=[{"step": 0, "engine": "fake", "operation": "opt_sp"}])
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


def test_step_name_rejects_all_digit_names(tmp_path: Path):
    """A digits-only name is unreachable — CLI keys resolve digits as step numbers."""
    data = _minimal_config(
        steps=[{"step": 1, "name": "2", "engine": "fake", "operation": "opt_sp"}]
    )
    with pytest.raises(ConfigError, match="all digits"):
        load_config(_write_yaml(tmp_path, data))


def test_step_name_validator_accepts_explicit_none():
    """Explicit `name=None` must pass the validator's early-return branch."""
    sc = StepConfig(step=1, name=None, engine="fake", operation="opt_sp")
    assert sc.name is None


def test_step_options_reject_a_yaml_date(tmp_path: Path):
    """An unquoted date in `options` fails at load, naming the key — not as a
    TypeError traceback from inside the cache fingerprint three layers later."""
    p = tmp_path / "input.yaml"
    p.write_text(
        "steps:\n  - step: 1\n    engine: fake\n    options: { calibration_date: 2024-01-01 }\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="calibration_date"):
        load_config(p)


def test_step_options_accept_nested_json_values():
    """Anything JSON can encode — nested dicts, lists, floats, bools — stays legal."""
    sc = StepConfig(
        step=1,
        engine="fake",
        options={"model": "uma-s-1p2", "windows": [0.5, 1.0], "flags": {"gpu": True}},
    )
    assert sc.options["windows"] == [0.5, 1.0]


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
    # `"²".isdigit()` is True and `int("²")` raises, so the old guard routed this into a
    # bare ValueError — out through `start_run` and the GUI past the contract that every
    # failure is a ChemRefineError with an exit code. It is simply not a step name.
    assert cfg.find_step("²") is None
    # Unicode decimals that `int` *does* accept still resolve, which is why the predicate
    # is `isdecimal` rather than an ASCII test.
    assert cfg.find_step("٢") is cfg.steps[1]


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


def _sample_cfg(tmp_path: Path, sample: dict) -> Config:
    """Load a minimal one-step config carrying ``sample``."""
    data = _minimal_config(
        steps=[{"step": 1, "engine": "fake", "operation": "opt_sp", "sample": sample}]
    )
    return load_config(_write_yaml(tmp_path, data))


def test_min_sample_requires_exactly_one_selector(tmp_path: Path):
    # neither count nor window_kcalmol
    with pytest.raises(ConfigError):
        _sample_cfg(tmp_path, {"method": "min"})
    # both at once
    with pytest.raises(ConfigError):
        _sample_cfg(tmp_path, {"method": "min", "count": 5, "window_kcalmol": 3.0})


def test_max_sample_requires_exactly_one_selector(tmp_path: Path):
    with pytest.raises(ConfigError):
        _sample_cfg(tmp_path, {"method": "max"})
    with pytest.raises(ConfigError):
        _sample_cfg(tmp_path, {"method": "max", "count": 5, "window_kcalmol": 3.0})


def test_the_shared_selector_rule_still_discriminates_and_keeps_its_own_bounds(tmp_path: Path):
    """`min` and `max` share one selector rule without collapsing into one another.

    The rule lives on a common base now — it was written out twice, differing only in the
    prefix of its message, which is how one copy gets tightened and the other does not. A
    shared base could plausibly have broken either the `method` discriminator or the
    per-variant `count` floors, so both are asserted here rather than assumed.
    """
    assert isinstance(
        _sample_cfg(tmp_path, {"method": "min", "count": 0}).steps[0].sample, MinSample
    )
    assert isinstance(
        _sample_cfg(tmp_path, {"method": "max", "window_kcalmol": 3.0}).steps[0].sample, MaxSample
    )
    # `min: 0` means "keep everything"; `max: 0` would mean "keep nothing", so it is refused.
    with pytest.raises(ConfigError):
        _sample_cfg(tmp_path, {"method": "max", "count": 0})
    # The message names the method it came from, read off the discriminator.
    with pytest.raises(ConfigError, match="max: set exactly one"):
        _sample_cfg(tmp_path, {"method": "max"})
    with pytest.raises(ConfigError, match="min: set exactly one"):
        _sample_cfg(tmp_path, {"method": "min"})


def test_min_sample_parses(tmp_path: Path):
    cfg = _sample_cfg(tmp_path, {"method": "min", "count": 5})
    assert isinstance(cfg.steps[0].sample, MinSample)
    assert cfg.steps[0].sample.count == 5


def test_min_window_sample_parses(tmp_path: Path):
    cfg = _sample_cfg(tmp_path, {"method": "min", "window_kcalmol": 3.0})
    assert isinstance(cfg.steps[0].sample, MinSample)
    assert cfg.steps[0].sample.window_kcalmol == 3.0


def test_max_sample_parses(tmp_path: Path):
    cfg = _sample_cfg(tmp_path, {"method": "max", "count": 5})
    assert isinstance(cfg.steps[0].sample, MaxSample)
    assert cfg.steps[0].sample.count == 5


def test_max_window_sample_parses(tmp_path: Path):
    cfg = _sample_cfg(tmp_path, {"method": "max", "window_kcalmol": 3.0})
    assert isinstance(cfg.steps[0].sample, MaxSample)
    assert cfg.steps[0].sample.window_kcalmol == 3.0


def test_sample_energy_type_defaults_to_electronic(tmp_path: Path):
    cfg = _sample_cfg(tmp_path, {"method": "min", "count": 1})
    assert cfg.steps[0].sample.energy_type == "electronic"


def test_sample_energy_type_accepts_short_aliases(tmp_path: Path):
    for alias, canonical in [
        ("G", "gibbs"),
        ("H", "enthalpy"),
        ("E_ZPE", "electronic_zero_point"),
        ("Electronic", "electronic"),  # case-insensitive long name
    ]:
        cfg = _sample_cfg(tmp_path, {"method": "min", "count": 1, "energy_type": alias})
        assert cfg.steps[0].sample.energy_type == canonical


def test_sample_energy_type_rejects_unknown(tmp_path: Path):
    with pytest.raises(ConfigError):
        _sample_cfg(tmp_path, {"method": "min", "count": 1, "energy_type": "bogus"})


def test_sample_energy_type_rejects_non_string(tmp_path: Path):
    with pytest.raises(ConfigError):
        _sample_cfg(tmp_path, {"method": "min", "count": 1, "energy_type": 123})


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
                "engine": "mlip",
                "operation": "opt_sp",
                "options": {"model": "mace_off23", "device": "cuda"},
            }
        ]
    )
    cfg = load_config(_write_yaml(tmp_path, data))
    assert cfg.steps[0].options == {"model": "mace_off23", "device": "cuda"}


def test_a_relative_model_path_resolves_against_the_config_dir(tmp_path: Path):
    """The value reaches a job that runs in a scratch directory, where `./` means nothing.

    Same rule as `template_dir` / `output_dir`: relative paths in the config resolve against
    the config file's own directory, so `chemrefine run sub/proj/input.yaml` works from
    anywhere. A field validator could not do this — it sees only the process working
    directory — which is why the resolution lives beside the config's other paths.
    """
    data = _minimal_config(
        steps=[
            {
                "step": 1,
                "engine": "mlip",
                "operation": "sp",
                "options": {"model_path": "./outputs/step2/train/train.model"},
            }
        ]
    )
    cfg = load_config(_write_yaml(tmp_path, data))
    resolved = cfg.steps[0].options["model_path"]
    assert Path(resolved).is_absolute()
    assert resolved == str((tmp_path / "outputs/step2/train/train.model").resolve())


def test_an_absolute_model_path_passes_through_unchanged(tmp_path: Path):
    data = _minimal_config(
        steps=[
            {
                "step": 1,
                "engine": "mlip",
                "operation": "sp",
                "options": {"model_path": "/models/mine.model", "device": "cpu"},
            }
        ]
    )
    cfg = load_config(_write_yaml(tmp_path, data))
    assert cfg.steps[0].options["model_path"] == "/models/mine.model"


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


# ---------------------------------------------------------------------------
# scratch_dir: optional + auto-derived per-calc work dir
# ---------------------------------------------------------------------------


def test_scratch_dir_defaults_to_none(tmp_path: Path):
    """Omitting ``scratch_dir`` lets the SLURM script auto-derive a per-calc work dir."""
    cfg = load_config(_write_yaml(tmp_path, _minimal_config()))
    assert cfg.scratch_dir is None


def test_scratch_dir_explicit_path_round_trips(tmp_path: Path):
    data = _minimal_config(scratch_dir="/scratch/user")
    cfg = load_config(_write_yaml(tmp_path, data))
    assert cfg.scratch_dir == Path("/scratch/user")


def test_scratch_dir_equal_output_dir_rejected(tmp_path: Path):
    """``scratch_dir == output_dir`` is ambiguous; require ``None`` instead."""
    data = _minimal_config(scratch_dir="./outputs", output_dir="./outputs")
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))


@pytest.mark.parametrize("field", ["template_dir", "output_dir", "scratch_dir"])
@pytest.mark.parametrize("bad", ['./out"dir', "./out$dir", "./out`dir"])
def test_directory_paths_with_shell_metacharacters_rejected(tmp_path: Path, field: str, bad: str):
    """These paths are interpolated into the generated SLURM script.

    ``"`` / ``$`` / backtick would close the quoted string or open a command
    substitution, so they are refused at load time rather than producing a
    broken — or dangerous — job script at submit time.
    """
    data = _minimal_config(**{field: bad})
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))


@pytest.mark.parametrize(
    "bad", ["/opt/orca-$(id -un)/orca", "/opt/`whoami`/orca", '/opt/o"rca/orca', "/opt/o\\x/orca"]
)
def test_executables_with_shell_metacharacters_rejected(tmp_path: Path, bad: str):
    """`executables` reaches generated bash too, and one route does not quote it.

    The runlog header embeds the binary raw inside `cat <<EOF`, a heredoc that has to
    stay unquoted so `$(hostname)` and `$WORK_DIR` expand — so a `$(...)` in the
    configured path is a command substitution the job runs. Quoting the header is not
    available; refusing the character at the boundary is.
    """
    data = _minimal_config(executables={"orca": bad})
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))


@pytest.mark.parametrize("bad", ["opt_sp$(id -un)", "opt_sp`whoami`", 'opt_sp"x', "opt_sp\\x"])
def test_operation_with_shell_metacharacters_rejected(tmp_path: Path, bad: str):
    """`operation` lands in the same unquoted heredoc the executables rule exists for.

    `name` is regex-validated and `engine` is registry-checked, but `operation` was a free
    string interpolated raw into the runlog header — so `operation: opt_sp$(id -un)` was a
    command substitution the job executed. The rule belongs to the *concept* ("every config
    value that reaches generated bash"), not to the fields that happened to have it first.
    """
    data = _minimal_config()
    data["steps"][0]["operation"] = bad
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))


@pytest.mark.parametrize("ok", ["opt_sp", "OPT+SP", "sp", "mlip_train", "goat"])
def test_operation_accepts_every_real_spelling(tmp_path: Path, ok: str):
    """The rule blocks shell metacharacters only — not the vocabulary engines actually use.

    `OPT+SP` matters: it is the legacy spelling, normalised to `opt_sp` before this
    validator sees it, and a stricter allowlist would have rejected it for no security gain.
    """
    data = _minimal_config()
    data["steps"][0]["operation"] = ok
    assert load_config(_write_yaml(tmp_path, data)).steps[0].operation is not None


def test_directory_paths_with_a_comma_rejected(tmp_path: Path):
    """A comma is the tab rule's sibling: sbatch splits ``--export`` on it.

    The array path hands each chunk its manifest through
    ``--export=ALL,CR_MANIFEST=<path>``, so a comma anywhere in the path truncates the
    variable and every task reads a manifest that does not exist — a whole step ledgered
    ``output missing`` for a legal directory name. Refused at the boundary like the tab.
    """
    data = _minimal_config()
    data["output_dir"] = "./out,puts"
    with pytest.raises(ConfigError, match="comma"):
        load_config(_write_yaml(tmp_path, data))


@pytest.mark.parametrize("ok", ["/opt/my orca/orca", "/opt/orca/orca", "orca"])
def test_executables_allow_spaces_and_bare_command_names(tmp_path: Path, ok: str):
    """A space is not a shell hazard here — the run site quotes it — and paths have them.

    A bare command name stays legal too: it is resolved on the executing host, which is
    how a `module load` inside the job is meant to provide the binary.
    """
    data = _minimal_config(executables={"orca": ok})
    assert load_config(_write_yaml(tmp_path, data)).executables["orca"] == ok


def test_a_metacharacter_in_the_config_files_own_directory_is_refused(tmp_path: Path):
    """The shell rule is re-asked after resolution, because a parent directory can carry one.

    Reproduced before it was fixed: a config whose YAML contains nothing but
    `output_dir: ./outputs`, sitting in a directory named `$(touch /tmp/MARKER)`, produced
    `export OUTPUT_DIR="…/$(touch /tmp/MARKER)/outputs"` in the generated script — and bash
    substitutes inside double quotes, so running the job created the file. The field
    validator could not see it: it runs on the path as written, and `resolve_relative_paths`
    anchors relative paths through `model_copy`, which runs no validators.

    A directory name is the payload here, which is exactly the shape that matters on a
    shared filesystem — the environment `docs/internals/security.md` is written for.
    """
    hostile = tmp_path / "$(echo pwned)"
    hostile.mkdir()
    config = hostile / "input.yaml"
    config.write_text(yaml.safe_dump(_minimal_config()), encoding="utf-8")
    with pytest.raises(ConfigError, match="resolved"):
        load_config(config)


def test_a_clean_config_directory_still_loads(tmp_path: Path):
    """The re-check must not refuse an ordinary tree — it is a rule, not a mood."""
    project = tmp_path / "proj-1_ok"
    project.mkdir()
    config = project / "input.yaml"
    config.write_text(yaml.safe_dump(_minimal_config()), encoding="utf-8")
    assert load_config(config).output_dir == project / "outputs"


def test_explicit_null_scratch_dir_is_accepted(tmp_path: Path):
    """An explicit ``scratch_dir: null`` still means "auto-derive under output_dir"."""
    data = _minimal_config(scratch_dir=None)
    cfg = load_config(_write_yaml(tmp_path, data))
    assert cfg.scratch_dir is None


# ---------------------------------------------------------------------------
# Path resolution: relative paths resolve against the config file's directory
# ---------------------------------------------------------------------------


def test_load_config_resolves_relative_paths_against_config_dir(tmp_path: Path):
    """A portable config: relative dirs/input resolve next to the YAML, not the CWD."""
    data = _minimal_config(
        template_dir="./t",
        output_dir="./out",
        scratch_dir="./scr",
        input="./seed.xyz",
    )
    cfg = load_config(_write_yaml(tmp_path, data))
    base = tmp_path.resolve()
    assert cfg.template_dir == base / "t"
    assert cfg.output_dir == base / "out"
    assert cfg.scratch_dir == base / "scr"
    assert cfg.input == base / "seed.xyz"


def test_load_config_leaves_absolute_paths_unchanged(tmp_path: Path):
    """Absolute paths pass through resolution untouched."""
    data = _minimal_config(
        template_dir="/abs/t",
        output_dir="/abs/out",
        scratch_dir="/abs/scr",
        input="/abs/seed.xyz",
    )
    cfg = load_config(_write_yaml(tmp_path, data))
    assert cfg.template_dir == Path("/abs/t")
    assert cfg.output_dir == Path("/abs/out")
    assert cfg.scratch_dir == Path("/abs/scr")
    assert cfg.input == Path("/abs/seed.xyz")


# ---------------------------------------------------------------------------
# Legacy-YAML normalizer (v1.3.1 / mlff-named configs load + run)
# ---------------------------------------------------------------------------


def test_legacy_top_level_renames():
    cfg = Config(
        template_dir="./t",
        orca_executable="/orca",  # type: ignore[call-arg]
        initial_xyz="./seed.xyz",
        steps=[{"step": 1, "engine": "orca", "operation": "opt_sp"}],
    )
    assert cfg.executables == {"orca": "/orca"}
    assert cfg.input == Path("seed.xyz")


def test_legacy_engine_renames():
    cfg = Config(
        template_dir="./t",
        steps=[
            {"step": 1, "engine": "DFT", "operation": "OPT+SP"},
            {"step": 2, "engine": "mlff", "operation": "opt_sp"},
        ],
    )
    assert [s.engine for s in cfg.steps] == ["orca", "mlip"]
    assert cfg.steps[0].operation == "opt_sp"


def test_legacy_mlff_block_becomes_options_and_extopt_engine():
    cfg = Config(
        template_dir="./t",
        steps=[
            {
                "step": 1,
                "engine": "MLFF",
                "operation": "OPT+SP",
                "mlff": {
                    "model_name": "uma-s-1",
                    "task_name": "omol",
                    "device": "cuda",
                    "bind": "x:1",
                },
            }
        ],
    )
    s = cfg.steps[0]
    assert s.engine == "mlip-extopt"
    assert s.options == {"model_name": "uma-s-1", "task_name": "omol", "device": "cuda"}
    assert "bind" not in s.options  # obsolete sub-key dropped


def test_legacy_train_operation_selects_trainer():
    cfg = Config(
        template_dir="./t",
        steps=[{"step": 1, "operation": "MLFF_TRAIN"}],
    )
    assert cfg.steps[0].engine == "mlip-train"


def test_legacy_sample_type_and_param_renames():
    cfg = Config(
        template_dir="./t",
        steps=[
            {
                "step": 1,
                "engine": "orca",
                "operation": "sp",
                "sample_type": {"method": "boltzmann", "parameters": {"weight": 95}},
            },
            {
                "step": 2,
                "engine": "orca",
                "operation": "sp",
                "sample_type": {"method": "integer", "parameters": {"num_structures": 3}},
            },
            {
                "step": 3,
                "engine": "orca",
                "operation": "sp",
                "sample_type": {
                    "method": "energy_window",
                    "parameters": {"energy": 8, "unit": "kcal/mol"},
                },
            },
        ],
    )
    assert isinstance(cfg.steps[0].sample, BoltzmannSample)
    assert cfg.steps[0].sample.percent_cumulative == 95
    # integer -> min, num_structures -> count
    assert isinstance(cfg.steps[1].sample, MinSample)
    assert cfg.steps[1].sample.count == 3
    # energy_window -> min, energy -> window_kcalmol (an explicit kcal/mol value
    # crosses unchanged; every other unit converts -- see the tests below)
    assert isinstance(cfg.steps[2].sample, MinSample)
    assert cfg.steps[2].sample.window_kcalmol == 8


def _legacy_window_sample(parameters: dict) -> MinSample:
    """The validated sample a v1 ``energy_window`` block with ``parameters`` becomes."""
    cfg = Config(
        template_dir="./t",
        steps=[
            {
                "step": 1,
                "engine": "orca",
                "operation": "sp",
                "sample_type": {"method": "energy_window", "parameters": parameters},
            }
        ],
    )
    sample = cfg.steps[0].sample
    assert isinstance(sample, MinSample)
    return sample


def test_legacy_energy_window_value_keeps_its_hartree_meaning():
    """v1's ``energy`` was hartree unless ``unit: kcal/mol`` said otherwise — so it converts.

    ``window_kcalmol`` is kcal/mol by definition; carrying the number across unchanged
    shrank the window ~627.5x for every v1 config that relied on the default, silently
    discarding structures the run should have kept. The conversion mirrors v1's own rule:
    only an explicit ``kcal/mol`` converted anything there, so any other spelling —
    including none at all — meant the number was hartree.
    """
    implicit = _legacy_window_sample({"energy": 0.5})
    assert implicit.window_kcalmol == pytest.approx(313.754737, abs=1e-4)
    explicit = _legacy_window_sample({"energy": 0.5, "unit": "hartree"})
    assert explicit.window_kcalmol == pytest.approx(313.754737, abs=1e-4)
    # v1 compared any unit it did not recognise as hartree; mirrored, so a config
    # that ran on v1 filters identically here rather than differently-wrong.
    odd = _legacy_window_sample({"energy": 0.5, "unit": "eV"})
    assert odd.window_kcalmol == pytest.approx(313.754737, abs=1e-4)
    kcal = _legacy_window_sample({"energy": 8, "unit": "kcal/mol"})
    assert kcal.window_kcalmol == 8


def test_legacy_energy_window_conversion_names_both_numbers():
    """The deprecation row states the conversion, not just the rename.

    The rename row alone ("use `window_kcalmol`") reads as an endorsement of the value —
    which is exactly how a silently reinterpreted window would go unnoticed. A row that
    names the hartree number and the kcal/mol number it became is checkable at a glance.
    """
    from chemrefine.config_legacy import normalize as _normalize_legacy

    rows: list[tuple[tuple[str | int, ...], str]] = []
    _normalize_legacy(
        {
            "steps": [
                {
                    "step": 1,
                    "engine": "orca",
                    "sample": {"method": "energy_window", "energy": 0.5},
                }
            ]
        },
        report=lambda loc, message: rows.append((loc, message)),
    )
    [(loc, message)] = [(loc, m) for loc, m in rows if "window_kcalmol:" in m]
    assert loc == ("steps", 0, "sample", "energy")
    assert "0.5" in message and "313.755" in message


def test_legacy_energy_window_non_numeric_value_is_left_for_validation():
    """A value the conversion cannot read passes through for pydantic to refuse.

    v1 would have crashed on it at filter time; v2 refuses it at the boundary with the
    field error every other bad sample value gets — the conversion must not turn that
    refusal into its own arithmetic traceback.
    """
    with pytest.raises(ValidationError):
        Config(
            template_dir="./t",
            steps=[
                {
                    "step": 1,
                    "engine": "orca",
                    "operation": "sp",
                    "sample_type": {"method": "energy_window", "parameters": {"energy": [1]}},
                }
            ],
        )


def test_legacy_normal_mode_sampling_renamed():
    cfg = Config(
        template_dir="./t",
        steps=[{"step": 1, "engine": "orca", "operation": "freq", "normal_mode_sampling": True}],
    )
    # bare nms maps to target: ts (main's default calc_type was rm_imag).
    assert cfg.steps[0].nms is True
    assert cfg.steps[0].options == {"target": "ts"}


def test_legacy_nms_rm_imag_maps_to_ts_and_displacement():
    cfg = Config(
        template_dir="./t",
        steps=[
            {
                "step": 1,
                "engine": "orca",
                "operation": "freq",
                "normal_mode_sampling": True,
                "normal_mode_sampling_parameters": {
                    "calc_type": "rm_imag",
                    "displacement_vector": 1.5,
                },
            }
        ],
    )
    s = cfg.steps[0]
    assert s.nms is True
    assert s.options == {"target": "ts", "displacement_value": 1.5}
    # the renamed knobs are the ones NmsOptions reads.
    from chemrefine.nms import NmsOptions

    opts = NmsOptions.from_raw(s.options)
    assert (opts.target, opts.displacement_value) == ("ts", 1.5)


def test_legacy_nms_random_passes_params_through():
    cfg = Config(
        template_dir="./t",
        steps=[
            {
                "step": 1,
                "engine": "orca",
                "operation": "freq",
                "normal_mode_sampling": True,
                "normal_mode_sampling_parameters": {
                    "calc_type": "random",
                    "num_random_displacements": 3,
                },
            }
        ],
    )
    assert cfg.steps[0].options == {"target": "random", "num_random_displacements": 3}


def test_legacy_calculation_type_raises_clear_error():
    with pytest.raises(ConfigError, match="calculation_type"):
        Config(template_dir="./t", steps=[{"step": 1, "calculation_type": "DFT"}])


def test_legacy_initial_xyz_does_not_override_existing_input():
    """When both spellings are present, the new ``input`` wins; the legacy key is dropped."""
    cfg = Config(
        template_dir="./t",
        input="./new.xyz",
        initial_xyz="./old.xyz",  # type: ignore[call-arg]
        steps=[{"step": 1, "engine": "orca", "operation": "opt_sp"}],
    )
    assert cfg.input == Path("new.xyz")


def test_legacy_sample_type_does_not_override_existing_sample():
    """When both spellings are present, the new ``sample`` wins; ``sample_type`` is dropped."""
    cfg = Config(
        template_dir="./t",
        steps=[
            {
                "step": 1,
                "engine": "orca",
                "operation": "opt_sp",
                "sample": {"method": "integer", "count": 3},
                "sample_type": {"method": "boltzmann", "parameters": {"weight": 99}},
            }
        ],
    )
    # The direct `sample` block (legacy `integer` -> `min`) wins; sample_type dropped.
    assert isinstance(cfg.steps[0].sample, MinSample)
    assert cfg.steps[0].sample.count == 3


def test_legacy_nms_false_with_no_parameters_normalizes_to_nothing():
    """``normal_mode_sampling: false`` is consumed without setting nms or options."""
    cfg = Config(
        template_dir="./t",
        steps=[{"step": 1, "engine": "orca", "operation": "freq", "normal_mode_sampling": False}],
    )
    assert cfg.steps[0].nms is False
    assert cfg.steps[0].options == {}


def test_normalizer_passes_through_non_list_steps():
    """A non-list ``steps`` is left for Pydantic to reject with its own message."""
    from chemrefine.config_legacy import normalize as _normalize_legacy

    raw = {"steps": "not-a-list"}
    assert _normalize_legacy(raw) == raw


def test_normalizer_handles_step_without_operation():
    """Engine renames still apply when ``operation`` is absent (validation rejects later)."""
    from chemrefine.config_legacy import _normalize_step

    s = _normalize_step({"step": 1, "engine": "DFT"}, 0, _silent)
    assert s["engine"] == "orca"
    assert "operation" not in s


def test_normalizer_is_idempotent_on_new_style():
    new = {
        "template_dir": "./t",
        "executables": {"orca": "/orca"},
        "input": "./s.xyz",
        "steps": [
            {
                "step": 1,
                "engine": "mlip-extopt",
                "operation": "opt_sp",
                "options": {"model_name": "uma-s-1", "task_name": "omol"},
                "sample": {"method": "min", "count": 5},
            }
        ],
    }
    cfg = Config(**new)
    assert cfg.executables == {"orca": "/orca"}
    assert cfg.steps[0].engine == "mlip-extopt"
    assert cfg.steps[0].options == {"model_name": "uma-s-1", "task_name": "omol"}


# The shipped-example sweeps live in tests/test_examples.py.


def test_slurm_array_knob_defaults_off_and_loads(tmp_path: Path):
    """`slurm_array: true` opts a run into array submission; absent = today's path."""
    assert load_config(_write_yaml(tmp_path, _minimal_config())).slurm_array is False
    data = _minimal_config()
    data["slurm_array"] = True
    assert load_config(_write_yaml(tmp_path, data)).slurm_array is True


def test_dispatch_knob_defaults_auto_and_validates(tmp_path: Path):
    """`dispatch` defaults to auto, loads local/slurm, and rejects anything else."""
    assert load_config(_write_yaml(tmp_path, _minimal_config())).dispatch == "auto"
    data = _minimal_config()
    data["dispatch"] = "local"
    assert load_config(_write_yaml(tmp_path, data)).dispatch == "local"
    data["dispatch"] = "bogus"
    with pytest.raises(ConfigError):
        load_config(_write_yaml(tmp_path, data))


def test_config_rejects_a_newline_in_a_directory_path():
    """A newline ends the generated `export DIR="..."` and makes the rest a command."""
    with pytest.raises(ValidationError):
        Config(
            output_dir=Path("outputs\nrm -rf /tmp/x"),
            steps=[StepConfig(step=1, engine="fake", operation="opt_sp")],
        )


def test_config_rejects_a_backslash_in_a_directory_path():
    """Inside double quotes a backslash escapes the very characters we screen for,
    which is enough to smuggle one past the check."""
    with pytest.raises(ValidationError):
        Config(
            scratch_dir=Path("scratch\\"),
            steps=[StepConfig(step=1, engine="fake", operation="opt_sp")],
        )


# --- config: legacy rewrites announce themselves -----------------------------


def test_every_legacy_rewrite_reports_where_it_happened():
    """Each rewrite reaches the sink anchored at the key that caused it.

    The point of the sink: before it, a rewrite spoke only to a logger, so
    ``validate_config`` handed the agent, the GUI and every MCP client an empty
    ``warnings`` list for a config written in a vocabulary due for removal in 3.0.
    """
    from chemrefine.config_legacy import normalize as _normalize_legacy

    seen, sink = _collect()
    _normalize_legacy(
        {
            "orca_executable": "/opt/orca/orca",
            "initial_xyz": "./seed.xyz",
            "steps": [
                {
                    "step": 1,
                    "mlff": {"model_name": "small"},
                    "normal_mode_sampling": True,
                    "sample_type": {"method": "integer", "parameters": {"num_structures": 5}},
                }
            ],
        },
        report=sink,
    )
    by_loc = dict(seen)
    assert by_loc[("orca_executable",)].startswith("`orca_executable` is deprecated")
    assert by_loc[("initial_xyz",)].startswith("`initial_xyz` is deprecated")
    assert by_loc[("steps", 0, "mlff")].startswith("step-level `mlff:` block is deprecated")
    assert by_loc[("steps", 0, "normal_mode_sampling")].startswith("`normal_mode_sampling*`")
    assert by_loc[("steps", 0, "sample_type")].startswith("`sample_type` is deprecated")
    assert "`min`" in by_loc[("steps", 0, "sample", "method")]
    assert "`count`" in by_loc[("steps", 0, "sample", "num_structures")]


def test_the_step_index_anchors_the_finding_to_its_own_step():
    """Two legacy steps produce two findings, each pointing at the right index."""
    from chemrefine.config_legacy import normalize as _normalize_legacy

    seen, sink = _collect()
    _normalize_legacy({"steps": [{"step": 1}, {"step": 2, "pyscf": {"xc": "pbe"}}]}, report=sink)
    assert [loc for loc, _ in seen] == [("steps", 1, "pyscf")]


def test_a_current_config_announces_nothing():
    """Idempotence, from the sink's side: nothing to rewrite, nothing to report."""
    from chemrefine.config_legacy import normalize as _normalize_legacy

    seen, sink = _collect()
    _normalize_legacy(
        {
            "executables": {"orca": "/opt/orca/orca"},
            "input": "./seed.xyz",
            "steps": [{"step": 1, "engine": "orca", "sample": {"method": "min", "count": 5}}],
        },
        report=sink,
    )
    assert seen == []


def test_reporting_replaces_the_log_rather_than_doubling_it(caplog):
    """``report=`` redirects; it does not add a second voice.

    The validator normalizes twice — once to collect, once inside
    ``Config.model_validate`` — so a sink that also logged would say everything twice on a
    path that used to say it once.
    """
    import logging

    from chemrefine.config_legacy import normalize as _normalize_legacy

    seen, sink = _collect()
    with caplog.at_level(logging.WARNING, logger="chemrefine.config_legacy"):
        _normalize_legacy({"orca_executable": "/opt/orca/orca"}, report=sink)
    assert len(seen) == 1
    assert caplog.records == []

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="chemrefine.config_legacy"):
        _normalize_legacy({"orca_executable": "/opt/orca/orca"})
    assert len(caplog.records) == 1


# --- config: legacy sample normalizer edges ---------------------------------


def test_normalize_sample_helpers_pass_through_non_dict():
    from chemrefine.config_legacy import _flatten_sample_type, _normalize_sample_block

    assert _flatten_sample_type("nope") == "nope"
    assert _normalize_sample_block("nope", (), _silent) == "nope"


def test_flatten_then_normalize_carries_through_extra_top_level_keys():
    from chemrefine.config_legacy import _flatten_sample_type, _normalize_sample_block

    flat = _flatten_sample_type(
        {"method": "boltzmann", "parameters": {"weight": 95}, "by_parent": True}
    )
    out = _normalize_sample_block(flat, (), _silent)
    assert out == {"method": "boltzmann", "percent_cumulative": 95, "by_parent": True}


def test_normalize_sample_block_without_method_passes_keys_through():
    """A block with no ``method`` key leaves it out (validation rejects it later)."""
    from chemrefine.config_legacy import _normalize_sample_block

    assert _normalize_sample_block({"count": 5}, (), _silent) == {"count": 5}
