"""Static end-to-end validation of every shipped example.

Each test sweeps ``examples/**/input.yaml`` so a broken example — missing
seed, renamed template dir, stale scan indices, a template without the
frequencies its NMS step needs — fails CI instead of a user's first run.
The knob-matrix test at the bottom pins the deliberate split between the
options the examples demonstrate and the options covered by tests only.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from chemrefine.config import (
    BoltzmannSample,
    Config,
    MaxSample,
    MinSample,
    StepConfig,
    load_config,
)
from chemrefine.engines._job import gpus_from_options
from chemrefine.engines._options import EngineOptions
from chemrefine.engines.api import ENGINES, OptionsDeclaring, get_engine
from chemrefine.engines.orca.engine import OrcaEngine
from chemrefine.engines.orca.inspect import inspect_template
from chemrefine.io import read_xyz_frames
from chemrefine.nms import NmsOptions

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = sorted(REPO.glob("examples/**/input.yaml"))
IDS = [str(p.parent.relative_to(REPO / "examples")) or "canonical" for p in EXAMPLES]

# Derived, not spelled: every engine driven by the ORCA parser is in the family, so a
# fourth member joins these gates by existing. The authoritative behavior pin lives in
# test_engines_invariants.test_every_orca_family_engine_refuses_an_unknown_operation_up_front
# — that file pins the set, this one derives it.
_ORCA_FAMILY = frozenset(n for n in ENGINES if isinstance(get_engine(n), OrcaEngine))
assert _ORCA_FAMILY, "the ORCA family derives empty — the example gates would all skip"


def _seed_path(cfg: Config) -> Path:
    """The pipeline's seed input: explicit ``input:`` or the step-1 default."""
    return Path(cfg.input) if cfg.input is not None else cfg.template_dir / "step1.xyz"


def _requests_gpu(step: StepConfig) -> bool:
    """Whether a step asks for a GPU, read through the engine's own options model.

    Goes through the same single reader the scheduler uses (`gpus_from_options` +
    the engine's `options_cls`) rather than re-deriving it from the raw dict here —
    re-deriving is what let this check drift from `_execution._header_name`.
    """
    options_cls = getattr(get_engine(step.engine), "options_cls", EngineOptions)
    return bool(gpus_from_options(step.options, options_cls))


def _resolved_template(cfg: Config, step: StepConfig) -> Path | None:
    """The template file a step would run with, or ``None`` for template-less engines.

    The same resolution `build_context` performs, so a shipped example is checked against
    the file the run would actually read.
    """
    from chemrefine.engines.api import TemplateDriven
    from chemrefine.ids import step_template_path

    engine = get_engine(step.engine)
    if not isinstance(engine, TemplateDriven):
        return None
    return step_template_path(
        cfg.template_dir, step.step, template=step.template, suffix=engine.template_suffix
    )


def test_examples_are_discovered() -> None:
    """The sweep must actually sweep — an empty glob would green-light anything."""
    assert len(EXAMPLES) >= 10


@pytest.mark.parametrize("yml", EXAMPLES, ids=IDS)
def test_example_loads(yml: Path) -> None:
    """Every shipped example must load through the v2 schema (and normalizer)."""
    load_config(yml)  # raises ConfigError on any failure


@pytest.mark.parametrize("yml", EXAMPLES, ids=IDS)
def test_example_seed_exists_and_parses(yml: Path) -> None:
    """The seed input must exist and yield at least one structure."""
    cfg = load_config(yml)
    seed = _seed_path(cfg)
    assert seed.exists(), f"seed input missing: {seed}"
    if seed.is_dir():
        assert list(seed.glob("*.xyz")), f"seed directory holds no .xyz files: {seed}"
    elif seed.suffix == ".csv":
        # utf-8-sig: pandas (the runtime reader) strips a UTF-8 BOM, so we must too.
        with seed.open(encoding="utf-8-sig") as fh:
            fieldnames = csv.DictReader(fh).fieldnames or []
        assert "smiles" in fieldnames, f"SMILES seed {seed} has no `smiles` column"
    else:
        assert read_xyz_frames(seed), f"seed {seed} parses to no frames"


@pytest.mark.parametrize("yml", EXAMPLES, ids=IDS)
def test_example_step_templates_resolve(yml: Path) -> None:
    """Every step's template (explicit or default stepN.*) must exist."""
    cfg = load_config(yml)
    for step in cfg.steps:
        _resolved_template(cfg, step)  # raises FileNotFoundError when missing


@pytest.mark.parametrize("yml", EXAMPLES, ids=IDS)
def test_example_slurm_headers_exist(yml: Path) -> None:
    """Each step's SLURM header (explicit, cuda-for-GPU, or global) must ship."""
    cfg = load_config(yml)
    for step in cfg.steps:
        if step.slurm_template:
            header = step.slurm_template
        elif _requests_gpu(step):
            header = "cuda.slurm.header"
        else:
            header = cfg.slurm_template
        path = cfg.template_dir / header
        assert path.is_file(), f"step {step.step}: SLURM header missing: {path}"


@pytest.mark.parametrize("yml", EXAMPLES, ids=IDS)
def test_example_nms_steps_compute_frequencies(yml: Path) -> None:
    """`nms: true` steps must run a frequency calculation (the B9 gate)."""
    cfg = load_config(yml)
    for step in cfg.steps:
        if not step.nms or step.engine not in _ORCA_FAMILY:
            continue
        template = _resolved_template(cfg, step)
        assert template is not None
        info = inspect_template(template)
        assert info.has_freq, (
            f"step {step.step}: nms: true but template {template.name} computes no frequencies"
        )


@pytest.mark.parametrize("yml", EXAMPLES, ids=IDS)
def test_example_scan_indices_fit_seed(yml: Path) -> None:
    """%geom Scan atom indices in a step-1 template must exist in the seed molecule."""
    cfg = load_config(yml)
    seed = _seed_path(cfg)
    if seed.suffix != ".xyz" or not seed.is_file():
        return
    n_atoms = len(read_xyz_frames(seed)[0])
    for step in cfg.steps:
        if step.step != 1 or step.engine not in _ORCA_FAMILY:
            continue
        template = _resolved_template(cfg, step)
        assert template is not None
        text = template.read_text()
        for match in re.finditer(r"^\s*Scan\s+[BAD]\s+([\d\s,]+?)\s*=", text, re.MULTILINE):
            for index in re.findall(r"\d+", match.group(1)):
                assert int(index) < n_atoms, (
                    f"step {step.step}: scan index {index} out of range for "
                    f"{n_atoms}-atom seed {seed.name}"
                )


@pytest.mark.parametrize("yml", EXAMPLES, ids=IDS)
def test_example_docker_guest_ships(yml: Path) -> None:
    """A %DOCKER template's GUEST file must ship with the example's templates."""
    cfg = load_config(yml)
    for step in cfg.steps:
        if step.engine not in _ORCA_FAMILY:
            continue
        template = _resolved_template(cfg, step)
        assert template is not None
        for match in re.finditer(r"GUEST\s+\"([^\"]+)\"", template.read_text(), re.IGNORECASE):
            guest = Path(match.group(1))
            # The runtime's own predicate, exactly: `_absolutize_template_paths` resolves
            # the full quoted relative path against the template's directory and, when
            # that names no file, silently leaves the relative path in place for ORCA to
            # fail on from its scratch dir. A basename check under template_dir passed a
            # broken `guests/cl.xyz` (basename shipped, path unresolvable) and failed a
            # working `sub/g.xyz` — the gate must judge what the run judges.
            resolved = (template.parent / guest).resolve()
            assert resolved.is_file(), (
                f"step {step.step}: %DOCKER guest {guest} does not resolve against "
                f"{template.parent} — the rendered input would keep the unresolvable "
                f"relative path and ORCA would fail to open it from the scratch dir"
            )


# ---------------------------------------------------------------------------
# Knob coverage matrix
# ---------------------------------------------------------------------------
#
# Every user-facing config knob belongs to exactly one of two sets: shown by at
# least one shipped example (REQUIRED — the examples are living documentation),
# or exercised by the test suite only (TESTS_ONLY — behavioural knobs that would
# distort the paper tutorials). A new schema field fails the universe test until
# it is filed into one of the sets, so no knob can silently join neither.

REQUIRED = {
    "config": {
        "template_dir",
        "scratch_dir",
        "output_dir",
        "input",
        "charge",
        "multiplicity",
        "max_cores",
        "slurm_template",
        "executables",
        "steps",
    },
    "step": {
        "step",
        "name",
        "engine",
        "operation",
        "template",
        "charge",
        "multiplicity",
        "options",
        "sample",
        "nms",
    },
    "sample": {"method", "percent_cumulative", "count", "window_kcalmol"},
    "nms": {"target", "displacement_value", "num_random_displacements"},
    "MlipOptions": {"model_name", "task_name", "device", "cores"},
    "PyscfOptions": {"method", "xc", "basis", "device", "cores"},
    # A training step must name all three: which library trains, what it starts from, and
    # where it runs. None of them has a default, so an example that omitted one would not run.
    "MlipTrainOptions": {"task_name", "model_name", "device"},
}

TESTS_ONLY = {
    "config": {"max_gpus", "slurm_array", "dispatch", "job_timeout_seconds"},
    "step": {"slurm_template", "on_failure"},
    "sample": {"by_parent", "temperature_k", "energy_type"},
    "nms": {"ts_mode_index", "seed"},
    "MlipOptions": {"model_path", "backend_python"},
    "PyscfOptions": {"df", "gpu", "backend_python"},
    # Whole models filed here: no shipped example runs `pyscf-extopt` or `qchem` — the
    # examples are frozen paper artifacts — so every knob is tests-only until a tutorial
    # demonstrating the engine ships. `strict_scf` in particular is an opt-*out*: it
    # defaults on, and a tutorial would have no reason to turn a correctness guard off.
    "PyscfExtOptOptions": {
        "method",
        "xc",
        "basis",
        "df",
        "gpu",
        "strict_scf",
        "save_tensors",
        "localized",
        "tensor_folder",
        "device",
        "cores",
        "backend_python",
    },
    "QchemOptions": {"nprocs", "save", "device", "cores", "backend_python"},
    "MlipTrainOptions": {
        "valid_fraction",
        "test_fraction",
        "seed",
        "gpus",
        "cores",
        "model_path",
        "backend_python",
    },
}

_SAMPLE_FIELDS = (
    set(BoltzmannSample.model_fields) | set(MinSample.model_fields) | set(MaxSample.model_fields)
)

# The engine half of the universe is derived: one section per OptionsDeclaring engine's
# model, keyed by the model's name, with the full field set (inherited included — the
# shared base knobs get a deliberate verdict per model). A new engine's model joins the
# universe by registering, and fails test_knob_universe_is_fully_filed until its knobs
# are filed above.
_ENGINE_MODELS = {
    engine.options_cls
    for engine in (get_engine(name) for name in ENGINES)
    if isinstance(engine, OptionsDeclaring)
}

_UNIVERSE = {
    "config": set(Config.model_fields),
    "step": set(StepConfig.model_fields),
    "sample": _SAMPLE_FIELDS,
    "nms": set(NmsOptions.model_fields),
    **{model.__name__: set(model.model_fields) for model in _ENGINE_MODELS},
}


def _raw_examples() -> list[dict[str, Any]]:
    return [yaml.safe_load(p.read_text()) for p in EXAMPLES]


def test_knob_universe_is_fully_filed() -> None:
    """Every schema field is deliberately REQUIRED or TESTS_ONLY — never neither."""
    assert set(REQUIRED) | set(TESTS_ONLY) <= set(_UNIVERSE), (
        "a filed section matches no universe section — its verdicts would go unchecked: "
        f"{sorted((set(REQUIRED) | set(TESTS_ONLY)) - set(_UNIVERSE))}"
    )
    for section, universe in _UNIVERSE.items():
        required = REQUIRED.get(section, set())
        tests_only = TESTS_ONLY.get(section, set())
        assert required.isdisjoint(tests_only), f"{section}: knob filed in both sets"
        assert required | tests_only == universe, (
            f"{section}: unfiled or stale knobs: {sorted(universe ^ (required | tests_only))}"
        )


def test_examples_cover_required_knobs() -> None:
    """Each REQUIRED knob appears, at its nesting level, in at least one example."""
    used: dict[str, set[str]] = {key: set() for key in _UNIVERSE}
    for doc in _raw_examples():
        used["config"] |= set(doc)
        for step in doc.get("steps", []):
            used["step"] |= set(step)
            used["sample"] |= set(step.get("sample") or {})
            options = set(step.get("options") or {})
            engine = step.get("engine")
            if step.get("nms"):
                used["nms"] |= options
            # Filed under the model the engine itself declares, so a knob counts as
            # demonstrated only for the model that can take it: mlip-extopt shares
            # MlipOptions with mlip, mlip-train's different model files apart with no
            # special case, and a future OptionsDeclaring engine joins with no edit here.
            engine_obj = get_engine(engine) if engine else None
            if isinstance(engine_obj, OptionsDeclaring):
                used[engine_obj.options_cls.__name__] |= options
    for section, required in REQUIRED.items():
        missing = required - used[section]
        assert not missing, f"{section}: no example uses {sorted(missing)}"


def test_examples_cover_required_variants() -> None:
    """The behavioural variants the examples must keep demonstrating."""
    engines: set[str] = set()
    operations: set[str] = set()
    sample_variants: set[tuple[str, ...]] = set()
    nms_targets: set[str] = set()
    seed_suffixes: set[str] = set()
    devices: set[str] = set()
    step_overrides: set[str] = set()

    for doc in _raw_examples():
        seed_suffixes.add(Path(doc.get("input", "step1.xyz")).suffix)
        for step in doc.get("steps", []):
            engines.add(step.get("engine", ""))
            operations.add(step.get("operation", ""))
            sample = step.get("sample") or {}
            if sample:
                selectors = tuple(sorted(set(sample) & {"count", "window_kcalmol"}))
                sample_variants.add((sample.get("method", ""), *selectors))
            options = step.get("options") or {}
            if step.get("nms"):
                nms_targets.add(options.get("target", "<inferred>"))
            if "device" in options:
                devices.add(options["device"])
            step_overrides |= set(step) & {"charge", "multiplicity"}

    assert engines >= {"orca", "mlip", "mlip-extopt", "mlip-train", "pyscf"}
    # `mlip_train` is not here: it was never an operation the way the others are — it named
    # a *step kind*, which `engine: mlip-train` already says. The config normalizer still
    # rewrites the old spelling, and `test_config.py` covers that; an example carrying it
    # would only be demonstrating a legacy form to new readers.
    assert operations >= {"goat", "docker", "solvator", "pes", "opt_sp", "sp"}
    assert sample_variants >= {
        ("boltzmann",),
        ("min", "count"),
        ("min", "window_kcalmol"),
        ("max", "count"),
    }
    assert nms_targets >= {"ts", "random", "<inferred>"}
    assert seed_suffixes >= {".xyz", ".csv"}
    assert devices >= {"cuda", "cpu"}
    assert step_overrides == {"charge", "multiplicity"}
