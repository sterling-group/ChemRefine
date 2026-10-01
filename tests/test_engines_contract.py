"""Per-engine parse contracts: trimmed real output → canonical result records.

Every registered engine must ship at least one case under
``tests/data/engines/<engine>/<case>/`` — a shortened **real** native output
(the trim rule: keep only the blocks the parsers read), the seed geometry,
a ``case.json`` meta, and the golden ``expected.json`` holding the canonical
records (:func:`chemrefine.cache.structure_record`). The contract test runs
the engine's public ``parse`` on the native fixture and compares **exactly**
(text→float parsing is IEEE-deterministic; a drifting golden means the
parser or the schema changed and must be reviewed). Regenerate goldens with
``pytest tests/test_engines_contract.py --update-goldens`` and review the
git diff — the same harness writes and checks, so they cannot drift apart.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chemrefine import cache
from chemrefine.config import StepConfig
from chemrefine.engines.api import (
    ENGINES,
    CalculationEngine,
    OperationInferring,
    OperationsDeclaring,
    get_engine,
)
from chemrefine.errors import ChemRefineError
from chemrefine.io import read_xyz_frames
from chemrefine.state import PipelineState, StepContext, StepInputs, Structure

DATA = Path(__file__).resolve().parent / "data" / "engines"
CASES = sorted(path for path in DATA.glob("*/*") if path.is_dir())
CASE_IDS = [f"{path.parent.name}-{path.name}" for path in CASES]

_CONTRACT_EXEMPT = {
    "fake",  # test scaffolding registered by conftest, not a shipped plugin
    "mlip-train",  # pass-through trainer: parses no calculation output
}


def _native_output(case_dir: Path) -> Path:
    """The case's native output file (the pipeline-named .out or .json)."""
    for candidate in (case_dir / "step1_0.out", case_dir / "step1_0.json"):
        if candidate.is_file():
            return candidate
    raise AssertionError(f"{case_dir}: no step1_0.out or step1_0.json native output")


def _context(case_dir: Path, meta: dict[str, Any]) -> StepContext:
    """A minimal read-only StepContext for parsing one archived output."""
    seed = read_xyz_frames(case_dir / "step1_0_inp.xyz")[0]
    return StepContext(
        step_cfg=StepConfig(step=1, engine=case_dir.parent.name, operation=meta.get("operation")),
        step_dir=case_dir,
        template_dir=case_dir,
        template=case_dir / f"step1.{get_engine(case_dir.parent.name).template_suffix}",
        operation=meta.get("operation"),
        scratch_dir=None,
        prev_state=PipelineState(structures=(Structure(id="0", atoms=seed),)),
        charge=int(meta.get("charge", 0)),
        multiplicity=int(meta.get("multiplicity", 1)),
        max_cores=1,
        slurm_template="cpu.slurm.header",
    )


@pytest.mark.parametrize("case_dir", CASES, ids=CASE_IDS)
def test_contract_case(case_dir: Path, request: pytest.FixtureRequest) -> None:
    """The engine's parse of the native fixture must match the golden records."""
    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    engine = get_engine(case_dir.parent.name)
    out = _native_output(case_dir)
    ctx = _context(case_dir, meta)

    results = engine.parse(StepInputs(files=((out, out, "0"),)), ctx)
    records = [cache.structure_record(s) for s in results.structures]
    document = {
        "result_format": cache.RESULT_FORMAT_VERSION,
        "structures": json.loads(json.dumps(records)),
    }

    golden_path = case_dir / "expected.json"
    if request.config.getoption("--update-goldens"):
        golden_path.write_text(json.dumps(document, indent=2) + "\n")
        return
    assert golden_path.is_file(), f"{case_dir}: missing golden — run with --update-goldens"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert document["result_format"] == golden["result_format"]
    assert document["structures"] == golden["structures"]


INFERRING_CASES = [
    path for path in CASES if isinstance(get_engine(path.parent.name), OperationInferring)
]


def _read_as(engine: CalculationEngine, out: Path, ctx: StepContext) -> object:
    """What one read of ``out`` on ``ctx`` yields: its records, or the refusal it raises."""
    try:
        results = engine.parse(StepInputs(files=((out, out, "0"),)), ctx)
    except ChemRefineError as e:
        return (type(e).__name__, str(e))
    return [cache.structure_record(s) for s in results.structures]


@pytest.mark.parametrize(
    "case_dir", INFERRING_CASES, ids=[f"{p.parent.name}-{p.name}" for p in INFERRING_CASES]
)
def test_operations_said_to_read_alike_read_a_real_output_alike(case_dir: Path) -> None:
    """``reads_alike`` decides whether a declared operation draws a warning, so it is held to
    the dispatch: two operations it calls alike read the same real output into the same
    records, or into the same refusal.

    Each operation is read as an inferred one, which is the read a template's run gets — a
    run kind with no reader of its own (``irc``) included.
    """
    engine = get_engine(case_dir.parent.name)
    assert isinstance(engine, OperationInferring) and isinstance(engine, OperationsDeclaring)
    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    out = _native_output(case_dir)
    base = _context(case_dir, meta)
    inferred = replace(base, step_cfg=base.step_cfg.model_copy(update={"operation": None}))
    words = sorted({*engine.operations, "irc"})
    read = {word: _read_as(engine, out, replace(inferred, operation=word)) for word in words}
    alike = [(a, b) for a, b in itertools.combinations(words, 2) if engine.reads_alike(a, b)]
    assert alike, "no two operations read alike — has the dispatch lost its shared reader?"
    for first, second in alike:
        assert read[first] == read[second], f"{first!r} and {second!r} read {out} differently"


def test_every_registered_engine_ships_a_contract_case() -> None:
    """Registering an engine without its shortened real output fails here."""
    for name in sorted(set(ENGINES) - _CONTRACT_EXEMPT):
        cases = list((DATA / name).glob("*/expected.json"))
        assert cases, (
            f"engine {name!r} ships no contract fixture under tests/data/engines/{name}/ "
            "— add a trimmed real output + golden (see this module's docstring)"
        )
