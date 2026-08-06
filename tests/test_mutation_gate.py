"""The mutation gate's anchors must still match the source they name.

``scripts/mutation_gate.py`` finds each mutation by a literal string from the file it
breaks, which couples it to source text that refactors move. When that happened the gate
did not merely miss one predicate: it stopped at the stale anchor, so every mutation after
it went unrun, and it stayed that way for as long as nobody read past the exit code.

The check belongs here rather than behind a flag on the script because ``pytest`` is what
the developer who moves the line actually runs — pre-commit runs only ruff and interrogate,
and the gate itself is a separate three-minute CI job. This fires in the edit loop, at the
commit that moves the code.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parent.parent
_GATE = REPO / "scripts" / "mutation_gate.py"

# Skipped inside the gate's own scratch copy, and that is load-bearing rather than
# incidental. `_INPUTS` copies `src`, `tests`, `examples` and `pyproject.toml` — not
# `scripts` — so the file is simply absent there. It must stay absent: the gate mutates one
# source file per run, which by construction breaks that mutation's own anchor, so an
# anchor check running inside the copy would fail for *every* mutation and report each as
# caught by this test rather than by the one that was supposed to catch it. A gate that is
# green for the wrong reason is worse than no gate, so the condition is "am I looking at a
# real checkout", and the answer is whether the script is here at all.
pytestmark = pytest.mark.skipif(
    not _GATE.is_file(), reason="no scripts/ — this is the mutation gate's own scratch copy"
)


def _load_gate() -> ModuleType:
    """Import ``scripts/mutation_gate.py`` by path.

    ``scripts/`` is not a package and has no reason to become one. The module is registered
    in ``sys.modules`` *before* it executes because ``@dataclass`` reads
    ``sys.modules[cls.__module__].__dict__`` while the class body runs, and a module absent
    from that table fails there rather than at the import.
    """
    spec = importlib.util.spec_from_file_location("chemrefine_mutation_gate", _GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_mutation_anchor_matches_the_source_exactly_once():
    """A moved line must fail here, not three minutes into a CI job that stops early."""
    gate = _load_gate()
    assert gate.stale_anchors(REPO, gate.MUTATIONS) == []


def test_a_moved_anchor_is_reported_by_id():
    """Every stale anchor is reported, and each names the mutation to update.

    Reporting them together is the point: the old behaviour raised on the first, which said
    nothing about whether the rest of the list still matched.
    """
    gate = _load_gate()
    moved = gate.Mutation(
        id="gone",
        path="src/chemrefine/throttle.py",
        old="a line that is not in throttle.py",
        new="irrelevant",
        breaks="nothing — this mutation exists only to be missing",
    )
    report = gate.stale_anchors(REPO, [moved, *gate.MUTATIONS])
    assert len(report) == 1
    assert "[gone]" in report[0]
    assert "found 0" in report[0]
