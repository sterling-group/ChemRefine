"""The mutation gate's anchors must still match the source they name.

``scripts/mutation_gate.py`` finds each mutation by a literal string from the file it
breaks, which couples it to source text that refactors move. When that happened the gate
did not merely miss one predicate: it stopped at the stale anchor, so every mutation after
it went unrun, and it stayed that way for as long as nobody read past the exit code.

The check belongs here rather than behind a flag on the script because ``pytest`` is what
the developer who moves the line actually runs — pre-commit runs only ruff and interrogate,
and the gate itself is a separate CI job of its own. This fires in the edit loop, at the
commit that moves the code.
"""

from __future__ import annotations

import importlib.util
import json
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
    not _GATE.is_file(),
    reason="no scripts/ — neither the sdist nor the gate's own scratch copy carries it",
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
    """A moved line must fail here, not partway into a CI job that stops early."""
    gate = _load_gate()
    assert gate.stale_anchors(REPO, gate.MUTATIONS) == []


def test_an_engine_files_its_own_mutations_beside_its_fixture(tmp_path: Path):
    """``tests/data/engines/<name>/mutations.json`` joins the gate with no edit to the script.

    The bundled list is one more central roster an engine would otherwise have to be added
    to; read from the fixture folder, an engine's guards travel with the engine, and the
    anchor check above covers them like any other entry.
    """
    gate = _load_gate()
    folder = tmp_path / "tests" / "data" / "engines" / "probe"
    folder.mkdir(parents=True)
    entry = {
        "id": "probe-guard",
        "path": "src/chemrefine/probe.py",
        "old": "if converged:",
        "new": "if True:",
        "tests": "tests/test_engines_probe.py",
        "breaks": "an unconverged probe run ranks as a result",
    }
    (folder / gate.ENGINE_MUTATIONS_NAME).write_text(json.dumps([entry]), encoding="utf-8")

    assert gate.engine_mutations(tmp_path) == (gate.Mutation(**entry),)
    assert gate.engine_mutations(tmp_path / "nowhere") == ()


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
        tests="tests/test_throttle.py",
        breaks="nothing — this mutation exists only to be missing",
    )
    report = gate.stale_anchors(REPO, [moved, *gate.MUTATIONS])
    assert len(report) == 1
    assert "[gone]" in report[0]
    assert "found 0" in report[0]


def test_a_renamed_test_file_is_reported_by_id():
    """The named test file is checked here too, for the reason the anchor is.

    A ``tests`` entry that no longer names a file costs only time — the run falls back to
    the whole suite and the verdict is unchanged — but silently paying 47s instead of 2s
    per mutation is how the gate drifts back to what it was. This is where a rename is
    cheap to notice.
    """
    gate = _load_gate()
    renamed = gate.Mutation(
        id="orphan",
        path="src/chemrefine/throttle.py",
        old="def has_room",
        new="irrelevant",
        tests="tests/test_a_file_that_was_renamed.py",
        breaks="nothing — this mutation exists only to name a missing test",
    )
    report = gate.stale_anchors(REPO, [renamed, *gate.MUTATIONS])
    assert len(report) == 1
    assert "[orphan]" in report[0]
    assert "tests/test_a_file_that_was_renamed.py" in report[0]


def test_a_renamed_source_file_is_reported_by_id():
    """A missing ``path`` is a report line like the other two, never a traceback.

    A ``git mv`` of a mutated source file made ``stale_anchors`` raise
    ``FileNotFoundError`` out of ``main()`` — one moved file hiding the state of the whole
    gate, the exact failure the aggregate report was built to prevent, arriving by the one
    read the function never guarded.
    """
    gate = _load_gate()
    moved = gate.Mutation(
        id="vanished",
        path="src/chemrefine/a_file_that_was_renamed.py",
        old="def has_room",
        new="irrelevant",
        tests="tests/test_throttle.py",
        breaks="nothing — this mutation exists only to name a missing source file",
    )
    report = gate.stale_anchors(REPO, [moved, *gate.MUTATIONS])
    assert len(report) == 1
    assert "[vanished]" in report[0]
    assert "src/chemrefine/a_file_that_was_renamed.py" in report[0]
    assert "the code moved" in report[0]


def test_a_timeout_is_a_catch_for_the_named_file_and_inconclusive_for_the_whole_suite(
    monkeypatch: pytest.MonkeyPatch,
):
    """The two timeouts mean different things, and only one of them is a red build.

    A mutation that makes a wait loop spin forever hangs the file that drives it — caught.
    The whole-suite fallback runs only after that file stayed green, so its timeout is as
    likely the suite outgrowing the budget on a slow runner as a hang; reported as caught,
    that was a false green from the one gate whose job is to refuse them.
    """
    gate = _load_gate()

    def hang(argv: list[str], **_: object) -> None:
        raise gate.subprocess.TimeoutExpired(argv, gate.TIMEOUT_SECONDS)

    monkeypatch.setattr(gate.subprocess, "run", hang)
    named = gate._run_suite(REPO, {}, "tests/test_throttle.py")
    assert named.caught and not named.inconclusive
    whole = gate._run_suite(REPO, {})
    assert whole.inconclusive and not whole.caught
    assert str(gate.TIMEOUT_SECONDS) in whole.why


def test_a_run_that_collects_no_tests_is_never_a_catch(monkeypatch: pytest.MonkeyPatch):
    """pytest exits 5 when nothing was collected, and "non-zero means red" read that as caught.

    An emptied test file, or one whose tests are all deselected by the default addopts,
    made every mutation naming it report `caught` while nothing ran — and `stale_anchors`
    cannot see it, because `is_file()` is true of a 0-byte file. Split the way the timeout
    is: a named file that ran nothing is not a catch (the whole-suite fallback then decides),
    and the whole suite collecting nothing proves nothing either way.
    """
    gate = _load_gate()

    def nothing_collected(argv: list[str], **_: object) -> object:
        return gate.subprocess.CompletedProcess(argv, 5, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", nothing_collected)
    named = gate._run_suite(REPO, {}, "tests/test_throttle.py")
    assert not named.caught and not named.inconclusive
    assert "collected no tests" in named.why
    whole = gate._run_suite(REPO, {})
    assert whole.inconclusive and not whole.caught
    assert "collected no tests" in whole.why


def test_an_inconclusive_mutation_fails_the_gate_without_being_called_a_survivor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """``main`` exits 2 for an inconclusive run and says so, apart from a survivor's 1.

    The tree copy, the isolation and baseline checks and the suite itself are stood in
    for: this is about the verdict's bookkeeping, not about running pytest twice more.
    """
    gate = _load_gate()
    # The longest id cannot be a substring of another, so `-k` selects exactly it.
    mutation = max(gate.MUTATIONS, key=lambda m: len(m.id))

    def copy_one(dest: Path) -> None:
        target = dest / mutation.path
        target.parent.mkdir(parents=True)
        target.write_text((REPO / mutation.path).read_text(encoding="utf-8"), encoding="utf-8")

    def verdicts(work: Path, env: dict[str, str], target: str | None = None) -> object:
        if target is not None:
            return gate.Verdict(False, "suite passed unchanged")
        return gate.Verdict(False, "did not finish", inconclusive=True)

    monkeypatch.setattr(gate, "_copy_tree", copy_one)
    monkeypatch.setattr(gate, "_assert_isolated", lambda work, env: None)
    monkeypatch.setattr(gate, "_assert_baseline_is_green", lambda work, env: None)
    monkeypatch.setattr(gate, "_run_suite", verdicts)

    assert gate.main(["-k", mutation.id]) == 2
    out = capsys.readouterr().out
    assert "INCONCLUSIVE" in out
    assert "did not finish" in out
    assert "SURVIVED" not in out and "survived" not in out
