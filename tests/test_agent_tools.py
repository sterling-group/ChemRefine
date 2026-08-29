"""The agent tool layer: JSON in, JSON out, filesystem truth, and a submit that never blocks.

These tests drive :mod:`chemrefine.agent_tools` the way an MCP client or the embedded
agent will — through the public functions only — against real config trees under
``tmp_path``. The two properties everything else leans on: :func:`start_run` launches a
*detached* child (pinned by inspecting the recorded ``Popen`` call, never by running
one) and refuses a held lock; the status/results/failures readers answer purely from
what the pipeline persists, so they are exercised against files written by the
pipeline's own writers (``io.save_step_csv``, ``cache.save_failure_records``), not
hand-rolled lookalikes.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from chemrefine import agent_tools, io, pipeline
from chemrefine.cache import save_failure_records
from chemrefine.errors import ConfigError, RunLockError
from chemrefine.state import FailureKind, FailureRecord


def _write_config(tmp_path: Path, *steps: dict[str, object]) -> Path:
    listed = list(steps) or [{"step": 1, "engine": "fake"}]
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump({"steps": listed}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The shared surface
# ---------------------------------------------------------------------------


def test_every_mutating_tool_name_names_a_tool():
    """``MUTATING_TOOLS`` must be a subset of the registered tool names.

    The confirmation gate matches by name (`agent.harness.build_agent`), so an entry
    naming nothing gates nothing — and, the direction that matters, a mutating tool
    renamed without this set would silently go ungated. The set once carried a ``"save"``
    that matched no tool; this is what makes that impossible to reintroduce.
    """
    phantom = agent_tools.MUTATING_TOOLS - {tool.__name__ for tool in agent_tools.TOOLS}
    assert not phantom, f"MUTATING_TOOLS entries naming no tool: {sorted(phantom)}"


# ---------------------------------------------------------------------------
# Introspection + validation re-exposures
# ---------------------------------------------------------------------------


def test_schema_and_engines_are_json_shaped():
    document = json.loads(json.dumps(agent_tools.get_schema()))
    assert "config" in document
    names = [d["name"] for d in json.loads(json.dumps(agent_tools.list_engines()))]
    assert "orca" in names


def test_validate_config_text_and_path_agree(tmp_path: Path):
    path = _write_config(tmp_path)
    from_text = agent_tools.validate_config(path.read_text(encoding="utf-8"), str(tmp_path))
    from_path = agent_tools.validate_config_path(str(path))
    assert from_text == from_path
    assert from_text["ok"] is True


def test_validate_config_without_base_dir_still_reports(tmp_path: Path):
    report = agent_tools.validate_config("steps: [unclosed")
    assert report["ok"] is False
    assert report["issues"][0]["kind"] == "yaml"


def test_summarize_config_rows_mirror_the_steps(tmp_path: Path):
    path = _write_config(
        tmp_path,
        {"step": 1, "name": "screen", "engine": "fake", "sample": {"method": "min", "count": 3}},
        {"step": 2, "engine": "fake", "on_failure": "skip"},
    )
    summary = agent_tools.summarize_config(str(path))
    assert summary["max_cores"] == 4  # the schema default, surfaced not invented
    first, second = summary["steps"]
    assert first["name"] == "screen"
    assert first["sample"]["count"] == 3
    assert second["on_failure"] == "skip"


# ---------------------------------------------------------------------------
# save_config — validation gates the write
# ---------------------------------------------------------------------------


def test_save_config_writes_a_runnable_config(tmp_path: Path):
    destination = tmp_path / "proj" / "input.yaml"
    text = yaml.safe_dump({"steps": [{"step": 1, "engine": "fake"}]})
    result = agent_tools.save_config(str(destination), text)
    assert result["written"] is True
    assert result["ok"] is True
    assert destination.read_text(encoding="utf-8") == text


def test_save_config_refuses_an_unrunnable_config_and_writes_nothing(tmp_path: Path):
    destination = tmp_path / "input.yaml"
    result = agent_tools.save_config(
        str(destination), yaml.safe_dump({"steps": [{"step": 1, "engine": "no-such"}]})
    )
    assert result["written"] is False
    assert result["ok"] is False
    assert result["issues"][0]["kind"] == "engine"
    assert not destination.exists()


def test_save_config_passes_warnings_through_without_blocking(tmp_path: Path):
    """Missing templates warn — exactly as `chemrefine validate` treats them."""
    destination = tmp_path / "input.yaml"
    result = agent_tools.save_config(
        str(destination), yaml.safe_dump({"steps": [{"step": 1, "engine": "orca"}]})
    )
    assert result["written"] is True
    assert any(w["kind"] == "template" for w in result["warnings"])
    assert destination.exists()


def _fortress(tmp_path: Path) -> Path:
    """An unwritable directory, or a skip where the wall cannot be built (root)."""
    if os.geteuid() == 0:
        pytest.skip("root writes everywhere; the permission wall cannot be built")
    fortress = tmp_path / "fortress"
    fortress.mkdir()
    fortress.chmod(0o555)
    return fortress


def test_save_config_answers_an_unwritable_destination_with_the_contract(tmp_path: Path):
    """A disk refusal is the module's ConfigError, not a raw OSError.

    The module promises every failure carries the documented exit code; a raw ``OSError``
    reached the GUI's handler, which re-raises anything outside the taxonomy — a 500 with
    a logged traceback for what is an ordinary bad destination (a read-only tree, ENOSPC
    on scratch). ``/api/save`` already answers this with a 400 "like every other bad input
    here"; these are its siblings, held to the same rule.
    """
    fortress = _fortress(tmp_path)
    text = yaml.safe_dump({"steps": [{"step": 1, "engine": "fake"}]})
    try:
        with pytest.raises(ConfigError, match="cannot write"):
            agent_tools.save_config(str(fortress / "input.yaml"), text)
    finally:
        fortress.chmod(0o755)


def test_write_template_answers_an_unwritable_destination_with_the_contract(tmp_path: Path):
    """The same rule for the inline template editor's write."""
    fortress = _fortress(tmp_path)
    path = tmp_path / "input.yaml"
    path.write_text(
        yaml.safe_dump({"template_dir": str(fortress), "steps": [{"step": 1, "engine": "orca"}]}),
        encoding="utf-8",
    )
    try:
        with pytest.raises(ConfigError, match="cannot write template"):
            agent_tools.write_template(str(path), 1, "! MyKeywords\n")
    finally:
        fortress.chmod(0o755)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def test_template_roundtrip_by_number_and_name(tmp_path: Path):
    path = _write_config(
        tmp_path,
        {"step": 1, "engine": "orca"},
        {"step": 2, "name": "refine", "engine": "orca"},
    )
    agent_tools.scaffold_templates(str(path))
    by_number = agent_tools.read_template(str(path), 2)  # walks past step 1's plan row
    assert "%pal" in by_number["text"]

    agent_tools.write_template(str(path), "refine", "! MyKeywords\n")
    by_name = agent_tools.read_template(str(path), "refine")
    assert by_name["text"] == "! MyKeywords\n"
    assert by_name["path"] == by_number["path"]


def test_reading_a_missing_template_names_the_fix(tmp_path: Path):
    path = _write_config(tmp_path, {"step": 1, "engine": "orca"})
    with pytest.raises(ConfigError, match="scaffold_templates"):
        agent_tools.read_template(str(path), 1)


def test_template_tools_refuse_nonsense_targets(tmp_path: Path):
    path = _write_config(tmp_path)
    with pytest.raises(ConfigError, match="no step matches"):
        agent_tools.read_template(str(path), 7)
    with pytest.raises(ConfigError, match="does not read a template"):
        agent_tools.read_template(str(path), 1)  # the fake engine is template-free


def test_scaffold_templates_reports_written_then_kept(tmp_path: Path):
    path = _write_config(tmp_path, {"step": 1, "engine": "orca"})
    first = agent_tools.scaffold_templates(str(path))
    assert any(p.endswith("step1.inp") for p in first["written"])
    assert first["kept"] == []
    second = agent_tools.scaffold_templates(str(path))
    assert second["written"] == []
    assert any(p.endswith("step1.inp") for p in second["kept"])


# ---------------------------------------------------------------------------
# start_run — detached, validated, lock-aware
# ---------------------------------------------------------------------------


class _RecordedPopen:
    """Stands in for the detached child: records how it was launched, goes nowhere."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.pid = 4242
        type(self).calls.append({"argv": argv, **kwargs})


@pytest.fixture
def recorded_popen(monkeypatch: pytest.MonkeyPatch) -> type[_RecordedPopen]:
    _RecordedPopen.calls = []
    monkeypatch.setattr(subprocess, "Popen", _RecordedPopen)
    return _RecordedPopen


def test_start_run_launches_a_detached_chemrefine(tmp_path: Path, recorded_popen):
    path = _write_config(tmp_path)
    result = agent_tools.start_run(str(path), max_cores=2)
    [call] = recorded_popen.calls
    assert call["argv"][1:4] == ["-m", "chemrefine", "run"]
    assert call["argv"][-2:] == ["--maxcores", "2"]
    assert call["start_new_session"] is True  # survives the agent session ending
    assert result["pid"] == 4242
    assert Path(result["log"]).parent.name == "agent_runs"


def test_start_run_passes_the_target_through(tmp_path: Path, recorded_popen):
    path = _write_config(
        tmp_path, {"step": 1, "name": "screen", "engine": "fake"}, {"step": 2, "engine": "fake"}
    )
    agent_tools.start_run(str(path), action="rerun", target="screen", max_gpus=0)
    [call] = recorded_popen.calls
    assert call["argv"][3:6] == ["rerun", str(path.resolve()), "screen"]
    assert call["argv"][-2:] == ["--maxgpus", "0"]


def test_two_runs_started_together_get_their_own_logs(tmp_path: Path, recorded_popen):
    """Two launches in the same second must not share one log file.

    The lock check in `start_run` is check-then-act — the child claims the lock, so both
    calls can pass it — and at one-second resolution both resolved to the same path, where
    `open("wb")` truncated the first child's log while it was still writing to it. The
    second driver dies on the lock either way; what is lost is the *first* one's log, which
    is exactly what `run_status` serves back when someone asks what went wrong.
    """
    path = _write_config(tmp_path)
    first = agent_tools.start_run(str(path))
    second = agent_tools.start_run(str(path))
    assert first["log"] != second["log"]


def test_start_run_refuses_before_launching(tmp_path: Path, recorded_popen):
    path = _write_config(tmp_path)
    with pytest.raises(ConfigError, match="unknown action"):
        agent_tools.start_run(str(path), action="format-disk")
    with pytest.raises(ConfigError, match="no step matches target"):
        agent_tools.start_run(str(path), action="rerun", target="nope")
    assert recorded_popen.calls == []


@pytest.mark.parametrize("action", ["run", "resume"])
def test_a_target_with_a_targetless_action_is_refused_here(
    tmp_path: Path, recorded_popen, action: str
):
    """`run`/`resume` take no target — refused where the docstring promises, not exit 2.

    The CLI's `run` and `resume` declare no positional target, so the detached child died
    on "unexpected extra argument" *after* this had returned a pid and a log path — an
    agent then polls a run that never started, with the reason only in the child log. The
    tool schema shows `action` and `target` side by side and nothing else says which pairs
    are legal.
    """
    path = _write_config(tmp_path, {"step": 1, "name": "screen", "engine": "fake"})
    with pytest.raises(ConfigError, match="takes no target"):
        agent_tools.start_run(str(path), action=action, target="screen")
    assert recorded_popen.calls == []


def test_start_run_refuses_a_live_lock_but_ignores_a_dead_one(tmp_path: Path, recorded_popen):
    path = _write_config(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    lock = outputs / pipeline.RUN_LOCK_NAME

    lock.write_text(
        json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "started": "now"}),
        encoding="utf-8",
    )
    with pytest.raises(RunLockError, match=str(os.getpid())):
        agent_tools.start_run(str(path))
    assert recorded_popen.calls == []

    lock.write_text("", encoding="utf-8")
    # A 0-byte lock — a driver killed between creating and writing it — is what
    # `run_lock` refuses as unreadable, so launching a child would hand back a pid
    # for a run that exits 10 before anyone watches its log.
    with pytest.raises(RunLockError, match="unreadable lock file"):
        agent_tools.start_run(str(path))
    assert recorded_popen.calls == []

    lock.write_text(
        json.dumps({"host": socket.gethostname(), "pid": 2**22 + 1, "started": "then"}),
        encoding="utf-8",
    )
    agent_tools.start_run(str(path))  # a dead holder is the stale case run_lock reclaims
    assert len(recorded_popen.calls) == 1


# ---------------------------------------------------------------------------
# lock_status (the pipeline seam start_run and the GUI read)
# ---------------------------------------------------------------------------


def test_lock_status_reads_without_touching(tmp_path: Path):
    assert pipeline.lock_status(tmp_path).held is False

    lock = tmp_path / pipeline.RUN_LOCK_NAME
    lock.write_text("not json", encoding="utf-8")
    # Present but unreadable is HELD, with no holder to name — the side `run_lock`
    # puts the same file on. Reported "not held", start_run sailed past its own gate
    # and launched a child that immediately exited 10 into an unwatched log.
    unreadable = pipeline.lock_status(tmp_path)
    assert (unreadable.held, unreadable.host, unreadable.alive) == (True, None, None)

    lock.write_text(
        json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "started": "now"}),
        encoding="utf-8",
    )
    live = pipeline.lock_status(tmp_path)
    assert (live.held, live.alive, live.pid) == (True, True, os.getpid())

    lock.write_text(
        json.dumps({"host": "somewhere-else", "pid": 1, "started": "now"}), encoding="utf-8"
    )
    foreign = pipeline.lock_status(tmp_path)
    assert (foreign.held, foreign.alive) == (True, None)  # cannot probe, must assume held


# ---------------------------------------------------------------------------
# Status / results / failures — filesystem truth
# ---------------------------------------------------------------------------


def _reported_tree(tmp_path: Path) -> Path:
    """A config whose output tree carries what the pipeline's own writers persist."""
    path = _write_config(
        tmp_path, {"step": 1, "name": "screen", "engine": "fake"}, {"step": 2, "engine": "fake"}
    )
    outputs = tmp_path / "outputs"
    io.save_step_csv(
        energies_hartree=[-1.0, -0.9], structure_ids=["0", "1"], step_number=1, output_dir=outputs
    )
    io.save_step_csv(
        energies_hartree=[-1.1], structure_ids=["0"], step_number=2, output_dir=outputs
    )
    step1 = outputs / "step1_screen"
    save_failure_records(
        step1,
        [FailureRecord(structure_id="2", kind=FailureKind.MISSING_OUTPUT, reason="no output")],
    )
    log_dir = outputs / "agent_runs"
    log_dir.mkdir(parents=True)
    (log_dir / "20260815T000000Z-run.log").write_text("line1\nline2\nline3\n", encoding="utf-8")
    return path


def test_run_status_reads_the_persisted_truth(tmp_path: Path):
    status = agent_tools.run_status(str(_reported_tree(tmp_path)), log_tail_lines=2)
    assert status["running"] is False
    assert status["holder"] is None
    screen, refine = status["steps"]
    assert (screen["reported_survivors"], screen["failures"]) == (2, 1)
    assert (refine["reported_survivors"], refine["failures"]) == (1, 0)
    assert status["log_tail"] == ["line2", "line3"]


def test_run_status_holder_carries_the_three_valued_liveness(tmp_path: Path):
    """The holder payload says not just *who* but whether the pid could be proven alive.

    ``alive`` is what an agent cannot re-derive over MCP: ``True``/``False`` for a
    same-host holder, ``null`` for a foreign one nothing here can probe — the difference
    between "held by a live run" and "held, unverifiable from this side".
    """
    path = _reported_tree(tmp_path)
    lock = tmp_path / "outputs" / pipeline.RUN_LOCK_NAME
    lock.write_text(
        json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "started": "now"}),
        encoding="utf-8",
    )
    status = agent_tools.run_status(str(path))
    assert status["running"] is True
    assert status["holder"]["pid"] == os.getpid()
    assert status["holder"]["alive"] is True


def test_run_status_on_a_fresh_tree_is_all_zeros(tmp_path: Path):
    status = agent_tools.run_status(str(_write_config(tmp_path)))
    assert status["steps"][0]["reported_survivors"] == 0
    assert status["log"] is None
    assert status["log_tail"] is None


@pytest.mark.parametrize("wanted", [0, -5])
def test_a_zero_or_negative_tail_means_no_tail(tmp_path: Path, wanted: int):
    """``log_tail_lines: 0`` is an empty tail — never the whole file.

    Python's ``-0 == 0``, so the bare slice ``[-0:]`` read a zero as "everything": the one
    value the module's pagination rule exists to forbid became the multi-MB driver log in
    a tool result. Negative values inverted the same way (``[-(-5):]`` drops five lines
    and keeps the rest). The log's *path* still reports, so a caller who asked for no
    lines can still name the file.
    """
    status = agent_tools.run_status(str(_reported_tree(tmp_path)), log_tail_lines=wanted)
    assert status["log_tail"] == []
    assert status["log"] is not None


def test_the_tail_is_read_from_the_end_not_by_reading_the_whole_log(tmp_path: Path):
    """A driver log outgrows its tail; the read must not grow with it.

    Spans several chunks so the backward walk really loops, and asserts on bytes read
    rather than on the answer alone — the answer was always right, it was the cost of
    getting it that scaled, once every five seconds for the length of the run.
    """
    path = _reported_tree(tmp_path)
    log = next((tmp_path / "outputs" / "agent_runs").glob("*.log"))
    log.write_text("".join(f"line {i:06d}\n" for i in range(60_000)), encoding="utf-8")
    assert log.stat().st_size > 4 * agent_tools._TAIL_CHUNK  # several backward steps

    read = 0
    real_open = Path.open

    def counting_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        if self == log:
            inner = handle.read

            def read_counting(size: int = -1) -> Any:
                nonlocal read
                chunk = inner(size)
                read += len(chunk)
                return chunk

            handle.read = read_counting
        return handle

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", counting_open)
        status = agent_tools.run_status(str(path), log_tail_lines=3)

    assert status["log_tail"] == ["line 059997", "line 059998", "line 059999"]
    assert read < log.stat().st_size // 4  # a whole-file read is what this rules out


def test_a_multibyte_character_split_across_a_chunk_boundary_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Decode once at the end, never per block.

    Reading backwards means chunk boundaries land at arbitrary byte offsets, so a
    multi-byte character gets cut in half. Decoding each block on its own would turn both
    halves into replacement characters — ``errors="replace"``, meant to make partial
    output safe, corrupting output that was never partial. A tiny chunk forces the case.
    """
    monkeypatch.setattr(agent_tools, "_TAIL_CHUNK", 8)
    path = _reported_tree(tmp_path)
    log = next((tmp_path / "outputs" / "agent_runs").glob("*.log"))
    log.write_text("ΔE = -12.5 kcal/mol · 1.09 Å\nrésumé ✓\n", encoding="utf-8")
    status = agent_tools.run_status(str(path), log_tail_lines=2)
    assert status["log_tail"] == ["ΔE = -12.5 kcal/mol · 1.09 Å", "résumé ✓"]


def test_get_results_paginates_and_filters(tmp_path: Path):
    path = _reported_tree(tmp_path)
    everything = agent_tools.get_results(str(path))
    assert everything["total"] == 3

    page = agent_tools.get_results(str(path), limit=1, offset=1)
    assert page["total"] == 3
    assert len(page["rows"]) == 1

    screen_only = agent_tools.get_results(str(path), step="screen")
    assert screen_only["total"] == 2
    assert {row["Step"] for row in screen_only["rows"]} == {"1"}

    with pytest.raises(ConfigError, match="no step matches"):
        agent_tools.get_results(str(path), step=9)


@pytest.mark.parametrize(
    ("limit", "offset", "expected"),
    [
        (-1, 0, 0),  # "no limit" is not a licence to slice from the end
        (-99, 0, 0),
        (0, 0, 0),
        (20, -2, 3),  # a pager walking back past zero starts at zero, not at the tail
        (2, -5, 2),
    ],
)
def test_negative_pagination_never_slices_from_the_end(
    tmp_path: Path, limit: int, offset: int, expected: int
):
    """The guard ``run_status``'s ``log_tail_lines`` already had, on its three siblings.

    A bare slice reads a negative as "count from the end", so ``limit=-1`` returned every
    row but the last while ``total`` still reported them all — a caller one row short with
    nothing in the payload to say so — and a negative ``offset`` re-served the tail under
    an offset no pager could page from. Both are the inversion the module's pagination
    rule exists to forbid, and both are now clamped to the empty/first page.
    """
    path = _reported_tree(tmp_path)
    page = agent_tools.get_results(str(path), limit=limit, offset=offset)
    assert page["total"] == 3  # the count is of the filtered set, always
    assert len(page["rows"]) == expected
    assert page["offset"] == max(0, offset)  # the answer describes the slice returned
    if expected:
        assert page["rows"] == agent_tools.get_results(str(path), limit=expected)["rows"]


def test_a_huge_limit_is_capped_and_says_so(tmp_path: Path):
    """The module's stated pagination rule, enforced rather than merely written down.

    ``limit`` went straight into a slice, so ``limit=10**9`` returned the whole ensemble —
    exactly what "a tool result cannot flood a model's context window" forbids. The cap
    cannot be silent either: ``total`` stays the unpaginated count and the answer echoes
    the ``limit`` actually applied, so a caller can always see there is more.
    """
    path = _reported_tree(tmp_path)
    page = agent_tools.get_results(str(path), limit=10**9)
    assert page["limit"] == agent_tools._MAX_ROWS
    assert page["total"] == 3  # three rows here, so the cap does not bite the result
    assert len(page["rows"]) == 3
    # A limit under the ceiling is passed through untouched.
    assert agent_tools.get_results(str(path), limit=2)["limit"] == 2


def test_get_failures_carries_the_taxonomy_and_the_next_move(tmp_path: Path):
    path = _reported_tree(tmp_path)
    everything = agent_tools.get_failures(str(path))
    [failure] = everything["failures"]
    assert failure == {
        "step": 1,
        "structure_id": "2",
        "kind": "output missing",
        "reason": "no output",
    }
    assert everything["exit_codes"]["ConfigError"] == 2
    # An indirect subclass, and the one the payload used to omit: this map was built from
    # `ChemRefineError.__subclasses__()`, which is direct-only, so a payload advertising
    # the whole taxonomy quietly shipped without the class a died-mid-run parse raises.
    assert everything["exit_codes"]["OutputTerminationError"] == 6
    assert everything["suggested_action"] == "rerun-errors"

    clean = agent_tools.get_failures(str(path), step=2)
    assert clean["failures"] == []
    assert clean["suggested_action"] is None
