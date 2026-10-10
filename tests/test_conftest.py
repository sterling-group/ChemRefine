"""The suite's own filesystem fixtures, held to what they promise.

On a local disk the wait ``_modes_take_hold`` makes is over before it begins, every
directory ``list_only`` leaves is listed, and no clock is too coarse for a rewrite. So a
run there executes none of what these fixtures do for a slower or stricter filesystem, and
a change that broke it would show only on one. These tests bring that filesystem along.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import conftest
import pytest

pytestmark = pytest.mark.skipif(os.geteuid() == 0, reason="root reads through any mode")


def _answering_for(
    monkeypatch: pytest.MonkeyPatch, entry: Path, *, times: int | None
) -> list[Path]:
    """Make ``os.lstat`` go on answering for ``entry``: ``times`` more times, or for good.

    What a filesystem does that resolves an entry from memory after the directory holding
    it was walled off. Returns the list each such answer is added to.
    """
    lstat = os.lstat
    answered: list[Path] = []

    def remembering(path: Any, **how: Any) -> os.stat_result:
        if Path(os.fsdecode(path)) == entry and (times is None or len(answered) < times):
            answered.append(entry)
            return lstat(entry.parent.parent)
        return lstat(path, **how)

    monkeypatch.setattr(os, "lstat", remembering)
    return answered


def test_chmod_returns_from_a_wall_only_once_the_filesystem_holds_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Through ``Path.chmod``, as every test builds a wall: it returns when the entry is refused."""
    wall = tmp_path / "wall"
    (wall / "inside").mkdir(parents=True)
    answered = _answering_for(monkeypatch, wall / "inside", times=3)

    wall.chmod(0)
    try:
        assert len(answered) == 3, "chmod returned while the entry still resolved"
        with pytest.raises(PermissionError):
            os.lstat(wall / "inside")
    finally:
        wall.chmod(0o755)


def test_a_wall_that_never_comes_fails_the_test_and_is_taken_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """No test goes on behind a wall that is not there, and none leaves one half-built."""
    wall = tmp_path / "wall"
    (wall / "inside").mkdir(parents=True)
    wall.chmod(0o750)
    _answering_for(monkeypatch, wall / "inside", times=None)
    monkeypatch.setattr(conftest, "_HOLD_SECONDS", 0.0)

    with pytest.raises(pytest.fail.Exception, match="still answers behind mode 000"):
        wall.chmod(0)

    assert stat.S_IMODE(wall.stat().st_mode) == 0o750


def test_a_mode_that_walls_nothing_off_is_set_with_no_question_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Read-only directories, files and loosened modes pass straight through."""
    folder = tmp_path / "folder"
    (folder / "inside").mkdir(parents=True)
    file = folder / "file"
    file.write_text("", encoding="utf-8")
    answered = _answering_for(monkeypatch, folder / "inside", times=None)

    folder.chmod(0o555)
    file.chmod(0)
    folder.chmod(0o755)

    assert answered == []
    assert stat.S_IMODE(file.stat().st_mode) == 0


def test_a_filesystem_without_the_listing_skips_the_list_only_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, list_only: Callable[[Path], Path]
):
    """A directory that may be read and not searched is not listed everywhere."""
    folder = tmp_path / "folder"
    folder.mkdir()
    scandir = os.scandir

    def refusing(path: Any) -> Any:
        if Path(os.fsdecode(path)) == folder:
            raise PermissionError(13, "Permission denied", str(folder))
        return scandir(path)

    monkeypatch.setattr(os, "scandir", refusing)

    with pytest.raises(pytest.skip.Exception, match="lists no directory it may not search"):
        list_only(folder)


def test_back_dating_tells_a_rewrite_in_place_on_a_clock_too_coarse_to(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    back_dated: Callable[[Path], Callable[[], bool]],
):
    """The same bytes under the same inode, within one tick of the filesystem's clock.

    The clock here keeps whole days, where the share that showed this keeps whole seconds:
    read before and after, such a file's time is the time it had.
    """
    document = tmp_path / "document"
    document.write_text("as it was", encoding="utf-8")
    status, day = Path.stat, 86_400 * 10**9

    def by_the_day(self: Path, **how: Any) -> Any:
        real = status(self, **how)
        if self != document:
            return real
        return SimpleNamespace(st_ino=real.st_ino, st_mtime_ns=real.st_mtime_ns // day * day)

    monkeypatch.setattr(Path, "stat", by_the_day)
    rewritten = back_dated(document)

    assert not rewritten()
    with document.open("r+", encoding="utf-8") as handle:
        handle.write("as it was")
    assert rewritten()
