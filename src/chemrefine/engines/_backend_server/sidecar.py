"""Sidecar files — kernel-assigned-port and auth-token handoff for the server.

When the server binds ``host:0`` the kernel picks a free port; it records the
actual ``host:port`` to the URL sidecar so the driver's bridge can find it,
and the per-run bearer token to the token sidecar so only the owning run can
call ``/calculate``. Engine-neutral: any out-of-process backend server /
client pair can use these. Both are written through :func:`write_atomic`, the one writer
for a file that live processes read while it may be rewritten — which is also how the
ExtOpt ``ProgExt`` wrapper is written.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic(target: Path, content: str, *, prefix: str, mode: int | None = None) -> Path:
    """Replace ``target`` with ``content`` in one rename, so no reader ever sees it partial.

    Tempfile + rename in the target's own directory: a process that opens or executes
    ``target`` at any moment gets the old file or the new one, whole — never the empty one an
    in-place write leaves between its truncate and its write. Shared by both sidecars and
    the ExtOpt ``ProgExt`` wrapper, which ORCA executes once per optimiser step while
    another job's retry may be preparing it again.

    ``mkstemp`` creates the temp file ``0600`` and the rename preserves that
    mode, so a secret written through here is never readable by other users,
    even transiently. ``mode`` replaces it *before* the rename (``0o755`` for a script),
    so the file is never visible under ``target`` with the wrong one either.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=prefix, dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        if mode is not None:
            Path(tmp_name).chmod(mode)
        Path(tmp_name).replace(target)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def write_server_url(url_file: str | Path, url: str) -> Path:
    """Atomically write ``host:port`` to ``url_file`` (tempfile + rename)."""
    return write_atomic(Path(url_file), url, prefix=".url.")


def read_server_url(url_file: str | Path) -> str:
    """Return the ``host:port`` recorded by :func:`write_server_url`."""
    return Path(url_file).read_text(encoding="utf-8").strip()


def write_server_token(token_file: str | Path, token: str) -> Path:
    """Atomically write the per-run bearer token, owner-readable only (0600)."""
    return write_atomic(Path(token_file), token, prefix=".token.")


def read_server_token(token_file: str | Path) -> str:
    """Return the token recorded by :func:`write_server_token`."""
    return Path(token_file).read_text(encoding="utf-8").strip()
