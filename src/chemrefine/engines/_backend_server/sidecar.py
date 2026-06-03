"""Sidecar URL file — kernel-assigned-port handoff for the backend server.

When the server binds ``host:0`` the kernel picks a free port; it records the
actual ``host:port`` to this sidecar file so the driver's bridge can find it.
Engine-neutral: any out-of-process backend server / client pair can use it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def write_server_url(url_file: str | Path, url: str) -> Path:
    """Atomically write ``host:port`` to ``url_file`` (tempfile + rename)."""
    target = Path(url_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".url.", dir=target.parent)
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            fh.write(url)
        Path(tmp_name).replace(target)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def read_server_url(url_file: str | Path) -> str:
    """Return the ``host:port`` previously written by :func:`write_server_url`."""
    return Path(url_file).read_text(encoding="utf-8").strip()
