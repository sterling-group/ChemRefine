"""Sidecar files — kernel-assigned-port and auth-token handoff for the server.

When the server binds ``host:0`` the kernel picks a free port; it records the
actual ``host:port`` to the URL sidecar so the driver's bridge can find it,
and the per-run bearer token to the token sidecar so only the owning run can
call ``/calculate``. Engine-neutral: any out-of-process backend server /
client pair can use these. Both are replaced whole and owner-only
(:func:`chemrefine.io.atomic_write` with mode ``0o600``), so the bridge never reads one
half-written and no other account on the node reads either.
"""

from __future__ import annotations

from pathlib import Path

from chemrefine.engines._backend_server.base import SERVER_TOKEN_FILENAME
from chemrefine.io import atomic_write


def write_server_url(url_file: str | Path, url: str) -> Path:
    """Atomically write ``host:port`` to ``url_file`` (tempfile + rename)."""
    path = Path(url_file)
    atomic_write(path, url.encode("utf-8"), mode=0o600)
    return path


def read_server_url(url_file: str | Path) -> str:
    """Return the ``host:port`` recorded by :func:`write_server_url`."""
    return Path(url_file).read_text(encoding="utf-8").strip()


def token_file_for(url_file: str | Path) -> Path:
    """Where a server's token sidecar sits: beside its URL sidecar, for writer and reader alike."""
    return Path(url_file).with_name(SERVER_TOKEN_FILENAME)


def write_server_token(token_file: str | Path, token: str) -> Path:
    """Atomically write the per-run bearer token, owner-readable only (0600)."""
    path = Path(token_file)
    atomic_write(path, token.encode("utf-8"), mode=0o600)
    return path


def read_server_token(token_file: str | Path) -> str:
    """Return the token recorded by :func:`write_server_token`."""
    return Path(token_file).read_text(encoding="utf-8").strip()
