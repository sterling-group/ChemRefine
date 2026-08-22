"""Bind the GUI app to loopback behind a fresh token and open the browser.

The serving posture in one place: **127.0.0.1 only** (reaching a cluster means SSH port
forwarding, not a bind flag), a per-session ``secrets.token_urlsafe`` secret carried
once in the launch URL's query string, waitress as the WSGI server (the same one the
``[server]`` extra already pins for the ExtOpt sidecar), and a **stable per-user port**
by default — hashed from the username, so an SSH forwarding setup written once keeps
working across sessions — with a kernel-assigned free port as the fallback when that one
is taken (``--port 0`` asks for a kernel port outright). The socket is bound here and
handed to waitress, so the effective port is known before the URL is printed or opened.

waitress is imported at module scope for the reason :mod:`.app` imports Flask there: this
is the module the CLI imports inside ``except ImportError``, so a deferred import would
leave that guard unable to fire and ``chemrefine gui`` would traceback instead of naming
the extra to install.
"""

from __future__ import annotations

import getpass
import hashlib
import logging
import os
import secrets
import socket
import webbrowser
from pathlib import Path

from waitress.server import create_server

from chemrefine.gui.app import create_app

logger = logging.getLogger(__name__)

_PORT_BASE = 20000
"""Bottom of the personal-port window — below Linux's ephemeral range (32768-60999), so
a long-lived listener never trips over a transient outgoing socket."""

_PORT_SPAN = 10000
"""Width of the personal-port window: hashed ports land in [20000, 29999]."""


def _personal_port() -> int:
    """A stable per-user port, so a forwarding stanza written once keeps working.

    Hashed rather than configured: every user on a shared login node gets their own
    default without coordination, and the same user gets the same port everywhere.
    """
    # getpass.getuser() raises in passwd-less environments (containers under an
    # arbitrary UID); same guard and rationale as slurm.dispatch._current_user — not
    # imported from there, the GUI must not pull in the scheduler stack.
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = str(os.getuid())
    digest = hashlib.sha1(user.encode(), usedforsecurity=False).hexdigest()
    return _PORT_BASE + int(digest[:8], 16) % _PORT_SPAN


def launch(
    config_path: Path | None = None, *, port: int | None = None, open_browser: bool = True
) -> None:
    """Serve the GUI until interrupted; print (and usually open) the tokened URL.

    ``port=None`` means the personal default (falling back to a kernel-assigned port);
    an explicit port that is taken raises ``OSError`` to the CLI unchanged.
    """
    token = secrets.token_urlsafe(16)
    app = create_app(token=token, config_path=config_path)
    # Bound here, not by waitress: a create_server that dies on a taken port leaves its
    # already-started handler threads behind, and the pre-bound socket is what waitress's
    # documented ``sockets=`` parameter exists for (same pattern as the ExtOpt server).
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if port is None:
        try:
            sock.bind(("127.0.0.1", _personal_port()))
        except OSError:  # squatted or hash-shared — any free port, and the URL says so
            sock.bind(("127.0.0.1", 0))
    else:
        sock.bind(("127.0.0.1", port))
    bound_port = sock.getsockname()[1]
    server = create_server(app, sockets=[sock])
    url = f"http://127.0.0.1:{bound_port}/?token={token}"
    logger.info("ChemRefine GUI: %s", url)
    if open_browser:
        webbrowser.open(url)
    server.run()
