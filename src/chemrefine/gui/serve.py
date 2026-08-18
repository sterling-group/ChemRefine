"""Bind the GUI app to loopback behind a fresh token and open the browser.

The serving posture in one place: **127.0.0.1 only** (reaching a cluster means SSH port
forwarding, not a bind flag), a per-session ``secrets.token_urlsafe`` secret carried
once in the launch URL's query string, waitress as the WSGI server (the same one the
``[server]`` extra already pins for the ExtOpt sidecar), and port 0 by default so the
kernel picks a free port — the effective one is read back off the bound socket before
the URL is printed or opened.

waitress is imported at module scope for the reason :mod:`.app` imports Flask there: this
is the module the CLI imports inside ``except ImportError``, so a deferred import would
leave that guard unable to fire and ``chemrefine gui`` would traceback instead of naming
the extra to install.
"""

from __future__ import annotations

import logging
import secrets
import webbrowser
from pathlib import Path

from waitress.server import create_server

from chemrefine.gui.app import create_app

logger = logging.getLogger(__name__)


def launch(config_path: Path | None = None, *, port: int = 0, open_browser: bool = True) -> None:
    """Serve the GUI until interrupted; print (and usually open) the tokened URL."""
    token = secrets.token_urlsafe(16)
    app = create_app(token=token, config_path=config_path)
    server = create_server(app, host="127.0.0.1", port=port)
    url = f"http://127.0.0.1:{server.effective_port}/?token={token}"
    logger.info("ChemRefine GUI: %s", url)
    if open_browser:
        webbrowser.open(url)
    server.run()
