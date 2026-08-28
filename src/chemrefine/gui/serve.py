"""Bind the GUI app to loopback behind a fresh token and open the browser.

The serving posture in one place: **127.0.0.1 only** (reaching a cluster means SSH port
forwarding, not a bind flag), a per-session ``secrets.token_urlsafe`` secret carried
once in the launch URL's query string, waitress as the WSGI server (the same one the
``[server]`` extra already pins for the ExtOpt sidecar), and a **stable per-user port**
by default — hashed from the username, so an SSH forwarding setup written once keeps
working across sessions — with a kernel-assigned free port as the fallback when that one
is taken (``--port 0`` asks for a kernel port outright). The socket is bound here and
handed to waitress, so the effective port is known before the URL is printed or opened.
A session with no way to show a *window* gets the SSH forwarding recipe printed instead —
never a text browser hijacking the terminal — and a remote session gets the recipe even
when something did open, because a local tunnel beats a forwarded display.

The token rides in that URL's query string, which means it also rides in the ``argv`` of
whatever browser process is launched. See ``docs/internals/security.md``: on a shared node
that is readable by other users, and ``--no-browser`` is the way to avoid it.

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
import sys
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


def _opens_a_window() -> bool:
    """Whether handing the URL to ``webbrowser`` opens a window rather than seizing this
    terminal.

    This used to be a ``DISPLAY`` check, and ``DISPLAY`` is the wrong question. ``ssh -X``
    sets it on a login node that has no graphical browser at all, and ``webbrowser``
    registers the console browsers — lynx, w3m, links — whenever ``TERM`` is set, so it
    then hands the URL to one of those. Their ``open()`` *waits* for the child, so a text
    browser draws over the user's shell and the server does not start until they quit it:
    exactly the outcome this guard was written to prevent, reached through the branch it
    did not check.

    :class:`webbrowser.BackgroundBrowser` is the discriminator. It is the class every
    windowed launcher registers as, and the only one whose ``open()`` returns without
    waiting; the console browsers are plain :class:`~webbrowser.GenericBrowser`. The
    ``DISPLAY`` test is kept ahead of it because it is free and certain — no display, no
    window — and because it answers before ``webbrowser`` builds its whole try-order.

    What this does **not** settle, stated so nobody reads more into it: ``xdg-open`` and
    friends are ``BackgroundBrowser`` and answer yes here, but they *delegate* — on a node
    where nothing graphical is installed they go on to hand the URL to a console browser
    anyway. Predicting that would mean asking the desktop's MIME associations, which is not
    portable. It is bounded rather than solved: those launchers run with
    ``start_new_session=True`` so the child has no controlling terminal, and the realistic
    instance of it — a login node reached over SSH — takes the recipe branch through
    :func:`_is_remote_session` regardless of what the launcher claims.
    """
    if sys.platform in ("darwin", "win32"):
        return True
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        return isinstance(webbrowser.get(), webbrowser.BackgroundBrowser)
    except webbrowser.Error:  # nothing registered at all
        return False


def _is_remote_session() -> bool:
    """Whether this shell arrived over SSH — the case a forwarding recipe exists for.

    Consulted only to decide whether to *also* print the recipe, never to decide whether
    to launch a browser. That distinction is what keeps the old rationale intact:
    tmux/screen and batch jobs drop ``SSH_CONNECTION``, so it must not gate the launch —
    but a session that does carry it is one where a local tunnel beats whatever the
    forwarded display is doing, and saying so costs two lines.
    """
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))


def _log_forwarding_recipe(port: int) -> None:
    """The copy-paste route from the machine with the browser to this loopback port."""
    # SSH_CONNECTION's third field is the address the user's ssh client actually
    # connected to — gethostname() is usually an internal node name a laptop can't
    # resolve, so it is only mentioned, never prescribed.
    fields = os.environ.get("SSH_CONNECTION", "").split()
    host = fields[2] if len(fields) == 4 else "<the host you ssh to>"
    # Worded for both callers: this prints when nothing opened here *and* when something
    # did but the session is remote, where a tunnel still beats a forwarded display.
    # "no browser here" was true only of the first and would have been a lie in the second.
    logger.info("to reach it from your machine: ssh -L %d:127.0.0.1:%d %s", port, port, host)
    logger.info(
        "or once in ~/.ssh/config on your machine — Host %s / LocalForward %d 127.0.0.1:%d — "
        "and every future login carries the tunnel",
        host,
        port,
        port,
    )
    logger.info(
        "then open the URL above in your local browser (this node is %s)", socket.gethostname()
    )


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
    # SO_REUSEADDR, because binding by hand bypasses the one waitress would have set: a
    # stop-and-restart with a browser tab still connected leaves the previous session's
    # connections in TIME_WAIT, and without the flag the personal-port bind fails for the
    # next ~60 s — silently moving a "stable per-user port" onto a kernel-assigned one,
    # which is exactly the SSH-forwarding breakage the stable port exists to prevent. Safe
    # for a loopback listener: the flag admits rebinding over TIME_WAIT remnants, not over
    # a *live* listener, so a genuinely squatted port still refuses below.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if port is None:
        try:
            sock.bind(("127.0.0.1", _personal_port()))
        except OSError:  # a live listener squats the port, or the hash collides — any
            sock.bind(("127.0.0.1", 0))  # free port instead, and the printed URL says so
    else:
        sock.bind(("127.0.0.1", port))
    bound_port = sock.getsockname()[1]
    server = create_server(app, sockets=[sock])
    url = f"http://127.0.0.1:{bound_port}/?token={token}"
    logger.info("ChemRefine GUI: %s", url)
    if open_browser:
        # `webbrowser.open` returning True is not evidence that a window opened: every
        # launcher registers as BackgroundBrowser, whose open() reports `poll() is None`
        # the instant Popen succeeds. On a node where `xdg-open` exists but finds no
        # browser it still returns True, having delegated to whatever the session
        # associates with http. So the recipe is withheld only when a browser opened *and*
        # this is the machine the user is sitting at.
        opened = _opens_a_window() and webbrowser.open(url)
        if not opened or _is_remote_session():
            _log_forwarding_recipe(bound_port)
    server.run()
