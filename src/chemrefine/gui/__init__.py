"""The click-through YAML-builder GUI (``chemrefine gui``) — an optional local web app.

Split like every optional server in this codebase: :mod:`.app` is the Flask app factory
(thin, fully-tested JSON endpoints over :mod:`chemrefine.agent_tools`,
:mod:`chemrefine.introspect` and :mod:`chemrefine.validate`), :mod:`.serve` binds it to
127.0.0.1 behind a per-session token and opens the browser. All layout and interaction
live in the static assets (vendored Alpine.js, no build step); the Python surface only
answers questions the library already answers, plus the one job the browser must never
do itself — YAML emission and parsing stay server-side so there is exactly one YAML
implementation in the system.

:data:`STATIC_DIR` lives here rather than in :mod:`.app` because it is a fact about this
package's own layout, and because it has one consumer that must not need Flask: the docs
build copies the assets into the online playground, and the ``[docs]`` extra installs no
server. Keeping it here is what lets :mod:`.app` import Flask at module scope, which is
what makes the CLI's ``except ImportError`` guard fire on a missing extra.
"""

from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"
"""The vendored frontend: ``index.html``, the two scripts, the stylesheet, the vendor bundle."""
