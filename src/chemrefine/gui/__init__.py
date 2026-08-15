"""The click-through YAML-builder GUI (``chemrefine gui``) — an optional local web app.

Split like every optional server in this codebase: :mod:`.app` is the Flask app factory
(thin, fully-tested JSON endpoints over :mod:`chemrefine.agent_tools`,
:mod:`chemrefine.introspect` and :mod:`chemrefine.validate`), :mod:`.serve` binds it to
127.0.0.1 behind a per-session token and opens the browser. All layout and interaction
live in the static assets (vendored Alpine.js, no build step); the Python surface only
answers questions the library already answers, plus the one job the browser must never
do itself — YAML emission and parsing stay server-side so there is exactly one YAML
implementation in the system.
"""
