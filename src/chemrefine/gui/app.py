"""Flask app factory for the YAML-builder GUI — thin JSON endpoints, one YAML emitter.

Every endpoint either re-exposes a tested library seam (schema, validation, scaffold,
templates, summary — the same calls the MCP server serves) or does one small job the
browser must not: ``/api/yaml`` and ``/api/parse`` are the *single* YAML
implementation in the system (server-side PyYAML), so the form pane and the text pane
cannot drift the way two serializers would. ``/api/browse`` exists because a browser
cannot list the local filesystem for the path pickers; it serves directory listings
only, for the same local user who launched the process.

Security posture: this is a local tool. :mod:`chemrefine.gui.serve` binds 127.0.0.1
only and mints a per-session token; every ``/api/*`` request must carry it in
``X-ChemRefine-Token`` (compared with :func:`secrets.compare_digest`). The static
assets are public — they contain no secrets — and the token travels once in the launch
URL's fragment-free query string, which the frontend keeps in memory.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from chemrefine import agent_tools, introspect
from chemrefine.errors import ChemRefineError

if TYPE_CHECKING:
    from flask import Flask

STATIC_DIR = Path(__file__).parent / "static"


def create_app(*, token: str | None, config_path: Path | None = None) -> Flask:
    """Build the GUI app.

    ``token`` is the per-session secret every ``/api/*`` call must present
    (``None`` disables the check — a unit-test affordance; :func:`.serve.launch`
    always passes one). ``config_path`` preloads an existing config into the builder.
    The flask import stays inside, like every optional-extra server here, so importing
    the package without the extra keeps working.
    """
    from flask import Flask, jsonify, request, send_from_directory
    from werkzeug.exceptions import HTTPException

    app = Flask("chemrefine-gui", static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["PROPAGATE_EXCEPTIONS"] = True

    def _authorized() -> bool:
        """Whether this request carries the session token (or the check is off)."""
        if token is None:
            return True
        presented = request.headers.get("X-ChemRefine-Token", "")
        return secrets.compare_digest(presented, token)

    @app.get("/")
    def index() -> Any:
        """The single page; everything after this is fetch calls from its JS."""
        return send_from_directory(str(STATIC_DIR), "index.html")

    @app.before_request
    def gate() -> Any:
        """Reject unauthorized ``/api/*`` requests before any handler runs."""
        if request.path.startswith("/api/") and not _authorized():
            return jsonify({"error": "unauthorized"}), 401
        return None

    @app.errorhandler(Exception)
    def surface(error: Exception) -> Any:
        """A ChemRefineError becomes its documented shape; anything else re-raises.

        The exit-code taxonomy is the GUI's error contract exactly as it is the CLI's
        and the MCP server's — one vocabulary for one failure. Routine HTTP errors
        (a 404 for a URL nobody serves, e.g. a browser probing /favicon.ico) pass
        through as themselves — re-raising them turned every stray request into a
        logged traceback.
        """
        if isinstance(error, HTTPException):
            return error
        if isinstance(error, ChemRefineError):
            return jsonify({"error": str(error), "exit_code": error.exit_code}), 400
        raise error

    @app.get("/api/bootstrap")
    def bootstrap() -> Any:
        """Everything the frontend needs to render: schema document + launch context."""
        initial: dict[str, Any] | None = None
        if config_path is not None and config_path.is_file():
            initial = {
                "path": str(config_path),
                "yaml_text": config_path.read_text(encoding="utf-8"),
            }
        return jsonify({"schema": introspect.schema_document(), "initial": initial})

    @app.post("/api/validate")
    def validate() -> Any:
        """The structured validation report for the current editor text."""
        payload = request.get_json(force=True)
        return jsonify(
            agent_tools.validate_config(payload["yaml_text"], payload.get("base_dir"))
        )

    @app.post("/api/yaml")
    def to_yaml() -> Any:
        """Form state (a raw config mapping) → YAML text. The one emitter."""
        payload = request.get_json(force=True)
        text = yaml.safe_dump(payload["config"], sort_keys=False, allow_unicode=True)
        return jsonify({"yaml_text": text})

    @app.post("/api/parse")
    def parse() -> Any:
        """YAML text → the raw mapping the form edits (or the reason it cannot)."""
        payload = request.get_json(force=True)
        try:
            raw = yaml.safe_load(payload["yaml_text"])
        except yaml.YAMLError as e:
            return jsonify({"error": f"malformed YAML: {e}"}), 400
        if not isinstance(raw, dict):
            return jsonify({"error": "config is not a YAML mapping"}), 400
        return jsonify({"config": raw})

    @app.get("/api/browse")
    def browse() -> Any:
        """One directory level for the path pickers (local machine, local user)."""
        requested = request.args.get("path") or str(Path.home())
        path = Path(requested).expanduser().resolve()
        if not path.is_dir():
            return jsonify({"error": f"not a directory: {path}"}), 400
        entries = sorted(
            (
                {"name": child.name, "path": str(child), "dir": child.is_dir()}
                for child in path.iterdir()
                if not child.name.startswith(".")
            ),
            key=lambda e: (not e["dir"], str(e["name"]).lower()),
        )
        return jsonify({"path": str(path), "parent": str(path.parent), "entries": entries})

    @app.post("/api/save")
    def save() -> Any:
        """Write the editor's YAML to disk — the artifact the whole GUI exists to make."""
        payload = request.get_json(force=True)
        destination = Path(payload["path"]).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload["yaml_text"], encoding="utf-8")
        return jsonify({"path": str(destination)})

    @app.post("/api/scaffold")
    def scaffold() -> Any:
        """Fill the saved config's template gaps with starters (kept/written report)."""
        payload = request.get_json(force=True)
        return jsonify(
            agent_tools.scaffold_templates(
                payload["config_path"], overwrite=bool(payload.get("overwrite", False))
            )
        )

    @app.get("/api/template")
    def read_template() -> Any:
        """One step's template text for the inline editor."""
        return jsonify(
            agent_tools.read_template(
                request.args["config_path"], _step_key(request.args["step"])
            )
        )

    @app.post("/api/template")
    def write_template() -> Any:
        """Save the inline editor's template text."""
        payload = request.get_json(force=True)
        return jsonify(
            agent_tools.write_template(
                payload["config_path"], _step_key(payload["step"]), payload["text"]
            )
        )

    @app.post("/api/summary")
    def summary() -> Any:
        """The dry-run-style execution summary for a saved config."""
        payload = request.get_json(force=True)
        return jsonify(agent_tools.summarize_config(payload["config_path"]))

    return app


def _step_key(value: str | int) -> int | str:
    """A step selector from the wire: digits mean the step number, anything else a name."""
    text = str(value)
    return int(text) if text.isdigit() else text
