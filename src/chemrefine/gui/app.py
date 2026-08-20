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

Flask is imported at module scope, the way :mod:`chemrefine.mcp_server` imports its SDK
and for the same reason: the CLI imports *this* module inside ``except ImportError`` to
turn a missing ``chemrefine[gui]`` into an install hint. An import deferred into
:func:`create_app` makes that guard unreachable, so ``chemrefine gui`` without the extra
raised a traceback past a handler written to prevent exactly that. Nothing here is
imported by a Flask-free caller — :data:`chemrefine.gui.STATIC_DIR` is where the docs
build gets the asset path.
"""

from __future__ import annotations

import importlib
import secrets
from pathlib import Path
from typing import Any

import yaml
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from chemrefine import agent_tools, introspect
from chemrefine.cache import atomic_write
from chemrefine.errors import ChemRefineError
from chemrefine.gui import STATIC_DIR


def create_app(*, token: str | None, config_path: Path | None = None) -> Flask:
    """Build the GUI app.

    ``token`` is the per-session secret every ``/api/*`` call must present
    (``None`` disables the check — a unit-test affordance; :func:`.serve.launch`
    always passes one). ``config_path`` preloads an existing config into the builder.
    """
    app = Flask("chemrefine-gui", static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["PROPAGATE_EXCEPTIONS"] = True

    def _authorized() -> bool:
        """Whether this request carries the session token (or the check is off)."""
        if token is None:
            return True
        presented = request.headers.get("X-ChemRefine-Token", "")
        # Compared as bytes: Werkzeug decodes headers as latin-1, and `compare_digest`
        # raises TypeError on a str holding a non-ASCII character — so a header with any
        # byte above 0x7F turned a plain 401 into a 500 and a logged traceback. Encoding
        # both sides answers every input in constant time instead of the one class of
        # wrong token we happened to think of.
        return secrets.compare_digest(presented.encode(), token.encode())

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
        """Everything the frontend needs to render: schema document + launch context.

        ``host`` names the machine the server runs on: over SSH port forwarding the
        browser's address bar always says 127.0.0.1, so this is the only way the UI can
        tell the user *where* Save… writes — the disambiguation from Download, which
        goes through the browser to the local machine.
        """
        import socket

        initial: dict[str, Any] | None = None
        if config_path is not None and config_path.is_file():
            initial = {
                "path": str(config_path),
                "yaml_text": config_path.read_text(encoding="utf-8"),
            }
        return jsonify(
            {
                "schema": introspect.schema_document(),
                "initial": initial,
                "host": socket.gethostname(),
            }
        )

    @app.post("/api/validate")
    def validate() -> Any:
        """The structured validation report for the current editor text."""
        payload = request.get_json(force=True)
        return jsonify(agent_tools.validate_config(payload["yaml_text"], payload.get("base_dir")))

    @app.post("/api/yaml")
    def to_yaml() -> Any:
        """Form state (a raw config mapping) → YAML text. The one emitter.

        Keys are emitted in the schema's declaration order — workflow settings first,
        ``steps`` last, and each step's keys in :class:`StepConfig` order — so the file
        reads like every shipped example regardless of the order the user clicked
        things together in.
        """
        payload = request.get_json(force=True)
        text = yaml.safe_dump(
            _canonical_order(payload["config"]), sort_keys=False, allow_unicode=True
        )
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
        try:
            entries = sorted(
                (
                    {"name": child.name, "path": str(child), "dir": child.is_dir()}
                    for child in path.iterdir()
                    if not child.name.startswith(".")
                ),
                key=lambda e: (not e["dir"], str(e["name"]).lower()),
            )
        except OSError as e:
            # A picker walks wherever the filesystem leads — another user's mode-700
            # directory, /lost+found — and an unreadable stop is an ordinary answer for
            # it, not the logged-traceback 500 the re-raising error handler would make.
            return jsonify({"error": f"cannot list {path}: {e}"}), 400
        return jsonify({"path": str(path), "parent": str(path.parent), "entries": entries})

    @app.post("/api/save")
    def save() -> Any:
        """Write the editor's YAML to disk — the artifact the whole GUI exists to make.

        Deliberately unvalidated — Save is for drafts too, and the Validate button is
        its own action — but atomic like :func:`chemrefine.agent_tools.save_config`'s
        write: a kill mid-save must not truncate the config a run is pointed at. An
        unwritable destination is a plain 400, like every other bad input here.
        """
        payload = request.get_json(force=True)
        destination = Path(payload["path"]).expanduser()
        try:
            atomic_write(destination, payload["yaml_text"].encode("utf-8"))
        except OSError as e:
            return jsonify({"error": f"cannot write {destination}: {e}"}), 400
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
            agent_tools.read_template(request.args["config_path"], _step_key(request.args["step"]))
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

    @app.post("/api/status")
    def status() -> Any:
        """Where the tree stands — lock holder, per-step progress, log tail."""
        payload = request.get_json(force=True)
        return jsonify(
            agent_tools.run_status(
                payload["config_path"],
                log_tail_lines=int(payload.get("log_tail_lines", 40)),
            )
        )

    @app.post("/api/results")
    def results() -> Any:
        """A paginated slice of steps.csv for the dashboard's results table."""
        payload = request.get_json(force=True)
        return jsonify(
            agent_tools.get_results(
                payload["config_path"],
                step=payload.get("step"),
                limit=int(payload.get("limit", 20)),
                offset=int(payload.get("offset", 0)),
            )
        )

    @app.post("/api/failures")
    def failures() -> Any:
        """The failure ledger plus the suggested recovery action."""
        payload = request.get_json(force=True)
        return jsonify(agent_tools.get_failures(payload["config_path"], step=payload.get("step")))

    # One user, one browser, one conversation: the chat state lives on the app instance
    # (history = PydanticAI's own message list; pending = the suspended run awaiting the
    # human's allow/deny verdicts).
    chat_state: dict[str, Any] = {"history": None, "pending": None}

    @app.get("/api/agent/availability")
    def agent_availability() -> Any:
        """Whether the chat panel can work here: extra installed, model configured."""
        try:
            # The probe *is* the import — whether the panel works here is exactly whether
            # this succeeds, which `find_spec` cannot answer (a package can be findable and
            # still fail to import). Spelled through `import_module` so the result is a
            # value rather than an unused binding needing a lint suppression to survive.
            importlib.import_module("pydantic_ai")
        except ImportError:
            return jsonify(
                {
                    "installed": False,
                    "configured": False,
                    "detail": "pip install 'chemrefine[agent]'",
                }
            )
        from chemrefine.agent.providers import ProviderConfig

        try:
            resolved = ProviderConfig.resolve()
            return jsonify({"installed": True, "configured": True, "detail": resolved.model})
        except ChemRefineError as e:
            return jsonify({"installed": True, "configured": False, "detail": str(e)})

    @app.post("/api/agent/chat")
    def agent_chat() -> Any:
        """One agent turn: a message, or the verdicts that resume a suspended run.

        Non-streaming by design (a local-model turn takes seconds to a minute inside
        this worker thread); the reply is either text or a list of approval requests
        the frontend renders as allow/deny cards. Provider errors — an unreachable
        Ollama, a bad key — come back as a 502 with the message, not a traceback: at
        the network boundary the failure is the answer.
        """
        from pydantic_ai import DeferredToolRequests, DeferredToolResults

        from chemrefine.agent.harness import build_web_agent
        from chemrefine.agent.providers import ProviderConfig

        payload = request.get_json(force=True)
        if payload.get("reset"):
            chat_state["history"] = None
            chat_state["pending"] = None
            return jsonify({"reply": None, "pending": None, "reset": True})
        resolved = ProviderConfig.resolve(
            payload.get("provider", "custom"),
            model=payload.get("model"),
            base_url=payload.get("base_url"),
        )
        agent = build_web_agent(
            resolved.build_model(),
            config_path=str(config_path) if config_path is not None else None,
        )
        try:
            if payload.get("approvals"):
                pending = chat_state["pending"]
                if pending is None:
                    return jsonify({"error": "no suspended run to resume"}), 400
                result = agent.run_sync(
                    message_history=pending,
                    deferred_tool_results=DeferredToolResults(
                        approvals={k: bool(v) for k, v in payload["approvals"].items()}
                    ),
                )
            else:
                result = agent.run_sync(payload["message"], message_history=chat_state["history"])
        except ChemRefineError:
            # A tool's own failure keeps its documented {error, exit_code} shape.
            raise
        except Exception as e:  # the model endpoint is the outside world
            return jsonify({"error": f"model endpoint failed: {e}"}), 502
        if isinstance(result.output, DeferredToolRequests):
            chat_state["pending"] = result.all_messages()
            return jsonify(
                {
                    "reply": None,
                    "pending": [
                        {"id": call.tool_call_id, "tool": call.tool_name, "args": call.args}
                        for call in result.output.approvals
                    ],
                }
            )
        chat_state["history"] = result.all_messages()
        chat_state["pending"] = None
        return jsonify({"reply": result.output, "pending": None})

    @app.post("/api/run")
    def run() -> Any:
        """Launch a detached run — same semantics as the agent's start_run.

        A held lock surfaces through the error handler as the documented
        ``{error, exit_code: 10}`` shape; the browser confirmed the action already,
        and the child outlives this server exactly as it outlives an agent session.
        """
        payload = request.get_json(force=True)
        return jsonify(
            agent_tools.start_run(
                payload["config_path"],
                action=payload.get("action", "run"),
                target=payload.get("target"),
                max_cores=payload.get("max_cores"),
                max_gpus=payload.get("max_gpus"),
            )
        )

    return app


def _step_key(value: str | int) -> int | str:
    """A step selector from the wire: digits mean the step number, anything else a name."""
    text = str(value)
    return int(text) if text.isdigit() else text


def _canonical_order(raw: Any) -> Any:
    """Reorder a raw config mapping into the schema's declaration order.

    The form assembles ``cfg`` in click order — a fresh session starts from
    ``{steps: []}``, so without this the emitted YAML led with the steps block and
    trailed the settings, unlike every example in the repository. The authority on
    ordering is the model itself: :class:`~chemrefine.config.Config`'s fields for the
    top level (``steps`` is declared last), :class:`~chemrefine.config.StepConfig`'s
    for each step. Unknown keys sort to the end, order preserved — validation is the
    place that complains about them, not the emitter.
    """
    from chemrefine.config import Config, StepConfig

    if not isinstance(raw, dict):
        return raw
    top_rank = {name: i for i, name in enumerate(Config.model_fields)}
    step_rank = {name: i for i, name in enumerate(StepConfig.model_fields)}
    ordered = dict(sorted(raw.items(), key=lambda kv: top_rank.get(kv[0], len(top_rank))))
    steps = ordered.get("steps")
    if isinstance(steps, list):
        ordered["steps"] = [
            dict(sorted(s.items(), key=lambda kv: step_rank.get(kv[0], len(step_rank))))
            if isinstance(s, dict)
            else s
            for s in steps
        ]
    return ordered
