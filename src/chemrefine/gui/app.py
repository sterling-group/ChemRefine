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
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypeVar

import yaml
from flask import Flask, jsonify, request, send_from_directory
from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError, ValidationInfo
from werkzeug.exceptions import HTTPException

from chemrefine import agent_tools, introspect, io
from chemrefine.cache import atomic_write
from chemrefine.errors import ChemRefineError, ConfigError
from chemrefine.gui import STATIC_DIR


def _refusal(error: ChemRefineError) -> dict[str, Any]:
    """A ChemRefineError in the GUI's documented shape: its message and its exit code."""
    return {"error": str(error), "exit_code": error.exit_code}


@dataclass(frozen=True)
class _Suspended:
    """A chat run paused on the human's verdicts: its messages, and the cards they answer."""

    messages: list[Any]
    cards: list[dict[str, Any]]


def _wire_int(value: Any, key: str) -> int:
    """One whole number off the wire — a JSON int or a decimal string — or the 400.

    The two frontends spell the same field differently: the form pane ships JSON
    numbers, the free-text boxes ship strings, and both are legitimate — a guard that
    took only one of them would break the other pane. A string gets one optional
    leading minus and then ``isdecimal`` (``--5`` is not a number, and the
    ``lstrip("-")`` trimming that admitted it is how a comment promising 400 sat on a
    line that raised 500). Everything else — ``7a``, ``3.5``, ``null``, a JSON
    boolean — raises :class:`ConfigError`, which :func:`surface` turns into the
    documented ``{error, exit_code}`` 400 like every other unusable input to this app.
    Absence is the caller's business, decided before the call: a request model's default
    (:data:`_WireInt`, :data:`_WireIntOrUnset`), or the ``not in (None, "")`` sentinel a
    GET uses for a field whose callee wants ``None``.
    """
    # The bool test precedes the int test it would otherwise vanish under: bool is an
    # int subclass, and a JSON `true` silently meaning 1 is not a contract anyone wrote.
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value)
    digits = text.removeprefix("-")
    if isinstance(value, str) and digits and digits.isdecimal():
        try:
            return int(text)
        except ValueError:
            # Past `sys.get_int_max_str_digits()` `int` refuses the conversion itself —
            # an escaped ValueError, and so a 500, for a count no request needs.
            raise ConfigError(f"{key} has {len(digits)} digits; no count is that long") from None
    raise ConfigError(f"{key} {value!r} is not a whole number")


def _body() -> dict[str, Any]:
    """The request's JSON object, or the 400 for a body that is not one.

    ``force=True`` parses a bare list, a string or ``null`` as readily as an object, and a
    handler that then reads ``.get`` off it raises ``AttributeError`` out of the handler —
    a logged-traceback 500 for what is the client's mistake. The shape check used to live
    inside one field reader, which closed it only for the endpoints whose *first* read went
    through that helper; the four whose first read was a ``.get`` (``/api/run``,
    ``/api/agent/check``, ``/api/agent/chat``, ``/api/structure-file``) crashed on ``[]``.
    The one place every POST handler starts is where the shape is decided, before any read.
    """
    payload = request.get_json(force=True)
    if not isinstance(payload, dict):
        raise ConfigError("request body must be a JSON object")
    return payload


def _whole(value: Any, info: ValidationInfo) -> Any:
    """:func:`_wire_int` as a request field's own reading — the field names itself."""
    return _wire_int(value, str(info.field_name))


def _whole_or_unset(value: Any, info: ValidationInfo) -> Any:
    """:func:`_whole`, with absent, ``null`` and ``""`` all meaning "unset" (``None``)."""
    return None if value is None or value == "" else _wire_int(value, str(info.field_name))


_WireInt = Annotated[int, BeforeValidator(_whole)]
"""A count off the wire: a JSON int or a decimal string, refused by :func:`_wire_int`."""

_WireIntOrUnset = Annotated[int | None, BeforeValidator(_whole_or_unset)]
"""A count whose callee reads ``None`` as "resolve it yourself" (``/api/run``'s budgets)."""


class _Request(BaseModel):
    """One endpoint's JSON body, declared: each field's name, JSON type and default.

    Strict, because the wire's types are the contract: a lax model reads the string
    ``"false"`` as ``False`` and ``1`` as a string, which is the guessing this layer exists
    to refuse. ``/api/agent/chat`` once read ``reset`` by truthiness while ``/api/scaffold``
    typed ``overwrite`` — each field was read by a hand-written helper call, and the one
    that was not typed was the one that wiped a conversation on ``{"reset": "false"}``.
    Declared per endpoint, a field cannot be read untyped. Keys a body carries beyond its
    model are ignored: the page sends its session's ``config_path`` wherever it chats.
    """

    model_config = ConfigDict(strict=True, frozen=True)


RequestT = TypeVar("RequestT", bound=_Request)

_WIRE_TYPES: dict[str, str] = {
    "string_type": "string",
    "bool_type": "boolean",
    "dict_type": "JSON object",
}
"""How a refusal names the shape a field must have — the wire's words, not Python's."""


def _wire_refusal(error: ValidationError) -> ConfigError:
    """A request model's first complaint as the 400 that names the field and what it needs.

    ``'approvals.call-1' must be a boolean, not str`` / ``request body is missing 'path'``
    — one sentence a person or an agent can act on, where pydantic's own report is a block
    written for a developer. A complaint with no wire word falls back to pydantic's message.
    """
    first = error.errors()[0]
    field = ".".join(str(part) for part in first["loc"])
    if first["type"] == "missing":
        return ConfigError(f"request body is missing {field!r}")
    expected = _WIRE_TYPES.get(first["type"])
    if expected is None:
        return ConfigError(f"{field!r}: {first['msg']}")
    return ConfigError(f"{field!r} must be a {expected}, not {type(first['input']).__name__}")


def _read(model: type[RequestT], payload: dict[str, Any]) -> RequestT:
    """``payload`` as ``model`` declares it, or the :class:`ConfigError` that names the field.

    Takes the payload rather than reading the body, so an endpoint that answers a bad
    field as something other than a 400 (``/api/agent/check``'s finding) can still refuse
    a body that is not an object at all, as every POST here does.
    """
    try:
        return model.model_validate(payload)
    except ValidationError as e:
        raise _wire_refusal(e) from e


class _ValidateRequest(_Request):
    """``/api/validate``: the editor's text, and the directory its relative paths mean."""

    yaml_text: str
    base_dir: str | None = None


class _YamlRequest(_Request):
    """``/api/yaml``: the form's raw mapping — its shape is the emitter's to judge."""

    config: Any


class _ParseRequest(_Request):
    """``/api/parse``: YAML text, for the form."""

    yaml_text: str


class _SaveRequest(_Request):
    """``/api/save``: where the editor's text goes, and the text."""

    path: str
    yaml_text: str


class _ConfigRequest(_Request):
    """A request about one saved config — ``/api/summary``, and the base of the rest."""

    config_path: str


class _ScaffoldRequest(_ConfigRequest):
    """``/api/scaffold``: ``overwrite`` replaces every existing template with a starter."""

    overwrite: bool = False


class _TemplateWriteRequest(_ConfigRequest):
    """``POST /api/template``: one step's template text; the step is the config's to find."""

    step: Any
    text: str


class _StructureFileRequest(_Request):
    """``/api/structure-file``: a ``path`` on this machine, or a dropped file's name and text."""

    path: str | None = None
    name: str | None = None
    text: str | None = None


class _StatusRequest(_ConfigRequest):
    """``/api/status``: how much of the driver log to tail."""

    log_tail_lines: _WireInt = 40


class _FailuresRequest(_ConfigRequest):
    """``/api/failures``: the ledger, optionally of one step (a selector the config judges)."""

    step: Any = None


class _ResultsRequest(_FailuresRequest):
    """``/api/results``: one page of ``steps.csv``."""

    limit: _WireInt = 20
    offset: _WireInt = 0


class _RunRequest(_ConfigRequest):
    """``/api/run``: the action, its step target, and the two budgets a child's argv carries."""

    action: str = "run"
    target: Any = None
    max_cores: _WireIntOrUnset = None
    max_gpus: _WireIntOrUnset = None


class _ProviderSettings(_Request):
    """The chat panel's settings — every one may be blank, falling back as the CLI's does."""

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class _ChatRequest(_ProviderSettings):
    """``/api/agent/chat``: a reset, a message, or the verdicts that resume a paused run."""

    reset: bool = False
    message: str | None = None
    approvals: dict[str, bool] | None = None
    config_path: str | None = None


def _arg(key: str) -> str:
    """One query argument a GET must carry, or the 400 that names it.

    :func:`_read`'s rule for the other half of the API. A query argument is always a
    string, so presence is the whole contract. Read bare, an absent one is Werkzeug's
    ``BadRequestKeyError`` — an :class:`HTTPException` that :func:`surface` passes through
    as an HTML page, which the page's fetch cannot read for a reason.
    """
    if key not in request.args:
        raise ConfigError(f"request is missing the {key!r} query argument")
    return request.args[key]


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
            return jsonify(_refusal(error)), 400
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

        initial = (
            _config_payload(config_path)
            if config_path is not None and config_path.is_file()
            else None
        )
        return jsonify(
            {
                "schema": introspect.schema_document(),
                "initial": initial,
                "host": socket.gethostname(),
            }
        )

    @app.get("/api/load")
    def load() -> Any:
        """Read a config back off disk, so the builder can adopt what something else wrote.

        The missing half of ``/api/save``. ``/api/bootstrap`` was the only reader and it
        fires once, for the launch path — so when the agent saved a config, the editor
        beside it went on showing the previous text and the user had to reload the page to
        see what had happened.

        A directory or an unreadable file is a plain 400 like every other bad input here.
        A binary file picked by mistake is the same 400: :func:`chemrefine.io.read_utf8`
        refuses it as a :class:`ConfigError` (``UnicodeDecodeError`` is a ``ValueError``,
        not an ``OSError``, and would otherwise reach :func:`surface` as a 500 with a
        traceback). A ``~name`` the host cannot resolve is
        :func:`chemrefine.agent_tools.expand_user_path`'s refusal, which :func:`surface`
        answers with the same 400 — every path a user types here goes through it.
        """
        path = agent_tools.expand_user_path(_arg("path"))
        if not path.is_file():
            return jsonify({"error": f"not a file: {path}"}), 400
        try:
            return jsonify(_config_payload(path))
        except (OSError, ConfigError) as e:
            return jsonify({"error": f"cannot read {path}: {e}"}), 400

    @app.post("/api/validate")
    def validate() -> Any:
        """The structured validation report for the current editor text."""
        body = _read(_ValidateRequest, _body())
        return jsonify(agent_tools.validate_config(body.yaml_text, body.base_dir))

    @app.post("/api/yaml")
    def to_yaml() -> Any:
        """Form state (a raw config mapping) → YAML text. The one emitter.

        Keys are emitted in the schema's declaration order — workflow settings first,
        ``steps`` last, and each step's keys in :class:`StepConfig` order — so the file
        reads like every shipped example regardless of the order the user clicked
        things together in.
        """
        body = _read(_YamlRequest, _body())
        text = yaml.safe_dump(_canonical_order(body.config), sort_keys=False, allow_unicode=True)
        return jsonify({"yaml_text": text})

    @app.post("/api/parse")
    def parse() -> Any:
        """YAML text → the raw mapping the form edits (or the reason it cannot)."""
        body = _read(_ParseRequest, _body())
        try:
            raw = yaml.safe_load(body.yaml_text)
        except yaml.YAMLError as e:
            return jsonify({"error": f"malformed YAML: {e}"}), 400
        if not isinstance(raw, dict):
            return jsonify({"error": "config is not a YAML mapping"}), 400
        return jsonify({"config": raw})

    @app.get("/api/browse")
    def browse() -> Any:
        """One directory level for the path pickers (local machine, local user)."""
        requested = request.args.get("path") or str(Path.home())
        path = agent_tools.expand_user_path(requested).resolve()
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
        body = _read(_SaveRequest, _body())
        destination = agent_tools.expand_user_path(body.path)
        try:
            atomic_write(destination, body.yaml_text.encode("utf-8"))
        except OSError as e:
            return jsonify({"error": f"cannot write {destination}: {e}"}), 400
        return jsonify({"path": str(destination)})

    @app.post("/api/scaffold")
    def scaffold() -> Any:
        """Fill the saved config's template gaps with starters (kept/written report).

        ``overwrite`` is typed like every other field — a JSON ``true``/``false``, never
        Python truthiness. Coerced with ``bool()``, the string ``"false"`` is true, and true
        here replaces every existing template with a starter: the one request on this app
        that can destroy a user's edits must refuse a value it cannot read rather than guess.
        """
        body = _read(_ScaffoldRequest, _body())
        return jsonify(agent_tools.scaffold_templates(body.config_path, overwrite=body.overwrite))

    @app.get("/api/template")
    def read_template() -> Any:
        """One step's template text for the inline editor."""
        return jsonify(agent_tools.read_template(_arg("config_path"), _step_key(_arg("step"))))

    @app.post("/api/template")
    def write_template() -> Any:
        """Save the inline editor's template text."""
        body = _read(_TemplateWriteRequest, _body())
        return jsonify(
            agent_tools.write_template(body.config_path, _step_key(body.step), body.text)
        )

    @app.get("/api/structure")
    def structure() -> Any:
        """One structure as extended-XYZ, for the Structure pane.

        Query arguments, not a path segment: ``test_every_route_is_behind_the_gate`` walks
        the url_map and refuses a parameterized rule, because one cannot be probed for the
        token gate by enumeration.

        No ``step`` means the input seeds — the view that works before anything has run.
        ``mode_index`` asks for the structure *with* a normal mode's displacement columns,
        which is what the viewer animates.
        """
        step = request.args.get("step")
        mode = request.args.get("mode_index")
        return jsonify(
            agent_tools.get_structure(
                _arg("config_path"),
                _step_key(step) if step not in (None, "") else None,
                structure_id=request.args.get("structure_id"),
                mode_index=_wire_int(mode, "mode_index") if mode not in (None, "") else None,
            )
        )

    @app.get("/api/structure-list")
    def structure_list() -> Any:
        """What the Structure pane's two combo boxes offer: the ids, and each one's modes.

        Same query-argument discipline and same ``step``/seeds sentinel as ``/api/structure``
        above. Offering only what exists is what stops the pane asking for a mode number the
        step never computed — the request that could only ever fail.
        """
        step = request.args.get("step")
        return jsonify(
            agent_tools.list_structures(
                _arg("config_path"),
                _step_key(step) if step not in (None, "") else None,
            )
        )

    @app.post("/api/structure-file")
    def structure_file() -> Any:
        """One structure file, drawn on its own — no workflow, no step, no run tree.

        Two ways in, because a file reaches this page two ways. ``path`` is a file on the
        machine the server runs on, chosen through the browser. ``name`` + ``text`` is a
        file *dropped* on the page, which over a forwarded port lives on the other machine
        entirely and has no path here — browsers hand over contents and a basename, never
        a path, which is also why this is a POST.

        Deliberately not ``/api/load``: that reads a workflow and brings its whole tree,
        and conflating "show me this molecule" with "open this project" is how one of them
        ends up doing the other by accident.
        """
        body = _read(_StructureFileRequest, _body())
        if body.path:
            return jsonify(agent_tools.read_structure_file(body.path))
        if body.text is None:
            raise ConfigError("send either a path, or a name and text")
        return jsonify(agent_tools.read_structure_text(body.name or "structure", body.text))

    @app.post("/api/summary")
    def summary() -> Any:
        """The dry-run-style execution summary for a saved config."""
        body = _read(_ConfigRequest, _body())
        return jsonify(agent_tools.summarize_config(body.config_path))

    @app.post("/api/status")
    def status() -> Any:
        """Where the tree stands — lock holder, per-step progress, log tail."""
        body = _read(_StatusRequest, _body())
        return jsonify(agent_tools.run_status(body.config_path, log_tail_lines=body.log_tail_lines))

    @app.post("/api/results")
    def results() -> Any:
        """A paginated slice of steps.csv for the dashboard's results table."""
        body = _read(_ResultsRequest, _body())
        return jsonify(
            agent_tools.get_results(
                body.config_path, step=body.step, limit=body.limit, offset=body.offset
            )
        )

    @app.post("/api/failures")
    def failures() -> Any:
        """The failure ledger plus the suggested recovery action."""
        body = _read(_FailuresRequest, _body())
        return jsonify(agent_tools.get_failures(body.config_path, step=body.step))

    # One user, one browser, one conversation: the chat state lives on the app instance
    # (history = PydanticAI's own message list; pending = the suspended run awaiting the
    # human's allow/deny verdicts, a `_Suspended`).
    chat_state: dict[str, Any] = {"history": None, "pending": None}

    def _still_pending() -> list[dict[str, Any]] | None:
        """The cards a failed turn leaves answerable — what the page may offer again."""
        suspended = chat_state["pending"]
        return suspended.cards if suspended is not None else None

    # One conversation, but four waitress threads — and a turn is a read-modify-write
    # around a model call that takes seconds to a minute. Two overlapping POSTs would
    # both run from the same history and the last writer would win: a turn's messages
    # silently gone, or `pending` clobbered mid-approval. Non-blocking, answering 409,
    # rather than queueing: a Send during a wedged turn should hear "busy" now, not hang
    # behind it. `reset` takes the same lock — a reset landing mid-turn would otherwise
    # be resurrected by the turn it interrupted.
    chat_lock = threading.Lock()

    def _preset_shapes() -> dict[str, dict[str, Any]]:
        """The provider field-shapes, from the module that owns them.

        Served rather than duplicated in JavaScript: a second copy of the URLs in the
        frontend would quietly beat ``CHEMREFINE_LLM_BASE_URL``, since an explicit URL
        wins at resolution. The page shows these as placeholders and never sends them back.
        """
        from chemrefine.agent import providers

        return providers.preset_shapes()

    @app.get("/api/agent/availability")
    def agent_availability() -> Any:
        """Whether the chat panel can work here: extra installed, model configured.

        Carries the preset shapes either way — the panel renders its fields from them
        before any check has run, and they are true whether or not the extra is installed.
        """
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
                    "presets": _preset_shapes(),
                }
            )
        from chemrefine.agent.providers import ProviderConfig

        try:
            resolved = ProviderConfig.resolve()
            detail, configured = resolved.model, True
        except ChemRefineError as e:
            detail, configured = str(e), False
        return jsonify(
            {
                "installed": True,
                "configured": configured,
                "detail": detail,
                "presets": _preset_shapes(),
            }
        )

    @app.post("/api/agent/check")
    def agent_check() -> Any:
        """Probe the panel's settings before a turn is spent on them.

        The same preflight ``chemrefine agent --check`` runs
        (:func:`chemrefine.agent.providers.check`), reached from the browser: is the
        endpoint there, and does it serve the model that was named. Answers 200 with a
        verdict either way, like :func:`agent_availability` beside it — an unreachable
        Ollama is this endpoint's *answer*, not its failure, so a ``ChemRefineError`` from
        resolution is caught here rather than becoming :func:`surface`'s 400.

        ``providers`` is imported directly and imports no SDK at runtime, so the check
        works before ``chemrefine[agent]`` is installed — the same property that lets the
        CLI preflight run on a bare machine.
        """
        from chemrefine.agent import providers

        payload = _body()
        try:
            # The settings are read inside the try for the reason the resolve is: a
            # provider, model, URL or key of the wrong JSON type is a verdict about the
            # panel's settings, answered as a finding like every other unusable value
            # here. A key that cannot be a header is `resolve`'s refusal too — judged on
            # the resolved key, so a blank box means what it means at every entry point.
            settings = _read(_ProviderSettings, payload)
            resolved = providers.ProviderConfig.resolve(
                settings.provider or "custom",
                model=settings.model,
                base_url=settings.base_url,
                api_key=settings.api_key,
            )
        except ChemRefineError as e:
            return jsonify({"ok": False, "findings": [str(e)]})
        report = providers.check(resolved)
        return jsonify({"ok": report.ok, "findings": list(report.findings)})

    @app.post("/api/agent/chat")
    def agent_chat() -> Any:
        """One agent turn at a time: contend on the lock, then run it.

        A turn is a read-modify-write of ``chat_state`` around the model call, so
        overlapping requests must not interleave — see ``chat_lock`` above for what a
        lost race costs. The contended answer is a 409 in the same ``{error: …}`` shape
        as every other refusal here, so the panel flashes it like any other.
        """
        if not chat_lock.acquire(blocking=False):
            return jsonify({"error": "a chat turn is already running; wait for it to finish"}), 409
        try:
            return _chat_turn()
        finally:
            chat_lock.release()

    def _chat_turn() -> Any:
        """The turn itself — a message, or the verdicts that resume a suspended run.

        Runs under ``agent_chat``'s lock, always. Non-streaming by design (a local-model
        turn takes seconds to a minute inside this worker thread); the reply is either
        text or a list of approval requests the frontend renders as allow/deny cards.
        Provider errors — an unreachable Ollama, a bad key — come back as a 502 with the
        message, not a traceback: at the network boundary the failure is the answer.
        """
        from pydantic_ai import DeferredToolRequests, DeferredToolResults

        from chemrefine.agent.harness import build_web_agent, run_turn
        from chemrefine.agent.providers import ProviderConfig

        try:
            # Everything decidable before a model is asked, refused with what is still
            # pending: nothing here touches the suspended run, so its cards still answer it.
            body = _read(_ChatRequest, _body())
            if body.reset:
                chat_state["history"] = None
                chat_state["pending"] = None
                return jsonify({"reply": None, "pending": None, "reset": True})
            if not body.approvals and body.message is None:
                raise ConfigError("request body is missing 'message'")
            resolved = ProviderConfig.resolve(
                body.provider or "custom",
                model=body.model,
                base_url=body.base_url,
                api_key=body.api_key,
            )
        except ChemRefineError as e:
            return jsonify({**_refusal(e), "pending": _still_pending()}), 400
        try:
            # Inside the try, not above it: `Agent("gpt-5-mini")` raises `UserError` at
            # construction for a name PydanticAI cannot place, and that is neither an
            # HTTPException nor a ChemRefineError — so `surface` re-raised it as a 500 with
            # a traceback, for a typo in the model box. It belongs with the other
            # outside-world failures, as the panel's 502.
            # The GUI's *current* file wins over the launch argument: the user may have
            # saved somewhere else since, and telling the agent about a path the editor
            # left behind is how "work on the file I just saved" reached the wrong one.
            # `or`, not `is None`, so an empty string from a page with nothing saved yet
            # falls back rather than becoming a config_path of "".
            agent = build_web_agent(
                resolved.build_model(),
                config_path=body.config_path
                or (str(config_path) if config_path is not None else None),
            )
            if body.approvals:
                suspended = chat_state["pending"]
                if suspended is None:
                    return jsonify({"error": "no suspended run to resume", "pending": None}), 400
                verdicts = DeferredToolResults(approvals=dict(body.approvals))
                # Taken, not read: the approved calls run before the model is asked anything,
                # so once the resume starts, a failure — the endpoint dying on the next
                # request, a refusal the model cannot get past — leaves no run for these
                # cards to answer, and answering them again would run those calls twice.
                chat_state["pending"] = None
                result = run_turn(
                    agent, message_history=suspended.messages, deferred_tool_results=verdicts
                )
            else:
                result = run_turn(agent, body.message, message_history=chat_state["history"])
        except ChemRefineError as e:
            # The documented {error, exit_code} shape, for a refusal the model could not
            # get past (run_turn ends the turn on it) and for a request this panel refuses —
            # with what is still pending, so the page offers only cards that can be answered.
            return jsonify({**_refusal(e), "pending": _still_pending()}), 400
        except Exception as e:  # the model endpoint is the outside world
            return jsonify(
                {"error": f"model endpoint failed: {e}", "pending": _still_pending()}
            ), 502
        # Computed once, before the branch, and reported on both: an approved write and a
        # fresh batch of approval cards arrive in the *same* turn, so a signal attached
        # only to the text branch would be silently dropped exactly when the agent is
        # working steadily.
        wrote_config = _config_written_by(result)
        if isinstance(result.output, DeferredToolRequests):
            cards = [
                {"id": call.tool_call_id, "tool": call.tool_name, "args": call.args}
                for call in result.output.approvals
            ]
            chat_state["pending"] = _Suspended(result.all_messages(), cards)
            return jsonify({"reply": None, "wrote_config": wrote_config, "pending": cards})
        chat_state["history"] = result.all_messages()
        chat_state["pending"] = None
        return jsonify({"reply": result.output, "pending": None, "wrote_config": wrote_config})

    @app.post("/api/run")
    def run() -> Any:
        """Launch a detached run — same semantics as the agent's start_run.

        A held lock surfaces through the error handler as the documented
        ``{error, exit_code: 10}`` shape; the browser confirmed the action already,
        and the child outlives this server exactly as it outlives an agent session.

        The two budgets are wire numbers (:data:`_WireIntOrUnset`) like the rest of them, and
        the two that reach a *child process*: ``start_run`` renders them onto its argv,
        where one Typer cannot read is exit 2 into a log nobody is watching — after this
        call has already returned a pid and a log path. Unset stays unset, because the
        callee reads ``None`` as "resolve it yourself".
        """
        body = _read(_RunRequest, _body())
        return jsonify(
            agent_tools.start_run(
                body.config_path,
                action=body.action,
                target=body.target,
                max_cores=body.max_cores,
                max_gpus=body.max_gpus,
            )
        )

    return app


def _config_payload(path: Path) -> dict[str, str]:
    """One config file as the page wants it — the shape ``bootstrap`` has always sent."""
    return {"path": str(path), "yaml_text": io.read_utf8(path, what="config")}


def _config_written_by(result: Any) -> str | None:
    """The path an agent turn saved a config to, or ``None`` if it saved none.

    Read off the turn's own messages rather than guessed at from the reply text. Four
    conditions, and each excludes a real case: the part must be a tool *return* (a request
    is not a write), from ``save_config`` (the other mutating tools touch templates, seeds
    and runs, never the YAML), with ``outcome`` success (a denied call still leaves a part
    behind), and carrying ``written: True`` — because ``save_config`` reports
    ``written: False`` *without raising* when validation refuses the draft, which is the
    case that would otherwise have the editor adopt a file that was never saved.
    """
    from pydantic_ai.messages import ToolReturnPart

    for message in result.new_messages():
        for part in getattr(message, "parts", []):
            if (
                isinstance(part, ToolReturnPart)
                and part.tool_name == "save_config"
                and part.outcome == "success"
                and isinstance(part.content, dict)
                and part.content.get("written") is True
            ):
                return str(part.content.get("path"))
    return None


def _step_key(value: str | int) -> int | str:
    """A step selector from the wire: digits mean the step number, anything else a name.

    ``isdecimal``, not ``isdigit``, and for the reason ``StepConfig.matches`` uses it:
    ``isdigit`` accepts the Unicode ``No`` category that ``int`` rejects, so ``step=²``
    raised a bare ``ValueError`` here — a 500 with a traceback out of the handler, where
    every other bad selector is the documented ``{error, exit_code}`` 400.
    """
    text = str(value)
    return int(text) if text.isdecimal() else text


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
