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
URL's fragment-free query string, which the page moves into the tab's ``sessionStorage``
before its first request.

Flask is imported at module scope, the way :mod:`chemrefine.mcp_server` imports its SDK
and for the same reason: the CLI imports *this* module inside ``except ImportError`` to
turn a missing ``chemrefine[gui]`` into an install hint. An import deferred into
:func:`create_app` makes that guard unreachable, and ``chemrefine gui`` without the
extra would raise a traceback past the handler written for it. Nothing here is imported
by a Flask-free caller — :data:`chemrefine.gui.STATIC_DIR` is where the docs build gets
the asset path.
"""

from __future__ import annotations

import importlib
import inspect
import secrets
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypeVar

import yaml
from flask import Flask, jsonify, request, send_from_directory
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)
from pydantic_core import PydanticCustomError
from werkzeug.exceptions import HTTPException

from chemrefine import agent_tools, introspect, io
from chemrefine.errors import ChemRefineError, ConfigError
from chemrefine.gui import STATIC_DIR
from chemrefine.io import atomic_write, expand_user_path


def _refusal(error: ChemRefineError) -> dict[str, Any]:
    """A ChemRefineError in the GUI's documented shape: its message and its exit code."""
    return {"error": str(error), "exit_code": error.exit_code}


@dataclass(frozen=True)
class _Suspended:
    """A chat run paused on the human's verdicts: its messages, and the cards they answer."""

    messages: list[Any]
    cards: list[dict[str, Any]]


@dataclass
class _Chat:
    """The one conversation the chat panel holds: its messages, and a run paused on verdicts.

    ``history`` is PydanticAI's own message list; ``pending`` the run waiting for the
    human's allow/deny cards, if one is.
    """

    history: list[Any] | None = None
    pending: _Suspended | None = None

    def answer(self, **reply: Any) -> dict[str, Any]:
        """``reply`` with the cards still answerable, which every chat reply carries.

        A refusal, a failure and a run's own answer alike, so the page offers cards again
        only while a suspended run is there for them to answer.
        """
        return {**reply, "pending": self.pending.cards if self.pending is not None else None}

    def ended(
        self,
        messages: list[Any],
        started: Sequence[Any] | None,
        *,
        cards: list[dict[str, Any]] | None = None,
        **reply: Any,
    ) -> dict[str, Any]:
        """Hold where a run ended, and answer with ``reply`` and the config it saved.

        Every run that started ends here — answered, paused on ``cards``, or failed
        part-way. A run paused anew is held for its verdicts; any other is the conversation
        the next message continues (the paused run it continued was taken when it started).
        Every answer carries the config the run saved (:func:`_config_written_by`): an
        approved save and a fresh batch of cards arrive in the same turn, and a run that
        failed may have saved before it did.
        """
        if cards is None:
            self.history = messages
        else:
            self.pending = _Suspended(messages, cards)
        return self.answer(**reply, wrote_config=_config_written_by(messages, started))


def _body() -> dict[str, Any]:
    """The request's JSON object, or the 400 for a body that is not one.

    ``force=True`` parses a bare list, a string or ``null`` as readily as an object, and a
    handler reading fields off one would raise ``AttributeError`` — a logged-traceback 500
    for the client's mistake. Every POST handler starts here, so the shape is decided
    before any field is read.
    """
    payload = request.get_json(force=True)
    if not isinstance(payload, dict):
        raise ConfigError("request body must be a JSON object")
    return payload


def _count(value: Any) -> Any:
    """One whole number off the wire — a JSON int or a decimal string — or the complaint.

    The two frontends spell the same field differently: the form pane ships JSON numbers,
    the free-text boxes and query strings ship text, and both are legitimate. Text is one
    optional leading minus and decimal digits (``--5`` is not a number); anything else —
    ``7a``, ``3.5``, ``null``, a JSON boolean — is refused. A field validator, so the
    refusal is the model's own complaint, worded by :func:`_wire_refusal` like every other.
    """
    # The bool test precedes the int test it would otherwise vanish under: bool is an int
    # subclass, and a JSON `true` meaning 1 is not a contract anyone wrote.
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.removeprefix("-").isdecimal():
        try:
            return int(value)
        except ValueError:
            # Past `sys.get_int_max_str_digits()`, `int` refuses the conversion itself.
            raise PydanticCustomError(
                "count", "{digits} digits is too long for a count", {"digits": len(value)}
            ) from None
    raise PydanticCustomError("count", "{value} is not a whole number", {"value": repr(value)})


def _count_or_unset(value: Any) -> Any:
    """:func:`_count`, with absent, ``null`` and ``""`` all meaning "unset" (``None``)."""
    return None if value is None or value == "" else _count(value)


_Count = Annotated[int, BeforeValidator(_count)]
"""A count off the wire: a JSON int or decimal text."""

_CountOrUnset = Annotated[int | None, BeforeValidator(_count_or_unset)]
"""A count whose callee reads ``None`` as "resolve it yourself" or "none asked for"."""


class _Request(BaseModel):
    """One endpoint's input — a JSON body or a query string — declared field by field.

    Strict, because the wire's types are the contract: ``"false"`` is not ``False`` and
    ``1`` is not a string, and a field declared here cannot be read any other way. Keys
    beyond the model are ignored: the page sends its session's ``config_path`` wherever it
    chats.

    A field the request may leave out defaults to what the tool behind the endpoint
    declares (:func:`_tool_default`), or to ``None``, the wire's "absent", which the
    callee resolves itself.
    """

    model_config = ConfigDict(strict=True, frozen=True)


def _tool_default(tool: Callable[..., Any], name: str) -> Any:
    """``tool``'s own default for its parameter ``name``, read off its signature.

    The MCP server and the chat agent derive each tool's schema from the same signature,
    so an endpoint and an agent asking one tool cannot be given two defaults. A parameter
    the tool requires is refused at import, where the request model is declared.
    """
    default = inspect.signature(tool).parameters[name].default
    if default is inspect.Parameter.empty:
        raise TypeError(f"{tool.__name__}() gives {name!r} no default")
    return default


RequestT = TypeVar("RequestT", bound=_Request)

_WIRE_TYPES: dict[str, str] = {
    "string_type": "string",
    "bool_type": "boolean",
    "dict_type": "JSON object",
}
"""How a refusal names the shape a field must have — the wire's words, not Python's."""


def _wire_refusal(error: ValidationError) -> ConfigError:
    """A request model's first complaint as the 400 that names the field and what it needs.

    ``'approvals.call-1' must be a boolean, not str`` / ``request is missing 'path'`` —
    one sentence a person or an agent can act on, where pydantic's own report is a block
    written for a developer. A complaint with no wire word (a count's) keeps its message.
    """
    first = error.errors()[0]
    field = ".".join(str(part) for part in first["loc"])
    if first["type"] == "missing":
        return ConfigError(f"request is missing {field!r}")
    expected = _WIRE_TYPES.get(first["type"])
    if expected is None:
        return ConfigError(f"{field!r}: {first['msg']}")
    return ConfigError(f"{field!r} must be a {expected}, not {type(first['input']).__name__}")


def _read(model: type[RequestT], payload: dict[str, Any]) -> RequestT:
    """``payload`` as ``model`` declares it, or the :class:`ConfigError` that names the field.

    Takes the payload rather than reading the request, so an endpoint that answers a bad
    field as something other than a 400 (``/api/agent/check``'s finding) can still refuse
    a body that is not an object at all, as every POST here does.
    """
    try:
        return model.model_validate(payload)
    except ValidationError as e:
        raise _wire_refusal(e) from e


def _query(model: type[RequestT]) -> RequestT:
    """The GET's query string as ``model`` declares it — :func:`_read` for the other half."""
    return _read(model, request.args.to_dict())


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

    overwrite: bool = _tool_default(agent_tools.scaffold_templates, "overwrite")


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

    log_tail_lines: _Count = _tool_default(agent_tools.run_status, "log_tail_lines")


class _FailuresRequest(_ConfigRequest):
    """``/api/failures``: the ledger, optionally of one step (a selector the config judges)."""

    step: Any = None


class _ResultsRequest(_FailuresRequest):
    """``/api/results``: one page of ``steps.csv``."""

    limit: _Count = _tool_default(agent_tools.get_results, "limit")
    offset: _Count = _tool_default(agent_tools.get_results, "offset")


class _RunRequest(_ConfigRequest):
    """``/api/run``: the action, its step target, and the two budgets a child's argv carries."""

    action: str = _tool_default(agent_tools.start_run, "action")
    target: Any = None
    max_cores: _CountOrUnset = None
    max_gpus: _CountOrUnset = None


class _ProviderSettings(_Request):
    """The chat panel's settings — every one may be blank, falling back as the CLI's does."""

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class _ChatRequest(_ProviderSettings):
    """``/api/agent/chat``: a reset, the verdicts that resume a paused run, or a message."""

    reset: bool = False
    approvals: dict[str, bool] | None = None
    message: str | None = Field(default=None, validate_default=True)
    config_path: str | None = None

    @field_validator("message")
    @classmethod
    def _says_something(cls, message: str | None, info: ValidationInfo) -> str | None:
        """A turn carries a message unless it resets or answers cards."""
        if message is None and not info.data.get("reset") and not info.data.get("approvals"):
            raise PydanticCustomError("missing", "Field required")
        return message


class _PathQuery(_Request):
    """``GET /api/load``: the config file to open."""

    path: str


class _BrowseQuery(_Request):
    """``GET /api/browse``: the directory to list; absent means the user's home."""

    path: str | None = None


class _TemplateQuery(_ConfigRequest):
    """``GET /api/template``: one step's template, by number or name."""

    step: str


class _StepQuery(_ConfigRequest):
    """``GET /api/structure-list``: one step's structures; no step means the input seeds."""

    step: str | None = None


class _StructureQuery(_StepQuery):
    """``GET /api/structure``: one structure, optionally with a normal mode's displacements."""

    structure_id: str | None = None
    mode_index: _CountOrUnset = None


def create_app(*, token: str, config_path: Path | None = None) -> Flask:
    """Build the GUI app.

    ``token`` is the per-session secret every ``/api/*`` call must present — required and
    non-empty, so no app is built with the gate open. ``config_path`` preloads an existing
    config into the builder.
    """
    if not token:
        raise ValueError("the GUI needs a session token; an empty one opens every endpoint")
    app = Flask("chemrefine-gui", static_folder=str(STATIC_DIR), static_url_path="/static")
    app.config["PROPAGATE_EXCEPTIONS"] = True

    def _authorized() -> bool:
        """Whether this request carries the session token."""
        presented = request.headers.get("X-ChemRefine-Token", "")
        # Compared as bytes: Werkzeug decodes headers as latin-1, and `compare_digest`
        # raises TypeError on a str holding a non-ASCII character — so a header with any
        # byte above 0x7F would turn a plain 401 into a 500 and a logged traceback.
        # Encoding both sides answers every input in constant time, whatever its bytes.
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
        through as themselves — re-raising them would turn every stray request into a
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

        The other half of ``/api/save``. ``/api/bootstrap`` fires once, for the launch
        path, so this is how the editor adopts a config the agent saves while the page is
        open.

        A missing path, a directory, an unreadable file and a binary one picked by mistake
        are all :func:`chemrefine.io.read_utf8`'s :class:`ConfigError`, which :func:`surface`
        answers with the plain 400 every other bad input here gets. So is a ``~name`` the
        host cannot resolve: :func:`chemrefine.io.expand_user_path` refuses it,
        and every path a user types here goes through it.
        """
        path = expand_user_path(_query(_PathQuery).path)
        return jsonify(_config_payload(path))

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
        requested = _query(_BrowseQuery).path or str(Path.home())
        path = expand_user_path(requested).resolve()
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
        destination = expand_user_path(body.path)
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
        query = _query(_TemplateQuery)
        return jsonify(agent_tools.read_template(query.config_path, query.step))

    @app.post("/api/template")
    def write_template() -> Any:
        """Save the inline editor's template text."""
        body = _read(_TemplateWriteRequest, _body())
        return jsonify(agent_tools.write_template(body.config_path, body.step, body.text))

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
        query = _query(_StructureQuery)
        return jsonify(
            agent_tools.get_structure(
                query.config_path,
                query.step or None,
                structure_id=query.structure_id,
                mode_index=query.mode_index,
            )
        )

    @app.get("/api/structure-list")
    def structure_list() -> Any:
        """What the Structure pane's two combo boxes offer: the ids, and each one's modes.

        Same query-argument discipline and same ``step``/seeds sentinel as ``/api/structure``
        above. Offering only what exists is what stops the pane asking for a mode number the
        step never computed — the request that could only ever fail.
        """
        query = _query(_StepQuery)
        return jsonify(agent_tools.list_structures(query.config_path, query.step or None))

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

    # One user, one browser, one conversation: the chat state lives on the app instance.
    chat = _Chat()

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

        A turn is a read-modify-write of ``chat`` around the model call, so
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
        message, not a traceback: at the network boundary the failure is the answer. A tool
        that crashes is this server's failure, a 500 named for the tool, with its traceback
        in the log rather than the reply.
        """
        from pydantic_ai import DeferredToolRequests, DeferredToolResults

        from chemrefine.agent.harness import (
            TurnFailed,
            answerable,
            build_web_agent,
            endpoint_failure_text,
            run_turn,
        )
        from chemrefine.agent.providers import ProviderConfig

        try:
            # Everything decidable before a model is built: a reset, and every refusal —
            # answered with what is still pending, since nothing here takes the paused run.
            body = _read(_ChatRequest, _body())
            if body.reset:
                chat.history = None
                chat.pending = None
                return jsonify(chat.answer(reply=None, reset=True))
            # The conversation this turn continues, which is the suspended run's whenever
            # one is waiting: its verdicts resume it, and a message sent past it (from a
            # second tab, a reloaded page) closes its calls as never run — continuing from
            # the history before it dropped the message that raised them. A config save is
            # this turn's when that history had not seen it.
            started = chat.history
            verdicts: DeferredToolResults | None = None
            if body.approvals:
                if chat.pending is None:
                    raise ConfigError("no suspended run to resume")
                # Exactly the calls waiting, checked here: the run would refuse any other set
                # only after the cards were taken, leaving nothing for them to resume.
                waiting = [card["id"] for card in chat.pending.cards]
                if set(body.approvals) != set(waiting):
                    raise ConfigError(f"the verdicts must answer exactly the calls {waiting}")
                started = chat.pending.messages
                verdicts = DeferredToolResults(approvals=dict(body.approvals))
            elif chat.pending is not None:
                started = answerable(chat.pending.messages)
            resolved = ProviderConfig.resolve(
                body.provider or "custom",
                model=body.model,
                base_url=body.base_url,
                api_key=body.api_key,
            )
        except ChemRefineError as e:
            return jsonify(chat.answer(**_refusal(e))), 400
        try:
            # Inside the try, not above it: `Agent("gpt-5-mini")` raises `UserError` at
            # construction for a name PydanticAI cannot place, and that is neither an
            # HTTPException nor a ChemRefineError — so `surface` would re-raise it as a 500
            # with a traceback, for a typo in the model box. It belongs with the other
            # outside-world failures, as the panel's 502.
            # The GUI's *current* file wins over the launch argument: the user may have
            # saved somewhere else since, and telling the agent about a path the editor
            # left behind would send "work on the file I just saved" to the wrong one.
            # `or`, not `is None`, so an empty string from a page with nothing saved yet
            # falls back rather than becoming a config_path of "".
            agent = build_web_agent(
                resolved.build_model(),
                config_path=body.config_path
                or (str(config_path) if config_path is not None else None),
            )
            # Taken, not read, once a run starts: verdicts run the approved calls before the
            # model is asked anything, and a message has closed the calls it went past —
            # either way a failure leaves no run for the cards to answer, and answering them
            # again would run those calls twice.
            chat.pending = None
            if verdicts is not None:
                result = run_turn(agent, message_history=started, deferred_tool_results=verdicts)
            else:
                result = run_turn(agent, body.message, message_history=started)
        except ChemRefineError as e:
            # The harness refusing its own settings while the agent is built (an unusable
            # CHEMREFINE_LLM_TIMEOUT), in the documented shape — with what is still pending,
            # so the page offers only cards that can be answered. A tool's refusal never
            # lands here: it is the tool's answer to the model.
            return jsonify(chat.answer(**_refusal(e))), 400
        except TurnFailed as e:
            # The run failed part-way, and what it did first is the conversation now: the
            # message, and every approved call that ran with its result. A tool's crash is
            # this server's own failure (logged with its traceback by the harness), the
            # endpoint's the outside world's.
            return jsonify(chat.ended(e.messages, started, error=str(e))), (
                500 if e.crashed else 502
            )
        except Exception as e:  # the model endpoint is the outside world
            return jsonify(chat.answer(error=endpoint_failure_text(e))), 502
        if isinstance(result.output, DeferredToolRequests):
            cards = [
                {"id": call.tool_call_id, "tool": call.tool_name, "args": call.args}
                for call in result.output.approvals
            ]
            return jsonify(chat.ended(result.all_messages(), started, cards=cards, reply=None))
        return jsonify(chat.ended(result.all_messages(), started, reply=result.output))

    @app.post("/api/run")
    def run() -> Any:
        """Launch a detached run — same semantics as the agent's start_run.

        A held lock surfaces through the error handler as the documented
        ``{error, exit_code: 10}`` shape; the browser confirmed the action already,
        and the child outlives this server exactly as it outlives an agent session.

        The two budgets are wire numbers (:data:`_CountOrUnset`) like the rest of them, and
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
    """One config file as the page wants it — the shape ``bootstrap`` sends."""
    return {"path": str(path), "yaml_text": io.read_utf8(path, what="config")}


def _config_written_by(messages: Sequence[Any], started: Sequence[Any] | None) -> str | None:
    """The path an agent turn saved a config to, or ``None`` if it saved none.

    Read off the turn's own messages rather than guessed at from the reply text:
    ``messages`` is the conversation the turn ended with — returned, or carried by its
    failure — and ``started`` the one it continued. Five conditions, and each excludes a
    real case: the part must be a tool *return* (a request is not a write), one this turn
    added (an earlier turn's save is not this one's), from ``save_config`` (the other
    mutating tools touch templates, seeds and runs, never the YAML), with ``outcome``
    success (a denied call still leaves a part behind), and carrying ``written: True`` —
    because ``save_config`` reports ``written: False`` *without raising* when validation
    refuses the draft, which is the case that would otherwise have the editor adopt a file
    that was never saved.

    The turn's own returns are the ones after as many as ``started`` holds: a run keeps the
    history it takes in, in order, ahead of what it adds. Counted by return, not located by
    message or by call — the run merges adjacent messages as it takes a history in, so a
    message's position moves, and a provider may reuse a call id in a later response, so
    an id seen before can still be this turn's save.
    """
    from pydantic_ai.messages import ToolReturnPart

    def returns(history: Sequence[Any]) -> list[Any]:
        return [p for message in history for p in message.parts if isinstance(p, ToolReturnPart)]

    for part in returns(messages)[len(returns(started or ())) :]:
        if (
            part.tool_name == "save_config"
            and part.outcome == "success"
            and isinstance(part.content, dict)
            and part.content.get("written") is True
        ):
            return str(part.content.get("path"))
    return None


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
