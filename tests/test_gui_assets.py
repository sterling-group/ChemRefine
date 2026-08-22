"""The GUI's static assets: the one layer no other gate can see.

Python endpoints are held to 100% branch coverage; the JavaScript and the Alpine
expressions in ``index.html`` are executed only by a browser, so every defect there has
so far been found by a human clicking. These tests close the two gaps that cost the
most: a **syntax error** (one stray brace blanks the whole page — every button, both
panes) and a **broken binding** (an ``@click`` naming a method that no longer exists
fails silently at click time, which is exactly how a reset button comes to "do
nothing").

Neither needs a browser. The syntax check shells out to ``node --check``, and three cases
execute the pure form logic through ``node -e``; the binding check is pure text analysis:
every handler an Alpine attribute calls must exist in the component, and every state root
it reads must be declared on it.

A Node is a hard test dependency (``nodejs-wheel-binaries`` in ``[test]``) rather than an
optional nicety, because these seven cases are the only ones that ever *execute* this
JavaScript and they used to skip: pytest printed "7 skipped" and exited 0, so a machine
without Node ran every gate green while checking the frontend not at all.
:func:`_node_or_skip` is what makes that impossible to ship — see its docstring.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from chemrefine import io

STATIC = Path(__file__).resolve().parent.parent / "src" / "chemrefine" / "gui" / "static"
INDEX = STATIC / "index.html"
OURS = ("app.js", "forms.js")

# Alpine attributes whose value is a JavaScript expression.
_ALPINE_ATTR = re.compile(
    r"""(?:@[\w.]+|x-(?:text|show|if|for|model|init|effect)|:[\w-]+)\s*=\s*"([^"]*)\"""",
    re.VERBOSE,
)
_STRING = re.compile(r"'[^']*'")
# A *bare* call / a chain's *first* name: the negative lookbehind drops `.join(`,
# `.holder`, and every other mid-chain member, which the component never declares.
_CALL = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(")
_ROOT = re.compile(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\.")
_XFOR_VARS = re.compile(r"x-for\s*=\s*\"\s*\(?([^)]*?)\)?\s+in\s")

# Names an expression may use that the component does not define: JS keywords and
# literals, JS/DOM globals, and Alpine's magics. Loop variables are collected from the
# page's own ``x-for`` declarations rather than listed here.
_AMBIENT = {
    "in",
    "of",
    "new",
    "typeof",
    "instanceof",
    "void",
    "delete",
    "return",
    "true",
    "false",
    "null",
    "undefined",
    "this",
    "Alpine",
    "Array",
    "Boolean",
    "JSON",
    "Math",
    "Number",
    "Object",
    "String",
    "console",
    "document",
    "window",
    "localStorage",
    "jsyaml",
    "URL",
    "fetch",
    "$event",
    "$el",
    "$refs",
    "$dispatch",
    "$watch",
    "$store",
    "$nextTick",
    "$data",
}


def _component_source() -> str:
    return "\n".join((STATIC / f).read_text(encoding="utf-8") for f in OURS)


def _expressions() -> list[str]:
    """Every Alpine expression on the page, with single-quoted literals blanked out.

    String contents are not code: ``row['Energy (kcal/mol)']`` must not be read as a
    reference to ``Energy``.
    """
    html = INDEX.read_text(encoding="utf-8")
    return [_STRING.sub("''", expression) for expression in _ALPINE_ATTR.findall(html)]


def _bound_names() -> set[str]:
    """Ambient names plus the loop variables the page's own ``x-for``s introduce."""
    html = INDEX.read_text(encoding="utf-8")
    loop_vars = {
        part.strip()
        for declaration in _XFOR_VARS.findall(html)
        for part in declaration.split(",")
        if part.strip()
    }
    return _AMBIENT | loop_vars


def _node() -> str | None:
    """A Node executable: one on ``PATH``, else the one ``nodejs-wheel-binaries`` ships.

    ``PATH`` first, deliberately. A distro packager building from the sdist runs this suite
    with a system Node and no wheel — a vendored-binary wheel is not something a
    distribution will ever depend on — and a developer's own Node is the runtime their
    users actually have. The wheel is the floor under everyone else: it is in ``[test]``,
    so a machine with no Node checks this JavaScript instead of skipping past it.
    """
    found = shutil.which("node")
    if found:
        return found
    try:
        import nodejs_wheel
    except ImportError:  # a suite run without `[test]` — a packager's, typically
        return None
    # The layout is the wheel's, not a public API: `nodejs_wheel/bin/node` on POSIX,
    # `nodejs_wheel/node.exe` on Windows (its `executable._program` builds exactly this
    # path). Derived because the package exports only callables and no path at all, and
    # guarded with `is_file()` so a future relayout degrades to "no node" rather than a
    # FileNotFoundError out of `subprocess.run`. CHEMREFINE_REQUIRE_NODE below is what
    # turns that quiet degradation into a red build — which is what makes deriving a path
    # from someone else's layout safe to do in the first place.
    if nodejs_wheel.__file__ is None:  # a namespace package: no directory to look in
        return None
    root = Path(nodejs_wheel.__file__).parent
    binary = root / "node.exe" if os.name == "nt" else root / "bin" / "node"
    return str(binary) if binary.is_file() else None


def _node_or_skip(purpose: str) -> str:
    """A Node, or skip — or fail, under ``CHEMREFINE_REQUIRE_NODE``.

    Skipping is the right answer for someone with neither a system Node nor ``[test]``
    installed. It is the wrong answer for any *gate*: pytest reports "N passed, 7 skipped"
    and exits 0, so the only checks that execute the GUI's JavaScript can go quiet while
    every gate stays green. That is not hypothetical — it is what they did here until
    ``[test]`` grew a Node. ``ci.yml`` and ``scripts/release-check.sh`` set the variable,
    the same technique and the same reason as ``CHEMREFINE_REQUIRE_LIVE`` in
    ``test_e2e_live.py``.
    """
    node = _node()
    if node is not None:
        return node
    message = f"no node available to {purpose}"
    if os.environ.get("CHEMREFINE_REQUIRE_NODE"):
        pytest.fail(f"{message} — CHEMREFINE_REQUIRE_NODE is set, so this may not be skipped")
    pytest.skip(message)


VENDORED = tuple(sorted(f"vendor/{p.name}" for p in (STATIC / "vendor").glob("*.js")))
"""Every vendored bundle, from the directory rather than from a hand-kept list.

The list used to be written out here, which meant a newly vendored bundle was not
parse-checked until someone remembered to add it — and per this test's own docstring, that
check is the only thing distinguishing a truncated download from a working one. Reading the
directory makes forgetting impossible; the assertion below keeps the glob honest."""


@pytest.mark.parametrize("filename", [*OURS, *VENDORED])
def test_the_javascript_parses(filename: str):
    """A syntax error here blanks the entire GUI — no button, no pane, no message.

    Vendored files are checked too: a truncated download is indistinguishable from a
    working one until the page is opened.
    """
    node = _node_or_skip("parse-check the GUI assets")
    result = subprocess.run(  # argv list, no shell
        [node, "--check", str(STATIC / filename)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"{filename} does not parse:\n{result.stderr}"


def test_every_vendored_bundle_is_found_and_licensed():
    """The glob must actually find things, and each bundle must ship its licence.

    A glob that silently matches nothing would turn the parse check above into zero cases
    while pytest still reported them as passing. And a vendored bundle without its licence
    text is a redistribution problem, not a style one — the two existing pairs set the
    convention (``ALPINE-LICENSE.md``, ``JS-YAML-LICENSE``: the extension tracks upstream).
    """
    assert len(VENDORED) >= 3, f"the vendor glob found {VENDORED} — has the directory moved?"
    licences = [p.name for p in (STATIC / "vendor").iterdir() if "LICENSE" in p.name.upper()]
    assert len(licences) == len(VENDORED), (
        f"{len(VENDORED)} bundles but {len(licences)} licence files: {sorted(licences)}"
    )


def test_every_handler_the_page_calls_exists():
    """An ``@click`` naming a method that isn't there fails silently, at click time.

    The component is one object literal, so a method exists iff ``name(`` or
    ``name:`` appears in the sources — enough to catch a rename or a deletion, which
    is the failure this guards.
    """
    source = _component_source()
    ambient = _bound_names()
    missing = sorted(
        {
            call
            for expression in _expressions()
            for call in _CALL.findall(expression)
            if call not in ambient
            and not re.search(rf"\b(?:async\s+)?{re.escape(call)}\s*\(", source)
            and not re.search(rf"\b{re.escape(call)}\s*:", source)
        }
    )
    assert not missing, f"index.html calls handlers the component does not define: {missing}"


def test_every_state_root_the_page_reads_exists():
    """Same guard for state: ``chat.busy`` is undefined-dot-busy if ``chat`` was renamed."""
    source = _component_source()
    ambient = _bound_names()
    missing = sorted(
        {
            root
            for expression in _expressions()
            for root in _ROOT.findall(expression)
            if root not in ambient and not re.search(rf"\b{re.escape(root)}\s*:", source)
        }
    )
    assert not missing, f"index.html reads state the component does not declare: {missing}"


def _options_after(marker: str) -> list[str]:
    """The ``<option value="…">`` values of the first ``<select>`` following ``marker``.

    Bounded by the element, not by a byte count. A fixed window did the same job until it
    did not: the sample dropdown's window cleared the *next* ``<select>``'s first option by
    six characters, so shortening a purely cosmetic label — ``(keep everything)`` → ``(all)``
    — reported a vocabulary mismatch for a vocabulary that had not changed. Slicing to
    ``</select>`` makes the helper mean what its name says, and makes an edit to the page's
    prose incapable of moving the answer.

    Text analysis, like the two guards above: the page is one file, the selects are
    hand-written where they are not schema-driven, and reading them back is what lets a
    hardcoded list be compared against the model that owns it.
    """
    html = INDEX.read_text(encoding="utf-8")
    after = html.split(marker, 1)[1]
    element = after[: after.index("</select>")]
    return re.findall(r'<option value="([^"]*)"', element)


def test_the_pages_hardcoded_vocabularies_match_the_models_that_own_them():
    """Three dropdowns spell out values that live in Python; require them to agree.

    Most of what the builder offers is schema-driven — engines, `operation:`, every option
    and NMS field come from `/api/bootstrap`. These three do not, and each is a place a
    future change can be made in Python alone and go silently missing from the UI:
    a fourth `SampleConfig` variant is a mypy error in `filtering._dispatch` and a
    `test_docs_drift` failure, but the page would simply never offer it, and a config that
    named it in the YAML pane would render *no* knobs at all (`sampleFields` → `[]`).

    They are not the only such place, which this docstring used to claim: the results
    table names `steps.csv` columns, which arrive through `/api/results` rather than
    through the schema. That fourth vocabulary has its own test below — the enumeration
    was the thing that had drifted, not the list.

    Kept as text guards rather than deriving the lists in JavaScript, because the labels
    are prose the models cannot supply ("min (lowest-energy)") — so the duplication is
    deliberate and this is what makes it honest. `provider` has no schema source at all
    (`_PRESETS` is private and stays that way); comparing against the dict directly is
    still the guard, and `providers` imports no SDK at runtime, so it costs nothing.
    """
    from chemrefine.agent import providers
    from chemrefine.introspect import schema_document

    # Read the schema document, not the models: it is exactly what `/api/bootstrap` hands
    # the page, so the comparison is against what the frontend could have derived.
    config_schema = schema_document()["config"]
    defs = config_schema["$defs"]
    variants = {
        name: schema["properties"]["method"]["const"]
        for name, schema in defs.items()
        if name.endswith("Sample") and "method" in schema.get("properties", {})
    }

    # `sample`: an empty option means "keep everything" (no `sample:` block), then one per
    # variant — and app.js maps each method back to the $defs name whose fields it renders.
    assert sorted(_options_after("<span>sample</span>")) == sorted(["", *variants.values()])
    by_method = dict(
        re.findall(r'(\w+):\s*"(\w+Sample)"', (STATIC / "app.js").read_text(encoding="utf-8"))
    )
    assert by_method == {method: name for name, method in variants.items()}

    # `on_failure`: a plain Literal, so the schema already carries the enum.
    assert (
        _options_after("<span>on_failure</span>")
        == defs["StepConfig"]["properties"]["on_failure"]["enum"]
    )

    # `provider`: mirrors the agent's preset table, which the schema does not publish.
    assert sorted(_options_after('x-model="chat.provider"')) == sorted(providers._PRESETS)


def test_the_results_table_names_columns_the_report_actually_writes():
    """The fourth hardcoded vocabulary, and the one nothing was watching.

    The results table reads `row.Conformer`, `row['Energy (kcal/mol)']` and two more
    straight off `/api/results`, which hands back `steps.csv` rows verbatim. Those names
    are owned by `io.save_step_csv` and reach the page through no schema, so renaming a
    column there left the table rendering the literal string `undefined` in one cell and
    `NaN` in three — with the whole suite green, which is exactly the failure the three
    guards above exist to prevent, in the one spot they did not cover.
    """
    html = INDEX.read_text(encoding="utf-8")
    # Scoped to the results table's own x-for: `row` is the loop variable in the run-status
    # table and the executables rows too, and those iterate objects this module does not
    # own. Bounded by the element, like `_options_after` above and for the same reason.
    marker = "runResults.rows"
    assert marker in html, "the results table's x-for has moved; this guard needs its new anchor"
    block = html.split(marker, 1)[1]
    block = block[: block.index("</table>")]
    # Both spellings the page uses: `row.Name` for the identifier-safe one, `row['A b']`
    # for the rest. Anything the table reads off a row has to be a column that exists.
    named = set(re.findall(r"\brow\.([A-Za-z_]\w*)", block)) | set(
        re.findall(r"\brow\['([^']+)'\]", block)
    )
    assert named, "the results table reads no columns — has it been rewritten?"
    unknown = named - set(io.STEPS_CSV_COLUMNS)
    assert not unknown, f"the page reads columns steps.csv does not write: {sorted(unknown)}"


def _run_in_node(script: str) -> str:
    """Evaluate ``script`` with ``forms.js`` already loaded; return its stdout.

    ``forms.js`` is plain top-level functions (no module system — the page loads it
    with a bare script tag), so sourcing it is a concatenation.
    """
    node = _node_or_skip("execute the GUI's pure form logic")
    source = (STATIC / "forms.js").read_text(encoding="utf-8") + "\n" + script
    result = subprocess.run(  # argv list, no shell
        [node, "-e", source], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _run_component_in_node(script: str) -> str:
    """Evaluate ``script`` against a real ``builder()`` instance; return its stdout.

    ``forms.js``'s docstring says logic left in ``app.js`` "cannot be reached by a test at
    all", because ``builder()`` reads ``window`` and ``localStorage`` as it is constructed.
    That is true of the *browser* globals, not of the logic: stubbing those three is
    enough, and what it buys is the ability to check behaviour the markup only describes —
    which fields a provider shows, and whether a credential reaches ``localStorage``.

    The stubbed ``localStorage`` is a plain object, so a test can read back exactly what
    the component wrote to it.
    """
    node = _node_or_skip("execute the GUI's component logic")
    preamble = """
      global.window = { location: { search: "?token=t" } };
      global.localStorage = { _d: {}, getItem(k){return this._d[k] ?? null;},
                              setItem(k,v){this._d[k]=v;}, removeItem(k){delete this._d[k];} };
      global.URLSearchParams = URLSearchParams;
    """
    # Both files, into one scope, because that is what the page gives them: classic
    # scripts sharing globals, where app.js reaches forms.js's helpers (
    # `seedWorkflowDefaults` → `topFields` → `fieldSpecs`). Loading app.js alone would
    # model a page that cannot exist. Concatenation order does not matter here — both
    # files are top-level function declarations, which hoist — which is also why the
    # page's own load order is not something this has to reproduce.
    source = preamble + _component_source() + "\n" + script
    result = subprocess.run(  # argv list, no shell
        [node, "-e", source], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_the_panel_offers_each_provider_only_the_fields_it_can_use():
    """The conditional-field rule, run rather than read off the markup.

    Which inputs appear is decided in JavaScript from the shapes ``/api/agent/availability``
    serves, so the `x-show` expressions alone prove nothing. This drives the real component
    with the real shapes: ``custom`` is the only provider with nowhere to go and so the only
    one that must be told where; the two local presets ship a dummy key and so must not ask
    for one.
    """
    from chemrefine.agent import providers  # imports no SDK at runtime

    shapes = json.dumps(providers.preset_shapes())
    out = _run_component_in_node(f"""
      const b = builder();
      b.chat.presets = {shapes};
      const rows = ["ollama", "vllm", "openai", "custom"].map((p) => {{
        b.chat.provider = p;
        return [p, b.needsBaseUrl(), b.needsApiKey()].join(":");
      }});
      console.log(rows.join(" "));
    """)
    assert out == "ollama:false:false vllm:false:false openai:false:true custom:true:true"


def test_the_api_key_never_reaches_localstorage_but_does_reach_the_request():
    """The panel's security claim, gated instead of asserted.

    The key is a live credential typed into a page whose provider and model *are*
    remembered, so the symmetry actively invites a fourth ``setItem`` — and a browser
    profile outlives the session the key was typed for. What must hold is both halves: it
    goes out with the request, and it is not written down.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.chat.model = "m";
      b.chat.baseUrl = "http://h/v1";
      b.chat.apiKey = "sk-secret-value";
      b.saveChatSettings();
      console.log(JSON.stringify({
        stored: Object.keys(localStorage._d).sort(),
        leaked: JSON.stringify(localStorage._d).includes("sk-secret-value"),
        sent: b._chatPayload({ message: "hi" }).api_key,
      }));
    """)
    result = json.loads(out)
    assert result["sent"] == "sk-secret-value"  # it does reach the request
    assert result["leaked"] is False  # and nowhere else
    assert result["stored"] == ["cr-baseurl", "cr-model", "cr-provider"]


def test_send_is_armed_by_a_passing_check_and_disarmed_by_any_edit():
    """A verdict belongs to the settings that earned it.

    ``ok: null`` (never checked) and ``ok: false`` (checked, failed) both mean "do not
    send", but only the first is the state a fresh panel is in — which is why the check
    result is a tri-state and not a boolean.
    """
    out = _run_component_in_node("""
      const b = builder();
      const seen = [b.chatReady()];
      b.chat.check = { ok: true, findings: [], busy: false };
      seen.push(b.chatReady());
      b.armCheck();
      seen.push(b.chatReady(), b.chat.check.ok === null);
      console.log(JSON.stringify(seen));
    """)
    assert json.loads(out) == [False, True, False, True]


def test_switching_to_the_agent_tab_reprobes_without_disarming_send():
    """The bug that only exists once tabs and the check gate are both in.

    Under a ``<details>`` the availability probe fired on open and close. As a tab it
    fires on every switch — so if arming lived inside it, or if the verdict shared
    ``chat.detail`` (which the probe overwrites unconditionally), every click on Agent
    would silently disarm Send mid-conversation. The probe must still run, and the
    verdict must survive it.
    """
    out = _run_component_in_node("""
      const b = builder();
      let probes = 0;
      b.chatAvailability = () => { probes += 1; };
      b.chat.check = { ok: true, findings: ["served"], busy: false };
      b.showTab("left", "agent");
      b.showTab("left", "builder");
      b.showTab("left", "agent");
      console.log(JSON.stringify({ probes, stillReady: b.chatReady(), tab: b.tabs.left }));
    """)
    result = json.loads(out)
    assert result["probes"] == 2  # every switch *to* the agent re-probes
    assert result["stillReady"] is True  # and none of them took the verdict away
    assert result["tab"] == "agent"


def test_each_column_switches_independently():
    """Two strips, one state object — a switch on one column must not move the other."""
    out = _run_component_in_node("""
      const b = builder();
      b.chatAvailability = () => {};
      const seen = [JSON.stringify(b.tabs)];
      b.showTab("right", "structure");
      seen.push(JSON.stringify(b.tabs));
      b.showTab("left", "agent");
      seen.push(JSON.stringify(b.tabs));
      console.log(JSON.stringify(seen));
    """)
    assert [json.loads(s) for s in json.loads(out)] == [
        {"left": "builder", "right": "yaml"},  # the defaults a fresh page opens on
        {"left": "builder", "right": "structure"},
        {"left": "agent", "right": "structure"},
    ]


def test_every_tab_button_targets_a_panel_that_exists():
    """A tab whose panel nothing renders is a button that blanks half the page.

    The strip and the panels are two independent lists in the markup, so a typo in either
    shows an empty column at click time and nowhere else — the same silent-at-click-time
    failure the handler guard exists for, one level up.
    """
    html = INDEX.read_text(encoding="utf-8")
    targets = set(re.findall(r"showTab\('(\w+)',\s*'(\w+)'\)", html))
    panels = set(re.findall(r"tabs\.(\w+) === '(\w+)'", html))
    assert targets, "no tab buttons found — has the strip been renamed?"
    assert targets <= panels, f"tabs with no panel: {sorted(targets - panels)}"
    # Both columns are tabbed, and each offers at least two panels — a one-tab strip is a
    # decoration, and would mean a panel lost its button.
    for side in ("left", "right"):
        assert len({t for s, t in targets if s == side}) >= 2, f"{side} column has one tab"


_RELOAD_HARNESS = """
  const b = builder();
  b.chatAvailability = () => {};
  const calls = [];
  b.api = async (method, url, body) => {
    calls.push(method + " " + url);
    if (url.startsWith("/api/load")) return { path: "/p/input.yaml", yaml_text: "steps: []\\n" };
    if (url === "/api/parse") return { config: { steps: [] } };
    if (url === "/api/yaml") return { yaml_text: "from-the-server\\n" };
    return {};
  };
  b.schema = { config: { properties: {} } };  // topFields reads it in seedWorkflowDefaults
"""


def test_an_agent_write_reloads_the_form_and_shows_the_pane():
    """The point of the whole item: the agent saves, the editor catches up.

    Also switches the right column back to the YAML — item 6 made it possible to be
    looking at the molecule while the agent writes, and a change nobody can see is not
    much of an improvement.
    """
    out = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      b.showTab("right", "structure");
      await b.loadConfigFrom("/p/input.yaml");
      console.log(JSON.stringify({
        calls, tab: b.tabs.right, savedPath: b.savedPath,
        clean: !b.dirty(), pending: b.pendingReload,
      }));
    """
    )
    result = json.loads(out)
    assert "GET /api/load?path=%2Fp%2Finput.yaml" in result["calls"]
    assert result["tab"] == "yaml"  # the pane the change happened in
    assert result["savedPath"] == "/p/input.yaml"
    assert result["pending"] is None
    # Clean straight after a reload: savedText is set from the *re-emitted* YAML, because
    # seedWorkflowDefaults() adds keys the file omits and comparing to the file's own
    # bytes would read dirty the instant it loaded.
    assert result["clean"] is True


def test_a_write_never_silently_discards_unsaved_edits():
    """Unsaved work outranks the agent. It offers; it does not take.

    ``rawEdit`` counts as unsaved even when the text matches, because ``syncYaml()``
    early-returns while raw editing — adopting underneath it would leave the two panes
    describing different configs.
    """
    dirty = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      b.yamlText = "steps: [{step: 1}]\\n";  // typed, never saved
      b.savedText = "";
      await b.loadConfigFrom("/p/input.yaml");
      const offered = JSON.parse(JSON.stringify(b.pendingReload));
      const before = calls.length;
      await b.acceptPendingReload();          // the user chooses to take it
      console.log(JSON.stringify({ offered, loadedOnlyAfterConsent: calls.length > before,
                                   pendingAfter: b.pendingReload }));
    """
    )
    result = json.loads(dirty)
    # Offered, not applied — and it says which kind of load is waiting, because the
    # notice reads differently for an agent write than for a file the user chose.
    assert result["offered"] == {"path": "/p/input.yaml", "reason": "agent"}
    assert result["loadedOnlyAfterConsent"] is True
    assert result["pendingAfter"] is None

    raw = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      b.rawEdit = true;                       // text matches, but the pane is authoritative
      await b.loadConfigFrom("/p/input.yaml");
      console.log(JSON.stringify({ pending: b.pendingReload, calls }));
    """
    )
    assert json.loads(raw)["pending"] == {"path": "/p/input.yaml", "reason": "agent"}
    assert not any(c.startswith("GET /api/load") for c in json.loads(raw)["calls"])


def test_the_playground_never_tries_to_read_a_file():
    """staticMode has no server behind it; a reload there would only flash an error."""
    out = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      b.staticMode = true;
      await b.loadConfigFrom("/p/input.yaml");
      console.log(JSON.stringify({ calls, pending: b.pendingReload }));
    """
    )
    assert json.loads(out) == {"calls": [], "pending": None}


def test_the_agent_is_sent_the_path_the_builder_has_open():
    """`config_path` is sourced from savedPath, not from chat state.

    Every neighbouring line in ``_chatPayload`` reads ``this.chat.*``, which makes the
    wrong source the natural one to reach for; what the agent needs to hear about is the
    file the *builder* has open.
    """
    out = _run_component_in_node("""
      const b = builder();
      const before = b._chatPayload({ message: "hi" }).config_path;
      b.savedPath = "/p/input.yaml";
      const after = b._chatPayload({ message: "hi" }).config_path;
      console.log(JSON.stringify({ before: before ?? null, after }));
    """)
    assert json.loads(out) == {"before": None, "after": "/p/input.yaml"}


def test_a_launched_config_is_clean_at_boot():
    """The regression that broke the whole reload feature for its most common case.

    ``chemrefine gui input.yaml`` boots with the form mirroring the file, and the user has
    typed nothing — but ``savedText`` was only ever written by saveTo() and the reload
    itself, so it stayed "" and ``dirty()`` answered true from the first paint. The very
    first agent write then arrived as "you have unsaved edits", about edits nobody made.
    """
    out = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      b.savedPath = "/p/input.yaml";        // as /api/bootstrap's `initial` sets it
      b.cfg = { steps: [] };
      await b.recordOnDisk();               // what init() now does for a launched config
      const clean = !b.dirty();
      await b.loadConfigFrom("/p/input.yaml");
      console.log(JSON.stringify({ clean, pending: b.pendingReload }));
    """
    )
    result = json.loads(out)
    assert result["clean"] is True  # a freshly loaded file is not an unsaved edit
    assert result["pending"] is None  # so the agent's write is adopted, not queued


def test_a_superseded_check_cannot_rearm_send():
    """A probe belongs to the settings that asked for it.

    The check takes a second or two against a real endpoint. Change a setting during it
    and ``armCheck()`` disarms Send — but the in-flight verdict then landed anyway and
    re-armed it, against settings nobody had checked. Which is the one thing the gate is
    for.
    """
    out = _run_component_in_node("""
      const b = builder();
      let release;
      b.api = () => new Promise((r) => { release = () => r({ ok: true, findings: ["served"] }); });
      const probe = b.runCheck();           // in flight, for the old settings
      b.chat.model = "something-else";
      b.armCheck();                          // the user edits a field
      release();
      await probe;
      console.log(JSON.stringify({ ok: b.chat.check.ok, ready: b.chatReady() }));
    """)
    assert json.loads(out) == {"ok": None, "ready": False}


def test_the_key_is_not_sent_to_a_provider_whose_key_field_is_hidden():
    """The key survives a provider switch; it must not travel with one that hides it.

    ollama and vllm ship a dummy key in their preset, so the panel hides the field — but
    the value was still in state and still went out, handing an OpenAI credential to
    whatever host the local preset points at.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.chat.presets = {
        openai: { default_url: "https://api.openai.com/v1", needs_key: true },
        vllm:   { default_url: "http://localhost:8000/v1", needs_key: false },
      };
      b.chat.provider = "openai";
      b.chat.apiKey = "sk-secret";
      const sentToOpenai = b._chatPayload({}).api_key ?? null;
      b.chat.provider = "vllm";              // the field is hidden now
      const sentToVllm = b._chatPayload({}).api_key ?? null;
      console.log(JSON.stringify({ sentToOpenai, sentToVllm, stillHeld: b.chat.apiKey }));
    """)
    assert json.loads(out) == {
        "sentToOpenai": "sk-secret",
        "sentToVllm": None,
        "stillHeld": "sk-secret",  # kept, so switching back does not mean retyping
    }


def test_renumbering_releases_every_panel_that_names_a_step():
    """Two panels select a step by number, and both must let go when the numbers move.

    They land differently on purpose: the results table has an em-dash placeholder to fall
    back to, while the Structure pane's selector has no empty option — so it falls back to
    the seeds, which are always there.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.api = async () => ({ yaml_text: "" });
      b.cfg = { steps: [{ step: 1 }, { step: 2 }, { step: 3 }] };
      b.stepKeys = [1, 2, 3];
      b.resultsStep = "3";
      b.viewer.step = "3";
      b.cfg.steps.splice(0, 1);              // delete step 1; 3 becomes 2
      b.renumber();
      console.log(JSON.stringify({ results: b.resultsStep, viewer: b.viewer.step }));
    """)
    assert json.loads(out) == {"results": "", "viewer": "input"}


def test_opening_a_workflow_adopts_it_and_makes_its_run_reachable():
    """The door a config could not come through except at launch.

    `savedPath` had three writers — the launch argument, Save…, and the agent reload — so
    looking at a finished run meant restarting the GUI pointed at its file. Opening one
    now sets the same state a launch would, which is what brings its Run panel, its
    results and its structures with it.
    """
    out = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      b.showTab("right", "structure");              // where you go to look at a finished run
      b.browse = { ...b.browse, open: true, mode: "open" };
      await b.pickFile({ name: "input.yaml", path: "/p/input.yaml", dir: false });
      await new Promise((r) => setTimeout(r, 0));   // pickFile is sync; the load is not
      console.log(JSON.stringify({
        savedPath: b.savedPath, modalOpen: b.browse.open,
        clean: !b.dirty(), tab: b.tabs.right, flash: b.flash,
      }));
    """
    )
    result = json.loads(out)
    assert result["savedPath"] == "/p/input.yaml"  # the Run panel keys off exactly this
    assert result["modalOpen"] is False
    assert result["clean"] is True  # a freshly opened file is not an unsaved edit
    # Opening a run is how you go and *look* at it, so the tab you chose is kept. Only the
    # agent's own write steals the pane back, and that is asserted separately above.
    assert result["tab"] == "structure"
    assert "loaded /p/input.yaml" in result["flash"]  # not "the agent wrote…"


def test_opening_does_not_discard_unsaved_edits_either():
    """The same guard the agent write gets, and a notice that says which one is waiting."""
    out = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      b.yamlText = "steps: [{step: 1}]\\n";   // typed, never saved
      b.savedText = "";
      b.browse = { ...b.browse, open: true, mode: "open" };
      await b.pickFile({ name: "other.yaml", path: "/p/other.yaml", dir: false });
      await new Promise((r) => setTimeout(r, 0));
      console.log(JSON.stringify({ pending: b.pendingReload, savedPath: b.savedPath }));
    """
    )
    result = json.loads(out)
    assert result["pending"] == {"path": "/p/other.yaml", "reason": "open"}
    assert result["savedPath"] is None  # nothing adopted until the user says so


def test_picking_a_file_still_means_the_other_two_things_in_the_other_two_modes():
    """One modal, three jobs — a new mode must not steal the other two.

    ``pick`` fills a path field, ``save`` fills the filename box, and only ``open`` loads.
    """
    out = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      let picked = null;
      b.browse = { ...b.browse, mode: "pick", onPick: (p) => { picked = p; } };
      b.pickFile({ name: "seeds.xyz", path: "/p/seeds.xyz", dir: false });
      const afterPick = { picked, savedPath: b.savedPath };
      b.browse = { ...b.browse, open: true, mode: "save", onPick: null };
      b.pickFile({ name: "old.yaml", path: "/p/old.yaml", dir: false });
      console.log(JSON.stringify({ afterPick, filename: b.browse.filename,
                                   savedPath: b.savedPath }));
    """
    )
    result = json.loads(out)
    assert result["afterPick"] == {"picked": "/p/seeds.xyz", "savedPath": None}
    assert result["filename"] == "old.yaml"  # save mode fills the name box
    assert result["savedPath"] is None  # and neither mode loads anything


def test_the_viewer_asks_for_seeds_by_omitting_the_step():
    """ "input" is a sentinel in the page, never a value the server sees.

    A step may be *named* anything, so sending ``step=input`` would be read as a step name
    and look up a step nobody has. Absence is the request for the seeds.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.savedPath = "/p/input.yaml";
      const asked = [];
      b.api = async (m, u) => { asked.push(u); return null; };   // stop after the request
      b.mountViewer = async () => {};
      // A mode number is set FIRST, so the seeds request has one to suppress. Left at its
      // default "" the `modeIndex !== ""` half excluded it on its own and the clause this
      // case is named for — `step !== "input"` — was never exercised at all.
      b.viewer.modeIndex = "6";
      await b.showStructure();                      // viewer.step defaults to "input"
      b.chooseViewerStep("2"); b.viewer.modeIndex = "6";
      await b.showStructure();
      console.log(JSON.stringify(asked));
    """)
    seeds, step = json.loads(out)
    assert "step=" not in seeds  # no step at all, not step=input
    assert "mode_index" not in seeds  # and the seeds have no mode, whatever the box holds
    assert "step=2" in step
    assert "mode_index=6" in step


def test_a_mode_that_could_not_be_drawn_does_not_follow_you_to_the_next_one():
    """The sequence that turned one bad request into "now I cannot show anything".

    ``modeIndex`` was cleared nowhere, so a mode that failed once rode along on every later
    Show, for every step and every structure. And because the box is hidden on the seeds
    view, the value was invisible as well as wrong — there was no way to see why, and no
    obvious way back.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.savedPath = "/p/input.yaml";
      const asked = [];
      let refuse = true;
      b.api = async (m, u) => {
        asked.push(u);
        return refuse ? null : { step: 2, structure_id: "0", mode_index: null, text: "" };
      };
      b.mountViewer = async () => {};
      b._gl = { stopAnimate(){}, removeAllModels(){}, addModel: () => ({ vibrate(){} }),
                setStyle(){}, addUnitCell(){}, zoomTo(){}, animate(){}, render(){} };

      b.chooseViewerStep("1");
      b.viewer.modeIndex = "6";
      await b.showStructure();                     // the .out is gone: refused
      const afterFailure = { mode: b.viewer.modeIndex, note: b.viewer.note };

      refuse = false;                              // the very next Show must work
      await b.showStructure();
      console.log(JSON.stringify({ afterFailure, asked, note: b.viewer.note }));
    """)
    result = json.loads(out)
    assert result["afterFailure"]["mode"] == ""  # cleared, so it cannot poison the next one
    assert result["afterFailure"]["note"]  # and the pane says something, not nothing
    assert "mode_index" in result["asked"][0]
    assert "mode_index" not in result["asked"][1]  # the retry is a plain structure request
    assert "step 2 · 0" in result["note"]


def test_choosing_a_step_drops_what_belonged_to_the_previous_one():
    """A structure id and a mode number are meaningless against a different step."""
    out = _run_component_in_node("""
      const b = builder();
      b.viewer.step = "1"; b.viewer.structureId = "7"; b.viewer.modeIndex = "6";
      b.chooseViewerStep("1");                     // the same step: nothing to drop
      const same = { ...b.viewer };
      b.chooseViewerStep("2");
      console.log(JSON.stringify({ same, moved: { ...b.viewer } }));
    """)
    result = json.loads(out)
    assert (result["same"]["structureId"], result["same"]["modeIndex"]) == ("7", "6")
    assert result["moved"]["step"] == "2"
    assert (result["moved"]["structureId"], result["moved"]["modeIndex"]) == ("", "")


def test_the_seed_view_survives_renumbering():
    """The sentinel is not a step number, so the staleness guard must not collect it.

    ``renumber()`` releases a panel whose chosen step has gone. "input" never goes — the
    seeds outlive any renumbering — and a step that *has* gone falls back to it rather
    than to an empty selection with no matching option.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.api = async () => ({ yaml_text: "" });
      const run = (chosen) => {
        b.cfg = { steps: [{step:1},{step:2},{step:3}] };
        b.stepKeys = [1,2,3];
        b.viewer.step = chosen;
        b.cfg.steps.splice(0, 1);
        b.renumber();
        return b.viewer.step;
      };
      console.log(JSON.stringify({ sentinel: run("input"), vanished: run("3") }));
    """)
    assert json.loads(out) == {"sentinel": "input", "vanished": "input"}


def test_opening_a_workflow_leaves_none_of_the_previous_one_on_screen():
    """Everything keyed off ``savedPath`` describes the file that *was* loaded.

    The Run panel refreshes only when it is toggled, so after an Open its status table,
    failure count and results rows kept describing the previous workflow under the new
    file's name — and the viewer kept pointing at a step number the new config need not
    have. A stale report and a half-open template editor came along too.
    """
    out = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      b.runStatus = { state: "failed", steps: [{ step: 1 }] };
      b.runFailures = { count: 3 };
      b.runResults = { rows: [{ id: "0" }] };
      b.resultsStep = "2";
      b.report = { ok: false };
      b.tmpl = { open: true, step: 2, path: "/old/tmpl.inp", text: "old" };
      b.viewer = { ...b.viewer, step: "7", structureId: "4", modeIndex: "6", note: "stale" };
      let stopped = 0, cleared = 0;
      b._gl = { stopAnimate: () => { stopped++; }, removeAllModels: () => { cleared++; },
                render: () => {} };
      await b.loadConfigFrom("/p/input.yaml", { reason: "open" });
      console.log(JSON.stringify({
        runStatus: b.runStatus, runFailures: b.runFailures, runResults: b.runResults,
        resultsStep: b.resultsStep, report: b.report, tmplOpen: b.tmpl.open,
        viewer: b.viewer, stopped, cleared,
      }));
    """
    )
    result = json.loads(out)
    assert result["runStatus"] is None
    assert result["runFailures"] is None
    assert result["runResults"] is None
    assert result["resultsStep"] == ""
    assert result["report"] is None  # it validated a file that is no longer in the form
    assert result["tmplOpen"] is False
    # Back to the one view every tree can answer, with nothing carried over from the old one.
    assert result["viewer"]["step"] == "input"
    assert (result["viewer"]["structureId"], result["viewer"]["modeIndex"]) == ("", "")
    assert result["viewer"]["note"] == ""
    # And the old tree's molecule is off the canvas rather than left there animating.
    assert (result["stopped"], result["cleared"]) == (1, 1)


def test_a_failed_open_keeps_the_browser_where_you_had_navigated_to():
    """A directory, an unreadable file or a bad ``~user`` must not dismiss the dialog.

    ``openConfigFrom`` closed the modal before the request, unlike ``saveTo()``, so a
    failure left a flash, no browser, and no way back to wherever you had browsed.
    """
    out = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      b.api = async (m, url) => (url.startsWith("/api/load") ? null : { config: { steps: [] } });
      b.browse = { ...b.browse, open: true, mode: "open", path: "/deep/in/a/tree" };
      await b.openConfigFrom("/deep/in/a/tree/notes.txt");
      console.log(JSON.stringify({
        modalOpen: b.browse.open, where: b.browse.path, savedPath: b.savedPath,
      }));
    """
    )
    assert json.loads(out) == {
        "modalOpen": True,
        "where": "/deep/in/a/tree",
        "savedPath": None,  # nothing was adopted, so the form keeps what it had
    }


def test_a_failed_accept_puts_the_offer_back():
    """The banner is the only thing that still names the file the agent wrote.

    ``acceptPendingReload`` cleared it before the load could fail — and it must clear it,
    or the load re-offers the path to itself — so a failed accept lost the write for good.
    """
    out = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      b.yamlText = "steps: [{step: 1}]\\n";        // unsaved edits, so the write is offered
      b.savedText = "";
      await b.loadConfigFrom("/p/written.yaml");
      const offered = b.pendingReload;
      b.api = async (m, url) => (url.startsWith("/api/load") ? null : { config: { steps: [] } });
      await b.acceptPendingReload();               // the file has gone in the meantime
      console.log(JSON.stringify({ offered, after: b.pendingReload }));
    """
    )
    result = json.loads(out)
    assert result["offered"] == {"path": "/p/written.yaml", "reason": "agent"}
    assert result["after"] == result["offered"]  # still there to try again


def test_a_launched_config_and_an_opened_one_seed_the_same_starter_step():
    """``adopt()``'s comment named ``init()`` as a caller while ``init()`` hand-rolled it.

    Only ``init()`` had the starter-step fallback, so a config whose ``steps:`` is empty
    showed one seeded step from the command line and an empty builder through Open… — the
    same file, two different forms.
    """
    out = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      global.fetch = async () => ({
        ok: true, status: 200,
        json: async () => ({ schema: { config: { properties: {} } }, host: "",
                             initial: { path: "/p/input.yaml", yaml_text: "steps: []\\n" } }),
      });
      await b.init();
      const launched = b.cfg.steps.length;
      const c = builder();
      c.chatAvailability = () => {};
      c.api = b.api; c.schema = b.schema;
      await c.loadConfigFrom("/p/input.yaml", { reason: "open" });
      console.log(JSON.stringify({ launched, opened: c.cfg.steps.length }));
    """
    )
    assert json.loads(out) == {"launched": 1, "opened": 1}


def test_a_bundle_that_loads_but_defines_nothing_is_a_failure_not_a_silence():
    """Loaded and defined are two facts, and only the first has an event.

    ``onload`` resolved ``window.$3Dmol`` unchecked, so a truncated or shimmed bundle
    resolved ``undefined`` and every later Show returned at the ``!lib`` guard — no note,
    no flash, for the rest of the session.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.chatAvailability = () => {};
      let tag = null;
      global.document = {
        createElement: () => (tag = {}),
        head: { appendChild: () => {} },
        getElementById: () => ({}),
      };
      b.savedPath = "/p/input.yaml";               // past the nothing-to-show guard
      const first = b.mountViewer();
      tag.onload();                                 // the script arrives, and defines nothing
      await first;
      const noted = b.viewer.note;
      // And the cached rejection is dropped, so the next open still gets to try.
      const retried = b.loadViewerLib() !== null;
      console.log(JSON.stringify({ noted, retried }));
    """)
    result = json.loads(out)
    assert "3D viewer could not start" in result["noted"]
    assert "defined nothing" in result["noted"]
    assert result["retried"] is True


def test_the_viewer_bundle_is_not_fetched_for_a_pane_that_can_show_nothing():
    """Half a megabyte to render the sentence "save the workflow first"."""
    out = _run_component_in_node("""
      const b = builder();
      b.chatAvailability = () => {};
      let created = 0;
      global.document = {
        createElement: () => { created++; return {}; },
        head: { appendChild: () => {} },
        getElementById: () => ({}),
      };
      b.showTab("right", "structure");             // opened with nothing saved
      await new Promise((r) => setTimeout(r, 0));
      console.log(JSON.stringify({ created }));
    """)
    assert json.loads(out) == {"created": 0}


def test_field_specs_carry_the_schema_bounds_and_default():
    """The spec a number input renders from: bounds for the spinner, default to step from.

    An empty box steps from ``min`` (or 0), which is why ``max_cores`` showed a grey 4
    and its arrow produced 1 — the default has to be a real value, and the bounds have
    to reach the input.
    """
    out = _run_in_node("""
      const spec = fieldSpec("max_cores", {type: "integer", minimum: 1, default: 4});
      const optional = fieldSpec("max_gpus", {anyOf: [{type: "integer", minimum: 0},
                                                      {type: "null"}], default: null});
      const exclusive = fieldSpec("temperature_k", {type: "number", exclusiveMinimum: 0,
                                                    default: 298.15});
      console.log(JSON.stringify({
        cores: [spec.kind, spec.min, spec.max, spec.fallbackNum],
        gpus: [optional.kind, optional.min, optional.fallbackNum],
        temp: [exclusive.min, exclusive.fallbackNum],
      }));
    """)
    assert json.loads(out) == {
        "cores": ["number", 1, None, 4],
        "gpus": ["number", 0, None],  # Optional[int] unwrapped; no default to display
        "temp": [0, 298.15],
    }


def test_coercion_clamps_and_splits_defaults_by_scope():
    """The two rules a user feels: bounds are enforced, and scope decides default-keeping.

    Workflow settings are written out even at their default (the file reads as a
    complete protocol, like the shipped examples); a step's knobs are deviation-only,
    so a step shows the two lines that matter instead of every default it inherited.
    """
    out = _run_in_node("""
      const cores = fieldSpec("max_cores", {type: "integer", minimum: 1, default: 4});
      const count = fieldSpec("count", {type: "integer", minimum: 0, default: null});
      console.log(JSON.stringify({
        typed_zero_clamps: coerceField(cores, "0"),
        typed_zero_clamps_kept: coerceField(cores, "0", true),
        default_dropped_for_steps: coerceField(cores, "4") === undefined,
        default_kept_for_settings: coerceField(cores, "4", true),
        real_value_survives_both: [coerceField(cores, "16"), coerceField(cores, "16", true)],
        blank_is_unset: coerceField(count, "") === undefined,
        no_default_means_nothing_to_drop: coerceField(count, "0"),
      }));
    """)
    assert json.loads(out) == {
        "typed_zero_clamps": 1,
        "typed_zero_clamps_kept": 1,
        "default_dropped_for_steps": True,
        "default_kept_for_settings": 4,
        "real_value_survives_both": [16, 16],
        "blank_is_unset": True,
        "no_default_means_nothing_to_drop": 0,
    }


def test_the_playgrounds_yaml_order_matches_the_servers():
    """``canonicalConfigOrder`` is the static half of the server's ``_canonical_order``.

    The playground dumps with js-yaml, which writes keys in click order — and a fresh
    session's cfg starts ``{steps: []}``, so its YAML led with the steps block and
    trailed the settings, the exact shape the server emitter was written to fix. The
    order authority is the same schema document; unknown keys keep their place, last.
    """
    out = _run_in_node("""
      const schemaDoc = {config: {
        properties: {charge: {}, max_cores: {}, steps: {}},
        $defs: {StepConfig: {properties: {step: {}, engine: {}, template: {}}}},
      }};
      const clicked = {steps: [{template: "t.inp", mystery: 1, step: 1, engine: "orca"}],
                       max_cores: 8, unknown_key: true, charge: 0};
      const ordered = canonicalConfigOrder(clicked, schemaDoc);
      console.log(JSON.stringify({
        top: Object.keys(ordered),
        step: Object.keys(ordered.steps[0]),
        untouched_scalar: canonicalConfigOrder("raw text", schemaDoc),
      }));
    """)
    assert json.loads(out) == {
        "top": ["charge", "max_cores", "steps", "unknown_key"],
        "step": ["step", "engine", "template", "mystery"],
        "untouched_scalar": "raw text",
    }


def test_templated_option_lists_bind_selected():
    """Every ``<select>`` whose options are ``x-for``-rendered must bind ``:selected``.

    Alpine renders templated options *after* the select binds, so a plain ``:value``
    on the select loses the race and the box displays its first option while the state
    says otherwise — the engine dropdown showed ``mlip`` for an ``orca`` step exactly
    this way. Checked structurally so the next select added cannot repeat it.
    """
    html = INDEX.read_text(encoding="utf-8")
    for block in re.findall(r"<select\b.*?</select>", html, re.DOTALL):
        if "x-for" not in block:
            continue
        assert ":selected" in block, f"templated select without :selected:\n{block[:200]}"


def test_the_validation_base_is_a_directory_even_for_a_bare_filename():
    """`chemrefine gui input.yaml` must validate against the config's own directory.

    `lastIndexOf("/")` is -1 for a bare filename, and slicing to -1 drops a character
    instead of yielding a directory — so the base became "input.yam", every relative
    path resolved under a directory that does not exist, and the report warned that
    templates were missing while they sat right beside the config. The launch argument
    is the route that carries a bare name: paths chosen through Save… are absolute.
    """
    out = _run_in_node("""
      console.log(JSON.stringify(["/home/u/proj/input.yaml", "sub/input.yaml",
                                  "input.yaml", "/input.yaml"].map(parentDir)));
    """)
    assert json.loads(out) == ["/home/u/proj", "sub", ".", "/"]
