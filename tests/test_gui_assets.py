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

import itertools
import json
import math
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


def _label_const(name: str) -> float:
    """One of app.js's label constants, read out of app.js.

    Restating them here is what made every one of them a second place to edit — the label
    sizes are eyed against a running viewer and moved more than once, and a test that
    hardcodes the old number fails for the wrong reason and gets "fixed" by copying the new
    one in. Read, they simply follow.
    """
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    match = re.search(rf"^const {name} = ([\d.]+);", source, re.MULTILINE)
    assert match, f"app.js has no `const {name} = …`"
    return float(match.group(1))


# The 3Dmol surface drawLabels() drives, as one stub rather than six hand-rolled ones.
# `atoms` is what the model reports; every label made is captured in `labels`. Sprite scale
# and position are recorded per write, because the whole point of _syncLabels() is that it
# writes them on every camera change without rebuilding anything.
_VIEWER_STUB = """
  const labels = [];
  const makeSprite = () => ({
    material: {},
    scale: { x: 1, y: 1, set(a, b) { this.x = a; this.y = b; } },
    position: { x: 0, y: 0, z: 0, set(a, b, c) { this.x = a; this.y = b; this.z = c; } },
  });
  // measureText needs a live 2D context, which node has not got, so the advances are read
  // out of the font instead: these are NotoSans-Bold's, in ems, which is what fontconfig
  // answers for `sans-serif:bold` on Linux. A flat per-character width would not do — the
  // whole point of the width budget is that "H10" is far wider than "111", and H is the
  // widest glyph the numbering can emit. The bundle then adds 2*padding and truncates:
  // `v = g + 2.5*l + 2*o` with borderThickness forced to 0, then `canvas.width = v`.
  const ADVANCE = { C: 0.637, H: 0.765, O: 0.687, N: 0.593 };   // digits fall through to 0.572
  const canvasFor = (text, fontSize) => ({
    width: Math.floor([...text].reduce((w, c) => w + (ADVANCE[c] ?? 0.572), 0) * fontSize + 4),
    height: 1.25 * fontSize + 4,
  });
  // What is actually attached to the scene, which is the thing removeAllLabels() iterates
  // `labels` to clear — so a sprite added back without being re-listed is unreachable.
  const scene = new Set();
  const viewerStub = (atoms, calls = []) => {
    const model = {
      selectedAtoms: () => atoms,
      // Every element the pane can meet is either in 3Dmol's vdW table or falls back to
      // its default sphere radius; 0.425 is carbon at sphere scale 0.25.
      getRadiusFromStyle: (atom, style) => ({ C: 1.7, H: 1.2, O: 1.52 }[atom.elem] ?? 1.5)
                                            * style.scale,
    };
    let view = [0, 0, 0, -150, 0, 0, 0, 1];
    return {
      model,
      labels,
      gl: {
        HEIGHT: 400, fov: 20, CAMERA_Z: 150,
        getView: () => view,
        setView(v) { view = v; },
        // The bundle keeps every Label on the viewer as `this.labels` and removeAllLabels
        // splices that array without freeing anything — which is why discardLabels() has to
        // dispose them itself, and why the stub has to expose both halves to be able to
        // check that it does.
        labels,
        // The bundle's own shape, because the bug lives in it: detach every sprite, empty
        // the list, THEN show() — and show() fires the view-change callback, which lands
        // back in the app while its own bookkeeping may not have been cleared yet.
        removeAllLabels() {
          for (const l of labels) scene.delete(l.sprite);
          labels.length = 0;
          calls.push("clear");
          this.show();
        },
        addUnitCell() { calls.push("cell"); },
        addLabel(text, style) {
          calls.push("label:" + text);
          const sprite = makeSprite();
          scene.add(sprite);
          const label = {
            text,
            style,
            stylespec: style,
            sprite,
            canvas: canvasFor(text, style.fontSize),
            disposed: false,
            rasterised: 1,
            dispose() { this.disposed = true; calls.push("dispose:" + this.text); },
            // The bundle's setContext() re-measures the text at the new fontSize, resizes
            // the canvas, builds a fresh material and Texture, and resets the sprite's
            // scale and position — every one of which the caller has to cope with.
            setContext() {
              this.rasterised++;
              this.disposed = false;
              this.canvas = canvasFor(this.text, this.stylespec.fontSize);
              this.sprite.material = {};
              this.sprite.scale.set(1, 1);
              this.sprite.position.set(0, 0, 0);
              calls.push("raster:" + this.stylespec.fontSize);
            },
          };
          labels.push(label);
          return label;
        },
        show() { calls.push("show"); if (this.onView) this.onView(); },
        render() { calls.push("render"); },
        setViewChangeCallback(fn) { this.onView = fn; },
        // `upscale` holds this at two or more whatever the display is, which is what the
        // label texture is really being stretched across.
        getRenderer: () => ({ devicePixelRatio: 2 }),
        modelGroup: { add: (s) => scene.add(s), remove: (s) => scene.delete(s) },
        scene,
        stopAnimate(){}, removeAllModels(){}, setStyle(){}, zoomTo(){}, resize(){},
      },
    };
  };
"""


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
    licences = {
        p.name.upper() for p in (STATIC / "vendor").iterdir() if "LICENSE" in p.name.upper()
    }
    # Paired to its own bundle, not counted against the total: equal counts are satisfied by
    # a fourth bundle arriving beside a second licence for one of the first three, which is
    # precisely the redistribution gap the check exists to close. The convention is the
    # bundle's basename, uppercased, plus -LICENSE (the extension tracks upstream).
    missing = [
        name
        for name in VENDORED
        for stem in [Path(name).name.removesuffix(".min.js").upper()]
        if not any(licence.startswith(f"{stem}-LICENSE") for licence in licences)
    ]
    assert missing == [], (
        f"vendored with no licence of their own: {missing} (have {sorted(licences)})"
    )
    # And nothing orphaned the other way: a licence whose bundle has been removed.
    stems = {Path(name).name.removesuffix(".min.js").upper() for name in VENDORED}
    orphans = [lic for lic in licences if lic.split("-LICENSE")[0] not in stems]
    assert orphans == [], f"licence files for bundles that are gone: {sorted(orphans)}"


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
      b.runTarget = "3";
      b.cfg.steps.splice(0, 1);              // delete step 1; 3 becomes 2
      b.renumber();
      console.log(JSON.stringify({
        results: b.resultsStep, viewer: b.viewer.step, target: b.runTarget,
      }));
    """)
    # Three selectors name a step by number, and the Run panel's target was the last to be
    # added and the longest without this. Left set, it disables Run and Resume — they refuse
    # a target — against a step that no longer exists, and the four targeted actions submit
    # a number that now means a different step.
    assert json.loads(out) == {"results": "", "viewer": "input", "target": ""}


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
      // Awaited, because pickFile's open branch loads. Unawaited, `savedPath is None`
      // below was true of every mode including open — it just had not happened *yet* —
      // so the assertion that "neither mode loads anything" proved nothing at all.
      await b.pickFile({ name: "seeds.xyz", path: "/p/seeds.xyz", dir: false });
      const afterPick = { picked, savedPath: b.savedPath };
      b.browse = { ...b.browse, open: true, mode: "save", onPick: null };
      await b.pickFile({ name: "old.yaml", path: "/p/old.yaml", dir: false });
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
      // Only the geometry requests: the pane also asks /api/structure-list to fill its
      // combo boxes, and this case is about what the *structure* request carries.
      console.log(JSON.stringify(asked.filter((u) => u.startsWith("/api/structure?"))));
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
      b._gl = { stopAnimate(){}, removeAllModels(){}, labels: [], removeAllLabels(){},
                addModel: () => ({ vibrate(){}, addPropertyLabels(){} }),
                setStyle(){}, addUnitCell(){}, mapAtomProperties(){}, zoomTo(){},
                animate(){}, render(){} };

      b.chooseViewerStep("1");
      b.viewer.modeIndex = "6";
      await b.showStructure();                     // the .out is gone: refused
      const afterFailure = { mode: b.viewer.modeIndex, note: b.viewer.note };

      refuse = false;                              // the very next Show must work
      await b.showStructure();
      console.log(JSON.stringify({ afterFailure, note: b.viewer.note,
        asked: asked.filter((u) => u.startsWith("/api/structure?")) }));
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
      b.runTarget = "2";
      let stopped = 0, cleared = 0;
      b._gl = { stopAnimate: () => { stopped++; }, removeAllModels: () => { cleared++; },
                labels: [], removeAllLabels: () => {}, render: () => {} };
      await b.loadConfigFrom("/p/input.yaml", { reason: "open" });
      console.log(JSON.stringify({
        runStatus: b.runStatus, runFailures: b.runFailures, runResults: b.runResults, calls,
        resultsStep: b.resultsStep, runTarget: b.runTarget, report: b.report,
        tmplOpen: b.tmpl.open,
        viewer: b.viewer, stopped, cleared,
      }));
    """
    )
    result = json.loads(out)
    # The previous workflow's status is gone, and the new file's has been asked for rather
    # than the panel left blank: nulling it alone emptied an expanded Run panel and only its
    # own @toggle or the Refresh button ever refilled it.
    assert result["runStatus"] != {"state": "failed", "steps": [{"step": 1}]}
    assert "POST /api/status" in result["calls"]
    assert result["runFailures"] != {"count": 3}
    assert result["runResults"] is None
    assert result["resultsStep"] == ""
    assert result["runTarget"] == ""  # a step number that meant something in the old workflow
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
      // And the cached rejection is dropped, so the next open still gets to try. The
      // handle being null is that fact; a truthiness test on loadViewerLib() is not —
      // it returns a Promise either way, retained rejection included.
      const dropped = b._glLib === null;
      const firstTag = tag;
      b.loadViewerLib();
      const retried = tag !== firstTag && tag !== null;  // a fresh script tag went out
      console.log(JSON.stringify({ noted, dropped, retried }));
    """)
    result = json.loads(out)
    assert "3D viewer could not start" in result["noted"]
    assert "defined nothing" in result["noted"]
    assert result["dropped"] is True
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


_LIST_HARNESS = """
  const b = builder();
  b.chatAvailability = () => {};
  b.savedPath = "/p/input.yaml";
  b.mountViewer = async () => {};
  const rows = [
    { id: "0", modes: { "6": -820.38, "7": 1464.97, "11": 3765.42 }, imaginary: [6] },
    { id: "1", modes: {}, imaginary: [] },
    { id: "2", modes: null, imaginary: [] },
  ];
  b.api = async (m, u) =>
    u.startsWith("/api/structure-list") ? { step: 1, structures: rows } : null;
"""


def test_the_mode_list_names_each_mode_by_its_frequency():
    """ "Mode 6" answers nothing; -820.4 cm-1 (imaginary) is what you came to look at.

    The list is the point of the change: a mode number had to be guessed, and a guess that
    named a mode the step never computed was a request that could only fail.
    """
    out = _run_component_in_node(
        _LIST_HARNESS
        + """
      await b.loadStructureList();
      console.log(JSON.stringify({ choices: b.modeChoices(),
                                   hint: b.structureHint(b.viewer.structures[0]) }));
    """
    )
    result = json.loads(out)
    # Sorted by index numerically, not by the string order the JSON object happens to carry.
    assert [row["index"] for row in result["choices"]] == [6, 7, 11]
    assert result["choices"][0]["label"] == "-820.4 cm⁻¹ (imaginary)"
    assert result["choices"][1]["label"] == "1465.0 cm⁻¹"
    assert result["hint"] == "3 modes, 1 imaginary"


def test_the_modes_offered_belong_to_the_structure_that_would_be_shown():
    """Blank means "the first one", which is what Show picks — so those are its modes.

    Offering structure 0's modes while Show would draw structure 2 is the same class of
    mismatch as the sticky mode index: a number that is valid somewhere and not here.
    """
    out = _run_component_in_node(
        _LIST_HARNESS
        + """
      await b.loadStructureList();
      const blank = b.modeChoices().length;              // falls back to the first row
      b.viewer.structureId = "1";                        // ran, no frequencies
      const none = b.modeChoices().length;
      b.viewer.structureId = "2";                        // cached before the table existed
      const unknown = b.modeChoices().length;
      b.viewer.structureId = "nosuch";                   // typed, matches nothing
      const typed = b.modeChoices().length;
      console.log(JSON.stringify({ blank, none, unknown, typed,
                                   hints: b.viewer.structures.map((r) => b.structureHint(r)) }));
    """
    )
    result = json.loads(out)
    assert result["blank"] == 3  # structure "0"'s modes, which is what Show would draw
    assert (result["none"], result["unknown"], result["typed"]) == (0, 0, 0)
    assert result["hints"] == ["3 modes, 1 imaginary", "no modes", "no frequency data"]


def test_a_list_that_arrives_after_you_moved_on_is_dropped():
    """The answer belongs to the step that asked for it.

    A slow list for step 1 landing after a switch to step 2 would offer step 1's ids under
    step 2's name — ids that are valid somewhere, which is the failure mode this whole
    change exists to stop.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.chatAvailability = () => {};
      b.savedPath = "/p/input.yaml";
      b.mountViewer = async () => {};
      let release;
      const held = new Promise((r) => { release = r; });
      const fresh = { step: 2, structures: [{ id: "b" }] };
      b.api = async (m, u) => (u.includes("step=1") ? held : fresh);
      b.viewer.step = "1";
      const slow = b.loadStructureList();
      b.viewer.step = "2";                       // the user moves on while it is in flight
      await b.loadStructureList();               // step 2's answer lands first
      release({ step: 1, structures: [{ id: "a" }] });
      await slow;
      console.log(JSON.stringify(b.viewer.structures.map((r) => r.id)));
    """)
    assert json.loads(out) == ["b"]  # never ["a"]


def test_switching_steps_asks_again_and_offers_nothing_in_the_meantime():
    """Stale ids are worse than none: the boxes stay typable either way."""
    out = _run_component_in_node(
        _LIST_HARNESS
        + """
      await b.loadStructureList();
      const before = b.viewer.structures.length;
      b.api = async () => null;                  // step 2 has not run
      b.chooseViewerStep("2");
      const cleared = b.viewer.structures.length;
      await new Promise((r) => setTimeout(r, 0));
      console.log(JSON.stringify({ before, cleared, after: b.viewer.structures.length }));
    """
    )
    result = json.loads(out)
    assert result["before"] == 3
    assert result["cleared"] == 0  # dropped the moment the step changed, not when the reply came
    assert result["after"] == 0


def test_a_step_that_has_not_run_is_not_announced_on_every_switch():
    """The lists are a convenience; both boxes stay typable without them.

    Driven through the *real* ``api``, with only ``fetch`` stubbed: stubbing ``api`` itself
    is what a previous version of this did, and it never reached the branch it named — the
    quiet flag lives inside the method the stub replaced.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.chatAvailability = () => {};
      b.savedPath = "/p/input.yaml";
      global.fetch = async () => ({
        ok: false, status: 400,
        json: async () => ({ error: "step 2 has no cached results yet" }),
      });
      await b.loadStructureList();
      const quiet = b.flash;
      await b.validateNow();                     // an ordinary call, same failing fetch
      console.log(JSON.stringify({ quiet, loud: b.flash }));
    """)
    result = json.loads(out)
    assert result["quiet"] == ""  # switching steps must not shout about it
    # The suppression is per-call, not a hole in the error path: everything else still says so.
    assert "no cached results" in result["loud"]


_RUN_HARNESS = """
  const b = builder();
  b.chatAvailability = () => {};
  b.savedPath = "/p/input.yaml";
  b.cfg = { steps: [{ step: 1 }, { step: 2 }] };
  b.refreshStatus = async () => {};
  const posted = [];
  const asked = [];
  b.api = async (m, u, body) => { posted.push(body); return { pid: 1, log: "/l" }; };
  global.window = { ...global.window, confirm: (text) => { asked.push(text); return true; } };
"""


def test_every_recovery_action_is_reachable_and_described_as_itself():
    """Three of six were offered, so half the recovery vocabulary needed a terminal.

    The blurb used to be a ternary with no default arm, so anything past its two named
    actions was confirmed as "re-attempt failed jobs" — which rebuild-cache, that submits
    nothing at all, is not. A confirmation that misdescribes what it confirms is worse than
    none, so each action's own sentence is asserted here.
    """
    out = _run_component_in_node(
        _RUN_HARNESS
        + """
      for (const a of ["run", "resume", "rerun", "rerun-errors",
                       "rebuild-cache", "rebuild-nms"]) {
        await b.launch(a);
      }
      console.log(JSON.stringify({ posted, asked }));
    """
    )
    result = json.loads(out)
    assert [p["action"] for p in result["posted"]] == [
        "run",
        "resume",
        "rerun",
        "rerun-errors",
        "rebuild-cache",
        "rebuild-nms",
    ]
    # Every dialog describes its own action, and no two share a sentence.
    blurbs = [text.split("This will ")[1] for text in result["asked"]]
    assert len(set(blurbs)) == 6
    assert "submits nothing" in blurbs[4]  # rebuild-cache, which attempts no jobs
    assert "normal-mode resolution" in blurbs[5]
    assert not any("undefined" in text for text in result["asked"])


def test_the_two_whole_pipeline_actions_never_carry_a_step():
    """``start_run`` refuses a target for run/resume, so sending one is a 400 after a click.

    The page says so before the click instead: the buttons are disabled while a step is
    chosen, and the payload carries null even if one is reached another way.
    """
    out = _run_component_in_node(
        _RUN_HARNESS
        + """
      b.runTarget = "2";
      await b.launch("rerun");        // takes one
      await b.launch("run");          // does not, even with the selection standing
      console.log(JSON.stringify({
        posted, asked,
        takes: ["run", "resume", "rerun", "rerun-errors", "rebuild-cache", "rebuild-nms"]
                 .map((a) => b.takesTarget(a)),
      }));
    """
    )
    result = json.loads(out)
    assert result["posted"][0]["target"] == "2"
    assert result["posted"][1]["target"] is None
    assert result["takes"] == [False, False, True, True, True, True]
    # And the dialog names the step, so a targeted action cannot be confirmed blind.
    assert "step 2 of /p/input.yaml" in result["asked"][0]
    assert "on /p/input.yaml" in result["asked"][1]


def test_declining_the_dialog_launches_nothing():
    """The gate is the whole point: real compute on the user's machine."""
    out = _run_component_in_node(
        _RUN_HARNESS
        + """
      global.window.confirm = () => false;
      await b.launch("run");
      console.log(JSON.stringify({ posted: posted.length, flash: b.flash }));
    """
    )
    assert json.loads(out) == {"posted": 0, "flash": ""}


def test_a_pasted_absolute_name_replaces_the_directory_rather_than_hanging_off_it():
    """`${dir}/${name}` is a string join, not a path join.

    The Save… filename box is free text, so pasting a full path produced
    `/home/u//abs/path` — a directory named "" and a file nobody meant — and a `~/…` was
    joined instead of expanded. Absolute is absolute, in this box as in every shell.
    """
    out = _run_component_in_node("""
      const cases = [
        ["/home/u", "input.yaml"],
        ["/home/u", "/abs/path/input.yaml"],
        ["/home/u", "~/projects/input.yaml"],
        ["/", "input.yaml"],
      ];
      console.log(JSON.stringify(cases.map(([d, n]) => joinPath(d, n))));
    """)
    assert json.loads(out) == [
        "/home/u/input.yaml",
        "/abs/path/input.yaml",  # not /home/u//abs/path/input.yaml
        "~/projects/input.yaml",  # the server expanduser()s it; joining would defeat that
        "/input.yaml",  # and no doubled slash at the root either
    ]


def test_a_typed_path_navigates_when_it_is_a_directory_and_is_taken_when_it_is_not():
    """The one path field on the page that could not be typed into.

    A cluster tree's interesting directory is eight levels down, and the modal offered
    only clicking. The server already knows which a path is — /api/browse lists a
    directory and refuses anything else — so the box needs no guess of its own.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.chatAvailability = () => {};
      const opened = [];
      b.openConfigFrom = async (p) => { opened.push(p); return true; };
      // Exact, not a prefix: "/deep/dir/input.yaml" encodes to a string that *contains*
      // the directory's own encoding, so a loose stub would answer for the file too.
      b.api = async (m, u) =>
        u.endsWith("path=%2Fdeep%2Fdir")
          ? { path: "/deep/dir", parent: "/deep", entries: [] }
          : null;                                  // not a directory

      b.browse = { ...b.browse, open: true, mode: "open" };
      b.browse.typed = "  /deep/dir  ";            // trimmed, then navigated
      await b.goToTyped();
      const walked = { path: b.browse.path, typed: b.browse.typed, opened: opened.length };

      b.browse.typed = "/deep/dir/input.yaml";     // a file: taken, not navigated
      await b.goToTyped();

      b.browse.typed = "";                         // nothing typed, nothing done
      await b.goToTyped();
      console.log(JSON.stringify({ walked, opened }));
    """)
    result = json.loads(out)
    assert result["walked"]["path"] == "/deep/dir"
    # The box shows where you are afterwards, so it is a thing to edit, not to retype.
    assert result["walked"]["typed"] == "/deep/dir"
    assert result["walked"]["opened"] == 0
    assert result["opened"] == ["/deep/dir/input.yaml"]  # once, and not for the blank box


def test_the_typed_path_does_the_other_two_jobs_in_the_other_two_modes():
    """One box, the same three jobs the listing's rows have — not an open-only shortcut."""
    out = _run_component_in_node("""
      const b = builder();
      b.chatAvailability = () => {};
      b.api = async () => null;                    // never a directory here
      let picked = null;
      b.browse = { ...b.browse, mode: "pick", onPick: (p) => { picked = p; }, path: "/here" };
      b.browse.typed = "/seeds/mol.xyz";
      await b.goToTyped();
      const afterPick = { picked, closed: !b.browse.open };
      b.browse = { ...b.browse, open: true, mode: "save", onPick: null, path: "/here" };
      b.browse.typed = "/elsewhere/other.yaml";
      await b.goToTyped();
      const typedAbsolute = b.browse.filename;
      // Where it would actually be written, which is the thing that matters.
      let wrote = null;
      // Only /api/save carries a body; goToTyped's own /api/browse probe is a GET, and must
      // still answer "not a directory" so the typed path is taken as a file.
      b.api = async (m, u, body) => {
        if (!body) return null;
        wrote = body.path;
        return { path: body.path };
      };
      await b.saveTo();
      // And a bare name still lands in the directory being browsed.
      b.browse = { ...b.browse, open: true, mode: "save", path: "/here" };
      b.browse.typed = "notes.yaml";
      await b.goToTyped();
      await b.saveTo();
      console.log(JSON.stringify({ afterPick, typedAbsolute, wrote, bare: wrote }));
    """)
    result = json.loads(out)
    assert result["afterPick"] == {"picked": "/seeds/mol.xyz", "closed": True}
    # The whole typed path reaches the filename box, not its basename. Stripping it there
    # took the decision away from joinPath(), which exists precisely so an absolute name
    # replaces the directory — so typing an absolute path in save mode quietly wrote the
    # file into whatever directory the listing happened to be showing.
    assert result["typedAbsolute"] == "/elsewhere/other.yaml"
    assert result["bare"] == "/here/notes.yaml"  # and a bare name still joins as before


def test_every_modal_starts_where_the_open_file_lives():
    """Only Open… did; Save… and the field pickers always started at $HOME.

    With a config open eight levels down a cluster tree, starting at $HOME is the same
    eight levels of clicking every time.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.chatAvailability = () => {};
      const asked = [];
      b.api = async (m, u) => { asked.push(u); return { path: "/x", parent: "/", entries: [] }; };
      b.savedPath = "/deep/tree/project/input.yaml";
      await b.openConfig();
      await b.openSave();
      await b.openBrowseWith("Pick a folder", () => {});
      const withFile = asked.slice();
      b.savedPath = null;                          // nothing open: the server picks $HOME
      asked.length = 0;
      await b.openSave();
      console.log(JSON.stringify({ withFile, without: asked }));
    """)
    result = json.loads(out)
    assert len(result["withFile"]) == 3
    for url in result["withFile"]:
        assert "path=%2Fdeep%2Ftree%2Fproject" in url
    assert result["without"] == ["/api/browse"]  # no path at all, which is $HOME


def test_recent_workflows_are_paths_and_only_paths():
    """A convenience stored in a browser profile, which outlives the session.

    Paths only: file contents in localStorage would be a copy of the user's work sitting
    in a store this page cannot promise anything about. Newest first, deduplicated, capped.
    """
    out = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      const seen = [];
      for (const p of ["/a/x.yaml", "/b/y.yaml", "/a/x.yaml"]) {
        b.api = async (m, u) =>
          u.startsWith("/api/load") ? { path: p, yaml_text: "steps: []\\n" }
                                    : { config: { steps: [] }, yaml_text: "" };
        await b.loadConfigFrom(p, { reason: "open" });
        seen.push([...b.recents]);
      }
      console.log(JSON.stringify({
        seen, stored: JSON.parse(localStorage.getItem("cr-recents")),
        keys: Object.keys(localStorage._d), short: shortPath("/a/b/c/d/input.yaml"),
      }));
    """
    )
    result = json.loads(out)
    assert result["seen"][-1] == ["/a/x.yaml", "/b/y.yaml"]  # newest first, no duplicate
    assert result["stored"] == ["/a/x.yaml", "/b/y.yaml"]
    # Nothing but the path list — no yaml text anywhere in this browser's store.
    assert result["keys"] == ["cr-recents"]
    assert result["short"] == "…/d/input.yaml"  # the directory that names it, and the file


def test_a_corrupt_recents_entry_cannot_blank_the_page():
    """It is a browser profile: hand-edited, upgraded across builds, shared with old ones.

    An uncaught throw inside ``builder()`` is not a broken list — it is a page that never
    renders, with every button gone and no message.
    """
    out = _run_component_in_node("""
      const rows = [];
      for (const raw of ['{"not": "a list"}', 'not json at all', '["/a.yaml", 7, null]']) {
        localStorage.setItem("cr-recents", raw);
        rows.push(builder().recents);
      }
      console.log(JSON.stringify(rows));
    """)
    assert json.loads(out) == [[], [], ["/a.yaml"]]  # never a throw, never a non-string


def test_every_3dmol_call_names_a_method_the_vendored_bundle_has():
    """The viewer's API is reached through hand-written stubs in these tests.

    A stub answers whatever it is asked, so a misspelt 3Dmol call passes every case here
    and then does nothing in a browser — silent at click time, which is the exact failure
    this module exists to close for Alpine bindings. The bundle is the authority: every
    method app.js calls on the viewer or the model must appear in it.
    """
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    bundle = (STATIC / "vendor" / "3dmol.min.js").read_text(encoding="utf-8")
    called = set(re.findall(r"this\._(?:gl|model)\.(\w+)\(", source))
    assert len(called) >= 8, (
        f"the scanner found almost nothing — has the spelling changed? {called}"
    )
    # Minified, so a method is `name(` in a class body: the name has to occur followed by
    # an open paren somewhere in the bundle. Loose on purpose — this catches typos and
    # removed methods, and a false pass costs nothing that the stubs did not already cost.
    missing = sorted(name for name in called if f"{name}(" not in bundle)
    assert missing == [], f"3Dmol has no such method(s): {missing}"


def test_the_five_numbering_modes_read_the_way_each_convention_does():
    """Chemcraft's "Labels on atoms" vocabulary, because it is the one people already have.

    Its menu is Clear labels / Show atoms seq. number / Show types+numbers in group / Show
    atoms types, and each numbering convention is somebody's default: ChemRefine's own tools
    report file order (``analyze_mode``'s ``top_atoms[].index``), papers and most GUIs count
    from one, and a spectroscopist reads per-element ordinals. Off by one is how the atom
    being discussed stops being the atom on screen.
    """
    out = _run_component_in_node("""
      const atoms = [
        { serial: 0, elem: "C" }, { serial: 1, elem: "H" },
        { serial: 2, elem: "H" }, { serial: 3, elem: "O" },
      ];
      const render = (mode) => { const c = {}; return atoms.map((a) => atomLabel(a, mode, c)); };
      console.log(JSON.stringify({
        off: render("off"), zero: render("zero"), one: render("one"),
        element: render("element"), symbol: render("symbol"),
        // A second render must restart the ordinals, not carry on from the first.
        again: render("element"),
        unknownOrdinal: atomLabel({ serial: 0 }, "element", {}),
        unknownSymbol: atomLabel({ serial: 0 }, "symbol", {}),
        notAMode: render("numbers"),
      }));
    """)
    result = json.loads(out)
    assert result["off"] == [None, None, None, None]  # nothing to draw, not empty strings
    assert result["zero"] == ["0", "1", "2", "3"]
    assert result["one"] == ["1", "2", "3", "4"]
    assert result["element"] == ["C1", "H1", "H2", "O1"]
    # Deliberately not unique: it answers "what is this atom", not "which atom is this".
    assert result["symbol"] == ["C", "H", "H", "O"]
    assert result["again"] == result["element"]  # a fresh tally each redraw
    # An element-less atom is labelled, never crashed on, in both element-bearing modes.
    assert result["unknownOrdinal"] == "?1"
    assert result["unknownSymbol"] == "?"
    # A mode nothing handles must read as "no label" rather than as a label: `null` here
    # becomes the literal text "null" on every atom once 3Dmol stringifies the property.
    assert result["notAMode"] == [None, None, None, None]


def test_every_numbering_the_page_offers_is_a_numbering_the_code_draws():
    """The five blob buttons and the five branches are two lists that must not drift.

    A sixth blob whose mode string nothing handles does not render nothing — 3Dmol's
    addPropertyLabels stringifies whatever the property holds and skips only a genuinely
    absent one, so `null` reaches the canvas as the four characters "null", on every atom.
    Neither file can catch that alone.
    """
    html = INDEX.read_text(encoding="utf-8")
    offered = re.findall(r"chooseLabels\('(\w+)'\)", html)
    assert len(offered) == 5, f"expected five blobs, found {offered}"
    assert offered[0] == "off", "the default has to be first — it is the one you come back to"

    out = _run_component_in_node(f"""
      const modes = {json.dumps(offered)};
      const drawn = modes.map((m) => atomLabel({{ serial: 0, elem: "C" }}, m, {{}}));
      console.log(JSON.stringify(drawn));
    """)
    drawn = json.loads(out)
    # Only `off` draws nothing; every other blob must produce actual text.
    assert drawn[0] is None
    assert all(isinstance(label, str) and label for label in drawn[1:]), dict(
        zip(offered, drawn, strict=True)
    )


def test_turning_labels_off_does_not_take_the_unit_cell_with_them():
    """``removeAllLabels()`` removes the a/b/c corner labels ``addUnitCell`` adds.

    Verified against the vendored bundle, which calls ``addLabel`` three times inside
    ``addUnitCell``. So clearing atom numbering would quietly strip a periodic structure's
    box — which is why the cell is re-added on every relabel rather than once at draw time.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      const calls = [];
      const stub = viewerStub([{ serial: 0, elem: "C", x: 0, y: 0, z: 0 }], calls);
      b._model = stub.model;
      b._gl = stub.gl;
      // A zoom at which the starting texture is already the right resolution, so nothing
      // re-rasterises and the sequence below is the label lifecycle alone. The resolution
      // work has its own test; mixing the two here made this one fail for its reasons.
      stub.gl.setView([0, 0, 0, 113, 0, 0, 0, 1]);
      b.viewer.labels = "one";
      b.drawLabels();
      const on = calls.splice(0);
      b.viewer.labels = "off";
      b.drawLabels();
      console.log(JSON.stringify({ on, off: calls }));
    """
    )
    result = json.loads(out)
    # Cleared, then the cell back, then the atom labelled — in that order.
    # The "show" is removeAllLabels()'s own: it detaches the sprites, empties its list
    # and then renders — which is the re-entrancy the bookkeeping clear must precede.
    assert result["on"] == ["clear", "show", "cell", "label:1", "render"]
    # And with numbering off the cell is still re-added; only the atom labels are skipped.
    # The dispose lands after the clear, never before it: removeAllLabels() detaches the
    # sprites and renders once, and only then is deleting the textures they drew with safe.
    assert result["off"] == ["clear", "show", "dispose:1", "cell", "render"]


def test_each_label_owns_its_own_position_rather_than_sharing_one():
    """``addPropertyLabels`` hands every label the same stylespec object.

    It deep-copies the spec once, then rewrites ``.position`` per atom in a loop — and
    ``Label`` keeps that object by reference (``this.stylespec = t || {}``), re-reading
    ``.position`` from it whenever ``setContext()`` runs again. So every label made that way
    is holding the LAST atom's coordinates, and the whole numbering collapses onto one atom
    the moment a lost WebGL context comes back. The explicit loop gives each its own.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      const atoms = [
        { serial: 0, elem: "C", x: 0, y: 0, z: 0 },
        { serial: 1, elem: "O", x: 1.2, y: 0, z: 0 },
        { serial: 2, elem: "H", x: -1.1, y: 0.9, z: 0 },
      ];
      const stub = viewerStub(atoms);
      b._model = stub.model;
      b._gl = stub.gl;
      b.viewer.labels = "element";
      b.drawLabels();
      console.log(JSON.stringify({
        texts: stub.labels.map((l) => l.text),
        anchors: stub.labels.map((l) => l.style.position),
        shared: stub.labels[0].style === stub.labels[1].style,
      }));
    """
    )
    result = json.loads(out)
    assert result["texts"] == ["C1", "O1", "H1"]
    # Each label's spec carries its own atom, not the last one three times over.
    assert result["anchors"] == [
        {"x": 0, "y": 0, "z": 0},
        {"x": 1.2, "y": 0, "z": 0},
        {"x": -1.1, "y": 0.9, "z": 0},
    ]
    assert result["shared"] is False


def test_the_label_style_is_bold_centred_and_behind_the_atoms_in_front_of_it():
    """Every value here was read out of the vendored bundle, and every one is pinned.

    ``bold`` is the only thing in that build that changes glyph weight — ``fontWeight``,
    ``fontStyle`` and ``strokeText`` do not occur in it at all — and it is a bare truthiness
    check, so it has to be a real boolean rather than a string the way its neighbours are.

    ``alignment`` matters more than it looks. Its default is ``topLeft``, i.e. ``(1, -1)``:
    half the label's own width and height from the atom, in *screen* pixels, at every zoom.
    Worse, an unrecognised alignment string also centres, because the renderer coerces the
    missing vector's components to zero — so a typo would look right and a later correct
    value could too. The literal is asserted rather than the rendering for exactly that
    reason.

    ``inFront`` becomes the material's ``depthTest``, inverted: ``false`` is what lets an
    atom in front hide the number behind it. ``true`` — which is what this drew before —
    makes every label float over the whole molecule whatever its depth.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      const stub = viewerStub([{ serial: 0, elem: "C", x: 0, y: 0, z: 0 }]);
      b._model = stub.model;
      b._gl = stub.gl;
      b.viewer.labels = "one";
      b.drawLabels();
      console.log(JSON.stringify({
        style: stub.labels[0].style,
        depthWrite: stub.labels[0].sprite.material.depthWrite,
      }));
    """
    )
    result = json.loads(out)
    style = result["style"]
    assert style["bold"] is True  # a boolean: `bold: "false"` renders bold in this build
    assert style["alignment"] == "center"
    assert style["inFront"] is False  # depth-tested, so atoms in front occlude it
    # The bundle defaults glyphs to WHITE — `Ge(e.fontColor, e.fontOpacity, {r:255,g:255,
    # b:255,a:1})` — so dropping this key leaves them invisible on a light background.
    assert style["fontColor"] == "black"
    # No box: it read badly, and with the glyphs now atom-sized it is not what carries them.
    assert style["showBackground"] is False
    # The bundle aliases borderOpacity onto the background's own colour object, so setting
    # it alone silently changes the fill's alpha. No border key belongs here.
    assert not any(key.startswith("border") for key in style)
    # Not a stylespec key, so it is written onto the material: without it every antialiased
    # glyph edge writes depth and bites a hole in whatever draws after it.
    assert result["depthWrite"] is False


def test_a_label_is_pushed_along_its_own_ray_so_it_stays_on_its_atom_at_every_angle():
    """The push must follow each atom's line to the eye, not one shared view axis.

    Depth-testing a label at the nucleus makes the atom punch a hole through its own number:
    all four corners of the sprite carry ONE depth — the anchor's — because the shader adds
    the quad offset after its own perspective divide, while the sphere writes its front
    *surface* depth, nearer by a full radius. So the anchor is pushed toward the camera by
    that atom's own radius (0.425 A for carbon at sphere scale 0.25) plus a clearance.

    *Which* direction is the part that has to be exact. Pushing every anchor along the same
    view axis leaves an atom's x and y untouched in eye space and only shortens its depth —
    and the shader divides by w, so a shorter depth means a bigger radius on screen. The
    label is thrown outward from the centre of the pane: nothing at all on the view axis,
    most at the edge, and swinging around as the structure turns. Pushing along the atom's
    own ray to the camera instead moves the anchor along the line the eye already collapses
    to a point, so the projected position does not move at all — which is why the label sits
    exactly on its atom here rather than approximately on it.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      const atoms = [
        { serial: 0, elem: "C", x: 0, y: 0, z: 0 },
        { serial: 1, elem: "H", x: 2, y: 0, z: 0 },
      ];
      const stub = viewerStub(atoms);
      b._model = stub.model;
      b._gl = stub.gl;
      b.viewer.labels = "one";
      b.drawLabels();                              // identity view: camera looks down +z
      // Snapshot, not a reference: _syncLabels() writes into these very objects, so
      // reading them after the rotation below would compare the rotation with itself.
      const snap = () => stub.labels.map((l) => ({ ...l.sprite.position }));
      const facing = snap();
      // Turn the molecule 180 degrees about y: "toward the camera" is now -z.
      stub.gl.setView([0, 0, 0, -150, 0, 1, 0, 0]);
      b._syncLabels();
      const turned = snap();
      // And a quarter turn about y, where the push is along -x — the case that catches a
      // formula right in z and wrong everywhere else.
      const s = 0.7071067811865476;
      stub.gl.setView([0, 0, 0, -150, 0, s, 0, s]);
      b._syncLabels();
      const quarter = snap();
      // And a quarter turn about x, which is the only one of the four that puts the push
      // on +y — so all three components have now been wrong-able independently.
      stub.gl.setView([0, 0, 0, -150, s, 0, 0, s]);
      b._syncLabels();
      console.log(JSON.stringify({ facing, turned, quarter, tipped: snap() }));
    """
    )
    result = json.loads(out)
    carbon, hydrogen = result["facing"]
    # Carbon: 1.7 vdW x 0.25 + 0.15 clearance. Hydrogen: 1.2 x 0.25 + 0.15. Its own radius,
    # so the smaller atom is pushed less and stays easier for a neighbour to hide.
    # The stub's view puts the camera 300 A away, so an atom on the axis is pushed straight
    # along +z and the distance is the whole of it.
    assert carbon["z"] == pytest.approx(0.575)
    assert (carbon["x"], carbon["y"]) == (0, 0)  # on the axis: the ray IS the axis
    # The hydrogen is 2 A off-axis, so its ray leans: it is pushed very slightly back toward
    # the axis as well as forward. That lean is the entire fix — an axis push would leave x
    # at exactly 2 and throw the label outward on screen instead.
    assert hydrogen["z"] == pytest.approx(0.45, rel=1e-4)
    assert hydrogen["x"] < 2  # leaning toward the eye, not straight along +z
    assert hydrogen["x"] == pytest.approx(2 - 0.45 * 2 / 300.00667, rel=1e-4)
    # The push length is the atom's own radius plus the clearance, whatever the direction.
    assert math.dist((hydrogen["x"], hydrogen["y"], hydrogen["z"]), (2, 0, 0)) == pytest.approx(
        0.45
    )
    # Rotated to face the other way, the push follows the camera rather than staying on +z.
    assert result["turned"][0]["z"] == pytest.approx(-0.575)
    # A quarter turn puts it on -x, so every component of the direction is exercised: a
    # formula correct in z and wrong in x and y reads fine until the structure is turned.
    assert result["quarter"][0]["x"] == pytest.approx(-0.575)
    assert result["quarter"][0]["z"] == pytest.approx(0, abs=1e-9)
    assert result["tipped"][0]["y"] == pytest.approx(0.575)
    assert result["tipped"][0]["z"] == pytest.approx(0, abs=1e-9)
    # Every label, at every one of the four orientations, is exactly its own push away from
    # the atom it names — the invariant the ray push guarantees and the axis push did not.
    for state in ("facing", "turned", "quarter", "tipped"):
        for label, atom, push in zip(
            result[state], ((0, 0, 0), (2, 0, 0)), (0.575, 0.45), strict=True
        ):
            assert math.dist((label["x"], label["y"], label["z"]), atom) == pytest.approx(push)


def test_label_size_tracks_the_zoom_so_it_stays_the_size_of_its_atom():
    """A sprite is a fixed size in screen pixels; the molecule is not.

    The vertex shader adds the quad offset after its own perspective divide, so nothing in
    the stylespec makes a label grow as you zoom in — it is the molecule shrinking around a
    number that does not that reads as the number drifting loose. The one live multiplier is
    `sprite.scale`, which the sprite plugin re-reads every frame, so the size is recomputed
    from the camera rather than the label rebuilt: no canvas, no texture upload.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      const stub = viewerStub([{ serial: 0, elem: "C", x: 0, y: 0, z: 0 }]);
      b._model = stub.model;
      b._gl = stub.gl;
      b.viewer.labels = "one";
      b.drawLabels();
      const sprite = stub.labels[0].sprite;
      // The drawn ink in CSS pixels — scale times the texels the glyphs occupy. Stated
      // this way it is invariant under re-rasterisation, which changes both factors and
      // must leave their product alone; the raw scale factor is not.
      const at = (zoom) => {
        stub.gl.setView([0, 0, 0, zoom, 0, 0, 0, 1]);
        b._syncLabels();
        return sprite.scale.y * 0.72 * stub.labels[0].stylespec.fontSize;
      };
      console.log(JSON.stringify({
        // distance = CAMERA_Z - zoom, so bigger is closer: 10 A, 20 A, 50 A away.
        near: at(140), mid: at(130), far: at(100),
        // Way past the band a 32 px texture survives, both ways.
        clampedIn: at(148), clampedOut: at(-100000),
      }));
    """
    )
    result = json.loads(out)
    # Closer camera, bigger number — the whole point, and monotonic in between.
    assert result["near"] > result["mid"] > result["far"]
    # Clamped at both ends, in CSS pixels of drawn ink — LABEL_INK_MAX_PX and
    # LABEL_INK_MIN_PX. Bounding the ink rather than the scale factor is what makes these
    # survive a change of texture resolution, which multiples of the texture's own height
    # did not: the texture is re-rasterised as the zoom moves, and the drawn size must not
    # move with it.
    assert result["clampedIn"] == pytest.approx(_label_const("LABEL_INK_MAX_PX"))
    assert result["clampedOut"] == pytest.approx(_label_const("LABEL_INK_MIN_PX"))


def test_a_wide_label_is_scaled_down_so_it_never_outgrows_the_atom_it_names():
    """Height alone cannot bound a label, because the quad's width is the text's.

    Every label's texture is the same 44 px tall — ``1.25 * fontSize + 2 * padding``, which
    has no term for the string — while its width is ``measureText(text) + 2 * padding``. So
    one scale factor driven by height alone drew "H1" 0.66 A across and "H10" 1.11 A across:
    the same nominal size, and the second one nearly two hydrogen blobs wide, sprawling over
    the bonds either side of the atom it belongs to. The width budget is what stops that, and
    it must bind only on the labels that need it — a one- or two-character label is already
    inside its budget and must not be shrunk to pay for a three-character one.

    The legibility floor outranks it, and that is deliberate: zoomed far enough out that
    meeting the width budget would put the ink under ``LABEL_INK_MIN_PX``, the label stays
    readable and overruns the budget instead. An unreadable label is not a smaller label,
    it is a missing one.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      // `element` mode emits the widest strings the page can produce, and a two-digit
      // ordinal on the widest glyph — H — is what makes them widest.
      const atoms = Array.from({ length: 10 }, (_, i) => (
        { serial: i, elem: "H", x: i * 3, y: 0, z: 0 }));
      const stub = viewerStub(atoms);
      b._model = stub.model;
      b._gl = stub.gl;
      b.viewer.labels = "element";
      b.drawLabels();
      const perAngstrom = (d) => 400 / (2 * d * Math.tan(Math.PI / 360 * 20));
      const at = (zoom) => {
        stub.gl.setView([0, 0, 0, zoom, 0, 0, 0, 1]);
        b._syncLabels();
        const perA = perAngstrom(150 - zoom);
        return stub.labels.map((l) => ({
          text: l.text,
          inkHigh: (l.sprite.scale.y * 0.72 * l.stylespec.fontSize) / perA,
          inkWide: l.sprite.scale.x * (l.canvas.width - 4) / perA,
        }));
      };
      console.log(JSON.stringify({ working: at(130), farOut: at(96) }));
    """
    )
    result = json.loads(out)
    ink = _label_const("LABEL_INK_HEIGHT")
    width = _label_const("LABEL_INK_WIDTH")
    floor = _label_const("LABEL_INK_MIN_PX")
    rows = result["working"]  # 20 A away: clear of both clamps, so the budgets decide
    short = [r for r in rows if len(r["text"]) == 2]  # H1 .. H9
    wide = [r for r in rows if len(r["text"]) == 3]  # H10
    assert len(short) == 9 and len(wide) == 1, [r["text"] for r in rows]
    # No label's ink is wider than one carbon blob, 2 * 1.7 * 0.25 A — the width budget.
    for row in rows:
        assert row["inkWide"] < width + 1e-9, row
    # The narrow ones are still sized by height, untouched by the budget: a hydrogen blob is
    # 0.6 A across and their ink stands half that tall, so the number reads smaller than the
    # smallest atom rather than sitting on it like a lid.
    for row in short:
        assert row["inkHigh"] == pytest.approx(ink)
        assert row["inkWide"] < width
    # And the wide one paid for its width by getting shorter, not by pushing the others out.
    assert wide[0]["inkWide"] == pytest.approx(width)
    assert wide[0]["inkHigh"] < ink
    # Zoomed out to 54 A the wide label is the first to reach the floor, and there it is
    # allowed past its width budget rather than shrunk out of legibility — 8 CSS px of ink,
    # the same as every other label, instead of the 7.2 px the budget alone would have asked
    # for. The short ones are still above the floor and still sized by height.
    far = result["farOut"]
    per_angstrom_far = 400 / (2 * 54 * math.tan(math.radians(10)))
    far_wide = next(r for r in far if len(r["text"]) == 3)
    # The wide label is the first to reach it, because the width budget was already asking
    # for less height than the height budget was. Between the two thresholds — a narrow
    # window, and the only place the ordering is observable — it is floored while its
    # shorter neighbours are not.
    assert far_wide["inkHigh"] * per_angstrom_far == pytest.approx(floor)
    assert far_wide["inkWide"] > width
    for row in far:
        if len(row["text"]) == 2:
            assert row["inkHigh"] == pytest.approx(ink)
            assert row["inkHigh"] * per_angstrom_far > floor


def test_discarding_labels_leaves_nothing_stranded_in_the_scene():
    """The viewer's list and the scene must agree, or a sprite becomes unreachable.

    ``removeAllLabels()`` detaches every sprite, empties its own list, and *then* renders —
    and that render fires the view-change callback, which lands back in ``_syncLabels()``.
    With the app's own bookkeeping not yet cleared, that sync runs against labels the viewer
    has just stopped listing, and a re-rasterise inside it ends on ``modelGroup.add()``:
    putting back a sprite nothing can ever take out again, because ``removeAllLabels()``
    iterates the list this one is no longer in.

    The result is the reported one — the previous molecule's numbers standing beside the new
    one, at the smallest size the code can draw, a fresh set every time a workflow is opened.
    Pinned as an invariant rather than as the symptom: after discarding, the scene holds
    exactly the sprites the viewer still lists.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      const atoms = [
        { serial: 0, elem: "C", x: 0, y: 0, z: 0 },
        { serial: 1, elem: "H", x: 1.1, y: 0, z: 0 },
      ];
      const stub = viewerStub(atoms);
      // Through the real mount, so the view-change callback is wired the way the page wires
      // it — that callback is the re-entrancy, and without it none of this can happen.
      b.loadViewerLib = async () => ({ createViewer: () => stub.gl });
      global.document = { getElementById: () => ({}) };
      b.savedPath = "/p/input.yaml";
      await b.mountViewer({ force: true });
      b._model = stub.model;
      // An ordinary window and an ordinary zoom, which is what makes the first draw
      // rasterise well above the starting 32 texels. At the pane's 260 px minimum it stays
      // at 32, the resolution band is never left, and none of this can happen — which is
      // why the fault looks intermittent rather than constant.
      stub.gl.HEIGHT = 620;
      stub.gl.setView([0, 0, 0, 130, 0, 0, 0, 1]);
      b.viewer.labels = "one";
      b.drawLabels();
      const drawn = { scene: stub.gl.scene.size, listed: stub.labels.length,
                      font: stub.labels[0].stylespec.fontSize };

      // The pane is hidden, so the viewer measures zero and its resize skips the render
      // that would refresh the key — leaving the app's idea of the camera stale. That is
      // what makes the re-entrant sync do work rather than return early.
      stub.gl.HEIGHT = 0;
      b.discardLabels();
      console.log(JSON.stringify({
        drawn, after: { scene: stub.gl.scene.size, listed: stub.labels.length },
      }));
    """
    )
    result = json.loads(out)
    assert result["drawn"]["scene"] == 2 and result["drawn"]["listed"] == 2
    # The precondition: the texture has been rasterised above its starting size, so the
    # re-entrant sync below has a reason to rebuild rather than to return unchanged.
    assert result["drawn"]["font"] > 40
    # Nothing listed, and nothing left behind: a sprite in the scene that the viewer does
    # not list can never be removed, re-sized or re-pushed again.
    assert result["after"] == {"scene": 0, "listed": 0}


def test_a_label_is_sized_by_its_own_atoms_depth_not_the_molecules():
    """The spheres are perspective-projected and the labels were not.

    A sprite carries no depth term at all — its shader adds the quad's corners after its own
    perspective divide — so a size computed once for the whole scene made every label the
    same on screen while the atoms around them were not. A hydrogen at the front of an
    8 A-deep molecule is drawn 1.4x the size of an identical hydrogen at the back, so one
    label sat inside its blob and the other overflowed it. Sized on each atom's own depth the
    ratio is constant instead, at every depth and every zoom.

    The depth is the atom's, not the pushed anchor's: the sphere it is matched against is
    itself drawn from its centre's depth, and the push is element-dependent, so pushing into
    the size would make an H and a C at one depth differ for no physical reason.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      // Three identical hydrogens, differing only in depth: 4 A in front of the rotation
      // centre, on it, and 4 A behind. Identity view, so +z is straight at the camera.
      const atoms = [
        { serial: 0, elem: "H", x: 0, y: 0, z: 4 },
        { serial: 1, elem: "H", x: 0, y: 0, z: 0 },
        { serial: 2, elem: "H", x: 0, y: 0, z: -4 },
      ];
      const stub = viewerStub(atoms);
      b._model = stub.model;
      b._gl = stub.gl;
      b.viewer.labels = "one";
      b.drawLabels();
      const at = (zoom) => {
        stub.gl.setView([0, 0, 0, zoom, 0, 0, 0, 1]);
        b._syncLabels();
        const centre = 150 - zoom;
        return stub.labels.map((l, i) => {
          const depth = centre - atoms[i].z;
          const perA = 400 / (2 * depth * Math.tan(Math.PI / 360 * 20));
          const inkPx = l.sprite.scale.y * 0.72 * l.stylespec.fontSize;
          // The blob this label has to sit inside, in the same CSS pixels: an H sphere is
          // 2 * 1.2 * 0.25 A across, projected from that atom's own depth.
          const blobPx = 0.6 * perA;
          return { depth, ratio: inkPx / blobPx, visible: l.sprite.visible };
        });
      };
      console.log(JSON.stringify({ fitted: at(122), closer: at(140) }));
    """
    )
    result = json.loads(out)
    for stage in ("fitted", "closer"):
        rows = result[stage]
        assert [round(r["depth"], 3) for r in rows] == sorted(
            [round(r["depth"], 3) for r in rows]
        ), rows
        # Every label the same fraction of its own atom, front to back. Before this, the
        # same three labels were the same size as each other while their atoms were not.
        expected = _label_const("LABEL_INK_HEIGHT") / 0.6  # the H blob it is measured in
        for row in rows:
            assert row["visible"] is True
            assert row["ratio"] == pytest.approx(expected, rel=1e-9), (stage, row)


def test_a_label_hides_when_its_own_atom_goes_behind_the_camera():
    """Hidden, not clamped, and hidden exactly when the atom is.

    Both a sprite and a sphere are clipped whole rather than sliced — each forces all four
    corners of its quad to one depth — and the camera's near plane is held at 1 or more
    unconditionally, so a label at a depth below that is drawing nothing either way. Clamping
    instead would hand the sizing a depth nobody can see, pin the scale to its ceiling and
    drive a full texture rebuild for an invisible label, then another on the way back.

    Reachable in ordinary use rather than theoretical: the zoom limit bounds the rotation
    centre alone, and each wheel notch multiplies the distance by about 0.68, so a handful of
    them from the default zoom puts a front atom behind the camera.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      const atoms = [
        { serial: 0, elem: "H", x: 0, y: 0, z: 4 },    // the near one
        { serial: 1, elem: "H", x: 0, y: 0, z: -4 },   // still well in front of the camera
      ];
      const stub = viewerStub(atoms);
      b._model = stub.model;
      b._gl = stub.gl;
      b.viewer.labels = "one";
      b.drawLabels();
      const at = (centre) => {
        stub.gl.setView([0, 0, 0, 150 - centre, 0, 0, 0, 1]);
        b._syncLabels();
        return stub.labels.map((l) => ({
          visible: l.sprite.visible, raster: l.rasterised,
        }));
      };
      const wide = at(20);              // both far in front of the camera
      const tight = at(4.5);            // the near atom is now 0.5 A behind the camera plane
      const backOff = at(20);           // and it comes back
      console.log(JSON.stringify({ wide, tight, backOff }));
    """
    )
    result = json.loads(out)
    assert [r["visible"] for r in result["wide"]] == [True, True]
    # The near atom has crossed the camera plane; its own sphere is gone and so is its number.
    assert [r["visible"] for r in result["tight"]] == [False, True]
    assert [r["visible"] for r in result["backOff"]] == [True, True]
    # And it was not re-rasterised while it was invisible — that is the cost a clamp would
    # have paid, twice, for a label nobody could see.
    assert result["tight"][0]["raster"] == result["wide"][0]["raster"]


def test_a_label_is_redrawn_at_the_resolution_it_is_being_displayed_at():
    """The one reason a number is ever softer than the atom beside it.

    A sphere is solved per pixel by its fragment shader, so it has no resolution of its own
    and is exactly as sharp as the framebuffer at any zoom. A label is a canvas rasterised
    into a texture and sampled with a filter this build forces to LINEAR, with no mipmaps —
    so it has a fixed number of texels, and drawing it larger than those interpolates. A
    fixed fontSize therefore cannot be sharp at more than one zoom, and the size that would
    be right spans more than an order of magnitude across pane sizes and structures.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      const calls = [];
      const stub = viewerStub([{ serial: 0, elem: "C", x: 0, y: 0, z: 0 }], calls);
      b._model = stub.model;
      b._gl = stub.gl;
      b.viewer.labels = "one";
      b.drawLabels();
      const label = stub.labels[0];
      calls.length = 0;
      // Device pixels per texel, which is 1 for a label as sharp as the geometry. The stub's
      // renderer reports devicePixelRatio 2, as `upscale` guarantees on any display.
      const density = () => label.sprite.scale.y * 2;
      const at = (zoom) => {
        stub.gl.setView([0, 0, 0, zoom, 0, 0, 0, 1]);
        b._syncLabels();
        return { font: label.stylespec.fontSize, density: density(),
                 raster: label.rasterised, depthWrite: label.sprite.material.depthWrite };
      };
      console.log(JSON.stringify({
        start: { font: label.stylespec.fontSize, density: density() },
        // 8 A, 4 A, 350 A and 20 A from the camera.
        zoomedIn: at(142), further: at(146), zoomedOut: at(-200), back: at(130),
        calls,
      }));
    """
    )
    result = json.loads(out)
    for stage in ("zoomedIn", "further", "zoomedOut", "back"):
        row = result[stage]
        # About one texel per device pixel at any zoom — that is the whole claim. Under it a
        # label is mildly soft; over it the glyphs go blocky. The band may only be left when
        # the rasterisation bound is what stopped it: texture memory is finite, so at the
        # extreme of zoom-in the ceiling binds and the label goes soft rather than the page
        # holding a 4000 px texture per atom. Stated with the escape rather than without it,
        # because a bare band assertion would be a claim this cannot keep.
        at_bound = row["font"] in (16, 192)
        assert at_bound or 0.55 - 1e-9 <= row["density"] <= 1.3 + 1e-9, (stage, row)
        # The material is rebuilt by setContext(), so the one property that is written
        # onto it rather than into the stylespec has to be written again — or every
        # antialiased glyph edge starts writing depth and biting holes in what follows.
        assert row["depthWrite"] is False, (stage, row)
    # Zooming in raised the resolution; zooming out lowered it again rather than holding a
    # texture far larger than anything on screen needs.
    assert result["further"]["font"] > result["zoomedIn"]["font"] > result["start"]["font"]
    assert result["zoomedOut"]["font"] < result["further"]["font"]
    # And it is not re-rasterising on every camera move — the band is wide enough that an
    # ordinary zoom crosses it a couple of times, not continuously.
    assert result["back"]["raster"] <= 6
    # Every rebuild frees the texture it replaces BEFORE building the next one. setContext()
    # makes a fresh canvas, material and Texture and frees none of them, so a rebuild that
    # skipped the dispose would leak a GL texture per label per rebuild — invisible until a
    # session has zoomed around a few structures.
    freed = [c for c in result["calls"] if c.startswith(("dispose:", "raster:"))]
    assert freed, "no rebuild happened at all"
    assert len(freed) % 2 == 0
    for disposal, rebuild in zip(freed[0::2], freed[1::2], strict=True):
        assert disposal.startswith("dispose:"), freed
        assert rebuild.startswith("raster:"), freed


def test_panning_re_pushes_the_labels_because_the_ray_moved():
    """The view key may not omit ``modelGroup.position`` now that the push follows a ray.

    Leaving ``view[0..2]`` out was right while every anchor was pushed along one shared view
    axis: panning slides the model and the camera together, so it cannot turn that axis. A
    ray to the camera is a different matter — panning changes where the camera is relative to
    each atom, so a pan that returned early would leave every label pushed along the ray it
    wanted before the pan, which is the same drift the ray push exists to remove.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      const stub = viewerStub([{ serial: 0, elem: "C", x: 4, y: 0, z: 0 }]);
      b._model = stub.model;
      b._gl = stub.gl;
      b.viewer.labels = "one";
      b.drawLabels();
      const before = { ...stub.labels[0].sprite.position };
      // A pan writes modelGroup.position, which getView() reports as view[0..2]; the
      // quaternion and the zoom are untouched.
      stub.gl.setView([6, 0, 0, -150, 0, 0, 0, 1]);
      const resynced = b._syncLabels();
      console.log(JSON.stringify(
        { resynced, before, after: { ...stub.labels[0].sprite.position } }));
    """
    )
    result = json.loads(out)
    assert result["resynced"] is True, "a pan left the labels on their pre-pan rays"
    # The atom has not moved in model coordinates, so the label must still be exactly its
    # own push from it — but along a different ray, so at a different point.
    assert result["before"] != result["after"]
    for state in ("before", "after"):
        pos = result[state]
        assert math.dist((pos["x"], pos["y"], pos["z"]), (4, 0, 0)) == pytest.approx(0.575)


def test_the_labels_are_resized_once_per_camera_move_and_not_once_per_show():
    """``show()`` is called by far more than camera moves, and each one re-enters the hook.

    ``removeAllLabels``, ``addLabel``, ``addUnitCell``, ``setBackgroundColor`` and ``resize``
    all call ``show()``, and the view-change callback fires from inside it. Without the view
    key every one of those would walk all the labels; without the re-entrancy guard the
    second ``show()`` this issues would call the callback again, for ever.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      const stub = viewerStub([{ serial: 0, elem: "C", x: 0, y: 0, z: 0 }]);
      b._model = stub.model;
      b._gl = stub.gl;
      b.viewer.labels = "one";
      b.drawLabels();
      const first = b._syncLabels();               // the camera has not moved since the draw
      stub.gl.setView([0, 0, 0, -80, 0, 0, 0, 1]);
      const moved = b._syncLabels();
      const again = b._syncLabels();               // same camera twice: no work owed
      console.log(JSON.stringify({ first, moved, again }));
    """
    )
    assert json.loads(out) == {"first": False, "moved": True, "again": False}


def test_the_viewer_is_built_without_fxaa_but_keeps_its_supersampling():
    """``antialias`` is not the geometry-smoothing flag it reads as, and it hurt the glyphs.

    The bundle renders into its own framebuffer and blits it with one of two shaders —
    ``this._antialias ? screenaa : screen`` — and ``screenaa`` is an FXAA pass. FXAA finds
    high-contrast edges and redistributes pixels along them, which is the wrong treatment
    for black bold glyphs and is what made the numbers look uneven. The labels are inside
    that pass: the depth-tested sprites are drawn before the framebuffer reaches the screen.

    Both keys are passed explicitly because the two defaults are chained — GLViewer defaults
    ``antialias`` to true, and the renderer then defaults ``upscale`` to whatever
    ``antialias`` is. Dropping one silently drops the other, and ``upscale`` is the one worth
    keeping: it holds the backing store at two device pixels per CSS pixel, which is the
    supersampling that actually smooths the spheres and sticks.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.chatAvailability = () => {};
      let options = null;
      b.loadViewerLib = async () => ({
        createViewer: (host, opts) => {
          options = opts;
          return { resize(){}, render(){}, setViewChangeCallback(){} };
        },
      });
      global.document = { getElementById: () => ({}) };
      b.savedPath = "/p/input.yaml";
      await b.mountViewer({ force: true });
      console.log(JSON.stringify(options));
    """)
    options = json.loads(out)
    assert options["antialias"] is False  # the FXAA blit, which smears the glyph edges
    assert options["upscale"] is True  # the >=2x backing store, which is the real smoothing
    assert options["backgroundColor"] == "white"


def test_the_view_change_hook_redraws_once_and_cannot_recurse():
    """The callback fires *after* the render it belongs to, so what it writes is a frame late.

    That is why it issues a second ``show()`` — and why it must guard, because that ``show()``
    re-enters the callback. Unguarded this is an unbounded recursion on every mouse move.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      const stub = viewerStub([{ serial: 0, elem: "C", x: 0, y: 0, z: 0 }]);
      let shows = 0, entries = 0;
      stub.gl.show = () => { shows++; stub.gl.onView(); };   // as the real show() does
      b.loadViewerLib = async () => ({ createViewer: () => stub.gl });
      global.document = { getElementById: () => ({}) };
      b.savedPath = "/p/input.yaml";
      await b.mountViewer({ force: true });
      b._model = stub.model;
      b.viewer.labels = "one";
      b.drawLabels();
      // Count how many times the callback's body actually runs, not just how many redraws
      // it causes: the guard's whole job is to stop the show() it issues from re-entering
      // it, and the view-key check would mask an unbounded recursion as merely one extra.
      const real = b._syncLabels.bind(b);
      b._syncLabels = () => { entries++; return real(); };
      const before = shows;
      stub.gl.setView([0, 0, 0, -80, 0, 0, 0, 1]);
      stub.gl.onView();                            // the camera moved
      const afterMove = shows - before;
      const entriesForOneMove = entries;
      stub.gl.onView();                            // fires again on the same camera
      console.log(JSON.stringify({ afterMove, afterIdle: shows - before, entriesForOneMove }));
    """
    )
    result = json.loads(out)
    # Exactly one extra redraw for the move: the labels' new size needs a frame to appear.
    assert result["afterMove"] == 1
    # And a callback on an unmoved camera owes nothing, so it costs no redraw at all.
    assert result["afterIdle"] == 1
    # One entry, not two: the show() above re-enters the callback and the guard turns it
    # back at the door. Without it that nesting is bounded only by the view-key check.
    assert result["entriesForOneMove"] == 1


def test_choosing_a_numbering_selects_exactly_one_and_redraws_nothing_else():
    """Radio behaviour: clicking one blob is what unselects the previous one.

    And it must call ``drawLabels()`` alone. ``drawLabels`` returns at once when nothing is
    drawn, so choosing a numbering on an empty pane costs nothing; routing it through
    ``mountViewer`` would fetch half a megabyte of viewer in order to label no atoms.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.chatAvailability = () => {};
      let drew = 0, mounted = 0;
      b.drawLabels = () => { drew++; };
      b.mountViewer = async () => { mounted++; };
      const seen = [b.viewer.labels];
      // The last two repeat "zero": re-clicking the lit blob must LEAVE it lit. A toggle
      // would clear it here, which is not what "clicking one unselects the old one" means.
      for (const mode of ["element", "zero", "zero", "off"]) {
        b.chooseLabels(mode);
        seen.push(b.viewer.labels);
      }
      console.log(JSON.stringify({ seen, drew, mounted }));
    """)
    result = json.loads(out)
    # One value at a time, so the markup's :class comparison can only ever light one blob.
    assert result["seen"] == ["off", "element", "zero", "zero", "off"]
    assert result["drew"] == 4  # every choice redraws
    assert result["mounted"] == 0  # and none of them fetches the bundle


def test_each_blob_wears_the_label_the_mode_it_selects_would_draw():
    """The blobs are self-describing only if the face and the mode agree — nothing else says so.

    A `0` blob wired to `chooseLabels('one')` reads as file order and turns on one-based
    numbering: the user's spec inverted, in the one control whose whole point is that you can
    see what it will do without reading anything. Three bindings per blob must name the same
    mode — the click, the highlight, and what a screen reader is told — and the face has to be
    what that mode actually draws.
    """
    html = INDEX.read_text(encoding="utf-8")
    row = html[html.index('class="blobs"') : html.index("</fieldset>")]
    blobs = re.findall(r"<button\b(.*?)>(.*?)</button>", row, re.DOTALL)
    assert len(blobs) == 5, f"expected five blobs, parsed {len(blobs)}"

    wired = []
    for attrs, face in blobs:
        clicks = re.findall(r"chooseLabels\('(\w+)'\)", attrs)
        highlights = re.findall(r":class=\"viewer\.labels === '(\w+)'", attrs)
        pressed = re.findall(r":aria-pressed=\"viewer\.labels === '(\w+)'\"", attrs)
        assert len(clicks) == len(highlights) == len(pressed) == 1, attrs
        # All three name the same mode, or the lit blob is not the one that acted.
        assert clicks == highlights == pressed, f"blob bindings disagree: {attrs}"
        wired.append((clicks[0], face.strip()))

    modes = [mode for mode, _ in wired]
    assert modes == ["off", "zero", "one", "element", "symbol"]
    assert len(set(modes)) == 5  # five buttons, five modes — no mode wired twice

    # And the face each blob wears is exactly what its mode draws on the first carbon of a
    # C H H O molecule. That is what makes the row readable without a legend.
    out = _run_component_in_node(f"""
      const modes = {json.dumps(modes)};
      const first = {{ serial: 0, elem: "C" }};
      console.log(JSON.stringify(modes.map((m) => atomLabel(first, m, {{}}) ?? "")));
    """)
    assert [face for _, face in wired] == json.loads(out)


def test_a_numbering_chosen_before_anything_is_drawn_applies_when_a_file_arrives():
    """The whole point of taking the control out of the gate, end to end.

    The blobs are now visible with an empty canvas, so choosing one there has to mean
    something later. It does because every draw path ends in drawLabels() — but nothing
    asserted that, so a draw path that forgot it would silently ignore the choice.
    """
    out = _run_component_in_node(
        _VIEWER_STUB
        + """
      const b = builder();
      b.chatAvailability = () => {};
      b.savedPath = null;                          // no workflow whatsoever
      const stub = viewerStub([{ serial: 0, elem: "C", x: 0, y: 0, z: 0 }]);
      const drawn = stub.labels;
      const gl = stub.gl;
      gl.addModel = () => stub.model;
      b.loadViewerLib = async () => ({ createViewer: () => gl });
      global.document = { getElementById: () => ({}) };

      b.chooseLabels("one");                       // chosen with nothing on screen
      const beforeAnyDraw = drawn.length;
      await b.mountViewerFor("1\\nc\\nC 0 0 0\\n");   // now a dropped file arrives
      console.log(JSON.stringify({ beforeAnyDraw, drawn: drawn.length, labels: b.viewer.labels }));
    """
    )
    result = json.loads(out)
    assert result["beforeAnyDraw"] == 0  # nothing to label yet, and nothing pretended there was
    assert result["labels"] == "one"  # the choice survived having nowhere to apply
    assert result["drawn"] == 1  # and it applied the moment there was something to apply it to


def test_clearing_the_canvas_takes_the_labels_choice_with_no_structure_to_apply_it_to():
    """Open… blanks the canvas; a blob click after it must not resurrect what was cleared.

    ``drawLabels()`` guards on ``_model``, and ``forgetPreviousWorkflow()`` dropped the
    models without dropping the handle to them — so choosing a numbering on the blank pane
    re-labelled, and for a periodic file re-boxed, the structure that had just been removed.
    """
    out = _run_component_in_node(
        _RELOAD_HARNESS
        + """
      const drawn = [];
      b._model = { addPropertyLabels: () => drawn.push("labelled") };
      b._gl = {
        stopAnimate(){}, labels: [], removeAllLabels(){}, removeAllModels(){}, render(){},
        addUnitCell: () => drawn.push("boxed"),
        mapAtomProperties: (fn) => [{ serial: 0, elem: "C", properties: {} }].forEach(fn),
      };
      await b.loadConfigFrom("/p/input.yaml", { reason: "open" });
      const modelAfter = b._model;
      b.chooseLabels("element");                   // the user picks a numbering on the blank pane
      console.log(JSON.stringify({ modelAfter, drawn, labels: b.viewer.labels }));
    """
    )
    result = json.loads(out)
    assert result["modelAfter"] is None  # the handle went with the models it pointed at
    assert result["drawn"] == []  # nothing relabelled, nothing re-boxed
    assert result["labels"] == "element"  # but the choice is remembered for the next structure


def test_the_numbering_control_is_not_behind_the_saved_workflow_gate():
    """Numbering acts on whatever is drawn, and a dropped file needs no workflow.

    The control was a dropdown inside ``x-show="!staticMode && savedPath"``, so opening a
    single file gave you a structure you could not number — the pane's one view-only
    operation, hidden by the one condition that has nothing to do with it. Checked the same
    structural way the canvas is, because a comment saying so would not hold.
    """
    html = INDEX.read_text(encoding="utf-8")
    # From the start of the control's OWN tag, and including it: a gate written on the
    # fieldset hides the blobs just as surely as one on an ancestor, and it can sit on
    # either side of the `class="blobs"` this finds the element by.
    at = html.rindex("<", 0, html.index('class="blobs"'))
    open_tags: list[str] = [html[at : html.index(">", at) + 1]]
    # Every container this page nests with, not `div` alone — `<section>`, `<details>` and
    # `<fieldset>` are all in use here, and a gate on any of them hides what it wraps.
    names = "div|section|details|fieldset|template|main|p"
    container = rf"<({names})\b([^>]*)>|</(?:{names})>"
    for match in re.finditer(container, html[:at]):
        if match.group(0).startswith("</"):
            if open_tags:
                open_tags.pop()
        else:
            open_tags.append(match.group(2))
    gated = [attrs for attrs in open_tags if "savedPath" in attrs]
    assert gated == [], f"the numbering blobs sit inside a savedPath-gated element: {gated}"


def test_labels_are_dropped_before_the_model_they_are_attached_to():
    """A label is positioned on an atom object, so it outlives the model unless removed.

    Both paths that drop the models — showing the next structure, and opening another
    workflow — must clear labels first, or the previous structure's numbering floats over
    whatever is drawn next.

    ``discardLabels()`` rather than ``removeAllLabels()`` directly, because the bundle's
    method splices its array and frees nothing: the Label, its material, its GL texture and
    its backing canvas all survive it, and ``_atomLabels`` on this side survives it too. A
    call site that reached past the wrapper would leak both.
    """
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    drops = source.count("this._gl.removeAllModels()")
    assert drops >= 2, f"expected both drop sites; found {drops}"
    # Immediately before, not merely somewhere earlier in the file: a prefix search would be
    # satisfied by the wrapper's own definition and prove nothing about the call site being
    # checked. Comment lines between the two are allowed, nothing else.
    paired = re.compile(
        r"this\.discardLabels\(\);\n(?:\s*//[^\n]*\n)*\s*this\._gl\.removeAllModels\(\)"
    )
    assert len(paired.findall(source)) == drops, (
        "a removeAllModels() without discardLabels() immediately before it"
    )
    # And nothing may call the bundle's own clear except that one wrapper.
    assert source.count("this._gl.removeAllLabels()") == 1, (
        "removeAllLabels() belongs to discardLabels() alone — it frees nothing by itself"
    )


def test_the_step_dropdowns_mark_the_chosen_row_across_the_type_boundary():
    """A step number is a number in the config and a string out of the DOM.

    Three selects need this and each had its own copy. `===` on the raw values is false
    for the row that is actually chosen, so the box displays the first option while the
    state says otherwise — the same class of mismatch the templated-select guard exists
    for, one level in.
    """
    out = _run_component_in_node("""
      const steps = [{ step: 1 }, { step: 2 }, { step: 10 }];
      console.log(JSON.stringify({
        fromDom: stepOptions(steps, "2").map((o) => o.selected),
        fromCfg: stepOptions(steps, 2).map((o) => o.selected),
        sentinel: stepOptions(steps, "").map((o) => o.selected),
        labels: stepOptions(steps, 1).map((o) => o.label),
        empty: stepOptions(undefined, 1),
      }));
    """)
    result = json.loads(out)
    assert result["fromDom"] == [False, True, False]  # the string the DOM hands back
    assert result["fromCfg"] == [False, True, False]  # and the number the config holds
    # A sentinel selects no step — "" must not match step 1 through some coercion.
    assert result["sentinel"] == [False, False, False]
    assert result["labels"] == ["step 1", "step 2", "step 10"]
    assert result["empty"] == []  # a config mid-edit can have no steps at all


def test_the_playground_refuses_every_disk_backed_action_in_one_voice():
    """Three hand-written refusals had drifted into three sentences for the same fact.

    The published playground has no server behind it. What matters is that each action
    stops *and* says why in terms of that one fact, and that the same code says nothing at
    all when a server is there.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.chatAvailability = () => {};
      b.api = async () => null;
      b.savedPath = "/p/input.yaml";
      b.staticMode = true;
      const said = [];
      for (const call of ["validateNow", "showStructure", "runCheck"]) {
        b.flash = "";
        await b[call]();
        said.push(b.flash);
      }
      b.staticMode = false;
      console.log(JSON.stringify({ said, served: b.playgroundRefuses("anything") }));
    """)
    result = json.loads(out)
    assert len(result["said"]) == 3
    for message in result["said"]:
        assert "needs the local chemrefine gui" in message
        assert "chemrefine[gui]" in message
    assert len(set(result["said"])) == 3  # each still names what it was that stopped
    assert result["served"] is False  # and with a server, it refuses nothing


def test_the_one_opener_still_carries_what_each_mode_needs():
    """Three spread-and-navigate copies became one; the pick mode's callback is the risk.

    ``pick`` is the only mode with an onPick, and it is the whole of that mode's job —
    dropping it on the way through a shared opener makes the field pickers silently do
    nothing when a file is chosen.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.chatAvailability = () => {};
      b.api = async () => ({ path: "/x", parent: "/", entries: [] });
      let picked = null;
      await b.openBrowseWith("Pick template_dir", (p) => { picked = p; });
      const pick = { mode: b.browse.mode, title: b.browse.title, hasPick: !!b.browse.onPick };
      b.browse.onPick("/chosen");
      await b.openConfig();
      const open = { mode: b.browse.mode, title: b.browse.title, onPick: b.browse.onPick };
      await b.openSave();
      const save = { mode: b.browse.mode, title: b.browse.title, onPick: b.browse.onPick };
      console.log(JSON.stringify({ pick, picked, open, save, opened: b.browse.open }));
    """)
    result = json.loads(out)
    assert result["pick"] == {"mode": "pick", "title": "Pick template_dir", "hasPick": True}
    assert result["picked"] == "/chosen"  # the callback survived the shared path
    # The other two carry no callback — a stale one would fire on the next file clicked.
    assert result["open"] == {"mode": "open", "title": "Open a workflow…", "onPick": None}
    assert result["save"] == {"mode": "save", "title": "Save workflow as…", "onPick": None}
    assert result["opened"] is True


_FILE_HARNESS = """
  const b = builder();
  b.chatAvailability = () => {};
  const drawn = [];
  const gl = {
    stopAnimate(){}, labels: [], removeAllLabels(){}, removeAllModels(){},
    addModel: (text) => { drawn.push(text); return { addPropertyLabels(){} }; },
    setStyle(){}, addUnitCell(){}, mapAtomProperties(){}, zoomTo(){}, render(){}, resize(){},
  };
  // The REAL mountViewer runs, with only the bundle and the host element stubbed: it is
  // the method that refuses to mount without a saved workflow, and stubbing it out is how
  // a test stops exercising the very guard a dropped file has to get past.
  b.loadViewerLib = async () => ({ createViewer: () => gl });
  global.document = { getElementById: () => ({}) };
  const posted = [];
  b.api = async (m, u, body) => {
    posted.push([u, body]);
    return body && (body.path || body.text)
      ? { path: body.path || body.name, format: "extxyz", text: "2\\nsi\\nSi 0 0 0\\n",
          atoms: 2, formula: "Si2", periodic: true }
      : null;
  };
"""


def test_a_dropped_file_is_drawn_without_a_workflow_being_open():
    """A file dropped on the page is on the browser's machine, so contents travel, not a path.

    And nothing about the workflow moves: this is "what is in this file", not "open this
    project". The two doors look alike, and conflating them is how one silently does the
    other.
    """
    out = _run_component_in_node(
        _FILE_HARNESS
        + """
      const file = { name: "POSCAR", text: async () => "SI POSCAR TEXT" };
      b.viewer.dragging = true;                    // as a dragover would have left it
      await b.dropStructure({ dataTransfer: { files: [file] } });
      console.log(JSON.stringify({
        posted, drawn, note: b.viewer.note, savedPath: b.savedPath,
        dragging: b.viewer.dragging, step: b.viewer.step,
      }));
    """
    )
    result = json.loads(out)
    [[url, body]] = result["posted"]
    assert url == "/api/structure-file"
    assert body == {"name": "POSCAR", "text": "SI POSCAR TEXT"}  # contents, never a path
    assert result["drawn"] == ["2\nsi\nSi 0 0 0\n"]
    # The formula and count, because a file that parsed into the wrong thing looks fine.
    assert "Si2" in result["note"] and "2 atoms" in result["note"]
    assert "periodic" in result["note"]  # so a drawn box is expected, not a surprise
    # No workflow was opened: savedPath and the step selector are untouched.
    assert result["savedPath"] is None
    assert result["step"] == "input"
    assert result["dragging"] is False  # the drag state is released on drop


def test_a_drop_that_could_not_be_read_says_so_in_the_pane():
    """The pane's own line, not only #flash — the same lesson as the mode failure."""
    out = _run_component_in_node(
        _FILE_HARNESS
        + """
      b.api = async () => null;
      await b.dropStructure({ dataTransfer: { files: [{ name: "notes.txt",
                                                        text: async () => "hello" }] } });
      const failed = { note: b.viewer.note, busy: b.viewer.busy, drawn: drawn.length };
      b.viewer.dragging = true;
      await b.dropStructure({ dataTransfer: { files: [] } });   // a drag with no file
      const releasedAnyway = b.viewer.dragging;
      console.log(JSON.stringify({ failed, after: b.viewer.note, releasedAnyway }));
    """
    )
    result = json.loads(out)
    assert "notes.txt" in result["failed"]["note"]
    assert result["failed"]["busy"] is False  # released even on the failing path
    assert result["failed"]["drawn"] == 0
    assert result["after"] == result["failed"]["note"]  # an empty drop changes nothing
    # The drag highlight is released even when the drop carried nothing to read, or the
    # zone stays lit with no way to turn it off short of dragging something else over it.
    assert result["releasedAnyway"] is False


def test_opening_a_structure_file_goes_through_the_picker_not_the_workflow_door():
    """The server's own disk — the only way to reach a cluster's files from this page.

    It borrows the browse modal in `pick` mode, so a typed path works here too; what it
    must not borrow is `open` mode, which loads a workflow.
    """
    out = _run_component_in_node(
        _FILE_HARNESS
        + """
      b.api = async (m, u, body) => {
        posted.push([u, body]);
        if (u.startsWith("/api/browse")) return { path: "/x", parent: "/", entries: [] };
        return { path: body.path, format: "extxyz", text: "1\\nsi\\nSi 0 0 0\\n",
                 atoms: 1, formula: "Si", periodic: false };
      };
      await b.openStructureFile();
      const mode = b.browse.mode;
      await b.browse.onPick("/on/the/server/POSCAR");
      console.log(JSON.stringify({
        mode, title: b.browse.title, posted: posted.map((p) => p[0]),
        body: posted[posted.length - 1][1], savedPath: b.savedPath, note: b.viewer.note,
      }));
    """
    )
    result = json.loads(out)
    assert result["mode"] == "pick"  # never "open": that door loads a workflow
    assert result["title"] == "Open a structure file…"
    assert result["posted"][-1] == "/api/structure-file"
    assert result["body"] == {"path": "/on/the/server/POSCAR"}
    assert result["savedPath"] is None  # still no workflow open
    assert "Si" in result["note"]


def test_the_viewer_canvas_is_not_behind_the_saved_workflow_gate():
    """A dropped file needs no workflow, so it must not need a workflow's container either.

    ``#viewer`` lived inside ``x-show="!staticMode && savedPath"``, so with nothing saved
    the element mountViewer looks up did not exist: the file read fine on the server and
    then drew nowhere, silently, because the mount returns at its ``!host`` guard. The
    step and structure controls *do* belong behind that gate — they name things only a
    workflow has — which is why this is a structural check and not a rule about the pane.
    """
    html = INDEX.read_text(encoding="utf-8")
    at = html.index('id="viewer"')
    # Walk the open/close tags before it and keep the ones still open: any div gated on
    # savedPath among them is an ancestor that would hide the canvas.
    depth_stack: list[str] = []
    for match in re.finditer(r"<div\b([^>]*)>|</div>", html[:at]):
        if match.group(0) == "</div>":
            if depth_stack:
                depth_stack.pop()
        else:
            depth_stack.append(match.group(1))
    gated = [attrs for attrs in depth_stack if "savedPath" in attrs]
    assert gated == [], f"#viewer sits inside a savedPath-gated element: {gated}"


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


def test_every_option_widget_kind_has_an_arm():
    """Four blocks render a field from a spec, and each must handle every kind fieldSpec emits.

    A missing arm is silent in both directions: the label renders and the input does not, so
    the knob simply cannot be set and nothing says so. A catch-all is worse — it draws a
    number box for a boolean and writes 1 where true was meant.

    This test was named in a comment in ``index.html`` for some time before it existed, and
    while it did not exist the Sample block was missing its ``text`` arm and the top-level
    block was the only one of the four without ``step="any"`` on the number input — which is
    the block that renders the one float the schema has. Copying markup four times is how
    that happens; this is the cheap half of the fix.
    """
    html = INDEX.read_text(encoding="utf-8")
    # The kinds fieldSpec() can return, taken from its own docstring in forms.js rather than
    # from a list here — a new kind must break this test, not slip past it.
    forms = (STATIC / "forms.js").read_text(encoding="utf-8")
    kinds = set(re.findall(r'kind: "(\w+)"', forms))
    assert kinds == {"text", "number", "checkbox", "select"}, kinds

    # Each block is a `<template x-for="field in …">`; its arms are the `field.kind === '…'`
    # tests inside it, up to the start of the next block.
    starts = [m.start() for m in re.finditer(r'x-for="field in ', html)]
    assert len(starts) == 4, f"expected four field-rendering blocks, found {len(starts)}"
    bounds = [*starts, len(html)]
    for index, (begin, end) in enumerate(itertools.pairwise(bounds)):
        block = html[begin:end]
        arms = set(re.findall(r"field\.kind === '(\w+)'", block))
        assert arms == kinds, f"block {index} handles {sorted(arms)}, not {sorted(kinds)}"
        # And no catch-all: an arm that fires for everything else puts the wrong widget on
        # whichever kind nobody thought about.
        assert "field.kind !==" not in block, f"block {index} still has a catch-all arm"

    # Every number input takes fractional values. The schema has a float among the top-level
    # fields, and the block that renders it was the one without this.
    for match in re.finditer(r'<input type="number"([^>]*)>', html):
        attrs = match.group(1)
        if "field." in attrs:  # a spec-driven field, not a hand-written charge/multiplicity
            assert 'step="any"' in attrs, attrs


def test_the_results_pager_moves_by_the_page_it_fetches():
    """The stride was a literal in the fetch and two more in the markup, in another file.

    Change the fetch limit and the buttons page by the old number — every page after the
    first skipping rows or repeating them, with nothing anywhere to say so. One constant, and
    the markup asks for a direction rather than an offset.
    """
    out = _run_component_in_node("""
      const b = builder();
      b.chatAvailability = () => {};
      const asked = [];
      b.api = async (m, u, body) => { asked.push(body); return { rows: [], total: 99 }; };
      b.resultsStep = "1";
      await b.loadResults("1", 0);
      b.runResults = { offset: 0 };
      b.pageResults(1);
      await new Promise((r) => setTimeout(r, 0));
      b.runResults = { offset: asked[asked.length - 1].offset };
      b.pageResults(-1);
      await new Promise((r) => setTimeout(r, 0));
      console.log(JSON.stringify(asked.map((a) => ({ limit: a.limit, offset: a.offset }))));
    """)
    asked = json.loads(out)
    stride = asked[0]["limit"]
    # Forward by exactly one page, then back to where it started.
    assert asked[1]["offset"] == stride
    assert asked[2]["offset"] == 0
    # And every request asks for that same page size.
    assert {a["limit"] for a in asked} == {stride}

    # The markup no longer carries the number at all — it asks for a direction.
    html = INDEX.read_text(encoding="utf-8")
    assert "pageResults(-1)" in html and "pageResults(1)" in html
    # No page-sized arithmetic left in the markup. `runResults.offset + 1` stays — that is
    # the 1-based row counter in the "showing 21-40 of 99" label, not a stride.
    assert not re.search(r"runResults\.offset [-+] (?!1\b)\d+", html), html


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
