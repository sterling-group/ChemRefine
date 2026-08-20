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


@pytest.mark.parametrize("filename", [*OURS, "vendor/alpine.min.js", "vendor/js-yaml.min.js"])
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

    Everything else the builder offers is schema-driven — engines, `operation:`, every
    option and NMS field come from `/api/bootstrap`. These three do not, and each is a
    place a future change can be made in Python alone and go silently missing from the UI:
    a fourth `SampleConfig` variant is a mypy error in `filtering._dispatch` and a
    `test_docs_drift` failure, but the page would simply never offer it, and a config that
    named it in the YAML pane would render *no* knobs at all (`sampleFields` → `[]`).

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
