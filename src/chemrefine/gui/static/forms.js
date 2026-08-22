/* forms.js — normalize pydantic's JSON Schema into flat field specs the page renders.
 *
 * The server's /api/bootstrap document is generated from the validating models, so this
 * file interprets rather than invents: each property becomes
 *   { key, kind: "text"|"number"|"checkbox"|"select", options, fallback, doc, path }
 * and the Alpine templates in index.html render every field the same generic way.
 * Only shapes pydantic v2 actually emits here are handled: plain types, enums
 * (Literal), and `anyOf: [X, {type: null}]` optionals. Anything unrecognized renders
 * as free text — the server-side validator remains the authority on every value.
 */
"use strict";

/** Numeric bounds from the schema (pydantic's ge/gt/le/lt), inclusive for the input. */
function numericBounds(p) {
  let min = null;
  let max = null;
  if (typeof p.minimum === "number") min = p.minimum;
  if (typeof p.exclusiveMinimum === "number") {
    // gt: for integers the next representable value; for floats HTML min is close enough.
    min = p.type === "integer" ? p.exclusiveMinimum + 1 : p.exclusiveMinimum;
  }
  if (typeof p.maximum === "number") max = p.maximum;
  if (typeof p.exclusiveMaximum === "number") {
    max = p.type === "integer" ? p.exclusiveMaximum - 1 : p.exclusiveMaximum;
  }
  return { min, max };
}

/** One property schema -> a field spec (null = not renderable, e.g. steps/objects). */
function fieldSpec(key, prop) {
  let p = prop || {};
  // Optional[X] arrives as anyOf [X, null] — unwrap to X.
  if (Array.isArray(p.anyOf)) {
    const real = p.anyOf.filter((alt) => alt.type !== "null");
    if (real.length !== 1) return null;
    p = { ...real[0], description: p.description, default: prop.default };
  }
  if (p.$ref || p.type === "object" || p.type === "array") return null;
  const doc = (p.description || "").split("\n\n")[0];
  const fallback = prop.default === undefined || prop.default === null ? "" : String(prop.default);
  if (Array.isArray(p.enum)) {
    return {
      key,
      kind: "select",
      options: p.enum,
      fallback,
      doc,
      path: false,
      min: null,
      max: null,
      fallbackNum: null,
    };
  }
  if (p.type === "boolean") {
    return {
      key,
      kind: "checkbox",
      options: [],
      fallback,
      doc,
      path: false,
      min: null,
      max: null,
      fallbackNum: null,
    };
  }
  if (p.type === "integer" || p.type === "number") {
    const { min, max } = numericBounds(p);
    // A real numeric default becomes the input's *displayed* value (muted), so the
    // spinner steps from it — an empty input steps from min-or-0, which is how
    // "max_cores shows 4 but the arrow gives 1" happened. null = genuinely unset-able.
    const fallbackNum = typeof prop.default === "number" ? prop.default : null;
    return { key, kind: "number", options: [], fallback, doc, path: false, min, max, fallbackNum };
  }
  const path = /(_dir|^input$)/.test(key);
  return {
    key,
    kind: "text",
    options: [],
    fallback,
    doc,
    path,
    min: null,
    max: null,
    fallbackNum: null,
  };
}

/** All renderable fields of an object schema, minus `skip`, in schema order. */
// app.js calls this through the global scope — these are classic scripts, not
// modules. tests/test_gui_assets.py executes this file in Node and is what proves
// the call still resolves.
// biome-ignore lint/correctness/noUnusedVariables: app.js is the caller
function fieldSpecs(objectSchema, skip) {
  const out = [];
  const properties = objectSchema?.properties || {};
  for (const [key, prop] of Object.entries(properties)) {
    if (skip?.includes(key)) continue;
    const spec = fieldSpec(key, prop);
    if (spec) out.push(spec);
  }
  return out;
}

/** Coerce an <input> string back to the schema's type ("" -> undefined = use default).
 *
 * Two rules beyond typing: numbers are clamped to the schema's bounds (the min/max
 * attributes stop the spinner, but nothing stops typing "0" into max_cores — the clamp
 * does), and — unless keepDefault — a value equal to the schema default coerces to
 * undefined, so step-level knobs stay deviation-only. Workflow settings pass
 * keepDefault=true: the file spells them out explicitly, like the shipped examples. */
// app.js calls this through the global scope — these are classic scripts, not
// modules. tests/test_gui_assets.py executes this file in Node and is what proves
// the call still resolves.
// biome-ignore lint/correctness/noUnusedVariables: app.js is the caller
function coerceField(field, raw, keepDefault = false) {
  if (raw === "" || raw === undefined || raw === null) return undefined;
  if (field.kind === "checkbox") {
    if (keepDefault) return raw;
    return raw === (field.fallback === "true") ? undefined : raw;
  }
  if (field.kind === "number") {
    const n = Number(raw);
    if (Number.isNaN(n)) return raw;
    let value = n;
    if (field.min !== null && field.min !== undefined && value < field.min) value = field.min;
    if (field.max !== null && field.max !== undefined && value > field.max) value = field.max;
    if (!keepDefault && field.fallbackNum !== null && value === field.fallbackNum) {
      return undefined;
    }
    return value;
  }
  if (!keepDefault && field.fallback !== "" && String(raw) === field.fallback) return undefined;
  return raw;
}

/** The directory part of a path, or "." when it names none.
 *
 * `lastIndexOf` returns -1 for a bare filename and slicing to -1 drops the last
 * *character* rather than a directory — so `chemrefine gui input.yaml` sent
 * base_dir "input.yam", every relative path in the config resolved under a directory
 * that does not exist, and the validation report blamed templates sitting right
 * beside it. Lives here, with the other pure helpers, because `builder()` reads
 * `window` and `localStorage` as it is constructed: logic left in app.js cannot be
 * reached by a test at all.
 */
// app.js calls this through the global scope — these are classic scripts, not
// modules. tests/test_gui_assets.py executes this file in Node and is what proves
// the call still resolves.
// biome-ignore lint/correctness/noUnusedVariables: app.js is the caller
function parentDir(path) {
  const cut = path.lastIndexOf("/");
  if (cut < 0) return ".";
  return cut === 0 ? "/" : path.slice(0, cut);
}

/** Reorder a raw config mapping into the schema's declaration order.
 *
 * The static playground's twin of the server's `_canonical_order` (gui/app.py): there
 * `/api/yaml` has the real emitter behind it, here `jsyaml.dump` writes keys in click
 * order — and a fresh session's cfg starts `{steps: []}`, so the playground's YAML led
 * with the steps block and trailed the settings, exactly the shape the server half was
 * written to fix. The order authority is the same schema document (property order is
 * the models' field order). Unknown keys sort to the end, order preserved (sort is
 * stable) — validation is the place that complains about them, not the emitter.
 */
// app.js calls this through the global scope — these are classic scripts, not
// modules. tests/test_gui_assets.py executes this file in Node and is what proves
// the call still resolves.
// biome-ignore lint/correctness/noUnusedVariables: app.js is the caller
function canonicalConfigOrder(config, schemaDoc) {
  if (!config || typeof config !== "object" || Array.isArray(config)) return config;
  const rankOf = (properties) =>
    Object.fromEntries(Object.keys(properties || {}).map((key, i) => [key, i]));
  const order = (mapping, rank) =>
    Object.fromEntries(
      Object.entries(mapping).sort(([a], [b]) => (rank[a] ?? 999) - (rank[b] ?? 999)),
    );
  const ordered = order(config, rankOf(schemaDoc?.config?.properties));
  if (Array.isArray(ordered.steps)) {
    const stepRank = rankOf(schemaDoc?.config?.$defs?.StepConfig?.properties);
    ordered.steps = ordered.steps.map((step) =>
      step && typeof step === "object" && !Array.isArray(step) ? order(step, stepRank) : step,
    );
  }
  return ordered;
}

/** Join a directory and a filename the way a path is joined, not the way strings are.
 *
 * `${dir}/${name}` is right only when `name` is bare. The Save… box is free text, so a
 * pasted absolute path produced `/home/u//abs/path` — a directory named "" and a file
 * that never existed — and a `~/…` was joined rather than expanded. An absolute name or
 * a `~` one *is* the answer: it replaces the directory rather than hanging off it, which
 * is what every shell and `Path.joinpath` already do.
 */
// biome-ignore lint/correctness/noUnusedVariables: app.js is the caller
function joinPath(dir, name) {
  if (name.startsWith("/") || name.startsWith("~")) return name;
  return dir.endsWith("/") ? dir + name : `${dir}/${name}`;
}

const RECENTS_KEY = "cr-recents";
const RECENTS_MAX = 8;

/** The recently-opened workflow paths, newest first — never contents, only paths.
 *
 * Reads defensively because localStorage is a browser profile: it survives upgrades, can
 * be edited by hand, and is shared with whatever an older build of this page wrote there.
 * A malformed value must not blank the whole component, which is what an uncaught throw
 * inside `builder()` does — the page never renders and every button is gone.
 */
// biome-ignore lint/correctness/noUnusedVariables: app.js is the caller
function readRecents() {
  try {
    const raw = JSON.parse(localStorage.getItem(RECENTS_KEY) || "[]");
    return Array.isArray(raw) ? raw.filter((p) => typeof p === "string").slice(0, RECENTS_MAX) : [];
  } catch {
    return [];
  }
}

/** `path` to the front of the recents list, deduplicated, capped — the new list. */
// biome-ignore lint/correctness/noUnusedVariables: app.js is the caller
function rememberRecent(current, path) {
  const next = [path, ...current.filter((p) => p !== path)].slice(0, RECENTS_MAX);
  try {
    localStorage.setItem(RECENTS_KEY, JSON.stringify(next));
  } catch {
    // A full or disabled store costs the convenience, never the load that just succeeded.
  }
  return next;
}

/** A path shortened to its last two segments, for a button that must stay button-sized.
 *
 * `/groups/sterling/mfshome/dal063121/projects/a3eda/ts-uncat/input.yaml` is a real path
 * from a real tree; the full string is on the button's `title`, where it can be read.
 */
// biome-ignore lint/correctness/noUnusedVariables: app.js is the caller
function shortPath(path) {
  const parts = path.split("/").filter(Boolean);
  return parts.length <= 2 ? path : `…/${parts.slice(-2).join("/")}`;
}
