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
  const fallback = prop.default === undefined || prop.default === null
    ? "" : String(prop.default);
  if (Array.isArray(p.enum)) {
    return { key, kind: "select", options: p.enum, fallback, doc, path: false,
             min: null, max: null, fallbackNum: null };
  }
  if (p.type === "boolean") {
    return { key, kind: "checkbox", options: [], fallback, doc, path: false,
             min: null, max: null, fallbackNum: null };
  }
  if (p.type === "integer" || p.type === "number") {
    const { min, max } = numericBounds(p);
    // A real numeric default becomes the input's *displayed* value (muted), so the
    // spinner steps from it — an empty input steps from min-or-0, which is how
    // "max_cores shows 4 but the arrow gives 1" happened. null = genuinely unset-able.
    const fallbackNum = typeof prop.default === "number" ? prop.default : null;
    return { key, kind: "number", options: [], fallback, doc, path: false,
             min, max, fallbackNum };
  }
  const path = /(_dir|^input$)/.test(key);
  return { key, kind: "text", options: [], fallback, doc, path,
           min: null, max: null, fallbackNum: null };
}

/** All renderable fields of an object schema, minus `skip`, in schema order. */
function fieldSpecs(objectSchema, skip) {
  const out = [];
  const properties = (objectSchema && objectSchema.properties) || {};
  for (const [key, prop] of Object.entries(properties)) {
    if (skip && skip.includes(key)) continue;
    const spec = fieldSpec(key, prop);
    if (spec) out.push(spec);
  }
  return out;
}

/** Coerce an <input> string back to the schema's type ("" -> undefined = use default).
 *
 * Two rules beyond typing: numbers are clamped to the schema's bounds (the min/max
 * attributes stop the spinner, but nothing stops typing "0" into max_cores — the clamp
 * does), and any value equal to the schema default coerces to undefined, so the YAML
 * carries only deviations — never a duplicate restatement of a default. */
function coerceField(field, raw) {
  if (raw === "" || raw === undefined || raw === null) return undefined;
  if (field.kind === "checkbox") {
    return raw === (field.fallback === "true") ? undefined : raw;
  }
  if (field.kind === "number") {
    const n = Number(raw);
    if (Number.isNaN(n)) return raw;
    let value = n;
    if (field.min !== null && field.min !== undefined && value < field.min) value = field.min;
    if (field.max !== null && field.max !== undefined && value > field.max) value = field.max;
    if (field.fallbackNum !== null && value === field.fallbackNum) return undefined;
    return value;
  }
  if (field.fallback !== "" && String(raw) === field.fallback) return undefined;
  return raw;
}
