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
    return { key, kind: "select", options: p.enum, fallback, doc, path: false };
  }
  if (p.type === "boolean") return { key, kind: "checkbox", options: [], fallback, doc, path: false };
  if (p.type === "integer" || p.type === "number") {
    return { key, kind: "number", options: [], fallback, doc, path: false };
  }
  const path = /(_dir|^input$)/.test(key);
  return { key, kind: "text", options: [], fallback, doc, path };
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

/** Coerce an <input> string back to the schema's type ("" -> undefined = use default). */
function coerceField(field, raw) {
  if (raw === "" || raw === undefined || raw === null) return undefined;
  if (field.kind === "checkbox") return raw ? true : undefined;
  if (field.kind === "number") {
    const n = Number(raw);
    return Number.isNaN(n) ? raw : n;
  }
  return raw;
}
