/* app.js — the Alpine component behind the builder.
 *
 * State model: `cfg` mirrors the raw YAML mapping (never a parallel representation —
 * what the form edits is what the file says). Every mutation triggers a debounced
 * POST /api/yaml so the right pane always shows the server-emitted YAML; "edit as
 * text" flips the direction (textarea -> /api/parse -> cfg). The token arrives once in
 * the launch URL's query string and is sent as X-ChemRefine-Token on every API call.
 */
"use strict";

function builder() {
  return {
    token: new URLSearchParams(window.location.search).get("token") || "",
    ready: false,
    schema: null,
    cfg: { steps: [] },
    yamlText: "",
    rawEdit: false,
    report: null,
    flash: "",
    savedPath: null,
    browse: { open: false, mode: "save", title: "", path: "", parent: "",
              entries: [], filename: "input.yaml", target: null },
    tmpl: { open: false, step: null, path: "", text: "" },
    _timer: null,

    // ---------------- boot ----------------
    async init() {
      const data = await this.api("GET", "/api/bootstrap");
      this.schema = data.schema;
      if (data.initial) {
        this.savedPath = data.initial.path;
        const parsed = await this.api("POST", "/api/parse",
                                      { yaml_text: data.initial.yaml_text });
        if (parsed && parsed.config) this.cfg = this.withSteps(parsed.config);
      }
      if (!this.cfg.steps.length) this.addStep();
      this.ready = true;
      this.syncYaml();
    },

    async api(method, url, body) {
      const options = { method, headers: { "X-ChemRefine-Token": this.token } };
      if (body !== undefined) {
        options.headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body);
      }
      const response = await fetch(url, options);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        this.flash = data.error || (response.status + " error");
        return null;
      }
      return data;
    },

    withSteps(config) { return { steps: [], ...config }; },

    // ---------------- schema-derived field lists ----------------
    get topFields() {
      return fieldSpecs(this.schema.config, ["steps", "executables"]);
    },
    get engineNames() { return Object.keys(this.schema.engines).sort(); },
    get operations() { return this.schema.operations || []; },
    get nmsFields() { return fieldSpecs(this.schema.nms); },
    nmsFieldsFor(step) {
      // Knobs only meaningful for one target stay hidden until that target is chosen;
      // passthroughKeys still counts the full set as declared, so nothing gets flagged.
      const target = (step.options || {}).target || "minimum";
      return this.nmsFields.filter((field) => {
        if (field.key === "num_random_displacements" || field.key === "seed") {
          return target === "random";
        }
        if (field.key === "ts_mode_index") return target === "ts";
        return true;
      });
    },
    engineFields(engine) {
      const descriptor = this.schema.engines[engine];
      if (!descriptor || !descriptor.options_schema) return [];
      return fieldSpecs(descriptor.options_schema, ["backend_python"]);
    },
    sampleFields(step) {
      if (!step.sample) return [];
      const byMethod = { boltzmann: "BoltzmannSample", min: "MinSample", max: "MaxSample" };
      const name = byMethod[step.sample.method];
      return name ? fieldSpecs(this.schema.config.$defs[name], ["method"]) : [];
    },
    isTemplateDriven(engine) {
      const descriptor = this.schema.engines[engine];
      return Boolean(descriptor && descriptor.template_driven);
    },
    templatePlaceholder(step) {
      const descriptor = this.schema.engines[step.engine];
      const suffix = descriptor && descriptor.template_suffix;
      return suffix ? "step" + step.step + "." + suffix + " (default)" : "(no template)";
    },
    passthroughKeys(step) {
      const declared = new Set(
        engineAndNmsKeys(this.engineFields(step.engine), step.nms ? this.nmsFields : []));
      return Object.keys(step.options || {}).filter((key) => !declared.has(key));
    },

    // ---------------- mutations (every one funnels into syncYaml) ----------------
    setTop(field, raw) {
      const value = coerceField(field, raw);
      if (value === undefined) delete this.cfg[field.key];
      else this.cfg[field.key] = value;
      this.syncYaml();
    },
    setStep(index, key, value) {
      if (value === undefined) delete this.cfg.steps[index][key];
      else this.cfg.steps[index][key] = value;
      this.syncYaml();
    },
    setEngine(index, engine) {
      this.cfg.steps[index].engine = engine;
      this.syncYaml();
    },
    setOption(index, field, raw) {
      const step = this.cfg.steps[index];
      const options = { ...(step.options || {}) };
      const value = coerceField(field, raw);
      if (value === undefined) delete options[field.key];
      else options[field.key] = value;
      if (Object.keys(options).length) step.options = options;
      else delete step.options;
      this.syncYaml();
    },
    setSampleMethod(index, method) {
      if (!method) delete this.cfg.steps[index].sample;
      else this.cfg.steps[index].sample = { method };
      this.syncYaml();
    },
    setSample(index, field, raw) {
      const sample = this.cfg.steps[index].sample;
      const value = coerceField(field, raw);
      if (value === undefined) delete sample[field.key];
      else sample[field.key] = value;
      this.syncYaml();
    },
    numOrUndef(raw) { return raw === "" ? undefined : Number(raw); },

    addStep() {
      this.cfg.steps.push({ step: this.cfg.steps.length + 1, engine: "orca" });
      this.syncYaml();
    },
    removeStep(index) {
      this.cfg.steps.splice(index, 1);
      this.renumber();
    },
    moveStep(index, delta) {
      const steps = this.cfg.steps;
      const [moved] = steps.splice(index, 1);
      steps.splice(index + delta, 0, moved);
      this.renumber();
    },
    renumber() {
      this.cfg.steps.forEach((step, i) => { step.step = i + 1; });
      this.syncYaml();
    },

    // ---------------- the two panes ----------------
    syncYaml() {
      if (this.rawEdit) return; // text is the source of truth right now
      clearTimeout(this._timer);
      this._timer = setTimeout(async () => {
        const data = await this.api("POST", "/api/yaml", { config: this.cfg });
        if (data) this.yamlText = data.yaml_text;
      }, 150);
    },
    async copyYaml() {
      // 127.0.0.1 is a secure context, so the async Clipboard API is available;
      // the execCommand path covers browsers that still refuse it.
      try {
        await navigator.clipboard.writeText(this.yamlText);
        this.flash = "YAML copied to clipboard";
      } catch (err) {
        const box = document.getElementById("yaml");
        box.select();
        document.execCommand("copy");
        window.getSelection().removeAllRanges();
        this.flash = "YAML copied to clipboard";
      }
    },
    async applyRaw() {
      const parsed = await this.api("POST", "/api/parse", { yaml_text: this.yamlText });
      if (parsed && parsed.config) {
        this.cfg = this.withSteps(parsed.config);
        this.rawEdit = false;
        this.flash = "text applied to the form";
        this.syncYaml();
      }
    },

    // ---------------- actions ----------------
    async validateNow() {
      const base = this.savedPath
        ? this.savedPath.slice(0, this.savedPath.lastIndexOf("/")) : null;
      this.report = await this.api("POST", "/api/validate",
                                   { yaml_text: this.yamlText, base_dir: base });
    },
    async scaffold() {
      const data = await this.api("POST", "/api/scaffold", { config_path: this.savedPath });
      if (data) {
        this.flash = "scaffold: " + data.written.length + " written, "
                   + data.kept.length + " kept";
      }
    },

    // ---------------- browse / save ----------------
    async openBrowse(fieldKey) {
      this.browse = { ...this.browse, open: true, mode: "pick", target: fieldKey,
                      title: "Pick " + fieldKey };
      await this.navigate(null);
    },
    async openSave() {
      this.browse = { ...this.browse, open: true, mode: "save", target: null,
                      title: "Save workflow as…" };
      await this.navigate(null);
    },
    async navigate(path) {
      const data = await this.api("GET",
        "/api/browse" + (path ? "?path=" + encodeURIComponent(path) : ""));
      if (data) {
        this.browse.path = data.path;
        this.browse.parent = data.parent;
        this.browse.entries = data.entries;
      }
    },
    pickFile(entry) {
      if (this.browse.mode === "pick" && this.browse.target) {
        this.cfg[this.browse.target] = entry.path;
        this.browse.open = false;
        this.syncYaml();
      } else {
        this.browse.filename = entry.name;
      }
    },
    async saveTo() {
      const path = this.browse.path + "/" + (this.browse.filename || "input.yaml");
      const data = await this.api("POST", "/api/save",
                                  { path, yaml_text: this.yamlText });
      if (data) {
        this.savedPath = data.path;
        this.browse.open = false;
        this.flash = "saved " + data.path;
      }
    },

    // ---------------- template editor ----------------
    async openTemplate(step) {
      const key = step.name || step.step;
      const data = await this.api("GET", "/api/template?config_path="
        + encodeURIComponent(this.savedPath) + "&step=" + encodeURIComponent(key));
      if (data) this.tmpl = { open: true, step: step.step, path: data.path, text: data.text };
      else this.flash += " — run Scaffold templates first?";
    },
    async saveTemplate() {
      const data = await this.api("POST", "/api/template",
        { config_path: this.savedPath, step: this.tmpl.step, text: this.tmpl.text });
      if (data) {
        this.flash = "saved " + data.path;
        this.tmpl.open = false;
      }
    },
  };
}

/** The union of declared option keys for the passthrough note. */
function engineAndNmsKeys(engineFields, nmsFields) {
  return engineFields.concat(nmsFields).map((field) => field.key);
}
