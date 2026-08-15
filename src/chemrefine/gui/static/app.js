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
    staticMode: false,
    ready: false,
    schema: null,
    cfg: { steps: [] },
    execRows: [],
    yamlText: "",
    rawEdit: false,
    report: null,
    flash: "",
    savedPath: null,
    runStatus: null,
    runFailures: null,
    runResults: null,
    resultsStep: "",
    _statusTimer: null,
    chat: { detail: "", msgs: [], pending: [], decisions: {}, draft: "", busy: false,
            provider: localStorage.getItem("cr-provider") || "ollama",
            model: localStorage.getItem("cr-model") || "",
            baseUrl: localStorage.getItem("cr-baseurl") || "" },
    browse: { open: false, mode: "save", title: "", path: "", parent: "",
              entries: [], filename: "input.yaml", onPick: null },
    tmpl: { open: false, step: null, path: "", text: "" },
    _timer: null,

    // ---------------- boot ----------------
    async init() {
      // One page, two worlds: served by the local Flask app there is an /api behind
      // us; copied onto the static docs site there is only schema.json, baked at docs
      // build time. Probe once and let every later call route accordingly.
      let data = null;
      try {
        data = await this.api("GET", "/api/bootstrap");
      } catch (err) {
        data = null;
      }
      if (data) {
        this.schema = data.schema;
        if (data.initial) {
          this.savedPath = data.initial.path;
          const parsed = await this.api("POST", "/api/parse",
                                        { yaml_text: data.initial.yaml_text });
          if (parsed && parsed.config) this.cfg = this.withSteps(parsed.config);
        }
      } else {
        this.staticMode = true;
        this.flash = "";
        this.schema = await (await fetch("schema.json")).json();
      }
      if (!this.cfg.steps.length) this.addStep();
      this.syncExecRows();
      this.ready = true;
      this.syncYaml();
    },

    async api(method, url, body) {
      if (this.staticMode) return this.staticApi(url, body);
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

    staticApi(url, body) {
      // The playground's stand-ins. YAML runs on vendored js-yaml here ONLY — the
      // local GUI keeps emission server-side, the system's single implementation.
      if (url === "/api/yaml") {
        return { yaml_text: jsyaml.dump(body.config, { noRefs: true }) };
      }
      if (url === "/api/parse") {
        try {
          const config = jsyaml.load(body.yaml_text);
          if (!config || typeof config !== "object" || Array.isArray(config)) {
            this.flash = "config is not a YAML mapping";
            return null;
          }
          return { config };
        } catch (err) {
          this.flash = "malformed YAML: " + err.message;
          return null;
        }
      }
      this.flash = "not available in the online playground — pip install chemrefine[gui]";
      return null;
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
    minOneOrUndef(raw) {
      // Multiplicity floor: the schema says >= 1, and 0 or negatives typed past the
      // spinner must not reach the YAML.
      if (raw === "") return undefined;
      const n = Number(raw);
      return Number.isNaN(n) ? undefined : Math.max(1, Math.round(n));
    },

    // ---------------- executables (dict → editable rows) ----------------
    syncExecRows() {
      this.execRows = Object.entries(this.cfg.executables || {})
        .map(([name, path]) => ({ name, path: String(path) }));
    },
    writeExecBack() {
      const entries = this.execRows
        .filter((row) => row.name.trim())
        .map((row) => [row.name.trim(), row.path]);
      if (entries.length) this.cfg.executables = Object.fromEntries(entries);
      else delete this.cfg.executables;
      this.syncYaml();
    },
    addExec() { this.execRows.push({ name: "", path: "" }); },
    removeExec(index) {
      this.execRows.splice(index, 1);
      this.writeExecBack();
    },
    setExecName(index, name) {
      this.execRows[index].name = name;
      this.writeExecBack();
    },
    setExecPath(index, path) {
      this.execRows[index].path = path;
      this.writeExecBack();
    },
    browseExec(index) {
      this.openBrowseWith("Pick the " + (this.execRows[index].name || "executable") +
                          " binary", (path) => {
        this.execRows[index].path = path;
        this.writeExecBack();
      });
    },

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
    downloadYaml() {
      const blob = new Blob([this.yamlText], { type: "text/yaml" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "input.yaml";
      link.click();
      URL.revokeObjectURL(link.href);
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
        this.syncExecRows();
        this.rawEdit = false;
        this.flash = "text applied to the form";
        this.syncYaml();
      }
    },

    // ---------------- actions ----------------
    async validateNow() {
      if (this.staticMode) {
        this.flash = "validation runs the real models — pip install chemrefine[gui]";
        return;
      }
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

    // ---------------- run dashboard ----------------
    async refreshStatus() {
      if (!this.savedPath) return;
      const status = await this.api("POST", "/api/status", { config_path: this.savedPath });
      if (status) this.runStatus = status;
      this.runFailures = await this.api("POST", "/api/failures",
                                        { config_path: this.savedPath });
      clearTimeout(this._statusTimer);
      if (status && status.running) {
        // Poll while a driver holds the tree; stop the moment it lets go.
        this._statusTimer = setTimeout(() => this.refreshStatus(), 5000);
      }
    },
    async launch(action) {
      const blurb = action === "run"
        ? "start the full pipeline from step 1 (invalidates the cache)"
        : action === "resume" ? "resume, honouring the cache" : "re-attempt failed jobs";
      if (!window.confirm(action + " on " + this.savedPath + "?\nThis will " + blurb
                          + " — real compute on this machine.")) {
        return;
      }
      const started = await this.api("POST", "/api/run",
                                     { config_path: this.savedPath, action });
      if (started) {
        this.flash = action + " started (pid " + started.pid + "); log: " + started.log;
        this.refreshStatus();
      }
    },
    async loadResults(step, offset = 0) {
      this.resultsStep = step;
      if (!step) {
        this.runResults = null;
        return;
      }
      this.runResults = await this.api("POST", "/api/results", {
        config_path: this.savedPath, step: Number(step), limit: 20, offset,
      });
    },

    // ---------------- agent chat ----------------
    saveChatSettings() {
      localStorage.setItem("cr-provider", this.chat.provider);
      localStorage.setItem("cr-model", this.chat.model);
      localStorage.setItem("cr-baseurl", this.chat.baseUrl);
    },
    async chatAvailability() {
      if (this.staticMode) return;
      const state = await this.api("GET", "/api/agent/availability");
      if (!state) return;
      if (!state.installed) this.chat.detail = state.detail;
      else if (!state.configured && !this.chat.model) {
        this.chat.detail = "pick a model above (or set CHEMREFINE_LLM_MODEL)";
      } else this.chat.detail = "";
    },
    _chatPayload(extra) {
      const payload = { provider: this.chat.provider, ...extra };
      if (this.chat.model) payload.model = this.chat.model;
      if (this.chat.baseUrl) payload.base_url = this.chat.baseUrl;
      return payload;
    },
    async _chatTurn(extra) {
      this.chat.busy = true;
      try {
        const data = await this.api("POST", "/api/agent/chat", this._chatPayload(extra));
        if (!data) return;
        if (data.pending) {
          this.chat.pending = data.pending;
          this.chat.decisions = {};
        } else if (data.reply !== null && data.reply !== undefined) {
          this.chat.msgs.push({ who: "agent", text: data.reply });
        }
      } catch (err) {
        this.flash = "chat request failed: " + err;
      } finally {
        this.chat.busy = false; // never leave the panel stuck on a failed request
      }
    },
    async sendChat() {
      const message = this.chat.draft.trim();
      if (!message || this.chat.busy) return;
      this.chat.msgs.push({ who: "you", text: message });
      this.chat.draft = "";
      await this._chatTurn({ message });
    },
    async decide(callId, allow) {
      // Every pending call needs a verdict before the run can resume.
      this.chat.decisions[callId] = allow;
      if (Object.keys(this.chat.decisions).length < this.chat.pending.length) return;
      const approvals = this.chat.decisions;
      this.chat.pending = [];
      this.chat.decisions = {};
      await this._chatTurn({ approvals });
    },
    async resetChat() {
      // Clear locally FIRST — reset must work even when the server call cannot
      // (a stuck busy flag, a dropped connection), or the panel stays dead.
      this.chat.msgs = [];
      this.chat.pending = [];
      this.chat.decisions = {};
      this.chat.draft = "";
      this.chat.busy = false;
      this.flash = "";
      try {
        await this.api("POST", "/api/agent/chat", { reset: true });
      } catch (err) {
        // Local state is already fresh; the server forgets on its next reset/turn.
      }
    },

    // ---------------- browse / save ----------------
    async openBrowse(fieldKey) {
      await this.openBrowseWith("Pick " + fieldKey, (path) => {
        this.cfg[fieldKey] = path;
        this.syncYaml();
      });
    },
    async openBrowseWith(title, onPick) {
      this.browse = { ...this.browse, open: true, mode: "pick", onPick, title };
      await this.navigate(null);
    },
    async openSave() {
      this.browse = { ...this.browse, open: true, mode: "save", onPick: null,
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
      if (this.browse.mode === "pick" && this.browse.onPick) {
        this.browse.onPick(entry.path);
        this.browse.open = false;
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
