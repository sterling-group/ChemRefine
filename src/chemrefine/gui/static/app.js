/* app.js — the Alpine component behind the builder.
 *
 * State model: `cfg` mirrors the raw YAML mapping (never a parallel representation —
 * what the form edits is what the file says). Every mutation triggers a debounced
 * POST /api/yaml so the right pane always shows the server-emitted YAML; "edit as
 * text" flips the direction (textarea -> /api/parse -> cfg). The token arrives once in
 * the launch URL's query string and is sent as X-ChemRefine-Token on every API call.
 */
"use strict";

// index.html calls this from x-data. Biome reads one file at a time and cannot see
// the page; tests/test_gui_assets.py checks that wiring, in both directions.
// biome-ignore lint/correctness/noUnusedVariables: the page is the caller
function builder() {
  return {
    token: new URLSearchParams(window.location.search).get("token") || "",
    staticMode: false,
    serverHost: "",
    ready: false,
    fatal: "",
    schema: null,
    cfg: { steps: [] },
    execRows: [],
    stepKeys: [], // stable per-card identity; see rekeySteps()
    _uid: 0,
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
    _chatGen: 0,
    chat: {
      detail: "",
      msgs: [],
      pending: [],
      decisions: {},
      draft: "",
      busy: false,
      provider: localStorage.getItem("cr-provider") || "ollama",
      model: localStorage.getItem("cr-model") || "",
      baseUrl: localStorage.getItem("cr-baseurl") || "",
    },
    browse: {
      open: false,
      mode: "save",
      title: "",
      path: "",
      parent: "",
      entries: [],
      filename: "input.yaml",
      onPick: null,
    },
    tmpl: { open: false, step: null, path: "", text: "" },
    _timer: null,

    // ---------------- boot ----------------
    async init() {
      // One page, two worlds: served by the local Flask app there is an /api behind
      // us; copied onto the static docs site there is only schema.json, baked at docs
      // build time. Probe once and let every later call route accordingly.
      // A *server* that answers is the local GUI even when it says no: only the
      // absence of one (fetch throws) means the static docs copy. Treating a 401 as
      // "no server" sent the local page into playground mode, where it then died
      // fetching a schema.json that only the docs build has — a blank page with a
      // banner telling the user to run the very thing they were running.
      let data = null;
      let served = true;
      try {
        const probe = await fetch("/api/bootstrap", {
          headers: { "X-ChemRefine-Token": this.token },
        });
        if (probe.ok) data = await probe.json();
      } catch {
        served = false; // nothing listening: this is the static docs copy
      }
      if (!data && served) {
        this.fatal =
          "This tab's session token is stale — the server was restarted. Open the " +
          "URL printed by `chemrefine gui` again.";
        this.ready = true;
        return;
      }
      if (data) {
        this.schema = data.schema;
        this.serverHost = data.host || "";
        if (data.initial) {
          this.savedPath = data.initial.path;
          const parsed = await this.api("POST", "/api/parse", {
            yaml_text: data.initial.yaml_text,
          });
          if (parsed?.config) this.cfg = this.withSteps(parsed.config);
        }
      } else {
        this.staticMode = true;
        this.flash = "";
        try {
          this.schema = await (await fetch("schema.json")).json();
        } catch {
          this.fatal = "Could not load the schema — reload the page.";
          this.ready = true;
          return;
        }
      }
      this.seedWorkflowDefaults();
      this.rekeySteps();
      if (!this.cfg.steps.length) this.addStep();
      this.syncExecRows();
      this.ready = true;
      this.syncYaml();
    },

    seedWorkflowDefaults() {
      // Workflow settings are spelled out in the file, like every shipped example — a
      // config that states its template_dir, charge and max_cores reads as a complete
      // protocol. (Step knobs stay deviation-only: a step listing every default would
      // bury the two lines that matter.) Only fills what the loaded config omits.
      for (const field of this.topFields) {
        if (field.fallback === "" || this.cfg[field.key] !== undefined) continue;
        this.cfg[field.key] =
          field.kind === "number"
            ? field.fallbackNum
            : field.kind === "checkbox"
              ? field.fallback === "true"
              : field.fallback;
      }
    },

    async api(method, url, body) {
      if (this.staticMode) return this.staticApi(url, body);
      const options = { method, headers: { "X-ChemRefine-Token": this.token } };
      if (body !== undefined) {
        options.headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body);
      }
      let response;
      try {
        response = await fetch(url, options);
      } catch {
        this.flash = "the ChemRefine server is unreachable — is `chemrefine gui` still running?";
        return null;
      }
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        this.flash = data.error || `${response.status} error`;
        return null;
      }
      return data;
    },

    staticApi(url, body) {
      // The playground's stand-ins. YAML runs on vendored js-yaml here ONLY — the
      // local GUI keeps emission server-side, the system's single implementation.
      if (url === "/api/yaml") {
        // Ordered the way the server's _canonical_order orders: jsyaml writes keys in
        // click order, and a fresh session's cfg leads with `steps`.
        const ordered = canonicalConfigOrder(body.config, this.schema);
        return { yaml_text: jsyaml.dump(ordered, { noRefs: true }) };
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
          this.flash = `malformed YAML: ${err.message}`;
          return null;
        }
      }
      this.flash = "not available in the online playground — pip install chemrefine[gui]";
      return null;
    },

    withSteps(config) {
      // `steps:` written but empty parses as null, and a half-typed raw edit can leave
      // a scalar there; either one made every later cfg.steps.map/length throw and
      // blanked the builder for good.
      const merged = { steps: [], ...config };
      if (!Array.isArray(merged.steps)) merged.steps = [];
      return merged;
    },

    // ---------------- schema-derived field lists ----------------
    get topFields() {
      // Presentation order only (the emitted YAML uses the schema's canonical order):
      // where things live, what the molecule is, what resources it gets, how it runs.
      const curated = [
        "input",
        "template_dir",
        "output_dir",
        "scratch_dir",
        "charge",
        "multiplicity",
        "max_cores",
        "max_gpus",
        "job_timeout_seconds",
        "dispatch",
        "slurm_template",
        "slurm_array",
      ];
      const rank = Object.fromEntries(curated.map((key, i) => [key, i]));
      return fieldSpecs(this.schema.config, ["steps", "executables"]).sort(
        (a, b) => (rank[a.key] ?? 99) - (rank[b.key] ?? 99),
      );
    },
    get engineNames() {
      return Object.keys(this.schema.engines).sort();
    },
    operationsFor(step) {
      // Each engine's own vocabulary (the ORCA family today, ExtOpt engines included —
      // they are ORCA-driven and parse the same outputs). An engine that declares none
      // treats the field as a free label.
      const descriptor = this.schema.engines[step.engine] || {};
      // A server predating per-engine vocabularies has no `operations` key at all;
      // fall back to the document-level list rather than claiming "not used".
      const ops = descriptor.operations
        ? [...descriptor.operations]
        : "operations" in descriptor
          ? []
          : [...(this.schema.operations || [])];
      if (step.operation && !ops.includes(step.operation)) ops.unshift(step.operation);
      return ops;
    },
    get nmsFields() {
      return fieldSpecs(this.schema.nms);
    },
    nmsFieldsFor(step) {
      // Knobs only meaningful for one target stay hidden until that target is chosen;
      // passthroughKeys still counts the full set as declared, so nothing gets flagged.
      const target = step.options?.target || "minimum";
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
      if (!descriptor?.options_schema) return [];
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
      return Boolean(descriptor?.template_driven);
    },
    templatePlaceholder(step) {
      const descriptor = this.schema.engines[step.engine];
      const suffix = descriptor?.template_suffix;
      return suffix ? `step${step.step}.${suffix} (default)` : "(no template)";
    },
    passthroughKeys(step) {
      const declared = new Set(
        engineAndNmsKeys(this.engineFields(step.engine), step.nms ? this.nmsFields : []),
      );
      return Object.keys(step.options || {}).filter((key) => !declared.has(key));
    },

    // ---------------- mutations (every one funnels into syncYaml) ----------------
    setTop(field, raw) {
      if (field.kind === "number" && raw === "") return; // mid-keystroke, not a clear
      // keepDefault: a workflow setting stays in the file even at its default value.
      const value = coerceField(field, raw, true);
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
    setStepCharge(index, raw) {
      // The step's baseline is the *inherited* workflow charge: stepping starts there,
      // and an override equal to it is no override at all — the YAML stays clean.
      if (raw === "") return; // mid-keystroke ("-", "1e"): never rewrite the box
      const inherited = this.cfg.charge ?? 0;
      const n = Number(raw);
      this.setStep(index, "charge", Number.isNaN(n) || n === inherited ? undefined : n);
    },
    setStepMult(index, raw) {
      // Same inherit-baseline rule, with the schema's floor of 1 enforced on typing.
      if (raw === "") return;
      const inherited = this.cfg.multiplicity ?? 1;
      const n = Math.max(1, Math.round(Number(raw)));
      this.setStep(index, "multiplicity", Number.isNaN(n) || n === inherited ? undefined : n);
    },

    // ---------------- executables (dict → editable rows) ----------------
    syncExecRows() {
      this.execRows = Object.entries(this.cfg.executables || {}).map(([name, path]) => ({
        name,
        path: String(path),
      }));
    },
    writeExecBack() {
      const entries = this.execRows
        .filter((row) => row.name.trim())
        .map((row) => [row.name.trim(), row.path]);
      const names = entries.map(([name]) => name);
      const duplicate = names.find((name, i) => names.indexOf(name) !== i);
      if (duplicate) {
        this.flash = `two executables named "${duplicate}" — only the last path is kept`;
      }
      if (entries.length) this.cfg.executables = Object.fromEntries(entries);
      else delete this.cfg.executables;
      this.syncYaml();
    },
    addExec() {
      this.execRows.push({ name: "", path: "" });
    },
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
      this.openBrowseWith(
        `Pick the ${this.execRows[index].name || "executable"} binary`,
        (path) => {
          this.execRows[index].path = path;
          this.writeExecBack();
        },
      );
    },

    addStep() {
      this.cfg.steps.push({ step: this.cfg.steps.length + 1, engine: "orca" });
      this.stepKeys.push(++this._uid);
      this.syncYaml();
    },
    removeStep(index) {
      this.cfg.steps.splice(index, 1);
      this.stepKeys.splice(index, 1);
      this.renumber();
    },
    moveStep(index, delta) {
      const steps = this.cfg.steps;
      const [moved] = steps.splice(index, 1);
      steps.splice(index + delta, 0, moved);
      const [key] = this.stepKeys.splice(index, 1);
      this.stepKeys.splice(index + delta, 0, key);
      this.renumber();
    },
    rekeySteps() {
      // x-for keyed by array index makes DOM state (a collapsed card, focus, the
      // template modal) stick to the *slot*: move a step and the collapse stays behind
      // on whatever lands there. These ids follow the step object instead.
      this.stepKeys = this.cfg.steps.map(() => ++this._uid);
    },
    renumber() {
      this.cfg.steps.forEach((step, i) => {
        step.step = i + 1;
      });
      // The results view names a step by number; after renumbering that number means a
      // different step (or none), so the panel must let go rather than keep serving
      // the old one under a selector that has silently snapped back to "—".
      if (
        this.resultsStep &&
        !this.cfg.steps.some((s) => String(s.step) === String(this.resultsStep))
      ) {
        this.resultsStep = "";
        this.runResults = null;
      }
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
      } catch {
        const box = document.getElementById("yaml");
        box.select();
        document.execCommand("copy");
        window.getSelection().removeAllRanges();
        this.flash = "YAML copied to clipboard";
      }
    },
    async leaveRawEdit() {
      if (this.rawEdit) return; // entering text mode: nothing to apply
      this.rawEdit = true; // hold the mode until the text parses
      await this.applyRaw();
    },
    async applyRaw() {
      const parsed = await this.api("POST", "/api/parse", { yaml_text: this.yamlText });
      if (parsed?.config) {
        this.cfg = this.withSteps(parsed.config);
        this.seedWorkflowDefaults(); // the form shows them; the file states them
        this.rekeySteps();
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
      const base = this.savedPath ? parentDir(this.savedPath) : null;
      this.report = await this.api("POST", "/api/validate", {
        yaml_text: this.yamlText,
        base_dir: base,
      });
    },
    async scaffold() {
      const data = await this.api("POST", "/api/scaffold", { config_path: this.savedPath });
      if (data) {
        this.flash = `scaffold: ${data.written.length} written, ${data.kept.length} kept`;
      }
    },

    // ---------------- run dashboard ----------------
    async refreshStatus() {
      if (!this.savedPath) return;
      const status = await this.api("POST", "/api/status", { config_path: this.savedPath });
      if (status) this.runStatus = status;
      this.runFailures = await this.api("POST", "/api/failures", { config_path: this.savedPath });
      clearTimeout(this._statusTimer);
      if (status?.running) {
        // Poll while a driver holds the tree; stop the moment it lets go.
        this._statusTimer = setTimeout(() => this.refreshStatus(), 5000);
      }
    },
    async launch(action) {
      const blurb =
        action === "run"
          ? "start the full pipeline from step 1 (invalidates the cache)"
          : action === "resume"
            ? "resume, honouring the cache"
            : "re-attempt failed jobs";
      if (
        !window.confirm(
          action +
            " on " +
            this.savedPath +
            "?\nThis will " +
            blurb +
            " — real compute on this machine.",
        )
      ) {
        return;
      }
      const started = await this.api("POST", "/api/run", { config_path: this.savedPath, action });
      if (started) {
        this.flash = `${action} started (pid ${started.pid}); log: ${started.log}`;
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
        config_path: this.savedPath,
        step: Number(step),
        limit: 20,
        offset,
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
        this.chat.detail = "pick a model in the settings below (or set CHEMREFINE_LLM_MODEL)";
      } else this.chat.detail = "";
    },
    _chatPayload(extra) {
      const payload = { provider: this.chat.provider, ...extra };
      if (this.chat.model) payload.model = this.chat.model;
      if (this.chat.baseUrl) payload.base_url = this.chat.baseUrl;
      return payload;
    },
    async _chatTurn(extra) {
      const generation = ++this._chatGen;
      this.chat.busy = true;
      try {
        const data = await this.api("POST", "/api/agent/chat", this._chatPayload(extra));
        if (!data) return false; // server refused; flash explains
        if (generation !== this._chatGen) return true; // a reset won the race: drop it
        if (data.pending) {
          this.chat.pending = data.pending;
          this.chat.decisions = {};
        } else if (data.reply !== null && data.reply !== undefined) {
          this.chat.msgs.push({ who: "agent", text: data.reply });
        }
        return true;
      } catch (err) {
        this.flash = `chat request failed: ${err}`;
        return false;
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
      const approvals = { ...this.chat.decisions };
      const heldPending = this.chat.pending;
      this.chat.pending = [];
      this.chat.decisions = {};
      const generation = this._chatGen;
      const landed = await this._chatTurn({ approvals });
      // A failed resume must not swallow the verdicts: without the cards the suspended
      // run has nothing left to answer it. Restore unless a reset intervened.
      if (!landed && this._chatGen === generation + 1) {
        this.chat.pending = heldPending;
        this.chat.decisions = approvals;
      }
    },
    async resetChat() {
      // Reset everything the user can see — conversation AND the settings drawer —
      // and say so: a reset that only clears hidden server state looks broken.
      // Local state clears FIRST, so reset works even when the server call cannot
      // (a stuck busy flag, a dropped connection).
      this._chatGen += 1; // orphan any in-flight turn
      this.chat.msgs = [];
      this.chat.pending = [];
      this.chat.decisions = {};
      this.chat.draft = "";
      this.chat.busy = false;
      this.chat.provider = "ollama";
      this.chat.model = "";
      this.chat.baseUrl = "";
      localStorage.removeItem("cr-provider");
      localStorage.removeItem("cr-model");
      localStorage.removeItem("cr-baseurl");
      this.flash = "agent chat reset — conversation cleared, provider settings back to defaults";
      try {
        await this.api("POST", "/api/agent/chat", { reset: true });
      } catch {
        // Local state is already fresh; the server forgets on its next reset/turn.
      }
      this.chatAvailability();
    },

    // ---------------- browse / save ----------------
    async openBrowse(fieldKey) {
      await this.openBrowseWith(`Pick ${fieldKey}`, (path) => {
        this.cfg[fieldKey] = path;
        this.syncYaml();
      });
    },
    async openBrowseWith(title, onPick) {
      this.browse = { ...this.browse, open: true, mode: "pick", onPick, title };
      await this.navigate(null);
    },
    async openSave() {
      this.browse = {
        ...this.browse,
        open: true,
        mode: "save",
        onPick: null,
        title: "Save workflow as…",
      };
      await this.navigate(null);
    },
    async navigate(path) {
      const data = await this.api(
        "GET",
        `/api/browse${path ? `?path=${encodeURIComponent(path)}` : ""}`,
      );
      if (data) {
        this.browse.path = data.path;
        this.browse.parent = data.parent;
        this.browse.entries = data.entries;
      }
    },
    pickHere() {
      // Directory fields (template_dir, output_dir, scratch_dir) are chosen by
      // navigating *into* the folder and taking it — clicking a file never applied.
      if (this.browse.onPick) this.browse.onPick(this.browse.path);
      this.browse.open = false;
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
      const path = `${this.browse.path}/${this.browse.filename || "input.yaml"}`;
      const data = await this.api("POST", "/api/save", { path, yaml_text: this.yamlText });
      if (data) {
        this.savedPath = data.path;
        this.browse.open = false;
        this.flash = `saved ${data.path}`;
      }
    },

    // ---------------- template editor ----------------
    async openTemplate(step) {
      const key = step.name || step.step;
      const data = await this.api(
        "GET",
        "/api/template?config_path=" +
          encodeURIComponent(this.savedPath) +
          "&step=" +
          encodeURIComponent(key),
      );
      if (data) this.tmpl = { open: true, step: step.step, path: data.path, text: data.text };
      else this.flash += " — run Scaffold templates first?";
    },
    async saveTemplate() {
      const data = await this.api("POST", "/api/template", {
        config_path: this.savedPath,
        step: this.tmpl.step,
        text: this.tmpl.text,
      });
      if (data) {
        this.flash = `saved ${data.path}`;
        this.tmpl.open = false;
      }
    },
  };
}

/** The union of declared option keys for the passthrough note. */
function engineAndNmsKeys(engineFields, nmsFields) {
  return engineFields.concat(nmsFields).map((field) => field.key);
}
