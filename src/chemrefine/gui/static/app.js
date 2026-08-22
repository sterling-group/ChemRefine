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
    // Which panel each column shows. Dotted (`tabs.left`), not two flat scalars, and read
    // through showTab() rather than assigned inline: the asset guards resolve a bare name
    // in an expression only when it is dotted or called, so `x-show="leftTab === 'agent'"`
    // and `@click="leftTab = 'agent'"` would both be checked by nothing at all.
    tabs: { left: "builder", right: "yaml" },
    report: null,
    flash: "",
    savedPath: null,
    // The YAML as it stands on disk, set whenever we write or read the file. dirty() is
    // the comparison; without it the agent reload could not tell "adopt this" from
    // "you have unsaved edits I am about to throw away".
    savedText: "",
    // A write the agent made that the form has not taken up, because taking it up would
    // have discarded unsaved edits: { path } until the user answers.
    pendingReload: null,
    runStatus: null,
    runFailures: null,
    runResults: null,
    resultsStep: "",
    _statusTimer: null,
    _chatGen: 0,
    _checkGen: 0,
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
      // Deliberately NOT from localStorage, and deliberately not written there either:
      // this is a live credential, and the browser profile outlives the session it was
      // typed for. It lives here for as long as the tab does, and goes out per request.
      apiKey: "",
      // The preflight verdict, in its own state rather than sharing `detail`, which
      // chatAvailability() overwrites unconditionally. Sharing them meant every probe
      // wiped the check result — invisible until the panel became a tab that re-probes
      // on each switch. `ok: null` is "not checked yet", distinct from a failed check.
      check: { ok: null, findings: [], busy: false },
      // Field shapes per provider, served by /api/agent/availability. Never copied into
      // this file: a second copy of the URLs would beat CHEMREFINE_LLM_BASE_URL.
      presets: {},
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
    mol: { step: "", structureId: "", modeIndex: "", busy: false, note: "", animating: false },
    _viewer: null,
    _viewerLib: null, // the in-flight or settled load of the vendored bundle
    _timer: null,

    // ---------------- boot ----------------
    async init() {
      // One page, three worlds, told apart by how the probe ends. Served by the local
      // Flask app, /api/bootstrap answers 200 (or 401 when the token is stale or the
      // URL lost its query — same server, wrong key). Copied onto the static docs
      // site, the probe still *answers* — the docs host says 404, it never refuses
      // the connection — and only schema.json exists, baked at docs build time. And a
      // bookmark on a dead port gets no answer at all: the fetch throws. Each world
      // used to be inferred from the wrong signal (a 401 read as "no server" sent the
      // local page into playground mode; a 404 read as "server said no" hid the
      // published playground behind a stale-token banner), so the dispatch reads the
      // actual status.
      let data = null;
      let status = null; // stays null when nothing answered on this origin at all
      try {
        const probe = await fetch("/api/bootstrap", {
          headers: { "X-ChemRefine-Token": this.token },
        });
        status = probe.status;
        if (probe.ok) data = await probe.json();
      } catch {
        // Connection refused — a dead port, not the docs copy; handled below.
      }
      if (status === 401) {
        this.fatal = this.token
          ? "This tab's session token is stale — the server was restarted. Open the " +
            "URL printed by `chemrefine gui` again."
          : "This URL is missing its ?token=… — open the exact URL `chemrefine gui` " + "printed.";
        this.ready = true;
        return;
      }
      if (status === null) {
        this.fatal =
          "No server is answering on this port — the bookmark outlived its session. " +
          "Run `chemrefine gui` and open the URL it prints.";
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
        // Any other *answered* status is the static docs copy (its host says 404).
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
      if (this.savedPath) {
        // A launched config is on disk and the form now mirrors it, so record that —
        // otherwise savedText stays "" and dirty() answers true from the first paint,
        // which made the agent's very first write arrive as "you have unsaved edits"
        // for a user who had typed nothing at all.
        await this.recordOnDisk();
      } else {
        this.syncYaml();
      }
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
      // Two panels name a step by number, and after renumbering that number means a
      // different step (or none) — so each must let go rather than keep serving the old
      // one under a selector that has silently snapped back to "—". The Molecule pane is
      // the second; it was added later and inherited the bug this guard was written for.
      const gone = (chosen) =>
        chosen && !this.cfg.steps.some((s) => String(s.step) === String(chosen));
      if (gone(this.resultsStep)) {
        this.resultsStep = "";
        this.runResults = null;
      }
      if (gone(this.mol.step)) {
        this.mol.step = "";
        this.mol.note = "";
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
    // The one path from "some YAML text" to "the form shows it". init(), applyRaw() and
    // the agent reload all need the same five steps in the same order, and each is
    // load-bearing: withSteps guards a `steps:` that parsed as null or a scalar,
    // seedWorkflowDefaults fills the fields the file omits, rekeySteps issues fresh card
    // identities so DOM state does not stick to a slot, syncExecRows rebuilds the
    // executables table. A third hand-written copy is how two of them drift.
    async adopt(yamlText) {
      const parsed = await this.api("POST", "/api/parse", { yaml_text: yamlText });
      if (!parsed?.config) return false;
      this.cfg = this.withSteps(parsed.config);
      this.seedWorkflowDefaults(); // the form shows them; the file states them
      this.rekeySteps();
      this.syncExecRows();
      this.syncYaml();
      return true;
    },
    async applyRaw() {
      if (await this.adopt(this.yamlText)) {
        this.rawEdit = false;
        this.flash = "text applied to the form";
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

    // A method, not a getter: the asset guards resolve a bare name in an expression only
    // when it is dotted or called, so `dirty` as a getter would be checked by nothing.
    dirty() {
      return this.yamlText !== this.savedText;
    },

    // ---------------- panels ----------------
    showTab(side, id) {
      this.tabs[side] = id;
      // Opening the agent re-probes availability, the job the panel's `<details>` toggle
      // used to do. Note what this must NOT do: touch the check verdict. Under a
      // `<details>` the probe fired on open and close; as a tab it fires on every switch,
      // so anything destructive in here becomes a per-click side effect — which is why
      // arming lives in armCheck(), driven by settings changes rather than by visibility.
      if (side === "left" && id === "agent") this.chatAvailability();
      // Mounted here rather than at init(), and this is not a preference. 3Dmol sizes its
      // canvas from the container's offsetWidth, and compensates for a hidden one only
      // when the *container's own* inline display is "none" — but x-show sets that on the
      // panel, so a viewer built while its tab is inactive reads 0 and stays 0.
      if (side === "right" && id === "molecule") this.mountViewer();
    },

    // ---------------- structure viewer ----------------
    // Loaded on first use, never at boot: the bundle is six times the rest of the
    // frontend put together, and someone who never opens this tab never pays for it. The
    // promise is cached, so two fast clicks inject one script tag.
    loadViewerLib() {
      if (!this._viewerLib) {
        this._viewerLib = new Promise((resolve, reject) => {
          if (window.$3Dmol) return resolve(window.$3Dmol);
          const tag = document.createElement("script");
          tag.src = "static/vendor/3dmol.min.js"; // relative, like every other asset here
          tag.onload = () => resolve(window.$3Dmol);
          tag.onerror = () => {
            // Drop the cached rejection: keeping it would make one dropped request
            // disable the Molecule tab until the page is reloaded.
            this._viewerLib = null;
            reject(new Error("could not load the 3Dmol bundle"));
          };
          document.head.appendChild(tag);
        });
      }
      return this._viewerLib;
    },
    async mountViewer() {
      if (this.staticMode) return; // the playground has no server to ask for geometry
      try {
        const lib = await this.loadViewerLib();
        const host = document.getElementById("viewer");
        if (!lib || !host) return;
        if (!this._viewer) {
          this._viewer = lib.createViewer(host, { backgroundColor: "white" });
        }
        // Both paths, every time: a viewer built while the tab was hidden still has to be
        // told the container has a size now.
        this._viewer.resize();
        this._viewer.render();
      } catch (err) {
        // createViewer throws a bare string on a WebGL failure, not an Error.
        this.mol.note = `the 3D viewer could not start: ${err.message || err}`;
      }
    },
    async showStructure() {
      if (this.staticMode) {
        this.flash = "the structure view needs the local chemrefine gui";
        return;
      }
      if (!this.savedPath || !this.mol.step) return;
      this.mol.busy = true;
      this.mol.note = "";
      try {
        const query = new URLSearchParams({ config_path: this.savedPath, step: this.mol.step });
        if (this.mol.structureId) query.set("structure_id", this.mol.structureId);
        if (this.mol.modeIndex !== "") query.set("mode_index", this.mol.modeIndex);
        const data = await this.api("GET", `/api/structure?${query}`);
        if (!data) return; // flash carries the reason
        await this.mountViewer();
        if (!this._viewer) return;
        // Before the model goes: animate() pushes a timer per call, so a second Show
        // would leave two driving the same model at different phases, and clearing the
        // mode box would not stop either.
        this._viewer.stopAnimate();
        this._viewer.removeAllModels();
        const model = this._viewer.addModel(data.text, "xyz");
        this._viewer.setStyle({}, { stick: { radius: 0.12 }, sphere: { scale: 0.25 } });
        // Extended XYZ carries the cell as Lattice="…" when the structure has one, and
        // 3Dmol turns that into crystal data — so this draws a box for a periodic
        // structure and nothing for a molecule, with no branch of our own.
        this._viewer.addUnitCell(model);
        this._viewer.zoomTo();
        this.mol.animating = this.mol.modeIndex !== "";
        if (this.mol.animating) {
          // The same extended-XYZ file carries three displacement columns per atom, which
          // is what 3Dmol reads as dx/dy/dz; vibrate() only builds the frames, animate()
          // plays them.
          model.vibrate(10, 1, true);
          this._viewer.animate({ loop: "backAndForth", interval: 60 });
        }
        this._viewer.render();
        this.mol.note = `step ${data.step} · ${data.structure_id}`;
      } finally {
        this.mol.busy = false;
      }
    },

    // ---------------- agent chat ----------------
    saveChatSettings() {
      // Three keys, not four: `chat.apiKey` is a live credential and is never persisted.
      // Adding it here would be the natural-looking edit and the wrong one.
      localStorage.setItem("cr-provider", this.chat.provider);
      localStorage.setItem("cr-model", this.chat.model);
      localStorage.setItem("cr-baseurl", this.chat.baseUrl);
    },
    async chatAvailability() {
      if (this.staticMode) return;
      const state = await this.api("GET", "/api/agent/availability");
      if (!state) return;
      this.chat.presets = state.presets || {};
      if (!state.installed) this.chat.detail = state.detail;
      else if (!state.configured && !this.chat.model) {
        this.chat.detail = "pick a model in the settings below (or set CHEMREFINE_LLM_MODEL)";
      } else this.chat.detail = "";
    },

    // Methods, not getters, and called with () from the markup on purpose: the asset
    // guards resolve a bare `name` in an expression only when it is dotted or called, so
    // `get needsApiKey()` read as `needsApiKey` would be checked by nothing at all and a
    // rename would ship green. The existing getters predate that guard.

    // The shape of the selected provider: which fields mean anything, and where it points
    // when nobody says otherwise. Server-supplied, so there is one preset table.
    providerShape() {
      return this.chat.presets[this.chat.provider] || { default_url: null, needs_key: true };
    },
    needsBaseUrl() {
      return this.providerShape().default_url === null; // only `custom` has nowhere to go
    },
    needsApiKey() {
      return this.providerShape().needs_key === true;
    },
    // Green light for Send. A check must have passed for the settings as they stand;
    // armCheck() takes it away the moment any of them changes.
    chatReady() {
      return this.chat.check.ok === true;
    },

    armCheck() {
      // Disarm on *settings change*, never from chatAvailability() — that runs on every
      // panel open, and once the panel is a tab it runs on every switch, which would drop
      // a passing verdict mid-conversation.
      this._checkGen += 1; // orphan any probe still in flight for the old settings
      this.chat.check = { ok: null, findings: [], busy: false };
    },
    async runCheck() {
      if (this.staticMode) {
        this.flash = "the agent runs with the local chemrefine gui";
        return;
      }
      // A probe takes a second or two, and the settings can change under it. The
      // generation is what makes the answer belong to the settings that asked for it:
      // armCheck() bumps it, so a verdict that arrives for superseded settings is
      // dropped rather than re-arming Send against an endpoint nobody checked.
      const generation = ++this._checkGen;
      this.chat.check.busy = true;
      try {
        const data = await this.api("POST", "/api/agent/check", this._chatPayload({}));
        if (generation !== this._checkGen) return; // the settings moved on
        // A refused request leaves ok null — unchecked, not failed — and `flash` explains.
        if (data) this.chat.check = { ok: data.ok, findings: data.findings, busy: false };
      } finally {
        if (generation === this._checkGen) this.chat.check.busy = false;
      }
    },

    // Reload the config the agent just wrote, into the form the user is watching.
    // Refuses to clobber unsaved edits: it offers instead, through pendingReload.
    async reloadSavedConfig(path, { force = false } = {}) {
      if (this.staticMode) return; // the playground has no server to read a file from
      if (!force && (this.dirty() || this.rawEdit)) {
        // rawEdit counts as dirty even when the text matches: syncYaml() early-returns
        // while raw editing, so adopting underneath it would desync the two panes.
        this.pendingReload = { path };
        return;
      }
      const data = await this.api("GET", `/api/load?path=${encodeURIComponent(path)}`);
      if (!data) return; // flash explains; the form keeps what it had
      if (await this.adopt(data.yaml_text)) {
        this.savedPath = data.path;
        this.pendingReload = null;
        this.rawEdit = false;
        await this.recordOnDisk();
        this.showTab("right", "yaml"); // the change is worth nothing behind another tab
        this.flash = `the agent wrote ${data.path} — loaded into the builder`;
      }
    },
    // Emit the current form and record the result as what is on disk. Awaited, not left
    // to syncYaml()'s debounce, because the caller is about to be judged clean or dirty
    // against it. The *emitted* YAML, not the file's bytes: seedWorkflowDefaults() adds
    // keys the file omits, so comparing against the file itself reads dirty at once.
    async recordOnDisk() {
      const emitted = await this.api("POST", "/api/yaml", { config: this.cfg });
      if (emitted) {
        this.yamlText = emitted.yaml_text;
        this.savedText = emitted.yaml_text;
      }
    },
    async acceptPendingReload() {
      const path = this.pendingReload?.path;
      this.pendingReload = null;
      if (path) await this.reloadSavedConfig(path, { force: true });
    },
    dismissPendingReload() {
      this.pendingReload = null;
    },

    _chatPayload(extra) {
      const payload = { provider: this.chat.provider, ...extra };
      if (this.chat.model) payload.model = this.chat.model;
      if (this.chat.baseUrl) payload.base_url = this.chat.baseUrl;
      // Only when the selected provider actually takes one. The key survives a provider
      // switch (so going back does not mean retyping it), but sending it to an endpoint
      // whose field the panel has hidden would hand a credential to a host the user did
      // not intend it for — a local vLLM box, say.
      if (this.chat.apiKey && this.needsApiKey()) payload.api_key = this.chat.apiKey;
      // Sourced from savedPath, not from chat state, though every line around it reads
      // this.chat.*: it is the file the *builder* has open, which is what the agent
      // should be told about.
      if (this.savedPath) payload.config_path = this.savedPath;
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
        // Outside that branch on purpose: an approved write and a fresh batch of approval
        // cards arrive in the same turn, so a reload hung off the reply branch would be
        // skipped exactly while the agent is working steadily. After the generation guard
        // above, so a turn a reset has orphaned cannot rewrite the form.
        if (data.wrote_config) await this.reloadSavedConfig(data.wrote_config);
        return true;
      } catch (err) {
        this.flash = `chat request failed: ${err}`;
        return false;
      } finally {
        // Only if this turn is still the current one: an orphaned turn resolving late
        // would otherwise clear the flag belonging to the turn that replaced it, and the
        // panel would accept a second message while the first was still running.
        if (generation === this._chatGen) this.chat.busy = false;
      }
    },
    async sendChat() {
      const message = this.chat.draft.trim();
      // Gated here as well as on the button: the input's @keydown.enter reaches this
      // directly, so a :disabled attribute alone would leave the Enter path live.
      if (!message || this.chat.busy || !this.chatReady()) return;
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
      this.chat.apiKey = "";
      this.armCheck(); // the verdict belonged to settings that no longer exist
      localStorage.removeItem("cr-provider");
      localStorage.removeItem("cr-model");
      localStorage.removeItem("cr-baseurl");
      // Never written by this build, removed anyway: an earlier one might have, and a
      // credential left in a browser profile is not something to leave to good intentions.
      localStorage.removeItem("cr-apikey");
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
        // Exactly the string that was written, so dirty() is a comparison and not a guess.
        this.savedText = this.yamlText;
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
