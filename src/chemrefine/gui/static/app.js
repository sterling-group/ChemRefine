/* app.js — the Alpine component behind the builder.
 *
 * State model: `cfg` mirrors the raw YAML mapping (never a parallel representation —
 * what the form edits is what the file says). Every mutation triggers a debounced
 * POST /api/yaml so the right pane always shows the server-emitted YAML; "edit as
 * text" flips the direction (textarea -> /api/parse -> cfg). The token arrives once in
 * the launch URL's query string and is sent as X-ChemRefine-Token on every API call.
 */
"use strict";

// What each run action does, in the words the confirmation dialog uses. A lookup, not a
// chain of ternaries: the chain had no default arm, so its last branch described every
// action it did not name — a rebuild-cache, which submits nothing, would have been
// confirmed as "re-attempt failed jobs". A dialog that misdescribes what it is confirming
// is worse than no dialog. The keys are `agent_tools._ACTIONS`, the whole recovery
// vocabulary; a missing one shows up immediately as "undefined" in the prompt.
// The chat settings that persist, as one vocabulary. They were nine bare string literals
// across three methods — read, write and reset — while the recents key next door was already
// a named constant. A typo in any one of the nine is a setting that silently stops
// persisting, which nothing would have caught.
const CHAT_KEYS = { provider: "cr-provider", model: "cr-model", baseUrl: "cr-baseurl" };
const CHAT_DEFAULTS = { provider: "ollama", model: "", baseUrl: "" };

// How many result rows a page holds. One constant: the fetch limit and the two paging
// buttons were three literals across two files, so changing the limit made the buttons page
// by the old stride — skipping or repeating rows, with nothing to say so.
const RESULTS_PAGE = 20;

const RUN_BLURBS = {
  run: "start the full pipeline from step 1, ignoring the cache",
  resume: "carry on where the tree left off, honouring the cache",
  rerun: "recompute, discarding what is cached for it",
  "rerun-errors": "re-attempt the ledgered failures, then carry on",
  "rebuild-cache": "re-parse the outputs already on disk — submits nothing",
  "rebuild-nms": "redo the normal-mode resolution from the outputs on disk",
};

const SPHERE_SCALE = 0.25;
// The one style the model is drawn with. Beside SPHERE_SCALE rather than written out at each
// draw path, because the label push is computed from that same radius: with the literal
// copied into the draw calls, changing it moved every sphere and left every label pushed by
// the radius the spheres used to have.
const MODEL_STYLE = { stick: { radius: 0.12 }, sphere: { scale: SPHERE_SCALE } };

// Three facts about the vendored bundle set the shape of everything below. All three were
// read out of the build, not out of 3Dmol's docs.
//
// 1. A label is a Sprite, and the sprite vertex shader adds the quad's corner offset AFTER
//    its own perspective divide:
//        finalPosition = projectionMatrix * modelViewMatrix * vec4(0,0,0,1);
//        finalPosition /= finalPosition.w;
//        finalPosition.xy += rotatedPosition;
//    Only the anchor is projected, so a label is a fixed size in SCREEN pixels at every
//    zoom, and no stylespec key changes that — the molecule shrinks around a number that
//    does not. `sizeAttenuation` looks like the knob and is not: SpriteMaterial writes
//    `scaleByViewPort` and SpritePlugin reads `scaleByViewport`, so that branch is dead
//    code. The one live multiplier is `sprite.scale`, re-read every frame — see
//    `_syncLabels()`, which is why the labels track the zoom at all.
//
// 2. `inFront: false` is what lets a label hide behind the atoms in front of it: the key
//    becomes the material's `depthTest`, and the depth-tested sprite pass runs immediately
//    after the opaque geometry, so the spheres are already in the depth buffer. But the
//    same shader gives all four corners ONE depth — the anchor's — while the sphere writes
//    its front SURFACE depth. A label left on the nucleus therefore loses LEQUAL to its own
//    sphere, which is nearer by a full radius (0.425 A for carbon at scale 0.25), and gets
//    a circular bite punched through it by the very atom it names. So each anchor is pushed
//    toward the camera by that atom's own radius. That direction is a function of the view,
//    which is why the sync below runs on rotation and not only on zoom.
//
//    The push must follow each atom's own ray to the eye, not one shared view axis. The
//    same perspective divide quoted above is why: shortening an off-axis atom's depth
//    without touching its eye-space x and y makes it project FURTHER from the centre of the
//    pane, so an axis push slides every label outward — nothing on the view axis, most at
//    the edge, and swinging around as the structure turns. Moving along the ray instead
//    moves the anchor along the line the eye collapses to a single pixel, so the projected
//    position does not change at all. See _syncLabels().
//
// 3. Dropping the background box costs the halo outright: the bundle never calls
//    strokeText, shadowBlur or shadowColor, and `borderThickness` strokes the background
//    box rather than the glyphs. Legibility now rests on the glyphs being bold and roughly
//    atom-sized. The tightest case is nitrogen, #3050F8 against black at 3.6:1 — the
//    large-text floor, which these now are.
const LABEL_STYLE = {
  // The texture's resolution, not its size on screen — LABEL_INK_HEIGHT sets that. 32 is
  // not inherited taste, it is where the resampling is least bad. The renderer draws into a
  // backing store of `devicePixelRatio` device pixels per CSS pixel and forces that ratio to
  // at least 2 (`_upscale` — see mountViewer), so a texel covers `2 * sprite.scale` device
  // pixels. With the sizing below that works out to `perAngstrom / fontSize`, and the pane
  // runs 260–620 CSS px against structures 10–30 A across, i.e. perAngstrom ≈ 9–62. A
  // fontSize of 32 keeps the ratio inside 0.7–1.4 over almost all of that; raising it drives
  // the whole range into minification, which this bundle cannot defend against — it builds
  // no mipmaps (`generateMipmap` appears nowhere in the build) and `filterFallback` returns
  // LINEAR whatever the texture asks for.
  fontSize: 32,
  // With no background this is bare transparent margin; 2 keeps the antialiased edges off
  // the texture border without inflating the quad. Not 0: `o = e.padding ? e.padding : 4`
  // reads 0 as absent and gives you 4.
  padding: 2,
  fontColor: "black",
  bold: true,
  alignment: "center",
  showBackground: false,
  inFront: false,
};

// A bold digit's cap height as a fraction of fontSize. The bundle sizes the label canvas
// `1.25 * fontSize + 2 * padding` tall and puts the baseline at `fontSize + padding`, so
// roughly half of that texture is leading and descender air no glyph ever touches — which
// is why sizing a label by its canvas height (what LABEL_WORLD_HEIGHT used to do) describes
// something the eye cannot see. Measured from the three faces `sans-serif:bold` resolves to
// on Linux: Noto Sans Bold 0.725, DejaVu Sans Bold 0.742, Liberation Sans Bold 0.698. The
// ±3% spread moves a label by well under a pixel.
const LABEL_CAP_EM = 0.72;

// Angstroms of scene a digit's cap spans — the ink, not the texture. The yardstick is the
// hydrogen blob, the smallest sphere the viewer draws: 2 * vdwRadii.H * SPHERE_SCALE =
// 2 * 1.2 * 0.25 = 0.600 A. At 0.36 a one-character label's ink box is 0.29 x 0.36 A, whose
// 0.46 A diagonal fits inside that blob with room to spare, so the number reads as smaller
// than the smallest atom it can name instead of sitting on it like a lid.
const LABEL_INK_HEIGHT = 0.36;

// Angstroms of scene the ink may span horizontally before the label is scaled down to fit.
// Height alone cannot bound a label, and that is the whole of the "too big" complaint: the
// quad's width is the text's, so at a fixed height "H10" drew 1.11 A across — 1.85 hydrogen
// blobs — while "1" drew 0.33 A. One carbon blob (2 * 1.7 * 0.25 = 0.850 A) is the budget,
// which binds only on the three-character strings `element` mode emits and leaves every
// one- and two-character label sized by height alone.
const LABEL_INK_WIDTH = 0.85;

// Angstroms the anchor sits in front of its own sphere's surface, along the ray to the
// camera. This is also the depth by which a neighbour must beat this atom to occlude the
// label, so: as small as will do.
const LABEL_CLEARANCE = 0.15;

// Legibility bounds on the drawn ink, in CSS pixels — a bound on what is seen, not on the
// scale factor. The old LABEL_SCALE_MIN/MAX were multiples of the texture's height, so they
// silently meant a different on-screen size the moment fontSize moved. These are the same
// two limits at today's texture: 0.35 * 0.72 * 32 = 8.1 px and 4 * 0.72 * 32 = 92 px.
const LABEL_INK_MIN_PX = 8;
const LABEL_INK_MAX_PX = 96;

// A label is the one thing in this scene that has a resolution of its own. A sphere is
// solved per pixel by its fragment shader — `lensqr = dot(mapping,mapping); if (lensqr >
// rsqr) discard; z = sqrt(rsqr - lensqr)` — so it is exactly as sharp as the framebuffer at
// any zoom, forever. A label is a <canvas> rasterised once into a Texture and then sampled
// with a filter this build forces to LINEAR, with no mipmaps anywhere in it. Draw it larger
// than the texels it was drawn with and you are interpolating: that is the blur, and it is
// the whole of the difference between crisp atoms and soft numbers.
//
// So the texture is re-rasterised whenever it drifts out of this band of device pixels per
// texel. Not a fixed fontSize: the size that would be right spans about 16x across the
// pane sizes, display densities and structure sizes this has to serve, so no constant can
// be right for more than one of them. The band is wide enough that an ordinary zoom crosses
// it once or twice rather than continuously, and biased high because magnification is what
// is actually visible — below 1 a label is mildly soft, above it the glyphs go blocky.
const LABEL_TEXELS_LO = 0.55;
const LABEL_TEXELS_HI = 1.3;
// Rasterisation bounds. The floor keeps a legible glyph when a label is tiny on screen; the
// ceiling bounds the texture memory of a large structure zoomed right in.
const LABEL_FONT_MIN = 16;
const LABEL_FONT_MAX = 192;

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
    // One entry per drawn label: its sprite, its texture height, the atom it names, and how
    // far in front of that atom it has to sit to clear the atom's own sphere.
    _atomLabels: [],
    _labelSync: false, // guards the re-entrant show() the view-change callback issues
    _labelViewKey: "", // the camera the labels are currently sized and pushed for
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
    // Which step the four targeted actions act on; "" means all of them. Held on the
    // component rather than read off the <select> at click time so the Run/Resume buttons
    // can say, before they are pressed, that they do not take one.
    runTarget: "",
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
      provider: localStorage.getItem(CHAT_KEYS.provider) || CHAT_DEFAULTS.provider,
      model: localStorage.getItem(CHAT_KEYS.model) || CHAT_DEFAULTS.model,
      baseUrl: localStorage.getItem(CHAT_KEYS.baseUrl) || CHAT_DEFAULTS.baseUrl,
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
      // The typed-path box. Every other path field on this page can be typed or pasted;
      // this modal was the one place a path could only be walked to, which is a long way
      // to click on a cluster tree whose interesting directory is eight levels down.
      typed: "",
    },
    // The last few workflows opened, newest first. Paths only, never file contents: this
    // is a browser profile, and it outlives the session that wrote to it.
    recents: readRecents(),
    tmpl: { open: false, step: null, path: "", text: "" },
    // `structures` is what /api/structure-list answered for the chosen step: one row per
    // structure, carrying its own mode table. Held here rather than fetched per keystroke
    // so the two combo boxes offer what the step actually holds without a request each.
    viewer: {
      step: "input",
      structureId: "",
      modeIndex: "",
      busy: false,
      note: "",
      structures: [],
      // Off by default: labels on a 60-atom structure are a wall of text, and the pane's
      // first job is to show the molecule. See atomLabel() for what the modes mean.
      labels: "off",
      // Whether a drag is currently over the drop zone — the only feedback a drag gets.
      dragging: false,
    },
    _gl: null, // the 3Dmol viewer instance, once the bundle is in
    _model: null, // the drawn model — labels and the unit cell both attach to it
    _glLib: null, // the in-flight or settled load of the vendored bundle
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
      let adopted = false; // whether a launched config already went through adopt()
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
          // Through adopt(), like every other route from YAML text to a populated form.
          // This used to be a fourth hand-written copy, which is exactly how it drifted.
          adopted = await this.adopt(data.initial.yaml_text);
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
      if (!adopted) this.refreshForm(); // adopt() has already done this for a loaded config
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

    // `quiet` suppresses the flash on a failed request, for a caller whose answer is a
    // convenience rather than the thing the user asked for — the structure lists behind the
    // combo boxes, where "this step has not run yet" is the ordinary case and announcing it
    // on every step change is noise. It never suppresses the null return: the caller still
    // has to cope with not getting an answer.
    async api(method, url, body, { quiet = false } = {}) {
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
        if (!quiet) {
          this.flash = "the ChemRefine server is unreachable — is `chemrefine gui` still running?";
        }
        return null;
      }
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        if (!quiet) this.flash = data.error || `${response.status} error`;
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
      // Three panels name a step by number, and after renumbering that number means a
      // different step (or none) — so each must let go rather than keep serving the old
      // one under a selector that has silently snapped back to "—". Each was added later
      // than the last and inherited the bug this guard was written for; the Run panel's
      // target was the third and went longest without it, which disabled Run and Resume
      // (`:disabled="!!runTarget"`) against a step that no longer existed.
      const gone = (chosen) =>
        chosen && !this.cfg.steps.some((s) => String(s.step) === String(chosen));
      if (gone(this.resultsStep)) {
        this.resultsStep = "";
        this.runResults = null;
      }
      if (gone(this.runTarget)) this.runTarget = "";
      // "input" is not a step number and never goes stale — the seeds outlive any
      // renumbering — so it must survive a guard written for step selections.
      if (this.viewer.step !== "input" && gone(this.viewer.step)) {
        this.viewer.step = "input";
        this.viewer.note = "";
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
      this.refreshForm();
      this.syncYaml();
      return true;
    },
    // The tail of adopt(), split out only because init()'s *empty* form needs the same
    // treatment without any YAML to parse. Keeping the starter step in here is what stops
    // the two paths diverging again: it used to live in init() alone, so a stepless file
    // opened from the command line showed one seeded step and the same file opened through
    // Open… showed an empty builder.
    refreshForm() {
      this.seedWorkflowDefaults(); // the form shows them; the file states them
      this.rekeySteps();
      if (!this.cfg.steps.length) this.addStep();
      this.syncExecRows();
    },
    async applyRaw() {
      if (await this.adopt(this.yamlText)) {
        this.rawEdit = false;
        this.flash = "text applied to the form";
      }
    },

    // ---------------- actions ----------------
    async validateNow() {
      if (this.playgroundRefuses("validation runs the real models, so it")) return;
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
    // The published playground has no server behind it, so everything that reads or writes
    // the disk stops here. One wording in one place: the three hand-written refusals had
    // drifted into three different sentences for the same fact, and the next caller would
    // have written a fourth. Answers whether it refused, so a caller is one line.
    playgroundRefuses(what) {
      if (!this.staticMode) return false;
      this.flash = `${what} needs the local chemrefine gui — pip install 'chemrefine[gui]'`;
      return true;
    },
    // Whether this action drives the whole pipeline, and so takes no step. start_run
    // refuses a target for these two, so offering one would be a 400 after the click
    // rather than a control that says what it accepts.
    takesTarget(action) {
      return action !== "run" && action !== "resume";
    },
    async launch(action) {
      // Only the four that accept one, and only when a step is actually chosen.
      const target = this.takesTarget(action) && this.runTarget ? this.runTarget : null;
      const where = target ? ` step ${target} of ` : " on ";
      if (
        !window.confirm(
          `${action}${where}${this.savedPath}?\nThis will ${RUN_BLURBS[action]} — ` +
            "real compute on this machine.",
        )
      ) {
        return;
      }
      const started = await this.api("POST", "/api/run", {
        config_path: this.savedPath,
        action,
        target,
      });
      if (started) {
        this.flash = `${action} started (pid ${started.pid}); log: ${started.log}`;
        this.refreshStatus();
      }
    },
    // One page forward or back. The stride lived in the markup twice and in the fetch once,
    // so changing the limit made the buttons page by the old one.
    pageResults(direction) {
      this.loadResults(this.resultsStep, this.runResults.offset + direction * RESULTS_PAGE);
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
        limit: RESULTS_PAGE,
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
      if (side === "right" && id === "structure") {
        this.mountViewer();
        this.loadStructureList(); // so the boxes offer something the moment the pane opens
      }
    },

    // A structure id and a mode number belong to the step that was showing when they were
    // typed. Carrying them to the next step is how one bad request became permanent: the
    // mode box is hidden on the seeds view, so its value was invisible as well as wrong.
    chooseViewerStep(step) {
      if (String(step) === String(this.viewer.step)) return;
      this.viewer.step = step;
      this.viewer.structureId = "";
      this.viewer.modeIndex = "";
      this.viewer.note = "";
      this.viewer.structures = []; // the previous step's ids are not this step's ids
      this.loadStructureList();
    },

    // What this step holds, so the two boxes can offer it. Failure is deliberately quiet in
    // the pane: the lists are a convenience, both boxes stay typable without them, and a
    // step that has not run yet is the ordinary case rather than an error to announce.
    async loadStructureList() {
      if (this.staticMode || !this.savedPath || !this.viewer.step) return;
      const asked = this.viewer.step;
      const url = apiUrl("/api/structure-list", {
        config_path: this.savedPath,
        step: this.viewer.step === "input" ? "" : this.viewer.step,
      });
      const data = await this.api("GET", url, undefined, { quiet: true });
      // The step can change while this is in flight, and the answer belongs to the step
      // that asked for it — offering step 1's ids under step 2 is worse than offering none.
      if (data && String(asked) === String(this.viewer.step)) {
        this.viewer.structures = data.structures;
      }
    },

    // The label beside a structure id in the dropdown — what distinguishes one from another
    // when the ids are "0", "1", "2". A mode count answers "which of these is the TS?".
    structureHint(row) {
      if (!row.modes) return "no frequency data";
      const count = Object.keys(row.modes).length;
      const imaginary = (row.imaginary || []).length;
      if (!count) return "no modes";
      return imaginary ? `${count} modes, ${imaginary} imaginary` : `${count} modes`;
    },

    // The mode list for whichever structure is named, or for the first one when the box is
    // blank — which is the structure Show would pick, so the modes offered are that
    // structure's modes and not some other one's.
    modeChoices() {
      const rows = this.viewer.structures;
      if (!rows.length) return [];
      const row = this.viewer.structureId
        ? rows.find((r) => String(r.id) === String(this.viewer.structureId))
        : rows[0];
      if (!row?.modes) return [];
      const imaginary = new Set((row.imaginary || []).map(String));
      return Object.keys(row.modes)
        .map(Number)
        .sort((a, b) => a - b)
        .map((index) => ({
          index,
          // The frequency is the whole point of the list: "mode 6" answers nothing, and
          // −820 cm⁻¹ (imaginary) is the reaction coordinate you came to look at.
          label: `${row.modes[index].toFixed(1)} cm⁻¹${imaginary.has(String(index)) ? " (imaginary)" : ""}`,
        }));
    },

    // ---------------- structure viewer ----------------
    // Loaded on first use, never at boot: the bundle is six times the rest of the
    // frontend put together, and someone who never opens this tab never pays for it. The
    // promise is cached, so two fast clicks inject one script tag.
    loadViewerLib() {
      if (!this._glLib) {
        this._glLib = new Promise((resolve, reject) => {
          if (window.$3Dmol) return resolve(window.$3Dmol);
          const tag = document.createElement("script");
          tag.src = "static/vendor/3dmol.min.js"; // relative, like every other asset here
          tag.onload = () => {
            // A bundle that loads but defines nothing used to resolve `undefined`, and
            // every later Show then returned at mountViewer's guard in silence — no note,
            // no flash, for the rest of the session. Load and define are separate facts.
            if (window.$3Dmol) return resolve(window.$3Dmol);
            this._glLib = null;
            reject(new Error("the 3Dmol bundle loaded but defined nothing"));
          };
          tag.onerror = () => {
            // Drop the cached rejection: keeping it would make one dropped request
            // disable the Structure tab until the page is reloaded.
            this._glLib = null;
            reject(new Error("could not load the 3Dmol bundle"));
          };
          document.head.appendChild(tag);
        });
      }
      return this._glLib;
    },
    // `force` is for the callers that have something to draw regardless of a workflow —
    // a dropped or opened file. Without it, opening the tab with nothing saved would fetch
    // half a megabyte of viewer to render a sentence about saving first.
    async mountViewer({ force = false } = {}) {
      if (this.staticMode) return; // the playground has no server to ask for geometry
      if (!force && !this.savedPath) return;
      try {
        const lib = await this.loadViewerLib();
        const host = document.getElementById("viewer");
        if (!host) return;
        if (!this._gl) {
          // `antialias` is not the geometry-smoothing flag it reads as. The bundle renders
          // the scene into its own framebuffer and then blits it with one of two shaders —
          // `this._antialias ? screenaa : screen` — and `screenaa` is an FXAA pass. FXAA
          // finds high-contrast edges and redistributes pixels along them, which is exactly
          // the wrong treatment for black bold glyphs on white: it is what turns the numbers
          // uneven and blocky. The labels are in that pass, because the depth-tested sprites
          // are drawn before renderFrameBuffertoScreen().
          //
          // Turning it off costs nothing here and `upscale` is why. The GLViewer constructor
          // defaults `antialias` to true and the Renderer then defaults `upscale` to whatever
          // `antialias` is, so dropping one silently drops the other; passing it explicitly
          // keeps the backing store at >= 2 device pixels per CSS pixel. That 2x supersample
          // is what actually smooths the sphere and stick silhouettes on an ordinary display,
          // and `screen` is a straight 1:1 texture fetch that leaves the glyphs alone.
          this._gl = lib.createViewer(host, {
            backgroundColor: "white",
            antialias: false,
            upscale: true,
          });
          // The only camera hook in the bundle. It fires from inside show(), *after*
          // renderer.render(), so what it writes reaches the screen one show() later —
          // hence the second show() here, and hence the guard, because that show()
          // re-enters this very callback and would otherwise recurse without end.
          this._gl.setViewChangeCallback(() => {
            if (this._labelSync) return;
            this._labelSync = true;
            try {
              if (this._syncLabels()) this._gl.show();
            } finally {
              this._labelSync = false;
            }
          });
        }
        // Both paths, every time: a viewer built while the tab was hidden still has to be
        // told the container has a size now.
        this._gl.resize();
        this._gl.render();
      } catch (err) {
        // createViewer throws a bare string on a WebGL failure, not an Error.
        this.viewer.note = `the 3D viewer could not start: ${err.message || err}`;
      }
    },
    // ---------------- a single structure file, on its own ----------------
    // Deliberately not loadConfigFrom(): that opens a *workflow* and brings its tree with
    // it — steps, cache, seeds, the Run panel. This answers "what is in this file" and
    // stops, which is why it neither touches savedPath nor disturbs whatever workflow is
    // open. The two doors look similar and must never become one.
    async showStructureFile(body, label) {
      this.viewer.busy = true;
      this.viewer.note = "";
      try {
        if (this._gl) this._gl.stopAnimate();
        const data = await this.api("POST", "/api/structure-file", body);
        if (!data) {
          this.viewer.note = `could not read ${label} — see the message below`;
          return;
        }
        await this.mountViewerFor(data.text);
        // The formula and the atom count, because a file that parsed into the wrong thing
        // (a unit cell where a supercell was meant) looks fine and reads wrong.
        const cell = data.periodic ? ", periodic" : "";
        this.viewer.note = `${data.path} — ${data.formula}, ${data.atoms} atoms${cell}`;
      } finally {
        this.viewer.busy = false;
      }
    },
    // Draw extended-XYZ text that came from anywhere. The tail showStructure() shares,
    // minus the animation: a file on disk carries no mode to play.
    // Draw extended-XYZ text, from wherever it came. `force` is what tells mountViewer it
    // has something to show without a workflow open — a dropped or opened file — and is off
    // for the pane's own Show, which cannot be reached without one.
    //
    // Both draw paths come through here. They used to be eight identical statements written
    // twice, with a comment on this one claiming the sharing that was not happening, and
    // they had already drifted.
    async mountViewerFor(text, { force = true } = {}) {
      await this.mountViewer({ force });
      if (!this._gl) return;
      this.discardLabels();
      this._gl.removeAllModels();
      this._model = this._gl.addModel(text, "xyz");
      this._gl.setStyle({}, MODEL_STYLE);
      // The cell and any labels together — drawLabels() re-adds the cell, because clearing
      // labels clears its a/b/c corner labels with them.
      this.drawLabels();
      this._gl.zoomTo();
    },
    // A file dropped from the user's own machine, which over a forwarded port is not the
    // machine this server runs on: the browser hands over contents and a basename, never
    // a path, so the contents are what travel.
    async dropStructure(event) {
      this.viewer.dragging = false;
      const file = event.dataTransfer?.files?.[0];
      if (!file) return;
      const text = await file.text();
      await this.showStructureFile({ name: file.name, text }, file.name);
    },
    // A file on the machine the server runs on, through the same browser modal everything
    // else uses — which is also the only way to reach a cluster's own files from here.
    async openStructureFile() {
      await this.openBrowse({
        mode: "pick",
        title: "Open a structure file…",
        onPick: (path) => this.showStructureFile({ path }, path),
      });
    },

    // Pick a numbering. Only drawLabels(), deliberately: it relabels a model that is already
    // there and returns at once when there is none, so choosing a numbering with an empty
    // pane costs nothing. Calling mountViewer() here would fetch half a megabyte of viewer
    // in order to label no atoms.
    chooseLabels(mode) {
      this.viewer.labels = mode;
      this.drawLabels();
    },

    // Take every label off the viewer and free what it holds. One method, because the two
    // halves have to happen together and in this order: removeAllLabels() detaches the
    // sprites and renders once, and only then is it safe to delete the textures they were
    // drawing with. It does not free them itself — it splices its array and leaves every
    // Label, its SpriteMaterial, its GL texture and its backing canvas alive — so without
    // this a session that opens ten structures keeps the labels of all ten.
    //
    // Clearing `_atomLabels` is the other half. It used to be left set when the viewer was
    // emptied, which meant every camera move for the rest of the session re-sized and
    // re-pushed sprites that were no longer in the scene, and forced one extra full show()
    // on a blank canvas to do it.
    discardLabels() {
      if (!this._gl) return;
      const stale = this._gl.labels.slice();
      this._gl.removeAllLabels();
      for (const label of stale) label.dispose();
      this._atomLabels = [];
      this._labelViewKey = "";
    },
    // The cell and the atom labels, redrawn together. They are one operation because
    // removeAllLabels() takes the a/b/c corner labels addUnitCell adds down with the atom
    // ones — so clearing atom numbering would silently remove a periodic structure's box.
    // Cheap enough to call on every change: it relabels an existing model, nothing refetches.
    drawLabels() {
      if (!this._gl || !this._model) return;
      this.discardLabels();
      // No branch of our own for periodic vs molecular: 3Dmol draws a box only when the
      // extended XYZ carried a Lattice="…", and nothing when it did not.
      this._gl.addUnitCell(this._model);
      if (this.viewer.labels !== "off") {
        // One tally per redraw, so element ordinals restart with the structure rather than
        // climbing across every one that has been shown.
        const counts = {};
        const mode = this.viewer.labels;
        for (const atom of this._model.selectedAtoms({})) {
          const text = atomLabel(atom, mode, counts);
          if (text === null) continue;
          // One stylespec object per label, and that is not fussiness. addPropertyLabels()
          // builds a single spec and rewrites its `.position` for each atom in turn, while
          // Label keeps the spec by reference (`this.stylespec = t || {}`) — so every label
          // it makes shares the LAST atom's coordinates. Invisible until something calls
          // setContext() again, which the bundle does by itself when a lost WebGL context
          // comes back, and then the whole numbering piles onto one atom.
          const label = this._gl.addLabel(
            text,
            { ...LABEL_STYLE, position: { x: atom.x, y: atom.y, z: atom.z } },
            undefined,
            true, // no show() per label: _syncLabels() and the render() below are the redraw
          );
          // Not a stylespec key — setContext() forwards only map, useScreenCoordinates,
          // alignment, depthTest and screenOffset — so the material is written directly.
          // Without it every antialiased glyph edge writes depth and bites a hole in
          // whatever draws after it.
          label.sprite.material.depthWrite = false;
          this._atomLabels.push({
            label,
            sprite: label.sprite,
            fontSize: LABEL_STYLE.fontSize,
            // The ink's size in texels, not the canvas's. setContext() gives the canvas
            // `1.25 * fontSize + 2 * padding` of height and `measureText(text) + 2 * padding`
            // of width (borderThickness is forced to 0 when showBackground is false), so the
            // glyphs own LABEL_CAP_EM * fontSize of the first and all but the padding of the
            // second. Sizing off these is what lets one budget bound the height and another
            // bound the width — the canvas box could only ever express the height, which is
            // why a three-character label used to sprawl.
            inkHeight: LABEL_CAP_EM * LABEL_STYLE.fontSize,
            inkWidth: Math.max(1, label.canvas.width - 2 * LABEL_STYLE.padding),
            anchor: { x: atom.x, y: atom.y, z: atom.z },
            // Its own radius, not a flat margin: the push is also the depth a neighbour has
            // to beat to occlude this label, so it should be as small as will clear.
            push: this._model.getRadiusFromStyle(atom, { scale: SPHERE_SCALE }) + LABEL_CLEARANCE,
          });
        }
      }
      this._syncLabels();
      this._gl.render();
    },

    // Put every label at the size and depth this camera calls for, and say whether anything
    // moved — so the many show() calls that are not camera moves (removeAllLabels, addLabel,
    // setBackgroundColor, a resize to the same size) cost nothing.
    //
    // Float writes only. SpritePlugin re-reads sprite.scale and the sprite's matrix on every
    // frame, so nothing here touches a canvas or uploads a texture. The rebuild route —
    // setLabelStyle() with a larger fontSize — would cost a backing-store reallocation, a
    // rasterise and a texImage2D per label per frame, which is why it is not taken. It would
    // not leak: setLabelStyle() calls dispose() before setContext(), and Texture.dispose()
    // dispatches the event the renderer registered deleteTexture against. A bare setContext()
    // is the one that leaks, and nothing here calls it.
    // Redraw one label's texture at `fontSize` texels and take its new measurements.
    //
    // Mirrors what the bundle's own setLabelStyle() does — remove, dispose, restyle,
    // setContext, add — minus its per-label show(), because the caller redraws once for all
    // of them. The dispose is not optional: setContext() replaces the canvas, the material
    // and the Texture and frees none of them, so a rebuild without it leaks a GL texture per
    // label per rebuild. setContext() also resets the sprite's scale and position, which the
    // caller sets immediately afterwards, and builds a fresh material, so depthWrite has to
    // be re-applied here.
    _rasteriseLabel(entry, fontSize) {
      const { label } = entry;
      this._gl.modelGroup.remove(label.sprite);
      label.dispose();
      label.stylespec = { ...label.stylespec, fontSize };
      label.setContext();
      this._gl.modelGroup.add(label.sprite);
      label.sprite.material.depthWrite = false;
      entry.fontSize = fontSize;
      entry.inkHeight = LABEL_CAP_EM * fontSize;
      entry.inkWidth = Math.max(1, label.canvas.width - 2 * LABEL_STYLE.padding);
    },

    // The scale that fits this label's ink inside both budgets and the legibility bounds.
    _labelScale(entry, perAngstrom) {
      return Math.min(
        LABEL_INK_MAX_PX / entry.inkHeight,
        Math.max(
          LABEL_INK_MIN_PX / entry.inkHeight,
          Math.min(
            (LABEL_INK_HEIGHT * perAngstrom) / entry.inkHeight,
            (LABEL_INK_WIDTH * perAngstrom) / entry.inkWidth,
          ),
        ),
      );
    },

    _syncLabels() {
      if (!this._gl || !this._atomLabels.length) return false;
      const view = this._gl.getView(); // [x, y, z, zoom, qx, qy, qz, qw]
      // The whole view, not a chosen subset of it. view[0..2] is modelGroup.position, and it
      // used to be left out on the grounds that panning slides the model and the camera
      // together — true, and enough, while every anchor was pushed along one shared view
      // axis, because a pan cannot turn that axis. It is not enough for a push along each
      // atom's own ray: panning changes where the camera sits relative to each atom, so a
      // pan that returned early here would leave every label on the ray it wanted before the
      // pan — 4.4 px off its atom at the default zoom, the very error the ray push removes.
      const key = `${view.join("|")}|${this._gl.HEIGHT}`;
      if (key === this._labelViewKey) return false;
      this._labelViewKey = key;

      // Toward the camera, in the coordinates the anchors are written in. modelGroup carries
      // only a translation and rotationGroup carries the whole rotation, so this is the third
      // ROW of the matrix the view quaternion builds — R transposed applied to the world +z
      // the camera looks down, which is the inverse rotation because R is orthonormal.
      // Unit length by construction, so it needs no normalising.
      //
      // The factors of two are not decoration. Without them the vector is still exactly
      // right for the identity and for any rotation about z, and wrong everywhere else —
      // zero for a half-turn about y — so labels would sit correctly until the moment the
      // structure was turned, and then be swallowed by their own atoms.
      const [qx, qy, qz, qw] = view.slice(4);
      const nx = 2 * (qx * qz - qw * qy);
      const ny = 2 * (qy * qz + qw * qx);
      const nz = 1 - 2 * (qx * qx + qy * qy);

      // CSS pixels per Angstrom at the model plane. The projection's vertical half-angle is
      // fov/2 — makePerspective takes tan(fov/2) — and HEIGHT is already CSS pixels, because
      // the device pixel ratio cancels out of the sprite's own size formula.
      const distance = Math.max(1, this._gl.CAMERA_Z - view[3]);
      // The projection's vertical half-angle — makePerspective takes tan(fov/2) — with
      // HEIGHT already in CSS pixels, because the device pixel ratio cancels out of the
      // sprite's own size formula. Hoisted because the only term that varies from one label
      // to the next is the depth underneath it.
      const tanHalfFov = Math.tan((Math.PI / 360) * this._gl.fov);
      const pxPerAngstromAt = (depth) => this._gl.HEIGHT / (2 * depth * tanHalfFov);

      // The camera, in the coordinates the anchors are written in. Labels hang off
      // modelGroup, which carries only the translation view[0..2]; rotationGroup carries the
      // whole rotation R and the translation (0, 0, view[3]); and the camera itself never
      // moves from (0, 0, CAMERA_Z). Setting R*(C + t) + (0,0,view[3]) equal to that gives
      // C = distance*n - t, reusing the n above — which is R transposed applied to +z, i.e.
      // the direction of the camera, not of the push.
      const cx = nx * distance - view[0];
      const cy = ny * distance - view[1];
      const cz = nz * distance - view[2];

      // Device pixels the renderer actually puts on screen per CSS pixel. Read off the
      // renderer rather than off `window`, because `upscale` holds it at two or more and
      // that is what the label texture is really being stretched across.
      const dpr = this._gl.getRenderer().devicePixelRatio || 1;

      for (const entry of this._atomLabels) {
        // Toward the camera from this atom. One vector read twice: the push below wants its
        // direction, the size wants its component along the view.
        const dx = cx - entry.anchor.x;
        const dy = cy - entry.anchor.y;
        const dz = cz - entry.anchor.z;

        // This atom's OWN depth — exactly the `w` the perspective divide uses for a point at
        // the anchor, not an approximation of it. That is the whole of this: it is the same
        // w the sphere beside it is drawn with. The sphere imposter offsets its billboard in
        // CLIP space and zeroes adjust.z, so all four corners carry the centre's w and the
        // drawn radius is exactly r/depth. One scene-wide figure, which is what this used to
        // compute, therefore cannot track anything — at 3Dmol's own default zoom the nearest
        // atom of an 8 A-deep molecule sits 14% nearer than the rotation centre and the
        // farthest 14% further, so a hydrogen at the front drew 0.52 of its own blob and an
        // identical hydrogen at the back drew 0.68 of its own. Zoomed in three times that
        // spread is 0.35 against 0.85. On this depth the ratio is LABEL_INK_HEIGHT / 2r at
        // every depth, every orientation, every zoom and every pane size.
        //
        // The anchor, never the pushed sprite position. The push exists to clear the label
        // from its own sphere in the depth buffer, and it is element-dependent — 0.45 A for
        // hydrogen against 0.575 for carbon — so sizing off it would restore a smaller copy
        // of this same bug (+9.9% at 5 A, +2.3% at 20 A) and make an H and a C at one depth
        // differ in size for no physical reason.
        const depth = dx * nx + dy * ny + dz * nz;

        // Hidden rather than clamped, and hidden exactly when its own sphere is. Both are
        // clipped whole rather than sliced — the sprite shader forces w = 1 before adding
        // the quad's corners, the imposter zeroes adjust.z, so in each case all four corners
        // share one z — and setSlabAndFog() holds camera.near at 1 or more unconditionally.
        // Clamping instead would hand _labelScale a depth nobody can see, pin the scale to
        // its ceiling and drive a full texture rebuild for an invisible label, then another
        // on the way back. Reachable in ordinary use: each wheel notch multiplies the
        // distance by 0.68, so six of them from the default zoom put a front atom behind the
        // camera.
        entry.sprite.visible = depth >= 1;
        if (!entry.sprite.visible) continue;

        // Fit the ink box inside both budgets, then hold it between the legibility bounds.
        // Two budgets, because one number cannot bound a box whose width is the text's.
        const perAngstrom = pxPerAngstromAt(depth);
        let k = this._labelScale(entry, perAngstrom);

        // One texel per device pixel is a sharp label; anything else is a resampled one.
        // Re-rasterise when it has drifted out of the band, at the size it is actually being
        // drawn — this is the only thing that makes a label as crisp as the spheres beside
        // it, which have no texture to outgrow.
        const texels = k * dpr;
        if (texels > LABEL_TEXELS_HI || texels < LABEL_TEXELS_LO) {
          const wanted = Math.round(entry.fontSize * texels);
          const fontSize = Math.min(LABEL_FONT_MAX, Math.max(LABEL_FONT_MIN, wanted));
          if (fontSize !== entry.fontSize) {
            this._rasteriseLabel(entry, fontSize);
            k = this._labelScale(entry, perAngstrom);
          }
        }
        entry.sprite.scale.set(k, k, 1);

        // Along this atom's own ray to the camera, never along a shared axis. The sprite
        // shader projects the anchor and divides by w before it adds the quad's corners, so
        // every point on the line from the eye through an atom lands on the same pixel: move
        // the anchor along that line and the label stays exactly on its atom, at every
        // orientation and zoom. Pushing along the view axis instead leaves x and y untouched
        // in eye space while shortening the depth, which under the perspective divide throws
        // the label radially outward — nothing at the centre of the pane, most at the edge,
        // and swinging around as the structure turns. That is the drift, and this is its
        // exact cure, not an approximation of one.
        const ray = Math.hypot(dx, dy, dz) || 1;
        const step = entry.push / ray;
        entry.sprite.position.set(
          entry.anchor.x + dx * step,
          entry.anchor.y + dy * step,
          entry.anchor.z + dz * step,
        );
      }
      return true;
    },
    async showStructure() {
      if (this.playgroundRefuses("the structure view reads a run tree, so it")) return;
      if (!this.savedPath || !this.viewer.step) return;
      this.viewer.busy = true;
      this.viewer.note = "";
      try {
        // No `step` at all is what asks for the input seeds; sending step="input" would
        // be read as a step *name*, since a step may be named anything. apiUrl() drops
        // every empty value, so the sentinel is the only case needing a line of its own.
        const url = apiUrl("/api/structure", {
          config_path: this.savedPath,
          step: this.viewer.step === "input" ? "" : this.viewer.step,
          structure_id: this.viewer.structureId,
          mode_index: this.viewer.step === "input" ? "" : this.viewer.modeIndex,
        });
        // Stop the old animation before anything can return early, or a failed Show
        // leaves the previous structure oscillating as though it were the answer.
        if (this._gl) this._gl.stopAnimate();
        const data = await this.api("GET", url);
        if (!data) {
          // Into the pane's own line as well as `flash`, which lives below the Run panel
          // and the report — a screen away from where the user is looking. Blanking this
          // and saying nothing is what made a failure look like nothing happening.
          this.viewer.note = "could not show that — see the message below";
          // A mode that cannot be drawn must not follow the user to the next structure.
          // It is the stickiness of this one field that turned one bad request into
          // "now I cannot show anything at all".
          this.viewer.modeIndex = "";
          return;
        }
        await this.mountViewerFor(data.text, { force: false });
        if (!this._gl) return;
        // From the answer, not from the form: the server decides whether a mode came back,
        // and reading the boxes again here is how the two could disagree.
        if (data.mode_index !== null && data.mode_index !== undefined) {
          // The same extended-XYZ file carries three displacement columns per atom, which
          // is what 3Dmol reads as dx/dy/dz; vibrate() only builds the frames, animate()
          // plays them.
          this._model.vibrate(10, 1, true);
          this._gl.animate({ loop: "backAndForth", interval: 60 });
        }
        this._gl.render();
        const where = data.step === null ? "seed" : `step ${data.step}`;
        this.viewer.note = `${where} · ${data.structure_id}`;
      } finally {
        this.viewer.busy = false;
      }
    },

    // ---------------- agent chat ----------------
    saveChatSettings() {
      // Three keys, not four: `chat.apiKey` is a live credential and is never persisted.
      // Adding it here would be the natural-looking edit and the wrong one.
      for (const [field, key] of Object.entries(CHAT_KEYS)) {
        localStorage.setItem(key, this.chat[field]);
      }
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
      if (this.playgroundRefuses("the agent drives a real tree, so it")) return;
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
    // Answers whether the path was dealt with: false means it did not load and the caller
    // is still holding the only route back to it (the browser dialog, or the offer banner).
    async loadConfigFrom(path, { force = false, reason = "agent" } = {}) {
      if (this.staticMode) return true; // the playground has no server to read a file from
      if (!force && (this.dirty() || this.rawEdit)) {
        // rawEdit counts as dirty even when the text matches: syncYaml() early-returns
        // while raw editing, so adopting underneath it would desync the two panes.
        this.pendingReload = { path, reason };
        return true; // handed over to the offer, which is now the route back
      }
      const data = await this.api("GET", apiUrl("/api/load", { path }));
      if (!data) return false; // flash explains; the form keeps what it had
      if (!(await this.adopt(data.yaml_text))) return false;
      this.savedPath = data.path;
      this.pendingReload = null;
      this.rawEdit = false;
      this.recents = rememberRecent(this.recents, data.path);
      this.forgetPreviousWorkflow();
      await this.recordOnDisk();
      // Only the agent's own writes steal the tab. Open… is how you go and *look* at a
      // finished run, and yanking the pane back to the YAML is precisely the wrong answer
      // when the tab you opened it on was the Structure one.
      if (reason !== "open") this.showTab("right", "yaml");
      this.flash =
        reason === "open"
          ? `loaded ${data.path}`
          : `the agent wrote ${data.path} — loaded into the builder`;
      return true;
    },

    // Everything on the page that describes the workflow that *was* loaded rather than the
    // form itself. Without this, Open… left the previous run's status table, failure count
    // and results rows on screen under the new file's name — they only refresh when the Run
    // panel is toggled — and pointed the viewer at a step number the new config may not have.
    forgetPreviousWorkflow() {
      this.runStatus = null;
      this.runFailures = null;
      this.runResults = null;
      this.resultsStep = "";
      this.report = null; // validation of a file that is no longer the one in the form
      this.tmpl = { open: false, step: null, path: "", text: "" };
      this.viewer = {
        ...this.viewer,
        step: "input",
        structureId: "",
        modeIndex: "",
        note: "",
        structures: [],
      };
      this.runTarget = ""; // a step number that meant something in the workflow just closed
      this.loadStructureList(); // savedPath is the new file by now, so these are its seeds
      // Nulling runStatus empties the panel; only its own @toggle and Refresh button ever
      // refill it, so an expanded Run panel went blank on Open… and stayed blank.
      this.refreshStatus();
      // The drawn molecule belongs to the old tree too; leaving it up (still animating)
      // reads as the new workflow's answer.
      if (this._gl) {
        this._gl.stopAnimate();
        this.discardLabels();
        this._gl.removeAllModels();
        // Detached with the models it points at. drawLabels() guards on `_model`, so
        // leaving it set meant choosing a numbering on the now-blank canvas re-labelled —
        // and re-boxed — the structure that had just been cleared off it.
        this._model = null;
        this._gl.render();
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
      const offer = this.pendingReload;
      if (!offer) return;
      // Cleared first so loadConfigFrom() cannot re-offer the same path to itself, but put
      // back if the load fails: the banner is the only thing that still names that file,
      // and losing it on a failed accept lost the agent's write for good.
      this.pendingReload = null;
      const loaded = await this.loadConfigFrom(offer.path, {
        force: true,
        reason: offer.reason ?? "agent",
      });
      if (!loaded) this.pendingReload = offer;
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
        if (data.wrote_config) await this.loadConfigFrom(data.wrote_config);
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
      this.chat.provider = CHAT_DEFAULTS.provider;
      this.chat.model = "";
      this.chat.baseUrl = "";
      this.chat.apiKey = "";
      this.armCheck(); // the verdict belonged to settings that no longer exist
      for (const key of Object.values(CHAT_KEYS)) localStorage.removeItem(key);
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
    // browseField, beside browseExec: both fill one path field from the picker. It was
    // openBrowse, which collided with the modal opener below — same name, different job.
    async browseField(fieldKey) {
      await this.openBrowseWith(`Pick ${fieldKey}`, (path) => {
        this.cfg[fieldKey] = path;
        this.syncYaml();
      });
    },
    // Where a modal should open: beside the file being worked on, since the next path is
    // almost always a sibling of it. Only Open… did this; Save… and the field pickers
    // always started at $HOME even with a config open elsewhere on the disk.
    startDir() {
      return this.savedPath ? parentDir(this.savedPath) : null;
    },
    // The one opener. Its three callers differed only in a mode string, a title and
    // whether they carried an onPick — three spread-and-navigate copies whose only
    // meaningful difference had been the one that remembered to start beside the file.
    async openBrowse({ mode, title, onPick = null }) {
      this.browse = { ...this.browse, open: true, mode, onPick, title };
      await this.navigate(this.startDir());
    },
    async openBrowseWith(title, onPick) {
      await this.openBrowse({ mode: "pick", title, onPick });
    },
    async openConfig() {
      // The counterpart to Save…, and the only way to look at a finished run without
      // restarting: `chemrefine gui <path>` was the single door a config could come
      // through, and it only opens once, at launch.
      await this.openBrowse({ mode: "open", title: "Open a workflow…" });
    },
    async openConfigFrom(path) {
      // Closed only once the file has actually loaded, the way saveTo() does it. Closing
      // first meant a directory, an unreadable file or a bad ~user dismissed the browser
      // and threw away wherever you had navigated to, leaving a flash and no way back.
      if (await this.loadConfigFrom(path, { reason: "open" })) this.browse.open = false;
    },
    async openSave() {
      await this.openBrowse({ mode: "save", title: "Save workflow as…" });
    },
    async navigate(path) {
      const data = await this.api("GET", apiUrl("/api/browse", { path }));
      if (data) {
        this.browse.path = data.path;
        this.browse.parent = data.parent;
        this.browse.entries = data.entries;
        // The box follows the listing, so it always shows where you are and is a starting
        // point to edit rather than an empty field to retype.
        this.browse.typed = data.path;
      }
    },
    // What the typed box means depends on what the modal is for, and the server is the one
    // that knows whether a path is a directory: /api/browse answers with the listing for a
    // directory and a 400 for anything else. So a directory navigates, and anything that is
    // not one is treated as the chosen file — which is how a full path pasted straight in
    // opens a workflow, fills a path field, or names a save target, without walking to it.
    async goToTyped() {
      const typed = this.browse.typed.trim();
      if (!typed) return;
      const listing = await this.api("GET", apiUrl("/api/browse", { path: typed }), undefined, {
        quiet: true,
      });
      if (listing) {
        this.browse.path = listing.path;
        this.browse.parent = listing.parent;
        this.browse.entries = listing.entries;
        this.browse.typed = listing.path;
        return;
      }
      // Not a directory: the same three jobs pickFile() does, on a path rather than a row.
      // `name` is only read in save mode, where it is the filename box's new contents.
      // The whole path as the name, not its basename. joinPath() exists so that an absolute
      // or ~ name replaces the directory rather than hanging off it, and stripping to the
      // basename here took that decision away from it: typing an absolute path in save mode
      // wrote the file into whatever directory the listing happened to be showing.
      await this.pickFile({ path: typed, name: typed });
    },
    pickHere() {
      // Directory fields (template_dir, output_dir, scratch_dir) are chosen by
      // navigating *into* the folder and taking it — clicking a file never applied.
      if (this.browse.onPick) this.browse.onPick(this.browse.path);
      this.browse.open = false;
    },
    // `async` so a caller can wait for the open to finish — goToTyped() needs to, and the
    // tests that drive the click path had to sleep a microtask to work around it not being.
    // The other two branches are still synchronous: an async function's body runs straight
    // through until its first await.
    async pickFile(entry) {
      if (this.browse.mode === "open") {
        await this.openConfigFrom(entry.path);
      } else if (this.browse.mode === "pick" && this.browse.onPick) {
        this.browse.onPick(entry.path);
        this.browse.open = false;
      } else {
        this.browse.filename = entry.name;
      }
    },
    async saveTo() {
      const path = joinPath(this.browse.path, this.browse.filename || "input.yaml");
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
        apiUrl("/api/template", {
          config_path: this.savedPath,
          step: key,
        }),
      );
      // No hint appended on failure: `api()` has already written the server's own reason,
      // and read_template's is "template … does not exist yet (scaffold_templates writes a
      // starter)" — which says it better. The old ` += " — run Scaffold templates first?"`
      // was redundant when that was the reason and nonsense when it was not, gluing itself
      // onto "the ChemRefine server is unreachable — is `chemrefine gui` still running?".
      if (data) this.tmpl = { open: true, step: step.step, path: data.path, text: data.text };
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
