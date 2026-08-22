# Security & Trust Boundaries

ChemRefine is a workflow driver. Several of its inputs are **executed**, not
merely parsed — which is inherent to what the tool does, and the same is true of
a Makefile or a CI config. This page states the boundary explicitly so you know
what to trust and what is hardened.

## Inputs that run as code

| Input | How it executes |
| --- | --- |
| `*.slurm.header` templates | Shell fragments spliced into the generated job script and run by `bash` / `sbatch` |
| `step{N}.py` templates (`mlip`, `pyscf`) | Rendered, then executed by the backend Python interpreter |
| `step{N}.inp` templates (`orca`) | Passed to ORCA, which can invoke external programs (e.g. ExtOpt wrappers) |
| The YAML config | Selects engines and supplies the `executables:` paths that get invoked |

**Only run configs, templates, and example directories from a source you trust.**
A malicious workflow can do anything your user account can do. That is not a
vulnerability in ChemRefine — it is the nature of a job runner.

The practical rule: treat a downloaded ChemRefine project the way you would treat
a downloaded shell script, not the way you would treat a data file.

## What *is* hardened

These are boundaries ChemRefine actively defends. A way past any of them is a
bug worth reporting.

### The ExtOpt compute server

The `*-extopt` engines start a local HTTP server that ORCA drives for gradients.

- It binds **loopback only** (`127.0.0.1`) on a **kernel-assigned port**
  (`:0`), so concurrent jobs on one node never collide and nothing is exposed
  off-host.
- `/calculate` requires a **per-run bearer token** (`secrets.token_hex(32)`),
  compared with `secrets.compare_digest` — on the **encoded bytes**, because
  headers arrive latin-1-decoded and `compare_digest` refuses a `str` holding a
  non-ASCII character, which would answer a malformed token with a 500 instead of
  a 401. `/healthz` stays open for the run block's readiness probe.
- The token sidecar is written **`0600`** via `mkstemp` + rename, so it is never
  world-readable, even transiently.

The token is what matters on a shared HPC node: loopback is reachable by *any*
user on that node, so binding locally is not by itself access control.

### The workflow-builder GUI

`chemrefine gui` starts the **second** server this package can run, and it is the
more consequential one: `/api/save` writes a file at a path the request names,
`/api/load` and `/api/structure` read from the tree, and `/api/run` launches a
detached pipeline. It is
defended the same way, for the same reason — loopback on a shared node is not
access control.

- It binds **loopback only** (`127.0.0.1`), by default on a **stable per-user port**
  hashed from the username (kernel-assigned when that one is taken; `--port 0` asks
  for a kernel port outright); the socket is bound first and the effective port read
  back off it. The predictable port is not a weakening: loopback ports are enumerable
  by any local user regardless, so the token below is the access control and the
  port number never was — what stability buys is an SSH forwarding setup written
  once. Reach it on a cluster with SSH port forwarding, not by binding wider — there
  is no flag to bind wider.
- Every `/api/*` request must carry a **per-session token**
  (`secrets.token_urlsafe(16)`), rejected in a `before_request` gate before any
  handler runs, and compared with `secrets.compare_digest` on the **encoded
  bytes** — same latin-1 reason as above, and the two were fixed a day apart.
- The token is handed over once in the URL `chemrefine gui` prints, so it reaches
  the page without a login form. It lasts only as long as the server process, and it
  is exposed in three places, not two. Two are per-user and unremarkable: the
  browser's history, and that terminal's scrollback. The third is not — **when the
  launch opens a browser for you, the tokened URL is passed as a command-line
  argument** (`webbrowser` runs `xdg-open <url>`), and on Linux `/proc/<pid>/cmdline`
  is world-readable unless the node mounts `procfs` with `hidepid`. On a shared node
  that hands the token — the *only* access control here, per the paragraph above — to
  any other user who looks while the browser is starting.

  **Use `--no-browser` on a shared or multi-user machine.** It prints the URL and
  opens nothing, so the token never reaches an `argv`. On a login node this is
  usually what happens anyway: with no graphical browser the launch declines to open
  one and prints the SSH forwarding recipe instead.
- `/` and the static assets are deliberately **ungated** — they are the same files
  the docs site publishes as the Playground and carry nothing about the tree. A
  cross-origin page cannot use that to reach the API: the token rides a custom
  header, which forces a CORS preflight the app does not answer.
- The chat panel's **API key is held in the page and nowhere else**. It is typed into
  the panel, sent with each agent request, and never written to `localStorage`, to a
  URL, or to a log — unlike the provider and model beside it, which are remembered.
  It dies with the tab; `CHEMREFINE_LLM_API_KEY` stays the way to avoid retyping it,
  and a blank field falls through to it. A key that could not be an HTTP header (a
  newline, a NUL) is refused before it becomes an `Authorization:` one.
- The chat and its preflight make **outbound requests to an endpoint the request
  names**, carrying that key. That is the point of the feature — it is how a model is
  reached — but it means the GUI is not purely local once the panel is used, and the
  scheme is constrained to HTTP(S) at the one place every caller resolves through.

### Data at rest

- The step cache is **JSON, never pickle** — loading a cache file cannot execute
  code. This is a deliberate choice, not an accident of format.
- The YAML config is read with `yaml.safe_load`, so tags cannot construct
  arbitrary Python objects.
- Caches, manifests, ledgers, and sidecars are written **atomically** (temp file
  + rename), so an interrupted run never leaves a half-written file that a later
  `resume` would misread.

### Generated job scripts

**Every config value that reaches the generated SLURM script is refused at
config-load time if it contains `"`, `$`, a backtick, a backslash, or a
newline.** Those would terminate a quoted string or introduce a command
substitution the job would then run.

The rule is asked **twice**: once on the values as written, and again on the
paths after they are resolved. The second pass is not belt-and-braces — a
relative `output_dir` is anchored to the config file's own directory, so a
*parent directory name* can carry every character the rule refuses while the YAML
itself contains none. A config holding only `output_dir: ./outputs`, placed in a
directory named `$(...)`, otherwise reaches the generated script as
`export OUTPUT_DIR="…/$(...)/outputs"` — and bash substitutes inside double
quotes. That matters most on exactly the shared filesystem this page is about.

The rule is attached to that property, not to a list of fields — a value that
reaches bash by a new route inherits it automatically:

| Value | Why it needs the rule |
| --- | --- |
| `template_dir`, `output_dir`, `scratch_dir` | interpolated into `export WORK_DIR=…` |
| `executables` | embedded raw in the runlog heredoc, which must stay unquoted so `$(hostname)` and `${SLURM_JOB_ID:-$$}` still expand |
| `operation` | the same heredoc |
| `options.tensor_folder` (pyscf) | reaches `cp -r "…"` — and bash substitutes *inside* double quotes, so the quoting there is not protection |

Other values in those same lines are safe by construction and need no check:
`engine` must be a registry key, `step.name` and the trainer's `job_name` are
matched against a character allowlist, `step`/`cores` are integers,
`structure_id` is minted by ChemRefine, and `output_globs` is an engine
constant.

Quoting alone is not the fix, and neither is quoting the heredoc: the expansion
around those fields is load-bearing. Refusing the character at the boundary is.

## Supply chain

Every release carries two artifacts beyond the package itself. **Build provenance**
(PEP 740 plus a GitHub attestation) says the wheel was built by this workflow from this
commit — verify it with `gh attestation verify`. A **CycloneDX SBOM** says what is inside
it, resolved at build time; reconstructing that from the version floors in
`pyproject.toml` after the fact gives the wrong answer, because the floors are not what
the resolver actually picked.

### Managed backend environments are not hash-pinned

`chemrefine backends install <extra>` runs an ordinary `pip install` into the managed
environment. A compromised index could therefore reach a backend env.

This is an accepted risk rather than an oversight. Closing it means a hash-pinned
lockfile per MLIP extra, and those extras track torch and CUDA builds that publish
frequently — every upstream release would break every lockfile until regenerated, which
is recurring work with no one to absorb it. Two things bound the exposure: a managed env
holds only the backend and its dependencies, never credentials or job data, and
`build_backend_env` installs the *same* ChemRefine source as the orchestrator driving it
(pinned by version for an index install, by URL or commit for a direct one), so the
package itself cannot be substituted.

If you need the guarantee, provision the environment yourself with your own pinned
requirements and point the step at it with `options.backend_python`.

## Reporting

Report vulnerabilities privately via
[GitHub Security Advisories](https://github.com/sterling-group/ChemRefine/security/advisories/new).
See the [security policy](../security-policy.md) for supported versions.
