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
  compared with `secrets.compare_digest`. `/healthz` stays open for the run
  block's readiness probe.
- The token sidecar is written **`0600`** via `mkstemp` + rename, so it is never
  world-readable, even transiently.

The token is what matters on a shared HPC node: loopback is reachable by *any*
user on that node, so binding locally is not by itself access control.

### Data at rest

- The step cache is **JSON, never pickle** — loading a cache file cannot execute
  code. This is a deliberate choice, not an accident of format.
- The YAML config is read with `yaml.safe_load`, so tags cannot construct
  arbitrary Python objects.
- Caches, manifests, ledgers, and sidecars are written **atomically** (temp file
  + rename), so an interrupted run never leaves a half-written file that a later
  `resume` would misread.

### Generated job scripts

`template_dir`, `output_dir`, and `scratch_dir` are interpolated into the
generated SLURM script inside double quotes. Paths containing `"`, `$`, or a
backtick are **rejected at config-load time** — they would otherwise terminate
the quoted string or introduce a command substitution.

## Reporting

Report vulnerabilities privately via
[GitHub Security Advisories](https://github.com/sterling-group/ChemRefine/security/advisories/new).
See [`SECURITY.md`](https://github.com/sterling-group/ChemRefine/blob/main/SECURITY.md)
for the supported-version policy.
