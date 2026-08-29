# Installing engines & backends

ORCA and Q-Chem are programs you install yourself and point ChemRefine at with
`executables:`. Everything else — the machine-learned potentials and PySCF — is a Python
stack, and ChemRefine can install it for you.

## One environment, or one per backend

The MLIP libraries have mutually incompatible dependency stacks (`e3nn==0.4.4` against
`>=0.5`, competing torch trees), so one Python process can only host one backend family.
Two ways to add one:

- **A single backend** — install its extra straight into the ChemRefine environment and
  run as usual:

    ```bash
    pip install "chemrefine[mlip]"        # FAIRChem / UMA (the default backend)
    ```

- **Several conflicting backends** — keep the core light and provision one *managed
  environment* per backend:

    ```bash
    chemrefine backends install mlip-mace mlip-fairchem
    ```

    Each environment is built once with the **same tool that created the environment you
    have activated** (conda / uv / venv, detected automatically), reused by every later
    run, and resolved **by name** at run time — no interpreter paths in your YAML.
    They are built **from the same source as the orchestrator**: a PyPI install pins the
    version, a Git or `pip install -e .` install reinstalls from that same repo or
    checkout — and on the **Python that backend supports**, which is not always yours (see
    below).

`chemrefine backends list` shows what is provisioned. Every run validates its steps'
backends **up front**: a step whose backend is neither importable nor provisioned fails
before any job is submitted, naming the fix.

## Available backends

These are the names `chemrefine backends install` accepts — and the only ones it accepts.
`[mlip]` and `[mlff]` are pip extras that alias `[mlip-fairchem]`; neither is a backend
name.

<!-- chemrefine:backends -->

The backends don't pin a CUDA build of `torch`, so for GPU install the matching `torch`
first (or let the extra resolve the default build). FAIRChem's default checkpoint is
`uma-s-1p2`. PySCF is the exception to one-environment-per-name: `pyscf` and `pyscf-gpu`
deliberately share one, because the GPU extra is a strict superset and two would duplicate
a large libcint/libxc tree for nothing.

`task_name` is the only key that selects a backend. To run a model you fine-tuned
yourself, name the library that trained it and point `model_path` at the checkpoint —
there is no `custom_<library>` task for any of them. (`custom_mace` still resolves, as a
MACE alias kept for v1 configs.)

## Each backend gets the Python it supports

A backend's dependency stack is isolated so it never has to match anyone else's, and the
**interpreter is part of that stack**. Some backends have no wheels for the newest Pythons —
today `mlip-orb` installs only on 3.12 (orb-models pins `dm-tree==0.1.8`, whose newest wheels
are cp312) and `mlip-chgnet` only up to 3.12 (chgnet ships no cp313 wheel), while `mlip-mace`
stops at 3.13 (its pinned torch line has no cp314 wheel). Without wheels, pip falls back to
building from source, which for those packages means compiling C++ or torch itself.

So `chemrefine backends install` builds each environment on the newest Python that backend's
extra installs on, and says which one it picked:

```console
$ chemrefine backends install mlip-orb        # on a 3.13 orchestrator
provisioning mlip-orb on Python 3.12 …
mlip-orb: ~/venvs/chemrefine/share/chemrefine/backends/mlip-orb/bin/python
```

The path shown is the default for a writable (venv/conda) install — the envs live
beside it. On a read-only or system Python they land under
`~/.chemrefine/<interpreter tag>/backends/` instead (e.g. `~/.chemrefine/cpython-313/…`
— tagged, because `$HOME` is routinely shared across clusters, and two machines on
different Pythons must not resolve each other's envs). `$CHEMREFINE_HOME` overrides
both.

Nothing about your run changes: the environment is still resolved by name, and a step still
launches `<env>/bin/python`. Where the interpreter comes from depends on the tool:

- **conda** resolves `python=3.12` from its channels, and **uv** downloads a managed build —
  neither needs anything on the machine.
- a plain **venv** needs a `python3.12` on `PATH` (a distribution package, pyenv, Homebrew, a
  loaded HPC module). If there is none but a `uv` binary is on `PATH`, that is used instead;
  if there is neither, the command says so and stops, rather than starting a doomed build.

`--python` overrides the choice with a version, a command name, or a path:

```bash
chemrefine backends install mlip-orb --python /opt/python3.12/bin/python3
```

It applies only when an environment is **created**. An existing one is extended on the Python
it already has — and if that Python is one the backend cannot install on (an environment
built before a cap was known), the command refuses and tells you to remove the directory,
because installing into it would report success having installed nothing.

The same reason makes the single-environment route unavailable for those backends on a Python
they exclude: `pip install "chemrefine[mlip-orb]"` on 3.13 succeeds and installs no orb, since
every requirement the extra declares is excluded there. Use the managed environment.

## Several backends in one run

Because each step resolves its own backend environment by name, one pipeline can mix
models whose dependency stacks conflict — e.g. a broad MACE screen refined by UMA:

```bash
chemrefine backends install mlip-mace mlip-fairchem   # once
```

```yaml
steps:
  - step: 1                       # broad screen — MACE
    engine: mlip
    operation: opt_sp
    options: { task_name: mace_off, model_name: medium }
    sample: { method: boltzmann, percent_cumulative: 99 }

  - step: 2                       # refine — UMA
    engine: mlip
    operation: opt_sp
    options: { task_name: omol, model_name: uma-s-1p2 }
    sample: { method: min, count: 5 }
```

Step 1 runs each structure with the `mlip-mace` environment's Python, step 2 with
`mlip-fairchem`'s — the conflicting stacks never share a process. The same works for
`mlip-extopt` steps (the gradient server launches from the step's environment) and for
`pyscf`. The `options.backend_python` knob overrides the resolution with an explicit
interpreter (an escape hatch — normally environments are resolved by name only).

## Every pip extra

The full `pip install "chemrefine[…]"` vocabulary, including the ones that are not compute
backends:

<!-- chemrefine:extras -->

`[server]` is flask + waitress — the ExtOpt HTTP server — and every backend extra pulls it
in. `[gui]` is the [workflow builder](../workflow/builder.md), `[mcp]` and `[agent]` the
two [AI agent](../workflow/agents.md) routes; `[dev]`, `[docs]` and `[test]` are for
working on ChemRefine itself. `[pyscf-gpu]` targets CUDA 12 — CUDA-11 hosts swap in the
`-cuda11x` wheels.

## FAIRChem model access

The UMA / OMol checkpoints are gated on Hugging Face: request access to the
[UMA repository](https://huggingface.co/facebook/UMA) and authenticate your machine with a
Hugging Face token. Follow the
[FAIRChem documentation](https://fair-chem.github.io/) for the current access +
authentication steps — they are defined upstream and may change.

## If a backend will not install

| Symptom | Likely cause |
|---------|--------------|
| `backend '…' is not available` at run start | The step's backend is neither importable nor provisioned — run `chemrefine backends install <extra>` (on a cluster: on a login node), or install `chemrefine[<extra>]` into the main environment. |
| `No matching distribution found for chemrefine==…` during `backends install` | Upgrade ChemRefine — older versions could only provision backends from a published PyPI release; current ones reinstall from the same Git/source install as the orchestrator. |
| A C++ wall (`dm-tree`, abseil, `enum class … : uint8_t`) or a torch source build during an install | An older ChemRefine, or a hand-made environment, on a Python that backend has no wheels for. `chemrefine backends install <extra>` now builds on a Python the backend supports — see [above](#each-backend-gets-the-python-it-supports). |
| `installs on Python 3.12 … neither python3.12 nor uv is on PATH` | The env tool in use (a plain venv) cannot produce the interpreter that backend needs. Install one — `pip install uv` is enough, it downloads the rest — or pass `--python /path/to/python3.12`. |
| `was built on Python 3.13, which chemrefine[…] does not install on` | The environment predates the backend's Python constraint and holds nothing. Remove the directory it names and re-run `chemrefine backends install <extra>`. |
| `Server crashed during startup` (MLIP) | Check the per-job `server_${SLURM_JOB_ID}.log`; common causes are out-of-memory at model load or a missing Hugging Face token for FAIRChem. |
