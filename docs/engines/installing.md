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
    checkout.

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
| `Server crashed during startup` (MLIP) | Check the per-job `server_${SLURM_JOB_ID}.log`; common causes are out-of-memory at model load or a missing Hugging Face token for FAIRChem. |
