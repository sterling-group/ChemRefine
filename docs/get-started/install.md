# Install

ChemRefine is a **light orchestrator core** plus optional **compute backends**. The core
— the pipeline, ORCA and Q-Chem driving, SLURM submission — installs anywhere in seconds
and needs no backend environment of its own. Machine-learned potentials, PySCF,
and the Qiskit molecular and quantum-experiment engines are added afterwards;
see [Installing engines & backends](../engines/installing.md).

## Requirements

--8<-- "README.md:requirements"

Which program each engine needs is in the [engine table](../engines/index.md), along with
its options.

The base install pulls the scientific stack (`numpy`, `pandas`, `ase`, `rdkit`) plus
`pyyaml`, `pydantic`, `typer` and `packaging` — the exact version floors live in
`pyproject.toml`, held by CI's own floors job.

## Install ChemRefine

A dedicated environment is **recommended, not required**. Two reasons, both concrete: a
managed backend environment is built with the tool that created the environment you have
**activated**, and the backends land beside that environment when it is writable — so
deleting the environment removes them with it. On a system Python they land in
`~/.chemrefine/<interpreter tag>` instead and outlive any uninstall.

=== "pip"

    ```bash
    python -m venv ~/chemrefine-env && source ~/chemrefine-env/bin/activate
    pip install chemrefine
    ```

=== "uv"

    [`uv`](https://docs.astral.sh/uv/) is a single binary that needs no pre-installed
    Python and can bootstrap one — the fastest path on a bare system:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.13        # only if the system has no Python
    uv tool install chemrefine    # isolated install, `chemrefine` on PATH
    ```

=== "conda"

    ```bash
    conda create -n chemrefine python=3.13 -y
    conda activate chemrefine
    pip install chemrefine
    ```

### From Git (unreleased changes)

The development version installs straight from GitHub — a drop-in replacement for
`chemrefine` in any command above:

```bash
pip install "chemrefine @ git+https://github.com/sterling-group/ChemRefine.git"
```

### From source (contributors)

```bash
git clone https://github.com/sterling-group/ChemRefine.git
cd ChemRefine
pip install -e ".[dev]"   # every tool the gates call — pytest, mypy, mkdocs,
                          # ruff, pre-commit, build, pip-audit
pre-commit install        # REQUIRED — installs the hook itself, which the line
                          # above does not; CI runs these same hooks
```

Installing the contributor tools does not select every quantum provider. For
the Qiskit examples, add the relevant backend profile as described below.

## Add Qiskit

The molecular `qiskit` engine and the artifact-producing `qiskit-experiment`
engine are discoverable in a core installation; executing their computations
requires the selected optional stack. The common local choices are:

| Calculation | Install in the active environment | Managed backend |
| --- | --- | --- |
| Molecular exact/VQE/ADAPT with the XYZ/PySCF driver | `python -m pip install "chemrefine[qiskit]"` | `chemrefine backends install qiskit` |
| Local Aer expectation values or finite shots | `python -m pip install "chemrefine[qiskit-aer]"` | `chemrefine backends install qiskit-aer` |
| UCJ/LUCJ, SQD and fermionic preparation | `python -m pip install "chemrefine[qiskit-fermionic]"` | `chemrefine backends install qiskit-fermionic` |

For a checkout, replace the pip package specifier with `-e ".[qiskit]"`,
`-e ".[qiskit-aer]"` or the corresponding extra. The driver-independent Python
integral API can use `[qiskit-core]` without PySCF. Runtime, cutting, RDM fitting
and resource estimation have their own profiles; the
[provider installation guide](../engines/installing.md#qiskit-nature) lists
their complete dependencies, Python requirements and compatible CUDA recipe.
The normal Aer extra supplies CPU Aer.

Start with [Qiskit calculations](../engines/qiskit.md) or
[quantum experiments](../engines/qiskit-experiment.md), and check the
[support status](../engines/qiskit-status.md) for known bugs and unverified
hardware paths. For repeatable calculations, preserve
[integral and circuit checkpoints](../engines/qiskit-checkpoints.md).

## Preview the documentation

From a source checkout, install the docs extra and start the local preview:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs serve -a 127.0.0.1:8000
```

Open `http://localhost:8000` on that computer. Use the Python interpreter from
the environment where the docs extra was installed; changing directories does
not switch an already active virtual environment. To validate the rendered
documentation without starting a server, run `python -m mkdocs build --strict`.

For a preview running on a remote server, leave MkDocs running there and run
this command in a terminal on the computer with the browser:

```bash
ssh -N -L 8000:127.0.0.1:8000 user@SERVER_HOST
```

Replace the username and `SERVER_HOST` with the server's SSH login, then open
`http://localhost:8000` locally. If local port 8000 is occupied, use
`-L 8001:127.0.0.1:8000` and open `http://localhost:8001`. The preview reads the
checkout where MkDocs runs; update that checkout to preview new branch changes.

## Check it works

```bash
chemrefine --version
chemrefine --help
chemrefine backends list     # known backends + which have an environment
```

Then run something: [Your first run](first-run.md).

## Supported platforms

ChemRefine's execution layer is POSIX by design — it is an HPC tool, and the choices that
make it robust on clusters bind it to POSIX semantics:

- local dispatch launches every job explicitly as `bash <script>`
  (`slurm/dispatch.py`), and generated job scripts begin with a `#!/bin/bash` shebang
  that ChemRefine writes itself (`slurm/script.py`);
- detached runs (`chemrefine run`, the agent's `start_run`) rely on POSIX process
  sessions to survive the parent exiting;
- the run lock's liveness probe (`os.kill(pid, 0)`) has no safe native-Windows
  equivalent.

| Platform | Status |
|---|---|
| Linux (workstations, clusters) | Supported — the primary target. |
| Windows | Supported **via WSL2 only**. Install ChemRefine inside the WSL2 distribution; everything behaves as on Linux. Native Windows (cmd/PowerShell, or Git-Bash) is out of scope. |
| macOS | The local execution path works (`bash` is present; there is no SLURM). Not part of CI. |

The pure-Python layers (schema, validation, the GUI, the MCP server, the agent chat)
import fine anywhere Python runs — but a layer that cannot execute workflows is not a
support claim we make, so the platform statement above is the honest one.

## If something goes wrong

| Symptom | Likely cause |
|---------|--------------|
| `chemrefine: command not found` | Activate the env where you installed ChemRefine (`pip show chemrefine` to confirm). |
| `PackageNotFoundError: ChemRefine` at runtime | `pip install -e .` again — the editable install was removed. |
| `ORCA not accessible` | Set `executables: { orca: ... }` in the YAML to an absolute path, or put ORCA on `$PATH`. |

Anything that goes wrong once a run has started is in [When a run
fails](../running/when-a-run-fails.md).

## License

ChemRefine is released under
[AGPL v3](https://github.com/sterling-group/ChemRefine/blob/main/LICENSE).
