# Install

ChemRefine is a **light orchestrator core** plus optional **compute backends**. The core
— the pipeline, ORCA and Q-Chem driving, SLURM submission — installs anywhere in seconds
and needs no backend environment of its own. Machine-learned potentials and PySCF are
added afterwards; see [Installing engines & backends](../engines/installing.md).

## Requirements

--8<-- "README.md:requirements"

Which program each engine needs is in the [engine table](../engines/index.md), along with
its options.

The base install pulls `numpy`, `pyyaml`, `pandas`, `ase`, `rdkit`, `pydantic >= 2`, and
`typer >= 0.12`.

## Install ChemRefine

A dedicated environment is **recommended, not required**. Two reasons, both concrete: a
managed backend environment is built with the tool that created the environment you have
**activated**, and the backends land beside that environment when it is writable — so
deleting the environment removes them with it. On a system Python they land in
`~/.chemrefine` instead and outlive any uninstall.

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
pip install -e ".[dev]"
pre-commit install   # REQUIRED — CI runs these same hooks
```

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
