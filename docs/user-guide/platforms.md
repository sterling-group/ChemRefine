# Platforms, Shells & Model Policy

## Supported platforms

ChemRefine's execution layer is POSIX by design — it is an HPC tool, and the choices
that make it robust on clusters bind it to POSIX semantics:

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

## Shells: your login shell never matters

ChemRefine does not run anything through your login shell:

- **Locally**, every job script is launched as `bash <script>` — an explicit
  interpreter, not `$SHELL`.
- **On SLURM**, the submitted script's first line is `#!/bin/bash`, written by
  ChemRefine — `sbatch` honors the shebang regardless of the shell your account uses.

A cluster whose users live in `zsh`, `fish`, or anything else is fine. The one
requirement: **`bash` must exist on the compute nodes** (universal on Linux clusters).
Your SLURM header files (`*.slurm.header`) contribute `#SBATCH` directives and
environment lines; they do not need their own shebang.

## Model licensing & site policy (AI features)

!!! warning "Your models, your policies, your responsibility"
    The [MCP server](mcp.md) and the [embedded agent](agent.md) connect ChemRefine to
    a language model **you** choose and provide. ChemRefine never ships, downloads, or
    endorses any model — `chemrefine agent --check` verifies a setup but will not pull
    weights or start services on your behalf.

    Before downloading model weights or calling a hosted endpoint from institutional
    hardware, check your institution's and cluster's acceptable-use, data-governance,
    and export-control policies. Some sites restrict specific model origins (for
    example, several US facilities restrict Chinese-origin models such as DeepSeek or
    Qwen); many restrict sending research data to external endpoints at all.
    Documentation examples list models of several origins side by side precisely so no
    single choice is implied — compliance with your site's rules is your
    responsibility, not something ChemRefine can judge for you.
