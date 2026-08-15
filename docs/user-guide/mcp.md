# MCP Server (AI agents)

`chemrefine mcp` serves ChemRefine's agent tools over the
[Model Context Protocol](https://modelcontextprotocol.io) on stdio, so any MCP client —
Claude Code, Claude Desktop, Cursor, and friends — can author, run, and triage
ChemRefine workflows from natural language. The agent writes the same reviewable
YAML-plus-templates artifact you would; ChemRefine's engine executes it without a model
in the loop.

## Setup

```bash
pip install 'chemrefine[mcp]'

# local runs
claude mcp add chemrefine -- chemrefine mcp

# SLURM cluster: the server must run where the scheduler and the output tree live,
# so launch it on the login node over SSH
claude mcp add chemrefine -- ssh login-node chemrefine mcp
```

Then, in a session: *"Build a conformer-refinement workflow for these three SMILES,
validate it, scaffold the templates, and start it with 16 cores."*

## What the server exposes

Every tool is a function from `chemrefine.agent_tools` — the same layer the CLI's
`validate`/`scaffold`/`schema` subcommands and the GUI use — plus one resource,
`chemrefine://guide`, the packaged operating guide an agent should read first
(working loop, conformer-funnel and transition-state recipes, hard rules).

| Purpose | Tools |
|---|---|
| Schema & registry | `get_schema`, `list_engines` |
| Seed structures | `lookup_smiles` (PubChem, needs network), `build_structures` (SMILES/XYZ, with charge-parity sanity checks) |
| Authoring | `validate_config`, `validate_config_path`, `summarize_config`, `save_config` (validation gates the write) |
| Templates | `scaffold_templates`, `read_template`, `write_template` |
| Execution | `start_run`, `run_status` |
| Results | `get_results` (paginated `steps.csv`), `get_failures` (ledger + suggested recovery) |
| Frequency analysis | `get_frequencies` (imaginary-mode counts, thermochemistry), `analyze_mode` (mode composition: which atoms and bonds move) |

## Design notes

- **Submitting never blocks.** `start_run` launches a detached `python -m chemrefine`
  child that owns the [run lock](../concepts/recovery.md) and outlives the agent
  session; it logs to `output_dir/agent_runs/`. Progress is read back with
  `run_status` — pure filesystem reads (`steps.csv`, `failed_jobs.json`, the lock, the
  log tail), identical whether the run is live, finished, or died.
- **Results are paginated** so a tool reply cannot flood a model's context window.
- **Errors are structured.** Tool failures carry ChemRefine's documented
  [exit-code taxonomy](cli.md#exit-codes), and `get_failures` suggests the recovery
  action (`rerun-errors` re-attempts only the pending failures).
- `analyze_mode` re-parses the structure's output with the engine's own parser
  (ORCA-format and Q-Chem) because the displacement tensor is deliberately not cached;
  everything else reads persisted state only.
