# AI agents

Two ways to let a language model author and triage ChemRefine workflows. Both drive the
same tool surface, and both produce the same reviewable artifact — a YAML config plus
templates that run afterwards without any model in the loop.

!!! warning "Check your site's model policy first"
    You choose and provide the model; ChemRefine ships, downloads and endorses none.
    `chemrefine agent --check` verifies a setup but will not pull weights or start
    services on your behalf.

    Before downloading model weights or calling a hosted endpoint from institutional
    hardware, check your institution's and cluster's acceptable-use, data-governance and
    export-control policies. Some sites restrict specific model origins; many restrict
    sending research data to external endpoints at all. The examples below list models of
    several origins side by side precisely so no single choice is implied — compliance
    with your site's rules is yours to judge, not something ChemRefine can.

## Route 1 — your own MCP client

`chemrefine mcp` serves the tools over the
[Model Context Protocol](https://modelcontextprotocol.io) on stdio, so any MCP client —
Claude Code, Claude Desktop, Cursor and friends — can drive ChemRefine from natural
language.

```bash
pip install 'chemrefine[mcp]'

# local runs
claude mcp add chemrefine -- chemrefine mcp

# SLURM cluster: the server must run where the scheduler and the output tree live,
# so launch it on the login node over SSH
claude mcp add chemrefine -- ssh login-node chemrefine mcp
```

The client's own tool-approval dialogs play the role of
[the confirmation gate](#the-confirmation-gate): nothing mutating runs without your yes.

## Route 2 — the embedded chat, no MCP client

`chemrefine agent` is a terminal chat over the same tools, for machines and people without
an MCP client. The model is yours to choose: any OpenAI-compatible endpoint (OpenAI,
OpenRouter, Groq, a local **Ollama** or **vLLM**) or a PydanticAI `provider:model` string.

```bash
pip install 'chemrefine[agent]'

# a local model via Ollama — keyless
ollama serve
ollama pull llama3.2:3b               # or qwen3:4b, mistral-nemo — site policy permitting
chemrefine agent --check --provider ollama --model llama3.2:3b   # verify first
chemrefine agent --provider ollama --model llama3.2:3b

# OpenAI (uses OPENAI_API_KEY)
chemrefine agent --model openai:gpt-5-mini

# any OpenAI-compatible endpoint
export CHEMREFINE_LLM_API_KEY=...
chemrefine agent --base-url https://openrouter.ai/api/v1 --model qwen/qwen3-coder

# work on an existing config
chemrefine agent input.yaml --provider ollama --model qwen3
```

Configuration precedence: flags → `CHEMREFINE_LLM_MODEL` / `CHEMREFINE_LLM_BASE_URL` /
`CHEMREFINE_LLM_API_KEY` → the provider preset. The [GUI's chat panel](builder.md#the-agent-chat-panel)
sits at the same tier as the flags — what you type there outranks the environment, and a
field you leave blank falls through to it. `--check` probes the endpoint's model
listing and names the fix (`ollama serve`, `ollama pull <model>`, set the key) —
verification only; it never downloads models or starts services for you. Other native
providers (Anthropic, Gemini, …) work through their own `pydantic-ai-slim[<provider>]`
extra and environment variables.

The same agent lives in the [workflow builder's chat panel](builder.md#the-agent-chat-panel),
where mutating calls appear as allow/deny cards instead of terminal prompts.

## What the tools are

Every tool is a function from `chemrefine.agent_tools` — the same layer the CLI's
`validate` / `scaffold` / `schema` subcommands and the GUI use — plus one resource,
`chemrefine://guide`, the packaged operating guide an agent should read first (working
loop, conformer-funnel and transition-state recipes, hard rules).

| Purpose | Tools |
|---|---|
| Schema & registry | `get_schema`, `list_engines` |
| Seed structures | `lookup_smiles` (PubChem, needs network), `build_structures` (SMILES/XYZ, with charge-parity sanity checks) |
| Authoring | `validate_config`, `validate_config_path`, `summarize_config`, `save_config` (validation gates the write) |
| Templates | `scaffold_templates`, `read_template`, `write_template` |
| Execution | `start_run`, `run_status` |
| Results | `get_results` (paginated `steps.csv`), `get_failures` (ledger + suggested recovery) |
| Frequency analysis | `get_frequencies` (imaginary-mode counts, thermochemistry), `analyze_mode` (mode composition: which atoms and bonds move) |
| Geometry | `get_structure` (extended XYZ: the cell as `Lattice=` when there is one, a mode's displacement columns when asked) |

## A session, end to end

> Read the `chemrefine://guide` resource. Build a two-step conformer workflow for the
> SMILES `CCO`: an MLIP screen keeping the 99% Boltzmann set, then an ORCA refinement
> window of 3 kcal/mol. Validate it, save it as `project/input.yaml`, scaffold the
> templates, and show me the ORCA keywords before anything runs.

The agent reads the guide, builds seed structures, writes and validates the YAML, and
scaffolds the starters — then shows you the config **before** anything runs. Results come
back through `run_status`, `get_results` and `get_failures`; for transition states,
`get_frequencies` / `analyze_mode` answer whether the imaginary mode is the reaction
coordinate.

To exercise the whole loop without starting real compute:

1. *"What engines are available, and which are configured via template?"*
2. *"Build a water structure from SMILES O into ./seeds"* — the gate appears; answer `y`
   and check `seeds/structure_0.xyz`.
3. *"Draft and save a one-step MLIP single-point config on those seeds as ./input.yaml"* —
   `save_config` validates before writing, and asks first.
4. *"Scaffold its templates"* — gate again; starters land in `templates/`.
5. *"Start the run"* — answer **n** once: the refusal is reported to the model as an
   answer, and it must adjust rather than retry.

The config such a session converges on looks like any hand-written one — that is the
point:

```yaml
template_dir: ./templates
output_dir: ./outputs
input: ./seeds

steps:
  - step: 1
    name: screen
    engine: mlip
    operation: sp
    options: { model_name: uma-s-1p2, task_name: omol }
    sample: { method: min, count: 1 }
```

## The confirmation gate

Every mutating tool — `save_config`, `write_template`, `scaffold_templates`,
`build_structures`, `start_run` — stops and asks first:

```
allow start_run({"config_path": ".../input.yaml", "max_cores": 16})? [y/N]
```

A declined call is reported to the model as a structured refusal it must respect — nothing
changes on disk without your yes, which also bounds what any prompt-injected instruction
could do. In an MCP client the client's own approval dialog plays this role.

Small local models make clumsier tool calls (the harness feeds validation errors back and
they retry) and reason more shallowly about chemistry; the gates, schemas and refusal
behaviour are identical to frontier models. For stronger chemistry without payment, any
free-tier OpenAI-compatible endpoint works via `--base-url`.

## How it behaves

- **Submitting never blocks.** `start_run` launches a detached `python -m chemrefine`
  child that owns the [run lock](../running/when-a-run-fails.md) and outlives the agent
  session; it logs to `output_dir/agent_runs/`. Progress is read back with `run_status` —
  pure filesystem reads (`steps.csv`, `failed_jobs.json`, the lock, the log tail),
  identical whether the run is live, finished, or died.
- **Results are paginated** so a tool reply cannot flood a model's context window.
- **Errors are structured.** Tool failures carry ChemRefine's documented
  [exit-code taxonomy](../running/when-a-run-fails.md#exit-codes), and `get_failures`
  suggests the recovery action.
- `analyze_mode` re-parses the structure's output with the engine's own parser
  (ORCA-format and Q-Chem) because the displacement tensor is deliberately not cached;
  `get_structure` re-parses the same way when you ask it for a mode; everything else
  reads persisted state only.
