# Agentic Workflows

Two ways to let an AI author and triage ChemRefine workflows. Both drive the same tool
surface; both produce the same reviewable artifact — a YAML config plus templates that
run without any model in the loop.

!!! warning "Check your site's model policy first"
    You choose and provide the model; ChemRefine ships and downloads none. See
    [Platforms, Shells & Model Policy](../user-guide/platforms.md#model-licensing-site-policy-ai-features)
    before pulling weights or calling endpoints from institutional hardware.

## Route 1 — an MCP client (Claude Code shown)

```bash
pip install 'chemrefine[mcp]'
claude mcp add chemrefine -- chemrefine mcp        # or: -- ssh login-node chemrefine mcp
```

Then, in a session:

> Read the `chemrefine://guide` resource. Build a two-step conformer workflow for the
> SMILES `CCO`: an MLIP screen keeping the 99% Boltzmann set, then an ORCA refinement
> window of 3 kcal/mol. Validate it, save it as `project/input.yaml`, scaffold the
> templates, and show me the ORCA keywords before anything runs.

The client's own tool-approval dialogs gate every mutating call (`save_config`,
`write_template`, `scaffold_templates`, `start_run`); results come back through
`run_status`, `get_results`, and `get_failures` — and for transition states,
`get_frequencies` / `analyze_mode` answer whether the imaginary mode is the reaction
coordinate.

## Route 2 — the embedded agent with a local model (no API key)

```bash
pip install 'chemrefine[agent]'
ollama serve                          # local, keyless
ollama pull llama3.2:3b               # or qwen3:4b, mistral-nemo — site policy permitting

chemrefine agent --check --provider ollama --model llama3.2:3b   # verify first
chemrefine agent --provider ollama --model llama3.2:3b
```

A session that exercises everything without starting real compute:

1. *"What engines are available, and which are configured via template?"*
2. *"Build a water structure from SMILES O into ./seeds"* — the y/N gate appears;
   answer `y` and check `seeds/structure_0.xyz`.
3. *"Draft and save a one-step MLIP single-point config on those seeds as
   ./input.yaml"* — `save_config` validates before writing and asks first.
4. *"Scaffold its templates"* — gate again; starters land in `templates/`.
5. *"Start the run"* — answer **n** once: the refusal is reported to the model as an
   answer, and it must adjust rather than retry.

The same agent lives in the [GUI's chat panel](../user-guide/gui.md#the-agent-chat-panel),
where mutating calls appear as allow/deny cards instead of terminal prompts.

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

Small local models make clumsier tool calls (the harness feeds validation errors back
and they retry) and reason more shallowly about chemistry; the gates, schemas, and
refusal behavior are identical to frontier models. For stronger chemistry without
payment, any free-tier OpenAI-compatible endpoint works via `--base-url` — see the
[embedded agent page](../user-guide/agent.md).
