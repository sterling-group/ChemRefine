# Embedded Agent (terminal chat)

`chemrefine agent` is a terminal chat that authors and triages workflows with the same
tools the [MCP server](mcp.md) exposes — for machines and users without an MCP client.
The model is yours to choose: any OpenAI-compatible endpoint (OpenAI, OpenRouter, Groq,
a local **Ollama** or **vLLM**) or a PydanticAI `provider:model` string.

```bash
pip install 'chemrefine[agent]'

# a local model via Ollama
chemrefine agent --provider ollama --model qwen3

# OpenAI (uses OPENAI_API_KEY)
chemrefine agent --model openai:gpt-5-mini

# any OpenAI-compatible endpoint
export CHEMREFINE_LLM_API_KEY=...
chemrefine agent --base-url https://openrouter.ai/api/v1 --model qwen/qwen3-coder

# work on an existing config
chemrefine agent input.yaml --provider ollama --model qwen3
```

Configuration precedence: flags → `CHEMREFINE_LLM_MODEL` / `CHEMREFINE_LLM_BASE_URL` /
`CHEMREFINE_LLM_API_KEY` → the provider preset. Other native providers (Anthropic,
Gemini, …) work through their own `pydantic-ai-slim[<provider>]` extra and environment
variables.

## What a session looks like

> *"Build a two-step workflow for these SMILES: cheap MLIP screen keeping the 99%
> Boltzmann set, then an ORCA refinement window of 3 kcal/mol. Validate it and scaffold
> the templates."*

The agent reads the packaged operating guide (the same `chemrefine://guide` the MCP
server serves), builds seed structures, writes and validates the YAML, and scaffolds
starters — then shows you the config **before** anything runs.

## The confirmation gate

Every mutating tool — `write_template`, `scaffold_templates`, `start_run`,
`build_structures` — stops and asks first:

```
allow start_run({"config_path": ".../input.yaml", "max_cores": 16})? [y/N]
```

A declined call is reported to the model as a structured refusal it must respect —
nothing changes on disk without your yes, which also bounds what any prompt-injected
instruction could do. Runs started through the agent are the same detached processes as
everywhere else: they survive the chat ending, and `run_status`/`get_results` pick them
up in the next session.
