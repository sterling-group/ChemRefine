# Job & Script Engines (building blocks)

The reusable bases an engine is *built from* — never edited to add an engine. `JobEngine` is
the per-structure lifecycle (build input → run → parse) plus the `build_structures` assembler;
`_execution` is the engine-independent scheduler its `submit` delegates to; `ScriptEngine` is
the kind whose per-structure input is a user `step{N}.py`.

## Job engine base + assembler

::: chemrefine.engines._job

## Scheduler

::: chemrefine.engines._execution

## Script engine kind

::: chemrefine.engines._script.engine

::: chemrefine.engines._script.render

::: chemrefine.engines._script.output
