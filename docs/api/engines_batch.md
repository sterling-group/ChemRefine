# Batch Base & Assembly

The `BatchEngine` template-method base gives every per-structure engine the same shape:
it owns `prepare` / `submit` / `parse` and calls the engine's primitives (`_build_input`,
`_parse_one`, `_run_block`, `_pal`, `_gpus`). Its `parse` feeds the shared assembler,
`build_structures`, which turns each engine's `ParsedResult`s into lineage-correct
[`Structure`](state.md)s — minting child IDs and threading parents through a fan-out, in
one place.

## Batch engine base

::: chemrefine.engines._batch

## Structure assembly

::: chemrefine.engines._assemble
