# ORCA Engine

The standard DFT engine and its supporting modules. The input file has a **writer**
(`input`, builds the `.inp`) and a **reader** (`inspect`, reads run-type + PAL). Output
parsing is **one section per module** (`geometry` / `energy` / `forces` / `frequencies` /
`status`, plus `ensembles` for multi-structure results), with `output` the read-once
coordinator that assembles each `ParsedResult` in a single pass. NMS is engine-independent
([Normal-Mode Sampling](nms.md)); this engine supplies only the `nms_input_info` hook and
carries the frequency values on each parsed structure.

## Engine

::: chemrefine.engines.orca.engine

## Input generation (writer)

::: chemrefine.engines.orca.input

## Template inspection (reader)

::: chemrefine.engines.orca.inspect

## Output coordinator

::: chemrefine.engines.orca.output

## Output sections

::: chemrefine.engines.orca.geometry

::: chemrefine.engines.orca.energy

::: chemrefine.engines.orca.forces

::: chemrefine.engines.orca.frequencies

::: chemrefine.engines.orca.status

::: chemrefine.engines.orca.ensembles

## ExtOpt base + protocol

::: chemrefine.engines.orca.extopt.engine

::: chemrefine.engines.orca.extopt.protocol

::: chemrefine.engines.orca.extopt.bridge

::: chemrefine.engines.orca.extopt.run_block
