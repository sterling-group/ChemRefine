# ORCA Engine

The standard DFT engine and its supporting modules: input generation, output
parsing, frequency parsing, template inspection, and the ORCA-driven ExtOpt base.
NMS itself is engine-independent ([Normal-Mode Sampling](nms.md)); this engine only
supplies the two NMS hooks (`nms_input_info` + `read_frequencies`) on the engine
class below.

## Engine

::: chemrefine.engines.orca.engine

## Input generation

::: chemrefine.engines.orca.input

## Output parsing

::: chemrefine.engines.orca.output

## Frequencies

::: chemrefine.engines.orca.frequencies

## Template inspection

::: chemrefine.engines.orca.inspect

## ExtOpt base + protocol

::: chemrefine.engines.orca.extopt.engine

::: chemrefine.engines.orca.extopt.protocol

::: chemrefine.engines.orca.extopt.bridge

::: chemrefine.engines.orca.extopt.run_block
