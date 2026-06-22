# Submit

The engine-independent batch submitter: `run_batch` runs a step's per-structure inputs
under the CPU+GPU budget (composing [SLURM](slurm.md) + [Throttle](throttle.md)). Every
[`BatchEngine`](engines_batch.md) delegates its `submit` here — submission is not an
engine responsibility.

::: chemrefine.submit
