# Engine Contract & Registry

The structural contract every engine satisfies, the `ENGINES` registry, and the
shared `SlurmBatchEngine` base (the throttled submit loop + per-step template
resolution). See [Adding an Engine](../developer/adding-an-engine.md) for the
authoring recipe.

::: chemrefine.engines.base
