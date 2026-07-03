# Engine Contract & Registry

The public face of the engine plugin subsystem: the `CalculationEngine` /
`NmsCapableEngine` / `ProvisionableEngine` Protocols, the `JobExecutable` provision
contract, the DTOs (`ParsedResult`, `NmsInputInfo`, `BackendRequirement`), and the
`ENGINES` registry. This is the only module the flat pipeline imports from `engines/`. The
reusable bases an engine is built from live in [Job & Script Engines](engines_job.md); see
[Adding an Engine](../developer/adding-an-engine.md) for the authoring recipe.

::: chemrefine.engines.api
