# Adding an Engine

ChemRefine's engines are **plugins**: they provide only engine-specific things. The
generic machinery is flat in `chemrefine/` — submission + budget (`chemrefine.submit`
composing `chemrefine.slurm` + `chemrefine.throttle`), structure assembly
(`chemrefine.engines._assemble`), caching, and the step lifecycle. The orchestrator only
ever sees the `CalculationEngine` Protocol and the `ENGINES` registry — it never imports a
concrete engine. Adding one is a self-contained package plus a single import line; no edit
to a god-class.

This page expands the recipe that lives in
[`engines/base.py`](../api/engines_base.md).

## How registration works

Registration is an **import side effect**. Each engine class is decorated with
`@register("<name>")`, which inserts it into the `ENGINES` dict. Importing
`chemrefine.engines` imports every bundled engine package, so by the time the
pipeline calls `get_engine(name)` the registry is fully populated.

```mermaid
flowchart LR
  A["import chemrefine.engines"] --> B["engines/__init__ imports\n_fake, mlip, orca, pyscf"]
  B --> C["each engine module runs\n@register('name')"]
  C --> D["ENGINES['name'] = EngineClass"]
  E["step.run_step"] --> F["get_engine('name')"]
  F --> D
  F --> G["fresh EngineClass() instance"]
```

So an engine becomes usable once (a) its class is decorated with `@register` and
(b) its package is imported from `engines/__init__.py`.

## Step-by-step

Everything engine-specific lives in a new `engines/<name>/` package; shared,
engine-neutral infrastructure stays flat (`chemrefine.submit`, `chemrefine.engines._batch`,
`chemrefine.engines._assemble`, the template renderer, the `_backend_server` gradient
service).

### 1. Pick a base for `engines/<name>/engine.py`

Decorate the class with `@register("<name>")` and choose the base that matches
the backend's shape. A **batch engine** (one job per structure) inherits `prepare` /
`submit` / `parse` from `BatchEngine` and supplies only *primitives*:

| Backend shape | Base class | You implement |
|---------------|-----------|---------------|
| A real binary run per structure | `BatchEngine` | the primitives `_build_input`, `_parse_one`, `_pal`, `_run_block` (+ `_gpus` if GPU-capable) and the ClassVars |
| User supplies a `step{N}.py` run per structure | `TemplateScriptEngine` (a thin `BatchEngine`) | usually only `_template_vars` (inject `$VAR`s from `step.options`) |
| ORCA optimises using *this* engine's gradients | `ExtOptOrcaEngine` | `_server_cmd` + the `backend` / `wrapper_filename` ClassVars, plus a `ComputeBackend` in `extopt_calc.py` |
| Not a per-structure batch (e.g. a training step) | implement the `CalculationEngine` Protocol directly | `prepare` / `submit` / `parse` + `input_digest` |

`submit` is **not** an engine responsibility for batch engines — `BatchEngine.submit`
delegates to `chemrefine.submit.run_batch`, which runs the batch under the budget using
the primitives above. `_parse_one` returns `ParsedResult`s and the shared
`chemrefine.engines._assemble.build_structures` mints child IDs + lineage.

### 2. Declare the metadata ClassVars

`name`, plus any base-required ClassVars (`label`, `template_suffix`, `output_suffix`,
`output_globs`), as `ClassVar[...]` annotations matching the other engines. There is no
`supports_nms` flag — NMS is a *capability* (step 7 below), detected via `isinstance`.

### 3. Validate the YAML knobs

Add a Pydantic model in `engines/<name>/options.py` with a `from_raw` classmethod
(mirror `engines/mlip/options.py` / `engines/pyscf/options.py`), and read it in the
engine's primitives. This keeps `step.options` a free dict at the orchestrator level
while giving the engine typed, `extra="forbid"` validation.

### 4. Wire registration

Import the class in `engines/<name>/__init__.py`, then list the package in
[`engines/__init__.py`](../api/engines_base.md) — the import is what registers it.

```python
# engines/<name>/__init__.py
from chemrefine.engines.<name>.engine import MyEngine

__all__ = ["MyEngine"]
```

```python
# engines/__init__.py  (add <name> to the import + __all__)
from chemrefine.engines import _fake, mlip, orca, pyscf, <name>
```

### 5. Legacy spellings go in the normalizer, not the registry

The registry stays alias-free. Old YAML engine names are rewritten to the
canonical name by `config._normalize_legacy` — the single place that knows the
legacy vocabulary.

### 6. Resources

An external binary reads its path from `ctx.executables.get("<name>")`. An
importable backend ships as a `pip install chemrefine[<name>]` extra and is
imported **lazily** (inside the function that needs it) so the package imports
cleanly when the optional dependency is absent — wrap the import so a missing
library reports the extra to install (see `mlip.calculator.optional_backend`).

### 7. Tests

Add `tests/test_engines_<name>*.py`, mirroring the existing engine tests. New code
ships at 100% line+branch coverage with a docstring on every public symbol.

## The lifecycle

Whatever base you pick, the engine satisfies the contract the step lifecycle
drives in order (plus `input_digest`, folded into the cache fingerprint). `submit`
blocks until the jobs finish, so there is no separate `wait`:

```
prepare → submit → parse
```

### Supporting NMS

Normal-mode sampling is **engine-independent**: the two-round algorithm lives in
`chemrefine.nms` and drives any engine through *two* hooks. NMS is a capability, not a
flag — to make an engine NMS-capable, just implement the
[`NmsCapableEngine`](../api/engines_base.md) Protocol's two methods (capability is then
detected with `isinstance`):

- `nms_input_info(ctx) -> NmsInputInfo` — introspect the step's input (is it a TS
  search? does it compute frequencies?), driving the default target and the freq gate.
- `read_frequencies(structure_id, step_dir, ctx) -> FrequencyData` — read a structure's
  imaginary frequencies + normal-mode tensor from its output.

Everything else — displacement, round-2 submission, resolution, retry — is generic.

See the [Engine Contract & Registry API](../api/engines_base.md) for the exact
signatures, and [Architecture & Code Flow](../concepts/architecture.md) for where
the lifecycle sits in the run.
