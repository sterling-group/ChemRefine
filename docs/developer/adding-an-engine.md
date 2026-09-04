# Adding an Engine

ChemRefine's engines are a **self-contained plugin subsystem** under `chemrefine/engines/`.
chemrefine *declares* a small contract; an engine *provides* it. The whole subsystem is one
place:

- **the contract** — [`engines/api.py`](../api/engines_api.md): the `CalculationEngine` /
  `NmsCapableEngine` / `JobExecutable` Protocols, the DTOs, and the `ENGINES` registry. This is
  the only module the flat pipeline imports from `engines/`.
- **the reusable building blocks** (underscored — *compose*, never edit to add an engine):
  `_job.py` (`JobEngine`), `_execution.py` (the scheduler), `_script/` (`ScriptEngine`),
  `_backend_server/` (the ExtOpt server).
- **the plugins** — one bare-named package each, auto-discovered; the
  [engine table](../engines/index.md) lists what that currently is.

Adding an engine touches exactly **one** thing: a new bare-named `engines/<name>/` package.
Plugins are auto-discovered — every bare-named subpackage is imported when
`chemrefine.engines` loads, so the addition is fully self-contained. You never edit a building
block (or any central list) to make a new engine exist.

## The steps

Adding an engine is the same handful of moves every time. Walk them in order — each has a section
below with the detail, and the [worked example](#worked-example-a-minimal-engine) at the end
strings them into one engine you can read top to bottom:

1. **Scaffold the package** — make `engines/<name>/` with `engine.py`, `options.py`, and an
   `__init__.py`.
2. **Pick a kind and write `engine.py`** — choose the base whose shape matches your backend
   ([Pick a kind](#pick-a-kind-and-provide-its-pieces)), decorate the class with
   `@register("<name>")`, declare its [ClassVars](#declare-the-metadata-classvars), and implement
   only the primitives that kind asks for.
3. **Validate the YAML knobs** in `options.py` ([Validate the YAML knobs](#validate-the-yaml-knobs)).
4. **Register it** — [one import line in *your own* `__init__.py`](#wire-registration); the
   package itself is auto-discovered.
5. **Wire resources** — a binary path or an optional `pip` extra ([Resources](#resources)).
6. **Support NMS** only if the engine computes frequencies ([Supporting NMS](#supporting-nms)).
7. **Add tests and a contract fixture** ([Tests](#tests)).

The sections that follow are those steps in detail.

## Pick a kind and provide its pieces

Decorate the class with `@register("<name>")` and choose the **kind** that matches the backend:

| Kind | Base | You provide |
|------|------|-------------|
| Per-structure program (own input format) | [`JobEngine`](../api/engines_job.md) | `build_input`, `run_block`, `parse_one`, `pal`, `gpus` + the ClassVars (`label` / `template_suffix` / `output_suffix` / `output_globs`) |
| User Python script | [`ScriptEngine`](../api/engines_job.md) (a `JobEngine`) | `_vars_from` (inject `$VAR`s from `step.options`), and `output_fields` when the template reports more than energy / geometry / gradient |
| ORCA optimises using *this* engine's gradients | `ExtOptOrcaEngine` | the ClassVars `backend` / `wrapper_filename` / `options_cls` / `calculator_cls`, plus a `ComputeBackend` in `extopt_calc.py` |
| Not a per-structure job (e.g. a training step) | `CalculationEngine` directly | `prepare` / `submit` / `parse`, plus `artifact` + `run_dir` for `ArtifactEngine`, and the `JobExecutable` members to run through the scheduler |

A `JobEngine` provides only **primitives** — the public provision surface chemrefine requests
(`build_input` / `run_block` / `parse_one` / `pal` / `gpus`, the `JobExecutable` contract). The
base owns the lifecycle: `submit` delegates to `engines._execution.run_batch` (the scheduler),
and `parse` feeds `build_structures`, which mints child IDs + lineage. Every `JobEngine`
*templates* — ORCA's input is a `.inp`, a `ScriptEngine`'s is a `.py`; that's the `build_input`
step, not an engine-specific feature.

## Declare the metadata ClassVars

`name`, plus the base-required ClassVars (`label`, `template_suffix`, `output_suffix`,
`output_globs`), as `ClassVar[...]` annotations matching the other engines. There is no
`supports_nms` flag — NMS is a *capability* (below), detected via `isinstance`.

## Validate the YAML knobs

Add a Pydantic model in `engines/<name>/options.py` subclassing
[`EngineOptions`](../api/engines_api.md) (it carries the shared `device` / `cores` /
`backend_python` fields + frozen / `extra="forbid"` config + `from_raw`); add your own
fields and read it in the primitives. This
keeps `step.options` a free dict at the orchestrator level while giving the engine typed
validation.

## Wire registration

Registration is an **import side effect** of `@register("<name>")`, and discovery is automatic:
importing `chemrefine.engines` imports every bare-named subpackage under `engines/`. The only
wiring you write is inside your own package — import the class in `engines/<name>/__init__.py`
so the discovery import runs your decorator:

```mermaid
flowchart LR
  A["import chemrefine.engines"] --> B["auto-discovers every\nbare-named engines/&lt;pkg&gt;/"]
  B --> C["each engine module runs\n@register('name')"]
  C --> D["ENGINES['name'] = EngineClass"]
  E["step.run_step"] --> F["get_engine('name')"]
  F --> D --> G["fresh EngineClass() instance"]
```

```python
# engines/<name>/__init__.py
from chemrefine.engines.<name>.engine import MyEngine

__all__ = ["MyEngine"]
```

Nothing outside `engines/<name>/` changes — underscored packages (building blocks) and plain
modules are never treated as plugins. Legacy YAML engine spellings are rewritten to the
canonical name by `config_legacy.normalize` (the single place that knows the legacy
vocabulary) — never the registry.

## Resources

An external binary reads its path from `ctx.executables.get("<name>")`. An importable backend
ships as a `pip install chemrefine[<name>]` extra and is imported **lazily** (inside the function
that needs it) so the package imports cleanly when the optional dependency is absent — declare
the extra + pip package + import name at registration so a missing library reports the extra to
install (see `mlip.registry.MlipLibrary` — one declaration per library, shared by its
calculator and its trainer).

## The lifecycle

Whatever kind you pick, the engine satisfies the contract the step lifecycle drives in order.
That contract is three methods and mentions caching nowhere: how a run is resumed is the
orchestrator's business, not an engine's. `submit` blocks until the jobs finish, so there is no
separate `wait`:

```
prepare → submit → parse
```

## Supporting NMS

Normal-mode sampling is **engine-independent**: the two-round algorithm lives in
`chemrefine.nms`. NMS is a capability, not a flag — implement the
[`NmsCapableEngine`](../api/engines_api.md) Protocol's one hook and populate two structure
fields; capability is detected with `isinstance`:

- `nms_input_info(ctx) -> NmsInputInfo` — introspect the step's input (is it a TS search? does it
  compute frequencies?), driving the default target and the freq gate.
- set `imaginary_freqs` (mode → cm⁻¹) and `normal_modes` (the displacement tensor) on each parsed
  `Structure`, in the *same* pass that reads geometry/energy — NMS reads them off the structure, so
  there is no separate output-reading hook and the file is parsed once.

Everything else — displacement, round-2 submission, resolution, retry — is generic.

An NMS-capable engine also implements the [`FrequencyOutputParsing`](../api/engines_api.md)
hook (`parse_frequency_output`) — the ctx-free re-parse the mode-viewer tools use on a
finished step; the invariant suite holds the two capabilities together.

The second bullet is a `JobEngine`'s own `parse_one` to satisfy, which is why ORCA and Q-Chem
do. A `ScriptEngine` parses through the shared reader instead, so it reports what its
[output contract](#the-parsed-result-contract) declares: add the two fields to its
`output_fields` and have the template assign them. Until it does, `nms: true` on a script step
is ignored with a warning from `chemrefine validate` — the capability is detected, not
assumed.

## Parsing output that depends on the input

Some programs write different sections depending on what was computed — an `opt` repeats its
energy every cycle, an ensemble generator emits many geometries, a correlated method prints
more than one total energy. The shipped reference for handling that is `orca/`, and the shape
is worth mirroring because it keeps the input→output coupling in one explicit chain instead of
scattered through the parser:

- **`inspect.py` reads the input, once, into a frozen `InputInfo`** — which keywords ran, is
  it a TS search, does it compute frequencies. Nothing else re-derives facts from the
  template.
- **An explicit `operation:` wins; otherwise the inspection decides** (ORCA:
  `OrcaEngine._resolve_operation`). If the vocabulary is real, declare `OperationsDeclaring` —
  `check_step` then refuses an unknown operation at preflight, and the schema document and
  agent guide serve the vocabulary with no further wiring.
- **A read-once coordinator hands shared text to per-section extractors** (see
  `orca/output/`: coordinator / energy / geometry / frequencies / status) — the file is read
  once, and each section owns its own block grammar.
- **The disciplines**, held by the contract goldens and the recorded-output sweep: *last
  match wins* (an `opt`'s repeated prints resolve to the final one); *finite refusal at the
  boundary* (a value that cannot be read raises `OutputParseError`, an abnormal ending
  `OutputTerminationError` with the `.err` tail — never a silent `NaN`); and every case ships
  a *trimmed real* contract fixture.

For **method-dependent scalars** — "the" energy differs between a plain SCF and a correlated
run that prints both — declare a priority chain over the *output text*: rows tried highest
level of theory first, the first whose pattern matches wins, each row at its own first/last
occurrence policy. Selection is by evidence in the output, never by inspecting the input to
pick a parser: whichever method actually ran left its line.

Parsing stays **per-engine** on purpose: the formats genuinely differ, and the shared ground
is the [`ParsedResult` contract](#the-parsed-result-contract) with its goldens, plus the
extracted seams — `NmsInputInfo`, `OperationsDeclaring`, `FrequencyOutputParsing`. A new
engine mirrors the *shape* above without importing a line of another engine's parser.

## Tests

Add `tests/test_engines_<name>*.py`, mirroring the existing engine tests. New code ships at 100%
line+branch coverage with a docstring on every public symbol.

The suite is tiered: unit tests (hermetic), **recorded** end-to-end tests that replay real
outputs without any binary installed, and **live** integration tests (`pytest -m integration`,
deselected by default) that run the real binaries and re-record the fixtures with `--record`.

### The parsed-result contract

Every engine's parse lands in one canonical, engine-independent record —
`chemrefine.cache.structure_record` is the normative schema (energies in Hartree, forces in
eV/Å, positions in Å, run-status flags, string-keyed imaginary modes). The pipeline writes it
as `step{N}_{id}.result.json` beside every parsed output, whatever native format the backend
produced.

A new engine **must ship at least one contract case** under
`tests/data/engines/<name>/<case>/`: a *shortened real* native output (trim it to the blocks
your parser reads), the seed geometry (`step1_0_inp.xyz`), a `case.json` meta, and the golden
`expected.json`. Generate the golden with

```bash
pytest tests/test_engines_contract.py --update-goldens
```

and review the diff. The suite fails until every registered engine ships a case
(`test_every_registered_engine_ships_a_contract_case`).

## Worked example: a minimal engine

To see the steps as one unit, here is a complete *illustrative* engine — `demoqm`, a fictional
quantum-chemistry program with its own `.inp` format and a `demoqm` binary. It's a `JobEngine`
(one job per structure, its own input file). This code is **not shipped** — don't look for
`engines/demoqm/` — but every signature matches the real base, so it reads exactly like the
shipped `orca/`.

The package is three files:

```text
engines/demoqm/
  __init__.py     # registration import
  engine.py       # the engine + its primitives
  options.py      # the YAML knobs
```

**Step 1 + 3 — `options.py`** validates `step.options`, reusing the shared fields and
frozen config from `EngineOptions`:

```python
from __future__ import annotations

from pydantic import Field

from chemrefine.engines._options import EngineOptions


class DemoqmOptions(EngineOptions):
    """Validated ``step.options`` for the demoqm engine."""

    basis: str = Field("sto-3g", min_length=1)
    """Orbital basis set, written into the input."""
```

**Step 2 — `engine.py`** picks `JobEngine`, declares the ClassVars, and implements only the
per-structure primitives (`prepare` / `submit` / `parse` come from the base — you don't write them):

```python
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from chemrefine.engines._job import JobEngine
from chemrefine.engines.api import ParsedResult, RunBlock, register
from chemrefine.engines.demoqm.options import DemoqmOptions
from chemrefine.state import StepContext


@register("demoqm")
class DemoqmEngine(JobEngine):
    """Illustrative QM engine: runs the ``demoqm`` binary once per structure."""

    name: ClassVar[str] = "demoqm"
    label: ClassVar[str] = "DemoQM"
    template_suffix: ClassVar[str] = "inp"  # input-file extension
    output_suffix: ClassVar[str] = "out"  # output-file extension
    output_globs: ClassVar[tuple[str, ...]] = ("*.out",)

    def build_input(
        self,
        *,
        xyz_path: Path,
        template_path: Path,
        input_path: Path,
        output_path: Path,
        ctx: StepContext,
    ) -> None:
        """Render this structure's ``.inp`` from the step template + its geometry."""
        opts = DemoqmOptions.from_raw(ctx.step_cfg.options)
        body = template_path.read_text(encoding="utf-8")
        input_path.write_text(
            body.replace("$BASIS", opts.basis).replace("$XYZ", xyz_path.name),
            encoding="utf-8",
        )

    def pal(self, ctx: StepContext) -> int:
        """Cores per job, before the scheduler clamps it to ``max_cores``."""
        return 1

    def run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> RunBlock:
        """The bash that runs inside the job's work dir."""
        demoqm = ctx.executables.get("demoqm", "demoqm")
        return RunBlock(body=f'{demoqm} {inp_path.name} > "$OUTPUT_DIR/{out_path.name}"')

    def parse_one(
        self, output_path: Path, structure_id: str, ctx: StepContext
    ) -> list[ParsedResult]:
        """Parse one ``.out`` into a ``ParsedResult`` (return ≥2 to fan out to an ensemble)."""
        text = output_path.read_text(encoding="utf-8", errors="replace")
        # your regexes; cf. engines/orca/output/
        symbols, positions, energy = _read_demoqm_out(text)
        return [
            ParsedResult(
                symbols=symbols,
                positions=positions,
                energy_hartree=energy,
                forces_ev_per_a=None,
                terminated_normally=True,
            )
        ]
```

`gpus` defaults to `0` (CPU); a GPU engine overrides it.

**Step 4 — register** with your own `__init__.py` — the bare-named package is auto-discovered,
so nothing outside `engines/demoqm/` changes:

```python
# engines/demoqm/__init__.py
from chemrefine.engines.demoqm.engine import DemoqmEngine

__all__ = ["DemoqmEngine"]
```

That's a working engine: `engine: demoqm` in a step now renders `step{N}.inp` per structure, runs
`demoqm`, and parses each result back into the pipeline.

For the real, shipped versions to copy: **`qchem/`** is the closest to the engine above — a
`JobEngine` wrapping a program with its own input format, and nothing else; **`orca/`** is the
same kind carrying the ExtOpt and NMS machinery too; **`pyscf/`** is a `ScriptEngine` (it
overrides only `_vars_from`); **`mlip/`** adds a backend server.

Pick by how the calculation is reached, not by what it computes: a program with its own
input format is a `JobEngine` like the one above; a Python library is a `ScriptEngine`
(chemrefine renders a `step{N}.py` that imports it) or a `_backend_server` backend (ORCA
drives it over the ExtOpt bridge) — never a binary wrapper.

## Adding an MLIP backend

A new MLIP *library* is not a new engine — the `mlip` / `mlip-extopt` / `mlip-train` engines
drive whichever backends the registry knows. One dropped-in module under
`engines/mlip/backends/`, auto-discovered like the engine packages, declares the library once
and hangs its capabilities off it:

```python
from chemrefine.engines.mlip.registry import CalculatorSpec, MlipLibrary

MY_MLIP = MlipLibrary(extra="mlip-my_mlip", package="my-mlip-lib", import_name="my_mlip_library")


@MY_MLIP.calculator("my_task")
def _build_my_mlip(spec: CalculatorSpec):
    from my_mlip_library import MyCalculator  # imported lazily, inside the builder

    return MyCalculator(model=spec.weights or spec.model_name, device=spec.device)


@MY_MLIP.trainer("my_task")  # optional — omit if the library cannot train
class MyTrainer(TrainerBase): ...  # or ApiTrainerBase, for API-driven libraries
```

Two obligations live outside the module, both enforced by the suite:

- **The pyproject extra.** Declare `mlip-my_mlip` under `[project.optional-dependencies]` —
  `test_every_backend_extra_is_declared_in_pyproject` fails until you do, because an extra
  nothing installs provisions an empty environment that then dies on the backend import. If
  the library supports only some Python versions, put a `python_version` marker on every
  requirement of the extra and record the supported versions in `test_provision.py`'s
  `capped` table.
- **Fake-module tests for the builder.** The registry imports every backend module at
  discovery, so the heavy import must stay inside the builder (an AST scan asserts it); test
  the builder by planting fake modules in `sys.modules` — the `_install_fake_*` pattern in
  `tests/test_engines_mlip_calculator.py`.

The registry-derived gates do the rest with no edit: the missing-dependency test tries your
tasks and expects the install hint naming the package and extra, the trainer-contract
invariants parametrize over `registered_trainers()`, and `chemrefine backends` lists the new
environment.

### Adding a calculator knob

The calculator surface is deliberately **closed**: `CalculatorSpec` is frozen with no
catch-all `**extra`, so a knob no field names has nowhere to hide — the property that ended a
bug class where builders silently swallowed knobs the dispatch was passing. The price is that
a new knob is declared in four places, each visible and typed:

1. `mlip/options.py` — the field on `MlipOptions`, and its name appended to
   `CALCULATOR_KNOBS`.
2. `mlip/registry.py` — the matching `CalculatorSpec` field (a builder that ignores it
   ignores it *visibly*).
3. `mlip/calculator.py` — thread it through `build_calculator` into the spec, and keep it on
   `MlipCalculator.__init__`.
4. `mlip/extopt_calc.py` — accept it in `MlipExtOptCalculator.__init__` and forward it.

Everything downstream derives from `CALCULATOR_KNOBS` and the model: the direct engine's
`$KNOB` template placeholder, the ExtOpt server's `--knob` CLI flag (with the model's own
default), and the server reading it back off the parsed args — no further edits.

See the [Engine Contract & Registry API](../api/engines_api.md) for the exact signatures, and
[Architecture & Code Flow](../internals/architecture.md) for where the lifecycle sits in the run.
