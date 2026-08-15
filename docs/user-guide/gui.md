# GUI (workflow builder)

`chemrefine gui` opens a click-through builder for the workflow YAML in your browser —
left pane: workflow settings and steps as forms; right pane: the `input.yaml` being
built, live. The layout follows the IQmol submission window: select on the left, read
the resulting input on the right.

!!! tip "Try it online"
    The builder also runs on this site, no install needed: **[open the
    playground](../playground.md)** (also in the top navigation). The online copy
    builds and copies/downloads YAML from the same live schema; running validation,
    saving files, and editing templates need the local `chemrefine gui`.

```bash
pip install 'chemrefine[gui]'
chemrefine gui                     # start from scratch
chemrefine gui input.yaml          # load an existing config into the builder
chemrefine gui --port 8901 --no-browser   # print the URL instead of opening it
```

The app binds **127.0.0.1 only**, behind a per-session token carried in the launch URL.
To use it against a cluster checkout, forward the port over SSH
(`ssh -L 8901:127.0.0.1:8901 login-node`) and run `chemrefine gui --port 8901
--no-browser` there.

## What the builder knows

The forms are rendered from `chemrefine schema`'s document — the same schema the loader
validates with — so every knob, enum and default is current by construction:

- **Steps**: add/remove/reorder (numbering stays contiguous automatically), engine
  dropdown from the live registry, operation, template override, `on_failure`,
  per-step charge/multiplicity.
- **Engine options**: a subform from the engine's own declared options model. Engines
  configured through their template (ORCA) say so instead of showing an invented form.
- **Sampling**: method dropdown (`boltzmann` / `min` / `max`) with exactly that
  method's fields.
- **NMS**: toggling `nms` reveals the NMS knobs.
- Unknown option keys in a loaded config are **kept untouched** (script templates may
  read them as placeholders); the step notes them.
- **Muted values are defaults** (or, for per-step charge/multiplicity, the inherited
  workflow value): the spinner steps from them like real values, but they stay out of
  the YAML until you change them — the file carries only your deviations. Setting a
  field back to its default removes it from the YAML again.

## The right pane is the file

YAML is emitted and parsed **server-side only** — the browser never serializes YAML, so
the form pane and the text pane cannot disagree. Tick *edit as text* to type YAML
directly, then *Apply to form* to continue clicking. *Copy* puts the current YAML on
the clipboard — handy for pasting straight into an editor on a cluster.

**Validate** runs the same structured check as `chemrefine validate`: errors and
warnings appear under the YAML, each anchored to the field that caused it. **Save…**
writes the file (with a directory browser); **Scaffold templates** then fills every
missing step template and SLURM header with a commented starter, and each
template-driven step gains an *Edit template…* editor.

## The Run panel

Once a config is saved (local GUI only — the playground stays build-and-copy), a
**Run** section appears under the YAML pane:

- **Run / Resume / Rerun errors** launch the same detached driver the CLI would —
  each behind a confirmation naming the config, because this is real compute. A tree
  already held by a live driver is refused (the run-lock, exit code 10).
- The **status table** shows per-step survivors, ledgered failures, and cache state,
  with the driver's log tail underneath; it polls every 5 seconds while a run is live
  and stops when the lock is released.
- The **results table** pages through `steps.csv` for a chosen step — energies,
  ΔE, and Boltzmann weights exactly as the pipeline reported them.

## The Agent chat panel

With the `[agent]` extra installed, an **Agent chat** section joins the right column —
the [embedded agent](agent.md) inside the GUI. Pick the provider/model in the panel
(saved in your browser; local **Ollama** works with no key — see the
[model-policy note](platforms.md#model-licensing-site-policy-ai-features)), then talk:
the agent uses the same tools as everywhere else. Mutating actions **suspend** the
agent and appear as allow/deny cards naming the exact call and its arguments; nothing
touches disk or starts compute until you click *allow*. The panel needs the local
server, so the online playground shows a note instead.

A first workflow, end to end: add steps → pick engines and options → Validate →
Save… → Scaffold templates → edit the ORCA keywords in the template editor → **Run**
(or `chemrefine run input.yaml` in your terminal — same thing).
