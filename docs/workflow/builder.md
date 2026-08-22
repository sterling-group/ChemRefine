# GUI (workflow builder)

`chemrefine gui` opens a click-through builder for the workflow YAML in your browser.
Two columns, each with its own tabs: on the left **Builder** (workflow settings and steps
as forms) or **Agent** (the chat); on the right **input.yaml** (the file being built,
live) or **Structure** (a 3D view). The layout follows the IQmol submission window —
select on the left, read the resulting input on the right — and the tabs mean the editor
stays visible while the agent works in the other column.

The Run panel and the validation report sit below the right column's panels, so they
stay visible whichever tab that column is showing.

!!! tip "Try it online"
    The builder also runs on this site, no install needed: **[open the
    playground](../playground.md)** (also in the top navigation). The online copy
    builds and copies/downloads YAML from the same live schema; running validation,
    saving files, and editing templates need the local `chemrefine gui`.

```bash
pip install 'chemrefine[gui]'
chemrefine gui                     # start from scratch
chemrefine gui input.yaml          # load an existing config into the builder
chemrefine gui --no-browser        # print the URL instead of opening it
```

The app binds **127.0.0.1 only**, behind a per-session token carried in the launch URL —
by default on a **stable per-user port** (hashed from your username), which is what makes
the one-time cluster setup below possible.

!!! warning "On a shared machine, use `--no-browser`"
    Opening a browser for you passes the tokened URL as a command-line argument, where
    any other user on the node can read it out of `/proc`. The token is the only thing
    guarding `/api/save` and `/api/run`, so on a multi-user host print the URL and paste
    it yourself. See [the security notes](../internals/security.md).

## From a cluster

The GUI runs where the scheduler and the output tree live — the same rule as
[the MCP server](agents.md#route-1-your-own-mcp-client) — so install `chemrefine[gui]`
on the cluster and run `chemrefine gui` inside your SSH session. A login node has no
browser worth opening: the launch checks whether anything here would open a *window*
— not merely whether a display is set, since `ssh -X` sets one on nodes whose only
"browser" is lynx — and prints the route to your own machine instead. Over SSH it
prints that route even when something did open locally, because a tunnel beats a
forwarded display. Two shapes:

- **One-time** — add the two lines the launch prints to `~/.ssh/config` on your own
  machine:

    ```
    Host login.hpc.example.edu
        LocalForward 21244 127.0.0.1:21244
    ```

    Every future connection to that host then carries the tunnel silently, and the
    routine becomes: run `chemrefine gui` on the cluster, copy the printed URL, paste it
    into your local browser. The port is hashed from your username, so it holds still
    across sessions; if something else already holds it, that launch falls back to a
    free port and prints an adjusted recipe for the session.

- **Ad-hoc** — `ssh -L 21244:127.0.0.1:21244 login.hpc.example.edu` in a second local
  terminal, for exactly one session. Also the answer when a *second* concurrent
  connection with the stanza reports `bind: Address already in use` — a warning, not a
  failure: that connection works, it just carries no tunnel of its own.

Any local machine qualifies — the requirements are an SSH client and a browser. Linux
and macOS terminals and Windows PowerShell run exactly the commands above (Windows 10+
ships OpenSSH; the config file is `C:\Users\<you>\.ssh\config`). **VS Code Remote-SSH**
needs none of it, on any OS: it auto-forwards the port and makes the printed URL
clickable. MobaXterm (*Tunneling* tab) and PuTTY (*Connection → SSH → Tunnels*) store
the same forward in their session settings. OpenSSH can even add a forward to a live
connection (press Enter, type `~C`, then `-L 21244:127.0.0.1:21244`) — though OpenSSH
≥ 9.2 keeps that command line disabled unless `EnableEscapeCommandline yes` is set.

Always paste the URL the **current** launch printed: yesterday's URL reaches today's
server on the same stable port, but its token died with its session, so the page
reports a stale token — that is the gate working, not the tunnel failing.

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

## The right column is the file

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
**Run** section appears below the right column's panels:

- **Run / Resume / Rerun errors** launch the same detached driver the CLI would —
  each behind a confirmation naming the config, because this is real compute. A tree
  already held by a live driver is refused (the run-lock, exit code 10).
- The **status table** shows per-step survivors, ledgered failures, and cache state,
  with the driver's log tail underneath; it polls every 5 seconds while a run is live
  and stops when the lock is released.
- The **results table** pages through `steps.csv` for a chosen step — energies,
  ΔE, and Boltzmann weights exactly as the pipeline reported them.

## The Structure pane

The right column's **Structure** tab draws structures — pick what to look at, and a
structure id if you want one other than the first.

**input (seeds)** is the default, and the only view that works before anything has run:
it draws what step 1 will be handed, so *"did I point `input:` at the molecule I meant?"*
is a click rather than a run. The seeds are numbered by the same code the pipeline uses,
so the id you inspect as `2` is the `2` that turns up in `steps.csv` afterwards.

Picking a **step** instead draws that step's cached structures, so it has to have run.

Give it a **mode #** and it animates that normal mode instead of drawing a still: the
displacement vectors are re-parsed from the structure's own output (the tensor is a
transient the pipeline displaces along and is deliberately not cached), so this is the
picture that goes with `analyze_mode`'s numbers — is the imaginary mode the reaction
coordinate, or a methyl rotor.

A SMILES `input:` (a `.csv`) is the one source the pane refuses: seeding from it *embeds*
the molecules and writes them into the output tree, which a read has no business doing.
Use `build_structures` to write `.xyz` seeds and point `input:` at those, or just run it.

Structures are served as extended XYZ, which carries a `Lattice="…"` line for a structure
with a cell; the viewer draws the box when one is there. ChemRefine's own pipeline is
molecular today — nothing in it sets a cell — so that path is groundwork rather than a
feature you can use yet.

The viewer is [3Dmol.js](https://3dmol.csb.pitt.edu/) (BSD-3), vendored under
`static/vendor/` and loaded the first time you open the tab, so a session that never
opens it never downloads it.

## The Agent chat panel

With the `[agent]` extra installed, the left column's **Agent** tab holds the
[embedded agent](agents.md) inside the GUI — switch to it and the `input.yaml` pane on
the right stays where it is, so you watch what the agent builds.

Pick the provider and the panel shows only the fields that provider can use: local
**Ollama** and **vLLM** need nothing but a model name, `custom` needs the endpoint's
base URL, and `openai` needs a key (see the
[model-policy note](agents.md#ai-agents)).

Then press **Check connection** before you type. It runs the same preflight as
`chemrefine agent --check` — is the endpoint there, does it serve the model you named —
and Send stays disabled until it passes, so a wrong port or a typo in the model name
costs you a click rather than a turn. Changing any setting asks for the check again.

Provider and model are remembered in your browser. **The API key is not** — it lives in
the tab for as long as the tab does, and is gone on reload. Set
`CHEMREFINE_LLM_API_KEY` before launching if you would rather not retype it.

Then talk:
the agent uses the same tools as everywhere else. Mutating actions **suspend** the
agent and appear as allow/deny cards naming the exact call and its arguments; nothing
touches disk or starts compute until you click *allow*. The panel needs the local
server, so the online playground shows a note instead.

When the agent saves the config, **the builder loads it** — the form and the YAML pane
update, and the right column switches back to `input.yaml` so you see what changed. The
agent is told about whichever file the builder currently has open, so *"add a freq step
to that"* means the one on your screen. If you have unsaved edits of your own, nothing is
overwritten: a notice offers the agent's version and you choose.

A first workflow, end to end: add steps → pick engines and options → Validate →
Save… → Scaffold templates → edit the ORCA keywords in the template editor → **Run**
(or `chemrefine run input.yaml` in your terminal — same thing).
