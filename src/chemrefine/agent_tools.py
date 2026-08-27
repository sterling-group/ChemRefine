"""The tool layer agents drive ChemRefine through — framework-neutral, JSON-shaped.

One module holds every operation an AI agent (or any remote caller) performs against a
ChemRefine tree, so the MCP server (:mod:`chemrefine.mcp_server`) and the embedded chat
agent register *the same functions* and cannot drift apart. Nothing here imports an
agent framework; every function takes JSON-compatible arguments, returns a
JSON-serializable dict, and raises :class:`~chemrefine.errors.ChemRefineError`
subclasses — whose documented ``exit_code`` taxonomy doubles as the structured failure
signal a caller branches on.

Two design rules, both taken from what breaks agent/HPC integrations in practice:

* **Submitting never blocks.** :func:`start_run` launches a *detached* ``python -m
  chemrefine`` child that owns the run lock and outlives the caller, and returns
  immediately; progress is read back by :func:`run_status` / :func:`get_results` /
  :func:`get_failures`, which are pure filesystem reads of what the pipeline already
  persists (``steps.csv``, ``failed_jobs.json``, the run lock, the child's log).
* **Results are paginated.** A refinement tree can hold thousands of structures;
  :func:`get_results` returns a slice with a total, never the whole ensemble, so a tool
  result cannot flood a model's context window.

Everything else delegates to the library seams built for exactly this:
:mod:`chemrefine.introspect` (schema), :mod:`chemrefine.validate` (structured report),
:mod:`chemrefine.scaffold` (starter templates), :func:`chemrefine.pipeline.lock_status`.
"""

from __future__ import annotations

import csv
import dataclasses
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from chemrefine import cache, introspect, io, pipeline, scaffold
from chemrefine.cache import load_failure_records
from chemrefine.config import Config, StepConfig, load_config
from chemrefine.engines.api import FrequencyOutputParsing, ParsedResult, get_engine
from chemrefine.errors import EXIT_CODES, ConfigError, RunLockError
from chemrefine.state import Structure
from chemrefine.validate import validate_config_file, validate_config_text

_ACTIONS = ("run", "resume", "rerun", "rerun-errors", "rebuild-cache", "rebuild-nms")
"""CLI actions :func:`start_run` may launch — the recovery vocabulary, nothing else."""

_MAX_ROWS = 200
"""Hard ceiling on :func:`get_results`' page size — the module's pagination rule, enforced.

The rule is stated at the top of this module ("a tool result cannot flood a model's
context window") and until now nothing held it: ``limit`` went straight into a slice, so
``limit=10**9`` returned the whole ensemble and the guarantee was decorative. A refinement
tree holds thousands of structures, and the caller is usually a model that pays for every
row twice — once reading, once quoting.

Truncation is never silent: ``total`` is the unpaginated count and the answer echoes the
``limit`` actually applied, so a caller can always see there is more and page for it."""


# ---------------------------------------------------------------------------
# Introspection + validation (thin re-exposures of the library seams)
# ---------------------------------------------------------------------------


def get_schema() -> dict[str, Any]:
    """The machine-readable schema document — read this before writing a config."""
    return introspect.schema_document()


def list_engines() -> list[dict[str, Any]]:
    """One descriptor per registered engine (capabilities, declared options schema)."""
    return [dataclasses.asdict(d) for d in introspect.describe_engines()]


def validate_config(yaml_text: str, base_dir: str | None = None) -> dict[str, Any]:
    """Validate config YAML text; every finding at once, never raises.

    ``base_dir`` resolves relative paths (and locates templates) as if the text lived in
    that directory — pass the directory the config will be saved to.
    """
    base = Path(base_dir) if base_dir is not None else None
    return validate_config_text(yaml_text, base_dir=base).to_json()


def validate_config_path(config_path: str) -> dict[str, Any]:
    """Validate a config file on disk — the report shape, not an exception."""
    return validate_config_file(Path(config_path)).to_json()


def summarize_config(config_path: str) -> dict[str, Any]:
    """A loaded config's execution summary: settings plus one row per step."""
    config = load_config(Path(config_path))
    return {
        "output_dir": str(config.output_dir),
        "template_dir": str(config.template_dir),
        "input": str(config.input) if config.input is not None else None,
        "charge": config.charge,
        "multiplicity": config.multiplicity,
        "max_cores": config.max_cores,
        "max_gpus": config.max_gpus,
        "dispatch": config.dispatch,
        "steps": [
            {
                "step": s.step,
                "name": s.name,
                "dir": s.dir_name(),
                "engine": s.engine,
                "operation": s.operation,
                "template": s.template,
                "options": s.options,
                "sample": s.sample.model_dump() if s.sample is not None else None,
                "nms": s.nms,
                "on_failure": s.on_failure,
            }
            for s in config.steps
        ],
    }


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def _step_template_plan(config: Config, step: int | str) -> scaffold.TemplatePlan:
    """The template plan row for one step, or the :class:`ConfigError` saying why not."""
    step_cfg = config.find_step(step)
    if step_cfg is None:
        raise ConfigError(f"no step matches {step!r}")
    for plan in scaffold.plan_templates(config):
        if plan.kind == "step" and plan.step == step_cfg.step:
            return plan
    raise ConfigError(f"step {step_cfg.step} (engine {step_cfg.engine!r}) does not read a template")


def read_template(config_path: str, step: int | str) -> dict[str, Any]:
    """One step's template text — the file the engine will actually render."""
    plan = _step_template_plan(load_config(Path(config_path)), step)
    if not plan.exists:
        raise ConfigError(
            f"template {plan.path} does not exist yet (scaffold_templates writes a starter)"
        )
    return {"path": str(plan.path), "text": plan.path.read_text(encoding="utf-8")}


def write_template(config_path: str, step: int | str, text: str) -> dict[str, Any]:
    """Replace one step's template with ``text`` (creating template_dir if needed).

    An unwritable destination is a :class:`~chemrefine.errors.ConfigError` naming the
    path — the module's stated contract — not a raw :class:`OSError` the GUI's error
    handler re-raises as a 500 whose traceback names neither.
    """
    plan = _step_template_plan(load_config(Path(config_path)), step)
    try:
        plan.path.parent.mkdir(parents=True, exist_ok=True)
        plan.path.write_text(text, encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot write template {plan.path}: {e}") from e
    return {"path": str(plan.path), "bytes": len(text.encode("utf-8"))}


def scaffold_templates(config_path: str, overwrite: bool = False) -> dict[str, Any]:
    """Write starter templates into every gap the config expects; report both sides."""
    config = load_config(Path(config_path))
    written = set(scaffold.scaffold_templates(config, overwrite=overwrite))
    plans = scaffold.plan_templates(config)
    return {
        "written": sorted(str(p) for p in written),
        "kept": sorted(str(p.path) for p in plans if p.path not in written),
    }


# ---------------------------------------------------------------------------
# Runs — detached submit, filesystem-read status
# ---------------------------------------------------------------------------


def start_run(
    config_path: str,
    action: str = "run",
    target: str | None = None,
    max_cores: int | None = None,
    max_gpus: int | None = None,
) -> dict[str, Any]:
    """Launch a detached ``python -m chemrefine <action>`` and return immediately.

    The child owns the run lock, logs to ``output_dir/agent_runs/``, and survives this
    process exiting — an agent session ending must not kill a three-day refinement.
    Refuses while a live driver holds the lock (:func:`~chemrefine.pipeline.lock_status`),
    and validates ``action`` / ``target`` before anything launches, so the likeliest
    mistakes fail here with a message rather than in a log nobody is watching yet.
    """
    if action not in _ACTIONS:
        raise ConfigError(f"unknown action {action!r}; one of {list(_ACTIONS)}")
    if target is not None and action in ("run", "resume"):
        # The CLI's `run`/`resume` take no positional target, so the child would exit 2 on
        # "unexpected extra argument" — *after* this returned a pid and a log path, leaving
        # an agent polling a run that never started. The schema shows `action` and `target`
        # side by side; this is the validation the docstring promises for that pairing.
        raise ConfigError(
            f"action {action!r} drives the whole pipeline and takes no target; "
            f"aim at a step with rerun, rerun-errors, rebuild-cache or rebuild-nms"
        )
    path = Path(config_path).resolve()
    config = load_config(path)
    if target is not None and config.find_step(target) is None:
        raise ConfigError(f"no step matches target {target!r}")
    status = pipeline.lock_status(config.output_dir)
    if status.held:
        raise RunLockError(
            f"a driver already holds {config.output_dir} "
            f"(pid {status.pid} on {status.host}, started {status.started})"
        )
    log_dir = config.output_dir / "agent_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Microseconds, not seconds: the lock check above is check-then-act — the *child*
    # claims the lock, so two calls close together can both pass it — and at one-second
    # resolution they resolved to one filename, where `open("wb")` truncated the first
    # child's log out from under it while it was still writing.
    log_path = log_dir / f"{datetime.now(UTC):%Y%m%dT%H%M%S.%fZ}-{action}.log"
    argv = [sys.executable, "-m", "chemrefine", action, str(path)]
    if target is not None:
        argv.append(target)
    if max_cores is not None:
        argv += ["--maxcores", str(max_cores)]
    if max_gpus is not None:
        argv += ["--maxgpus", str(max_gpus)]
    with log_path.open("wb") as log:
        # No shell, and nothing in `argv` is free text: the interpreter is `sys.executable`,
        # `action` was matched against `_ACTIONS` above, `path` is a resolved config file,
        # `target` names a step the config was asked for, and the two budgets are ints.
        # Passed as argv rather than interpolated, so none of it can become a command.
        proc = subprocess.Popen(  # noqa: S603
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    return {"pid": proc.pid, "log": str(log_path), "output_dir": str(config.output_dir)}


def _steps_csv_rows(output_dir: Path) -> list[dict[str, str]]:
    """Every row of the cumulative ``steps.csv`` (``[]`` before the first step reports)."""
    path = output_dir / "steps.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _latest_log(output_dir: Path) -> Path | None:
    """The newest ``agent_runs`` log, or ``None`` when no agent ever started a run."""
    log_dir = output_dir / "agent_runs"
    logs = sorted(log_dir.glob("*.log")) if log_dir.is_dir() else []
    return logs[-1] if logs else None


_TAIL_CHUNK = 64 * 1024
"""How much of a log to pull back per step when tailing it — comfortably over 40 lines."""


def _tail_lines(path: Path, wanted: int) -> list[str]:
    """The last ``wanted`` lines of ``path``, read from the end rather than whole.

    The tail used to be a slice of ``read_text()``, which materialises the entire log and
    then a list of every line in it, to keep forty. That is a driver log for a run that
    can span days, and the GUI re-asks for it every five seconds for the whole of that
    run — so the cost was not the file's size but paying it again on every poll.

    Decoding happens once, after the blocks are joined, never per block: a multi-byte
    character straddling a chunk boundary would otherwise be split into replacement
    characters by the very ``errors="replace"`` that is meant to make partial output safe.
    """
    if wanted <= 0:
        return []
    end = path.stat().st_size
    block = b""
    with path.open("rb") as handle:
        # One newline more than asked for: the first line in the block is usually a
        # fragment, and stopping at exactly `wanted` could hand back a truncated line.
        while end > 0 and block.count(b"\n") <= wanted:
            step = min(_TAIL_CHUNK, end)
            end -= step
            handle.seek(end)
            block = handle.read(step) + block
    return block.decode("utf-8", errors="replace").splitlines()[-wanted:]


def run_status(config_path: str, log_tail_lines: int = 40) -> dict[str, Any]:
    """Where the tree stands: lock holder, per-step progress, the latest log's tail.

    Everything is read from what the pipeline persists — nothing here talks to the
    driver, so the answer is the same whether the run is live, finished, or died.

    ``log_tail_lines`` of ``0`` (or below) means *no tail*: an empty list, with the log's
    path still reported. It cannot mean anything larger — Python's ``-0 == 0``, so the
    bare slice read ``[-0:]`` as "the whole file", inverting a zero into the one value the
    module's pagination rule exists to forbid (a multi-MB driver log in a tool result).
    The tail itself is read backwards from the end (:func:`_tail_lines`) rather than by
    reading the log and slicing it, because the GUI re-asks for this every five seconds
    for the length of the run.
    """
    config = load_config(Path(config_path))
    status = pipeline.lock_status(config.output_dir)
    reported: dict[int, int] = {}
    for row in _steps_csv_rows(config.output_dir):
        step = int(row["Step"])
        reported[step] = reported.get(step, 0) + 1
    steps = []
    for s in config.steps:
        step_dir = config.step_dir(s)
        steps.append(
            {
                "step": s.step,
                "dir": s.dir_name(),
                "engine": s.engine,
                "reported_survivors": reported.get(s.step, 0),
                "failures": len(load_failure_records(step_dir)),
                "cached": (step_dir / "_cache").is_dir(),
            }
        )
    log_path = _latest_log(config.output_dir)
    tail: list[str] | None = None
    if log_path is not None:
        tail = _tail_lines(log_path, max(0, log_tail_lines))
    return {
        "running": status.held,
        "holder": (
            # `alive` is the three-valued liveness a caller cannot re-derive: True/False
            # for a same-host holder, null for a foreign one no status read can probe —
            # which is what tells an agent "held by a live run" from "held, unverifiable".
            {
                "host": status.host,
                "pid": status.pid,
                "started": status.started,
                "alive": status.alive,
            }
            if status.host is not None
            else None
        ),
        "steps": steps,
        "log": str(log_path) if log_path is not None else None,
        "log_tail": tail,
    }


def _steps_for(config: Config, step: int | str | None) -> tuple[StepConfig, ...]:
    """The step(s) a query names — all of them, or exactly the one that matches."""
    if step is None:
        return tuple(config.steps)
    step_cfg = config.find_step(step)
    if step_cfg is None:
        raise ConfigError(f"no step matches {step!r}")
    return (step_cfg,)


def get_results(
    config_path: str,
    step: int | str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """A paginated slice of ``steps.csv`` — survivors with energies and weights.

    Rows are exactly what the pipeline reported (already sorted by energy within each
    step, with the ``Energy type`` column naming which energy that step filtered on).
    ``total`` counts the filtered rows so a caller pages without fetching everything.

    ``limit`` and ``offset`` are clamped into range, for the reason ``run_status`` clamps
    ``log_tail_lines``: handed to a bare slice, a negative counts from the *end* instead
    of failing. ``limit=-1`` — an ordinary spelling of "no limit" — returned every row but
    the last while ``total`` still reported them all, so the payload disagreed with itself
    and the caller was quietly one row short; a negative ``offset`` re-served the tail
    under an offset a pager cannot page from. Above, ``limit`` is capped at
    :data:`_MAX_ROWS`. Both the clamped ``offset`` and the applied ``limit`` come back, so
    the answer always describes the slice actually returned and never implies it is
    everything.
    """
    config = load_config(Path(config_path))
    rows = _steps_csv_rows(config.output_dir)
    if step is not None:
        wanted = {s.step for s in _steps_for(config, step)}
        rows = [row for row in rows if int(row["Step"]) in wanted]
    start = max(0, offset)
    applied = min(max(0, limit), _MAX_ROWS)
    return {
        "total": len(rows),
        "offset": start,
        "limit": applied,
        "rows": rows[start : start + applied],
    }


# ---------------------------------------------------------------------------
# Chemistry grounding — structures in, spectroscopic judgment calls out
# ---------------------------------------------------------------------------

_PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/TXT"
)


def lookup_smiles(name: str) -> dict[str, Any]:
    """A compound name → its canonical SMILES, via PubChem's PUG REST service.

    The one tool here that needs the network, kept separate so everything else works on
    an offline compute node; a failed lookup says so and names the offline alternative
    (pass a SMILES to :func:`build_structures` directly).
    """
    from urllib.error import URLError
    from urllib.parse import quote
    from urllib.request import Request, urlopen

    from chemrefine import USER_AGENT

    url = _PUBCHEM_URL.format(quote(name))
    # A `Request` rather than a bare URL for one reason: the header. urllib's default
    # announces `Python-urllib/3.x`, and NCBI's E-utilities usage policy asks callers to
    # identify themselves — an unnamed client is the one they throttle first, and other
    # edges (Groq's, for instance) reject that token outright.
    request = Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 — scheme is fixed https
    try:
        with urlopen(request, timeout=15) as response:  # noqa: S310 — scheme is fixed https
            smiles = response.read().decode("utf-8").strip().splitlines()[0]
    except (URLError, OSError, IndexError) as e:
        raise ConfigError(
            f"PubChem lookup for {name!r} failed ({e}); offline or unknown name — "
            "pass a SMILES to build_structures instead"
        ) from e
    return {"name": name, "smiles": smiles}


def _parity_warning(symbols: tuple[str, ...], charge: int, multiplicity: int) -> str | None:
    """The impossibility message when electron count and multiplicity disagree, else None.

    ``multiplicity - 1`` unpaired electrons must share parity with the electron count —
    a neutral even-electron molecule cannot be a doublet. This is the classic silent
    setup error: every engine will happily run it and produce garbage.
    """
    from ase.data import atomic_numbers

    electrons = sum(atomic_numbers[s] for s in symbols) - charge
    if electrons % 2 != (multiplicity - 1) % 2:
        return (
            f"{electrons} electrons (charge {charge}) cannot have multiplicity "
            f"{multiplicity}: unpaired-electron parity does not match"
        )
    return None


def build_structures(
    out_dir: str,
    smiles: list[str] | None = None,
    xyz_text: str | None = None,
    charge: int = 0,
    multiplicity: int = 1,
) -> dict[str, Any]:
    """Build seed ``.xyz`` structures from SMILES or raw XYZ text, sanity-checked.

    Writes ``structure_{i}.xyz`` files into ``out_dir`` — point the config's ``input:``
    at that directory. SMILES go through ChemRefine's own embedding path
    (:func:`chemrefine.io.embed_smiles`: RDKit, deterministic seed, UFF clean-up), and a
    bad SMILES raises rather than skips — the caller named this exact molecule. Raw XYZ
    text is written then re-read through the pipeline's reader, so a malformed block
    fails here instead of at step 1. Both paths check the charge/multiplicity electron
    parity per structure (:func:`_parity_warning`) and, for SMILES, that RDKit's formal
    charge agrees with ``charge`` — warnings, not errors, because open-shell intent is
    the caller's call.

    **A call owns the whole seed set.** ``input:`` directory-seeding reads every ``.xyz``
    in ``out_dir``, so a ``structure_*.xyz`` an earlier call left behind — a longer list, a
    failed attempt — would seed the run with a molecule this call never reported. The
    stale set is cleared before writing, and a SMILES list that fails partway is cleaned
    up exactly as the XYZ branch always was: the directory afterwards holds this call's
    structures, or none.
    """
    if (smiles is None) == (xyz_text is None):
        raise ConfigError("provide exactly one of smiles or xyz_text")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("structure_*.xyz"):
        stale.unlink()
    written: list[str] = []
    build_warnings: list[str] = []
    if smiles is not None:
        from rdkit import Chem

        for i, one in enumerate(smiles):
            try:
                rows = io.embed_smiles(one)
            except ValueError as e:
                for done in written:
                    Path(done).unlink(missing_ok=True)
                raise ConfigError(str(e)) from e
            path = io.write_single_xyz(rows, out / f"structure_{i}.xyz", comment=f"SMILES: {one}")
            written.append(str(path))
            mol = Chem.MolFromSmiles(one)
            formal = Chem.GetFormalCharge(mol)
            if formal != charge:
                build_warnings.append(
                    f"structure_{i}: SMILES formal charge {formal} != requested {charge}"
                )
            parity = _parity_warning(tuple(row[0] for row in rows), charge, multiplicity)
            if parity is not None:
                build_warnings.append(f"structure_{i}: {parity}")
    else:
        path = out / "structure_0.xyz"
        path.write_text(xyz_text or "", encoding="utf-8")
        try:
            frames = io.read_xyz_frames(path)
        except (ValueError, IndexError, KeyError, OSError) as e:
            path.unlink(missing_ok=True)
            raise ConfigError(f"xyz_text is not valid XYZ: {e}") from e
        written.append(str(path))
        for i, atoms in enumerate(frames):
            parity = _parity_warning(tuple(atoms.get_chemical_symbols()), charge, multiplicity)
            if parity is not None:
                build_warnings.append(f"frame {i}: {parity}")
    return {
        "written": written,
        "warnings": build_warnings,
        "charge": charge,
        "multiplicity": multiplicity,
    }


def _required_step(config: Config, step: int | str) -> StepConfig:
    """The one step ``step`` names, or the :class:`ConfigError` saying it doesn't."""
    step_cfg = config.find_step(step)
    if step_cfg is None:
        raise ConfigError(f"no step matches {step!r}")
    return step_cfg


def _mode_payload(table: dict[int, float] | None) -> dict[str, float] | None:
    """A mode table as JSON, sorted by index — ``None`` stays ``None``, never ``{}``.

    Sorted because these are read by people and by models, in the order they are printed;
    ``None`` preserved because "no frequency table" and "a table with nothing in it" are
    different answers about whether a structure is a verified minimum.
    """
    return None if table is None else {str(mode): cm1 for mode, cm1 in sorted(table.items())}


def get_frequencies(
    config_path: str, step: int | str, structure_id: str | None = None
) -> dict[str, Any]:
    """Cached frequency and thermochemistry facts for a step's structures.

    Read from the step cache — the mode tables and the thermochemistry fields are persisted
    with every parsed structure, so this answers "is it a minimum (0 imaginary) or a TS
    (exactly 1)?" without touching output files. ``imaginary_count: null`` means the
    calculation produced no frequency table at all — distinct from a table with zero
    imaginary modes, and not evidence of a minimum.

    ``frequencies`` is every mode, of which ``imaginary_freqs`` is the subset worth
    alarming about; it is what lets a caller name a mode by its frequency rather than by an
    index alone. It is ``null`` on a tree cached before it was persisted — ``rebuild-cache``
    re-parses the outputs and fills it in.
    """
    config = load_config(Path(config_path))
    step_cfg = _required_step(config, step)
    cached = cache.load(config.step_dir(step_cfg))
    if cached is None:
        raise ConfigError(
            f"step {step_cfg.step} has no cached results yet — run it (or rebuild-cache) first"
        )
    structures = cached.results.structures
    if structure_id is not None:
        structures = tuple(s for s in structures if s.id == structure_id)
        if not structures:
            raise ConfigError(f"no structure {structure_id!r} in step {step_cfg.step}'s cache")
    return {
        "step": step_cfg.step,
        "operation": cached.operation,
        "structures": [
            {
                "id": s.id,
                "imaginary_count": None if s.imaginary_freqs is None else len(s.imaginary_freqs),
                "imaginary_freqs": _mode_payload(s.imaginary_freqs),
                "frequencies": _mode_payload(s.frequencies),
                "energy_hartree": s.energy_hartree,
                "gibbs_hartree": s.gibbs_hartree,
                "enthalpy_hartree": s.enthalpy_hartree,
                "energy_zpe_hartree": s.energy_zpe_hartree,
                "converged": s.converged,
                "terminated_normally": s.terminated_normally,
            }
            for s in structures
        ],
    }


def get_structure(
    config_path: str,
    step: int | str | None = None,
    structure_id: str | None = None,
    mode_index: int | None = None,
) -> dict[str, Any]:
    """One structure as extended-XYZ text — geometry, and optionally a mode.

    The geometry half of :func:`get_frequencies`, reading the same step cache: symbols and
    positions are persisted with every parsed structure, so this needs no output file.
    Extended XYZ because it carries three displacement columns for a mode alongside the
    geometry, in one text format a viewer can read directly — see
    :func:`chemrefine.io.extended_xyz_text`.

    That writer also emits the cell as ``Lattice="…"``, but nothing reaching here has one:
    no engine parser captures a cell (:class:`~chemrefine.engines.api.ParsedResult` has no
    such field) and :func:`chemrefine.cache.structure_record` persists only symbols and
    positions, so it is dropped on the way in and again through the cache. The capability
    is groundwork for periodic support, not something this function returns today.

    ``step`` of ``None`` means the **input seeds** — what step 1 will be given, before
    anything has run. That is the one view available on a tree that has never been
    computed, and the question it answers ("did I point this at the molecule I meant?") is
    the cheapest one to get wrong. The seeds come from
    :func:`chemrefine.pipeline.bootstrap`, so the ids here are the ids the run will use.

    ``mode_index`` animates rather than describes: it re-parses the structure's output for
    the normal-mode tensor, exactly as :func:`analyze_mode` does and for the same reason —
    the tensor is a transient the pipeline displaces along and is deliberately not cached
    (see :mod:`chemrefine.cache`). Without it, only the cache is touched.
    """
    config = load_config(Path(config_path))
    if step is None:
        return _seed_structure(config, structure_id, mode_index)
    step_cfg = _required_step(config, step)
    if mode_index is not None:
        # Geometry AND displacement from the same parsed frame, never one of each: the
        # cache is a separate source with its own ordering, and a mode drawn onto
        # positions it was not computed for is a picture of the wrong molecule moving.
        # It is also what lets this work on a tree whose cache was rebuilt away.
        if structure_id is None:
            raise ConfigError("mode_index needs a structure_id — a mode belongs to one structure")
        from ase import Atoms  # deferred like every other ase import in this module

        frame = _mode_frame(config, step_cfg, structure_id)
        displacements = _mode_displacements(frame, mode_index)
        atoms = Atoms(symbols=list(frame.symbols), positions=np.asarray(frame.positions))
        return {
            "step": step_cfg.step,
            "structure_id": structure_id,
            "mode_index": mode_index,
            "format": "extxyz",
            "text": io.extended_xyz_text(atoms, displacements=displacements),
        }
    cached = cache.load(config.step_dir(step_cfg))
    if cached is None:
        raise ConfigError(
            f"step {step_cfg.step} has no cached results yet — run it (or rebuild-cache) first"
        )
    structures = cached.results.structures
    if structure_id is not None:
        structures = tuple(s for s in structures if s.id == structure_id)
        if not structures:
            raise ConfigError(f"no structure {structure_id!r} in step {step_cfg.step}'s cache")
    if not structures:
        raise ConfigError(f"step {step_cfg.step} cached no structures")
    chosen = structures[0]
    return {
        "step": step_cfg.step,
        "structure_id": chosen.id,
        "mode_index": None,
        "format": "extxyz",
        "text": io.extended_xyz_text(chosen.atoms),
    }


def list_structures(config_path: str, step: int | str | None = None) -> dict[str, Any]:
    """What a step actually holds: every structure id, and the modes each one has.

    The enumeration :func:`get_structure` lacks. Without it a caller has to *guess* an id
    and a mode index and read the refusal to find out it guessed wrong — which is how a
    mode number that belongs to no structure became a request that could only fail.

    ``step`` of ``None`` means the **input seeds**, as everywhere else here. Seeds have no
    modes, and a step that computed no frequencies (a GOAT search, an MLIP screen) reports
    ``modes: {}`` — an empty offering is the honest answer to "which mode?" and stops the
    question being asked at all.

    Reads the step cache only, never an output file, so it works on a tree copied off a
    cluster: the ids and the mode table are both persisted. ``modes`` is ``null`` rather
    than ``{}`` where the frequency table predates being cached — unknown, not empty; see
    :func:`get_frequencies`.
    """
    config = load_config(Path(config_path))
    if step is None:
        return {
            "step": None,
            "structures": [
                {"id": s.id, "modes": {}, "imaginary": []} for s in _seeds_for_reading(config)
            ],
        }
    step_cfg = _required_step(config, step)
    cached = cache.load(config.step_dir(step_cfg))
    if cached is None:
        raise ConfigError(
            f"step {step_cfg.step} has no cached results yet — run it (or rebuild-cache) first"
        )
    return {
        "step": step_cfg.step,
        "structures": [
            {
                "id": s.id,
                "modes": _mode_payload(s.frequencies),
                "imaginary": sorted(s.imaginary_freqs or {}),
            }
            for s in cached.results.structures
        ],
    }


_MAX_STRUCTURE_BYTES = 8 * 1024 * 1024
"""Ceiling on a structure file read for the viewer — a picture, not a trajectory.

ASE will happily read a multi-gigabyte LAMMPS dump into memory, and this is a synchronous
read inside a request. Eight megabytes is far past any single geometry (a 100k-atom PDB is
under ten) and far short of a file that would take the server down with it."""


def read_structure_file(path: str) -> dict[str, Any]:
    """Any structure file ASE can read, as extended-XYZ text for a viewer.

    Deliberately nothing to do with a workflow: no config, no step, no cache, no run tree.
    It answers "what is in this file" and stops there, which is why it is separate from
    :func:`get_structure` rather than another branch inside it.

    ASE reads about ninety formats and most of them are periodic — CIF, VASP ``POSCAR``,
    Quantum Espresso, CASTEP, LAMMPS, FHI-aims. Those carry a cell, and
    :func:`chemrefine.io.extended_xyz_text` emits it as ``Lattice="…"``, so this is the
    first path in ChemRefine that produces one at all: everything else is molecular by
    construction (``pipeline.bootstrap`` accepts only ``.xyz``, a directory or a SMILES
    ``.csv``, no engine parser captures a cell, and the cache stores symbols and positions).
    **Viewing a periodic file is not computing on one** — nothing here makes such a
    structure runnable, and the pipeline would refuse it as an ``input:``.

    The last frame of a multi-frame file, matching every other reader in ChemRefine (the
    energy, the geometry, the frequency table all take the last match): a trajectory's last
    frame is its result.
    """
    target = Path(path).expanduser()
    if not target.is_file():
        raise ConfigError(f"not a file: {target}")
    size = target.stat().st_size
    if size > _MAX_STRUCTURE_BYTES:
        raise ConfigError(
            f"{target} is {size / 1e6:.0f} MB, past the {_MAX_STRUCTURE_BYTES / 1e6:.0f} MB "
            "this reads — it is sized for one geometry, not a trajectory"
        )
    return _structure_payload(target, str(target))


def read_structure_text(name: str, text: str) -> dict[str, Any]:
    """The same read, for a file whose bytes the caller has but whose path is not ours.

    A file dropped onto the page comes from the browser's machine, which over a forwarded
    port is not the machine this server runs on — so there is no path to hand
    :func:`read_structure_file`, only contents and a name. The name still matters: ASE
    picks its parser from the filename, and ``POSCAR`` and ``.cif`` are not guessable from
    a first line.

    Written to a temporary file rather than parsed from a string, so that format detection
    and every one of ASE's ninety readers work exactly as they do for a path — several of
    them seek, and a few open sibling files relative to their own.
    """
    import tempfile

    if len(text.encode("utf-8")) > _MAX_STRUCTURE_BYTES:
        raise ConfigError(
            f"{name} is past the {_MAX_STRUCTURE_BYTES / 1e6:.0f} MB this reads — "
            "it is sized for one geometry, not a trajectory"
        )
    # The basename only: this string came from a browser, and a name like `../../etc/passwd`
    # must not become part of a path. It names the temporary file so ASE can see the suffix,
    # inside a directory that is ours and is then removed.
    #
    # `.name` alone is not that guard. It strips the directories off `../../etc/POSCAR`, but
    # a *bare* `..` is a name to pathlib and comes back whole — which resolved to the parent
    # of the staging directory rather than a file inside it. The three components that are
    # not filenames are named here, rather than trusted to a helper that never claimed to
    # sanitize anything.
    safe = Path(name).name
    if safe in ("", ".", ".."):
        safe = "structure"
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / safe
        target.write_text(text, encoding="utf-8")
        return _structure_payload(target, safe)


def _structure_payload(target: Path, shown: str) -> dict[str, Any]:
    """Parse one structure file and shape the viewer's answer; ``shown`` is what to call it.

    The name is passed separately because a dropped file's real home is the user's own
    machine — echoing back the temporary path it was staged at would name somewhere that
    does not exist and will not exist a moment later.
    """
    from ase import Atoms
    from ase.io import read

    try:
        # `index=-1` selects one frame, so this is an Atoms; the signature says
        # `Atoms | list[Atoms]` because the same call with a slice returns the list.
        atoms = cast("Atoms", read(target, index=-1))
    except Exception as e:
        # ASE raises whatever the format's reader raises — a dozen exception types across
        # ninety parsers, none of them documented as a set. The one thing they have in
        # common is that the caller pointed at something ASE could not make a structure
        # from, which is a ConfigError like every other unusable input to this module.
        raise ConfigError(f"cannot read a structure from {shown}: {e}") from e
    if not len(atoms):
        raise ConfigError(f"{shown} parsed, but holds no atoms")
    return {
        "path": shown,
        "format": "extxyz",
        "text": io.extended_xyz_text(atoms),
        "atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        # Whether the viewer will draw a box, decided here rather than by the page sniffing
        # the text it was handed.
        "periodic": bool(atoms.cell.rank),
    }


def _seeds_for_reading(config: Config) -> tuple[Structure, ...]:
    """The input seeds, for a caller that is only going to look at them.

    Reuses the real seeder rather than re-reading ``input:``, so the ids shown are the ids
    the run will use — a seed inspected as ``2`` is the ``2`` that appears in ``steps.csv``
    afterwards. It also inherits the seeder's own guards, including the non-finite-coordinate
    check that exists because a seed is the one geometry no parse boundary ever sees.

    The SMILES-CSV form of ``input:`` is refused rather than served: seeding from it
    *embeds* the molecules and writes them under ``output_dir/_seed``, and a read has no
    business doing that — nor importing RDKit to answer a GET. The message says so and names
    the way round it. Every read of the seeds comes through here so that refusal cannot be
    true of one entry point and not the next.
    """
    # is_dir() first, in bootstrap's order: it takes a directory as a directory whatever it
    # is called, so testing the suffix first refused a folder named `batch.csv` here and
    # seeded it happily there.
    seeds_from_smiles = (
        config.input is not None
        and not config.input.is_dir()
        and config.input.suffix.lower() == ".csv"
    )
    if seeds_from_smiles:
        raise ConfigError(
            f"{config.input} seeds from SMILES, which has to embed the molecules before "
            "they can be drawn — run the workflow, or use build_structures to write .xyz "
            "seeds and point input: at those"
        )
    return pipeline.bootstrap(config).structures


def _seed_structure(
    config: Config, structure_id: str | None, mode_index: int | None
) -> dict[str, Any]:
    """One of the input seeds, as :func:`chemrefine.pipeline.bootstrap` will number them.

    Reuses the real seeder rather than re-reading ``input:`` here, so the ids the viewer
    shows are the ids the run will use — a seed the user inspects as ``2`` is the ``2``
    that appears in ``steps.csv`` afterwards. It also inherits the seeder's own guards,
    including the non-finite-coordinate check that exists because a seed is the one
    geometry no parse boundary ever sees.

    The SMILES-CSV form of ``input:`` is refused rather than served — see
    :func:`_seeds_for_reading`.
    """
    if mode_index is not None:
        raise ConfigError("the input seeds have no normal modes — name a step to animate one")
    structures = _seeds_for_reading(config)
    if structure_id is not None:
        structures = tuple(s for s in structures if s.id == structure_id)
        if not structures:
            raise ConfigError(f"no seed structure {structure_id!r} in {config.input}")
    if not structures:
        raise ConfigError(f"{config.input} holds no structures")
    chosen = structures[0]
    return {
        "step": None,
        "structure_id": chosen.id,
        "mode_index": None,
        "format": "extxyz",
        "text": io.extended_xyz_text(chosen.atoms),
    }


def _mode_displacements(frame: ParsedResult, mode_index: int) -> NDArray[np.float64]:
    """One normal mode's per-atom displacement, out of a frame's ``(n, 3, modes)`` tensor.

    Takes the frame rather than fetching one, so a caller that already has it — both of
    them do — does not parse the output file a second time to ask about the same mode.
    """
    modes = cast("NDArray[np.float64]", frame.normal_modes)
    n_modes = modes.shape[2]
    if not 0 <= mode_index < n_modes:
        raise ConfigError(f"mode_index {mode_index} out of range (0..{n_modes - 1})")
    return modes[:, :, mode_index]


def _locate_output(step_dir: Path, structure_id: str, recorded: Path) -> Path | None:
    """Where a structure's output actually is now, or ``None`` if it is really gone.

    The manifest records the path the file was *written* to, resolved absolute
    (:func:`chemrefine.cache.save_manifest`). Everything else about a tree is addressed
    relatively — ``Config.step_dir`` derives from ``output_dir``, which resolves against
    the config file's own directory — so a tree that is copied or moved keeps working
    everywhere except here, where the recorded path still names the machine it ran on.
    That produced "rerun the step to regenerate it" about a file sitting in the copy.

    Three probes, cheapest first: the recorded path; the same basename directly under this
    tree's ``step_dir/structure_id``; and finally a search below that, which is what finds
    an output written into an ``attemptN/`` sub-directory by a retry or by NMS.
    """
    if recorded.is_file():
        return recorded
    home = step_dir / structure_id
    direct = home / recorded.name
    if direct.is_file():
        return direct
    # Newest wins: a retried structure has the same basename in several attempt dirs, and
    # the latest attempt is the one the manifest would have been rewritten to name.
    candidates = sorted(home.rglob(recorded.name), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _mode_frame(config: Config, step_cfg: StepConfig, structure_id: str) -> ParsedResult:
    """The parsed frame carrying a normal-mode tensor, or the reason there is none.

    Shared by :func:`analyze_mode` and :func:`get_structure` — one describes the mode and
    the other draws it, and they must agree about which frame they are talking about.
    """
    step_dir = config.step_dir(step_cfg)
    manifest = cache.load_manifest(step_dir)
    if manifest is None:
        raise ConfigError(f"step {step_cfg.step} has no manifest — it has not run here")
    recorded = next((out for _inp, out, sid in manifest.files if sid == structure_id), None)
    if recorded is None:
        raise ConfigError(f"no structure {structure_id!r} in step {step_cfg.step}'s manifest")
    output = _locate_output(step_dir, structure_id, recorded)
    if output is None:
        raise ConfigError(
            f"cannot find {recorded.name} for structure {structure_id!r} under "
            f"{step_dir / structure_id} — the manifest records {recorded}, which is where "
            "it was written. If this tree was copied or moved, the outputs did not come "
            "with it; otherwise rerun the step to regenerate them."
        )
    frame = next(
        (f for f in _parse_output_frames(step_cfg.engine, output) if f.normal_modes is not None),
        None,
    )
    if frame is None:
        raise ConfigError(
            f"{output} carries no normal-mode tensor — was this a frequency calculation?"
        )
    return frame


def _parse_output_frames(engine_name: str, output: Path) -> list[ParsedResult]:
    """Re-parse one output file with the engine's own ctx-free parser.

    Dispatch is the :class:`FrequencyOutputParsing` capability, never a name roster, so
    an engine whose outputs carry the frequency/normal-mode sections joins by declaring
    the hook. A name that is not registered at all propagates ``get_engine``'s
    ``EngineNotFoundError`` (exit 3) — the truer refusal, and it lists the registry.
    """
    engine = get_engine(engine_name)
    if isinstance(engine, FrequencyOutputParsing):
        return engine.parse_frequency_output(output)
    raise ConfigError(
        f"mode analysis is not supported for engine {engine_name!r}: "
        "its outputs carry no re-parseable normal-mode section"
    )


def analyze_mode(
    config_path: str,
    step: int | str,
    structure_id: str,
    mode_index: int,
    top_atoms: int = 5,
) -> dict[str, Any]:
    """Which atoms and bonds a normal mode moves — the reaction-coordinate check.

    The semantic half of TS validation: :func:`get_frequencies` says *whether* there is
    exactly one imaginary mode; this says *what that mode does* — the dominant atomic
    displacements and the bond-length change rates along the mode — so the caller can
    judge whether 512i cm⁻¹ is the intended H-transfer coordinate or a methyl rotor.
    The displacement tensor is deliberately not cached (it is a transient the pipeline
    displaces along), so the structure's output file is re-parsed with the engine's own
    parser; a tree whose outputs were cleaned gets told to rerun or rebuild instead.

    ``frequency_cm1`` is that mode's frequency whether or not it is imaginary, and
    ``frequencies`` is the whole table — both from the same re-parse as the displacements,
    so a real mode is named rather than being handed back its own index.

    ``top_atoms`` is clamped at zero for the reason :func:`get_results` clamps its
    pagination: ``[:top_atoms]`` on a negative counts from the end, so ``top_atoms=-1``
    quietly returned every atom *but* the least-displaced one — the opposite of a shorter
    list, and on the tool whose whole job is to say which atoms move most.
    """
    config = load_config(Path(config_path))
    step_cfg = _required_step(config, step)
    frame = _mode_frame(config, step_cfg, structure_id)
    displacement = _mode_displacements(frame, mode_index)
    norms = np.linalg.norm(displacement, axis=1)
    total = float(norms.sum()) or 1.0
    leaders = np.argsort(norms)[::-1][: max(0, top_atoms)]
    imaginary = frame.imaginary_freqs or {}
    # From the whole table, not the imaginary subset: asked about a real mode, this used to
    # answer with its own index echoed back and `frequency_cm1: null`, because the only
    # frequencies anyone had kept were the ones NMS needed.
    table = frame.frequencies or {}
    return {
        "structure_id": structure_id,
        "mode_index": mode_index,
        "frequency_cm1": table.get(mode_index),
        "is_imaginary": mode_index in imaginary,
        "imaginary_freqs": _mode_payload(imaginary) or {},
        "frequencies": _mode_payload(table) or {},
        "top_atoms": [
            {
                "index": int(i),
                "symbol": frame.symbols[int(i)],
                "displacement": float(norms[int(i)]),
                "fraction": float(norms[int(i)] / total),
            }
            for i in leaders
        ],
        "bond_changes": _bond_change_rates(frame, displacement),
    }


def _bond_change_rates(frame: ParsedResult, displacement: Any) -> list[dict[str, Any]]:
    """How fast each bonded distance changes along the mode, largest movers first.

    Bonded = within 1.3x the covalent-radius sum (the conventional slack that keeps
    stretched TS bonds counted). The rate is the directional derivative of the pair
    distance along the mode — sign says forming (negative) vs breaking (positive) —
    which needs no arbitrary displacement magnitude the way a finite step would.
    """
    from ase.data import atomic_numbers, covalent_radii

    positions = np.asarray(frame.positions, dtype=float)
    rates: list[dict[str, Any]] = []
    n = len(frame.symbols)
    for i in range(n):
        for j in range(i + 1, n):
            bond = positions[i] - positions[j]
            distance = float(np.linalg.norm(bond))
            cutoff = 1.3 * (
                covalent_radii[atomic_numbers[frame.symbols[i]]]
                + covalent_radii[atomic_numbers[frame.symbols[j]]]
            )
            if distance > cutoff or distance == 0.0:
                continue
            rate = float(bond @ (displacement[i] - displacement[j]) / distance)
            rates.append(
                {
                    "atoms": f"{frame.symbols[i]}{i}-{frame.symbols[j]}{j}",
                    "distance": distance,
                    "rate": rate,
                }
            )
    rates.sort(key=lambda r: abs(r["rate"]), reverse=True)
    return rates[:5]


def get_failures(config_path: str, step: int | str | None = None) -> dict[str, Any]:
    """Every ledgered failure, with the exit-code taxonomy and the suggested recovery.

    The ledger records *all* failures for visibility; only an ``on_failure: stop``
    step's failures are pending for re-attempt, which is why the suggestion is
    ``rerun-errors`` (re-attempt just those, then continue) rather than a full rerun.
    """
    config = load_config(Path(config_path))
    failures = [
        {
            "step": s.step,
            "structure_id": record.structure_id,
            "kind": record.kind.value,
            "reason": record.reason,
        }
        for s in _steps_for(config, step)
        for record in load_failure_records(config.step_dir(s))
    ]
    return {
        "failures": failures,
        "exit_codes": EXIT_CODES,
        "suggested_action": "rerun-errors" if failures else None,
    }


def save_config(path: str, yaml_text: str) -> dict[str, Any]:
    """Validate config YAML text and, only if it is runnable, write it to ``path``.

    The agent-side closer of the authoring loop (draft → validate → **save** →
    scaffold): an MCP client with its own file tools never needs this, but the terminal
    chat and the GUI's agent panel have no other way to put the reviewed artifact on
    disk. Validation gates the write — an unrunnable config is *returned* as its report
    (``written: false``) rather than saved, so no tool call can leave a broken
    ``input.yaml`` where a later ``start_run`` would trip over it; warnings (missing
    templates and the like) do not block, exactly as ``chemrefine validate`` treats
    them. The write itself is atomic (:func:`chemrefine.cache.atomic_write`), and
    relative paths inside the text are judged against the file's own directory, the way
    :func:`~chemrefine.config.load_config` will resolve them later.
    """
    destination = Path(path).expanduser().resolve()
    report = validate_config_text(yaml_text, base_dir=destination.parent)
    payload: dict[str, Any] = {"path": str(destination), "written": False, **report.to_json()}
    if not report.ok:
        return payload
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        cache.atomic_write(destination, yaml_text.encode("utf-8"))
    except OSError as e:
        # The module contract: a failure carries the documented exit code, so the GUI's
        # handler answers 400 and an MCP client gets a typed refusal — not a 500 traceback.
        raise ConfigError(f"cannot write {destination}: {e}") from e
    payload["written"] = True
    return payload


# ---------------------------------------------------------------------------
# The shared surface — what every harness registers
# ---------------------------------------------------------------------------

TOOLS = (
    get_schema,
    list_engines,
    validate_config,
    validate_config_path,
    summarize_config,
    save_config,
    read_template,
    write_template,
    scaffold_templates,
    start_run,
    run_status,
    get_results,
    get_failures,
    lookup_smiles,
    build_structures,
    get_frequencies,
    analyze_mode,
    get_structure,
    list_structures,
    read_structure_file,
)
"""Every tool this module offers, in working-loop order — the one list both harnesses
register (:mod:`chemrefine.mcp_server` and the embedded agent), living here so neither
optional extra has to import the other's SDK to know the surface."""

MUTATING_TOOLS = frozenset(
    {"save_config", "write_template", "scaffold_templates", "start_run", "build_structures"}
)
"""Tool names that change files or launch work — what a harness gates behind approval.

MCP clients gate on their side (every client confirms tool calls); the embedded chat
agent wraps exactly these in its own confirmation prompt. Every name must match a
member of :data:`TOOLS` — the gate matches by name, so an entry naming nothing gates
nothing, and a mutating tool renamed without this set would silently go ungated. A test
holds the subset relation."""


def guide_text() -> str:
    """The packaged agent guide (``data/agent_guide.md``) — knowledge beside the tools.

    Read from the wheel so both harnesses serve the identical text: the MCP server as
    the ``chemrefine://guide`` resource, the embedded agent inside its instructions.
    """
    from importlib import resources

    return resources.files("chemrefine").joinpath("data/agent_guide.md").read_text(encoding="utf-8")
