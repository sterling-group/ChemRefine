"""Translate v1.3.1 YAML into the current schema — the only module that knows the old names.

Every legacy spelling ChemRefine still accepts is rewritten here, once, before validation:
:meth:`chemrefine.config.Config._normalize_legacy_yaml` calls :func:`normalize` and nothing
else in the package knows these names exist. A config that is already current passes through
untouched, and each rewrite logs one deprecation warning naming what to change.

It translates a v1.3.1 step into the step v1.3.1 ran. A key v1.3.1 read keeps the meaning
it had there — including the default v1.3.1 read in its absence, written out where v2's
would differ — and a key or block v1.3.1 never read configured nothing, so it is dropped.
The one exception is a value v1.3.1's own docs told users to write and v1.3.1 then
ignored: its README's step-level ``model_name`` / ``task_name`` on an MLFF step are
honoured, as 2.0 honours the CHGNet ``model_name`` v1.3.1 ignored.
Every rewrite, written-out default and dropped key reaches the :data:`DeprecationSink` at
its own ``loc``. :func:`normalize` reads no file and changes nothing it is handed, so the
validator can run it twice over one mapping.

**Removal horizon: 3.0.** This layer is single-homed, and every key it handles was renamed
before 2.0, so deleting the module and its one call site is the whole removal; see
``docs/get-started/upgrading-from-v1.md``.

Kept out of :mod:`chemrefine.config` so the schema reads as the schema. Someone learning
what a step *is* should not have to read two hundred lines of translation for older ones.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from chemrefine.errors import ConfigError
from chemrefine.quantities import HARTREE_TO_KCALMOL

logger = logging.getLogger(__name__)

DeprecationSink = Callable[[tuple[str | int, ...], str], None]
"""``(loc, message) -> None`` — where a rewrite announces itself.

Every legacy spelling this module rewrites is a finding, and one only a logger hears
reaches only whoever is watching stderr. The agent, the GUI's Validate button and every MCP
client read :class:`~chemrefine.validate.ValidationReport` and nothing else, and would be
told a config using a removed-in-3.0 key is clean. ``loc`` follows pydantic's convention,
the same one :class:`~chemrefine.validate.ValidationIssue` anchors every other finding
with."""


def _log_deprecation(loc: tuple[str | int, ...], message: str) -> None:
    """The default sink: each rewrite as a deprecation warning in the log."""
    logger.warning("%s", message)


#: v1.3.1's engine names → canonical, matched in any case. v1.3.1 ran an ``MLFF`` step through
#: ORCA, with a machine-learned gradient server standing in for ORCA's own method — which is
#: ``mlip-extopt``.
_ENGINE_RENAMES = {
    "mlff": "mlip-extopt",
    "dft": "orca",
}

#: v1.3.1's step-level engine blocks: each → the engine whose ``options`` it holds now, and
#: why it configures nothing on any other step, where it is dropped rather than moved.
_LEGACY_BLOCKS: dict[str, tuple[str, str]] = {
    "mlff": (
        "mlip-extopt",
        "v1.3.1 used it only for the gradient server of an `engine: MLFF` step",
    ),
    "mlip": ("mlip-extopt", "v1.3.1 read it only on an `engine: MLIP` step"),
    "trainer": ("mlip-train", "v1.3.1 used it only for a training step"),
}

#: The keys v1.3.1 read from each block, by the engine that block configures. Any other key in
#: one configured nothing, so it is dropped rather than moved.
_V1_BLOCK_KEYS = {
    "mlip-extopt": frozenset({"model_name", "task_name", "bind", "device"}),
    "mlip-train": frozenset({"valid_fraction", "seed", "device"}),
}

#: A key of v1.3.1's that means another option now: its ``trainer.valid_fraction`` split off
#: the *test* set, which is ``test_fraction``; ``valid_fraction`` is the validation set.
_V1_TRAINER_RENAMES = {"valid_fraction": "test_fraction"}

#: Option sub-keys that are auto-managed now and dropped from a moved block.
_OBSOLETE_OPTION_KEYS = frozenset({"bind"})

#: The ``task_name`` prefixes v1.3.1 dispatched on before it looked at ``model_name``: MACE,
#: then FAIRChem, then CHGNet. Only a task matching none of them let a ``model_name`` starting
#: ``sevenn`` choose SevenNet — whose release v1.3.1 then read from ``task_name``.
_V1_TASK_PREFIXES = ("mace", "omol", "omat", "odac", "uma", "fairchem", "chgnet")

#: The MACE families whose v1.3.1 loader took ``model_name`` — the only tasks under which
#: v1.3.1's ``medium`` default ran as written.
_V1_MACE_LOADERS = frozenset({"mace_off", "mace_mp"})

#: Old sample-method name → canonical v2 name. ``integer``/``high_energy`` became
#: ``min``/``max``; ``energy_window`` folded into ``min`` + a ``window_kcalmol`` knob.
_SAMPLE_METHOD_RENAMES = {
    "integer": "min",
    "high_energy": "max",
    "energy_window": "min",
}

#: Old sample parameter key → canonical v2 key (applies across methods).
_SAMPLE_KEY_RENAMES = {
    "num_structures": "count",
    "weight": "percent_cumulative",
    "energy": "window_kcalmol",
    "window_kcal": "window_kcalmol",
    "temperature": "temperature_k",
}

#: v1.3.1's two gradient-server blocks. On its own MLFF/MLIP step v1.3.1 read only the one
#: the step's engine named (``step.get(engine)``); the other configured nothing there.
_V1_SERVER_BLOCKS = frozenset({"mlff", "mlip"})

#: What v1.3.1 read from ``normal_mode_sampling_parameters`` — and only on a step with
#: ``normal_mode_sampling: true`` — each → the ``options`` key it is now.
_V1_NMS_KEYS = {
    "calc_type": "target",
    "displacement_vector": "displacement_value",
    "num_random_displacements": "num_random_displacements",
}

#: What each v1.3.1 sampling method read from ``sample_type.parameters``. Any other key
#: there configured nothing, so it is dropped rather than translated.
_V1_SAMPLE_KEYS = {
    "energy_window": frozenset({"energy", "unit"}),
    "boltzmann": frozenset({"weight", "temperature"}),
    "integer": frozenset({"num_structures"}),
    "high_energy": frozenset({"num_structures"}),
}

#: What each v1.3.1 sampling method ran where its block named no value for the key it reads,
#: in v1.3.1's own spelling, and why — written out before translation, since a v2 ``min`` /
#: ``max`` names exactly one selector (:func:`_translate_sample_type`). ``0`` structures is
#: every structure to v1.3.1 and v2 alike; ``energy`` is read in the block's ``unit``.
_V1_SAMPLE_DEFAULTS: dict[str, tuple[str, float, str]] = {
    "energy_window": ("energy", 0.5, "v1.3.1's `energy_window` used 0.5 when `energy` named none"),
    "integer": (
        "num_structures",
        0,
        "v1.3.1's `integer` kept every structure when `num_structures` named none",
    ),
    "high_energy": (
        "num_structures",
        1,
        "v1.3.1's `high_energy` kept one structure when `num_structures` named none",
    ),
}


def _mapping(value: Any, *, what: str, of: str) -> dict[str, Any]:
    """A copy of ``value`` as the mapping ``what`` must hold; ``None`` is an empty one.

    Every legacy key that holds a mapping is read through here, so a scalar in its place is
    refused by name from both loaders rather than reaching a ``dict()`` call. A
    ``ValueError`` is the one exception both callers account for: pydantic wraps it into a
    row when this runs inside :class:`~chemrefine.config.Config`'s before-validator, and
    :func:`chemrefine.validate.validate_config_text` suppresses it on its reporting pass.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"{what} must be a mapping of {of}, got {type(value).__name__}: {value!r}")


def _spelled(loc: tuple[str | int, ...]) -> str:
    """``loc`` as the dotted key a message names: ``("steps", 0, "x")`` is ``steps[0].x``."""
    spelled = ""
    for part in loc:
        spelled += f"[{part}]" if isinstance(part, int) else (f".{part}" if spelled else part)
    return spelled


def _carry(
    target: dict[str, Any],
    key: str,
    value: Any,
    *,
    legacy: str,
    current: str,
    origin: dict[str, str] | None = None,
) -> None:
    """Carry a legacy value to its current spelling ``key`` in ``target``, refusing a clash.

    A config that sets both spellings to different values is neither v1.3.1's, which never
    read the current key, nor 2.0's, which never reads the legacy one, so either precedence
    would be a guess; refused by both names, as :class:`~chemrefine.engines._options.EngineOptions`
    refuses two spellings of one option. The same value written twice is one setting, kept.
    Values are compared as written: what a path or a command names is the loader's to read,
    and two spellings of one can name two things — ``./orca`` is not the ``orca`` on
    ``PATH``, nor ``./~/x`` the home's ``~/x``.
    ``origin`` records which spelling put each key into ``target``, so a clash between two
    legacy spellings names the two the file holds; a key without one is ``current``'s.
    Raised as a ``ValueError``, the one exception both loaders account for (:func:`_mapping`).
    """
    if key not in target:
        target[key] = value
        if origin is not None:
            origin[key] = legacy
        return
    if target[key] != value:
        earlier = origin.get(key, current) if origin is not None else current
        raise ValueError(
            f"`{legacy}` and `{earlier}` are two spellings of one setting with different "
            f"values ({value!r} and {target[key]!r}); keep one"
        )


def _folded(value: Any) -> str | None:
    """``value`` in lower case when it is text, else ``None``: how this module matches a name.

    v1 read engine and operation names in any case, and so does the config
    (``StepConfig._fold_engine``, ``StepConfig._fold_operation``), so a name is recognised
    here in any case; what this module writes back is only ever a rename.
    """
    return value.lower() if isinstance(value, str) else None


def normalize(raw: dict[str, Any], *, report: DeprecationSink | None = None) -> dict[str, Any]:
    """Rewrite v1.3.1's YAML keys to the current schema.

    The single place that knows the old vocabulary. Idempotent — new-style input
    passes through unchanged — and announces every rewrite, each at the key it
    changed. ``calculation_type`` is intentionally unsupported and raises
    :class:`~chemrefine.errors.ConfigError`.

    ``report`` redirects those announcements to a caller instead of the log, which is how
    :func:`chemrefine.validate.validate_config_text` turns them into report warnings. It is
    a redirect rather than an addition on purpose: the validator runs this twice over the
    user's mapping — once to collect, once inside ``Config.model_validate`` — and a sink that
    also logged would say everything twice. Each copy the model re-checks after that hands
    it the model's own current spellings, which pass through with nothing to announce.
    """
    sink = report or _log_deprecation
    out = dict(raw)
    if "orca_executable" in out:
        sink(
            ("orca_executable",), "`orca_executable` is deprecated; use `executables: {orca: ...}`"
        )
        execs = _mapping(out.get("executables"), what="executables", of="tool names to paths")
        _carry(
            execs,
            "orca",
            out.pop("orca_executable"),
            legacy="orca_executable",
            current="executables.orca",
        )
        out["executables"] = execs
    if "initial_xyz" in out:
        sink(("initial_xyz",), "`initial_xyz` is deprecated; use `input`")
        _carry(out, "input", out.pop("initial_xyz"), legacy="initial_xyz", current="input")
    if isinstance(out.get("steps"), list):
        out["steps"] = [_normalize_step(s, i, sink) for i, s in enumerate(out["steps"])]
    return out


def _v1_server_step(s: dict[str, Any]) -> bool:
    """Whether ``s`` is spelled for v1.3.1's gradient server: ``engine: MLFF``, or a block.

    Read before anything is rewritten, because the rewrite is what it decides: a step spelled
    this way gets the defaults v1.3.1 ran it with written out (:func:`_write_out_v1_server`),
    and a step written in v2's own terms keeps v2's.
    """
    return _folded(s.get("engine")) == "mlff" or not _V1_SERVER_BLOCKS.isdisjoint(s)


def _train_operation(s: dict[str, Any], loc: tuple[str | int, ...], sink: DeprecationSink) -> bool:
    """Rewrite v1.3.1's training operation to the trainer engine; whether it did.

    ``MLFF_TRAIN`` — and ``MLIP_TRAIN`` on another engine — was a training *operation* that
    implied the trainer engine. Decided before the engine rename, so an ``engine: MLFF``
    training step is not told to use the gradient server's engine. Any other spelling is the
    config's own rule (``StepConfig._fold_operation``), not a v1 one.
    """
    op = _folded(s.get("operation"))
    if op != "mlff_train" and (op != "mlip_train" or _folded(s.get("engine")) == "mlip-train"):
        return False
    sink(
        (*loc, "operation"),
        f"operation `{op}` implying the trainer engine is deprecated; write "
        f"`engine: mlip-train` + `operation: mlip_train`",
    )
    s["engine"] = "mlip-train"
    s["operation"] = "mlip_train"
    return True


def _rename_engine(s: dict[str, Any], loc: tuple[str | int, ...], sink: DeprecationSink) -> None:
    """Rewrite a v1.3.1 engine name to its canonical one, reported.

    A legacy name is matched in any case; a current one passes through as written, its case
    the config's own rule (``StepConfig._fold_engine``). ``mlip`` is both: beside an
    ``mlip:`` block it is v1.3.1's other spelling of a gradient-server step — v1.3.1 read the
    block with ``mlff``'s keys and defaults, but never started the server, so the template
    ran ORCA's own method — and alone it is v2's direct engine, passed through.
    """
    engine = _folded(s.get("engine"))
    if engine == "mlip" and "mlip" in s:
        sink(
            (*loc, "engine"),
            "engine `mlip` beside an `mlip:` block is deprecated; use `mlip-extopt` — v1.3.1 "
            "never started the gradient server for it, so the template ran ORCA's own method: "
            "name `! ExtOpt` in it to run the model the block names, or write `engine: orca` "
            "to keep what v1.3.1 computed",
        )
        s["engine"] = "mlip-extopt"
    elif engine is not None and engine in _ENGINE_RENAMES:
        renamed = _ENGINE_RENAMES[engine]
        # The most common legacy spelling of all — a config whose only legacy
        # feature is `engine: mlff` must be named before 3.0 removes the map.
        sink((*loc, "engine"), f"engine `{engine}` is deprecated; use `{renamed}`")
        s["engine"] = renamed


def _move_blocks(
    s: dict[str, Any],
    loc: tuple[str | int, ...],
    sink: DeprecationSink,
    *,
    v1_engine: str | None,
    origin: dict[str, str],
) -> None:
    """Move each of v1.3.1's engine blocks into ``options`` on the step v1.3.1 used it on.

    Mutates ``s``. On any other step the block configured nothing in v1.3.1, so it is
    dropped, reported — as is, on v1.3.1's own MLFF/MLIP step (``v1_engine``, read before the
    rename), the server block its engine did not name. A block moves the keys v1.3.1 read
    from it (:data:`_V1_BLOCK_KEYS`), under the name each means now
    (:data:`_V1_TRAINER_RENAMES`), and drops the rest, each reported — so a key v1.3.1 never
    read cannot meet one it did as two spellings of the same option. ``origin`` records
    each moved key's block spelling: what a step-level key yields to, and a clash names.
    """
    engine = _folded(s.get("engine"))
    for block, (owner, elsewhere) in _LEGACY_BLOCKS.items():
        if block not in s:
            continue
        held = _mapping(s.pop(block), what=_spelled((*loc, block)), of="option names to values")
        unread = (
            block in _V1_SERVER_BLOCKS and v1_engine in _V1_SERVER_BLOCKS and block != v1_engine
        )
        if engine != owner or unread:
            sink(
                (*loc, block),
                f"step-level `{block}:` block is deprecated, and {elsewhere}; it configures "
                f"nothing on this step and is dropped",
            )
            continue
        sink((*loc, block), f"step-level `{block}:` block is deprecated; use `options:`")
        options = _mapping(
            s.get("options"), what=_spelled((*loc, "options")), of="option names to values"
        )
        for k, v in held.items():
            key = str(k)
            spelled = _spelled((*loc, block, key))
            if key in _OBSOLETE_OPTION_KEYS:
                # Announced, not silently swallowed: a key that vanishes without a row
                # reads as either kept or mistyped.
                sink(
                    (*loc, block, key),
                    f"`{spelled}` is auto-managed now and was dropped from the moved block",
                )
                continue
            if key not in _V1_BLOCK_KEYS[owner]:
                sink(
                    (*loc, block, key),
                    f"`{spelled}` configured nothing in v1.3.1, which read only "
                    f"{', '.join(f'`{name}`' for name in sorted(_V1_BLOCK_KEYS[owner]))}; "
                    f"dropped",
                )
                continue
            now = _V1_TRAINER_RENAMES.get(key, key)
            if now != key:
                sink(
                    (*loc, block, key),
                    f"`{spelled}` was v1.3.1's test split; written as `{now}: {v}`",
                )
            _carry(
                options,
                now,
                v,
                legacy=spelled,
                current=_spelled((*loc, "options", now)),
                origin=origin,
            )
        s["options"] = options


def _write_out_v1_training(
    s: dict[str, Any], loc: tuple[str | int, ...], sink: DeprecationSink
) -> None:
    """Write out what v1.3.1 trained a step with, one row per key.

    v1.3.1 trained MACE alone, on the device its ``trainer.device`` named (``cuda`` unset),
    and split off a *test* set of ``trainer.valid_fraction`` (``0.1`` unset) and nothing
    else: validation was the MACE template's own ``valid_fraction``. v2 requires the
    library and device, and carves a validation set of its own by default, so each is
    written out — ``task_name: mace_off``, ``device: cuda``, ``test_fraction: 0.1`` and
    ``valid_fraction: 0`` — where the step names none.
    """
    options = _mapping(
        s.get("options"), what=_spelled((*loc, "options")), of="option names to values"
    )
    for key, value, why in (
        ("task_name", "mace_off", "v1.3.1 trained MACE alone"),
        ("device", "cuda", "v1.3.1 trained on `cuda` when `trainer.device` named none"),
        (
            "test_fraction",
            0.1,
            "v1.3.1 split off a 0.1 test set when `trainer.valid_fraction` named none",
        ),
        (
            "valid_fraction",
            0,
            "v1.3.1 made no validation split of its own — the template's MACE `valid_fraction` did",
        ),
    ):
        if key not in options:
            options[key] = value
            sink((*loc, "options", key), f"{why}; written out as `{key}: {value}`")
    s["options"] = options


def _names_a_file(value: object) -> bool:
    """Whether a v1.3.1 ``model_name`` names a checkpoint file rather than a release.

    v1.3.1 had no rule of its own: MACE's loader decided, loading anything that is not a
    release name or an ``https:`` URL as a file. Over everything v1.3.1 could load, a ``/``
    draws that line — its server ran in a fresh scratch directory, so every checkpoint it
    could open was absolute or climbed out of that directory, and no release of any library
    names a path — while a bare name is a release, or a typo the release check names when
    the run starts. Decided from the text alone: the file may not exist when the config
    loads, since an upstream step can be what trains it.
    """
    return isinstance(value, str) and "/" in value and not value.startswith("https:")


def _move_step_keys(
    s: dict[str, Any], loc: tuple[str | int, ...], sink: DeprecationSink, *, origin: dict[str, str]
) -> None:
    """Move v1.3.1's step-level ``model_name`` / ``task_name`` into a server step's ``options``.

    v1.3.1's README put them on an MLFF step itself, but v1.3.1 read only the block, so the
    step ran its defaults. Written where v1.3.1's docs said to write them, they are honoured:
    each moves into ``options`` — before the defaults are written out, so no default stands in
    for a value the step names — unless a block moved the key too (``origin``), whose value is
    the one v1.3.1 read. Beside a key the step's own ``options:`` hold, it is one setting or
    a refused clash (:func:`_carry`). Each move, and each value a block outranks, is reported.
    """
    for key in ("model_name", "task_name"):
        if key not in s:
            continue
        value = s.pop(key)
        options = _mapping(
            s.get("options"), what=_spelled((*loc, "options")), of="option names to values"
        )
        if key in origin:
            sink(
                (*loc, key),
                f"step-level `{key}` is deprecated, and `{origin[key]}: {options[key]}` "
                f"outranks `{value}`, as v1.3.1 read only the block; dropped",
            )
            continue
        sink(
            (*loc, key),
            f"step-level `{key}` is deprecated; use `options:` — v1.3.1 read it only from the "
            f"block and ran its default, and the translated step runs `{key}: {value}`",
        )
        _carry(
            options,
            key,
            value,
            legacy=_spelled((*loc, key)),
            current=_spelled((*loc, "options", key)),
            origin=origin,
        )
        s["options"] = options


def _write_out_v1_server(
    s: dict[str, Any], loc: tuple[str | int, ...], sink: DeprecationSink, *, origin: dict[str, str]
) -> None:
    """Write out the defaults v1.3.1 ran a gradient-server step with, one row per key.

    v1.3.1 read a missing ``task_name`` as ``mace_off``, a missing ``model_name`` as
    ``medium`` and a missing ``device`` as ``cuda``. v2's own defaults differ, so a step that
    named none of them would run another library's model on another device. Each is written
    out where v1.3.1 used it:

    * ``task_name: mace_off`` when the step names none;
    * ``model_name: medium`` under the families whose loader took it
      (:data:`_V1_MACE_LOADERS`), when the step names neither it nor a ``model_path``. Under
      any other task v1.3.1 ignored the default or failed on it, so the library's own default
      runs now — and under ``mace_omol``, whose loader ignored ``model_name`` altogether, a
      value the step names is dropped. Under those two families a ``model_name`` naming a
      checkpoint (:func:`_names_a_file`) is what their loader read as a file, so it is
      carried to ``model_path`` as written (:func:`_carry`, under the spelling ``origin``
      says the file holds it by) — the config reads its leading ``~`` as a home, as the
      job's shell did (:data:`chemrefine.config.STEP_OPTION_PATHS`);
    * ``device: cuda`` when the step names none.

    First, v1.3.1's SevenNet spelling: a ``model_name`` starting ``sevenn`` beside a
    ``task_name`` none of v1.3.1's earlier dispatch prefixes matched (:data:`_V1_TASK_PREFIXES`)
    ran SevenNet with the ``task_name`` as its release, which is ``task_name: sevenn`` with the
    release as ``model_name`` now.
    """
    options = _mapping(
        s.get("options"), what=_spelled((*loc, "options")), of="option names to values"
    )
    task, name = options.get("task_name"), options.get("model_name")
    if (
        isinstance(task, str)
        and isinstance(name, str)
        and name.startswith("sevenn")
        and not task.startswith(_V1_TASK_PREFIXES)
    ):
        options["task_name"], options["model_name"] = "sevenn", task
        sink(
            (*loc, "options", "task_name"),
            f"v1.3.1 chose SevenNet by `model_name: {name}` and read its release from "
            f"`task_name`; written as `task_name: sevenn`, `model_name: {task}`",
        )
    if "task_name" not in options:
        options["task_name"] = "mace_off"
        sink(
            (*loc, "options", "task_name"),
            "v1.3.1 ran a step that names no `task_name` as `mace_off`; written out as "
            "`task_name: mace_off`",
        )
    task = options["task_name"]
    if isinstance(task, str) and task in _V1_MACE_LOADERS:
        if _names_a_file(name := options.get("model_name")):
            del options["model_name"]
            sink(
                (*loc, "options", "model_path"),
                f"`model_name: {name}` names a checkpoint file, which is `model_path` now; a "
                f"relative path is read from the config file's directory, where v1.3.1 read "
                f"it from the job's scratch directory",
            )
            _carry(
                options,
                "model_path",
                name,
                legacy=origin.get("model_name", _spelled((*loc, "options", "model_name"))),
                current=_spelled((*loc, "options", "model_path")),
            )
        elif "model_name" not in options and "model_path" not in options:
            options["model_name"] = "medium"
            sink(
                (*loc, "options", "model_name"),
                f"v1.3.1 ran `{task}` with no `model_name` as `medium`; written out as "
                f"`model_name: medium`",
            )
    elif task == "mace_omol" and "model_name" in options:
        ignored = options.pop("model_name")
        sink(
            (*loc, "options", "model_name"),
            f"v1.3.1's `mace_omol` loader ignored `model_name: {ignored}`; dropped, so the "
            f"family's own model runs, as it did",
        )
    if "device" not in options:
        options["device"] = "cuda"
        sink(
            (*loc, "options", "device"),
            "v1.3.1 ran a step that names no `device` on `cuda`; written out as "
            "`device: cuda` — on a host with no GPU, write `device: cpu`",
        )
    s["options"] = options


def _normalize_nms_keys(
    s: dict[str, Any], loc: tuple[str | int, ...], sink: DeprecationSink
) -> None:
    """Rewrite legacy ``normal_mode_sampling{,_parameters}`` into ``nms`` + ``options``.

    The switch is ``nms``'s legacy spelling, carried by :func:`_carry`. v1.3.1 read the
    parameters only on a step that sampled, and only the keys of :data:`_V1_NMS_KEYS` —
    ``calc_type`` as ``target`` (``rm_imag`` is ``ts``), ``displacement_vector`` as
    ``displacement_value`` — so every other key, and every key on a step that does not
    sample, is dropped, reported. Where a sampling step names no ``target``, v1.3.1's
    ``rm_imag`` default is written out as ``target: ts``, reported. Mutates ``s`` in place.
    """
    if "normal_mode_sampling" not in s and "normal_mode_sampling_parameters" not in s:
        return
    sink(
        (*loc, "normal_mode_sampling"),
        "`normal_mode_sampling*` is deprecated; use `nms` + `options`",
    )
    if "normal_mode_sampling" in s:
        _carry(
            s,
            "nms",
            bool(s.pop("normal_mode_sampling")),
            legacy=_spelled((*loc, "normal_mode_sampling")),
            current=_spelled((*loc, "nms")),
        )
    block = (*loc, "normal_mode_sampling_parameters")
    params = _mapping(
        s.pop("normal_mode_sampling_parameters", None),
        what=_spelled(block),
        of="knob names to values",
    )
    read = ", ".join(f"`{name}`" for name in _V1_NMS_KEYS)
    if not s.get("nms"):
        for key in params:
            sink(
                (*block, key),
                f"`{_spelled((*block, key))}` configured nothing in v1.3.1 on a step that "
                f"does not sample; dropped",
            )
        return
    opts = _mapping(s.get("options"), what=_spelled((*loc, "options")), of="option names to values")
    for key, value in params.items():
        now = _V1_NMS_KEYS.get(str(key))
        if now is None:
            sink(
                (*block, key),
                f"`{_spelled((*block, key))}` configured nothing in v1.3.1, which read only "
                f"{read}; dropped",
            )
            continue
        if now == "target":
            value = {"rm_imag": "ts"}.get(str(value).lower(), str(value).lower())
        _carry(
            opts,
            now,
            value,
            legacy=_spelled((*block, key)),
            current=_spelled((*loc, "options", now)),
        )
    if "target" not in opts:
        opts["target"] = "ts"
        sink(
            (*loc, "options", "target"),
            "v1.3.1 ran `rm_imag` when `calc_type` named none; written out as `target: ts`",
        )
    s["options"] = opts


def _normalize_step(step: Any, index: int, sink: DeprecationSink) -> Any:
    """Rewrite one legacy step dict to the current schema (helper for :func:`normalize`).

    ``index`` is the step's position in the list, which is what makes every finding this
    raises point at the step that caused it rather than at the file.
    """
    if not isinstance(step, dict):
        return step
    s = dict(step)
    loc: tuple[str | int, ...] = ("steps", index)

    if "calculation_type" in s:
        raise ConfigError(
            "`calculation_type` is no longer supported; use `engine:` + `operation:` "
            "(see docs/get-started/upgrading-from-v1.md)"
        )

    # Whether v1.3.1 ran this step on its gradient server or as a training step, and the
    # engine it named, read before anything is rewritten: they decide which block v1.3.1
    # read and the defaults written out below.
    v1_engine = _folded(s.get("engine"))
    v1_server = _v1_server_step(s)
    v1_training = "trainer" in s
    if _train_operation(s, loc, sink):
        v1_training = True
    else:
        _rename_engine(s, loc, sink)
    origin: dict[str, str] = {}  # each moved `options` key → the legacy spelling it came from
    _move_blocks(s, loc, sink, v1_engine=v1_engine, origin=origin)
    engine = _folded(s.get("engine"))
    if v1_server and engine == "mlip-extopt":
        _move_step_keys(s, loc, sink, origin=origin)
        _write_out_v1_server(s, loc, sink, origin=origin)
    elif v1_training and engine == "mlip-train":
        _write_out_v1_training(s, loc, sink)

    _normalize_nms_keys(s, loc, sink)

    # One `sample`, from either spelling: a current block has its legacy method/key names
    # rewritten, a v1.3.1 `sample_type` is translated as v1.3.1 ran it, and the two are
    # compared in the current vocabulary, so a block beside its own translation is one.
    if isinstance(s.get("sample"), dict):
        s["sample"] = _normalize_sample_block(s["sample"], (*loc, "sample"), sink)
    if "sample_type" in s:
        sink((*loc, "sample_type"), "`sample_type` is deprecated; use `sample`")
        translated = _translate_sample_type(s.pop("sample_type"), (*loc, "sample_type"), sink)
        _carry(
            s,
            "sample",
            translated,
            legacy=_spelled((*loc, "sample_type")),
            current=_spelled((*loc, "sample")),
        )

    return s


def _translate_sample_type(block: Any, loc: tuple[str | int, ...], sink: DeprecationSink) -> Any:
    """A v1.3.1 ``sample_type`` block as the ``sample`` v1.3.1 ran.

    v1.3.1 read the ``method`` and, from ``parameters``, only the keys that method takes
    (:data:`_V1_SAMPLE_KEYS`): every other key configured nothing and is dropped, reported —
    ``parameters.method`` under any method, as the block's own ``method`` is the one that
    ran. What a method read in its key's absence is written out in v1.3.1's own spelling
    (:data:`_V1_SAMPLE_DEFAULTS`) and translated with the rest as a ``sample`` block
    (:func:`_normalize_sample_block`), which converts it as it would the file's own and
    reports it as written out. A method v1.3.1 did not know passes through, for the model to
    refuse as v1.3.1 did.
    """
    if not isinstance(block, dict):
        return block
    at = (*loc, "parameters")
    params = _mapping(block.get("parameters"), what=_spelled(at), of="parameter names to values")
    for key in block:
        if key not in ("method", "parameters"):
            sink(
                (*loc, key),
                f"`{_spelled((*loc, key))}` configured nothing in v1.3.1, which read "
                f"`method` and `parameters`; dropped",
            )
    if "method" in params:
        del params["method"]
        sink(
            (*at, "method"),
            f"`{_spelled((*at, 'method'))}` configured nothing in v1.3.1, which ran "
            f"`{_spelled((*loc, 'method'))}`; dropped",
        )
    method = block.get("method")
    written: dict[str, str] = {}
    if isinstance(method, str) and method in _V1_SAMPLE_KEYS:
        read = _V1_SAMPLE_KEYS[method]
        for key in [k for k in params if k not in read]:
            sink(
                (*at, key),
                f"`{_spelled((*at, key))}` configured nothing in v1.3.1's `{method}`, which "
                f"read {', '.join(f'`{name}`' for name in sorted(read))}; dropped",
            )
            del params[key]
        if method in _V1_SAMPLE_DEFAULTS:
            default, value, why = _V1_SAMPLE_DEFAULTS[method]
            if default not in params:
                params[default] = value
                written[default] = why
    return _normalize_sample_block(
        {**params, "method": method}, loc, sink, keys_at=at, written_out=written
    )


def _legacy_number(value: Any) -> float | None:
    """The number a legacy scalar carries, however YAML spelled it — else ``None``.

    Quoting is part of the legacy surface: ``energy: "0.005"`` is a string to YAML but
    a number to v1, whose reader coerced it — so an ``isinstance(int, float)`` gate let
    the quoted spelling skip the hartree conversion below and arrive ~627x smaller.
    Booleans are not numbers here, and a string that does not parse is left for the
    model's own coercion error.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _normalize_sample_block(
    sample: Any,
    loc: tuple[str | int, ...],
    sink: DeprecationSink,
    *,
    keys_at: tuple[str | int, ...] | None = None,
    written_out: Mapping[str, str] | None = None,
) -> Any:
    """Rewrite a flat ``sample`` dict's legacy method/key names to the v2 vocabulary.

    Idempotent: a current-vocabulary block (``min`` / ``max`` / ``boltzmann`` with
    ``count`` / ``window_kcalmol`` / ``percent_cumulative``) passes through unchanged.
    Announces one deprecation per legacy method or key actually rewritten — the method at
    ``loc``, the keys at ``keys_at`` (``loc`` unless they sit deeper in the file). A key
    the caller wrote out rather than found (``written_out``, each with why) is announced as
    written out, in the spelling it becomes.

    The legacy ``energy`` key carries a **unit conversion**, not just a rename. v1's
    ``energy_window`` read ``unit`` with a default of ``hartree`` and converted only an
    explicit ``kcal/mol`` — any other spelling, including none at all, meant the number
    was compared against hartree energies as-is. ``window_kcalmol`` is kcal/mol by
    definition, so carrying the number across unchanged silently shrank the window
    ~627.5x for every v1 config that relied on the default — and the rename row it got
    ("use `window_kcalmol`") read as an endorsement of the value. The conversion mirrors
    v1's own rule (an explicit ``kcal/mol`` passes through, everything else is hartree),
    and its announcement names both numbers so the translation is checkable at a glance.

    ``unit`` is read by that conversion and nowhere else: beside an ``energy`` it is
    consumed, reported, and beside none it is left in place for the model, which refuses a
    key it does not know. Two keys landing on one v2 key are one :func:`_carry` clash, in
    either order.
    """
    if not isinstance(sample, dict):
        return sample
    at = keys_at or loc
    written = written_out or {}
    out: dict[str, Any] = {}
    method = sample.get("method")
    if isinstance(method, str) and method in _SAMPLE_METHOD_RENAMES:
        new_method = _SAMPLE_METHOD_RENAMES[method]
        sink((*loc, "method"), f"sample method `{method}` is deprecated; use `{new_method}`")
        out["method"] = new_method
    elif method is not None:
        out["method"] = method
    unit = str(sample.get("unit", "hartree")).strip().lower()
    origin: dict[str, str] = {}
    for raw_key, v in sample.items():
        key = str(raw_key)
        if key == "method":
            continue
        if key == "unit" and "energy" in sample:
            sink(
                (*at, key),
                "sample key `unit` is read by the `energy` conversion and nowhere else; dropped",
            )
            continue
        new_key = _SAMPLE_KEY_RENAMES.get(key, key)
        number = _legacy_number(v) if key == "energy" and unit != "kcal/mol" else None
        value = v if number is None else number * HARTREE_TO_KCALMOL
        if key in written:
            sink((*at, key), f"{written[key]}; written out as `{new_key}: {value:.6g}`")
        elif number is not None:
            sink(
                (*at, key),
                f"sample key `energy` is deprecated; use `window_kcalmol` — and the unit "
                f"moves with it: v1 read {v!r} as hartree ({unit!r} converted nothing "
                f"there), so this becomes window_kcalmol: {value:.6g}",
            )
        elif new_key != key:
            sink((*at, key), f"sample key `{key}` is deprecated; use `{new_key}`")
        _carry(
            out,
            new_key,
            value,
            legacy=_spelled((*at, key)),
            current=_spelled((*at, new_key)),
            origin=origin,
        )
    return out
