"""Translate v1.3.1 YAML into the current schema — the only module that knows the old names.

Every legacy spelling ChemRefine still accepts is rewritten here, once, before validation:
:meth:`chemrefine.config.Config._normalize_legacy_yaml` calls :func:`normalize` and nothing
else in the package knows these names exist. A config that is already current passes through
untouched, and each rewrite logs one deprecation warning naming what to change.

It translates a v1.3.1 step into the step v1.3.1 ran. A key v1.3.1 read keeps the meaning
it had there — including the default v1.3.1 read in its absence, written out where v2's
would differ — and a key or block v1.3.1 never read configured nothing, so it is dropped.
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
from collections.abc import Callable
from pathlib import Path
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

#: The keys v1.3.1 read from a gradient-server block. Any other key in one configured
#: nothing, so it is dropped rather than moved.
_V1_SERVER_KEYS = frozenset({"model_name", "task_name", "bind", "device"})

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
    a redirect rather than an addition on purpose: the validator runs this twice — once to
    collect, once inside ``Config.model_validate`` — and a sink that also logged would say
    everything twice.
    """
    sink = report or _log_deprecation
    out = dict(raw)
    if "orca_executable" in out:
        sink(
            ("orca_executable",), "`orca_executable` is deprecated; use `executables: {orca: ...}`"
        )
        execs = _mapping(out.get("executables"), what="executables", of="tool names to paths")
        execs.setdefault("orca", out.pop("orca_executable"))
        out["executables"] = execs
    if "initial_xyz" in out:
        if "input" not in out:
            sink(("initial_xyz",), "`initial_xyz` is deprecated; use `input`")
            out["input"] = out["initial_xyz"]
        out.pop("initial_xyz")
    if isinstance(out.get("steps"), list):
        out["steps"] = [_normalize_step(s, i, sink) for i, s in enumerate(out["steps"])]
    return out


def _v1_server_step(s: dict[str, Any]) -> bool:
    """Whether ``s`` is spelled for v1.3.1's gradient server: ``engine: MLFF``, or a block.

    Read before anything is rewritten, because the rewrite is what it decides: a step spelled
    this way gets the defaults v1.3.1 ran it with written out (:func:`_write_out_v1_server`),
    and a step written in v2's own terms keeps v2's.
    """
    return _folded(s.get("engine")) == "mlff" or "mlff" in s or "mlip" in s


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


def _move_blocks(s: dict[str, Any], loc: tuple[str | int, ...], sink: DeprecationSink) -> None:
    """Move each of v1.3.1's engine blocks into ``options`` on the step v1.3.1 used it on.

    Mutates ``s``. On any other step the block configured nothing in v1.3.1, so it is
    dropped, reported. A gradient-server block moves the keys v1.3.1 read from it
    (:data:`_V1_SERVER_KEYS`) and drops the rest, each reported — so a key v1.3.1 never read
    cannot meet one it did as two spellings of the same option.
    """
    engine = _folded(s.get("engine"))
    for block, (owner, elsewhere) in _LEGACY_BLOCKS.items():
        if block not in s:
            continue
        held = _mapping(s.pop(block), what=f"steps[{loc[1]}].{block}", of="option names to values")
        if engine != owner:
            sink(
                (*loc, block),
                f"step-level `{block}:` block is deprecated, and {elsewhere}; it configures "
                f"nothing on this step and is dropped",
            )
            continue
        sink((*loc, block), f"step-level `{block}:` block is deprecated; use `options:`")
        options = _mapping(
            s.get("options"), what=f"steps[{loc[1]}].options", of="option names to values"
        )
        for k, v in held.items():
            key = str(k)
            if key in _OBSOLETE_OPTION_KEYS:
                # Announced, not silently swallowed: a key that vanishes without a row
                # reads as either kept or mistyped.
                sink(
                    (*loc, block, key),
                    f"`{block}.{key}` is auto-managed now and was dropped from the moved block",
                )
            elif owner == "mlip-extopt" and key not in _V1_SERVER_KEYS:
                sink(
                    (*loc, block, key),
                    f"`{block}.{key}` configured nothing in v1.3.1, which read only "
                    f"{', '.join(f'`{name}`' for name in sorted(_V1_SERVER_KEYS))}; dropped",
                )
            else:
                options.setdefault(key, v)
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


def _write_out_v1_server(
    s: dict[str, Any], loc: tuple[str | int, ...], sink: DeprecationSink
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
      checkpoint (:func:`_names_a_file`) is what their loader read as a file, so it moves to
      ``model_path``, with a leading ``~`` expanded as the job's shell expanded it;
    * ``device: cuda`` when the step names none.

    First, v1.3.1's SevenNet spelling: a ``model_name`` starting ``sevenn`` beside a
    ``task_name`` none of v1.3.1's earlier dispatch prefixes matched (:data:`_V1_TASK_PREFIXES`)
    ran SevenNet with the ``task_name`` as its release, which is ``task_name: sevenn`` with the
    release as ``model_name`` now.
    """
    options = _mapping(
        s.get("options"), what=f"steps[{loc[1]}].options", of="option names to values"
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
    if isinstance(task, str) and task in _V1_MACE_LOADERS and "model_path" not in options:
        if "model_name" not in options:
            options["model_name"] = "medium"
            sink(
                (*loc, "options", "model_name"),
                f"v1.3.1 ran `{task}` with no `model_name` as `medium`; written out as "
                f"`model_name: medium`",
            )
        elif _names_a_file(name := options["model_name"]):
            del options["model_name"]
            options["model_path"] = str(Path(name).expanduser())
            sink(
                (*loc, "options", "model_path"),
                f"`model_name: {name}` names a checkpoint file, which is `model_path` now; a "
                f"relative path is read from the config file's directory, where v1.3.1 read "
                f"it from the job's scratch directory",
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

    main's knobs are renamed: ``calc_type`` → ``target`` (``rm_imag`` → ``ts``, the
    default; ``random`` unchanged), ``displacement_vector`` → ``displacement_value``;
    other keys pass through. Mutates ``s`` in place.
    """
    if "normal_mode_sampling" not in s and "normal_mode_sampling_parameters" not in s:
        return
    sink(
        (*loc, "normal_mode_sampling"),
        "`normal_mode_sampling*` is deprecated; use `nms` + `options`",
    )
    nms_on = bool(s.pop("normal_mode_sampling", False))
    if nms_on:
        s["nms"] = True
    params = _mapping(
        s.pop("normal_mode_sampling_parameters", None),
        what=f"steps[{loc[1]}].normal_mode_sampling_parameters",
        of="knob names to values",
    )
    opts = _mapping(s.get("options"), what=f"steps[{loc[1]}].options", of="option names to values")
    calc_type = str(params.pop("calc_type", "rm_imag")).lower()  # main default rm_imag → ts
    if nms_on:
        opts.setdefault("target", {"rm_imag": "ts"}.get(calc_type, calc_type))
    if "displacement_vector" in params:
        opts.setdefault("displacement_value", params.pop("displacement_vector"))
    for k, v in params.items():
        opts.setdefault(k, v)
    if opts:
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

    # Whether v1.3.1 ran this step on its gradient server, read before anything is
    # rewritten: it decides the defaults written out below.
    v1_server = _v1_server_step(s)
    if not _train_operation(s, loc, sink):
        _rename_engine(s, loc, sink)
    _move_blocks(s, loc, sink)
    if v1_server and _folded(s.get("engine")) == "mlip-extopt":
        _write_out_v1_server(s, loc, sink)

    _normalize_nms_keys(s, loc, sink)

    # sample_type{method, parameters} → sample{method, …}; legacy sample method /
    # key names (integer/high_energy/energy_window, num_structures/energy/…) are
    # rewritten to the v2 vocabulary (min/max/boltzmann, count/window_kcalmol)
    # whether they arrive via the old `sample_type` block or a direct `sample`.
    if "sample_type" in s:
        if "sample" not in s:
            sink((*loc, "sample_type"), "`sample_type` is deprecated; use `sample`")
            s["sample"] = _flatten_sample_type(s["sample_type"], what=f"steps[{index}].sample_type")
        s.pop("sample_type")
    # Normalize the resulting `sample` block (from sample_type, or a direct
    # block, possibly using legacy method/key names) to the v2 vocabulary.
    if isinstance(s.get("sample"), dict):
        s["sample"] = _normalize_sample_block(s["sample"], (*loc, "sample"), sink)

    return s


def _flatten_sample_type(sample_type: Any, *, what: str = "sample_type") -> Any:
    """Flatten a legacy ``sample_type`` ``{method, parameters: {…}}`` into a flat dict.

    ``what`` names the block in the refusal of a ``parameters`` that is not a mapping.
    """
    if not isinstance(sample_type, dict):
        return sample_type
    flat: dict[str, Any] = {"method": sample_type.get("method")}
    flat.update(
        _mapping(
            sample_type.get("parameters"), what=f"{what}.parameters", of="parameter names to values"
        )
    )
    for k, v in sample_type.items():  # carry through any non-nested extras
        if k not in ("method", "parameters"):
            flat.setdefault(k, v)
    return flat


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


def _normalize_sample_block(sample: Any, loc: tuple[str | int, ...], sink: DeprecationSink) -> Any:
    """Rewrite a flat ``sample`` dict's legacy method/key names to the v2 vocabulary.

    Idempotent: a current-vocabulary block (``min`` / ``max`` / ``boltzmann`` with
    ``count`` / ``window_kcalmol`` / ``percent_cumulative``) passes through unchanged.
    Announces one deprecation per legacy method or key actually rewritten.

    The legacy ``energy`` key carries a **unit conversion**, not just a rename. v1's
    ``energy_window`` read ``unit`` with a default of ``hartree`` and converted only an
    explicit ``kcal/mol`` — any other spelling, including none at all, meant the number
    was compared against hartree energies as-is. ``window_kcalmol`` is kcal/mol by
    definition, so carrying the number across unchanged silently shrank the window
    ~627.5x for every v1 config that relied on the default — and the rename row it got
    ("use `window_kcalmol`") read as an endorsement of the value. The conversion mirrors
    v1's own rule (an explicit ``kcal/mol`` passes through, everything else is hartree),
    and its announcement names both numbers so the translation is checkable at a glance.
    """
    if not isinstance(sample, dict):
        return sample
    out: dict[str, Any] = {}
    method = sample.get("method")
    if isinstance(method, str) and method in _SAMPLE_METHOD_RENAMES:
        new_method = _SAMPLE_METHOD_RENAMES[method]
        sink((*loc, "method"), f"sample method `{method}` is deprecated; use `{new_method}`")
        out["method"] = new_method
    elif method is not None:
        out["method"] = method
    unit = str(sample.get("unit", "hartree")).strip().lower()
    for raw_key, v in sample.items():
        key = str(raw_key)
        if key in ("method", "unit"):
            # `unit` is consumed by the `energy` conversion below rather than merely
            # dropped: v1 read it there and nowhere else, so it has no v2 key of its own.
            continue
        new_key = _SAMPLE_KEY_RENAMES.get(key, key)
        if key == "energy" and unit != "kcal/mol" and (number := _legacy_number(v)) is not None:
            converted = number * HARTREE_TO_KCALMOL
            sink(
                (*loc, key),
                f"sample key `energy` is deprecated; use `window_kcalmol` — and the unit "
                f"moves with it: v1 read {v!r} as hartree ({unit!r} converted nothing "
                f"there), so this becomes window_kcalmol: {converted:.6g}",
            )
            out.setdefault(new_key, converted)
            continue
        if new_key != key:
            sink((*loc, key), f"sample key `{key}` is deprecated; use `{new_key}`")
        out.setdefault(new_key, v)
    return out
