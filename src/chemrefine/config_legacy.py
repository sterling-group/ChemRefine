"""Translate v1.3.1 YAML into the current schema — the only module that knows the old names.

Every legacy spelling ChemRefine still accepts is rewritten here, once, before validation:
:meth:`chemrefine.config.Config._normalize_legacy_yaml` calls :func:`normalize` and nothing
else in the package knows these names exist. A config that is already current passes through
untouched, and each rewrite logs one deprecation warning naming what to change.

**Removal horizon: 3.0.** This layer is correct and single-homed, but it is the largest
concentration of branches in the config path and it has no reason to grow — every key it
handles was renamed before 2.0. Deleting the module and its one call site is the whole
removal; see ``docs/get-started/upgrading-from-v1.md``.

Kept out of :mod:`chemrefine.config` so the schema reads as the schema. Someone learning
what a step *is* should not have to read two hundred lines of translation for older ones.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from chemrefine.errors import ConfigError
from chemrefine.quantities import HARTREE_TO_KCALMOL

logger = logging.getLogger(__name__)

DeprecationSink = Callable[[tuple[str | int, ...], str], None]
"""``(loc, message) -> None`` — where a rewrite announces itself.

Every legacy spelling this module rewrites is a finding, and until it had somewhere to go
other than a logger it was a finding only for whoever happened to be watching stderr. The
agent, the GUI's Validate button and every MCP client read
:class:`~chemrefine.validate.ValidationReport` and nothing else, so they were told a config
using a removed-in-3.0 key was clean. ``loc`` follows pydantic's convention, the same one
:class:`~chemrefine.validate.ValidationIssue` anchors every other finding with."""


def _log_deprecation(loc: tuple[str | int, ...], message: str) -> None:
    """The default sink: what this module has always done, unchanged."""
    logger.warning("%s", message)


#: Old engine name → canonical. Includes the ``mlff`` ↔ ``mlip`` rename aliases.
_ENGINE_RENAMES = {
    "mlff": "mlip",
    "mlff-extopt": "mlip-extopt",
    "mlff-train": "mlip-train",
    "dft": "orca",
}

#: Legacy step-level engine-config block → the canonical engine it implies.
#: v1.3.1 ran MLFF/PySCF through ORCA (the ``mlff:`` block configured the
#: gradient server), so they map to the ``*-extopt`` engines.
_LEGACY_BLOCKS = {
    "mlff": "mlip-extopt",
    "pyscf": "pyscf-extopt",
    "trainer": "mlip-train",
}

#: Option sub-keys that are auto-managed now and dropped from a moved block.
_OBSOLETE_OPTION_KEYS = frozenset({"bind"})

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


def normalize(raw: dict[str, Any], *, report: DeprecationSink | None = None) -> dict[str, Any]:
    """Rewrite legacy (v1.3.1 / ``mlff``-named) YAML keys to the current schema.

    The single place that knows the old vocabulary. Idempotent — new-style input
    passes through unchanged — and announces one deprecation per legacy feature
    rewritten. ``calculation_type`` is intentionally unsupported and raises
    :class:`~chemrefine.errors.ConfigError`.

    ``report`` redirects those announcements to a caller instead of the log, which is how
    :func:`chemrefine.validate.validate_config_text` turns them into report warnings. It is
    a redirect rather than an addition on purpose: the validator runs this twice — once to
    collect, once inside ``Config.model_validate`` — and a sink that also logged would say
    everything twice on a path that used to say it once.
    """
    sink = report or _log_deprecation
    out = dict(raw)
    if "orca_executable" in out:
        sink(
            ("orca_executable",), "`orca_executable` is deprecated; use `executables: {orca: ...}`"
        )
        execs = dict(out.get("executables") or {})
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


def _move_engine_block(
    s: dict[str, Any], loc: tuple[str | int, ...], sink: DeprecationSink
) -> str | None:
    """Fold a legacy engine block (``mlff:``/``pyscf:``/``trainer:``) into ``options``.

    Mutates ``s`` (pops the block, merges its keys into ``options``) and returns the
    canonical engine that block implies, or ``None`` when no block is present.
    """
    options = dict(s.get("options") or {})
    block_engine: str | None = None
    for block, engine_name in _LEGACY_BLOCKS.items():
        if isinstance(s.get(block), dict):
            sink((*loc, block), f"step-level `{block}:` block is deprecated; use `options:`")
            for k, v in s.pop(block).items():
                if k not in _OBSOLETE_OPTION_KEYS:
                    options.setdefault(k, v)
            block_engine = engine_name
    if options or "options" in s:
        s["options"] = options
    return block_engine


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
    params = dict(s.pop("normal_mode_sampling_parameters", None) or {})
    opts = dict(s.get("options") or {})
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

    # Engine name: a moved engine-config block decides it, else the rename map.
    block_engine = _move_engine_block(s, loc, sink)
    if block_engine is not None:
        s["engine"] = block_engine
    elif isinstance(s.get("engine"), str):
        eng = s["engine"].lower()
        s["engine"] = _ENGINE_RENAMES.get(eng, eng)

    # Operation: ``OPT+SP`` → ``opt_sp``, ``GOAT`` → ``goat`` (engines lower/replace too).
    if isinstance(s.get("operation"), str):
        s["operation"] = s["operation"].lower().replace("+", "_")
        # ``MLFF_TRAIN`` was a training *operation* that implied the trainer engine.
        if s["operation"] in ("mlff_train", "mlip_train"):
            s["engine"] = "mlip-train"
            s["operation"] = "mlip_train"

    _normalize_nms_keys(s, loc, sink)

    # sample_type{method, parameters} → sample{method, …}; legacy sample method /
    # key names (integer/high_energy/energy_window, num_structures/energy/…) are
    # rewritten to the v2 vocabulary (min/max/boltzmann, count/window_kcalmol)
    # whether they arrive via the old `sample_type` block or a direct `sample`.
    if "sample_type" in s:
        if "sample" not in s:
            sink((*loc, "sample_type"), "`sample_type` is deprecated; use `sample`")
            s["sample"] = _flatten_sample_type(s["sample_type"])
        s.pop("sample_type")
    # Normalize the resulting `sample` block (from sample_type, or a direct
    # block, possibly using legacy method/key names) to the v2 vocabulary.
    if isinstance(s.get("sample"), dict):
        s["sample"] = _normalize_sample_block(s["sample"], (*loc, "sample"), sink)

    return s


def _flatten_sample_type(sample_type: Any) -> Any:
    """Flatten a legacy ``sample_type`` ``{method, parameters: {…}}`` into a flat dict."""
    if not isinstance(sample_type, dict):
        return sample_type
    flat: dict[str, Any] = {"method": sample_type.get("method")}
    flat.update(sample_type.get("parameters") or {})
    for k, v in sample_type.items():  # carry through any non-nested extras
        if k not in ("method", "parameters"):
            flat.setdefault(k, v)
    return flat


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
        if (
            key == "energy"
            and unit != "kcal/mol"
            and isinstance(v, (int, float))
            and not isinstance(v, bool)
        ):
            converted = v * HARTREE_TO_KCALMOL
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
