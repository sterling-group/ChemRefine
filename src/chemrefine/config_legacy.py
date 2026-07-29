"""Translate v1.3.1 YAML into the current schema — the only module that knows the old names.

Every legacy spelling ChemRefine still accepts is rewritten here, once, before validation:
:meth:`chemrefine.config.Config._normalize_legacy_yaml` calls :func:`normalize` and nothing
else in the package knows these names exist. A config that is already current passes through
untouched, and each rewrite logs one deprecation warning naming what to change.

**Removal horizon: 3.0.** This layer is correct and single-homed, but it is the largest
concentration of branches in the config path and it has no reason to grow — every key it
handles was renamed before 2.0. Deleting the module and its one call site is the whole
removal; see ``docs/migrating-v1-to-v2.md``.

Kept out of :mod:`chemrefine.config` so the schema reads as the schema. Someone learning
what a step *is* should not have to read two hundred lines about what a step used to be.
"""

from __future__ import annotations

import logging
from typing import Any

from chemrefine.errors import ConfigError

logger = logging.getLogger(__name__)


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


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Rewrite legacy (v1.3.1 / ``mlff``-named) YAML keys to the current schema.

    The single place that knows the old vocabulary. Idempotent — new-style input
    passes through unchanged — and logs one warning per legacy feature rewritten.
    ``calculation_type`` is intentionally unsupported and raises
    :class:`~chemrefine.errors.ConfigError`.
    """
    out = dict(raw)
    if "orca_executable" in out:
        logger.warning("`orca_executable` is deprecated; use `executables: {orca: ...}`")
        execs = dict(out.get("executables") or {})
        execs.setdefault("orca", out.pop("orca_executable"))
        out["executables"] = execs
    if "initial_xyz" in out:
        if "input" not in out:
            logger.warning("`initial_xyz` is deprecated; use `input`")
            out["input"] = out["initial_xyz"]
        out.pop("initial_xyz")
    if isinstance(out.get("steps"), list):
        out["steps"] = [_normalize_step(s) for s in out["steps"]]
    return out


def _move_engine_block(s: dict[str, Any]) -> str | None:
    """Fold a legacy engine block (``mlff:``/``pyscf:``/``trainer:``) into ``options``.

    Mutates ``s`` (pops the block, merges its keys into ``options``) and returns the
    canonical engine that block implies, or ``None`` when no block is present.
    """
    options = dict(s.get("options") or {})
    block_engine: str | None = None
    for block, engine_name in _LEGACY_BLOCKS.items():
        if isinstance(s.get(block), dict):
            logger.warning("step-level `%s:` block is deprecated; use `options:`", block)
            for k, v in s.pop(block).items():
                if k not in _OBSOLETE_OPTION_KEYS:
                    options.setdefault(k, v)
            block_engine = engine_name
    if options or "options" in s:
        s["options"] = options
    return block_engine


def _normalize_nms_keys(s: dict[str, Any]) -> None:
    """Rewrite legacy ``normal_mode_sampling{,_parameters}`` into ``nms`` + ``options``.

    main's knobs are renamed: ``calc_type`` → ``target`` (``rm_imag`` → ``ts``, the
    default; ``random`` unchanged), ``displacement_vector`` → ``displacement_value``;
    other keys pass through. Mutates ``s`` in place.
    """
    if "normal_mode_sampling" not in s and "normal_mode_sampling_parameters" not in s:
        return
    logger.warning("`normal_mode_sampling*` is deprecated; use `nms` + `options`")
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


def _normalize_step(step: Any) -> Any:
    """Rewrite one legacy step dict to the current schema (helper for :func:`_normalize_legacy`)."""
    if not isinstance(step, dict):
        return step
    s = dict(step)

    if "calculation_type" in s:
        raise ConfigError(
            "`calculation_type` is no longer supported; use `engine:` + `operation:` "
            "(see docs/migrating-v1-to-v2.md)"
        )

    # Engine name: a moved engine-config block decides it, else the rename map.
    block_engine = _move_engine_block(s)
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

    _normalize_nms_keys(s)

    # sample_type{method, parameters} → sample{method, …}; legacy sample method /
    # key names (integer/high_energy/energy_window, num_structures/energy/…) are
    # rewritten to the v2 vocabulary (min/max/boltzmann, count/window_kcalmol)
    # whether they arrive via the old `sample_type` block or a direct `sample`.
    if "sample_type" in s:
        if "sample" not in s:
            logger.warning("`sample_type` is deprecated; use `sample`")
            s["sample"] = _flatten_sample_type(s["sample_type"])
        s.pop("sample_type")
    # Normalize whatever `sample` we now have (from sample_type, or a direct
    # block, possibly using legacy method/key names) to the v2 vocabulary.
    if isinstance(s.get("sample"), dict):
        s["sample"] = _normalize_sample_block(s["sample"])

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


def _normalize_sample_block(sample: Any) -> Any:
    """Rewrite a flat ``sample`` dict's legacy method/key names to the v2 vocabulary.

    Idempotent: a current-vocabulary block (``min`` / ``max`` / ``boltzmann`` with
    ``count`` / ``window_kcalmol`` / ``percent_cumulative``) passes through unchanged.
    Logs one deprecation warning per legacy method or key actually rewritten.
    """
    if not isinstance(sample, dict):
        return sample
    out: dict[str, Any] = {}
    method = sample.get("method")
    if isinstance(method, str) and method in _SAMPLE_METHOD_RENAMES:
        new_method = _SAMPLE_METHOD_RENAMES[method]
        logger.warning("sample method `%s` is deprecated; use `%s`", method, new_method)
        out["method"] = new_method
    elif method is not None:
        out["method"] = method
    for raw_key, v in sample.items():
        key = str(raw_key)
        if key in ("method", "unit"):  # energy_window unit (kcal/mol) is implicit now
            continue
        new_key = _SAMPLE_KEY_RENAMES.get(key, key)
        if new_key != key:
            logger.warning("sample key `%s` is deprecated; use `%s`", key, new_key)
        out.setdefault(new_key, v)
    return out
