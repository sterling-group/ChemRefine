"""PySCF backend for the shared ExtOpt server.

Wraps :mod:`chemrefine.engines.pyscf._runtime` so the shared ExtOpt
server can serve PySCF gradients via the same
:class:`ComputeBackend` contract MLIP uses. **Single channel:** the calculator
holds the step's validated :class:`~chemrefine.engines.pyscf.options.PyscfExtOptOptions`,
built once, at server construction, from the argv the engine wrote out of the step's
YAML options — the wrapper and the per-call POST carry nothing. Optional
active-space tensor extraction is gated on the server-constructed
``save_tensors`` flag; only the per-call correlation ``tag`` rides the
request (so dumps don't overwrite each other).

The SCF + gradient body needs PySCF + (optionally) gpu4pyscf
installed; tests under :file:`tests/test_engines_pyscf_extopt_calc.py`
mock those imports so the call graph can be exercised in CI.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from chemrefine.engines._backend_server.base import (
    CalculationData,
    ComputeBackend,
    tokens_from_options,
)
from chemrefine.engines.pyscf import _runtime
from chemrefine.engines.pyscf.options import PyscfExtOptOptions
from chemrefine.errors import JobFailureError

logger = logging.getLogger(__name__)

_SERVER_KNOBS: tuple[str, ...] = (
    "method",
    "xc",
    "basis",
    "df",
    "gpu",
    "save_tensors",
    "localized",
    "tensor_folder",
    "strict_scf",
)
"""The :class:`PyscfExtOptOptions` fields the server is built from — its CLI surface.

Each rides as a flag spelled like its field (flag == YAML key == field name), with its
default and its choices read off the model, so the parser and the model cannot disagree."""

_OPT_OUT: frozenset[str] = frozenset({"strict_scf"})
"""Boolean knobs whose safe state is on, spelled as their opt-out (``--no-strict-scf``).

A guard spelled as a positive flag fails open: anything that drops the token turns it off.
Spelled as the opt-out, a dropped token leaves it on — see
:func:`~chemrefine.engines._backend_server.base.tokens_from_options`."""

_BOOLEANS = frozenset(
    name for name in _SERVER_KNOBS if PyscfExtOptOptions.model_fields[name].annotation is bool
)
_VALUE_FLAGS = tuple((name, f"--{name}") for name in _SERVER_KNOBS if name not in _BOOLEANS)
_BOOL_FLAGS = tuple(name for name in _SERVER_KNOBS if name in _BOOLEANS - _OPT_OUT)
_FALSE_FLAGS = tuple(
    (name, f"--no-{name.replace('_', '-')}") for name in _SERVER_KNOBS if name in _OPT_OUT
)

_FLAG_HELP = {
    "method": "SCF method",
    "xc": "DFT exchange-correlation functional (required for --method dft)",
    "basis": "Orbital basis set (required)",
    "df": "Enable density fitting / RI",
    "gpu": "Attempt gpu4pyscf if installed",
    "save_tensors": "Dump active-space 1e/2e MO tensors after the SCF",
    "localized": "Boys-localize occupied / virtual orbitals before tensor extraction",
    "tensor_folder": "Directory (relative to $WORK_DIR) for save_tensors .npz output",
    "strict_scf": "Serve a gradient even when the SCF did not converge",
}
"""CLI presentation only; names, defaults and choices come from the options model."""

# The correlation tag becomes a filename component; the request is untrusted
# (any same-host client can POST to the loopback server), so squash anything
# that could traverse out of tensor_folder.
_TAG_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


class PyscfExtOptCalculator(ComputeBackend):
    """ExtOpt-side adapter for PySCF / gpu4pyscf gradient calls."""

    name = "pyscf"

    def __init__(self, options: PyscfExtOptOptions) -> None:
        """Hold the step's validated options, its level of theory checked once, here.

        Checked at construction rather than per gradient: a model built directly rather
        than through :meth:`PyscfExtOptOptions.from_raw` can still name no level of
        theory, and the rule answers with the basis it proved named.
        """
        self.options = options
        self._basis = PyscfExtOptOptions.require_level_of_theory(options.model_dump())

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register the server knobs on a shared server / client parser — one model, no table.

        A value knob's default and its choices (a ``Literal``'s values) are its field's. A
        boolean is a ``store_true`` whose argparse default is off, whatever the model's:
        the engine emits every resolved value as a token (``--df`` when on, nothing when
        off — see :meth:`server_cli_from_options`), so an omitted token must mean off for
        a ``df: false`` step to survive the trip. Only a hand-run server reaches that
        default, and a hand-run without ``--df`` runs the bare SCF. An opt-out
        (:data:`_OPT_OUT`) inverts it: ``--no-strict-scf`` stores false, and its absence
        leaves the guard on.
        """
        for name, flag in _VALUE_FLAGS:
            field = PyscfExtOptOptions.model_fields[name]
            kwargs: dict[str, Any] = {"default": field.default, "help": _FLAG_HELP[name]}
            if get_origin(field.annotation) is Literal:
                kwargs["choices"] = get_args(field.annotation)
            parser.add_argument(flag, **kwargs)
        for name in _BOOL_FLAGS:
            parser.add_argument(f"--{name}", action="store_true", help=_FLAG_HELP[name])
        for name, flag in _FALSE_FLAGS:
            parser.add_argument(flag, dest=name, action="store_false", help=_FLAG_HELP[name])

    @classmethod
    def server_cli_from_options(cls, options: dict[str, Any]) -> list[str]:
        """Translate validated YAML options into the flags :meth:`add_cli_args` registers.

        The same three groups, so the engine writes exactly what the server parses; the
        server injects the per-call correlation ``tag`` itself (single-channel, like MLIP).
        """
        return tokens_from_options(
            options, value_flags=_VALUE_FLAGS, bool_flags=_BOOL_FLAGS, false_flags=_FALSE_FLAGS
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PyscfExtOptCalculator:
        """Build the calculator from the server's argv, through the model's strict read.

        :meth:`PyscfExtOptOptions.from_raw` refuses what it refuses for a step, so a
        hand-run server that names no basis fails at startup rather than on every gradient.
        """
        return cls(
            PyscfExtOptOptions.from_raw({name: getattr(args, name) for name in _SERVER_KNOBS})
        )

    def calc(self, data: CalculationData) -> tuple[float, list[list[float]]]:
        """Run the SCF + (optional) gradient, return ``(energy, gradient)`` in atomic units.

        Single channel: the SCF knobs come from this instance (built once on the
        server from the step's YAML options); only the per-call correlation
        ``tag`` is read off the request, to key ``save_tensors`` dumps.

        A non-converged SCF raises unless the step set ``strict_scf: false``. PySCF returns
        the last iterate rather than raising, so nothing downstream would notice: ORCA would
        step on a gradient from a non-stationary density and record its own geometry
        convergence in the ``.out``, which says nothing about the backend. Raising turns it
        into the ordinary failure the bridge already reports — the server's 500 becomes a
        :class:`~chemrefine.errors.JobFailureError` client-side, and the detail lands in the
        ExtOpt server log beside the structure's other artifacts.
        """
        opts = self.options
        mol = _runtime.build_mol(
            symbols=data.symbols,
            positions_angstrom=data.positions_angstrom,
            charge=data.charge,
            multiplicity=data.multiplicity,
            basis=self._basis,
        )
        energy, gradient, meta, mf = _runtime.run_dft(
            mol,
            method=opts.method,
            xc=opts.xc,
            use_df=opts.df,
            want_gpu=opts.gpu,
            nthreads=data.nthreads,
            dograd=data.dograd,
        )
        logger.info(
            "PySCF calc: E=%.10f Eh converged=%s gpu=%s t=%.3fs",
            energy,
            meta["converged"],
            meta["gpu_used"],
            meta["elapsed_seconds"],
        )
        if opts.gpu and not meta["gpu_used"]:
            # `_build_scf` answers a GPU that could not be set up by falling back to CPU
            # and *returning* the reason; this is where that reason reaches a log. Without
            # it a `device: cuda` step whose gpu4pyscf import passes preflight on the login
            # node and fails to construct on the compute node would run every gradient on
            # CPU, with `gpu=False` in the line above the only trace and the exception dropped.
            logger.warning("%s", meta["gpu_msg"])
        if opts.strict_scf and not meta["converged"]:
            raise JobFailureError(
                f"PySCF SCF did not converge (E={energy:.10f} Eh, method={opts.method}, "
                f"xc={opts.xc}, basis={self._basis}); a gradient from a non-stationary "
                f"density is not usable. Tighten the SCF, or set `strict_scf: false` in "
                f"the step's options to accept it."
            )

        if opts.save_tensors:
            # ``tag`` is the per-call correlation id the bridge derives from the
            # ``.extinp.tmp`` stem (one file per ORCA geometry step); the server
            # injects it into ``settings`` so dumps don't overwrite each other.
            # Sanitised before use as a filename — the value arrives over HTTP.
            tag = _TAG_UNSAFE_RE.sub("_", str(data.settings.get("tag") or "untagged"))
            nuc, h1, h2 = _runtime.get_active_space_tensors(mol, mf, localized=opts.localized)
            target = Path(opts.tensor_folder) / f"{tag}.npz"
            _runtime.save_tensors(path=target, nuc=nuc, h1=h1, h2=h2)
            logger.info("PySCF tensors saved: %s", target)

        return energy, gradient
