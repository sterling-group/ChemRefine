"""PySCF backend for the shared ExtOpt server.

Wraps :mod:`chemrefine.engines.pyscf._runtime` so the shared ExtOpt
server can serve PySCF gradients via the same
:class:`ComputeBackend` contract MLIP uses. **Single channel:** the SCF
knobs (method / xc / basis / df / gpu / tensor settings) are baked into
the calculator once, at server construction, from the step's YAML
options — the wrapper and the per-call POST carry nothing. Optional
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
from typing import Any

from chemrefine.engines._backend_server.base import (
    CalculationData,
    ComputeBackend,
    tokens_from_options,
)
from chemrefine.engines.pyscf import _runtime
from chemrefine.engines.pyscf.options import PyscfExtOptOptions
from chemrefine.errors import JobFailureError

logger = logging.getLogger(__name__)

# CLI flag names. The Pydantic ``PyscfOptions`` model owns the *defaults*;
# this tuple lists which fields are exposed on the ExtOpt CLI surface
# (server + client + engine run_block) so the three callers stay in lockstep.
# Flag spelling == YAML key == Pydantic field (underscores), so the generic
# ``--{key}`` token builder below needs no per-flag special-casing.
_KEY_VALUE_FLAGS: tuple[str, ...] = ("method", "xc", "basis", "tensor_folder")
_BOOL_FLAGS: tuple[str, ...] = ("df", "gpu", "save_tensors", "localized")
# Spelled as the opt-*out* so the guard fails closed: a dropped token leaves it on.
_FALSE_FLAGS: tuple[tuple[str, str], ...] = (("strict_scf", "--no-strict-scf"),)

# The correlation tag becomes a filename component; the request is untrusted
# (any same-host client can POST to the loopback server), so squash anything
# that could traverse out of tensor_folder.
_TAG_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


class PyscfExtOptCalculator(ComputeBackend):
    """ExtOpt-side adapter for PySCF / gpu4pyscf gradient calls."""

    name = "pyscf"

    def __init__(
        self,
        *,
        method: str = "dft",
        xc: str | None = None,
        basis: str | None = None,
        df: bool = True,
        gpu: bool = False,
        save_tensors: bool = False,
        localized: bool = False,
        tensor_folder: str = "tensors",
        strict_scf: bool = True,
    ) -> None:
        self.method = method
        self.xc = xc
        self.basis = basis
        self.df = df
        self.gpu = gpu
        self.save_tensors = save_tensors
        self.localized = localized
        self.tensor_folder = tensor_folder
        # Every default here restates PyscfExtOptOptions' — the model is the canonical
        # source, and a directly-constructed calculator must be the same calculator a
        # YAML step's defaults build. `df` sat off here for a release after the model
        # flipped it on, so a bare construction solved a different SCF shape than a
        # default step; the lockstep test now holds every shared knob equal.
        self.strict_scf = strict_scf

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register PySCF flags on a shared server / client parser.

        Defaults mirror :class:`PyscfExtOptOptions` — with one deliberate exception.
        ``--df`` is a ``store_true`` whose argparse default stays ``False`` although the
        model's is ``True``: the engine emits every *resolved* value as a token
        (``--df`` when on, nothing when off — see :meth:`server_cli_from_options`), so
        the argparse default is what an omitted token means, and it has to be the off
        state for a ``df: false`` step to survive the trip. No generated run block can
        reach the argparse default; only a hand-run server does, and a hand-run without
        ``--df`` runs the bare SCF. The Pydantic model stays the canonical name +
        default source everywhere else. Adding a knob here means also adding it to
        ``PyscfExtOptOptions`` (or vice-versa) — the ``_KEY_VALUE_FLAGS`` /
        ``_BOOL_FLAGS`` tuples gate which knobs are CLI-exposed.
        """
        defaults = PyscfExtOptOptions()
        parser.add_argument(
            "--method",
            default=defaults.method,
            choices=["dft", "hf"],
            help="SCF method (dft | hf)",
        )
        parser.add_argument(
            "--xc",
            default=defaults.xc,
            help="DFT exchange-correlation functional (required for --method dft)",
        )
        parser.add_argument(
            "--basis",
            default=defaults.basis,
            help="Orbital basis set (required)",
        )
        parser.add_argument(
            "--df",
            action="store_true",
            help="Enable density fitting / RI",
        )
        parser.add_argument(
            "--gpu",
            action="store_true",
            help="Attempt gpu4pyscf if installed",
        )
        parser.add_argument(
            "--save_tensors",
            action="store_true",
            help="Dump active-space 1e/2e MO tensors after the SCF",
        )
        parser.add_argument(
            "--localized",
            action="store_true",
            help="Boys-localize occupied / virtual orbitals before tensor extraction",
        )
        parser.add_argument(
            "--tensor_folder",
            default=defaults.tensor_folder,
            help="Directory (relative to $WORK_DIR) for save_tensors .npz output",
        )
        # The one negative flag: `strict_scf` is on unless the step opts out, so argparse
        # must default it on too — the other booleans here can default off because their
        # off state is the harmless one.
        parser.add_argument(
            "--no-strict-scf",
            dest="strict_scf",
            action="store_false",
            default=defaults.strict_scf,
            help="Serve a gradient even when the SCF did not converge",
        )

    @classmethod
    def server_cli_from_options(cls, options: dict[str, Any]) -> list[str]:
        """Translate validated YAML options into ``--flag value`` / ``--flag`` tokens.

        PySCF flag spelling == YAML key, so each value flag maps to ``--{key}``; the server
        still injects the per-call correlation ``tag`` itself (single-channel, like MLIP).
        ``strict_scf`` is the exception, emitted as its opt-out so a missing token cannot
        turn the guard off — see
        :func:`~chemrefine.engines._backend_server.base.tokens_from_options`.
        """
        return tokens_from_options(
            options,
            value_flags=tuple((key, f"--{key}") for key in _KEY_VALUE_FLAGS),
            bool_flags=_BOOL_FLAGS,
            false_flags=_FALSE_FLAGS,
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PyscfExtOptCalculator:
        """Read the PySCF-relevant fields off the shared server CLI namespace."""
        return cls(
            method=args.method,
            xc=args.xc,
            basis=args.basis,
            df=args.df,
            gpu=args.gpu,
            save_tensors=args.save_tensors,
            localized=args.localized,
            tensor_folder=args.tensor_folder,
            strict_scf=args.strict_scf,
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
        # The level-of-theory rule, at the server's own boundary. The engine's strict read
        # enforces it before any job is generated, but this class is constructible bare —
        # the lockstep test holds its defaults equal to the model's, which are now None —
        # and a hand-run server carries only what its argv said. Reconstructing the
        # named-knobs view lets the options model's one rule (and wording) answer here too.
        named: dict[str, Any] = {"method": self.method}
        if self.basis is not None:
            named["basis"] = self.basis
        if self.xc is not None:
            named["xc"] = self.xc
        basis = PyscfExtOptOptions.require_level_of_theory(named)
        mol = _runtime.build_mol(
            symbols=data.symbols,
            positions_angstrom=data.positions_angstrom,
            charge=data.charge,
            multiplicity=data.multiplicity,
            basis=basis,
        )
        energy, gradient, meta, mf = _runtime.run_dft(
            mol,
            method=self.method,
            xc=self.xc,
            use_df=self.df,
            want_gpu=self.gpu,
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
        if self.strict_scf and not meta["converged"]:
            raise JobFailureError(
                f"PySCF SCF did not converge (E={energy:.10f} Eh, method={self.method}, "
                f"xc={self.xc}, basis={self.basis}); a gradient from a non-stationary "
                f"density is not usable. Tighten the SCF, or set `strict_scf: false` in "
                f"the step's options to accept it."
            )

        if self.save_tensors:
            # ``tag`` is the per-call correlation id the bridge derives from the
            # ``.extinp.tmp`` stem (one file per ORCA geometry step); the server
            # injects it into ``settings`` so dumps don't overwrite each other.
            # Sanitised before use as a filename — the value arrives over HTTP.
            tag = _TAG_UNSAFE_RE.sub("_", str(data.settings.get("tag") or "untagged"))
            nuc, h1, h2 = _runtime.get_active_space_tensors(mol, mf, localized=self.localized)
            target = Path(self.tensor_folder) / f"{tag}.npz"
            _runtime.save_tensors(path=target, nuc=nuc, h1=h1, h2=h2)
            logger.info("PySCF tensors saved: %s", target)

        return energy, gradient
