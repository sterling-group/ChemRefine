"""ORCA-driven PySCF engine (ORCA optimises, PySCF provides gradients).

Mirrors :class:`~chemrefine.engines.mlip.extopt_engine.MlipExtOptEngine` — a pure declaration
over :class:`~chemrefine.engines.orca.extopt.engine.ExtOptOrcaEngine`, differing only in the
backend name + option/calculator classes (method / xc / basis / df / gpu / tensor knobs live on
:class:`PyscfExtOptCalculator`). The one extra behaviour is copying the ``save_tensors`` output
directory back into each structure's dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from chemrefine.engines._backend_server.base import ComputeBackend
from chemrefine.engines._options import EngineOptions
from chemrefine.engines.api import register
from chemrefine.engines.orca.extopt.engine import ExtOptOrcaEngine
from chemrefine.engines.pyscf.backend import PyscfBackend
from chemrefine.engines.pyscf.extopt_calc import PyscfExtOptCalculator
from chemrefine.engines.pyscf.options import PyscfOptions
from chemrefine.errors import ConfigError
from chemrefine.state import StepContext, StepInputs


@register("pyscf-extopt")
class PyscfExtOptEngine(PyscfBackend, ExtOptOrcaEngine):
    """ORCA optimisation backed by a PySCF gradient server."""

    name: ClassVar[str] = "pyscf-extopt"
    backend: ClassVar[str] = "pyscf"
    wrapper_filename: ClassVar[str] = "pyscf_extopt.sh"
    options_cls: ClassVar[type[EngineOptions]] = PyscfOptions
    calculator_cls: ClassVar[type[ComputeBackend]] = PyscfExtOptCalculator

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Refuse ``save_tensors`` on an open-shell step before anything submits.

        The tensor transform is restricted-only (:func:`chemrefine.engines.pyscf._runtime.
        get_active_space_tensors` states why), and left to run it fails *after* the SCF and
        the gradient succeeded — inside the server, as a 500 whose actionable half lands in
        the server log rather than in the user's ledger. Here the step's effective
        multiplicity is in hand and nothing has been spent, so the refusal names the knob
        and the spin with the documented exit code.
        """
        opts = PyscfOptions.from_raw(ctx.step_cfg.options)
        if opts.save_tensors and ctx.multiplicity != 1:
            raise ConfigError(
                f"step {ctx.step_cfg.step}: save_tensors supports closed-shell systems "
                f"only (RHF/RKS), and this step's effective multiplicity is "
                f"{ctx.multiplicity}. Drop `save_tensors: true`, or run the step as a "
                f"closed-shell system."
            )
        return super().prepare(ctx)

    def output_dirs(self, ctx: StepContext) -> tuple[str, ...]:
        """Copy the ``save_tensors`` output directory back into the structure dir.

        The server writes ``<tensor_folder>/<tag>.npz`` under ``$WORK_DIR``; a relative
        ``tensor_folder`` is copied back so the tensors persist beside the structure's other
        artifacts. An absolute ``tensor_folder`` already persists at its own location, so
        nothing extra is copied.
        """
        opts = PyscfOptions.from_raw(ctx.step_cfg.options)
        if opts.save_tensors and not Path(opts.tensor_folder).is_absolute():
            return (opts.tensor_folder,)
        return ()
