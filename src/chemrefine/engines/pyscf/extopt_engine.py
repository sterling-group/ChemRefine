"""ORCA-driven PySCF engine (ORCA optimises, PySCF provides gradients).

Mirrors :class:`~chemrefine.engines.mlip.extopt_engine.MlipExtOptEngine` — a pure declaration
over :class:`~chemrefine.engines.orca.extopt.engine.ExtOptOrcaEngine`, differing only in the
backend name + option/calculator classes. Its knobs are
:class:`~chemrefine.engines.pyscf.options.PyscfExtOptOptions` — the SCF selection the direct
engine also reads, plus the ones only the gradient server acts on. The one extra behaviour is
copying the ``save_tensors`` output directory back into each structure's dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from chemrefine.config import StepConfig
from chemrefine.engines._backend_server.base import ComputeBackend
from chemrefine.engines.api import register
from chemrefine.engines.orca.extopt.engine import ExtOptOrcaEngine
from chemrefine.engines.pyscf.backend import PyscfBackend
from chemrefine.engines.pyscf.extopt_calc import PyscfExtOptCalculator
from chemrefine.engines.pyscf.options import PyscfExtOptOptions
from chemrefine.errors import ConfigError
from chemrefine.state import StepContext, StepInputs


@register("pyscf-extopt")
class PyscfExtOptEngine(PyscfBackend, ExtOptOrcaEngine):
    """ORCA optimisation backed by a PySCF gradient server."""

    name: ClassVar[str] = "pyscf-extopt"
    backend: ClassVar[str] = "pyscf"
    wrapper_filename: ClassVar[str] = "pyscf_extopt.sh"
    options_cls: ClassVar[type[PyscfExtOptOptions]] = PyscfExtOptOptions
    calculator_cls: ClassVar[type[ComputeBackend]] = PyscfExtOptCalculator

    def _opts(self, ctx: StepContext) -> PyscfExtOptOptions:
        """This step's validated options, read through the model this engine declares.

        Through :attr:`options_cls` rather than by naming the model again: two readers of one
        knob is what the ClassVar exists to prevent. The ClassVar is narrowed to this engine's
        own model, so the read is typed without a cast or a narrowing assertion.
        """
        return self.options_cls.from_raw(ctx.step_cfg.options)

    def check_step(self, step_cfg: StepConfig, *, charge: int, multiplicity: int) -> None:
        """Refuse ``save_tensors`` on an open-shell step, on top of the base's strict read.

        The tensor transform is restricted-only (:func:`chemrefine.engines.pyscf._runtime.
        get_active_space_tensors` states why), and left to run it fails *after* the SCF and
        the gradient succeeded — inside the server, as a 500 whose actionable half lands in
        the server log rather than in the user's ledger. The refusal needs only the config
        and the step's effective multiplicity, so it belongs on the preflight hook: the
        run's t=0 walk and ``chemrefine validate`` both make it before anything is spent,
        and :meth:`prepare` repeats it for the paths that skip the preflight.
        """
        super().check_step(step_cfg, charge=charge, multiplicity=multiplicity)
        opts = self.options_cls.from_raw(step_cfg.options)
        if opts.save_tensors and multiplicity != 1:
            raise ConfigError(
                f"step {step_cfg.step}: save_tensors supports closed-shell systems "
                f"only (RHF/RKS), and this step's effective multiplicity is "
                f"{multiplicity}. Drop `save_tensors: true`, or run the step as a "
                f"closed-shell system."
            )

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Re-make the preflight refusals, then write the inputs.

        :meth:`check_step` already ran at the run's preflight walk; repeating it here
        covers the recovery paths that reach ``prepare`` without one, for the price of
        one cheap validation.
        """
        self.check_step(ctx.step_cfg, charge=ctx.charge, multiplicity=ctx.multiplicity)
        return super().prepare(ctx)

    def output_dirs(self, ctx: StepContext) -> tuple[str, ...]:
        """Copy the ``save_tensors`` output directory back into the structure dir.

        The server writes ``<tensor_folder>/<tag>.npz`` under ``$WORK_DIR``; a relative
        ``tensor_folder`` is copied back so the tensors persist beside the structure's other
        artifacts. An absolute ``tensor_folder`` already persists at its own location, so
        nothing extra is copied.
        """
        opts = self._opts(ctx)
        if opts.save_tensors and not Path(opts.tensor_folder).is_absolute():
            return (opts.tensor_folder,)
        return ()
