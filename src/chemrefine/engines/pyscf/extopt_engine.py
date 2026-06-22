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
from chemrefine.engines.pyscf.extopt_calc import PyscfExtOptCalculator
from chemrefine.engines.pyscf.options import PyscfOptions
from chemrefine.state import StepContext


@register("pyscf-extopt")
class PyscfExtOptEngine(ExtOptOrcaEngine):
    """ORCA optimisation backed by a PySCF gradient server."""

    name: ClassVar[str] = "pyscf-extopt"
    backend: ClassVar[str] = "pyscf"
    wrapper_filename: ClassVar[str] = "pyscf_extopt.sh"
    options_cls: ClassVar[type[EngineOptions]] = PyscfOptions
    calculator_cls: ClassVar[type[ComputeBackend]] = PyscfExtOptCalculator

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
