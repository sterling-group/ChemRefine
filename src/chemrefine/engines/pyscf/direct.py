"""Direct in-process PySCF engine — no ORCA.

TODO: implement the full in-process flow once a real PySCF tutorial is
captured. The skeleton mirrors :class:`MlffDirectEngine` — write a
per-structure settings JSON, evaluate PySCF inline, read the energy
back. The actual gradient/energy computation should call the same
helper used by the server (see :mod:`chemrefine.engines.pyscf.server`)
to keep the two backends consistent.
"""

from __future__ import annotations

import logging

from chemrefine.engines.base import register
from chemrefine.state import JobBatch, StepContext, StepInputs, StepResults

logger = logging.getLogger(__name__)


@register("pyscf-direct")
class PyscfDirectEngine:
    """Placeholder direct-mode PySCF engine."""

    name = "pyscf-direct"
    supports_nms = False

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Placeholder — see module docstring."""
        raise NotImplementedError(
            "pyscf-direct not yet ported — see TODO in engines/pyscf/direct.py"
        )

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Placeholder — see module docstring."""
        raise NotImplementedError(
            "pyscf-direct not yet ported — see TODO in engines/pyscf/direct.py"
        )

    def wait(self, batch: JobBatch) -> None:
        """No-op — submit is in-process when this engine ships."""
        return None

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Placeholder — see module docstring."""
        raise NotImplementedError(
            "pyscf-direct not yet ported — see TODO in engines/pyscf/direct.py"
        )

    def normal_mode_sample(self, results: StepResults, ctx: StepContext) -> StepResults:
        """Not supported."""
        raise NotImplementedError("pyscf-direct does not support NMS")
