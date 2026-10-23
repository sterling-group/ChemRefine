"""xtb engine — importing this module registers ``xtb``.

One engine ships in this package: :class:`~chemrefine.engines.xtb.engine.XtbEngine`, a
per-structure :class:`~chemrefine.engines._job.JobEngine` that runs the xtb binary on each
structure's XYZ geometry with the step's xcontrol template (:mod:`.input`) and a command
line built from the step's operation and options (:mod:`.command`), threaded, and parses
its stdout and the files beside it through :mod:`.output` — a coordinator over one reader
per file (stdout, ``gradient``, ``xtbopt.xyz``, ``g98.out``).
"""

from chemrefine.engines.xtb.engine import XtbEngine

__all__ = ["XtbEngine"]
