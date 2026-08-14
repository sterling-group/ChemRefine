"""Q-Chem engine — importing this module registers ``qchem``.

One engine ships in this package: :class:`~chemrefine.engines.qchem.engine.QchemEngine`,
a per-structure :class:`~chemrefine.engines._job.JobEngine` that renders the user's
``step{N}.in`` template per structure (:mod:`.input`), runs ``qchem`` with CLI-side
parallelism and the ``QC``/``QCAUX``/``QCSCRATCH`` environment (threads by default, MPI
opt-in), and parses the ``.out`` (:mod:`.output` — energy/geometry placeholder plus the
real NMS frequency feed). :mod:`.inspect` reads the template's ``JOBTYPE`` and
``mem_total`` facts for NMS and the SLURM memory request.
"""

from chemrefine.engines.qchem.engine import QchemEngine

__all__ = ["QchemEngine"]
