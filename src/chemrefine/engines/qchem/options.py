"""Pydantic validator for Q-Chem engine YAML options.

Mirrors the other engines' ``options.py`` modules: one validated model, the single reader
of the step's knobs. ``cores`` and ``device`` come from
:class:`~chemrefine.engines._options.EngineOptions`; ``cores`` is Q-Chem's ``-nt`` thread
count, because Q-Chem takes its parallelism on the command line and not in the input file —
the YAML is where the number natively lives, where ORCA's lives in the template's ``%pal``.
"""

from __future__ import annotations

from pydantic import Field

from chemrefine.engines._options import EngineOptions


class QchemOptions(EngineOptions):
    """Validated knobs for the Q-Chem backend.

    ``cores`` (inherited) is the OpenMP thread count rendered as ``-nt``; the two knobs
    declared here are the MPI opt-in and the scratch save.
    """

    nprocs: int | None = Field(None, ge=1)
    """MPI ranks — **opt-in**, and deliberately so.

    Set, the job runs ``qchem -mpi -np {nprocs}`` (plus ``-nt {cores}`` when ``cores`` > 1)
    and is laid out as ``--ntasks={nprocs} --cpus-per-task={cores}``, charging
    ``nprocs x cores`` against ``max_cores``. Unset, the job is pure OpenMP — the solid
    path: Q-Chem's MPI covers only some methods, so qqchem gates it behind a per-version
    ``mpi_support`` flag and this stays a deliberate choice, never a default. The MPI
    install facts (``QCRSH``/``QCMPI`` exports, MPI module loads) belong in the SLURM
    header beside the other machine environment."""

    save: bool = False
    """Copy the key scratch files (MO coefficients — the ``.gbw``-analogue) back.

    The job always runs with a savename, so Q-Chem keeps those files at
    ``$QCSCRATCH/<structure stem>`` instead of deleting them on exit; this knob decides
    whether that directory comes home to the structure's own output dir (qqchem's
    ``--save``). Off by default — MO files are sizeable, and most steps never restart."""
