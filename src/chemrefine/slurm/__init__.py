"""SLURM mechanics: script generation, ``sbatch`` submission, ``squeue`` polling.

Engine-agnostic throughout. It knows how to assemble a SLURM script from a
cluster-specific header template plus an engine-provided
:class:`~chemrefine.state.RunBlock` (the bash that actually invokes the calculation), how
to submit that script, and how to poll for completion. PAL-budget bookkeeping lives in
:mod:`chemrefine.throttle`; this package only deals with the SLURM commands themselves.

Two halves, along the line that matters — whether a function talks to anything outside
this process:

* :mod:`~chemrefine.slurm.script` turns values into text and returns it. Pure, so its
  tests need no cluster and no patching, and so the shell-safety invariant can enumerate
  every value that reaches generated bash from its signatures.
* :mod:`~chemrefine.slurm.dispatch` runs those scripts — ``sbatch``, ``squeue``, and the
  local-process fallback that lets the same pipeline run on a laptop. Everything that
  shells out is here, which is why the tests patch ``subprocess.run``.

The names are re-exported here, so ``from chemrefine import slurm`` then ``slurm.submit``
keeps working; a caller has no reason to know which half something came from.
"""

from __future__ import annotations

from chemrefine.slurm.dispatch import (
    QueueState,
    dispatch_locally,
    finished_jobs,
    header_name_for_device,
    is_finished,
    poll_jobs,
    resolve_gpu_budget,
    sbatch_available,
    submit,
    submit_array,
    terminate_local_jobs,
    wait_for_jobs,
)
from chemrefine.slurm.script import (
    build_array_script,
    build_script,
    write_array_manifests,
)

__all__ = [
    "QueueState",
    "build_array_script",
    "build_script",
    "dispatch_locally",
    "finished_jobs",
    "header_name_for_device",
    "is_finished",
    "poll_jobs",
    "resolve_gpu_budget",
    "sbatch_available",
    "submit",
    "submit_array",
    "terminate_local_jobs",
    "wait_for_jobs",
    "write_array_manifests",
]
