"""Shared, engine-neutral out-of-process backend server.

Hosts one registered :class:`~chemrefine.engines._backend_server.base.ComputeBackend`
per process and serves its computed properties (energy + gradient today) over HTTP,
so a heavy backend (MLIP, PySCF, ...) loads once and answers many requests. Reusable
by any driver that needs out-of-process gradients; the ORCA-specific driver lives in
:mod:`chemrefine.engines.orca.extopt`.

Where it is used, and where it deliberately is not
--------------------------------------------------

The ExtOpt engines (``mlip-extopt`` / ``pyscf-extopt``) are the only users: there, one
*optimization* makes tens to hundreds of gradient calls against one geometry's model, so a
resident server per job pays for itself immediately. Each per-structure job launches its own
server inside its own script (:mod:`chemrefine.engines.orca.extopt.run_block`) — port from
the kernel, URL through a sidecar file in ``$WORK_DIR``, teardown in the script's one EXIT
handler — so jobs stay independent units the scheduler, the ledger and the recovery
commands can reason about.

The direct engines (``mlip`` / ``pyscf``) stay **serverless** on purpose. Their unit of work
is one structure = one process, which is what makes per-structure throttling, retries, the
failure ledger and ``rerun-errors`` compose; a shared resident server would tie N structures'
fates to one process and put a stateful dependency between jobs the scheduler treats as
independent. The cost is one backend load per structure — negligible for the small models
the gate runs, real for a foundation model (a UMA load is ~40 s of the ~45 s job). If batch
labeling with heavy models becomes routine, the trade to revisit is a *step-scoped* server
job that the structure jobs then talk to — through this same module — not an in-process
batch loop, which would forfeit the per-structure machinery.
"""
