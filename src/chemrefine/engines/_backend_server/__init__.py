"""Shared, engine-neutral out-of-process backend server.

Hosts one registered :class:`~chemrefine.engines._backend_server.base.ComputeBackend`
per process and serves its computed properties (energy + gradient today) over HTTP,
so a heavy backend loads once and answers many requests. Reusable by any driver that
needs out-of-process gradients; each driving engine's protocol — how its program calls out,
and how its jobs start and stop the server — lives in that engine's own package.

Where it is used, and where it deliberately is not
--------------------------------------------------

Its users are the engines that declare a served backend
(:class:`~chemrefine.engines._backend_server.base.ExtOptServed`): there, one *optimization*
makes tens to hundreds of gradient calls against one geometry's model, so a resident server
per job pays for itself immediately. Each per-structure job launches its own server inside
its own script — port from the kernel, URL through a sidecar file in ``$WORK_DIR``, teardown
in the script's one EXIT handler — so jobs stay independent units the scheduler, the ledger
and the recovery commands can reason about.

The direct script engines stay **serverless** on purpose. Their unit of work
is one structure = one process, which is what makes per-structure throttling, retries, the
failure ledger and ``rerun-errors`` compose; a shared resident server would tie N structures'
fates to one process and put a stateful dependency between jobs the scheduler treats as
independent. The cost is one backend load per structure — negligible for the small models
the gate runs, real for a foundation model (a load can be ~40 s of a ~45 s job). If batch
labeling with heavy models becomes routine, the trade to revisit is a *step-scoped* server
job that the structure jobs then talk to — through this same module — not an in-process
batch loop, which would forfeit the per-structure machinery.
"""
