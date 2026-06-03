"""Shared, engine-neutral out-of-process backend server.

Hosts one registered :class:`~chemrefine.engines._backend_server.base.ComputeBackend`
per process and serves its computed properties (energy + gradient today) over HTTP,
so a heavy backend (MLFF, PySCF, ...) loads once and answers many requests. Reusable
by any driver that needs out-of-process gradients; the ORCA-specific driver lives in
:mod:`chemrefine.engines.orca.extopt`.
"""
