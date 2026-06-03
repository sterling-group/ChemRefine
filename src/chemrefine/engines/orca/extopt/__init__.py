"""ORCA external-optimizer (``!ExtOpt`` / ``ProgExt``) driver.

ORCA optimises while an external program supplies energy + gradient at each step.
This package owns the ORCA-specific pieces: the engine, the SLURM run-block that
starts the backend server alongside ORCA, the ``.extinp.tmp``/``.engrad`` ProgExt
file protocol, and the bridge ORCA spawns to relay each call to the backend server
(:mod:`chemrefine.engines._backend_server`).
"""
