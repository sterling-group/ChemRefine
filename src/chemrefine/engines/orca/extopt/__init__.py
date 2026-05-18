"""ExtOpt file-protocol helpers used by MLFF/PySCF when ORCA drives the optimizer.

The protocol is: ORCA writes ``{base}.extinp.tmp`` containing the
current geometry, an external server reads it, writes energy +
gradients to ``{base}.engrad``, and ORCA picks the engrad up to take
its next optimisation step.

TODO: port the v3 :file:`utils_extopt.py` helpers (``read_extinp``,
``write_engrad``, plus the wrapper-script generator that becomes the
``ProgExt`` target in the ORCA input). These functions are
straightforward but rely on the exact byte-for-byte layout that ORCA
expects; needs a real run to verify.
"""

from chemrefine.engines.orca.extopt.protocol import (  # noqa: F401
    read_extinp,
    write_engrad,
    write_wrapper_script,
)
