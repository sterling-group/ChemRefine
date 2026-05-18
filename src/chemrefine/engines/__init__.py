"""Engine plugin layer.

Importing this package self-registers every bundled engine into
:data:`chemrefine.engines.base.ENGINES`. The orchestrator imports
:mod:`chemrefine.engines` once and then looks up engines by name from
that dict — it never imports a concrete engine module directly.

New engines join the registry by importing their package here (the
side-effect ``register`` decorator does the actual binding).
"""

from chemrefine.engines import (
    _fake,  # noqa: F401 - side-effect: registers "fake"
    mlff,  # noqa: F401 - side-effect: registers "mlff" and "mlff-direct"
    orca,  # noqa: F401 - side-effect: registers "orca"
)
