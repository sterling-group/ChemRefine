"""Engine plugin subsystem.

Importing this package self-registers every bundled engine: each plugin package listed
below registers itself via :func:`~chemrefine.engines.api.register` at import time, so
``import chemrefine.engines`` populates the registry (:data:`chemrefine.engines.api.ENGINES`).
The pipeline then looks engines up by name with :func:`get_engine` and never imports a
concrete engine module directly.

The contract lives in :mod:`chemrefine.engines.api`; the reusable building blocks an engine
is built from are the underscored modules (:mod:`._job`, :mod:`._execution`, :mod:`._script`,
:mod:`._backend_server`). Add a new engine by creating a bare-named ``engines/<name>/``
package and adding it to the import below — nothing else here changes.
"""

from chemrefine.engines import _fake, mlip, orca, pyscf
from chemrefine.engines.api import get_engine, register

__all__ = ["_fake", "get_engine", "mlip", "orca", "pyscf", "register"]
