"""Built-in MLIP backends — importing this package registers them all.

Each module self-registers its builder(s) via ``@register_backend`` from
:mod:`chemrefine.engines.mlip.calculator`. Add a new MLIP by dropping a
module here and listing it on the import line below; the heavy third-party
import stays inside the builder so this package imports cheaply even when an
optional dependency is missing.
"""

from chemrefine.engines.mlip.backends import chgnet, fairchem, mace, orb, sevenn

__all__ = ["chgnet", "fairchem", "mace", "orb", "sevenn"]
