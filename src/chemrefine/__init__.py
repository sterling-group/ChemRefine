"""ChemRefine: automated computational-chemistry workflow manager."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ChemRefine")
except PackageNotFoundError:  # pragma: no cover - only triggers when run from a non-installed checkout
    __version__ = "0.0.0+unknown"
