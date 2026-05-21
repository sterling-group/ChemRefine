"""ChemRefine: automated computational-chemistry workflow manager."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ChemRefine")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
