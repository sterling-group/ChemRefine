"""ChemRefine: automated computational-chemistry workflow manager."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ChemRefine")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

USER_AGENT = f"ChemRefine/{__version__} (+https://github.com/sterling-group/ChemRefine)"
"""The ``User-Agent`` ChemRefine sends on outbound HTTP requests.

Standard product-token form — name, version, and a URL identifying the project. It lives
beside :data:`__version__` because it is derived from it and is the same kind of fact: who
this package is when it speaks to something outside itself.

Applied by the callers that build requests with :mod:`urllib`, which otherwise sends the
interpreter's default token; clients that supply a product token of their own are
unaffected. Services commonly throttle or refuse requests that do not identify the caller,
and some APIs ask callers to identify themselves as a condition of use.
"""
