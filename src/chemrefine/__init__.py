"""ChemRefine: automated computational-chemistry workflow manager."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ChemRefine")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

USER_AGENT = f"ChemRefine/{__version__} (+https://github.com/sterling-group/ChemRefine)"
"""How ChemRefine identifies itself to third-party HTTP services.

Beside ``__version__`` because it *is* the package's identity, one tier up. Every HTTP
client we depend on sends a product token of its own — the OpenAI SDK, httpx, requests —
and :mod:`urllib`, which the two hand-rolled outbound calls use, is the one that does not:
its default announces ``Python-urllib/3.x``, which CDNs and WAFs drop on sight. Groq's edge
answers that token with a flat **403**, so ``chemrefine agent --check`` reported a perfectly
good API key as "authentication rejected" — and the GUI's chat panel, whose Send button is
gated on that preflight, could not be used with Groq at all while the chat itself worked
fine through the SDK. Sending a real name is the fix, and it is also what NCBI's E-utilities
usage policy asks of anything calling PubChem.
"""
