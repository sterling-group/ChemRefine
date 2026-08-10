"""Every role-marked docstring cross-reference must resolve to a real object.

The prose in ``src/`` is load-bearing: rationale lives in docstrings, and they point at
each other with Sphinx roles (``:func:`chemrefine.cache.fingerprint```). Nothing checks
those targets — mkdocs never resolves Sphinx roles, so a rename leaves the pointer
dangling silently and the reader chasing a name that no longer exists. Six had already
rotted when this test landed: ``chemrefine.config._normalize_legacy`` three times (the
function is ``config_legacy.normalize``), ``chemrefine.slurm._run_body_lines`` twice
(stranded by the slurm split into a package), and a bare ``_normalize_legacy`` once —
which is the evidence prose needs the same mechanical check the code gets.

Absolute references only (targets starting ``chemrefine.``): a bare local name has no
single right module to resolve against, and the absolute form is what the codebase uses
for anything a reader would navigate to.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import chemrefine

_PACKAGE_ROOT = Path(chemrefine.__file__).parent

_ROLE_RE = re.compile(
    # The dot is required so a bare local name that merely starts with "chemrefine"
    # (`:func:`chemrefine_home``) is not mistaken for an absolute path.
    r":(?:func|class|meth|mod|attr|data|exc):`~?(chemrefine(?:\.[A-Za-z0-9_.]+)?)`"
)


def _iter_refs() -> list[tuple[str, str]]:
    """Every ``(location, target)`` role reference in the package's source text.

    Scanned as text rather than via ``__doc__`` because the roles appear in comments and
    attribute docstrings too, and those never reach a runtime ``__doc__``.
    """
    refs: list[tuple[str, str]] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for m in _ROLE_RE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            refs.append((f"{path.relative_to(_PACKAGE_ROOT.parent)}:{line}", m.group(1)))
    return refs


def _resolves(target: str) -> bool:
    """Whether ``target`` names an importable module, or an attribute chain off one.

    The attribute walk needs three fallbacks beyond ``getattr``, all for the same reason:
    a documented field is not always a class attribute at runtime. A dataclass field
    without a default lives only in ``__dataclass_fields__`` (``StepContext.template``),
    a pydantic field only in ``model_fields`` (``Config.max_gpus``), and a TypedDict key
    only in ``__annotations__``. Each is accepted as a leaf only — nothing can be chained
    off a field that does not exist as a value.
    """
    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        rest = parts[i:]
        break
    else:
        return False
    for j, name in enumerate(rest):
        if hasattr(obj, name):
            obj = getattr(obj, name)
            continue
        is_leaf = j == len(rest) - 1
        return is_leaf and any(
            name in getattr(obj, table, {})
            for table in ("__dataclass_fields__", "model_fields", "__annotations__")
        )
    return True


def test_every_absolute_docstring_reference_resolves():
    """A dangling reference fails here, named with the file and line that carries it."""
    refs = _iter_refs()
    assert refs, "no role-marked references found — the scanner itself has broken"
    dangling = [f"{where} -> {target}" for where, target in refs if not _resolves(target)]
    assert dangling == [], (
        "these docstring cross-references point at nothing; rename the target in the "
        "prose or restore the object:\n" + "\n".join(dangling)
    )
