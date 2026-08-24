"""Every role-marked docstring cross-reference must resolve to a real object.

The prose is load-bearing: rationale lives in docstrings, and they point at each other
with Sphinx roles (``:func:`chemrefine.cache.structure_digest```). Nothing checks those
targets — mkdocs never resolves Sphinx roles, so a rename leaves the pointer dangling
silently and the reader chasing a name that no longer exists. Six had already rotted when
this test landed: ``chemrefine.config._normalize_legacy`` three times (the function is
``config_legacy.normalize``), ``chemrefine.slurm._run_body_lines`` twice (stranded by the
slurm split into a package), and a bare ``_normalize_legacy`` once — which is the evidence
prose needs the same mechanical check the code gets.

``tests/`` and ``scripts/`` are scanned on the same footing as ``src/``, because the
rationale a maintainer actually reads is as often in a test docstring as in the module it
covers. Scanning only ``src/`` let ``chemrefine.cache.parents_digest`` rot in a perf test
— and in the sentence above, which named the same removed function — through the release
that deleted it, because the one test that called it is deselected by default and nothing
read its prose.

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
_REPO_ROOT = _PACKAGE_ROOT.parent.parent

#: Directories scanned beside the package itself, when the checkout is there to hold them.
#: Both are repository artifacts: an installed package has neither, and the package's own
#: prose is the part that must be checked everywhere, so their absence narrows this test
#: rather than breaking it.
_EXTRA_ROOTS = ("tests", "scripts")

_ROLE_RE = re.compile(
    # The dot is required so a bare local name that merely starts with "chemrefine"
    # (`:func:`chemrefine_home``) is not mistaken for an absolute path.
    r":(?:func|class|meth|mod|attr|data|exc):`~?(chemrefine(?:\.[A-Za-z0-9_.]+)?)`"
)


def _scanned_paths() -> list[Path]:
    """Every ``.py`` file whose prose this checks — the package, plus the checkout's own."""
    paths = list(_PACKAGE_ROOT.rglob("*.py"))
    for name in _EXTRA_ROOTS:
        root = _REPO_ROOT / name
        if root.is_dir():
            paths.extend(root.rglob("*.py"))
    return sorted(set(paths))


def _iter_refs() -> list[tuple[str, str]]:
    """Every ``(location, target)`` role reference in the scanned source text.

    Scanned as text rather than via ``__doc__`` because the roles appear in comments and
    attribute docstrings too, and those never reach a runtime ``__doc__``.
    """
    refs: list[tuple[str, str]] = []
    for path in _scanned_paths():
        text = path.read_text(encoding="utf-8")
        for m in _ROLE_RE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            try:
                where = path.relative_to(_REPO_ROOT)
            except ValueError:  # an installed package, outside any checkout
                where = path.relative_to(_PACKAGE_ROOT.parent)
            refs.append((f"{where}:{line}", m.group(1)))
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


_DOCS = _REPO_ROOT / "docs"

#: A ``docs/…md`` path named in prose. Same claim as a role reference — "go and read
#: this" — and the same failure when it rots, but pointing at a file rather than an object.
_DOC_PATH_RE = re.compile(r"docs/[A-Za-z0-9_./-]+\.md")

#: Where such a path can appear. `docs/` itself is excluded: mkdocs resolves *its* links,
#: and `mkdocs build --strict` already fails on a broken one.
_PROSE_FILES = ("src/chemrefine/**/*.py", "CONTRIBUTING.md", "README.md")


def test_every_documentation_path_named_in_prose_exists():
    """A docs page named from the code must be a page that exists.

    Nothing watched this, and the re-organisation proved why: five references in ``src/``
    and ``CONTRIBUTING.md`` pointed at pages that had moved — one of them
    (``config_legacy``'s "see docs/migrating-v1-to-v2.md") inside an error message a user
    reads when their v1 config is rejected. ``--strict`` cannot see them because they are
    not in ``docs/``, so they rot silently in exactly the way a link inside ``docs/``
    no longer can.
    """
    if not _DOCS.is_dir():
        import pytest

        pytest.skip("docs/ is a repository artifact and does not ship in the sdist")

    found, missing = 0, []
    for pattern in _PROSE_FILES:
        for path in sorted(_REPO_ROOT.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            for m in _DOC_PATH_RE.finditer(text):
                found += 1
                if not (_REPO_ROOT / m.group(0)).is_file():
                    line = text.count("\n", 0, m.start()) + 1
                    missing.append(f"{path.relative_to(_REPO_ROOT)}:{line} -> {m.group(0)}")
    assert found, "no docs/ paths found in prose — the scanner itself has broken"
    assert missing == [], (
        "these prose references name a documentation page that does not exist:\n"
        + "\n".join(missing)
    )
