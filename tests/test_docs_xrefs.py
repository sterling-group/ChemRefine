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

An absolute target (``chemrefine.…``) must import and resolve. A bare one has no single
right module to resolve against, so it is held to where a reader would look for it: a name
the package defines, a name its own file defines or imports, or a builtin. A bare
``:mod:`` names a module of the package (relative to the file's package, or from its root)
or one the package depends on: the standard library or a declared dependency, by name, so
the answer is the same in every environment. A dotted bare target is walked from the
module that defines its first name at module level; one rooted outside the package
(``ase.Atoms``, ``sys.exit``) is left to the library that owns it.
"""

from __future__ import annotations

import ast
import builtins
import functools
import importlib
import importlib.util
import itertools
import re
import sys
import tomllib
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import chemrefine

_PACKAGE_ROOT = Path(chemrefine.__file__).parent

#: The tree under test, located from this file rather than from the imported package —
#: the idiom ``test_mutation_gate`` and ``test_docs_commit_types`` already use. Derived
#: from the package it was two directories above an *installed* ``chemrefine``, which is
#: a site-packages ancestor holding none of this, so every guard below stood down in the
#: one run that most needed them: the suite executed from the unpacked sdist.
_REPO_ROOT = Path(__file__).resolve().parent.parent

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
            refs.append((_where(path, text, m.start()), m.group(1)))
    return refs


def _where(path: Path, text: str, offset: int) -> str:
    """``file:line`` for a match, the file named relative to the checkout when there is one."""
    try:
        name = path.relative_to(_REPO_ROOT)
    except ValueError:  # an installed package, outside any checkout
        name = path.relative_to(_PACKAGE_ROOT.parent)
    return f"{name}:{text.count(chr(10), 0, offset) + 1}"


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


_BARE_ROLE_RE = re.compile(r":(func|class|meth|mod|attr|data|exc):`~?(\.?[A-Za-z_][\w.]*)`")


def _bound(target: ast.expr) -> Iterator[str]:
    """The names an assignment target binds: a name, ``self.x`` or ``cls.x``, or an unpacking."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Attribute):
        if isinstance(target.value, ast.Name) and target.value.id in ("self", "cls"):
            yield target.attr
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _bound(element)
    elif isinstance(target, ast.Starred):
        yield from _bound(target.value)


def _definitions(nodes: Iterable[ast.AST]) -> Iterator[str]:
    """The names ``nodes`` bind by definition: defs, classes, assignment targets."""
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                yield from _bound(target)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            yield from _bound(node.target)


def _defined(tree: ast.AST) -> Iterator[str]:
    """Every name a module defines, nested ones included.

    A method is a ``def`` inside a class, and a pydantic or dataclass field an annotated
    assignment in one; a role names either by its bare name.
    """
    return _definitions(ast.walk(tree))


def _imported(tree: ast.AST) -> Iterator[str]:
    """Every name a module binds by import."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.asname or alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                yield alias.asname or alias.name


def _module_of(path: Path) -> str:
    """The dotted module name of a file under the package."""
    parts = path.relative_to(_PACKAGE_ROOT.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


@functools.cache
def _package_names() -> tuple[frozenset[str], dict[str, frozenset[str]]]:
    """Every name the package defines, and its module-level names with their modules."""
    every: set[str] = set()
    module_level: defaultdict[str, set[str]] = defaultdict(set)
    for path in _PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        every.update(_defined(tree))
        for name in _definitions(tree.body):
            module_level[name].add(_module_of(path))
    return frozenset(every), {name: frozenset(mods) for name, mods in module_level.items()}


@functools.cache
def _dependencies() -> frozenset[str]:
    """The distributions ``pyproject.toml`` declares, extras included, by normalised name."""
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"].values()
    return frozenset(
        re.split(r"[^\w.-]", spec, maxsplit=1)[0].lower().replace("_", "-")
        for spec in itertools.chain(project["dependencies"], *extras)
    )


def _is_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:  # a dotted name whose parent is no package
        return False


def _names_a_module(target: str, path: Path) -> bool:
    """A module beside ``path`` or of the package from its root, or a stdlib or declared one."""
    if path.is_relative_to(_PACKAGE_ROOT):
        if target.startswith("."):
            package = _module_of(path.parent / "__init__.py")
            return _is_module(importlib.util.resolve_name(target, package))
    else:  # tests/ or scripts/, whose modules import each other by their bare names
        beside = path.parent.joinpath(*target.split("."))
        if beside.with_suffix(".py").is_file() or (beside / "__init__.py").is_file():
            return True
    if _importable_as(target, path):
        return True
    top = target.split(".")[0]
    return top in sys.stdlib_module_names or _declared(top)


def _declared(top: str) -> bool:
    """Whether ``top`` is the import name of a distribution ``pyproject.toml`` declares."""
    dist = top.lower().replace("_", "-")
    return any(name == dist or name.startswith(f"{dist}-") for name in _dependencies())


def _importable_as(name: str, path: Path) -> list[str]:
    """The importable modules a bare ``name`` can mean in ``path``.

    Beside it in the package, of the package from its root, or top-level — the one order
    both :func:`_names_a_module` and a module-qualified reference look in, kept to those
    that import, which is what an attribute chain off the name can be checked against.
    """
    candidates = [f"chemrefine.{name}", name]
    if path.is_relative_to(_PACKAGE_ROOT):
        candidates.insert(0, f"{_module_of(path.parent / '__init__.py')}.{name}")
    return [module for module in candidates if _is_module(module)]


def _bare_resolves(path: Path, role: str, target: str, own: frozenset[str]) -> bool:
    """Whether a bare ``target`` in ``path`` names what the module docstring says it may.

    A dotted target starts with a module-level name of the package, with a module the file
    can mean by that name (``lifecycle.finalize``), checked on that module, or with a name
    the file itself binds or a builtin (``Agent.run``, ``str.join``). Anything else names
    nothing — a misspelled or renamed module is exactly that. What the scanner cannot judge
    is a relative reference, and one into a declared library this environment lacks.
    """
    if role == "mod":
        return _names_a_module(target, path)
    every, module_level = _package_names()
    first, _, rest = target.partition(".")
    if not rest:
        return first in every or first in own or hasattr(builtins, first)
    modules = module_level.get(first)
    if modules is not None:
        return any(_resolves(f"{module}.{target}") for module in modules)
    if not first.isidentifier() or (_declared(first) and not _is_module(first)):
        return True
    owners = _importable_as(first, path)
    if not owners:
        return first in own or hasattr(builtins, first)
    return any(_resolves(f"{owner}.{rest}") for owner in owners)


def test_every_bare_docstring_reference_names_something_that_exists():
    """A bare role rots exactly as an absolute one does: a rename leaves it dangling."""
    dangling, checked = [], 0
    for path in _scanned_paths():
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:  # a step template under tests/data: $PLACEHOLDERs, not Python
            own: frozenset[str] = frozenset()
        else:
            own = frozenset(_defined(tree)) | frozenset(_imported(tree))
        for m in _BARE_ROLE_RE.finditer(text):
            role, target = m.groups()
            if target == "chemrefine" or target.startswith("chemrefine."):
                continue  # the absolute check's
            checked += 1
            if not _bare_resolves(path, role, target, own):
                dangling.append(f"{_where(path, text, m.start())} -> {target}")
    assert checked, "no bare role references found — the scanner itself has broken"
    assert dangling == [], (
        "these bare docstring cross-references point at nothing; correct the name or "
        "write it absolute (chemrefine.…):\n" + "\n".join(dangling)
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

    A missing ``docs/`` fails rather than skips. Every tree this suite runs in ships one —
    the sdist by its include list, the mutation gate by ``_INPUTS`` — and the wheel ships
    no tests at all, so there is no run where its absence is normal. Skipping there would
    be silence in the exact shape of the regression: ``docs/`` and ``examples/`` have
    already fallen out of the sdist once, with every gate green.
    """
    assert _DOCS.is_dir(), (
        f"no docs/ beside {_REPO_ROOT} — the sdist and the mutation gate's scratch copy "
        "both carry it, so its absence is a packaging regression, which is the thing this "
        "guard is here to notice rather than stand down for"
    )

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
