"""The leading underscore has to mean something: nothing crosses a module for one.

:func:`chemrefine.config.reject_shell_unsafe` states the rule this enforces, in its own
docstring — *"Public, not underscored, because the engine option models import it"*. A
helper another module needs is public and says who needs it; a helper with a leading
underscore is one module's business and can be renamed, re-signatured or deleted without
reading the rest of the package.

Stated once and applied once, the rule quietly stopped holding: four helpers were reached
across a module boundary anyway, so the underscore no longer predicted whether a change
was local, and a maintainer grepping a private name found callers where the naming said
there could be none.

Checked over ``src/`` only. Tests reach into internals on purpose — that is what a unit
test is — and holding them to this would say the opposite of what the rule is for.
"""

from __future__ import annotations

import ast
from pathlib import Path

import chemrefine

_PACKAGE_ROOT = Path(chemrefine.__file__).parent

#: Crossings that have been argued for. Empty on purpose: the rule is the default, so an
#: exception has to arrive as a diff that says why, next to the name it exempts.
_ALLOWED: frozenset[tuple[str, str]] = frozenset()


def _is_submodule(dotted: str, name: str) -> bool:
    """Whether ``from <dotted> import <name>`` names a *module*, not a name inside one.

    An underscore on a module means something else here, and ``engines/__init__.py`` says
    so: the underscored modules (``_job``, ``_execution``, ``_script``, ``_provision``,
    ``_backend_server``) are the reusable building blocks an engine is assembled from, and
    the underscore is what keeps them out of plugin auto-discovery — bare-named packages
    are plugins. Importing one is the documented way to build an engine, not a boundary
    being crossed.
    """
    parent = _PACKAGE_ROOT.parent.joinpath(*dotted.split("."))
    return (parent / f"{name}.py").is_file() or (parent / name / "__init__.py").is_file()


def _private_names(tree: ast.AST) -> set[str]:
    """Every underscored name a module defines — functions, classes, methods, constants.

    Methods count because the rule is about the name, not its nesting: ``options_cls.
    _accepted_names()`` reached across a module exactly as a bare function would have.
    Dunders are excluded — those are the data model, not a privacy claim.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            found.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found.add(node.target.id)
    return {n for n in found if n.startswith("_") and not n.startswith("__")}


def _crossings(path: Path, tree: ast.AST, owners: dict[str, set[str]]) -> list[str]:
    """Underscored names this module reaches for that only another module defines.

    ``self._x`` / ``cls._x`` are a class talking to itself and never counted. A name this
    module also defines is its own. Only names some *chemrefine* module owns are
    considered, so ``namedtuple._asdict`` and numpy internals cannot register.
    """
    module = path.stem
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("chemrefine"):
            out += [
                f"imports {a.name} from {node.module}"
                for a in node.names
                if a.name.startswith("_")
                and not a.name.startswith("__")
                and not _is_submodule(node.module or "", a.name)
                and module not in owners.get(a.name, set())
            ]
        elif isinstance(node, ast.Attribute):
            name = node.attr
            if not name.startswith("_") or name.startswith("__"):
                continue
            if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
                continue
            holders = owners.get(name, set())
            if holders and module not in holders:
                out.append(f"reads {ast.unparse(node)} (defined in {sorted(holders)})")
    return out


def test_no_module_reaches_for_another_modules_private_name():
    """A helper another module needs is public and says who needs it.

    The four that broke this — ``cache._atomic_write``, ``config._resolve_relative_paths``,
    ``EngineOptions._accepted_names``, ``run_block._build_extopt_run_block`` — were each
    a genuinely shared concern that had simply never been promoted. Nothing about them
    was wrong except the name, and the name is what a maintainer trusts when deciding
    whether a change is local.
    """
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(_PACKAGE_ROOT.rglob("*.py"))
    }
    owners: dict[str, set[str]] = {}
    for path, tree in trees.items():
        for name in _private_names(tree):
            owners.setdefault(name, set()).add(path.stem)

    offences = [
        f"{path.relative_to(_PACKAGE_ROOT.parent)}: {detail}"
        for path, tree in trees.items()
        for detail in _crossings(path, tree, owners)
        if (path.stem, detail.split()[1]) not in _ALLOWED
    ]
    assert not offences, (
        "a module reached past another's leading underscore:\n  "
        + "\n  ".join(offences)
        + "\n\nMake the helper public and say in its docstring who imports it and why — "
        "the form config.reject_shell_unsafe uses — or keep it private and stop crossing."
    )
