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

A boundary holds names too. Outside an engine's own package, core describes an engine by
its role, so adding, renaming or removing one never sends anybody through the core to fix
its prose: :func:`test_core_names_no_engine` holds the names, read off the registries. An
engine's vocabulary — its keywords, its file formats — is left to review.

A boundary holds writes too. Outside :mod:`chemrefine.io`, a call that creates, replaces,
moves or removes a path stands inside ``with tree_write(path):``, io's one net, so a write
a run's output tree refuses is the same named error whichever layer made it:
:func:`test_the_run_tree_is_written_only_through_io` holds that call by call. A function
whose write is no part of a run's tree — one the job runs, one writing a file a person
owns — is listed by name, with what it writes.

A boundary holds questions too. What is in a run's tree is asked through
:func:`chemrefine.io.absent` and :func:`chemrefine.io.tree_entries`, whose answers do not
depend on the Python version and never read a refusal as nothing being there:
:func:`test_a_run_tree_is_asked_what_it_holds_only_through_io` holds every ``Path``
existence question and ``glob`` in the package. A function that asks one of something that
is no run's tree read by its driver is listed by name, with what it asks about.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import NamedTuple

import pytest

import chemrefine
from chemrefine import io

_PACKAGE_ROOT = Path(chemrefine.__file__).parent

#: Crossings that have been argued for, as ``(package-relative path, name)`` pairs.
#: Empty on purpose: the rule is the default, so an exception has to arrive as a diff
#: that says why, next to the name it exempts.
_ALLOWED: frozenset[tuple[str, str]] = frozenset()


def _module_key(path: Path) -> str:
    """One module's identity for the ownership map: its package-relative path.

    The bare ``path.stem`` this replaces made every two same-named modules one owner —
    and this package has 15 ``__init__.py``, 7 ``engine.py`` and doubled ``base.py`` /
    ``registry.py`` / ``options.py`` / ``backend.py`` — so a private-name crossing
    between, say, two ``engine.py`` files read as self-access and the gate could not see
    that whole class of violation.
    """
    return path.relative_to(_PACKAGE_ROOT.parent).as_posix()


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
    module = _module_key(path)
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


def _exempted_name(detail: str) -> str:
    """The bare private name a crossing is about — the second half of an ``_ALLOWED`` key.

    The two crossing shapes spell the name differently: an import detail reads
    ``imports _x from …`` (the second token is the name) while an attribute detail reads
    ``reads cache._atomic_write (…)`` (the second token is the dotted expression). Keyed
    on the raw token, the first attribute exemption anyone wrote per ``_ALLOWED``'s own
    docstring — a ``(path, name)`` pair — silently failed to apply, and the gate went on
    failing over an entry that looked correct. Taking the last dotted segment gives both
    shapes the one spelling the docstring promises.
    """
    return detail.split()[1].rsplit(".", 1)[-1]


def test_an_exemption_key_matches_both_crossing_shapes():
    """One ``_ALLOWED`` spelling serves an import crossing and an attribute crossing alike.

    With no crossing in the tree the helper above otherwise runs on nothing, so this is
    what keeps its two-shape promise from rotting into exactly the mismatch it fixes.
    """
    assert _exempted_name("imports _atomic_write from chemrefine.cache") == "_atomic_write"
    assert _exempted_name("reads cache._atomic_write (defined in ['cache.py'])") == "_atomic_write"


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
            owners.setdefault(name, set()).add(_module_key(path))

    offences = [
        f"{path.relative_to(_PACKAGE_ROOT.parent)}: {detail}"
        for path, tree in trees.items()
        for detail in _crossings(path, tree, owners)
        if (_module_key(path), _exempted_name(detail)) not in _ALLOWED
    ]
    assert not offences, (
        "a module reached past another's leading underscore:\n  "
        + "\n  ".join(offences)
        + "\n\nMake the helper public and say in its docstring who imports it and why — "
        "the form config.reject_shell_unsafe uses — or keep it private and stop crossing."
    )


def _underscored_engine_imports(tree: ast.AST) -> list[str]:
    """Every import of an underscored ``chemrefine.engines`` module in ``tree``, dotted."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".")
            if parts[:2] == ["chemrefine", "engines"]:
                if len(parts) > 2 and parts[2].startswith("_"):
                    found.append(node.module or "")
                elif len(parts) == 2:
                    found += [
                        f"{node.module}.{a.name}" for a in node.names if a.name.startswith("_")
                    ]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if (
                    parts[:2] == ["chemrefine", "engines"]
                    and len(parts) > 2
                    and parts[2].startswith("_")
                ):
                    found.append(alias.name)
    return found


#: Core files that may name an engine, each with the reason it must.
_CORE_MAY_NAME: dict[str, str] = {
    "config_legacy.py": "translates v1.3.1's spellings into the engines that replace them, "
    "which is naming them",
    "data/agent_guide.md": "tells an agent which shipped engine serves which job, which it "
    "cannot do without their names",
}


def _engine_vocabulary(shipped: Mapping[str, type]) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """What names a shipped engine: its words, matched whole in any case, and its classes,
    matched exactly.

    The words are the registry keys, the plugin package names, the template labels, and each
    MLIP library's import name and trainer label; the classes are every plugin class in the
    MRO of a shipped engine, of its options model and of its gradient calculator. All of it
    is read off the registries, so an engine that joins is named by existing. Operation
    names and ``executables`` keys are a config's vocabulary rather than an engine's name,
    and are left out.
    """
    from chemrefine.engines.api import TemplateDriven, get_engine
    from chemrefine.engines.mlip import registry

    packages = {cls.__module__.split(".")[2] for cls in shipped.values()}
    words = set(shipped) | packages
    for name in shipped:
        engine = get_engine(name)
        if isinstance(engine, TemplateDriven):
            words.add(engine.label)
    for task in registry.registered_backends():
        spec = registry.backend_spec(task)
        words.add(spec.library.import_name)
        if spec.trainer is not None:
            words.add(spec.trainer.label)

    def plugin_classes(cls: type) -> set[str]:
        return {
            c.__name__
            for c in cls.__mro__
            if c.__module__.split(".")[:2] == ["chemrefine", "engines"]
            and c.__module__.split(".")[2:3] != []
            and c.__module__.split(".")[2] in packages
        }

    classes: set[str] = set()
    for cls in shipped.values():
        classes |= plugin_classes(cls)
        for attr in ("options_cls", "calculator_cls"):
            if (model := getattr(cls, attr, None)) is not None:
                classes |= plugin_classes(model)

    def alternation(names: set[str]) -> str:
        return "|".join(sorted(map(re.escape, names), key=len, reverse=True))

    return (
        re.compile(rf"\b(?:{alternation(words)})\b", re.IGNORECASE),
        re.compile(rf"\b(?:{alternation(classes)})\b"),
    )


def _core_files() -> Iterator[tuple[str, str]]:
    """Every UTF-8 text file of core, as ``(package-relative path, text)``.

    Core is everything under the package but an engine's own package (a bare-named one under
    ``engines/``) and the GUI's vendored libraries, which are other people's text.
    """
    for path in sorted(_PACKAGE_ROOT.rglob("*")):
        rel = path.relative_to(_PACKAGE_ROOT)
        if not path.is_file() or "__pycache__" in rel.parts:
            continue
        if rel.parts[0] == "engines" and len(rel.parts) > 2 and not rel.parts[1].startswith("_"):
            continue
        if rel.parts[:3] == ("gui", "static", "vendor"):
            continue
        try:
            yield rel.as_posix(), path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def test_core_names_no_engine(shipped_engines: dict[str, type]):
    """Outside an engine's own package, the code and its text describe an engine by its role.

    A name in core is a fact about one engine kept where that engine cannot keep it true:
    renamed, removed or joined by another, the core goes on saying the old thing, and a new
    engine has to read the core to learn which sentences are about someone else. Said by
    role — "an engine configured through its template", "an MPI program" — the sentence is
    true of every engine it is true of, and stays so.
    """
    words, classes = _engine_vocabulary(shipped_engines)
    offences = [
        f"{rel}:{lineno}: {match.group(0)}"
        for rel, text in _core_files()
        if rel not in _CORE_MAY_NAME
        for lineno, line in enumerate(text.splitlines(), 1)
        for pattern in (words, classes)
        for match in pattern.finditer(line)
    ]
    assert not offences, (
        "core names an engine:\n  "
        + "\n  ".join(offences)
        + "\n\nSay it by the engine's role, or move the fact into the engine's own package."
    )


def test_every_core_exemption_still_names_an_engine(shipped_engines: dict[str, type]):
    """An exemption that names nothing has outlived its reason; it goes."""
    words, classes = _engine_vocabulary(shipped_engines)
    texts = dict(_core_files())
    stale = [
        rel
        for rel in _CORE_MAY_NAME
        if not (words.search(texts[rel]) or classes.search(texts[rel]))
    ]
    assert stale == []


def test_the_flat_pipeline_imports_the_engine_subsystem_through_its_public_face():
    """Outside ``engines/``, an underscored engine module is nobody's to import.

    ``engines/api.py`` opens by calling itself "the one module the flat pipeline imports
    from the engine subsystem", and ``engines/__init__.py`` explains the underscore: the
    underscored packages are the building blocks an *engine* is assembled from. The first
    test in this file exempts imports of those modules because building an engine is
    exactly what they are for — which left the other half of the claim unchecked, and
    ``validate`` imported the GPU-demand helper from ``_job`` for as long as nothing looked.
    Held here: what the orchestrator needs is answered on the public face — the GPU demand,
    once reached for in ``_job``, is the option model's own ``gpu_demand`` — rather than
    reached for where it happens to live.
    """
    engines_root = _PACKAGE_ROOT / "engines"
    offences = [
        f"{path.relative_to(_PACKAGE_ROOT.parent)}: imports {dotted}"
        for path in sorted(_PACKAGE_ROOT.rglob("*.py"))
        if engines_root not in path.parents
        for dotted in _underscored_engine_imports(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not offences, (
        "the flat pipeline reached into an engine building block:\n  "
        + "\n  ".join(offences)
        + "\n\nRe-export what it needs from chemrefine.engines.api (or chemrefine.engines) "
        "and import it from there."
    )


# ---------------------------------------------------------------------------
# Writes — a run's output tree is written through io's net
# ---------------------------------------------------------------------------

#: io's net, by the dotted name an import binds — read off the function, so renaming it is
#: an error here. A raw write standing inside a ``with`` of it is a write of the run's tree.
_NET = f"{io.tree_write.__module__}.{io.tree_write.__qualname__}"

#: Functions that create, replace, link, move, copy or remove a path, by the dotted name an
#: import binds: the standard library's, numpy's savers and the other library writers this
#: package calls — one joins when the package first calls it — and io's own raw replace,
#: which a run's tree reaches through ``write_tree_file``.
_WRITER_FUNCTIONS = frozenset(
    {
        "ase.db.connect",
        "ase.io.write",
        "chemrefine.io.atomic_write",
        "logging.FileHandler",
        "numpy.save",
        "numpy.savetxt",
        "numpy.savez",
        "numpy.savez_compressed",
        "os.chmod",
        "os.link",
        "os.makedirs",
        "os.mkdir",
        "os.open",
        "os.remove",
        "os.removedirs",
        "os.rename",
        "os.renames",
        "os.replace",
        "os.rmdir",
        "os.symlink",
        "os.truncate",
        "os.unlink",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.copytree",
        "shutil.move",
        "shutil.rmtree",
        "tempfile.NamedTemporaryFile",
        "tempfile.TemporaryDirectory",
        "tempfile.mkdtemp",
        "tempfile.mkstemp",
        "torch.save",
    }
)

#: Functions that write a file only when handed the keyword that names one.
_WRITER_KEYWORDS = {"logging.basicConfig": "filename"}

#: Methods that create, change or remove what they are called on: a path, or a frame whose
#: ``to_csv`` names a file. Matched by name, since a receiver's type is not in the syntax.
_WRITER_METHODS = frozenset(
    {
        "chmod",
        "hardlink_to",
        "mkdir",
        "rmdir",
        "symlink_to",
        "to_csv",
        "touch",
        "unlink",
        "write_bytes",
        "write_text",
    }
)

#: A path's ``rename`` and ``replace`` take one target and nothing else; ``str.replace``
#: takes two, and a frame's ``rename`` takes keywords.
_ONE_TARGET_METHODS = frozenset({"rename", "replace"})

#: Functions that write outside io's net because what they write is not this process
#: writing a run's output tree, as ``(package-relative path, function)`` — a method or a
#: nested function dotted under its enclosing names — each with what it writes instead: a
#: file inside the job, whose refusal is the job's failure; a file a person owns — a
#: config, a template, a seed; a path outside every run tree; no file at all. An entry
#: excuses its whole function, so it names one that writes nothing of a run's tree raw.
_RAW_WRITERS: dict[tuple[str, str], str] = {
    ("agent_tools.py", "build_structures"): "seed files, a person's, in the directory the "
    "caller names",
    ("agent_tools.py", "read_structure_text"): "a scratch directory of its own, gone when it "
    "returns",
    ("agent_tools.py", "save_config"): "the config, a person's file: a ConfigError names it",
    ("agent_tools.py", "write_template"): "a step template, a person's file: a ConfigError "
    "names it",
    ("cache.py", "_npz_bytes"): "a buffer in memory, no file",
    ("engines/_backend_server/server.py", "main"): "the gradient server's log and loading "
    "marker, written inside the job",
    ("engines/_backend_server/sidecar.py", "write_server_token"): "the gradient server's "
    "token, written inside the job",
    ("engines/_backend_server/sidecar.py", "write_server_url"): "the gradient server's "
    "address, written inside the job",
    ("engines/_provision.py", "build_backend_env"): "a managed environment, outside every "
    "run tree: a BackendProvisionError says what the build could not do",
    ("engines/mlip/backends/chgnet.py", "ChgnetTrainer.train_with_library"): "the trained "
    "model, written by the training job",
    ("engines/mlip/backends/orb.py", "OrbTrainer.train_with_library"): "the checkpoints and "
    "the trained model, written by the training job",
    ("engines/orca/extopt/protocol.py", "write_engrad"): "the bridge's answer to the "
    "program, written inside the job",
    ("engines/pyscf/_runtime.py", "save_tensors"): "the tensors a calculation saves, written "
    "inside the job",
    ("gui/app.py", "create_app.save"): "the editor's YAML, a person's file: a 400 names it",
    ("scaffold.py", "scaffold_templates"): "starter templates, a person's files: a "
    "ConfigError names what landed",
}


class _RawWrite(NamedTuple):
    """One call that writes to the filesystem outside io's net."""

    function: str
    line: int
    primitive: str


def _bindings(tree: ast.AST) -> tuple[dict[str, str], set[str]]:
    """What each name a module imports means, dotted, and which of them name a module.

    Every import counts wherever it stands, so a function-local ``from ase.io import write``
    resolves as surely as one at the top.
    """
    names: dict[str, str] = {}
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                names[bound] = alias.name if alias.asname else bound
                modules.add(bound)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names, modules


def _resolved(expr: ast.expr, names: dict[str, str]) -> str | None:
    """``a.b.c`` with ``a`` replaced by what it imports, or ``None`` past a call or subscript."""
    parts: list[str] = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if not isinstance(expr, ast.Name):
        return None
    return ".".join([names.get(expr.id, expr.id), *reversed(parts)])


def _open_mode(call: ast.Call, position: int) -> str | None:
    """The mode an ``open`` spells, ``"?"`` for one it computes, ``None`` for the default.

    The ``mode=`` keyword, else the positional at ``position``: a function is handed the
    path first and the mode second, a path's own method the mode first. The reading is by
    name alone, so the first argument of any ``open`` method is read as its mode, something
    that is no path included: a computed one, or a literal holding ``w``, ``a``, ``x`` or
    ``+``, reads as a write.
    """
    spelled = next((keyword.value for keyword in call.keywords if keyword.arg == "mode"), None)
    if spelled is None and len(call.args) > position:
        spelled = call.args[position]
    if spelled is None:
        return None
    if isinstance(spelled, ast.Constant):
        return spelled.value if isinstance(spelled.value, str) else None
    return "?"


def _primitive(call: ast.Call, names: dict[str, str], modules: set[str]) -> str | None:
    """How ``call`` writes to the filesystem, or ``None`` when it does not."""
    func = call.func
    dotted = _resolved(func, names)
    if dotted in _WRITER_FUNCTIONS:
        return dotted
    file_keyword = _WRITER_KEYWORDS.get(dotted or "")
    if file_keyword is not None:
        return dotted if any(k.arg == file_keyword for k in call.keywords) else None
    owner = func.value if isinstance(func, ast.Attribute) else None
    method = owner is not None and not (isinstance(owner, ast.Name) and owner.id in modules)
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name in ("open", "fdopen"):
        mode = _open_mode(call, 0 if method else 1)
        writes = mode is not None and (mode == "?" or not set(mode).isdisjoint("wax+"))
        return f"{name}({mode!r})" if writes else None
    if not method:
        return None  # a module's own function, judged by its dotted name above
    one_target = name in _ONE_TARGET_METHODS and len(call.args) == 1 and not call.keywords
    return f".{name}" if name in _WRITER_METHODS or one_target else None


def _calls(
    tree: ast.AST, net: Callable[[ast.withitem], bool]
) -> Iterator[tuple[str, bool, ast.Call]]:
    """Every call in ``tree`` as ``(function, under the net, call)``.

    The function is the call's enclosing names, dotted — a method under its class, a nested
    function under the one that defines it — which is how an allowlist names it. Under the
    net is lexical: a call is under it where it stands, not because a caller up the stack
    holds one. ``with a, b:`` is ``with a: with b:``, so an item after the net stands under
    it, and a ``def`` or ``lambda`` written under the net runs later, outside it.
    """

    def walk(
        node: ast.AST, scope: tuple[str, ...], sealed: bool
    ) -> Iterator[tuple[str, bool, ast.Call]]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope, sealed = (*scope, node.name), False
        elif isinstance(node, ast.Lambda):
            scope, sealed = (*scope, "<lambda>"), False
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                yield from walk(item, scope, sealed)
                sealed = sealed or net(item)
            for statement in node.body:
                yield from walk(statement, scope, sealed)
            return
        if isinstance(node, ast.Call):
            yield ".".join(scope) or "<module>", sealed, node
        for child in ast.iter_child_nodes(node):
            yield from walk(child, scope, sealed)

    return walk(tree, (), False)


def _raw_writes(tree: ast.AST) -> Iterator[_RawWrite]:
    """Every call in ``tree`` that writes a path and stands inside no ``with`` of the net."""
    names, modules = _bindings(tree)

    def net(item: ast.withitem) -> bool:
        call = item.context_expr
        return isinstance(call, ast.Call) and _resolved(call.func, names) == _NET

    for function, sealed, call in _calls(tree, net):
        if not sealed and (primitive := _primitive(call, names, modules)) is not None:
            yield _RawWrite(function, call.lineno, primitive)


def _package_raw_writes() -> Iterator[tuple[str, _RawWrite]]:
    """Every raw write in the package outside io, as ``(package-relative path, write)``."""
    home = Path(io.__file__)
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if path == home:
            continue
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        for write in _raw_writes(ast.parse(path.read_text(encoding="utf-8"))):
            yield rel, write


def test_the_run_tree_is_written_only_through_io():
    """Outside io, a call that writes a path stands inside ``tree_write``, or is listed.

    A write a run's output tree refuses is one failure whichever layer made it: an
    :class:`~chemrefine.errors.OutputTreeError` naming the file, raised by the one net,
    :func:`chemrefine.io.tree_write`. A raw write standing anywhere else leaves that
    contract, and a full disk reaches the user as a traceback with the exit code of a
    halted step — unless what it writes is no part of a run's tree, which
    :data:`_RAW_WRITERS` says by function. Checked over the whole package but io itself,
    where the primitives live: a shipped engine is discovered from the package's own
    directory, so it is held by existing.
    """
    offences = [
        f"{rel}:{write.line}: {write.function} — {write.primitive}"
        for rel, write in _package_raw_writes()
        if (rel, write.function) not in _RAW_WRITERS
    ]
    assert not offences, (
        "a raw filesystem write outside io's net:\n  "
        + "\n  ".join(offences)
        + "\n\nWrite the file with io.write_tree_file (io.remove_tree_file removes one), or "
        "stand the call inside `with io.tree_write(path):`. A function whose write is no "
        "part of a run's tree — one the job runs, one writing a person's file — is listed "
        "in _RAW_WRITERS with what it writes."
    )


def test_every_listed_raw_writer_still_writes_raw():
    """An entry whose function holds no raw write has outlived its reason; it goes."""
    raw = {(rel, write.function) for rel, write in _package_raw_writes()}
    assert [key for key in _RAW_WRITERS if key not in raw] == []


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ('def f(p):\n    p.write_text("x")\n', [("f", ".write_text")]),
        ('def f(p):\n    open(p, "w")\n', [("f", "open('w')")]),
        ('def f(p):\n    p.open(mode="a")\n', [("f", "open('a')")]),
        ("def f(p, m):\n    open(p, m)\n    p.open(m)\n", [("f", "open('?')")] * 2),
        ('import os\ndef f(fd):\n    os.fdopen(fd, "wb")\n', [("f", "fdopen('wb')")]),
        ("def f(p, q):\n    p.replace(q)\n", [("f", ".replace")]),
        ("from shutil import copy2 as cp\ndef f(a, b):\n    cp(a, b)\n", [("f", "shutil.copy2")]),
        ("def f(p):\n    from ase.io import write\n    write(p, [])\n", [("f", "ase.io.write")]),
        (
            'from chemrefine import io\ndef f(p):\n    io.atomic_write(p, b"")\n',
            [("f", "chemrefine.io.atomic_write")],
        ),
        (
            "import logging\ndef f(p):\n    logging.basicConfig(filename=p)\n",
            [("f", "logging.basicConfig")],
        ),
        ("class C:\n    def m(self):\n        self.path.mkdir()\n", [("C.m", ".mkdir")]),
        ("def f(p):\n    return lambda: p.unlink()\n", [("f.<lambda>", ".unlink")]),
        ('from pathlib import Path\nPath("x").touch()\n', [("<module>", ".touch")]),
        (
            'from chemrefine import io\ndef f(p):\n    with open(p, "w") as h, io.tree_write(p):\n'
            "        pass\n",
            [("f", "open('w')")],
        ),
        (
            "from chemrefine import io\ndef f(p):\n    with io.tree_write(p):\n"
            "        def later():\n            p.unlink()\n    return later\n",
            [("f.later", ".unlink")],
        ),
        (
            "from chemrefine import io\ndef f(p):\n    with io.cleanup():\n        p.unlink()\n",
            [("f", ".unlink")],
        ),
        (
            "from elsewhere import tree_write\ndef f(p):\n    with tree_write(p):\n"
            "        p.unlink()\n",
            [("f", ".unlink")],
        ),
        ('def f(p):\n    open(p).read()\n    p.open("rb")\n    p.read_text()\n', []),
        ('def f(s, df):\n    s.replace("a", "b")\n    df.rename(columns={})\n', []),
        (
            "import logging\nimport webbrowser\ndef f(url):\n    webbrowser.open(url)\n"
            "    logging.basicConfig(level=10)\n",
            [],
        ),
        (
            'from chemrefine import io\ndef f(p):\n    io.write_tree_file(p, "x")\n'
            "    io.remove_tree_file(p)\n",
            [],
        ),
        (
            "from chemrefine import io\ndef f(p):\n    with io.tree_write(p):\n"
            "        p.parent.mkdir(parents=True, exist_ok=True)\n        p.write_text('x')\n",
            [],
        ),
        (
            "from chemrefine.io import tree_write\nfrom ase.db import connect\ndef f(p):\n"
            "    with tree_write(p), connect(str(p)) as db:\n        db.write(None)\n",
            [],
        ),
        (
            "import os\nfrom chemrefine import io\ndef f(claim, lock):\n"
            "    with io.tree_write(lock):\n        try:\n            os.link(claim, lock)\n"
            "        except FileExistsError:\n            pass\n",
            [],
        ),
    ],
    ids=[
        "write-text",
        "open-for-write",
        "path-open-to-append",
        "computed-mode",
        "fdopen",
        "path-replace",
        "aliased-import",
        "local-import",
        "raw-replace",
        "log-file",
        "method",
        "lambda",
        "module-level",
        "item-before-the-net",
        "defined-under-the-net-runs-outside-it",
        "cleanup-is-no-net",
        "a-namesake-is-no-net",
        "reads",
        "str-replace-and-frame-rename",
        "a-modules-own-open-and-a-console-log",
        "the-whole-file-forms",
        "under-the-net",
        "item-after-the-net",
        "normal-flow-handled-inside-the-net",
    ],
)
def test_the_write_gate_finds_what_it_is_for(source: str, found: list[tuple[str, str]]):
    """The gate passes on a clean tree, so its walker is shown trees that are not."""
    writes = [(write.function, write.primitive) for write in _raw_writes(ast.parse(source))]
    assert writes == found


def test_every_writer_the_gate_names_is_one_it_finds():
    """A writer in the vocabulary that the walker cannot see would be a gate on nothing."""
    planted: dict[str, str] = {}
    for name in _WRITER_FUNCTIONS:
        module, _, function = name.rpartition(".")
        planted[f"import {module}\ndef f(p):\n    {name}(p, p)\n"] = name
        planted[f"from {module} import {function}\ndef f(p):\n    {function}(p, p)\n"] = name
    for name, file_keyword in _WRITER_KEYWORDS.items():
        module = name.rpartition(".")[0]
        planted[f"import {module}\ndef f(p):\n    {name}({file_keyword}=p)\n"] = name
    for name in _WRITER_METHODS:
        planted[f"def f(p):\n    p.{name}()\n"] = f".{name}"
    for name in _ONE_TARGET_METHODS:
        planted[f"def f(p, q):\n    p.{name}(q)\n"] = f".{name}"
    missed = [
        source
        for source, primitive in planted.items()
        if [write.primitive for write in _raw_writes(ast.parse(source))] != [primitive]
    ]
    assert missed == []


# ---------------------------------------------------------------------------
# Questions — what a run's tree holds is asked through io
# ---------------------------------------------------------------------------

#: ``Path`` methods whose answer over a run's tree cannot be taken at its word, by name. Under
#: a directory the account cannot search, ``exists``, ``is_file`` and ``is_dir`` raise on
#: Python 3.11 to 3.13 and answer ``False`` on 3.14; ``glob`` and ``rglob`` answer an empty
#: list for a directory they cannot list, on every Python.
_ASKING_METHODS = frozenset({"exists", "glob", "is_dir", "is_file", "rglob"})

_A_PERSONS_FILE = "a file or directory a person names: its reader words what it cannot use"
_A_STATUS_READ = "a status read of a tree it does not drive: it reports, and holds no lock"
_A_MANAGED_ENVIRONMENT = "a managed environment, outside every run tree"
_INSIDE_THE_JOB = "a file of the job's own, asked inside the job"

#: Functions that ask one of those of something that is no run's tree read by its driver, as
#: ``(package-relative path, function)`` like :data:`_RAW_WRITERS`, each with what it asks
#: about instead. An entry excuses its whole function, so it names one that asks nothing of
#: a run's tree while driving it.
_RAW_ASKERS: dict[tuple[str, str], str] = {
    **dict.fromkeys(
        [
            ("agent_tools.py", "_seeds_for_reading"),
            ("agent_tools.py", "build_structures"),
            ("agent_tools.py", "read_structure_file"),
            ("cache.py", "template_digest"),
            ("config.py", "Config.missing_executables"),
            ("engines/orca/input.py", "_referenced_file"),
            ("gui/app.py", "create_app.bootstrap"),
            ("gui/app.py", "create_app.browse"),
            ("pipeline.py", "_seed"),
            ("pipeline.py", "bootstrap"),
            ("scaffold.py", "plan_templates"),
            ("step.py", "derive_step_key"),
            ("validate.py", "_inspect_steps"),
            ("validate.py", "validate_config_text"),
        ],
        _A_PERSONS_FILE,
    ),
    **dict.fromkeys(
        [
            ("agent_tools.py", "_latest_log"),
            ("agent_tools.py", "_locate_output"),
            ("agent_tools.py", "_steps_csv_rows"),
            ("cache.py", "has_results"),
            ("pipeline.py", "lock_status"),
        ],
        _A_STATUS_READ,
    ),
    **dict.fromkeys(
        [
            ("cli.py", "backends_install"),
            ("cli.py", "backends_list"),
            ("cli.py", "backends_path"),
            ("engines/_provision.py", "_env_distribution"),
            ("engines/_provision.py", "_env_lib"),
            ("engines/_provision.py", "_existing_env_tool"),
            ("engines/_provision.py", "_install_spec"),
            ("engines/_provision.py", "build_backend_env"),
            ("engines/_provision.py", "detect_env_tool"),
            ("engines/_provision.py", "require_backend"),
            ("engines/_provision.py", "resolve_launcher"),
        ],
        _A_MANAGED_ENVIRONMENT,
    ),
    **dict.fromkeys(
        [
            ("engines/mlip/backends/chgnet.py", "ChgnetTrainer.train_with_library"),
            ("engines/mlip/calculator.py", "build_calculator"),
            ("engines/orca/extopt/bridge.py", "resolve_server_token"),
        ],
        _INSIDE_THE_JOB,
    ),
}


class _RawAsk(NamedTuple):
    """One call that asks a path what is there through a method io's questions replace."""

    function: str
    line: int
    method: str


def _raw_asks(tree: ast.AST) -> Iterator[_RawAsk]:
    """Every call in ``tree`` that asks a path what is there through one of the methods.

    Matched by the method's name on anything that is no imported module: a receiver's type
    is not in the syntax, and a module's own function of the name — ``glob.glob``,
    ``os.path.exists`` — is another function with another answer.
    """
    _names, modules = _bindings(tree)
    for function, _sealed, call in _calls(tree, lambda item: False):
        func = call.func
        if not isinstance(func, ast.Attribute) or func.attr not in _ASKING_METHODS:
            continue
        owner: ast.expr = func.value
        while isinstance(owner, ast.Attribute):
            owner = owner.value
        if not (isinstance(owner, ast.Name) and owner.id in modules):
            yield _RawAsk(function, call.lineno, f".{func.attr}")


def _package_raw_asks() -> Iterator[tuple[str, _RawAsk]]:
    """Every such question in the package outside io, as ``(package-relative path, call)``."""
    home = Path(io.__file__)
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if path == home:
            continue
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        for ask in _raw_asks(ast.parse(path.read_text(encoding="utf-8"))):
            yield rel, ask


def test_a_run_tree_is_asked_what_it_holds_only_through_io():
    """Outside io, no call asks a ``Path`` whether something is there, or globs — or is listed.

    Whether a job left its output, whether a record was ever written and which attempts a
    structure has are each asked in a run's own tree, where another account's mode or a
    directory that cannot be searched must be a named refusal. ``Path.exists``, ``is_file``
    and ``is_dir`` raise there on one Python and answer ``False`` on the next, and ``glob``
    answers an empty list everywhere — a traceback, or work read as never done. So the
    question is :func:`chemrefine.io.absent`, the listing :func:`chemrefine.io.tree_entries`,
    and a reader that reads at once takes absence from its own ``FileNotFoundError``. A
    function asking about something else — :data:`_RAW_ASKERS` says what — is listed.
    """
    offences = [
        f"{rel}:{ask.line}: {ask.function} — {ask.method}"
        for rel, ask in _package_raw_asks()
        if (rel, ask.function) not in _RAW_ASKERS
    ]
    assert not offences, (
        "a path asked what is there outside io:\n  "
        + "\n  ".join(offences)
        + "\n\nIn a run's tree ask io.absent(path), list with io.tree_entries(directory), or "
        "read and take absence from FileNotFoundError: Path.exists/is_file/is_dir answer "
        "differently across Python versions under a directory that cannot be searched, and "
        "glob reads one it cannot list as empty. A function that asks about something that "
        "is no run's tree read by its driver is listed in _RAW_ASKERS with what it asks about."
    )


def test_every_listed_asker_still_asks():
    """An entry whose function asks nothing of the kind has outlived its reason; it goes."""
    asking = {(rel, ask.function) for rel, ask in _package_raw_asks()}
    assert [key for key in _RAW_ASKERS if key not in asking] == []


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("def f(p):\n    return p.is_file()\n", [("f", ".is_file")]),
        (
            "def f(p):\n    return p.parent.exists() or p.is_dir()\n",
            [("f", ".exists"), ("f", ".is_dir")],
        ),
        (
            'def f(d):\n    return sorted(d.glob("attempt*")) + list(d.rglob("*"))\n',
            [("f", ".glob"), ("f", ".rglob")],
        ),
        ("class C:\n    def m(self):\n        return self.path.exists()\n", [("C.m", ".exists")]),
        (
            "from chemrefine import io\ndef f(p):\n    with io.tree_write(p):\n"
            "        return p.exists()\n",
            [("f", ".exists")],
        ),
        (
            "import glob\nimport os\ndef f(p):\n    return glob.glob(p), os.path.exists(p)\n",
            [],
        ),
        (
            "from chemrefine import io\ndef f(p):\n    return io.absent(p), io.tree_entries(p)\n",
            [],
        ),
        ("def f(p):\n    return p.read_text(), p.stat(), list(p.iterdir())\n", []),
    ],
    ids=[
        "is-file",
        "exists-and-is-dir",
        "glob-and-rglob",
        "method",
        "the-write-net-is-no-excuse",
        "a-modules-own-function",
        "ios-questions",
        "reads-and-listings-that-raise",
    ],
)
def test_the_question_gate_finds_what_it_is_for(source: str, found: list[tuple[str, str]]):
    """The gate passes on a clean tree, so its walker is shown trees that are not."""
    asks = [(ask.function, ask.method) for ask in _raw_asks(ast.parse(source))]
    assert asks == found
