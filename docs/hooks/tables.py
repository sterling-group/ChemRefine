"""MkDocs hook: the engine and backend tables, generated from the registry.

Every roster the docs used to type by hand already exists in the code — the engines in
:data:`chemrefine.engines.api.ENGINES` with their capabilities detected by ``isinstance``,
the backends in each provisionable engine's ``backend_extras()``, the MLIP task names in
:mod:`chemrefine.engines.mlip.registry`, the pip extras in ``pyproject.toml``. Typing them
a second time in prose is what let ``qchem`` go missing from seven pages and exit code
``10`` from the CLI list. So the pages carry a directive and this hook fills it in at
build time: a registered engine is described the moment it registers, and no roster can
disagree with the registry because there is only one.

Facts only. What a knob *means* — that Q-Chem takes its parallelism on the command line
where ORCA takes it in ``%pal`` — is not in the models (they declare no field
descriptions), so the per-engine option tables stay hand-written beside these.

Runs at ``on_page_markdown``, so what it emits is ordinary markdown that ``--strict``
still link-checks. An unknown ``chemrefine:`` directive raises rather than rendering
as an HTML comment nobody would notice.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

#: ``<!-- chemrefine:name -->`` alone on a line. Anchored so prose *about* a directive
#: (this module's own docs, for one) is never mistaken for a directive.
_DIRECTIVE = re.compile(r"^[ \t]*<!--[ \t]*chemrefine:([a-z][a-z-]*)[ \t]*-->[ \t]*$", re.MULTILINE)

#: A self-referential extra (``chemrefine[server]``) — rendered as the extra it pulls in
#: rather than as a dependency line, so ``[gui] = [server]`` reads as the alias it is.
_SELF_EXTRA = re.compile(r"^chemrefine\[([^\]]+)\]$")


def _cell(text: str) -> str:
    """Escape the one character that would silently split a markdown table row."""
    return text.replace("|", "\\|")


def _code_list(values: Iterable[object], empty: str) -> str:
    """Render an iterable of names as inline code, or ``empty`` when there are none."""
    items = tuple(values)
    return ", ".join(f"`{_cell(str(v))}`" for v in items) if items else empty


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """A GitHub-flavoured markdown table; the header row fixes the column count."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _engines_table() -> str:
    """Every registered engine: its template, `operation:` vocabulary, NMS, backends."""
    from chemrefine.introspect import describe_engines

    rows: list[tuple[str, ...]] = []
    for d in describe_engines():
        template = f"`step{{N}}.{d.template_suffix}` ({d.label})" if d.template_driven else "none"
        rows.append(
            (
                f"`{d.name}`",
                template,
                # Empty is not "no operations": OperationsDeclaring is opt-in, and an
                # engine that does not declare one treats `operation:` as a free label.
                _code_list(d.operations, "any — treated as a label"),
                "yes" if "nms" in d.capabilities else "—",
                _code_list(d.backend_extras, "none needed"),
            )
        )
    return _table(("Engine", "Step template", "`operation:`", "NMS", "Backend env"), rows)


def _backends_table() -> str:
    """The names ``chemrefine backends install`` accepts, and what each unlocks."""
    from chemrefine.engines import known_backend_extras
    from chemrefine.engines.mlip.registry import (
        backend_spec,
        registered_backends,
        registered_trainers,
    )
    from chemrefine.introspect import describe_engines

    engines = describe_engines()
    trainable = registered_trainers()
    tasks_by_extra: dict[str, list[str]] = {}
    for task in sorted(registered_backends()):
        tasks_by_extra.setdefault(backend_spec(task).library.extra, []).append(task)

    rows: list[tuple[str, ...]] = []
    for extra in sorted(known_backend_extras()):
        tasks = tasks_by_extra.get(extra, [])
        # A training engine needs a backend with a trainer: `trainer_for` raises
        # ConfigError ("can be run but not trained") for chgnet, orb and sevenn, so
        # listing `mlip-train` against those would advertise a step that cannot run.
        users = [
            d.name
            for d in engines
            if extra in d.backend_extras
            and ("artifact" not in d.capabilities or any(t in trainable for t in tasks))
        ]
        rows.append(
            (
                f"`{extra}`",
                _code_list(users, "—"),
                _code_list(tasks, "—"),
                _code_list(_third_party(extra), "—"),
            )
        )
    return _table(("Backend", "Engines", "`task_name`(s)", "Installs"), rows)


def _extras_table() -> str:
    """Every pip extra in ``pyproject.toml`` — the half the install page never listed."""
    optional = _optional_dependencies()
    rows: list[tuple[str, ...]] = []
    for extra in sorted(optional):
        third_party = _third_party(extra)
        aliased = _aliases(extra)
        rows.append(
            (
                f"`[{extra}]`",
                _code_list(third_party, "—"),
                _code_list((f"[{a}]" for a in aliased), "—"),
            )
        )
    return _table(("Extra", "Installs", "Also pulls"), rows)


def _optional_dependencies() -> dict[str, list[str]]:
    """``[project.optional-dependencies]`` — the source of truth for what is installable."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = data["project"]["optional-dependencies"]
    return result


def _third_party(extra: str) -> list[str]:
    """One extra's real dependencies — its ``chemrefine[...]`` self-references removed."""
    return [d for d in _optional_dependencies().get(extra, []) if not _SELF_EXTRA.match(d)]


def _aliases(extra: str) -> list[str]:
    """The other extras this one pulls in, flattened from its ``chemrefine[a,b]`` entries."""
    names: list[str] = []
    for dep in _optional_dependencies().get(extra, []):
        if match := _SELF_EXTRA.match(dep):
            names += [part.strip() for part in match.group(1).split(",")]
    return names


#: Directive name -> the markdown it expands to. Adding a table means adding one entry.
_TABLES: dict[str, Callable[[], str]] = {
    "engines": _engines_table,
    "backends": _backends_table,
    "extras": _extras_table,
}


def on_page_markdown(markdown: str, **kwargs: Any) -> str:
    """Expand every ``<!-- chemrefine:… -->`` directive on one page.

    Raises
    ------
    ValueError
        If a directive names no known table. A typo would otherwise render as an HTML
        comment — invisible on the built page and invisible in review.
    """

    def expand(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in _TABLES:
            raise ValueError(
                f"unknown directive <!-- chemrefine:{name} -->; known tables: {sorted(_TABLES)}"
            )
        return _TABLES[name]()

    return _DIRECTIVE.sub(expand, markdown)
