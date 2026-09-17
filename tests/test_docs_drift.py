"""Drift guards: the hand-written configuration reference is held to the models.

Two pages duplicate the Pydantic models in prose tables — ``docs/workflow/configuration.md``
for the config's own keys and ``docs/engines/index.md`` for each engine's ``options`` (or
``docs/engines/<name>.md``, when an engine has a page of its own) — and they are the places in
the docs that can silently rot as fields are added or their defaults move. Two guards make
that drift loud. Every field of every model that :func:`chemrefine.introspect.schema_document`
exposes must appear (as a whole word) in the section that owns it; and every row of that
section's option table must state the model's default — or name, in ``_STATED_RULES``, the
code that decides the value instead — and exactly the aliases the model accepts. The reverse
direction — a page naming a field the schema lost — is already covered by
``test_docs_examples``, which validates the YAML fences against the real models.

The tables stay hand-written on purpose, and these guards are what that costs. ``docs/hooks/
tables.py`` generates the rosters that are pure fact (which engines exist, what each
drives, which backends install); the option tables are not, because the models declare no
``description=`` on any field — generating them would trade the page's teaching for a
list of defaults. So the page keeps the prose and the guards keep the page honest about
*which* knobs exist and *what* they default to.
"""

from __future__ import annotations

import ast
import pydoc
import re
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest
from pydantic import BaseModel

from chemrefine.config import BoltzmannSample, Config, MaxSample, MinSample, StepConfig
from chemrefine.engines._options import EngineOptions
from chemrefine.engines.api import ENGINES, OptionsDeclaring, get_engine
from chemrefine.nms import NmsOptions

_DOCS = Path(__file__).resolve().parent.parent / "docs"
_CONFIG_PAGE = _DOCS / "workflow" / "configuration.md"
_ENGINE_PAGES = _DOCS / "engines"
_ENGINES_PAGE = _ENGINE_PAGES / "index.md"

# A guard on the repository's prose, not on the package: the sdist ships the suite so a
# distro packager can run it, but `docs/` (14 MB of site assets) deliberately does not
# ship — so with no page to check, the guard has nothing to say. In the repo the file
# always exists and the guard always runs.
pytestmark = pytest.mark.skipif(
    not _CONFIG_PAGE.is_file(),
    reason="docs/ is a repository artifact and does not ship in the sdist",
)


class _Documented(NamedTuple):
    """A model's home in the docs: its page, and the ``## `` section that owns its rows."""

    page: Path
    model: type[BaseModel]
    heading: str | None
    """A regex for the owning section's ``## `` heading, or ``None`` when the model owns
    the whole page (an engine with a page of its own)."""


def _engine_page(name: str) -> Path:
    """The page that owns an engine's option table: its own if it has one, else the index.

    ``docs/engines/<name>.md`` is an engine's to write — an engine whose component tables
    run to a thousand lines should not pour them into the shared page — and the generated
    engine table links to it the moment it exists (``docs/hooks/tables.py``), so the index
    needs no hand edit for a new engine's page to be reachable. A family sharing one model
    documents on one page, as PySCF does on the index.
    """
    own = _ENGINE_PAGES / f"{name}.md"
    return own if own.is_file() else _ENGINES_PAGE


def _section(heading: str, text: str) -> str:
    """The ``## `` section whose heading matches ``heading``, up to the next ``## ``.

    A shared page names every model's knobs, so searching the whole of it lets one
    engine's row document another engine's field: ``backend_python`` was accepted on a
    ``qchem`` step and absent from the Q-Chem table, and the guard passed on the MLIP and
    PySCF rows. A family documenting several engines under one heading (``pyscf``,
    ``pyscf-extopt``) names each in the heading's backticks, which is what an engine's
    heading pattern keys on.
    """
    match = re.search(rf"^## {heading}.*$", text, re.MULTILINE)
    assert match, f"no `## {heading}` section to document it in"
    rest = text[match.end() :]
    following = re.search(r"^## ", rest, re.MULTILINE)
    return rest[: following.start()] if following else rest


def _documented_universe() -> dict[str, _Documented]:
    """Every model the docs must describe, with the page and section that own it.

    The config's own models are documented on the configuration page, each under its own
    heading; an engine's ``options`` model on the engines page — inside that engine's own
    section, beside the generated table and its rules — or on the engine's own page when it
    has one. Naming one target rather than searching every page keeps the guards specific:
    a knob documented on the wrong page, or in another model's section, is still a knob a
    reader will not find.
    """
    universe = {
        "Config": _Documented(_CONFIG_PAGE, Config, "Top-level keys"),
        "StepConfig": _Documented(_CONFIG_PAGE, StepConfig, "Per-step keys"),
        "BoltzmannSample": _Documented(_CONFIG_PAGE, BoltzmannSample, r"Sample \("),
        "MinSample": _Documented(_CONFIG_PAGE, MinSample, r"Sample \("),
        "MaxSample": _Documented(_CONFIG_PAGE, MaxSample, r"Sample \("),
        "NmsOptions": _Documented(_CONFIG_PAGE, NmsOptions, "NMS options"),
    }
    for name in sorted(ENGINES):
        engine = get_engine(name)
        if isinstance(engine, OptionsDeclaring):
            page = _engine_page(name)
            heading = rf".*`{re.escape(name)}`" if page == _ENGINES_PAGE else None
            universe[f"options[{name}]"] = _Documented(page, engine.options_cls, heading)
    return universe


def _owned_text(target: _Documented) -> str:
    """The prose the guards read for a model: its section, or the whole of its own page."""
    text = target.page.read_text(encoding="utf-8")
    return text if target.heading is None else _section(target.heading, text)


def _case_id(part: object) -> str:
    return part if isinstance(part, str) else ""


_UNIVERSE = sorted(_documented_universe().items())


@pytest.mark.parametrize(("model", "target"), _UNIVERSE, ids=_case_id)
def test_every_schema_field_is_named_in_the_configuration_page(model: str, target: _Documented):
    text = _owned_text(target)
    missing = sorted(
        field
        for field in target.model.model_fields
        if not re.search(rf"\b{re.escape(field)}\b", text)
    )
    assert not missing, (
        f"{model} field(s) {missing} are not mentioned in {target.page.name}; the schema "
        "moved and the hand-written reference did not. Document them (or generate the page)."
    )


# ---------------------------------------------------------------------------
# The option tables' defaults and aliases
# ---------------------------------------------------------------------------

_SELECTOR_TABLES = {"BoltzmannSample", "MinSample", "MaxSample"}
"""Models whose section documents by selector rather than by an option table.

``sample`` takes one method and one selector, and the page says so in prose and a
``method | Selector(s) | Keeps`` table with no ``Default`` column. The name guard reads
that section like any other; the default guard has no table there to hold to the models.
"""

_TRAIN = "chemrefine.engines.mlip.train.engine.MlipTrainEngine"
_PYSCF = "chemrefine.engines.pyscf.options.PyscfOptions"
_STATED_RULES: dict[tuple[str, str], str] = {
    ("StepConfig", "template"): "chemrefine.ids.step_template_path",
    ("StepConfig", "slurm_template"): "chemrefine.engines._execution._header_name",
    ("StepConfig", "charge"): "chemrefine.config.StepConfig.effective_charge",
    ("StepConfig", "multiplicity"): "chemrefine.config.StepConfig.effective_multiplicity",
    ("NmsOptions", "target"): "chemrefine.nms._resolved_options",
    ("options[mlip-train]", "task_name"): f"{_TRAIN}.check_step",
    ("options[mlip-train]", "device"): f"{_TRAIN}.check_step",
    ("options[mlip-train]", "cores"): f"{_TRAIN}.pal",
    ("options[mlip-train]", "extra"): (
        "chemrefine.engines.mlip.options.MlipTrainOptions._no_template_knobs"
    ),
    ("options[pyscf]", "xc"): f"{_PYSCF}.require_level_of_theory",
    ("options[pyscf]", "basis"): f"{_PYSCF}.require_level_of_theory",
    ("options[pyscf]", "gpu"): f"{_PYSCF}._derive_gpu_from_device",
    ("options[pyscf-extopt]", "xc"): f"{_PYSCF}.require_level_of_theory",
    ("options[pyscf-extopt]", "basis"): f"{_PYSCF}.require_level_of_theory",
    ("options[pyscf-extopt]", "gpu"): f"{_PYSCF}._derive_gpu_from_device",
}
"""Rows whose ``Default`` cell states a rule, keyed ``(model, field)`` to the code deciding it.

What a reader gets for these is not the field's default: the model's value is a placeholder
that code reads past — a template name derived from the step number, a header that falls
back to the workflow's, an NMS target inferred from the template, a training knob the engine
requires although the inherited field has a default. The page says so in words, and the
guard holds such a row to the deciding code existing rather than to a literal, so a rule
that is renamed or removed fails here instead of leaving the page describing nothing.
"""


def _cells(line: str) -> list[str]:
    """A table line's cells, stripped; an escaped ``\\|`` inside a cell does not split it."""
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]]


def _option_rows(section: str) -> list[tuple[str, str]]:
    """``(key cell, default cell)`` for every row of the section's option table.

    The first table whose header names a ``Default`` column, read by header position, so a
    ``Type`` column between the two cannot shift a default under another key. A section
    with no such table — the selector-shaped ``sample`` one — yields no rows.
    """
    lines = section.splitlines()
    for start, line in enumerate(lines):
        header = _cells(line)
        if "Default" in header:
            key, default = header.index("Key"), header.index("Default")
            rows: list[tuple[str, str]] = []
            for row in lines[start + 2 :]:
                if not row.startswith("|"):
                    break
                cells = _cells(row)
                rows.append((cells[key], cells[default]))
            return rows
    return []


def _row_names(key_cell: str) -> tuple[list[str], set[str]]:
    """The fields a key cell documents and the aliases it claims for them.

    A cell ``charge`` / ``multiplicity`` is two fields on one row; ``model_name`` (aliases
    ``model``, ``size``) is one field and its other spellings; a marker such as
    *(ExtOpt only)* carries no backticks and is ignored.
    """
    alias = re.search(r"\((?:alias|aliases) ([^)]*)\)", key_cell)
    names = re.findall(r"`([^`]+)`", key_cell[: alias.start()] if alias else key_cell)
    aliases = set(re.findall(r"`([^`]+)`", alias.group(1))) if alias else set()
    return names, aliases


def _spellings(default: object) -> set[str]:
    """The literals a table may print for a model default.

    ``None`` / ``True`` / ``False``, an int, a float with or without a trailing ``.0``
    (``600`` for ``600.0``), a bare string (``""`` when empty), a path with or without
    ``./``, and ``{}`` for an empty mapping.
    """
    if isinstance(default, float) and default.is_integer():
        return {str(default), str(int(default))}
    if isinstance(default, str):
        return {default or '""'}
    if isinstance(default, Path):
        return {str(default), f"./{default}"}
    return {repr(default)}


def test_every_stated_rule_names_a_documented_field_and_real_code():
    """A rule that outlived its model, its field or its code would excuse a row from nothing."""
    universe = dict(_UNIVERSE)
    assert set(universe) >= _SELECTOR_TABLES
    for (model, field), rule in _STATED_RULES.items():
        assert field in universe[model].model.model_fields, f"{model} has no field {field!r}"
        assert pydoc.locate(rule) is not None, f"{model}.{field}: {rule} does not exist"


@pytest.mark.parametrize(
    ("model", "target"),
    [item for item in _UNIVERSE if item[0] not in _SELECTOR_TABLES],
    ids=_case_id,
)
def test_every_documented_default_and_alias_is_the_models(model: str, target: _Documented):
    """The option table's ``Default`` column and alias notes are the model's, row by row.

    The name guard says a knob is mentioned; this one says what the page claims about it
    holds. A default that moved in the model and not on the page is the reader's wrong
    number, and an alias the page names that the model dropped is a step failing on the
    spelling the docs taught. A row that states a rule instead of a literal must be in
    ``_STATED_RULES``; a ``—`` cell must be a field the model itself requires; and every
    field must have a row, since a knob the table skips is one the reader cannot set.
    """
    fields = target.model.model_fields
    spellings = (
        target.model._spellings_by_field() if issubclass(target.model, EngineOptions) else {}
    )
    documented: set[str] = set()
    for key_cell, default_cell in _option_rows(_owned_text(target)):
        names, aliases = _row_names(key_cell)
        for field in names:
            if field not in fields:
                continue  # another engine's knob, on a table its family shares
            documented.add(field)
            assert aliases == spellings.get(field, {field}) - {field}, (
                f"{model}.{field}: the page names aliases {sorted(aliases)}"
            )
            if (model, field) in _STATED_RULES:
                continue
            if default_cell.startswith("—"):
                assert fields[field].is_required(), f"{model}.{field} is not required"
                continue
            literal = re.match(r"`([^`]*)`", default_cell)
            assert literal, (
                f"{model}.{field}: state the default as a literal, or record the rule that "
                "decides it in _STATED_RULES"
            )
            default = fields[field].get_default(call_default_factory=True)
            assert literal.group(1) in _spellings(default), (
                f"{model}.{field}: the page says `{literal.group(1)}`, the model {default!r}"
            )
    assert documented == set(fields), (
        f"{model}: the option table has no row for {sorted(set(fields) - documented)}"
    )


# ---------------------------------------------------------------------------
# Prose that counts something the tree can grow
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parent.parent / "src" / "chemrefine"
_TESTS = Path(__file__).resolve().parent

# Populations the tree *grows*: a registry gains an entry, a recording is captured, a module
# imports one more thing. Deliberately not "readers" / "call sites" / "places" — those are
# usually an argument's shape ("two readers of one knob cannot disagree"), not a census.
_COUNTABLE = (
    r"engines|backends|trainers|builders|libraries|plugins|heads|extras"
    r"|modules|archives|recordings|recorded outputs|recorded blocks|recorded runs"
)
_MAGNITUDE = r"\b(?:\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|seventeen)\b"
_COUNTED_PROSE = re.compile(rf"{_MAGNITUDE}\s+(?:{_COUNTABLE})\b", re.IGNORECASE)

_ALLOWED = (
    # Structural constants — fixed by a format or by geometry, not by what the tree holds.
    "three Cartesian",
    "three splits",
)


def _prose_blocks(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line number, text)`` for every docstring and comment block in ``path``.

    Every bare string expression, not only what :func:`ast.get_docstring` returns. This
    codebase hangs prose off nearly every ClassVar and module constant, and an *attribute*
    docstring is a bare ``Expr`` the AST helper does not report — so reading only the helper
    would leave the guard blind to most of the surface it claims to cover.

    Comments by scan, since a decision written as a ``#`` block rots exactly as a docstring
    does.
    """
    source = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            yield node.lineno, node.value.value
    for number, line in enumerate(source.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            yield number, stripped


def test_no_docstring_counts_something_the_tree_can_grow() -> None:
    """A magnitude in prose rots the moment an engine, backend or recording arrives.

    Nothing recounts it, so it is wrong silently and stays wrong: this guard was written
    after five such claims were found already false — a plugin roster that had not heard of
    Q-Chem, an importer count off by eight, and a corpus tally naming more than twice the
    outputs the archives hold.

    The fix is never a fresh number. It is the invariant the number was standing in for:
    "every registered trainer" rather than five of them, "the recorded outputs" rather than
    108 of them. Where a count really is structural — three Cartesian components, the three
    splits a dataset has — add it to ``_ALLOWED`` with the reason.
    """
    offenders: list[str] = []
    for path in sorted([*_SRC.rglob("*.py"), *_TESTS.glob("test_*.py")]):
        for number, text in _prose_blocks(path):
            flat = " ".join(text.split())
            for match in _COUNTED_PROSE.finditer(flat):
                phrase = match.group(0)
                window = flat[max(0, match.start() - 30) : match.end() + 30]
                if any(ok in window for ok in _ALLOWED):
                    continue
                offenders.append(f"{path.relative_to(_SRC.parent.parent)}:{number}: {phrase!r}")
    assert offenders == [], (
        "prose counts something the tree can grow; state the invariant instead:\n  "
        + "\n  ".join(offenders)
    )
