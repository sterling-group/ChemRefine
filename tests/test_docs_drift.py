"""Drift guard: the hand-written configuration reference must name every schema field.

Two pages duplicate the Pydantic models in prose tables — ``docs/workflow/configuration.md``
for the config's own keys and ``docs/engines/index.md`` for each engine's ``options`` — and
they are the places in the docs that can silently rot as fields are added. This guard makes
that drift loud: every field of every model that
:func:`chemrefine.introspect.schema_document` exposes must appear (as a whole word) on the
page that owns it. The reverse direction — a page naming a field the schema lost — is
already covered by ``test_docs_examples``, which validates the YAML fences against the real
models.

The tables stay hand-written on purpose, and this guard is what that costs. ``docs/hooks/
tables.py`` generates the rosters that are pure fact (which engines exist, what each
drives, which backends install); the option tables are not, because the models declare no
``description=`` on any field — generating them would trade the page's teaching for a
list of defaults. So the page keeps the prose and the guard keeps the page honest about
*which* knobs exist.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from chemrefine.config import BoltzmannSample, Config, MaxSample, MinSample, StepConfig
from chemrefine.engines.api import ENGINES, OptionsDeclaring, get_engine
from chemrefine.nms import NmsOptions

_DOCS = Path(__file__).resolve().parent.parent / "docs"
_CONFIG_PAGE = _DOCS / "workflow" / "configuration.md"
_ENGINES_PAGE = _DOCS / "engines" / "index.md"

# A guard on the repository's prose, not on the package: the sdist ships the suite so a
# distro packager can run it, but `docs/` (14 MB of site assets) deliberately does not
# ship — so with no page to check, the guard has nothing to say. In the repo the file
# always exists and the guard always runs.
pytestmark = pytest.mark.skipif(
    not _CONFIG_PAGE.is_file(),
    reason="docs/ is a repository artifact and does not ship in the sdist",
)


def _documented_universe() -> dict[str, tuple[Path, set[str]]]:
    """Every field name the docs must mention, with the page that owns it.

    The config's own models are documented on the configuration page; an engine's
    ``options`` model on the engines page, beside the generated table and that engine's
    own rules. Splitting the target rather than searching both keeps the guard specific:
    a knob documented on the wrong page is still a knob a reader will not find.
    """
    universe: dict[str, tuple[Path, set[str]]] = {
        "Config": (_CONFIG_PAGE, set(Config.model_fields)),
        "StepConfig": (_CONFIG_PAGE, set(StepConfig.model_fields)),
        "BoltzmannSample": (_CONFIG_PAGE, set(BoltzmannSample.model_fields)),
        "MinSample": (_CONFIG_PAGE, set(MinSample.model_fields)),
        "MaxSample": (_CONFIG_PAGE, set(MaxSample.model_fields)),
        "NmsOptions": (_CONFIG_PAGE, set(NmsOptions.model_fields)),
    }
    for name in sorted(ENGINES):
        engine = get_engine(name)
        if isinstance(engine, OptionsDeclaring):
            universe[f"options[{name}]"] = (_ENGINES_PAGE, set(engine.options_cls.model_fields))
    return universe


@pytest.mark.parametrize(
    ("model", "target"),
    sorted(_documented_universe().items()),
    ids=lambda part: part if isinstance(part, str) else "",
)
def test_every_schema_field_is_named_in_the_configuration_page(
    model: str, target: tuple[Path, set[str]]
):
    _DOC, fields = target
    text = _DOC.read_text(encoding="utf-8")
    missing = sorted(field for field in fields if not re.search(rf"\b{re.escape(field)}\b", text))
    assert not missing, (
        f"{model} field(s) {missing} are not mentioned in {_DOC.name}; the schema moved "
        "and the hand-written reference did not. Document them (or generate the page)."
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
