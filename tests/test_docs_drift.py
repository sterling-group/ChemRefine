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

import re
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
