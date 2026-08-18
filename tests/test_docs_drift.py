"""Drift guard: the hand-written configuration reference must name every schema field.

``docs/user-guide/configuration.md`` duplicates the Pydantic models in prose tables —
the one place in the docs that can silently rot as fields are added. This guard makes
that drift loud: every field of every model that
:func:`chemrefine.introspect.schema_document` exposes must appear (as a whole word)
somewhere in the page. The reverse direction — the page naming a field the schema lost —
is already covered by ``test_docs_examples``, which validates the page's YAML fences
against the real models.

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

_DOC = Path(__file__).resolve().parent.parent / "docs" / "user-guide" / "configuration.md"

# A guard on the repository's prose, not on the package: the sdist ships the suite so a
# distro packager can run it, but `docs/` (14 MB of site assets) deliberately does not
# ship — so with no page to check, the guard has nothing to say. In the repo the file
# always exists and the guard always runs.
pytestmark = pytest.mark.skipif(
    not _DOC.is_file(), reason="docs/ is a repository artifact and does not ship in the sdist"
)


def _documented_universe() -> dict[str, set[str]]:
    """Every field name the reference must mention, grouped by the model owning it."""
    universe: dict[str, set[str]] = {
        "Config": set(Config.model_fields),
        "StepConfig": set(StepConfig.model_fields),
        "BoltzmannSample": set(BoltzmannSample.model_fields),
        "MinSample": set(MinSample.model_fields),
        "MaxSample": set(MaxSample.model_fields),
        "NmsOptions": set(NmsOptions.model_fields),
    }
    for name in sorted(ENGINES):
        engine = get_engine(name)
        if isinstance(engine, OptionsDeclaring):
            universe[f"options[{name}]"] = set(engine.options_cls.model_fields)
    return universe


@pytest.mark.parametrize(
    ("model", "fields"),
    sorted(_documented_universe().items()),
    ids=lambda part: part if isinstance(part, str) else "",
)
def test_every_schema_field_is_named_in_the_configuration_page(model: str, fields: set[str]):
    text = _DOC.read_text(encoding="utf-8")
    missing = sorted(field for field in fields if not re.search(rf"\b{re.escape(field)}\b", text))
    assert not missing, (
        f"{model} field(s) {missing} are not mentioned in {_DOC.name}; the schema moved "
        "and the hand-written reference did not. Document them (or generate the page)."
    )
