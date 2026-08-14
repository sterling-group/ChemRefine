"""Drift guard: the hand-written configuration reference must name every schema field.

``docs/user-guide/configuration.md`` duplicates the Pydantic models in prose tables —
the one place in the docs that can silently rot as fields are added. Generating the page
from the schema is deliberate future work; until then this guard makes the drift loud:
every field of every model that :func:`chemrefine.introspect.schema_document` exposes
must appear (as a whole word) somewhere in the page. The reverse direction — the page
naming a field the schema lost — is already covered by ``test_docs_examples``, which
validates the page's YAML fences against the real models.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from chemrefine.config import BoltzmannSample, Config, MaxSample, MinSample, StepConfig
from chemrefine.engines.api import ENGINES, OptionsDeclaring, get_engine
from chemrefine.nms import NmsOptions

_DOC = Path(__file__).resolve().parent.parent / "docs" / "user-guide" / "configuration.md"


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
    missing = sorted(
        field for field in fields if not re.search(rf"\b{re.escape(field)}\b", text)
    )
    assert not missing, (
        f"{model} field(s) {missing} are not mentioned in {_DOC.name}; the schema moved "
        "and the hand-written reference did not. Document them (or generate the page)."
    )
