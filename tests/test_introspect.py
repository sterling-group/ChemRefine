"""The introspection document: produced from the validating models, never hand-written.

What these tests hold: :func:`chemrefine.introspect.schema_document` is the one view a
consumer that cannot import chemrefine (GUI forms, an agent via MCP) renders from, so it
must (a) describe *every* registered engine — including a drop-in one it has never heard
of, which the conftest-registered fake engine stands in for — (b) be internally
consistent (a capability and the field it implies may not disagree), and (c) serialize
to JSON whole, because every consumer receives it serialized.
"""

from __future__ import annotations

import dataclasses
import json

from chemrefine.config import Config
from chemrefine.engines.api import ENGINES
from chemrefine.introspect import EngineDescriptor, describe_engines, schema_document


def _by_name() -> dict[str, EngineDescriptor]:
    return {d.name: d for d in describe_engines()}


def test_every_registered_engine_is_described():
    """One descriptor per registry entry, sorted — the fake engine included.

    The fake engine is the drop-in third-party shape (three methods, no ClassVars, no
    options model); if it is describable, an engine this module has never imported is.
    """
    described = [d.name for d in describe_engines()]
    assert described == sorted(ENGINES)
    assert "fake" in described


def test_descriptors_are_internally_consistent():
    """A capability and the field it implies may not disagree.

    Template facts exist exactly for template-driven engines; backend extras exactly for
    provisionable ones; the capability vocabulary is closed. A descriptor violating one
    of these would make a GUI render a template picker for a template-free engine or an
    install hint for nothing.
    """
    for d in describe_engines():
        assert (d.template_suffix is not None) == d.template_driven
        assert (d.label is not None) == d.template_driven
        assert (d.backend_extras != ()) == ("provisionable" in d.capabilities)
        assert set(d.capabilities) <= {"artifact", "nms", "provisionable", "streaming"}
        assert list(d.capabilities) == sorted(d.capabilities)


def test_options_schema_tracks_the_declaring_capability():
    """``options_schema`` is the declared model's schema, or honestly absent.

    ORCA and the fake engine declare no model (see the invariants suite), so inventing a
    schema for them would advertise knobs — ``device`` — the engine never reads. A
    declared schema must carry the model's fields: ``mlip``'s ``model_name`` is pinned as
    the canary.
    """
    by_name = _by_name()
    assert by_name["orca"].options_schema is None
    assert by_name["fake"].options_schema is None
    mlip_schema = by_name["mlip"].options_schema
    assert mlip_schema is not None
    assert "model_name" in mlip_schema["properties"]


def test_orca_descriptor_pins_the_template_facts():
    """The GUI's "ORCA is configured via its template" hinges on these three fields."""
    orca = _by_name()["orca"]
    assert orca.template_driven
    assert orca.template_suffix == "inp"
    assert orca.options_schema is None


def test_operations_belong_to_the_family_that_interprets_them():
    """The ORCA family declares the parser dispatch's vocabulary; nobody else invents one.

    The GUI's dropdown renders exactly a descriptor's ``operations`` — an engine that
    treats the field as a free label (the script engines, qchem, the fake) must report
    an empty tuple, or the UI would offer ORCA's ensemble operations to an engine that
    would silently ignore them. The declaring set is pinned so a change is a decision:
    a new engine that grows a real ``operation:`` vocabulary declares
    ``OperationsDeclaring`` and extends this list.
    """
    from chemrefine.engines.orca.output.coordinator import known_operations

    by_name = _by_name()
    declaring = {name for name, d in by_name.items() if d.operations}
    assert sorted(declaring) == ["mlip-extopt", "orca", "pyscf-extopt"], (
        "the set of operation-declaring engines moved — extend this pin if that was a "
        "decision, and make sure the new vocabulary reaches the agent guide"
    )
    for name in declaring:
        assert by_name[name].operations == tuple(sorted(known_operations())), name


def test_schema_document_serializes_whole_and_carries_the_config_schema():
    """The document must round-trip JSON and contain what the loader validates with.

    ``config`` is :meth:`Config.model_json_schema` verbatim — same top-level keys, the
    step/sample models under ``$defs`` — and ``engines`` mirrors the registry. A version
    stamp lets a consumer cache against the installed package.
    """
    document = schema_document()
    round_tripped = json.loads(json.dumps(document))
    assert round_tripped["chemrefine_version"] == document["chemrefine_version"]
    assert document["config"] == Config.model_json_schema()
    assert "StepConfig" in document["config"]["$defs"]
    assert set(document["engines"]) == set(ENGINES)
    assert "target" in document["nms"]["properties"]
    fake = document["engines"]["fake"]
    assert fake == dataclasses.asdict(_by_name()["fake"])


def test_the_operation_vocabulary_is_served_canonical_and_sorted():
    """The config schema keeps ``operation`` a free string (engines interpret it), so
    the document's top-level list is the union of every descriptor's declared
    vocabulary — a future declarer's operations join with no edit here — served
    canonical and sorted, the legacy ``dft`` excluded."""
    operations = schema_document()["operations"]
    assert operations == sorted(operations)
    assert set(operations) == {op for d in describe_engines() for op in d.operations}
    assert set(operations) >= {"opt_sp", "sp", "freq", "pes", "goat", "docker", "solvator"}
    assert "dft" not in operations
