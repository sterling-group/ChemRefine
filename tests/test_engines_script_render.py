"""Tests for the engine-neutral Python-script template renderer."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from chemrefine.engines._script.render import build_input


def test_renderer_harvests_engine_neutral_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any script engine can persist JSON-compatible diagnostics without plugin coupling."""
    template = tmp_path / "step1.py"
    template.write_text(
        'energy_hartree = -1.25\nengine_metadata = {"provider": "demo", "evaluations": 3}\n',
        encoding="utf-8",
    )
    rendered = tmp_path / "rendered.py"
    output = tmp_path / "step1_0.json"

    build_input(
        xyz_path=tmp_path / "step1_0_inp.xyz",
        template_path=template,
        output_path=rendered,
        output_json_path=output,
        charge=0,
        multiplicity=1,
    )
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(rendered))

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "energy_hartree": -1.25,
        "engine_metadata": {"provider": "demo", "evaluations": 3},
    }
