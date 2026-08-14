"""Scaffolding: the plan mirrors what a run would read, starters fill only the gaps.

The plan must list exactly the files the run's own resolution would open — step
templates via :func:`chemrefine.ids.step_template_path`, headers via
:func:`chemrefine.validate.effective_header` — and scaffolding must never touch a file
the user already edited unless told to. Starter bodies are pinned by their load-bearing
line (the ``%pal`` an ORCA input needs, the ``$molecule`` block Q-Chem replaces), not by
full text, so wording can evolve without rewriting these tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from chemrefine.config import Config, load_config
from chemrefine.scaffold import TemplatePlan, _starter_for, plan_templates, scaffold_templates


def _config(tmp_path: Path, *steps: dict[str, object]) -> Config:
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump({"steps": list(steps)}), encoding="utf-8")
    return load_config(path)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def test_the_plan_lists_step_templates_and_the_shared_header(tmp_path: Path):
    config = _config(
        tmp_path,
        {"step": 1, "engine": "orca"},
        {"step": 2, "engine": "qchem"},
    )
    plans = plan_templates(config)
    names = [(p.kind, p.path.name, p.step, p.engine) for p in plans]
    assert names == [
        ("step", "step1.inp", 1, "orca"),
        ("step", "step2.in", 2, "qchem"),
        ("slurm-header", "cpu.slurm.header", None, None),
    ]
    assert not any(p.exists for p in plans)


def test_a_template_free_engine_plans_nothing(tmp_path: Path):
    """The fake engine reads no template and no header — an empty plan, not a stub."""
    assert plan_templates(_config(tmp_path, {"step": 1, "engine": "fake"})) == ()


def test_a_gpu_step_plans_the_cuda_header(tmp_path: Path):
    config = _config(tmp_path, {"step": 1, "engine": "mlip", "options": {"device": "cuda"}})
    headers = [p.path.name for p in plan_templates(config) if p.kind == "slurm-header"]
    assert headers == ["cuda.slurm.header"]


def test_a_step_template_override_is_planned_under_its_own_name(tmp_path: Path):
    config = _config(tmp_path, {"step": 1, "engine": "orca", "template": "custom.inp"})
    [step_plan] = [p for p in plan_templates(config) if p.kind == "step"]
    assert step_plan.path.name == "custom.inp"


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------


def test_scaffold_fills_every_gap_and_is_idempotent(tmp_path: Path):
    config = _config(tmp_path, {"step": 1, "engine": "orca"})
    written = scaffold_templates(config)
    assert [p.name for p in written] == ["step1.inp", "cpu.slurm.header"]
    assert all(p.is_file() for p in written)
    assert "%pal" in (config.template_dir / "step1.inp").read_text(encoding="utf-8")
    assert scaffold_templates(config) == ()  # nothing left to write


def test_scaffold_keeps_an_edited_file_unless_told_otherwise(tmp_path: Path):
    config = _config(tmp_path, {"step": 1, "engine": "orca"})
    config.template_dir.mkdir()
    template = config.template_dir / "step1.inp"
    template.write_text("! MyEditedKeywords\n", encoding="utf-8")

    written = scaffold_templates(config)
    assert template not in written
    assert template.read_text(encoding="utf-8") == "! MyEditedKeywords\n"

    written = scaffold_templates(config, overwrite=True)
    assert template in written
    assert "%pal" in template.read_text(encoding="utf-8")


def test_each_bundled_engine_gets_its_own_starter(tmp_path: Path):
    """Pinned by the load-bearing line each engine's format cannot do without."""
    config = _config(
        tmp_path,
        {"step": 1, "engine": "orca"},
        {"step": 2, "engine": "qchem"},
        {"step": 3, "engine": "mlip"},
        {"step": 4, "engine": "pyscf", "options": {"basis": "def2-svp", "xc": "pbe"}},
        {"step": 5, "engine": "mlip-train", "options": {"task_name": "mace_off", "device": "cpu"}},
    )
    scaffold_templates(config)

    def read(name: str) -> str:
        return (config.template_dir / name).read_text(encoding="utf-8")

    assert "%pal" in read("step1.inp")
    assert "$molecule" in read("step2.in")
    assert "MlipCalculator" in read("step3.py")
    assert "pyscf" in read("step4.py")
    assert "NOT runnable" in read("step5.yaml")


# ---------------------------------------------------------------------------
# Starter fallbacks (third-party engine shapes the bundled set cannot exercise)
# ---------------------------------------------------------------------------


def _plan(
    name: str, engine: str | None, kind: Literal["step", "slurm-header"] = "step"
) -> TemplatePlan:
    return TemplatePlan(
        path=Path(f"templates/{name}"),
        exists=False,
        kind=kind,
        step=1 if kind == "step" else None,
        engine=engine,
    )


def test_unknown_engines_fall_back_by_suffix_then_generically():
    assert "$XYZ_PATH" in _starter_for(_plan("step1.py", "somebody-elses-engine"))
    assert "%pal" in _starter_for(_plan("step1.inp", "somebody-elses-orca"))
    assert "ChemRefine step template" in _starter_for(_plan("step1.toml", "exotic"))
    assert "ChemRefine step template" in _starter_for(_plan("step1.toml", None))


def test_header_starters_cover_cuda_and_everything_else():
    cuda = _starter_for(_plan("cuda.slurm.header", None, kind="slurm-header"))
    assert "--gres=gpu:1" in cuda
    custom = _starter_for(_plan("special.header", None, kind="slurm-header"))
    assert custom.startswith("#!/bin/bash")
    assert "--gres" not in custom
