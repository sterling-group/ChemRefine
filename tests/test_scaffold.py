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


def test_an_override_in_a_subdirectory_is_scaffolded_not_a_traceback(tmp_path: Path):
    """``template: sub/custom.inp`` is a documented shape; the writer makes its parents.

    Only ``template_dir`` itself used to be created, so a subdirectory override raised a
    raw ``FileNotFoundError`` — past the CLI's ``ChemRefineError`` handler — while
    ``agent_tools.write_template`` created parents for the very same plan path.
    """
    config = _config(tmp_path, {"step": 1, "engine": "orca", "template": "sub/custom.inp"})
    written = scaffold_templates(config)
    assert config.template_dir / "sub" / "custom.inp" in written
    assert (config.template_dir / "sub" / "custom.inp").is_file()


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


# ---------------------------------------------------------------------------
# Starters must survive their own engine's input writer
# ---------------------------------------------------------------------------


def test_the_template_driven_starters_render_the_real_geometry(tmp_path: Path):
    """Each ``.inp``/``.in`` starter, rendered by its engine's writer, carries the seed.

    The starter and the writer live in different modules and had never met in a test —
    which is how the Q-Chem starter's comment, merely *mentioning* its coordinate block,
    made the writer splice the geometry into the comment and leave job 1 running the
    starter's placeholder atom. Rendering every template-driven starter through the real
    writer pins the pair: the seed's coordinates must land, and the placeholder must go.
    """
    from chemrefine.engines.orca import input as orca_input
    from chemrefine.engines.qchem import input as qchem_input
    from chemrefine.io import write_single_xyz
    from chemrefine.scaffold import _STEP_STARTERS

    xyz = write_single_xyz(
        [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.96)], tmp_path / "step1_0_inp.xyz"
    )

    qchem_tpl = tmp_path / "step1.in"
    qchem_tpl.write_text(_STEP_STARTERS["qchem"], encoding="utf-8")
    rendered = qchem_input.build_input(
        xyz_path=xyz,
        template_path=qchem_tpl,
        output_path=tmp_path / "q" / "step1_0.in",
        charge=0,
        multiplicity=1,
    ).read_text(encoding="utf-8")
    body = rendered.split("$end", 1)[1]  # everything after the comment block
    assert "O  0.000000 0.000000 0.000000" in body
    assert "H 0.0 0.0 0.0" not in rendered  # the starter's placeholder atom is gone

    orca_tpl = tmp_path / "step1.inp"
    orca_tpl.write_text(_STEP_STARTERS["orca"], encoding="utf-8")
    rendered = orca_input.build_input(
        xyz_path=xyz,
        template_path=orca_tpl,
        output_path=tmp_path / "o" / "step1_0.inp",
        charge=0,
        multiplicity=1,
    ).read_text(encoding="utf-8")
    assert f"* xyzfile 0 1 {xyz}" in rendered


def test_an_unwritable_template_dir_is_a_config_error_not_a_traceback(tmp_path: Path):
    """A disk refusal carries the documented exit code, wherever the caller sits.

    This seam serves the CLI, the GUI and the MCP tools, and all three promise failures
    the taxonomy names: the CLI catches only ``ChemRefineError``, and the GUI's handler
    re-raises anything else as a 500 with a logged traceback — for what is an ordinary
    read-only tree or an exhausted quota (ENOSPC on HPC scratch is the everyday case).
    """
    import os

    import pytest

    from chemrefine.errors import ConfigError

    if os.geteuid() == 0:
        pytest.skip("root writes everywhere; the permission wall cannot be built")
    fortress = tmp_path / "fortress"
    fortress.mkdir()
    fortress.chmod(0o555)
    path = tmp_path / "input.yaml"
    path.write_text(
        yaml.safe_dump(
            {"template_dir": str(fortress / "templates"), "steps": [{"step": 1, "engine": "orca"}]}
        ),
        encoding="utf-8",
    )
    try:
        with pytest.raises(ConfigError, match="cannot scaffold"):
            scaffold_templates(load_config(path))
    finally:
        fortress.chmod(0o755)


def test_steps_sharing_a_template_scaffold_it_once(tmp_path: Path):
    """Same engine, one shared ``template:`` — every step keeps a plan, the file is written once.

    Each sharing step needs a plan of its own (per-step lookups like ``read_template``
    answer from ``plan.step``), but the *file* must land once: ``exists`` is snapshotted
    before any write, so writing per sharing step re-did the same work at best.
    """
    config = _config(
        tmp_path,
        {"step": 1, "engine": "orca", "template": "shared.inp"},
        {"step": 2, "engine": "orca", "template": "shared.inp"},
    )
    step_plans = [p for p in plan_templates(config) if p.kind == "step"]
    assert [(p.step, p.path.name) for p in step_plans] == [(1, "shared.inp"), (2, "shared.inp")]
    written = scaffold_templates(config)
    assert [p.name for p in written if p.name.endswith(".inp")] == ["shared.inp"]


def test_a_cross_engine_template_share_is_refused_at_planning(tmp_path: Path):
    """Engines of different formats naming one template file refuse by name, never clobber.

    Both plans snapshot ``exists=False`` before either writes, so with ``overwrite``
    still False the second engine's starter replaced the first — against the docstring's
    "existing files are left alone". Refusing in ``plan_templates`` puts the refusal at
    the seam every consumer (CLI, GUI chips, agent tools) reads through.
    """
    import pytest

    from chemrefine.errors import ConfigError

    config = _config(
        tmp_path,
        {"step": 1, "engine": "orca", "template": "shared.tpl"},
        {"step": 2, "engine": "qchem", "template": "shared.tpl"},
    )
    with pytest.raises(ConfigError, match=r"shared\.tpl"):
        plan_templates(config)


def test_a_mid_scaffold_failure_names_what_already_landed(tmp_path: Path, monkeypatch):
    """A disk refusal partway through says which starters are already on disk.

    The failure leaves the earlier starters behind, and a retry reads a half-written
    last file as "exists — kept"; naming what landed is what makes that state
    inspectable rather than invisible.
    """
    import pytest

    from chemrefine.errors import ConfigError

    config = _config(
        tmp_path,
        {"step": 1, "engine": "orca"},
        {"step": 2, "engine": "qchem"},
    )
    real_write = Path.write_text

    def fail_on_qchem(self: Path, *args: object, **kwargs: object):
        if self.name == "step2.in":
            raise OSError(28, "No space left on device")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_on_qchem)
    with pytest.raises(ConfigError, match=r"already written before the failure: .*step1\.inp"):
        scaffold_templates(config)
