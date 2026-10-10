"""Scaffolding: the plan mirrors what a run would read, starters fill only the gaps.

The plan must list exactly the files the run's own resolution would open — step
templates via :func:`chemrefine.ids.step_template_path`, headers via
:func:`chemrefine.validate.effective_header` — and scaffolding must never touch a file
the user already edited unless told to. Starter bodies are pinned by their load-bearing
line (the ``%pal`` an ORCA input needs, the ``$molecule`` block Q-Chem replaces), not by
full text, so wording can evolve without rewriting these tests.
"""

from __future__ import annotations

import ast
from pathlib import Path
from string import Template
from typing import Literal

import pytest
import yaml

from chemrefine import io
from chemrefine.config import Config, load_config
from chemrefine.engines._script.engine import ScriptEngine
from chemrefine.errors import EngineNotFoundError, WriteRefusedError
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


def test_an_unknown_engine_is_refused_at_planning_time(tmp_path: Path):
    """The plan resolves engines exactly as a run would, so an unknown one fails here.

    Every consumer of this seam — the CLI, the GUI's chips, the agent tools — inherits
    the refusal before a byte is written, rather than a starter file for an engine that
    does not exist.
    """
    with pytest.raises(EngineNotFoundError, match="unknown engine 'nope'"):
        plan_templates(_config(tmp_path, {"step": 1, "engine": "nope"}))


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
        {"step": 6, "engine": "mlip-extopt"},
        {"step": 7, "engine": "pyscf-extopt"},
    )
    scaffold_templates(config)

    def read(name: str) -> str:
        return (config.template_dir / name).read_text(encoding="utf-8")

    assert "%pal" in read("step1.inp")
    assert "$molecule" in read("step2.in")
    assert "MlipCalculator" in read("step3.py")
    assert "converged = mlip.last_converged" in read("step3.py")
    assert "pyscf" in read("step4.py")
    assert "converged = bool(mf.converged)" in read("step4.py")
    assert "NOT runnable" in read("step5.yaml")
    for extopt in ("step6.inp", "step7.inp"):
        # ORCA computes a step itself unless its `!` line names ExtOpt.
        assert "! ExtOpt Opt" in read(extopt)
        assert "B3LYP" not in read(extopt)


# ---------------------------------------------------------------------------
# Starters come from the engine
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


def test_a_drop_in_engine_declares_its_own_starter():
    """``template_starter`` on the engine is what ``chemrefine scaffold`` writes for it.

    Keyed by name in ``scaffold.py``, a starter was one more central list an engine had to
    be added to — and a drop-in got a suffix-shaped guess however good its own example was.
    Declared on the class it travels with the engine, and is the only source: registration
    refuses a template engine that declares none. The contract comment is still filled in
    here, so a script engine's starter names its own fields.
    """
    from chemrefine.engines.api import ENGINES, register

    @register("starter-probe")
    class _Probe:
        name = "starter-probe"
        label = "Probe"
        template_suffix = "toml"
        template_starter = "# probe starter\n$OUTPUT_CONTRACT[probe]\n"

        def prepare(self, ctx):
            """Unused: the probe exists to be looked up, not run."""

        def submit(self, inputs, ctx):
            """Unused: see ``prepare``."""

        def parse(self, inputs, ctx):
            """Unused: see ``prepare``."""

    try:
        starter = _starter_for(_plan("step1.toml", "starter-probe"))
    finally:
        ENGINES.pop("starter-probe", None)

    assert starter == "# probe starter\n[probe]\n"


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
    from chemrefine.engines.orca import inspect
    from chemrefine.engines.orca.engine import OrcaEngine
    from chemrefine.engines.orca.extopt.engine import ExtOptOrcaEngine
    from chemrefine.engines.qchem import input as qchem_input
    from chemrefine.engines.qchem.engine import QchemEngine
    from chemrefine.ids import require_template
    from chemrefine.io import write_single_xyz

    xyz = write_single_xyz(
        [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.96)], tmp_path / "step1_0_inp.xyz"
    )

    qchem_tpl = tmp_path / "step1.in"
    qchem_tpl.write_text(QchemEngine.template_starter, encoding="utf-8")
    rendered = qchem_input.build_input(
        xyz_path=xyz,
        template=require_template(qchem_tpl, label="Q-Chem"),
        output_path=tmp_path / "q" / "step1_0.in",
        charge=0,
        multiplicity=1,
    ).read_text(encoding="utf-8")
    body = rendered.split("$end", 1)[1]  # everything after the comment block
    assert "O  0.000000 0.000000 0.000000" in body
    assert "H 0.0 0.0 0.0" not in rendered  # the starter's placeholder atom is gone

    orca_tpl = tmp_path / "step1.inp"
    orca_tpl.write_text(OrcaEngine.template_starter, encoding="utf-8")
    rendered = orca_input.build_input(
        xyz_path=xyz,
        template=require_template(orca_tpl, label="ORCA"),
        output_path=tmp_path / "o" / "step1_0.inp",
        charge=0,
        multiplicity=1,
    ).read_text(encoding="utf-8")
    assert f"* xyzfile 0 1 {xyz}" in rendered

    extopt_tpl = tmp_path / "extopt.inp"
    extopt_tpl.write_text(ExtOptOrcaEngine.template_starter, encoding="utf-8")
    rendered = orca_input.build_input(
        xyz_path=xyz,
        template=require_template(extopt_tpl, label="ORCA"),
        output_path=tmp_path / "e" / "step1_0.inp",
        charge=0,
        multiplicity=1,
        extra_blocks='%method\n  ProgExt "/w.sh"\nend',
    ).read_text(encoding="utf-8")
    assert "! ExtOpt Opt" in rendered
    assert f"* xyzfile 0 1 {xyz}" in rendered
    assert inspect.inspect_template(ExtOptOrcaEngine.template_starter) == inspect.OrcaInputInfo(
        operation="opt_sp", is_ts=False, has_freq=False, has_extopt=True, pal=4, maxcore=None
    )


def _hands_the_species_to_its_code(template: str) -> bool:
    """Whether a step's charge and multiplicity reach what a script template *runs*.

    Rendered with two numbers nothing else would put there, both must be constants of the
    parsed program. A placeholder named in a comment is no part of it, and neither is one
    inside a string.
    """
    charge, multiplicity = 97, 89
    values = {"CHARGE": charge, "MULTIPLICITY": multiplicity}
    script = Template(template)
    rendered = script.safe_substitute({n: values.get(n, 0) for n in script.get_identifiers()})
    constants = [n.value for n in ast.walk(ast.parse(rendered)) if isinstance(n, ast.Constant)]
    return charge in constants and multiplicity in constants


def test_every_script_template_chemrefine_ships_hands_the_species_to_its_code(
    tmp_path: Path, shipped_engines: dict[str, type]
):
    """A starter, an example or a live case that left them out would teach leaving them out.

    A user's script may set its own charge and spin, and nothing asks it not to. One of ours
    that does not pass the step's on runs every charged or open-shell step as whatever its
    library assumes, the result filed under the species the config declares. Asked of the
    code, not the text: an example named both placeholders in its opening comment and
    handed neither to its calculator.
    """
    repo = Path(__file__).resolve().parents[1]
    scripted = sorted(n for n, cls in shipped_engines.items() if issubclass(cls, ScriptEngine))
    config = _config(tmp_path, *({"step": i, "engine": n} for i, n in enumerate(scripted, 1)))
    starters = [path for path in scaffold_templates(config) if path.suffix == ".py"]
    assert len(starters) == len(scripted) > 0, "one starter for each script engine"
    shipped = {
        **{f"the {name} starter": path for name, path in zip(scripted, starters, strict=True)},
        **{
            path.relative_to(repo).as_posix(): path
            for home in ("examples", "tests/data/e2e/cases")
            for path in sorted((repo / home).rglob("templates/*.py"))
        },
    }
    assert len(shipped) > len(starters), "the examples and the live cases are found"

    without = [
        name
        for name, path in shipped.items()
        if not _hands_the_species_to_its_code(path.read_text(encoding="utf-8"))
    ]
    assert without == []


def test_a_template_dir_the_filesystem_refuses_is_a_write_refusal_naming_it(tmp_path: Path):
    """A ``template_dir`` that cannot be created is a refused write, wherever the caller sits.

    This seam serves the CLI, the GUI and the MCP tools, and all three promise failures
    the taxonomy names. A read-only tree or an exhausted quota (ENOSPC on HPC scratch is
    the everyday case) is the machine's to fix, so it carries the refused write's class and
    code and names the directory — never the config's, whose reader would look for a typo.
    """
    import os

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
        with pytest.raises(WriteRefusedError) as refusal:
            scaffold_templates(load_config(path))
    finally:
        fortress.chmod(0o755)
    assert type(refusal.value) is WriteRefusedError
    assert str(refusal.value) == f"cannot write {fortress / 'templates'}: Permission denied"


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


def test_a_starter_that_fails_to_write_leaves_the_edited_template_intact(
    tmp_path: Path, monkeypatch
):
    """With ``overwrite``, a torn write must not leave a stub where an edited template was.

    ``write_text`` truncates before it writes, so a kill or a full disk between the two
    left an empty ``step1.inp`` where the user's file had been. Each starter is written
    whole or not at all now — the rule the template editor and the config writer already
    follow — pinned by failing the write at its last step and finding the old text there.
    """
    import os

    config = _config(tmp_path, {"step": 1, "engine": "orca"})
    template = tmp_path / "templates" / "step1.inp"
    template.parent.mkdir()
    template.write_text("! Mine\n", encoding="utf-8")

    def refuse(_fd: int) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr(os, "fsync", refuse)
    with pytest.raises(WriteRefusedError, match="No space left on device"):
        scaffold_templates(config, overwrite=True)
    assert template.read_text(encoding="utf-8") == "! Mine\n"
    assert not list(template.parent.glob(".tmp_*"))


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


def test_a_scaffold_refused_partway_names_the_starter_and_a_second_run_writes_the_rest(
    tmp_path: Path, monkeypatch
):
    """A disk refusal partway through names the starter it stopped at, and loses nothing.

    Starters land in plan order, each whole or not at all, so the named file is where the
    plan stopped: the ones before it are on disk, it is not, and the same command again
    keeps the first and writes the rest.
    """
    config = _config(
        tmp_path,
        {"step": 1, "engine": "orca"},
        {"step": 2, "engine": "qchem"},
    )
    templates = tmp_path / "templates"
    real_write = io.atomic_write

    def full_at_the_second(path: Path, data: bytes, *, mode: int | None = None) -> None:
        if path.name == "step2.in":
            raise OSError(28, "No space left on device")
        real_write(path, data, mode=mode)

    with monkeypatch.context() as full_disk:
        full_disk.setattr(io, "atomic_write", full_at_the_second)
        with pytest.raises(WriteRefusedError) as refusal:
            scaffold_templates(config)
    assert str(refusal.value) == f"cannot write {templates / 'step2.in'}: No space left on device"
    assert [p.name for p in templates.iterdir()] == ["step1.inp"]
    assert [p.name for p in scaffold_templates(config)] == ["step2.in", "cpu.slurm.header"]


def test_a_template_directory_behind_a_wall_is_the_writes_refusal_not_the_plans(
    tmp_path: Path, unsearchable
):
    """The plan asks whether each starter is there, and the write that follows answers.

    Under a directory this account cannot search, ``Path.is_file`` raises
    ``PermissionError`` on Python 3.11 to 3.13 — the plan itself was a traceback — and
    answers ``False`` on 3.14. A starter that cannot be seen is planned as missing on
    every Python, and the write names what the filesystem refuses.
    """
    private = tmp_path / "private"
    templates = private / "templates"
    templates.mkdir(parents=True)
    path = tmp_path / "input.yaml"
    path.write_text(
        yaml.safe_dump({"template_dir": str(templates), "steps": [{"step": 1, "engine": "orca"}]}),
        encoding="utf-8",
    )
    config = load_config(path)
    unsearchable(private)

    assert [(plan.path.name, plan.exists) for plan in plan_templates(config)] == [
        ("step1.inp", False),
        ("cpu.slurm.header", False),
    ]
    with pytest.raises(WriteRefusedError, match=r"^cannot write .*: Permission denied$"):
        scaffold_templates(config)
