"""Tests for ``QchemEngine`` — run block, layout, memory, NMS hook, and the submission proof."""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms

from chemrefine import pipeline
from chemrefine.config import Config, StepConfig
from chemrefine.engines.api import NmsCapableEngine, get_engine
from chemrefine.errors import ConfigError
from chemrefine.io import write_single_xyz
from chemrefine.state import PipelineState, StepContext, Structure

DATA = Path(__file__).resolve().parent / "data" / "engines" / "qchem"

_SP_TEMPLATE = "$rem\n  jobtype sp\n$end\n"


def _ctx(
    tmp_path: Path,
    *,
    options: dict[str, object] | None = None,
    executables: dict[str, str] | None = None,
    max_cores: int = 8,
    template: str = _SP_TEMPLATE,
) -> StepContext:
    """A StepContext with a real template on disk, shaped like the ORCA test helper's."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "step1.in").write_text(template, encoding="utf-8")
    return StepContext(
        step_cfg=StepConfig(step=1, engine="qchem", options=options or {}),
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=template_dir,
        template=template_dir / "step1.in",
        scratch_dir=None,
        prev_state=PipelineState(
            structures=(Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])),)
        ),
        charge=0,
        multiplicity=1,
        max_cores=max_cores,
        slurm_template="cpu.slurm.header",
        executables=executables or {},
    )


def _body(tmp_path: Path, **kwargs):
    """The run block body for a step shaped by ``kwargs`` (passed to ``_ctx``)."""
    ctx = _ctx(tmp_path, **kwargs)
    return get_engine("qchem").run_block(ctx, Path("step1_0.in"), Path("step1_0.out")).body


# ---------------------------------------------------------------------------
# The run block
# ---------------------------------------------------------------------------


def test_the_default_run_block_is_serial_threads(tmp_path: Path):
    """No options: bare ``qchem -nt 1``, QCSCRATCH on the work dir, savename derived."""
    body = _body(tmp_path)
    assert 'export QCSCRATCH="$WORK_DIR"' in body
    assert "export OMP_NUM_THREADS=1" in body
    assert "export QC_THREADS=1" in body
    assert 'QCSAVE="step1_0.in"; QCSAVE="${QCSAVE%.*}"' in body
    assert 'qchem -nt 1 step1_0.in "$OUTPUT_DIR/step1_0.out" "$QCSAVE"' in body
    assert "QC=" not in body.replace("QCSCRATCH", "").replace("QC_THREADS", "").replace(
        "QCSAVE", ""
    ), "no QC/QCAUX exports without the executables keys"


def test_cores_clamp_to_the_budget_in_every_spelling(tmp_path: Path):
    """``-nt``, both thread exports and the layout all carry min(cores, max_cores)."""
    body = _body(tmp_path, options={"cores": 16}, max_cores=8)
    assert "-nt 8 " in body
    assert "export OMP_NUM_THREADS=8" in body and "export QC_THREADS=8" in body
    assert get_engine("qchem").slurm_layout(_ctx(tmp_path, options={"cores": 16})) == (1, 8)


def test_qc_root_exports_env_and_derives_the_default_qcaux(tmp_path: Path):
    """``qc`` alone exports QC, PATH, and the manual's own ``$QC/qcaux`` default, visibly."""
    body = _body(tmp_path, executables={"qc": "/inst/qchem700"})
    assert "export QC=/inst/qchem700" in body
    assert 'export QCAUX="$QC/qcaux"' in body
    assert 'export PATH="$PATH:$QC/bin:$QC/bin/perl"' in body
    assert '"$QC/bin/qchem" -nt 1' in body


def test_an_explicit_qcaux_beats_the_derived_default(tmp_path: Path):
    """Sibling-layout installs (qcaux beside the QC root) name theirs and it wins."""
    body = _body(tmp_path, executables={"qc": "/inst/qchem700", "qcaux": "/inst/qcaux"})
    assert "export QCAUX=/inst/qcaux" in body
    assert '"$QC/qcaux"' not in body


def test_qcaux_alone_is_exported_without_inventing_qc(tmp_path: Path):
    """A module-provided qchem with an odd qcaux exports only what was configured."""
    body = _body(tmp_path, executables={"qcaux": "/inst/qcaux"})
    assert "export QCAUX=/inst/qcaux" in body
    assert "export QC=" not in body.replace("export QCAUX", "")
    assert body.splitlines()[-1].startswith("qchem ")


def test_an_explicit_executable_with_spaces_stays_one_word(tmp_path: Path):
    """The config-supplied path is shlex-quoted at the one place it is rendered."""
    body = _body(tmp_path, executables={"qchem": "/opt/my qchem/qchem"})
    assert "'/opt/my qchem/qchem' -nt 1" in body


def test_mpi_is_opt_in_and_emits_qqchems_flags(tmp_path: Path):
    """``nprocs`` set: ``-mpi -np P -nt N`` with threads, ``-mpi -np P`` without."""
    body = _body(tmp_path, options={"nprocs": 4, "cores": 2})
    assert "-mpi -np 4 -nt 2 " in body
    serial_threads = _body(tmp_path, options={"nprocs": 4})
    assert "-mpi -np 4 step1_0.in" in serial_threads and "-nt" not in serial_threads


def test_save_toggles_the_cleanup_copy(tmp_path: Path):
    """``save: true`` copies the savename dir home inside the script's one EXIT trap."""
    engine = get_engine("qchem")
    kept = engine.run_block(
        _ctx(tmp_path, options={"save": True}), Path("step1_0.in"), Path("step1_0.out")
    )
    assert kept.cleanup == 'cp -r "$QCSCRATCH/$QCSAVE" "$OUTPUT_DIR/" 2>/dev/null || true'
    assert engine.run_block(_ctx(tmp_path), Path("step1_0.in"), Path("step1_0.out")).cleanup == ""


# ---------------------------------------------------------------------------
# Layout, memory, NMS
# ---------------------------------------------------------------------------


def test_mpi_layout_is_ranks_by_threads(tmp_path: Path):
    """MPI spells ``(nprocs, cores)`` and charges the product against the budget."""
    engine = get_engine("qchem")
    ctx = _ctx(tmp_path, options={"nprocs": 4, "cores": 2})
    assert engine.slurm_layout(ctx) == (4, 2)
    assert engine.pal(ctx) == 8


def test_an_mpi_product_over_the_budget_is_refused(tmp_path: Path):
    """The factorization is the job's shape — over budget errors instead of reshaping."""
    from chemrefine.engines import _execution

    ctx = _ctx(tmp_path, options={"nprocs": 4, "cores": 4}, max_cores=8)
    (tmp_path / "templates" / "cpu.slurm.header").write_text("#!/bin/bash\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="16 cores, more than max_cores=8"):
        _execution._BatchPlan.of(get_engine("qchem"), ctx, local=True)


def test_memory_mb_is_the_templates_mem_total(tmp_path: Path):
    """``mem_total`` is the whole-job declaration; no headroom factor, per qqchem."""
    engine = get_engine("qchem")
    with_mem = _ctx(tmp_path, template="$rem\n  jobtype sp\n  mem_total 8000\n$end\n")
    assert engine.memory_mb(with_mem) == 8000
    assert engine.memory_mb(_ctx(tmp_path)) is None


def test_the_engine_is_nms_capable(tmp_path: Path):
    """The hook + JobEngine's artifact_paths satisfy the NMS protocol."""
    engine = get_engine("qchem")
    assert isinstance(engine, NmsCapableEngine)
    chained = _ctx(
        tmp_path,
        template=(
            "$rem\n  jobtype opt\n$end\n\n@@@\n\n$molecule\nread\n$end\n"
            "$rem\n  jobtype freq\n$end\n"
        ),
    )
    info = engine.nms_input_info(chained)
    assert info.computes_frequencies and not info.is_transition_state
    ts = engine.nms_input_info(_ctx(tmp_path, template="$rem\n  jobtype ts\n$end\n"))
    assert ts.is_transition_state


# ---------------------------------------------------------------------------
# The submission proof: the whole pipeline, a stub binary, no licence
# ---------------------------------------------------------------------------


def test_a_qchem_step_runs_end_to_end_through_local_dispatch(tmp_path: Path):
    """prepare → generated script → local background run → parse → cache, argv asserted.

    The stub stands where the real ``qchem`` wrapper would: it checks the argv shape the
    run block promises (``-nt N input output savename``) and writes the trimmed real
    fixture output to the outfile argument — so everything else in the chain is the real
    thing, including the throttler, the script's EXIT trap, and the cache write.
    """
    stub = tmp_path / "fake-qchem"
    stub.write_text(
        "#!/bin/bash\n"
        '[ "$1" = "-nt" ] || { echo "expected -nt, got: $*" >&2; exit 64; }\n'
        '[ "$2" = "2" ] || { echo "expected -nt 2, got: $*" >&2; exit 65; }\n'
        '[ -f "$3" ] || { echo "input $3 missing" >&2; exit 66; }\n'
        '[ -n "$5" ] || { echo "savename missing" >&2; exit 67; }\n'
        f'cp "{DATA / "sp" / "step1_0.out"}" "$4"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "step1.in").write_text(_SP_TEMPLATE, encoding="utf-8")
    (template_dir / "cpu.slurm.header").write_text("#!/bin/bash\n", encoding="utf-8")
    seed = write_single_xyz([("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74)], tmp_path / "input.xyz")
    config = Config(
        template_dir=template_dir,
        output_dir=tmp_path / "outputs",
        input=seed,
        max_cores=2,
        dispatch="local",
        executables={"qchem": str(stub)},
        steps=[StepConfig(step=1, engine="qchem", options={"cores": 2})],
    )

    outcomes = pipeline.run(config)

    assert len(outcomes) == 1 and not outcomes[0].cache_hit
    survivors = outcomes[0].state.structures
    assert [s.id for s in survivors] == ["0"]
    assert survivors[0].energy_hartree == -721.7792413729  # the fixture's last energy
    step_dir = config.output_dir / "step1"
    script = (step_dir / "0" / "step1_0.slurm").read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=1" in script and "#SBATCH --cpus-per-task=2" in script
    assert (step_dir / "_cache" / "step.json").is_file(), "the step must cache"
    assert (step_dir / "step1_ensemble.xyz").is_file()
    # A second run is a pure cache hit — no submission, same survivors.
    rerun = pipeline.run(config)
    assert rerun[0].cache_hit


# ---------------------------------------------------------------------------
# The operation vocabulary — declared, refused up front, resolved before parsing
# ---------------------------------------------------------------------------


def test_an_unknown_operation_is_refused_up_front():
    """A typo'd ``operation:`` fails at the run's t=0 walk, not after every job ran.

    ``operation`` only picks the parser, so left to run it costs the whole step and
    ledgers every output UNPARSEABLE.
    """
    engine = get_engine("qchem")
    bad = StepConfig(step=1, engine="qchem", operation="goat")
    with pytest.raises(ConfigError, match="unknown Q-Chem operation"):
        engine.check_step(bad, charge=0, multiplicity=1)


def test_known_and_absent_operations_pass_preflight():
    """Every declared spelling — either case, ``+`` or ``_`` — and ``None`` pass."""
    engine = get_engine("qchem")
    for operation in (None, "sp", "opt_sp", "OPT+SP", "freq"):
        engine.check_step(
            StepConfig(step=1, engine="qchem", operation=operation), charge=0, multiplicity=1
        )


def test_an_explicit_operation_beats_the_template(tmp_path: Path):
    """``operation:`` wins over inspection — explicit beats inferred, always."""
    engine = get_engine("qchem")
    ctx = _ctx(tmp_path, template="$rem\n  jobtype opt\n$end\n")
    assert engine._resolve_operation(ctx) == "opt_sp"  # inspected: opt -> opt_sp
    explicit = StepContext(
        step_cfg=StepConfig(step=1, engine="qchem", operation="sp"),
        step_dir=ctx.step_dir,
        template_dir=ctx.template_dir,
        template=ctx.template,
        scratch_dir=None,
        prev_state=ctx.prev_state,
        charge=0,
        multiplicity=1,
        max_cores=8,
        slurm_template="cpu.slurm.header",
    )
    assert engine._resolve_operation(explicit) == "sp"
