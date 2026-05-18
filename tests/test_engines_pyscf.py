"""Tests for the PySCF engine package (ported from the unmerged ``origin/pyscf`` PR)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines.base import ENGINES, get_engine
from chemrefine.state import PipelineState, StepContext, Structure

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_pyscf_engines_are_registered():
    assert "pyscf" in ENGINES
    assert "pyscf-direct" in ENGINES


def test_pyscf_engine_supports_nms_is_false():
    assert get_engine("pyscf").supports_nms is False


# ---------------------------------------------------------------------------
# PyscfEngine — ORCA-driven mode (verified, no real PySCF needed)
# ---------------------------------------------------------------------------


def _pyscf_ctx(tmp_path: Path, **option_overrides) -> StepContext:
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "step1.inp").write_text("! HF def2-SVP\n", encoding="utf-8")
    (template_dir / "cpu.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --partition=normal\n", encoding="utf-8"
    )
    options = {"method": "dft", "xc": "pbe", "basis": "def2-svp"}
    options.update(option_overrides)
    step_cfg = StepConfig(
        step=1,
        engine="pyscf",
        operation="opt_sp",
        options=options,
    )
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=template_dir,
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(structures=(seed,)),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        orca_executable="orca",
    )


def test_pyscf_extra_blocks_carries_method_settings(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path, basis="cc-pvdz", xc="b3lyp")
    extra = engine._extra_blocks(ctx)
    assert "%method" in extra
    assert "ProgExt" in extra
    assert "pyscf_extopt.sh" in extra
    assert "--basis cc-pvdz" in extra
    assert "--xc b3lyp" in extra


def test_pyscf_extra_blocks_default_bind(tmp_path: Path):
    """Default bind for PySCF should be 127.0.0.1:8889 (distinct from MLFF)."""
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path)
    assert "127.0.0.1:8889" in engine._extra_blocks(ctx)


def test_pyscf_extra_blocks_includes_gpu_and_df_flags(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path, df=True, gpu=True)
    extra = engine._extra_blocks(ctx)
    assert "--df" in extra
    assert "--gpu" in extra


def test_pyscf_run_block_starts_pyscf_server(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path)
    run_block = engine._run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "python -m chemrefine.engines.pyscf.server" in run_block
    assert "--default-method dft" in run_block
    assert "--default-xc pbe" in run_block
    assert "--default-basis def2-svp" in run_block
    assert "SERVER_PID=$!" in run_block
    assert "kill $SERVER_PID" in run_block


def test_pyscf_run_block_emits_gpu_flag_when_set(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path, gpu=True, df=True)
    run_block = engine._run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "--default-df" in run_block
    assert "--default-gpu" in run_block


def test_pyscf_prepare_writes_inp_with_method_block(tmp_path: Path):
    engine = get_engine("pyscf")
    ctx = _pyscf_ctx(tmp_path)
    inputs = engine.prepare(ctx)
    inp_text = inputs.files[0][0].read_text()
    assert "%method" in inp_text
    assert "ProgExt" in inp_text
    assert "pyscf_extopt.sh" in inp_text


# ---------------------------------------------------------------------------
# Placeholders raise NotImplementedError with TODO hint
# ---------------------------------------------------------------------------


def test_pyscf_direct_prepare_raises_with_todo():
    engine = get_engine("pyscf-direct")
    with pytest.raises(NotImplementedError):
        engine.prepare(ctx=None)  # type: ignore[arg-type]


def test_pyscf_server_main_still_placeholder():
    """Only the model-loading + ``waitress.serve`` glue stays a TODO."""
    from chemrefine.engines.pyscf.server import main as server_main

    with pytest.raises(NotImplementedError):
        server_main()


def test_pyscf_client_main_still_placeholder():
    """The ExtOpt wrapper that reads ``.extinp.tmp`` is the remaining TODO."""
    from chemrefine.engines.pyscf.client import main as client_main

    with pytest.raises(NotImplementedError):
        client_main()


# ---------------------------------------------------------------------------
# Client CLI + HTTP RPC — verified
# ---------------------------------------------------------------------------


def test_pyscf_client_parse_args_defaults():
    from chemrefine.engines.pyscf.client import parse_args

    args = parse_args(["job.extinp.tmp"])
    assert args.bind == "127.0.0.1:8889"
    assert args.method == "dft"
    assert args.xc == "pbe"
    assert args.basis == "def2-svp"
    assert args.df is False
    assert args.gpu is False
    assert args.inputfile == "job.extinp.tmp"


def test_pyscf_client_settings_from_args_round_trip():
    from chemrefine.engines.pyscf.client import parse_args, settings_from_args

    args = parse_args(
        ["--method", "hf", "--basis", "cc-pvdz", "--xc", "b3lyp", "--df", "--gpu", "f"]
    )
    settings = settings_from_args(args)
    assert settings == {
        "method": "hf",
        "xc": "b3lyp",
        "basis": "cc-pvdz",
        "df": True,
        "gpu": True,
    }


def test_pyscf_client_submit_calculation_round_trip():
    from io import BytesIO

    from chemrefine.engines.pyscf import client

    expected = b'{"energy": -2.5, "gradient": [[0.1, 0.2, 0.3]]}'
    with patch.object(client, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(expected)
        energy, gradient = client.submit_calculation(
            server_url="127.0.0.1:8889",
            atom_types=["H"],
            coordinates=[[0.0, 0.0, 0.0]],
            charge=0,
            mult=1,
            dograd=True,
            nthreads=1,
            settings={"method": "dft", "xc": "pbe", "basis": "def2-svp", "df": False, "gpu": False},
        )
    assert energy == -2.5
    assert gradient == [[0.1, 0.2, 0.3]]


def test_pyscf_client_submit_calculation_url_error_becomes_jobfailure():
    from urllib.error import URLError

    from chemrefine.engines.pyscf import client
    from chemrefine.errors import JobFailureError

    with (
        patch.object(client, "urlopen", side_effect=URLError("connection refused")),
        pytest.raises(JobFailureError, match="unreachable"),
    ):
        client.submit_calculation(
                server_url="x",
                atom_types=["H"],
                coordinates=[[0.0, 0.0, 0.0]],
                charge=0,
                mult=1,
                dograd=False,
                nthreads=1,
                settings={},
            )


# ---------------------------------------------------------------------------
# Server CLI + Flask app factory — verified
# ---------------------------------------------------------------------------


def test_pyscf_server_parse_args_defaults():
    from chemrefine.engines.pyscf.server import parse_args

    args = parse_args([])
    assert args.bind == "127.0.0.1:8889"
    assert args.method == "dft"
    assert args.xc == "pbe"
    assert args.basis == "def2-svp"
    assert args.df is False
    assert args.gpu is False


def test_pyscf_server_defaults_from_args_round_trip():
    from chemrefine.engines.pyscf.server import defaults_from_args, parse_args

    args = parse_args(["--default-method", "hf", "--default-basis", "cc-pvdz", "--default-gpu"])
    defaults = defaults_from_args(args)
    assert defaults["method"] == "hf"
    assert defaults["basis"] == "cc-pvdz"
    assert defaults["gpu"] is True


def test_pyscf_server_create_app_registers_calculate_route():
    from chemrefine.engines.pyscf.server import create_app

    app = create_app({"method": "dft", "xc": "pbe", "basis": "def2-svp", "df": False, "gpu": False})
    rules = {r.rule for r in app.url_map.iter_rules()}
    assert "/calculate" in rules


def test_pyscf_server_calculate_route_surfaces_run_calc_failure():
    """Until ``run_calc`` is ported, the route should return a 501 with a TODO-pointing error."""
    from chemrefine.engines.pyscf import server

    app = server.create_app(
        {"method": "dft", "xc": "pbe", "basis": "def2-svp", "df": False, "gpu": False}
    )
    client_ = app.test_client()
    resp = client_.post(
        "/calculate",
        json={
            "atom_types": ["H"],
            "coordinates": [[0.0, 0.0, 0.0]],
            "charge": 0,
            "mult": 1,
            "nthreads": 1,
            "settings": {},
        },
    )
    assert resp.status_code == 501
    assert "not yet ported" in resp.get_json()["error"]
