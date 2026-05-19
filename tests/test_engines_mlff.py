"""Tests for the MLFF engine package.

The expensive bits (real MACE/UMA inference) are out of scope here.
What we verify:

* Engines register correctly.
* ``MlffCalculator`` rejects unknown backends and reads the right
  setup branch for each backend's task_name pattern (without actually
  loading a model — that path is exercised in the ``mlff`` extra's
  integration tests).
* ``MlffEngine`` produces the right ``%method`` block and SLURM
  ``run_block`` content.
* Placeholder modules raise ``NotImplementedError`` with a TODO hint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines.base import ENGINES, get_engine
from chemrefine.engines.mlff.calculator import MlffCalculator
from chemrefine.state import PipelineState, StepContext, Structure

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_mlff_engines_are_registered():
    assert "mlff" in ENGINES
    assert "mlff-direct" in ENGINES


def test_mlff_engine_supports_nms_is_false():
    assert get_engine("mlff").supports_nms is False


def test_mlff_direct_engine_supports_nms_is_false():
    assert get_engine("mlff-direct").supports_nms is False


# ---------------------------------------------------------------------------
# MlffCalculator backend dispatch
# ---------------------------------------------------------------------------


def test_mlff_calculator_unknown_backend_raises():
    with pytest.raises(ValueError):
        MlffCalculator(model_name="unknown_model", task_name="not_a_real_task")


def test_mlff_calculator_dispatches_mace_off():
    with patch.object(MlffCalculator, "_build_mace", return_value="MACE_OFF_CALC") as mock:
        calc = MlffCalculator(model_name="medium", task_name="mace_off")
    mock.assert_called_once()
    assert calc.calculator == "MACE_OFF_CALC"


def test_mlff_calculator_dispatches_mace_mp():
    with patch.object(MlffCalculator, "_build_mace", return_value="MACE_MP_CALC") as mock:
        MlffCalculator(model_name="medium", task_name="mace_mp")
    mock.assert_called_once()


def test_mlff_calculator_dispatches_fairchem_for_omol():
    with patch.object(
        MlffCalculator, "_build_fairchem", return_value="FAIRCHEM_CALC"
    ) as mock:
        MlffCalculator(model_name="uma-s-1", task_name="omol")
    mock.assert_called_once()


def test_mlff_calculator_dispatches_sevenn_by_model_name():
    with patch.object(MlffCalculator, "_build_sevenn", return_value="SEVENN_CALC") as mock:
        MlffCalculator(model_name="sevenn-tiny", task_name="custom")
    mock.assert_called_once()


def test_mlff_calculator_dispatches_chgnet_for_chgnet_task():
    with patch.object(MlffCalculator, "_build_chgnet", return_value="CHGNET_CALC") as mock:
        MlffCalculator(model_name="ignored", task_name="chgnet")
    mock.assert_called_once()


def test_mlff_calculator_dispatches_orb_by_model_name():
    with patch.object(MlffCalculator, "_build_orb", return_value="ORB_CALC") as mock:
        MlffCalculator(model_name="orb-d3", task_name="custom")
    mock.assert_called_once()


def test_mlff_calculator_custom_model_path_picks_custom_mace(tmp_path: Path):
    """When ``model_path`` is given, the custom-MACE branch wins."""
    model_file = tmp_path / "fake.model"
    model_file.touch()
    with patch.object(
        MlffCalculator, "_build_custom_mace", return_value="CUSTOM_MACE_CALC"
    ) as mock:
        MlffCalculator(
            model_name="ignored", task_name="ignored", model_path=str(model_file)
        )
    mock.assert_called_once()


def test_mlff_calculator_custom_model_path_missing_raises(tmp_path: Path):
    """The custom-MACE branch should validate that the model file exists."""
    missing = tmp_path / "does_not_exist.model"
    with pytest.raises(FileNotFoundError):
        MlffCalculator(
            model_name="ignored", task_name="mace_off", model_path=str(missing)
        )


# ---------------------------------------------------------------------------
# MlffEngine — ORCA-driven mode
# ---------------------------------------------------------------------------


def _mlff_ctx(tmp_path: Path, **option_overrides) -> StepContext:
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "step1.inp").write_text("! B3LYP def2-SVP\n", encoding="utf-8")
    (template_dir / "cpu.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --partition=normal\n",
        encoding="utf-8",
    )
    options = {"model_name": "uma-s-1", "task_name": "omol", "device": "cuda"}
    options.update(option_overrides)
    step_cfg = StepConfig(
        step=1,
        engine="mlff",
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


def test_mlff_extra_blocks_contains_progext_pointing_to_wrapper(tmp_path: Path):
    engine = get_engine("mlff")
    ctx = _mlff_ctx(tmp_path)
    extra = engine._extra_blocks(ctx)
    assert "%method" in extra
    assert "ProgExt" in extra
    assert "mlff_extopt.sh" in extra
    assert "Ext_Params" in extra
    assert "127.0.0.1:8888" in extra


def test_mlff_custom_bind_address_round_trips(tmp_path: Path):
    engine = get_engine("mlff")
    ctx = _mlff_ctx(tmp_path, bind="10.0.0.5:5000")
    assert "10.0.0.5:5000" in engine._extra_blocks(ctx)


def test_mlff_run_block_starts_and_stops_server(tmp_path: Path):
    engine = get_engine("mlff")
    ctx = _mlff_ctx(tmp_path)
    run_block = engine._run_block(
        ctx,
        inp_path=ctx.step_dir / "step1_structure_0.inp",
        out_path=ctx.step_dir / "step1_structure_0.out",
    )
    assert "python -m chemrefine.engines.mlff.server" in run_block
    assert "--model" in run_block
    assert "--bind" in run_block
    assert "SERVER_PID=$!" in run_block
    assert "kill $SERVER_PID" in run_block
    assert ctx.orca_executable in run_block


def test_mlff_prepare_writes_inp_with_method_block(tmp_path: Path):
    engine = get_engine("mlff")
    ctx = _mlff_ctx(tmp_path)
    inputs = engine.prepare(ctx)
    inp_text = inputs.files[0][0].read_text()
    assert "%method" in inp_text
    assert "ProgExt" in inp_text


# ---------------------------------------------------------------------------
# MlffDirectEngine — in-process scoring
# ---------------------------------------------------------------------------


def test_mlff_direct_round_trip_with_mocked_calculator(tmp_path: Path):
    engine = get_engine("mlff-direct")
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    step_cfg = StepConfig(
        step=1,
        engine="mlff-direct",
        operation="opt_sp",
        options={"model_name": "uma-s-1", "task_name": "omol"},
    )
    seeds = (
        Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])),
        Structure(id="1", atoms=Atoms("H2", positions=[[0, 0, 0], [0.80, 0, 0]])),
    )
    ctx = StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=template_dir,
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(structures=seeds),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        orca_executable="orca",
    )
    with (
        patch.object(MlffCalculator, "_build", return_value=None),
        patch.object(MlffCalculator, "single_point", side_effect=[(-1.5, []), (-2.0, [])]),
    ):
        inputs = engine.prepare(ctx)
        engine.submit(inputs, ctx)
        results = engine.parse(inputs, ctx)
    assert [s.id for s in results.structures] == ["0", "1"]
    # Energies should be sorted in eV → Hartree (negative)
    assert all(s.energy_hartree is not None and s.energy_hartree < 0 for s in results.structures)


def test_mlff_direct_does_not_support_nms(tmp_path: Path):
    engine = get_engine("mlff-direct")
    from chemrefine.state import StepResults

    with pytest.raises(NotImplementedError):
        engine.normal_mode_sample(StepResults(structures=()), ctx=None)  # type: ignore[arg-type]


def test_mlff_direct_wait_is_noop():
    from chemrefine.state import JobBatch

    engine = get_engine("mlff-direct")
    engine.wait(JobBatch(jobs={}))  # must not raise


def test_mlff_direct_get_calculator_is_cached(tmp_path: Path):
    """``_get_calculator`` builds once, then returns the cached instance."""
    from chemrefine.engines.mlff.direct import MlffDirectEngine

    step_cfg = StepConfig(
        step=1,
        engine="mlff-direct",
        operation="opt_sp",
        options={"model_name": "uma-s-1", "task_name": "omol"},
    )
    seed = Structure(id="0", atoms=Atoms("H"))
    ctx = StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path,
        template_dir=tmp_path,
        scratch_dir=tmp_path,
        prev_state=PipelineState(structures=(seed,)),
        charge=0,
        multiplicity=1,
        max_cores=1,
        slurm_template="cpu.slurm.header",
        orca_executable="orca",
    )
    engine = MlffDirectEngine()
    with patch.object(MlffCalculator, "_build", return_value=None):
        a = engine._get_calculator(ctx)
        b = engine._get_calculator(ctx)
    assert a is b


# ---------------------------------------------------------------------------
# Placeholders — trainer / server / client
# ---------------------------------------------------------------------------


def test_trainer_placeholder_raises_with_todo():
    from chemrefine.engines.mlff.trainer import run_training
    from chemrefine.state import StepResults

    with pytest.raises(NotImplementedError):
        run_training(StepResults(structures=()), ctx=None)  # type: ignore[arg-type]


def test_server_main_still_placeholder():
    """Only the model-loading + ``waitress.serve`` glue stays a TODO."""
    from chemrefine.engines.mlff.server import main as server_main

    with pytest.raises(NotImplementedError):
        server_main()


def test_client_main_still_placeholder():
    """The ExtOpt wrapper that reads ``.extinp.tmp`` is the remaining TODO."""
    from chemrefine.engines.mlff.client import main as client_main

    with pytest.raises(NotImplementedError):
        client_main()


# ---------------------------------------------------------------------------
# Client CLI + HTTP RPC — verified
# ---------------------------------------------------------------------------


def test_mlff_client_parse_args_defaults():
    from chemrefine.engines.mlff.client import parse_args

    args = parse_args(["job.extinp.tmp"])
    assert args.bind == "127.0.0.1:8888"
    assert args.model_name == "uma-s-1"
    assert args.task_name == "omol"
    assert args.device == "cuda"
    assert args.inputfile == "job.extinp.tmp"


def test_mlff_client_parse_args_overrides():
    from chemrefine.engines.mlff.client import parse_args

    args = parse_args(
        ["--bind", "10.0.0.5:9000", "--model_name", "medium", "--device", "cpu", "f.extinp.tmp"]
    )
    assert args.bind == "10.0.0.5:9000"
    assert args.model_name == "medium"
    assert args.device == "cpu"


def test_mlff_client_submit_calculation_round_trip():
    from io import BytesIO

    from chemrefine.engines.mlff import client

    expected = b'{"energy": -1.5, "gradient": [[0.0, 0.0, 0.0]]}'
    with patch.object(client, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(expected)
        energy, gradient = client.submit_calculation(
            server_url="127.0.0.1:8888",
            atom_types=["H"],
            coordinates=[[0.0, 0.0, 0.0]],
            charge=0,
            mult=1,
            dograd=True,
            nthreads=1,
        )
    assert energy == -1.5
    assert gradient == [[0.0, 0.0, 0.0]]


def test_mlff_client_submit_calculation_http_error_becomes_jobfailure():
    from urllib.error import HTTPError

    from chemrefine.engines.mlff import client
    from chemrefine.errors import JobFailureError

    err = HTTPError("http://x/calculate", 500, "internal error", {}, None)
    with (
        patch.object(client, "urlopen", side_effect=err),
        pytest.raises(JobFailureError, match="HTTP 500"),
    ):
        client.submit_calculation(
                server_url="x",
                atom_types=["H"],
                coordinates=[[0.0, 0.0, 0.0]],
                charge=0,
                mult=1,
                dograd=False,
                nthreads=1,
            )


def test_mlff_client_submit_calculation_server_returns_error_field():
    from io import BytesIO

    from chemrefine.engines.mlff import client
    from chemrefine.errors import JobFailureError

    with patch.object(client, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(b'{"error": "no cuda"}')
        with pytest.raises(JobFailureError, match="no cuda"):
            client.submit_calculation(
                server_url="x",
                atom_types=["H"],
                coordinates=[[0.0, 0.0, 0.0]],
                charge=0,
                mult=1,
                dograd=False,
                nthreads=1,
            )


def test_mlff_client_submit_calculation_url_error_becomes_jobfailure():
    from urllib.error import URLError

    from chemrefine.engines.mlff import client
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
        )


def test_mlff_client_submit_calculation_non_json_response_becomes_jobfailure():
    from io import BytesIO

    from chemrefine.engines.mlff import client
    from chemrefine.errors import JobFailureError

    with patch.object(client, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(b"not json at all")
        with pytest.raises(JobFailureError, match="non-JSON"):
            client.submit_calculation(
                server_url="x",
                atom_types=["H"],
                coordinates=[[0.0, 0.0, 0.0]],
                charge=0,
                mult=1,
                dograd=False,
                nthreads=1,
            )


def test_mlff_client_submit_calculation_missing_fields_becomes_jobfailure():
    from io import BytesIO

    from chemrefine.engines.mlff import client
    from chemrefine.errors import JobFailureError

    with patch.object(client, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(b'{"foo": 1}')
        with pytest.raises(JobFailureError, match="missing fields"):
            client.submit_calculation(
                server_url="x",
                atom_types=["H"],
                coordinates=[[0.0, 0.0, 0.0]],
                charge=0,
                mult=1,
                dograd=False,
                nthreads=1,
            )


def test_mlff_server_calculate_route_handles_calculator_errors():
    """Server should turn an exception in single_point into a 500 with error body."""
    from chemrefine.engines.mlff import calculator as calc_mod
    from chemrefine.engines.mlff.server import create_app

    with (
        patch.object(calc_mod.MlffCalculator, "_build", return_value=None),
        patch.object(calc_mod.MlffCalculator, "single_point", side_effect=RuntimeError("boom")),
    ):
        app = create_app(model_name="x", task_name="mace_off", device="cpu")
        client_ = app.test_client()
        resp = client_.post(
            "/calculate",
            json={
                "atom_types": ["H"],
                "coordinates": [[0.0, 0.0, 0.0]],
                "charge": 0,
                "mult": 1,
                "nthreads": 1,
            },
        )
    assert resp.status_code == 500
    assert "boom" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# Server CLI + Flask app factory — verified
# ---------------------------------------------------------------------------


def test_mlff_server_parse_args_requires_a_model():
    """At least one of ``--model`` or ``--model-path`` must be given."""
    from chemrefine.engines.mlff.server import parse_args

    with pytest.raises(SystemExit):
        parse_args([])


def test_mlff_server_parse_args_accepts_model_only():
    from chemrefine.engines.mlff.server import parse_args

    args = parse_args(["--model", "medium"])
    assert args.model == "medium"
    assert args.task_name == "omol"
    assert args.bind == "127.0.0.1:8888"


def test_mlff_server_parse_args_accepts_model_path_only(tmp_path: Path):
    from chemrefine.engines.mlff.server import parse_args

    args = parse_args(["--model-path", str(tmp_path / "fake.model")])
    assert args.model is None
    assert args.model_path is not None


def test_mlff_server_create_app_registers_calculate_route():
    from chemrefine.engines.mlff.server import create_app

    app = create_app(
        model_name="medium", task_name="mace_off", device="cpu", model_path=None
    )
    # The route should be visible in the Flask URL map.
    rules = {r.rule for r in app.url_map.iter_rules()}
    assert "/calculate" in rules


def test_mlff_server_calculate_route_returns_json_with_mocked_calculator():
    from chemrefine.engines.mlff import calculator as calc_mod
    from chemrefine.engines.mlff.server import create_app

    with (
        patch.object(calc_mod.MlffCalculator, "_build", return_value=None),
        patch.object(calc_mod.MlffCalculator, "single_point", return_value=(-1.0, [[0.0, 0.0, 0.0]])),
    ):
        app = create_app(model_name="medium", task_name="mace_off", device="cpu")
        client_ = app.test_client()
        resp = client_.post(
            "/calculate",
            json={
                "atom_types": ["H"],
                "coordinates": [[0.0, 0.0, 0.0]],
                "charge": 0,
                "mult": 1,
                "nthreads": 1,
            },
        )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["energy"] == -1.0
    assert data["gradient"] == [[0.0, 0.0, 0.0]]
