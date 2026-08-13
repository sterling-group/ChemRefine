"""Tests for the shared ``engines (_backend_server + orca/extopt)`` infrastructure.

Covers:

* :mod:`chemrefine.engines.orca.extopt.protocol` — ``.extinp.tmp``,
  ``.engrad``, wrapper script, and sidecar URL helpers.
* :mod:`chemrefine.engines._backend_server.server` — CLI parsing + Flask app
  factory (with a mock backend).
* :mod:`chemrefine.engines.orca.extopt.bridge` — CLI parsing, HTTP RPC,
  and the ``main`` glue end-to-end.
* :mod:`chemrefine.engines._backend_server.registry` — every registered
  backend imports + conforms to the ComputeBackend contract.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from chemrefine.engines._backend_server import registry, server, sidecar
from chemrefine.engines._backend_server.base import (
    DEFAULT_BIND_HOST,
    SERVER_URL_FILENAME,
    CalculationData,
)
from chemrefine.engines.orca.extopt import bridge, protocol
from chemrefine.errors import JobFailureError

# ---------------------------------------------------------------------------
# CalculationData + Base contract
# ---------------------------------------------------------------------------


def test_calculation_data_round_trips_fields():
    data = CalculationData(
        symbols=("H", "O", "H"),
        positions_angstrom=np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
        charge=0,
        multiplicity=1,
        nthreads=4,
        dograd=True,
        settings={"method": "dft"},
    )
    assert data.symbols == ("H", "O", "H")
    assert data.positions_angstrom.shape == (3, 3)
    assert data.dograd is True
    assert data.settings == {"method": "dft"}


def test_default_bind_host_is_loopback():
    assert DEFAULT_BIND_HOST == "127.0.0.1"


def test_server_url_filename_constant():
    assert SERVER_URL_FILENAME == "server.url"


# ---------------------------------------------------------------------------
# protocol.read_extinp / write_engrad / write_wrapper_script
# ---------------------------------------------------------------------------


def _write_extinp(tmp_path: Path, *, dograd: int = 1) -> Path:
    xyz = tmp_path / "struct.xyz"
    xyz.write_text("2\ncomment\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n", encoding="utf-8")
    inp = tmp_path / "step1_structure_0.extinp.tmp"
    inp.write_text(
        "struct.xyz   # XYZ filename\n"
        "0            # charge\n"
        "1            # mult\n"
        "4            # ncores\n"
        f"{dograd}            # dograd\n",
        encoding="utf-8",
    )
    return inp


def test_read_extinp_parses_header_and_xyz(tmp_path: Path):
    inp = _write_extinp(tmp_path)
    data = protocol.read_extinp(inp, settings={"method": "dft"})
    assert data.symbols == ("H", "H")
    assert data.positions_angstrom.shape == (2, 3)
    assert data.charge == 0
    assert data.multiplicity == 1
    assert data.nthreads == 4
    assert data.dograd is True
    assert data.settings == {"method": "dft"}


def test_read_extinp_handles_dograd_zero(tmp_path: Path):
    inp = _write_extinp(tmp_path, dograd=0)
    assert protocol.read_extinp(inp).dograd is False


@pytest.mark.parametrize("kept_lines", [0, 1, 4])
def test_read_extinp_rejects_a_truncated_file(tmp_path: Path, kept_lines: int):
    """A short `.extinp.tmp` is a classified job failure, not an IndexError.

    ORCA killed mid-write (walltime, a full disk) leaves a partial file. Indexing it
    blind raised IndexError *inside the wrapper script*, so ORCA got no `.engrad` and
    the step died with a bare traceback in the runlog instead of a failure the ledger
    could record.
    """
    inp = _write_extinp(tmp_path)
    inp.write_text(
        "\n".join(inp.read_text(encoding="utf-8").splitlines()[:kept_lines]), encoding="utf-8"
    )

    with pytest.raises(JobFailureError, match="truncated"):
        protocol.read_extinp(inp)


@pytest.mark.parametrize(
    ("xyz_text", "match"),
    [
        ("", "not an atom count"),  # empty file: no count line at all
        ("two\ncomment\n", "not an atom count"),  # count line that is not a number
        ("2\ncomment\nH 0.0 0.0 0.0\n", "truncated"),  # killed mid-atom-block
        ("2\ncomment\nH 0.0 0.0 0.0\nH 0.74 0.0\n", "bad atom row"),  # row cut short
        ("1\ncomment\nH 0.0 zero 0.0\n", "bad atom row"),  # corrupt coordinate token
    ],
)
def test_read_extinp_rejects_a_bad_xyz(tmp_path: Path, xyz_text: str, match: str):
    """The referenced `.xyz` is held to the same rule as the `.extinp.tmp` header.

    ORCA writes both files per ProgExt call, so the same kill or full disk that
    truncates one truncates the other — and this one raised a bare IndexError in the
    wrapper's runlog instead of a failure naming the file.
    """
    inp = _write_extinp(tmp_path)
    (tmp_path / "struct.xyz").write_text(xyz_text, encoding="utf-8")

    with pytest.raises(JobFailureError, match=match):
        protocol.read_extinp(inp)


def test_read_extinp_handles_absolute_xyz_path(tmp_path: Path):
    xyz = tmp_path / "abs_struct.xyz"
    xyz.write_text("1\nc\nH 1.0 2.0 3.0\n", encoding="utf-8")
    inp = tmp_path / "job.extinp.tmp"
    inp.write_text(f"{xyz}\n0\n1\n1\n1\n", encoding="utf-8")
    data = protocol.read_extinp(inp)
    assert data.symbols == ("H",)
    assert data.positions_angstrom[0, 0] == 1.0


def test_write_engrad_emits_full_format(tmp_path: Path):
    out = tmp_path / "job.engrad"
    protocol.write_engrad(
        path=out,
        n_atoms=2,
        energy_hartree=-1.123456789012,
        gradients_hartree_per_bohr=[[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]],
    )
    text = out.read_text(encoding="utf-8")
    assert "Number of atoms" in text
    assert "Total energy [Eh]" in text
    assert "Gradient [Eh/Bohr]" in text
    assert "2\n" in text
    # Six gradient components on six lines (two atoms by three components) plus one energy line.
    numeric_lines = [
        line
        for line in text.splitlines()
        if line and not line.startswith("#") and "atoms" not in line
    ]
    # numeric_lines now: n_atoms ('2'), energy, then 6 gradient values
    assert len(numeric_lines) == 8


def test_write_engrad_skips_gradient_block_when_dograd_false(tmp_path: Path):
    out = tmp_path / "job.engrad"
    protocol.write_engrad(
        path=out,
        n_atoms=1,
        energy_hartree=-0.5,
        gradients_hartree_per_bohr=None,
        dograd=False,
    )
    text = out.read_text(encoding="utf-8")
    assert "Gradient" not in text


def test_write_engrad_dograd_true_requires_gradient(tmp_path: Path):
    with pytest.raises(ValueError, match="dograd=True but no gradient"):
        protocol.write_engrad(
            path=tmp_path / "x.engrad",
            n_atoms=1,
            energy_hartree=0.0,
            gradients_hartree_per_bohr=None,
            dograd=True,
        )


def test_write_wrapper_script_emits_exec_call_and_is_executable(tmp_path: Path):
    out = tmp_path / "mlip_extopt.sh"
    url_file = tmp_path / "server.url"
    protocol.write_wrapper_script(
        path=out,
        backend="mlip",
        url_file=url_file,
    )
    text = out.read_text(encoding="utf-8")
    assert "#!/usr/bin/env bash" in text
    assert 'URL_FILE="' + str(url_file) + '"' in text
    assert "chemrefine.engines.orca.extopt.bridge" in text
    assert "--backend mlip" in text
    # The bridge must run with the orchestrator's interpreter, not a bare
    # ``python`` that may resolve to a system interpreter without chemrefine.
    assert f"exec {shlex.quote(sys.executable)} -m" in text
    assert out.stat().st_mode & 0o100  # owner-execute bit


def test_write_wrapper_script_threads_extra_args(tmp_path: Path):
    out = tmp_path / "pyscf_extopt.sh"
    protocol.write_wrapper_script(
        path=out,
        backend="pyscf",
        url_file=tmp_path / "u",
        extra_args="--method dft --xc pbe",
    )
    assert "--method dft --xc pbe" in out.read_text(encoding="utf-8")


def test_extopt_modules_have_main_entry_guard():
    """server + client must be runnable via ``python -m`` — the run_block starts the
    server with ``python -m ..._backend_server.server`` and the wrapper calls
    ``python -m ...orca.extopt.bridge``, so both need an ``if __name__ == '__main__'``
    guard that invokes ``main()``. Without it the module imports and exits 0 without
    starting, and the ExtOpt server "crashes during startup".
    """
    from chemrefine.engines._backend_server import server
    from chemrefine.engines.orca.extopt import bridge

    for mod in (server, bridge):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in src, f"{mod.__name__} missing -m entry guard"
        assert "main()" in src


def test_wrapper_passes_input_file_as_final_positional(tmp_path: Path):
    """Argv contract: ORCA's input file ("$1") is the client's sole positional.

    Settings ride as ``--flag`` options *before* it, so the client parses the
    ``.extinp.tmp`` as its positional no matter how many settings are baked in.
    This is the contract that lets us drop ORCA ``Ext_Params`` and keep one
    settings channel. (Confirm against a real ORCA ``ProgExt`` run.)
    """
    out = tmp_path / "pyscf_extopt.sh"
    protocol.write_wrapper_script(
        path=out,
        backend="pyscf",
        url_file=tmp_path / "u",
        extra_args="--method dft --xc pbe",
    )
    exec_line = next(
        ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.startswith("exec ")
    )
    assert exec_line.rstrip().endswith('"$1"')
    assert "--backend pyscf" in exec_line
    assert "--method dft --xc pbe" in exec_line


# ---------------------------------------------------------------------------
# Sidecar URL file
# ---------------------------------------------------------------------------


def test_write_and_read_server_url_round_trip(tmp_path: Path):
    target = tmp_path / "subdir" / "server.url"
    sidecar.write_server_url(target, "127.0.0.1:54321")
    assert sidecar.read_server_url(target) == "127.0.0.1:54321"


def test_write_server_url_cleans_up_temp_on_failure(tmp_path: Path):
    """A rename failure should remove the tempfile (no leftovers)."""
    target = tmp_path / "server.url"
    with patch("os.replace", side_effect=OSError("denied")), pytest.raises(OSError):
        sidecar.write_server_url(target, "x:1")
    # Ensure no tempfiles linger
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".url.")]
    assert leftovers == []


def test_write_and_read_server_token_round_trip(tmp_path: Path):
    target = tmp_path / "server.token"
    sidecar.write_server_token(target, "s3cret")
    assert sidecar.read_server_token(target) == "s3cret"


def test_server_token_file_is_owner_readable_only(tmp_path: Path):
    """The token is a secret — 0600, never group/world readable."""
    target = tmp_path / "server.token"
    sidecar.write_server_token(target, "s3cret")
    assert target.stat().st_mode & 0o777 == 0o600


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_discovers_mlip_and_pyscf():
    """The registry finds the ExtOpt backends from the engine registry — no central list."""
    assert registry.known_backends() == ["mlip", "pyscf"]


def test_discovery_is_the_extopt_served_capability():
    """Detection is the `ExtOptServed` Protocol: both declarations or nothing.

    An engine carrying only `backend` (or only `calculator_cls`) is not an ExtOpt engine —
    the old `getattr` probe read the same two attributes but nothing typed the contract, so
    a half-declared engine was silently skipped with no name for what it was missing.
    """
    from chemrefine.engines._backend_server.base import ExtOptServed
    from chemrefine.engines.api import get_engine

    assert isinstance(get_engine("mlip-extopt"), ExtOptServed)
    assert isinstance(get_engine("pyscf-extopt"), ExtOptServed)
    assert not isinstance(get_engine("orca"), ExtOptServed)

    class _HalfDeclared:
        backend = "half"

    assert not isinstance(_HalfDeclared(), ExtOptServed)


def test_load_calculator_returns_class():
    cls = registry.load_calculator("mlip")
    assert isinstance(cls, type)


def test_load_calculator_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        registry.load_calculator("not_a_backend")


def test_every_registered_backend_conforms_to_base_protocol():
    """Loaded backends should be ``ComputeBackend``-conformant classes.

    We only check class-level shape (``name`` + ``calc`` + ``from_args``)
    here — real instantiation requires backend dependencies (torch /
    pyscf) the test env doesn't install.
    """
    for name in registry.known_backends():
        cls = registry.load_calculator(name)
        assert hasattr(cls, "name")
        assert callable(cls.calc)
        assert callable(cls.from_args)
        assert callable(cls.add_cli_args)
        assert callable(cls.settings_from_args)
        assert callable(cls.server_cli_from_options)


# ---------------------------------------------------------------------------
# Shared layer is backend-agnostic (no backend literals leaking through)
# ---------------------------------------------------------------------------


def test_shared_server_source_has_no_backend_specific_flags():
    """The shared server module must not enumerate any backend's CLI flags."""
    import re

    text = Path(server.__file__).read_text(encoding="utf-8")
    forbidden = re.compile(r"--(?:model|task-name|device|model-path|method|xc|basis|df|gpu)\b")
    matches = forbidden.findall(text)
    assert matches == [], f"shared server leaks backend CLI flags: {matches}"


def test_shared_client_source_has_no_backend_specific_flags():
    """The shared client module must not enumerate any backend's CLI flags."""
    import re

    text = Path(bridge.__file__).read_text(encoding="utf-8")
    forbidden = re.compile(r"--(?:model|task-name|device|model-path|method|xc|basis|df|gpu)\b")
    matches = forbidden.findall(text)
    assert matches == [], f"shared client leaks backend CLI flags: {matches}"


# ---------------------------------------------------------------------------
# Server CLI
# ---------------------------------------------------------------------------


def test_server_parse_args_requires_backend():
    with pytest.raises(SystemExit):
        server.parse_args([])


def test_server_parse_args_defaults():
    args = server.parse_args(["--backend", "mlip"])
    assert args.backend == "mlip"
    assert args.bind == "127.0.0.1:0"
    assert args.nthreads == 4
    assert args.log_level == "INFO"
    assert args.method == "dft"  # PySCF default available regardless of backend


def test_server_parse_args_accepts_all_overrides():
    args = server.parse_args(
        [
            "--backend",
            "pyscf",
            "--bind",
            "127.0.0.1:54321",
            "--url-file",
            "/tmp/foo",
            "--log-file",
            "/tmp/srv.log",
            "--log-level",
            "DEBUG",
            "--method",
            "hf",
            "--xc",
            "b3lyp",
            "--basis",
            "cc-pvdz",
            "--df",
            "--gpu",
        ]
    )
    assert args.backend == "pyscf"
    assert args.df is True
    assert args.gpu is True


# ---------------------------------------------------------------------------
# Server Flask app
# ---------------------------------------------------------------------------


class _MockCalculator:
    name = "mock"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> _MockCalculator:
        return cls()

    def calc(self, data: CalculationData) -> tuple[float, list[list[float]]]:
        n = len(data.symbols)
        return -1.0, [[0.0, 0.0, 0.0]] * n


def test_create_app_serves_healthz_route():
    app = server.create_app(_MockCalculator())
    rules = {r.rule for r in app.url_map.iter_rules()}
    assert "/healthz" in rules
    assert "/calculate" in rules
    test_client = app.test_client()
    resp = test_client.get("/healthz")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok", "backend": "mock"}


def test_create_app_calculate_route_returns_energy_and_gradient():
    app = server.create_app(_MockCalculator())
    resp = app.test_client().post(
        "/calculate",
        json={
            "atom_types": ["H", "H"],
            "coordinates": [[0, 0, 0], [0.74, 0, 0]],
            "charge": 0,
            "mult": 1,
            "nthreads": 1,
            "dograd": True,
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["energy"] == -1.0
    assert data["gradient"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


_CALC_PAYLOAD = {
    "atom_types": ["H"],
    "coordinates": [[0, 0, 0]],
    "charge": 0,
    "mult": 1,
    "nthreads": 1,
    "dograd": True,
}


def test_calculate_rejects_requests_without_token():
    """With a token configured, an unauthenticated POST is 401 — any same-node
    user can reach the loopback port, so possession of the token is the gate."""
    app = server.create_app(_MockCalculator(), token="s3cret")
    resp = app.test_client().post("/calculate", json=_CALC_PAYLOAD)
    assert resp.status_code == 401
    assert resp.get_json() == {"error": "unauthorized"}


def test_calculate_rejects_wrong_token():
    app = server.create_app(_MockCalculator(), token="s3cret")
    resp = app.test_client().post(
        "/calculate", json=_CALC_PAYLOAD, headers={"Authorization": "Bearer wrong"}
    )
    assert resp.status_code == 401


def test_calculate_accepts_bearer_token():
    app = server.create_app(_MockCalculator(), token="s3cret")
    resp = app.test_client().post(
        "/calculate", json=_CALC_PAYLOAD, headers={"Authorization": "Bearer s3cret"}
    )
    assert resp.status_code == 200
    assert resp.get_json()["energy"] == -1.0


def test_healthz_stays_open_with_token_configured():
    """The run_block readiness curl carries no token — healthz must stay open."""
    app = server.create_app(_MockCalculator(), token="s3cret")
    assert app.test_client().get("/healthz").status_code == 200


def test_create_app_calculate_route_returns_500_on_calculator_error():
    class _Boom:
        name = "boom"

        def calc(self, data):
            raise RuntimeError("backend exploded")

    app = server.create_app(_Boom())
    resp = app.test_client().post(
        "/calculate",
        json={
            "atom_types": ["H"],
            "coordinates": [[0, 0, 0]],
            "charge": 0,
            "mult": 1,
            "nthreads": 1,
        },
    )
    assert resp.status_code == 500
    # The backend's own message stays in the server log (readable only by the job
    # owner); the response carries a correlation id to find it by.
    body = resp.get_json()["error"]
    assert "backend exploded" not in body
    assert "request " in body
    assert "see the ExtOpt server log" in body


def test_create_app_logs_the_real_error_with_its_correlation_id(caplog):
    """The detail is not discarded — it is logged against the id the client was given."""
    import logging
    import re

    class _Boom:
        name = "boom"

        def calc(self, data):
            raise RuntimeError("backend exploded")

    caplog.set_level(logging.ERROR, logger="chemrefine.engines._backend_server.server")
    app = server.create_app(_Boom())
    resp = app.test_client().post(
        "/calculate",
        json={
            "atom_types": ["H"],
            "coordinates": [[0, 0, 0]],
            "charge": 0,
            "mult": 1,
            "nthreads": 1,
        },
    )
    request_id = re.search(r"request ([0-9a-f]+)", resp.get_json()["error"]).group(1)
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert request_id in logged
    assert "backend exploded" in logged


def test_create_app_logs_with_correlation_tag(caplog):
    """``payload['tag']`` should appear in the server's INFO log line."""
    import logging

    caplog.set_level(logging.INFO, logger="chemrefine.engines._backend_server.server")
    app = server.create_app(_MockCalculator())
    app.test_client().post(
        "/calculate",
        json={
            "atom_types": ["H"],
            "coordinates": [[0, 0, 0]],
            "charge": 0,
            "mult": 1,
            "nthreads": 1,
            "tag": "req-abc",
        },
    )
    assert any("req=req-abc" in rec.message for rec in caplog.records)


def test_calculate_route_folds_top_level_tag_into_settings():
    """The bridge sends ``tag`` at the payload top level; the server folds it into
    ``settings`` so backends (e.g. PySCF tensor dumps) read it via the one
    ``settings`` channel without a separate argument."""
    captured: dict = {}

    class _Recorder:
        name = "rec"

        def calc(self, data: CalculationData) -> tuple[float, list[list[float]]]:
            captured["settings"] = dict(data.settings)
            return 0.0, []

    app = server.create_app(_Recorder())
    app.test_client().post(
        "/calculate",
        json={
            "atom_types": ["H"],
            "coordinates": [[0, 0, 0]],
            "charge": 0,
            "mult": 1,
            "nthreads": 1,
            "settings": {"method": "dft"},
            "tag": "step3_structure_0",
        },
    )
    assert captured["settings"]["tag"] == "step3_structure_0"
    assert captured["settings"]["method"] == "dft"  # existing settings preserved


# ---------------------------------------------------------------------------
# Client CLI + HTTP RPC
# ---------------------------------------------------------------------------


def test_client_parse_args_requires_backend_and_inputfile():
    with pytest.raises(SystemExit):
        bridge.parse_args(["job.extinp.tmp"])  # missing --backend


def test_client_parse_args_defaults():
    args = bridge.parse_args(["--backend", "mlip", "job.extinp.tmp"])
    assert args.backend == "mlip"
    assert args.bind is None
    assert args.url_file is None
    assert args.method == "dft"
    assert args.inputfile == "job.extinp.tmp"


def test_client_settings_from_args_is_empty_single_channel():
    """Both shipped backends are single-channel: knobs live on the server, the POST
    carries no per-call settings (the server injects only the correlation ``tag``)."""
    pyscf_args = bridge.parse_args(
        ["--backend", "pyscf", "--method", "hf", "--basis", "cc-pvdz", "--df", "--gpu", "f"]
    )
    assert bridge.settings_from_args(pyscf_args) == {}

    mlip_args = bridge.parse_args(["--backend", "mlip", "--device", "cpu", "f"])
    assert bridge.settings_from_args(mlip_args) == {}


def _data() -> CalculationData:
    return CalculationData(
        symbols=("H",),
        positions_angstrom=np.array([[0.0, 0.0, 0.0]]),
        charge=0,
        multiplicity=1,
        nthreads=1,
        dograd=True,
        settings={},
    )


def test_submit_calculation_round_trip():
    expected = b'{"energy": -1.5, "gradient": [[0.0, 0.0, 0.0]]}'
    with patch.object(bridge, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(expected)
        energy, gradient = bridge.submit_calculation(
            server_url="127.0.0.1:54321",
            data=_data(),
        )
    assert energy == -1.5
    assert gradient == [[0.0, 0.0, 0.0]]


def test_submit_calculation_sends_bearer_token():
    with patch.object(bridge, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(b'{"energy": -1.0, "gradient": []}')
        bridge.submit_calculation(server_url="x", data=_data(), token="s3cret")
    request = mock_open.call_args[0][0]
    assert request.get_header("Authorization") == "Bearer s3cret"


def test_submit_calculation_omits_header_without_token():
    with patch.object(bridge, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(b'{"energy": -1.0, "gradient": []}')
        bridge.submit_calculation(server_url="x", data=_data())
    assert mock_open.call_args[0][0].get_header("Authorization") is None


def test_resolve_server_token_reads_work_dir_sidecar(tmp_path: Path, monkeypatch):
    """Default resolution: $WORK_DIR/server.token — the wrapper's --bind path
    still finds the token without any extra flag."""
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    sidecar.write_server_token(tmp_path / "server.token", "tok123")
    args = argparse.Namespace(token_file=None)
    assert bridge.resolve_server_token(args) == "tok123"


def test_resolve_server_token_explicit_flag_wins(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    sidecar.write_server_token(tmp_path / "server.token", "wrong")
    explicit = tmp_path / "elsewhere.token"
    sidecar.write_server_token(explicit, "right")
    args = argparse.Namespace(token_file=str(explicit))
    assert bridge.resolve_server_token(args) == "right"


def test_resolve_server_token_missing_file_degrades_to_none(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    args = argparse.Namespace(token_file=None)
    assert bridge.resolve_server_token(args) is None


def test_submit_calculation_http_error_becomes_jobfailure():
    from urllib.error import HTTPError

    err = HTTPError("http://x/calculate", 500, "internal", {}, None)
    with (
        patch.object(bridge, "urlopen", side_effect=err),
        pytest.raises(JobFailureError, match="HTTP 500"),
    ):
        bridge.submit_calculation(server_url="x", data=_data())


def test_submit_calculation_url_error_becomes_jobfailure():
    from urllib.error import URLError

    with (
        patch.object(bridge, "urlopen", side_effect=URLError("refused")),
        pytest.raises(JobFailureError, match="unreachable"),
    ):
        bridge.submit_calculation(server_url="x", data=_data())


def test_submit_calculation_non_json_response_becomes_jobfailure():
    with patch.object(bridge, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(b"<html>")
        with pytest.raises(JobFailureError, match="non-JSON"):
            bridge.submit_calculation(server_url="x", data=_data())


def test_submit_calculation_error_field_becomes_jobfailure():
    with patch.object(bridge, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(b'{"error": "boom"}')
        with pytest.raises(JobFailureError, match="boom"):
            bridge.submit_calculation(server_url="x", data=_data())


def test_submit_calculation_missing_fields_becomes_jobfailure():
    with patch.object(bridge, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(b'{"foo": 1}')
        with pytest.raises(JobFailureError, match="missing fields"):
            bridge.submit_calculation(server_url="x", data=_data())


def test_submit_calculation_threads_tag_into_payload():
    """``tag`` should appear in the JSON payload the server receives."""
    captured: dict = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"energy": 0.0, "gradient": []}'

    def _fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data.decode())
        return _FakeResp()

    with patch.object(bridge, "urlopen", side_effect=_fake_urlopen):
        bridge.submit_calculation(server_url="x", data=_data(), tag="abc-123")
    assert captured["payload"]["tag"] == "abc-123"


def test_resolve_server_url_prefers_explicit_bind(tmp_path: Path):
    args = bridge.parse_args(["--backend", "mlip", "--bind", "1.2.3.4:5", "x"])
    assert bridge.resolve_server_url(args) == "1.2.3.4:5"


def test_resolve_server_url_reads_url_file(tmp_path: Path):
    url_file = tmp_path / "server.url"
    url_file.write_text("127.0.0.1:9999\n", encoding="utf-8")
    args = bridge.parse_args(["--backend", "mlip", "--url-file", str(url_file), "x"])
    assert bridge.resolve_server_url(args) == "127.0.0.1:9999"


def test_client_engrad_path_for_strips_extinp_tmp():
    assert bridge._engrad_path_for("a/b/foo.extinp.tmp") == Path("a/b/foo.engrad")


def test_client_engrad_path_for_falls_back_to_with_suffix():
    """An input without ``.extinp.tmp`` still gets ``.engrad`` via ``with_suffix``."""
    assert bridge._engrad_path_for("foo.tmp") == Path("foo.engrad")


def test_client_tag_for_strips_double_extinp_suffix():
    """``_tag_for`` strips the full ``.extinp.tmp`` (not just ``.tmp``)."""
    assert bridge._tag_for("a/b/step3_structure_0.extinp.tmp") == "step3_structure_0"


def test_client_tag_for_falls_back_to_stem():
    assert bridge._tag_for("foo.bar") == "foo"


# ---------------------------------------------------------------------------
# Client main — end-to-end with a mocked server
# ---------------------------------------------------------------------------


def test_client_main_writes_engrad(tmp_path: Path, monkeypatch):
    """End-to-end: read extinp, mock the HTTP call, write engrad."""
    inp = _write_extinp(tmp_path)
    url_file = tmp_path / "server.url"
    url_file.write_text("127.0.0.1:1234\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["bridge.py", "--backend", "mlip", "--url-file", str(url_file), str(inp)],
    )

    expected = b'{"energy": -1.0, "gradient": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}'
    with patch.object(bridge, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(expected)
        rc = bridge.main()
    assert rc == 0
    engrad = tmp_path / "step1_structure_0.engrad"
    assert engrad.is_file()
    text = engrad.read_text(encoding="utf-8")
    assert "Total energy [Eh]" in text
    assert "Gradient [Eh/Bohr]" in text


def test_client_main_tags_calls_with_extinp_jobname(tmp_path: Path, monkeypatch):
    """Without an explicit ``--tag``, ``main`` derives a per-structure tag from the
    ``.extinp.tmp`` jobname so per-call artefacts land in a per-structure file."""
    inp = _write_extinp(tmp_path)  # writes step1_structure_0.extinp.tmp
    url_file = tmp_path / "server.url"
    url_file.write_text("127.0.0.1:1234\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["bridge.py", "--backend", "mlip", "--url-file", str(url_file), str(inp)],
    )
    captured: dict = {}

    class _Resp(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _stub(req, timeout):
        captured["payload"] = json.loads(req.data.decode())
        return _Resp(b'{"energy": -1.0, "gradient": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}')

    with patch.object(bridge, "urlopen", side_effect=_stub):
        rc = bridge.main()
    assert rc == 0
    assert captured["payload"]["tag"] == "step1_structure_0"


# ---------------------------------------------------------------------------
# Server main — end-to-end with fake waitress + mace
# ---------------------------------------------------------------------------


def test_server_main_serves_with_fake_waitress(tmp_path: Path, monkeypatch):
    """``main()`` parses argv, binds a real socket for port discovery, writes the
    sidecar URL, then runs the server **on that pre-bound socket** via
    ``create_server(...).run()`` (not ``waitress.serve``, which would start a
    second server on the default port and leave the advertised port dead)."""
    import sys
    import types
    from unittest.mock import MagicMock, patch

    # Fake mace.calculators so MlipExtOptCalculator.from_args succeeds without
    # pulling a real backend in.
    # All three families, because one builder imports them together — a fake with only the
    # one this test calls would fail at the import rather than at anything it asserts.
    mace_factory = MagicMock(return_value="MACE_OFF_CALC")
    mace_mod = types.ModuleType("mace.calculators")
    mace_mod.mace_off = mace_factory
    mace_mod.mace_mp = MagicMock(return_value="MACE_MP_CALC")
    mace_mod.mace_omol = MagicMock(return_value="MACE_OMOL_CALC")
    mace_parent = types.ModuleType("mace")
    mace_parent.calculators = mace_mod
    monkeypatch.setitem(sys.modules, "mace", mace_parent)
    monkeypatch.setitem(sys.modules, "mace.calculators", mace_mod)

    # Fake waitress.server.create_server -> a server whose .run() blocks IRL.
    fake_server = MagicMock()
    create_server_mock = MagicMock(return_value=fake_server)
    waitress_mod = types.ModuleType("waitress")
    waitress_server_mod = types.ModuleType("waitress.server")
    waitress_server_mod.create_server = create_server_mock
    waitress_mod.server = waitress_server_mod
    monkeypatch.setitem(sys.modules, "waitress", waitress_mod)
    monkeypatch.setitem(sys.modules, "waitress.server", waitress_server_mod)

    # Fake socket so we don't actually bind an OS socket.
    fake_sock = MagicMock()
    fake_sock.getsockname.return_value = ("127.0.0.1", 54321)

    url_file = tmp_path / "server.url"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extopt-server",
            "--backend",
            "mlip",
            "--bind",
            "127.0.0.1:0",
            "--url-file",
            str(url_file),
            "--model",
            "small",
            "--task-name",
            "mace_off",
            "--device",
            "cpu",
        ],
    )

    with patch("socket.socket", return_value=fake_sock):
        rc = server.main()
    assert rc == 0
    create_server_mock.assert_called_once()
    fake_server.run.assert_called_once_with()
    assert url_file.read_text(encoding="utf-8") == "127.0.0.1:54321"
    # main() also writes the per-run bearer token next to the URL sidecar,
    # owner-readable only.
    token_file = tmp_path / "server.token"
    assert token_file.is_file()
    assert len(token_file.read_text(encoding="utf-8").strip()) == 64  # token_hex(32)
    assert token_file.stat().st_mode & 0o777 == 0o600


def test_server_main_logs_why_it_cannot_start_when_the_server_deps_are_missing(monkeypatch, caplog):
    """A missing flask/waitress is a logged, actionable line — not a bare traceback.

    The import used to precede ``logging.basicConfig``, so the crash happened before the
    ``--log-file`` existed: the run block's ``cat "$LOG_FILE"`` printed ``No such file or
    directory`` and the traceback was stranded in the job's ``.err``, which nothing pointed
    at. Probing after logging is configured puts the reason where the failure path already
    looks, and exits nonzero so the readiness loop still bails out fast.
    """
    import importlib.util
    import sys

    real_find_spec = importlib.util.find_spec

    def _no_waitress(name: str, *args: object) -> object:
        return None if name == "waitress" else real_find_spec(name, *args)

    monkeypatch.delitem(sys.modules, "waitress", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", _no_waitress)
    monkeypatch.setattr(sys, "argv", ["extopt-server", "--backend", "mlip"])

    rc = server.main()

    assert rc == 1
    assert "waitress" in caplog.text
    assert "chemrefine[server]" in caplog.text
    assert "backends install" in caplog.text
