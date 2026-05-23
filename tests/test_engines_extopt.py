"""Tests for the shared ``engines/_extopt`` infrastructure.

Covers:

* :mod:`chemrefine.engines._extopt.protocol` — ``.extinp.tmp``,
  ``.engrad``, wrapper script, and sidecar URL helpers.
* :mod:`chemrefine.engines._extopt.server` — CLI parsing + Flask app
  factory (with a mock backend).
* :mod:`chemrefine.engines._extopt.client` — CLI parsing, HTTP RPC,
  and the ``main`` glue end-to-end.
* :mod:`chemrefine.engines._extopt.registry` — every registered
  backend imports + conforms to the BaseExtOptCalculator contract.
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from chemrefine.engines._extopt import client, protocol, registry, server
from chemrefine.engines._extopt.base import (
    DEFAULT_BIND_HOST,
    SERVER_URL_FILENAME,
    CalculationData,
)
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
        path=out, n_atoms=1, energy_hartree=-0.5,
        gradients_hartree_per_bohr=None, dograd=False,
    )
    text = out.read_text(encoding="utf-8")
    assert "Gradient" not in text


def test_write_engrad_dograd_true_requires_gradient(tmp_path: Path):
    with pytest.raises(ValueError, match="dograd=True but no gradient"):
        protocol.write_engrad(
            path=tmp_path / "x.engrad", n_atoms=1, energy_hartree=0.0,
            gradients_hartree_per_bohr=None, dograd=True,
        )


def test_write_wrapper_script_emits_exec_call_and_is_executable(tmp_path: Path):
    out = tmp_path / "mlff_extopt.sh"
    url_file = tmp_path / "server.url"
    protocol.write_wrapper_script(
        path=out, backend="mlff", url_file=url_file,
    )
    text = out.read_text(encoding="utf-8")
    assert "#!/usr/bin/env bash" in text
    assert 'URL_FILE="' + str(url_file) + '"' in text
    assert "chemrefine.engines._extopt.client" in text
    assert "--backend mlff" in text
    assert out.stat().st_mode & 0o100  # owner-execute bit


def test_write_wrapper_script_threads_extra_args(tmp_path: Path):
    out = tmp_path / "pyscf_extopt.sh"
    protocol.write_wrapper_script(
        path=out, backend="pyscf", url_file=tmp_path / "u",
        extra_args="--method dft --xc pbe",
    )
    assert "--method dft --xc pbe" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Sidecar URL file
# ---------------------------------------------------------------------------


def test_write_and_read_server_url_round_trip(tmp_path: Path):
    target = tmp_path / "subdir" / "server.url"
    protocol.write_server_url(target, "127.0.0.1:54321")
    assert protocol.read_server_url(target) == "127.0.0.1:54321"


def test_write_server_url_cleans_up_temp_on_failure(tmp_path: Path):
    """A rename failure should remove the tempfile (no leftovers)."""
    target = tmp_path / "server.url"
    with patch("os.replace", side_effect=OSError("denied")), pytest.raises(OSError):
        protocol.write_server_url(target, "x:1")
    # Ensure no tempfiles linger
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".url.")]
    assert leftovers == []


# ---------------------------------------------------------------------------
# atoms_to_payload
# ---------------------------------------------------------------------------


def test_atoms_to_payload_converts_units(tmp_path: Path):
    """eV → Hartree for energy, eV/Å → Hartree/Bohr for gradient."""

    class _FakeAtoms:
        def get_potential_energy(self):
            return -27.211386245988  # exactly -1 Hartree

        def get_forces(self):
            return np.array([[1.0, 0.0, 0.0]])

    energy_h, grad = protocol.atoms_to_payload(_FakeAtoms())
    assert energy_h == pytest.approx(-1.0, rel=1e-12)
    # 1 eV/Å force → -1 eV/Å gradient → in Hartree/Bohr:
    assert grad[0][0] == pytest.approx(-1.0 * 0.529177210903 / 27.211386245988, rel=1e-12)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_mlff_and_pyscf():
    assert set(registry.CALCULATORS) == {"mlff", "pyscf"}


def test_load_calculator_returns_class():
    cls = registry.load_calculator("mlff")
    assert isinstance(cls, type)


def test_load_calculator_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        registry.load_calculator("not_a_backend")


def test_every_registered_backend_conforms_to_base_protocol():
    """Loaded backends should be ``BaseExtOptCalculator``-conformant classes.

    We only check class-level shape (``name`` + ``calc`` + ``from_args``)
    here — real instantiation requires backend dependencies (torch /
    pyscf) the test env doesn't install.
    """
    for name in registry.CALCULATORS:
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
    forbidden = re.compile(
        r'--(?:model|task-name|device|model-path|method|xc|basis|df|gpu)\b'
    )
    matches = forbidden.findall(text)
    assert matches == [], f"shared server leaks backend CLI flags: {matches}"


def test_shared_client_source_has_no_backend_specific_flags():
    """The shared client module must not enumerate any backend's CLI flags."""
    import re

    text = Path(client.__file__).read_text(encoding="utf-8")
    forbidden = re.compile(
        r'--(?:model|task-name|device|model-path|method|xc|basis|df|gpu)\b'
    )
    matches = forbidden.findall(text)
    assert matches == [], f"shared client leaks backend CLI flags: {matches}"


# ---------------------------------------------------------------------------
# Server CLI
# ---------------------------------------------------------------------------


def test_server_parse_args_requires_backend():
    with pytest.raises(SystemExit):
        server.parse_args([])


def test_server_parse_args_defaults():
    args = server.parse_args(["--backend", "mlff"])
    assert args.backend == "mlff"
    assert args.bind == "127.0.0.1:0"
    assert args.nthreads == 4
    assert args.log_level == "INFO"
    assert args.method == "dft"  # PySCF default available regardless of backend


def test_server_parse_args_accepts_all_overrides():
    args = server.parse_args([
        "--backend", "pyscf",
        "--bind", "127.0.0.1:54321",
        "--url-file", "/tmp/foo",
        "--log-file", "/tmp/srv.log",
        "--log-level", "DEBUG",
        "--method", "hf",
        "--xc", "b3lyp",
        "--basis", "cc-pvdz",
        "--df",
        "--gpu",
    ])
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
            "charge": 0, "mult": 1, "nthreads": 1, "dograd": True,
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["energy"] == -1.0
    assert data["gradient"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_create_app_calculate_route_returns_500_on_calculator_error():
    class _Boom:
        name = "boom"

        def calc(self, data):
            raise RuntimeError("backend exploded")

    app = server.create_app(_Boom())
    resp = app.test_client().post(
        "/calculate",
        json={
            "atom_types": ["H"], "coordinates": [[0, 0, 0]],
            "charge": 0, "mult": 1, "nthreads": 1,
        },
    )
    assert resp.status_code == 500
    assert "backend exploded" in resp.get_json()["error"]


def test_create_app_logs_with_correlation_tag(caplog):
    """``payload['tag']`` should appear in the server's INFO log line."""
    import logging

    caplog.set_level(logging.INFO, logger="chemrefine.engines._extopt.server")
    app = server.create_app(_MockCalculator())
    app.test_client().post(
        "/calculate",
        json={
            "atom_types": ["H"], "coordinates": [[0, 0, 0]],
            "charge": 0, "mult": 1, "nthreads": 1, "tag": "req-abc",
        },
    )
    assert any("req=req-abc" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Client CLI + HTTP RPC
# ---------------------------------------------------------------------------


def test_client_parse_args_requires_backend_and_inputfile():
    with pytest.raises(SystemExit):
        client.parse_args(["job.extinp.tmp"])  # missing --backend


def test_client_parse_args_defaults():
    args = client.parse_args(["--backend", "mlff", "job.extinp.tmp"])
    assert args.backend == "mlff"
    assert args.bind is None
    assert args.url_file is None
    assert args.method == "dft"
    assert args.inputfile == "job.extinp.tmp"


def test_client_settings_from_args_round_trip():
    args = client.parse_args(
        ["--backend", "pyscf", "--method", "hf", "--basis", "cc-pvdz", "--df", "--gpu", "f"]
    )
    settings = client.settings_from_args(args)
    assert settings == {
        "method": "hf",
        "xc": "pbe",
        "basis": "cc-pvdz",
        "df": True,
        "gpu": True,
    }


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
    with patch.object(client, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(expected)
        energy, gradient = client.submit_calculation(
            server_url="127.0.0.1:54321", data=_data(),
        )
    assert energy == -1.5
    assert gradient == [[0.0, 0.0, 0.0]]


def test_submit_calculation_http_error_becomes_jobfailure():
    from urllib.error import HTTPError

    err = HTTPError("http://x/calculate", 500, "internal", {}, None)
    with (
        patch.object(client, "urlopen", side_effect=err),
        pytest.raises(JobFailureError, match="HTTP 500"),
    ):
        client.submit_calculation(server_url="x", data=_data())


def test_submit_calculation_url_error_becomes_jobfailure():
    from urllib.error import URLError

    with (
        patch.object(client, "urlopen", side_effect=URLError("refused")),
        pytest.raises(JobFailureError, match="unreachable"),
    ):
        client.submit_calculation(server_url="x", data=_data())


def test_submit_calculation_non_json_response_becomes_jobfailure():
    with patch.object(client, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(b"<html>")
        with pytest.raises(JobFailureError, match="non-JSON"):
            client.submit_calculation(server_url="x", data=_data())


def test_submit_calculation_error_field_becomes_jobfailure():
    with patch.object(client, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(b'{"error": "boom"}')
        with pytest.raises(JobFailureError, match="boom"):
            client.submit_calculation(server_url="x", data=_data())


def test_submit_calculation_missing_fields_becomes_jobfailure():
    with patch.object(client, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(b'{"foo": 1}')
        with pytest.raises(JobFailureError, match="missing fields"):
            client.submit_calculation(server_url="x", data=_data())


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

    with patch.object(client, "urlopen", side_effect=_fake_urlopen):
        client.submit_calculation(server_url="x", data=_data(), tag="abc-123")
    assert captured["payload"]["tag"] == "abc-123"


def test_resolve_server_url_prefers_explicit_bind(tmp_path: Path):
    args = client.parse_args(["--backend", "mlff", "--bind", "1.2.3.4:5", "x"])
    assert client.resolve_server_url(args) == "1.2.3.4:5"


def test_resolve_server_url_reads_url_file(tmp_path: Path):
    url_file = tmp_path / "server.url"
    url_file.write_text("127.0.0.1:9999\n", encoding="utf-8")
    args = client.parse_args(["--backend", "mlff", "--url-file", str(url_file), "x"])
    assert client.resolve_server_url(args) == "127.0.0.1:9999"


def test_client_engrad_path_for_strips_extinp_tmp():
    assert client._engrad_path_for("a/b/foo.extinp.tmp") == Path("a/b/foo.engrad")


def test_client_engrad_path_for_falls_back_to_with_suffix():
    """An input without ``.extinp.tmp`` still gets ``.engrad`` via ``with_suffix``."""
    assert client._engrad_path_for("foo.tmp") == Path("foo.engrad")


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
        ["client.py", "--backend", "mlff", "--url-file", str(url_file), str(inp)],
    )

    expected = b'{"energy": -1.0, "gradient": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}'
    with patch.object(client, "urlopen") as mock_open:
        mock_open.return_value.__enter__.return_value = BytesIO(expected)
        rc = client.main()
    assert rc == 0
    engrad = tmp_path / "step1_structure_0.engrad"
    assert engrad.is_file()
    text = engrad.read_text(encoding="utf-8")
    assert "Total energy [Eh]" in text
    assert "Gradient [Eh/Bohr]" in text


# ---------------------------------------------------------------------------
# Server main — end-to-end with fake waitress + mace
# ---------------------------------------------------------------------------


def test_server_main_serves_with_fake_waitress(tmp_path: Path, monkeypatch):
    """``main()`` parses argv, binds a real socket for port discovery,
    writes the sidecar URL, then calls ``waitress.serve(server)``."""
    import sys
    import types
    from unittest.mock import MagicMock, patch

    # Fake mace.calculators so MlffExtOptCalculator.from_args succeeds without
    # pulling a real backend in.
    mace_factory = MagicMock(return_value="MACE_OFF_CALC")
    mace_mod = types.ModuleType("mace.calculators")
    mace_mod.mace_off = mace_factory
    mace_parent = types.ModuleType("mace")
    mace_parent.calculators = mace_mod
    monkeypatch.setitem(sys.modules, "mace", mace_parent)
    monkeypatch.setitem(sys.modules, "mace.calculators", mace_mod)

    # Fake waitress + waitress.server.
    fake_server = MagicMock()
    create_server_mock = MagicMock(return_value=fake_server)
    serve_mock = MagicMock()
    waitress_mod = types.ModuleType("waitress")
    waitress_mod.serve = serve_mock
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
        sys, "argv",
        [
            "extopt-server",
            "--backend", "mlff",
            "--bind", "127.0.0.1:0",
            "--url-file", str(url_file),
            "--model", "medium",
            "--task-name", "mace_off",
            "--device", "cpu",
        ],
    )

    with patch("socket.socket", return_value=fake_sock):
        rc = server.main()
    assert rc == 0
    create_server_mock.assert_called_once()
    serve_mock.assert_called_once_with(fake_server)
    assert url_file.read_text(encoding="utf-8") == "127.0.0.1:54321"
