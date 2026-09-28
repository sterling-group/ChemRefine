"""Tier-3 live integration: the e2e cases run against the real binaries.

The same case definitions the recordings were made from (``tests/data/e2e/cases``) run here
for real — ORCA local dispatch, the Python backends through their managed envs. Deselected by
default (``addopts = "-m 'not integration'"``); opt in with ``pytest -m integration`` on a
machine with the binaries, and leave out the one case whose load cannot be made small with
``-m "integration and not heavy"``.

What a case needs is read off its config: ORCA for any ORCA-family step, and whatever
:func:`~chemrefine.engines._provision.preflight_backends` requires of the rest — the same
answer the pipeline itself would give. A run is held to its declared budget and to what
every live run must leave (:func:`replay.assert_live_invariants`), and to the facts only a
live run shows (:data:`LIVE_CHECKS`). Then it is packed and the pack is replayed and checked
like any shipped recording; only an archive that passes replaces one, under ``--record``.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import replay
from _pytest.mark.structures import ParameterSet
from test_e2e_replay import check_recording

from chemrefine.config import Config, load_config
from chemrefine.engines import _execution
from chemrefine.engines._provision import preflight_backends
from chemrefine.engines.api import get_engine
from chemrefine.engines.orca.engine import OrcaEngine
from chemrefine.errors import ConfigError

pytestmark = pytest.mark.integration


def _real_orca() -> str | None:
    """The quantum-chemistry ORCA, or None.

    `which("orca")` alone is not enough: desktop Linux ships /usr/bin/orca —
    the GNOME screen reader. Real ORCA installs keep their helper binaries
    (orca_2json, otool_xtb, …) beside the main executable, so require one.
    """
    found = shutil.which("orca")
    if found is None:
        return None
    install_dir = Path(found).resolve().parent
    if not (install_dir / "orca_2json").exists():
        return None
    return found


_ORCA = _real_orca()
"""Resolved once: it reads ``PATH``, which no fixture rewrites."""


def _cases() -> list[ParameterSet]:
    """Every case directory, marked ``heavy`` where its ``case.yaml`` says so."""
    return [
        pytest.param(
            case.name,
            marks=[pytest.mark.heavy] if replay.CaseSpec.read(case).heavy else [],
            id=case.name,
        )
        for case in sorted(replay.CASES_DIR.iterdir())
        if (case / "input.yaml").is_file()
    ]


def _needs_orca(config: Config) -> bool:
    return any(isinstance(get_engine(step.engine), OrcaEngine) for step in config.steps)


def _skip_unless_runnable(config: Config) -> None:
    """Skip a case this machine cannot run — or fail, under ``CHEMREFINE_REQUIRE_LIVE``.

    Skipping is right for a developer running the tier on a laptop with only some of the
    stacks installed. It is wrong for the release gate, whose whole purpose is that the
    real binaries ran: set the variable and a missing stack fails by name instead of
    quietly shrinking what the gate covered. The backends are judged by the pipeline's
    own preflight, so a case is skipped exactly when a run of it would be refused.
    """
    missing = []
    if _needs_orca(config) and _ORCA is None:
        missing.append("ORCA on PATH")
    try:
        preflight_backends(config.steps)
    except ConfigError as e:
        missing.append(str(e))
    if not missing:
        return
    message = f"live case needs {'; '.join(missing)}"
    if os.environ.get("CHEMREFINE_REQUIRE_LIVE"):
        pytest.fail(f"{message} — CHEMREFINE_REQUIRE_LIVE is set, so this may not be skipped")
    pytest.skip(message)


def _pyscf_extopt(config: Config) -> None:
    """``save_tensors`` delivers through ``output_dirs``, the one copy-back path that moves a
    whole directory; the tensors are not archived, so only the live run can show them."""
    assert list(config.output_dir.rglob("tensors/*.npz")), "save_tensors delivered nothing"


LIVE_CHECKS: dict[str, Callable[[Config], None]] = {
    "pyscf_extopt": _pyscf_extopt,
}
"""Facts a live run shows that its recording cannot carry, by case."""


@pytest.mark.parametrize("name", _cases())
def test_live_case(
    name: str, tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the case's scenario for real, hold it to its budget and invariants, pack, replay.

    Every case runs under :class:`replay.SubmissionRecorder`, recording or not, so the log a
    recording needs is made the way every live run makes it.
    """
    run_dir = tmp_path / "run"
    shutil.copytree(replay.CASES_DIR / name, run_dir)
    spec = replay.CaseSpec.read(run_dir)
    config = load_config(run_dir / "input.yaml")
    _skip_unless_runnable(config)
    if _needs_orca(config):
        assert _ORCA is not None
        # ORCA needs its absolute path for MPI; the case config ships a placeholder.
        config = config.model_copy(
            update={"executables": {**config.executables, "orca": str(Path(_ORCA).resolve())}}
        )

    recorder = replay.SubmissionRecorder(_execution.run_batch, config.output_dir)
    monkeypatch.setattr(_execution, "run_batch", recorder)
    started = time.monotonic()
    try:
        replay.drive(config, spec)
    finally:
        recorder.write(run_dir)
    spec.assert_within_budget(time.monotonic() - started)

    replay.assert_live_invariants(config, run_dir)
    if name in LIVE_CHECKS:
        LIVE_CHECKS[name](config)

    archive = replay.pack_case(run_dir, name, dest_dir=tmp_path / "pack")
    check_recording(archive, tmp_path / "replayed", monkeypatch)
    if request.config.getoption("--record"):
        shutil.copy2(archive, replay.DATA_DIR / archive.name)
