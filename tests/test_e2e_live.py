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
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import replay
from _pytest.mark.structures import ParameterSet
from test_e2e_replay import check_recording

from chemrefine import ids
from chemrefine.config import Config, load_config
from chemrefine.engines import _execution
from chemrefine.engines._provision import preflight_backends
from chemrefine.engines.api import get_engine
from chemrefine.engines.orca.engine import OrcaEngine
from chemrefine.errors import ConfigError
from chemrefine.io import read_xyz_frames

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
    """What only a live run of ``pyscf_extopt`` shows: its servers, gradients and tensors.

    Every job, the retry too, started a server of its own, which logged beside it (the first
    attempt's log sealed with it). PySCF printed a gradient header into the job's runlog for
    each call, density-fitted throughout: unrestricted DFT for step 1's LiH+ doublet,
    restricted HF for step 2. Step 2's server saved the one- and two-electron tensors of each
    call under the tag of the ORCA job that asked (the optimiser's, and each NumFreq
    displacement's own), and ``output_dirs`` copied every file it saved home from the scratch
    dir. The Fock matrix built from them over the occupied orbitals is not diagonal, since
    they are Boys-localised rather than canonical, and its occupied-virtual block vanishes,
    as it does at a converged HF solution in any orbitals that span the occupied space.
    """
    budget, freq, _no_answer = config.steps
    runlogs = sorted(config.output_dir.rglob("*.runlog"))
    assert runlogs, "no job left a runlog"
    for runlog in runlogs:
        servers = list(runlog.parent.glob("server_*.log"))
        assert len(servers) == 1, f"{runlog.parent}: {len(servers)} server logs, not one"

    header = re.compile(r"^-+ (\w+) gradients -+$", re.MULTILINE)
    for step, kind in ((budget, "DFUKS"), (freq, "DFRHF")):
        logs = sorted(config.step_dir(step).rglob("*.runlog"))
        assert logs, f"step {step.step} left no runlog"
        for runlog in logs:
            said = set(header.findall(runlog.read_text(encoding="utf-8", errors="replace")))
            assert said == {kind}, f"{runlog.name}: gradients {said}, not {kind}"

    step_dir = config.step_dir(freq)
    tensors = sorted(step_dir.rglob("tensors/*.npz"))
    assert tensors, "save_tensors delivered nothing"
    (home,) = {path.parent.parent for path in tensors}
    (server_log,) = home.glob("server_*.log")
    log = server_log.read_text(encoding="utf-8", errors="replace")
    asked = set(re.findall(r"\[req=(\S+)\] calc backend=pyscf", log))
    saved = {Path(name).name for name in re.findall(r"PySCF tensors saved: (\S+)", log)}
    assert {path.name for path in tensors} == saved == {f"{tag}.npz" for tag in asked}
    assert len(asked) > 1, "the optimiser's calls and each displacement's, tagged apart"

    (lih,) = read_xyz_frames(ids.step_survivors_path(step_dir, freq.step))
    nocc = (int(lih.numbers.sum()) - freq.effective_charge(config.charge)) // 2
    occ = slice(0, nocc)
    for path in tensors:
        with np.load(path, allow_pickle=False) as npz:
            h1e, h2e = npz["h1e"], npz["h2e"]
        coulomb = np.einsum("pqii->pq", h2e[:, :, occ, occ])
        exchange = np.einsum("piiq->pq", h2e[:, occ, occ, :])
        fock = h1e + 2 * coulomb - exchange
        f_oo = fock[occ, occ]
        assert np.abs(f_oo - np.diag(np.diag(f_oo))).max() > 1e-3, f"{path.name}: canonical"
        assert np.abs(fock[occ, nocc:]).max() < 1e-4, f"{path.name}: not a converged HF density"


def _mlip_train(config: Config) -> None:
    """What only a live fit shows: each job's own record of its start, and the libraries' words.

    Each runlog names what its fit started from, in the library's spelling, the default
    included; MACE's own log says it loaded MACE-OFF, not the Materials Project model a
    bare ``small`` would be; and ORB's default loader trained with its stress output off,
    as the job said.
    """
    fits = {
        step.options["task_name"]: config.step_dir(step) / ids.TRAINING_ID
        for step in config.steps
        if step.engine == "mlip-train"
    }
    starts = {}
    for task, run_dir in fits.items():
        (runlog,) = run_dir.glob("*.runlog")
        text = runlog.read_text(encoding="utf-8")
        (starts[task],) = re.findall(r"^\s*started_from=(\S+)\s*$", text, re.MULTILINE)
    assert starts == {
        "mace_off": "small_off",
        "sevenn": "7net-0",
        "chgnet": "0.3.0",
        "orb": "orb_v3_conservative_inf_omat",
    }, starts
    mace_log = "\n".join(
        path.read_text(encoding="utf-8") for path in (fits["mace_off"] / "logs").glob("*.log")
    )
    loaded = re.search(r"Using foundation model (.+?) as initial checkpoint", mace_log)
    assert loaded is not None and "mace-off" in loaded[1], "MACE fine-tuned MACE-OFF"
    (orb_err,) = fits["orb"].glob("*.err")
    assert "carries a stress output" in orb_err.read_text(encoding="utf-8"), "ORB warned"


LIVE_CHECKS: dict[str, Callable[[Config], None]] = {
    "mlip_train": _mlip_train,
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
