"""Tier-3 live integration: the e2e cases run against the real binaries.

The same case definitions the recordings were made from (``tests/data/e2e/cases``)
run here for real — ORCA local dispatch, MLIP backends via their managed envs.
Deselected by default (``addopts = "-m 'not integration'"``); opt in with
``pytest -m integration`` on a machine with the binaries (ORCA ≥ 6 on PATH,
≥ 4 physical cores for the %pal values, the mace stack for the MLIP cases).
Passing runs re-pack the recorded archives when ``--record`` is given —
recording and replay share one case definition, so they cannot drift.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import pytest
import replay

from chemrefine import pipeline
from chemrefine.config import load_config
from chemrefine.engines import backend_env_python

pytestmark = pytest.mark.integration

CASES_DIR = Path(__file__).resolve().parent / "data" / "e2e" / "cases"


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


def _backend_available(import_name: str, extra: str) -> bool:
    """Whether a compute backend can run here — importable, or a managed env exists.

    Deliberately the same two conditions :func:`chemrefine.engines._provision.require_backend`
    accepts, so a case is skipped exactly when the pipeline would refuse to start it, and
    never when it would have run.

    Called per test rather than answered once at import, because ``$CHEMREFINE_HOME`` decides
    where a managed env is looked for and fixtures run after this module is imported. Sampled
    at import, the answer is the one from *before* the environment the case will run in, and
    the agreement with ``require_backend`` that this function exists for is only an agreement
    when both read the same environment.
    """
    return importlib.util.find_spec(import_name) is not None or backend_env_python(extra).is_file()


_ORCA = _real_orca()
"""Resolved once: it reads ``PATH``, which no fixture rewrites."""

_REQUIRES = {
    "conformers": {"orca"},
    "nms_minimum": {"orca"},
    "ts_pes": {"orca"},
    "host_guest": {"orca"},
    "mlip_screen": {"mace"},
    "mlip_train": {"mace"},
    "mlip_extopt": {"orca", "mace"},
    "pyscf_sp": {"pyscf"},
    "pyscf_extopt": {"orca", "pyscf"},
}


def _skip_unless_available(requirements: set[str]) -> None:
    """Skip a case whose backend is absent — or fail, under ``CHEMREFINE_REQUIRE_LIVE``.

    Skipping is right for a developer running the tier on a laptop with only some of the
    stacks installed. It is wrong for the release gate, whose whole purpose is that the
    real binaries ran: ``release-check.sh`` guards ORCA by hand precisely because "the
    tier-3 suite skips every live case and the gate passes having tested nothing", but the
    same hole was open for every other backend. Setting the variable closes it — a missing
    stack then fails by name instead of quietly shrinking what the gate covered.
    """
    missing = []
    if "orca" in requirements and _ORCA is None:
        missing.append("ORCA on PATH")
    if "mace" in requirements and not _backend_available("mace", "mlip-mace"):
        missing.append("a mace stack (importable or `chemrefine backends install mlip-mace`)")
    if "pyscf" in requirements and not _backend_available("pyscf", "pyscf"):
        missing.append("pyscf (importable or `chemrefine backends install pyscf`)")
    if not missing:
        return
    message = f"live case needs {', '.join(missing)}"
    if os.environ.get("CHEMREFINE_REQUIRE_LIVE"):
        pytest.fail(f"{message} — CHEMREFINE_REQUIRE_LIVE is set, so this may not be skipped")
    pytest.skip(message)


@pytest.mark.parametrize("name", sorted(_REQUIRES))
def test_live_case(name: str, tmp_path: Path, request: pytest.FixtureRequest) -> None:
    """Run the case for real, assert the workflow invariants, optionally record."""
    requirements = _REQUIRES[name]
    _skip_unless_available(requirements)
    shutil.copytree(CASES_DIR / name, tmp_path, dirs_exist_ok=True)

    config = load_config(tmp_path / "input.yaml")
    if "orca" in requirements:
        assert _ORCA is not None
        # ORCA needs its absolute path for MPI; the case config ships a placeholder.
        config = config.model_copy(
            update={"executables": {**config.executables, "orca": str(Path(_ORCA).resolve())}}
        )

    outcomes = pipeline.run(config)

    assert outcomes[-1].state.structures, "the live pipeline must end with survivors"
    outputs = tmp_path / "outputs"
    assert list(outputs.rglob("*.result.json")), "every parsed job leaves its canonical record"
    if "orca" in requirements:
        assert list(outputs.rglob("*.property.json")), "ORCA >= 6 drops its native property.json"

    if name == "conformers":
        ensemble = next(outputs.glob("step1/0/*.finalensemble.xyz"))
        assert len(ensemble.read_text().splitlines()) >= 3 * 12, "GOAT must yield >= 3 conformers"
        assert all(s.gibbs_hartree is not None for s in outcomes[1].state.structures)
    elif name == "nms_minimum":
        assert list((outputs / "step1" / "0" / "attempt1").iterdir()), "NMS round 2 must run"
        assert not outcomes[0].state.structures[0].imaginary_freqs
    elif name == "ts_pes":
        (ts,) = outcomes[1].state.structures
        assert ts.imaginary_freqs is not None and len(ts.imaginary_freqs) == 1
    elif name == "host_guest":
        assert outcomes[1].state.structures, "solvator output must parse"
    elif name in ("pyscf_sp", "pyscf_extopt"):
        # Water/HF/STO-3G is ~-74.96 Eh. Asserting the number, not just that something
        # parsed, is the point of running PySCF for real: the contract goldens for both
        # engines were hand-made from our own footer format rather than captured, so
        # until now nothing had checked that this backend computes anything at all.
        (structure,) = outcomes[0].state.structures
        assert structure.energy_hartree is not None
        assert -75.5 < structure.energy_hartree < -74.5, structure.energy_hartree

    if request.config.getoption("--record"):
        archive = replay.pack_case(tmp_path, name)
        assert archive.is_file()
