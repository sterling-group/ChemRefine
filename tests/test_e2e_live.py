"""Tier-3 live integration: the e2e cases run against the real binaries.

The same case definitions the recorded tier replays (``tests/data/e2e_cases``)
run here for real — ORCA local dispatch, MLIP backends via their managed envs.
Deselected by default (``addopts = "-m 'not integration'"``); opt in with
``pytest -m integration`` on a machine with the binaries (ORCA ≥ 6 on PATH,
≥ 4 physical cores for the %pal values, the mace stack for the MLIP cases).
Passing runs re-pack the recorded archives when ``--record`` is given —
recording and replay share one case definition, so they cannot drift.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest
import replay

from chemrefine import pipeline
from chemrefine.config import load_config
from chemrefine.engines import backend_env_python

pytestmark = pytest.mark.integration

CASES_DIR = Path(__file__).resolve().parent / "data" / "e2e_cases"


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
_MACE = importlib.util.find_spec("mace") is not None or backend_env_python("mlip-mace").is_file()

_REQUIRES = {
    "conformers": {"orca"},
    "nms_minimum": {"orca"},
    "ts_pes": {"orca"},
    "hostguest": {"orca"},
    "mlip_screen": {"mace"},
    "mlip_extopt": {"orca", "mace"},
}


def _skip_unless_available(requirements: set[str]) -> None:
    missing = []
    if "orca" in requirements and _ORCA is None:
        missing.append("ORCA on PATH")
    if "mace" in requirements and not _MACE:
        missing.append("a mace stack (importable or `chemrefine backends install mlip-mace`)")
    if missing:
        pytest.skip(f"live case needs {', '.join(missing)}")


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
    elif name == "hostguest":
        assert outcomes[1].state.structures, "solvator output must parse"

    if request.config.getoption("--record"):
        archive = replay.pack_case(tmp_path, name)
        assert archive.is_file()
