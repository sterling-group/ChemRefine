"""The chemistry-grounding tools: structures in, spectroscopic judgment calls out.

``build_structures`` is pinned to ChemRefine's own embedding path (deterministic seed,
UFF clean-up) plus the two sanity checks that catch the classic silent setup errors —
electron-parity vs multiplicity and SMILES formal charge vs the requested charge.
``get_frequencies`` reads only what the step cache persists; ``analyze_mode`` re-parses
a real recorded ORCA frequency output (the NH3 inversion TS contract fixture, one
imaginary mode at -820 cm^-1) through the same parser the pipeline uses, so the numbers
asserted here are the numbers an agent would reason over.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import yaml
from ase import Atoms

from chemrefine import agent_tools, cache
from chemrefine.config import load_config
from chemrefine.errors import ConfigError
from chemrefine.state import StepInputs, StepResults, Structure

_FREQ_OUT = Path(__file__).resolve().parent / "data" / "engines" / "orca" / "freq" / "step1_0.out"


def _write_config(tmp_path: Path, *steps: dict[str, object]) -> Path:
    listed = list(steps) or [{"step": 1, "engine": "fake"}]
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump({"steps": listed}), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# lookup_smiles
# ---------------------------------------------------------------------------


class _CannedResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _CannedResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_lookup_smiles_hits_pubchem_and_trims(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, Any] = {}

    def fake_urlopen(url: str, timeout: float) -> _CannedResponse:
        seen["url"] = url
        return _CannedResponse(b"CC(=O)OC1=CC=CC=C1C(=O)O\n")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = agent_tools.lookup_smiles("aspirin")
    assert result["smiles"] == "CC(=O)OC1=CC=CC=C1C(=O)O"
    assert "aspirin" in seen["url"]


def test_lookup_smiles_offline_names_the_alternative(monkeypatch: pytest.MonkeyPatch):
    def refuse(url: str, timeout: float) -> _CannedResponse:
        raise OSError("network unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(ConfigError, match="pass a SMILES"):
        agent_tools.lookup_smiles("aspirin")


# ---------------------------------------------------------------------------
# build_structures
# ---------------------------------------------------------------------------


def test_build_structures_from_smiles_is_clean_for_a_sane_request(tmp_path: Path):
    result = agent_tools.build_structures(str(tmp_path / "seeds"), smiles=["O"])
    [written] = result["written"]
    assert Path(written).name == "structure_0.xyz"
    assert Path(written).is_file()
    assert result["warnings"] == []


def test_build_structures_flags_charge_and_parity_mistakes(tmp_path: Path):
    """An ammonium SMILES with charge 0 is both a formal-charge and a parity mistake."""
    result = agent_tools.build_structures(str(tmp_path / "seeds"), smiles=["[NH4+]"])
    assert any("formal charge 1 != requested 0" in w for w in result["warnings"])
    assert any("cannot have multiplicity 1" in w for w in result["warnings"])


def test_build_structures_flags_an_impossible_multiplicity(tmp_path: Path):
    result = agent_tools.build_structures(
        str(tmp_path / "seeds"), smiles=["O"], multiplicity=2
    )
    assert any("cannot have multiplicity 2" in w for w in result["warnings"])


def test_build_structures_raises_on_the_named_bad_smiles(tmp_path: Path):
    with pytest.raises(ConfigError, match="invalid SMILES"):
        agent_tools.build_structures(str(tmp_path / "seeds"), smiles=["!!!"])


def test_build_structures_requires_exactly_one_source(tmp_path: Path):
    with pytest.raises(ConfigError, match="exactly one"):
        agent_tools.build_structures(str(tmp_path))
    with pytest.raises(ConfigError, match="exactly one"):
        agent_tools.build_structures(
            str(tmp_path), smiles=["O"], xyz_text="2\n\nH 0 0 0\nH 1 0 0\n"
        )


def test_build_structures_from_xyz_text_validates_through_the_pipelines_reader(tmp_path: Path):
    result = agent_tools.build_structures(
        str(tmp_path / "seeds"),
        xyz_text="3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n",
    )
    [written] = result["written"]
    assert Path(written).is_file()
    assert result["warnings"] == []


def test_build_structures_rejects_malformed_xyz_and_cleans_up(tmp_path: Path):
    with pytest.raises(ConfigError, match="not valid XYZ"):
        agent_tools.build_structures(str(tmp_path / "seeds"), xyz_text="not xyz at all")
    assert not (tmp_path / "seeds" / "structure_0.xyz").exists()


def test_build_structures_parity_checks_each_xyz_frame(tmp_path: Path):
    result = agent_tools.build_structures(
        str(tmp_path / "seeds"), xyz_text="1\na lone hydrogen\nH 0.0 0.0 0.0\n"
    )
    assert any(w.startswith("frame 0") for w in result["warnings"])


# ---------------------------------------------------------------------------
# get_frequencies — cached truth only
# ---------------------------------------------------------------------------


def _cache_step(tmp_path: Path, structures: tuple[Structure, ...]) -> Path:
    path = _write_config(tmp_path, {"step": 1, "engine": "fake"})
    config = load_config(path)
    step_cfg = config.steps[0]
    cache.save(
        step_cfg=step_cfg,
        key=cache.StepKey(parent_ids=(), fingerprint="f", reuse_fingerprint=""),
        results=StepResults(structures=structures),
        step_dir=config.step_dir(step_cfg),
        chemrefine_version="test",
    )
    return path


def _structure(sid: str, imaginary: dict[int, float] | None) -> Structure:
    return Structure(
        id=sid,
        atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
        energy_hartree=-1.0,
        gibbs_hartree=-0.9,
        imaginary_freqs=imaginary,
    )


def test_get_frequencies_reads_the_cached_verdict(tmp_path: Path):
    path = _cache_step(
        tmp_path, (_structure("0", {6: -820.4}), _structure("1", {}), _structure("2", None))
    )
    result = agent_tools.get_frequencies(str(path), 1)
    ts, minimum, unknown = result["structures"]
    assert (ts["imaginary_count"], ts["imaginary_freqs"]) == (1, {"6": -820.4})
    assert (minimum["imaginary_count"], minimum["imaginary_freqs"]) == (0, {})
    assert (unknown["imaginary_count"], unknown["imaginary_freqs"]) == (None, None)
    assert ts["gibbs_hartree"] == -0.9


def test_get_frequencies_filters_to_one_structure(tmp_path: Path):
    path = _cache_step(tmp_path, (_structure("0", {}), _structure("1", {})))
    [only] = agent_tools.get_frequencies(str(path), 1, structure_id="1")["structures"]
    assert only["id"] == "1"
    with pytest.raises(ConfigError, match="no structure '9'"):
        agent_tools.get_frequencies(str(path), 1, structure_id="9")


def test_get_frequencies_without_a_cache_says_run_first(tmp_path: Path):
    path = _write_config(tmp_path)
    with pytest.raises(ConfigError, match="run it"):
        agent_tools.get_frequencies(str(path), 1)


# ---------------------------------------------------------------------------
# analyze_mode — re-parsed truth from a real recorded output
# ---------------------------------------------------------------------------


def _freq_tree(tmp_path: Path) -> Path:
    """A config whose step 1 manifest points at the recorded NH₃ frequency output."""
    path = _write_config(tmp_path, {"step": 1, "engine": "orca"})
    config = load_config(path)
    step_dir = config.step_dir(config.steps[0])
    out = step_dir / "0" / "step1_0.out"
    out.parent.mkdir(parents=True)
    shutil.copy(_FREQ_OUT, out)
    cache.save_manifest(
        StepInputs(files=((out.with_suffix(".inp"), out, "0"),)),
        step_dir,
        operation="freq",
        engine="orca",
    )
    return path


def test_analyze_mode_reads_the_imaginary_mode_off_the_real_output(tmp_path: Path):
    result = agent_tools.analyze_mode(str(_freq_tree(tmp_path)), 1, "0", mode_index=6)
    assert result["is_imaginary"] is True
    assert result["frequency_cm1"] == pytest.approx(-820.38)
    assert result["imaginary_freqs"] == {"6": pytest.approx(-820.38)}
    # The NH₃ inversion coordinate moves every atom; hydrogens dominate the motion.
    symbols = [atom["symbol"] for atom in result["top_atoms"]]
    assert set(symbols) == {"N", "H"}
    assert symbols[0] == "H"
    fractions = [atom["fraction"] for atom in result["top_atoms"]]
    assert fractions == sorted(fractions, reverse=True)
    assert result["bond_changes"], "NH bonds must appear as movers"
    assert all({"atoms", "distance", "rate"} <= set(b) for b in result["bond_changes"])


def test_analyze_mode_speaks_qchem_too(tmp_path: Path):
    """The Q-Chem branch re-parses with Q-Chem's own parser; a minimum's mode is real."""
    path = _write_config(tmp_path, {"step": 1, "engine": "qchem"})
    config = load_config(path)
    step_dir = config.step_dir(config.steps[0])
    out = step_dir / "0" / "step1_0.out"
    out.parent.mkdir(parents=True)
    qchem_freq = _FREQ_OUT.parent.parent.parent / "qchem" / "freq" / "step1_0.out"
    shutil.copy(qchem_freq, out)
    cache.save_manifest(
        StepInputs(files=((out.with_suffix(".in"), out, "0"),)),
        step_dir,
        operation="freq",
        engine="qchem",
    )
    result = agent_tools.analyze_mode(str(path), 1, "0", mode_index=0)
    assert result["is_imaginary"] is False
    assert result["frequency_cm1"] is None
    assert result["imaginary_freqs"] == {}
    assert len(result["top_atoms"]) == 5


def test_analyze_mode_names_a_real_mode_range(tmp_path: Path):
    with pytest.raises(ConfigError, match="out of range"):
        agent_tools.analyze_mode(str(_freq_tree(tmp_path)), 1, "0", mode_index=99)


def test_analyze_mode_refuses_the_missing_pieces(tmp_path: Path):
    no_manifest = _write_config(tmp_path, {"step": 1, "engine": "orca"})
    with pytest.raises(ConfigError, match="no manifest"):
        agent_tools.analyze_mode(str(no_manifest), 1, "0", mode_index=0)

    tree = _freq_tree(tmp_path)
    with pytest.raises(ConfigError, match="no structure 'zz'"):
        agent_tools.analyze_mode(str(tree), 1, "zz", mode_index=0)

    config = load_config(tree)
    (config.step_dir(config.steps[0]) / "0" / "step1_0.out").unlink()
    with pytest.raises(ConfigError, match="no longer exists"):
        agent_tools.analyze_mode(str(tree), 1, "0", mode_index=0)


def test_analyze_mode_refuses_a_non_frequency_output(tmp_path: Path):
    path = _write_config(tmp_path, {"step": 1, "engine": "orca"})
    config = load_config(path)
    step_dir = config.step_dir(config.steps[0])
    out = step_dir / "0" / "step1_0.out"
    out.parent.mkdir(parents=True)
    dft = _FREQ_OUT.parent.parent / "dft" / "step1_0.out"
    shutil.copy(dft, out)
    cache.save_manifest(
        StepInputs(files=((out.with_suffix(".inp"), out, "0"),)),
        step_dir,
        operation="opt_sp",
        engine="orca",
    )
    with pytest.raises(ConfigError, match="no normal-mode tensor"):
        agent_tools.analyze_mode(str(path), 1, "0", mode_index=0)


def test_analyze_mode_refuses_an_unsupported_engine(tmp_path: Path):
    path = _write_config(tmp_path, {"step": 1, "engine": "fake"})
    config = load_config(path)
    step_dir = config.step_dir(config.steps[0])
    out = step_dir / "0" / "step1_0.out"
    out.parent.mkdir(parents=True)
    out.write_text("whatever", encoding="utf-8")
    cache.save_manifest(
        StepInputs(files=((out.with_suffix(".inp"), out, "0"),)),
        step_dir,
        operation=None,
        engine="fake",
    )
    with pytest.raises(ConfigError, match="not supported for engine 'fake'"):
        agent_tools.analyze_mode(str(path), 1, "0", mode_index=0)


def test_step_lookup_failure_is_shared_by_both_tools(tmp_path: Path):
    path = _write_config(tmp_path)
    with pytest.raises(ConfigError, match="no step matches"):
        agent_tools.get_frequencies(str(path), 9)
