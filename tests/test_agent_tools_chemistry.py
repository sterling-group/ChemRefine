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
    result = agent_tools.build_structures(str(tmp_path / "seeds"), smiles=["O"], multiplicity=2)
    assert any("cannot have multiplicity 2" in w for w in result["warnings"])


def test_build_structures_raises_on_the_named_bad_smiles(tmp_path: Path):
    with pytest.raises(ConfigError, match="invalid SMILES"):
        agent_tools.build_structures(str(tmp_path / "seeds"), smiles=["!!!"])


def test_a_bad_smiles_late_in_the_list_leaves_no_partial_seed_set(tmp_path: Path):
    """A list that fails partway is cleaned up like the XYZ branch always was.

    The loop writes as it goes, so a bad SMILES at index 2 raised with structures 0 and 1
    already on disk — and a corrected retry with a shorter list then reported one file
    while ``input:`` directory-seeding read two, seeding a molecule the successful call
    never produced.
    """
    seeds = tmp_path / "seeds"
    with pytest.raises(ConfigError, match="invalid SMILES"):
        agent_tools.build_structures(str(seeds), smiles=["O", "CCO", "!!!"])
    assert list(seeds.glob("structure_*.xyz")) == []


def test_a_new_build_clears_the_previous_calls_seed_set(tmp_path: Path):
    """A call owns the whole seed set — a longer earlier call's extras must not survive.

    ``structure_{i}.xyz`` names are positional, so a shorter second call rewrites the
    front of the set and, without the clear, leaves the tail: two files on disk, one in
    ``written``, and a directory-seeded run computing on both.
    """
    seeds = tmp_path / "seeds"
    agent_tools.build_structures(str(seeds), smiles=["O", "CCO"])
    result = agent_tools.build_structures(str(seeds), smiles=["O"])
    assert [Path(p).name for p in result["written"]] == ["structure_0.xyz"]
    assert [p.name for p in seeds.glob("structure_*.xyz")] == ["structure_0.xyz"]


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
        key=cache.StepKey(parent_ids=(), fingerprint="f"),
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


@pytest.mark.parametrize(("top_atoms", "expected"), [(2, 2), (0, 0), (-1, 0), (-3, 0)])
def test_a_negative_top_atoms_is_no_atoms_never_all_but_some(
    tmp_path: Path, top_atoms: int, expected: int
):
    """``[:top_atoms]`` on a negative counts from the end — here, the wrong end.

    NH₃ has four atoms, so ``top_atoms=-1`` returned three of them: every atom *except*
    the least-displaced one, dressed as a shorter list, on the tool whose entire job is
    to report which atoms move most. Same inversion the pagination parameters had, same
    clamp.
    """
    result = agent_tools.analyze_mode(
        str(_freq_tree(tmp_path)), 1, "0", mode_index=6, top_atoms=top_atoms
    )
    assert len(result["top_atoms"]) == expected


def test_get_structure_returns_extended_xyz_from_the_cache(tmp_path: Path):
    """Geometry out of the step cache, in the one format that carries everything.

    Extended XYZ because the same text has to serve three cases the viewer cannot tell
    apart in advance: a molecule, a periodic cell (``Lattice="…"``), and a normal mode
    (three more columns per atom). Nothing here needs an output file — symbols and
    positions are persisted with every parsed structure.
    """
    path = _cache_step(tmp_path, (_structure("0", {}), _structure("1", {})))
    result = agent_tools.get_structure(str(path), 1)
    assert result["format"] == "extxyz"
    assert result["structure_id"] == "0"  # the first, with no id asked for
    lines = result["text"].splitlines()
    assert lines[0] == "2"  # the H2 the fixture caches
    assert "Properties=species:S:1:pos:R:3" in lines[1]
    assert len(lines[2].split()) == 4  # symbol + xyz, no displacement columns

    assert agent_tools.get_structure(str(path), 1, structure_id="1")["structure_id"] == "1"
    with pytest.raises(ConfigError, match="no structure '9'"):
        agent_tools.get_structure(str(path), 1, structure_id="9")


def test_get_structure_without_a_cache_says_run_first(tmp_path: Path):
    with pytest.raises(ConfigError, match="run it"):
        agent_tools.get_structure(str(_write_config(tmp_path)), 1)


def test_get_structure_on_a_step_that_kept_nothing(tmp_path: Path):
    """A cache with no survivors is not the same as no cache, and says so differently.

    A filter can leave a step with zero structures — that ran, and it kept nothing.
    Telling the caller to run it again would send them round a loop that changes nothing.
    """
    path = _cache_step(tmp_path, ())
    with pytest.raises(ConfigError, match="cached no structures"):
        agent_tools.get_structure(str(path), 1)


def test_a_displacement_array_must_match_the_atom_count(tmp_path: Path):
    """One 3-vector per atom, or the writer refuses rather than mis-pairing them.

    ase would happily attach a shorter array and raise something obscure later, or pair
    displacements with the wrong atoms — the failure this exists to make loud.
    """
    import numpy as np

    from chemrefine import io as crio

    water = Atoms("H2O", positions=[[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]])
    with pytest.raises(ValueError, match="one 3-vector per atom"):
        crio.extended_xyz_text(water, displacements=np.zeros((2, 3)))


def test_a_periodic_structure_carries_its_cell(tmp_path: Path):
    """The condensed-matter path, such as it is today.

    ChemRefine's own pipeline is molecular — nothing in it sets a cell, and the step cache
    does not round-trip one — so this asserts the *writer* rather than a cached structure:
    when a structure does have a cell, the text says so, and that is the whole of what the
    viewer needs to draw a box.
    """
    from ase import Atoms

    from chemrefine import io as crio

    periodic = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]], cell=[5, 5, 5], pbc=True)
    text = crio.extended_xyz_text(periodic)
    assert 'Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0"' in text
    assert 'pbc="T T T"' in text
    # A molecule says so just as explicitly, rather than omitting the key.
    assert 'pbc="F F F"' in crio.extended_xyz_text(Atoms("H", positions=[[0, 0, 0]]))


def test_get_structure_with_a_mode_carries_displacement_columns(tmp_path: Path):
    """The animated normal mode: seven columns, which is what 3Dmol reads as dx/dy/dz.

    The tensor is deliberately not cached — it is a transient the pipeline displaces
    along — so this re-parses the recorded output exactly as ``analyze_mode`` does, and
    the two share one frame helper so they cannot disagree about which frame they mean.
    """
    path = _freq_tree(tmp_path)
    result = agent_tools.get_structure(str(path), 1, structure_id="0", mode_index=6)
    lines = result["text"].splitlines()
    assert "displacement:R:3" in lines[1]
    assert len(lines[2].split()) == 7
    assert result["mode_index"] == 6

    # The same range check analyze_mode applies, from the same helper.
    with pytest.raises(ConfigError, match="out of range"):
        agent_tools.get_structure(str(path), 1, structure_id="0", mode_index=99)

    # A mode belongs to one structure, so asking for one without saying which is a
    # refusal rather than a guess at the first.
    with pytest.raises(ConfigError, match="needs a structure_id"):
        agent_tools.get_structure(str(path), 1, mode_index=6)


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
