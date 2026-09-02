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
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import yaml
from ase import Atoms

from chemrefine import agent_tools, cache
from chemrefine.config import load_config
from chemrefine.engines.orca.output import coordinator as orca_coordinator
from chemrefine.errors import ConfigError, EngineNotFoundError
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

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _CannedResponse:
        seen["request"] = request
        return _CannedResponse(b"CC(=O)OC1=CC=CC=C1C(=O)O\n")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = agent_tools.lookup_smiles("aspirin")
    assert result["smiles"] == "CC(=O)OC1=CC=CC=C1C(=O)O"
    assert "aspirin" in seen["request"].full_url


def test_lookup_smiles_identifies_chemrefine_to_pubchem(monkeypatch: pytest.MonkeyPatch):
    """NCBI's usage policy asks callers to say who they are; urllib will not by default.

    ``get_header`` with the title-cased spelling because that is how urllib stores it.
    """
    from chemrefine import USER_AGENT

    seen: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _CannedResponse:
        seen["request"] = request
        return _CannedResponse(b"CCO\n")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    agent_tools.lookup_smiles("ethanol")
    assert seen["request"].get_header("User-agent") == USER_AGENT
    assert "urllib" not in USER_AGENT


def test_lookup_smiles_offline_names_the_alternative(monkeypatch: pytest.MonkeyPatch):
    def refuse(request: urllib.request.Request, timeout: float) -> _CannedResponse:
        raise OSError("network unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(ConfigError, match="pass a SMILES"):
        agent_tools.lookup_smiles("aspirin")


def test_lookup_smiles_survives_a_truncated_reply(monkeypatch: pytest.MonkeyPatch):
    """IncompleteRead — a reply that arrives and then breaks — gets the same offline hint.

    `http.client.HTTPException` is neither an OSError nor a URLError, so a dropped
    connection mid-body escaped as a traceback where every other network failure became
    the actionable ConfigError.
    """
    from http.client import IncompleteRead

    def die_mid_body(request: urllib.request.Request, timeout: float) -> _CannedResponse:
        raise IncompleteRead(b"CC")

    monkeypatch.setattr(urllib.request, "urlopen", die_mid_body)
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


@pytest.mark.parametrize("empty", ["", "\n\n"])
def test_build_structures_refuses_an_xyz_text_with_no_frames(tmp_path: Path, empty: str):
    """Zero frames is a failure here, not a zero-byte "seed set" reported as written.

    An empty string parses without error, so the tool answered ``written:
    [structure_0.xyz]`` (0 bytes) with no warnings — a success against its own
    fails-here contract, deferring the failure to a step with nothing to compute.
    """
    with pytest.raises(ConfigError, match="no structures"):
        agent_tools.build_structures(str(tmp_path / "seeds"), xyz_text=empty)
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


def _seeded_config(tmp_path: Path, **extra: object) -> Path:
    """A config whose ``input:`` points at a directory of two seed files."""
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "a.xyz").write_text("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n", "utf-8")
    (seeds / "b.xyz").write_text("2\nh2\nH 0 0 0\nH 0.74 0 0\n", "utf-8")
    path = tmp_path / "input.yaml"
    path.write_text(
        yaml.safe_dump({"input": "seeds", "steps": [{"step": 1, "engine": "orca"}], **extra}),
        encoding="utf-8",
    )
    return path


def test_the_input_seeds_are_viewable_before_anything_has_run(tmp_path: Path):
    """The one view that works on a tree nothing has computed in.

    Numbered by ``pipeline.bootstrap``, not by a second reader written here, so the id a
    user inspects as ``1`` is the ``1`` that turns up in ``steps.csv`` afterwards — and so
    the seeds inherit the bootstrap's own guards, including the non-finite check that
    exists because a seed is the geometry no parse boundary ever sees.
    """
    path = _seeded_config(tmp_path)
    first = agent_tools.get_structure(str(path))
    assert first["step"] is None  # not a step: this is what step 1 will be handed
    assert first["structure_id"] == "0"
    assert first["text"].splitlines()[0] == "3"  # the water, first in natural sort

    second = agent_tools.get_structure(str(path), structure_id="1")
    assert second["text"].splitlines()[0] == "2"  # ids number on across files

    with pytest.raises(ConfigError, match="no seed structure '9'"):
        agent_tools.get_structure(str(path), structure_id="9")


def test_a_seed_directory_that_yields_no_frames(tmp_path: Path):
    """A file is present, so bootstrap does not refuse — but it holds no structures.

    ``_seed_from_directory`` refuses an *empty* directory; a directory holding a frameless
    ``.xyz`` passes that check and seeds nothing, which is a different sentence to say.
    """
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "empty.xyz").write_text("", encoding="utf-8")
    path = tmp_path / "input.yaml"
    path.write_text(
        yaml.safe_dump({"input": "seeds", "steps": [{"step": 1, "engine": "orca"}]}), "utf-8"
    )
    with pytest.raises(ConfigError, match="holds no structures"):
        agent_tools.get_structure(str(path))


def test_a_seed_has_no_mode_to_animate(tmp_path: Path):
    """Modes belong to a computed structure; asking for one here is a refusal, not zeros."""
    with pytest.raises(ConfigError, match="no normal modes"):
        agent_tools.get_structure(str(_seeded_config(tmp_path)), mode_index=0)


def test_a_single_xyz_input_seeds_too(tmp_path: Path):
    """``input:`` may be one file rather than a directory, and every frame is a seed."""
    path = tmp_path / "input.yaml"
    (tmp_path / "two.xyz").write_text("1\nfirst\nH 0 0 0\n1\nsecond\nO 0 0 0\n", encoding="utf-8")
    path.write_text(
        yaml.safe_dump({"input": "two.xyz", "steps": [{"step": 1, "engine": "orca"}]}), "utf-8"
    )
    assert agent_tools.get_structure(str(path), structure_id="1")["text"].splitlines()[2][0] == "O"


def test_smiles_seeds_are_refused_rather_than_embedded_behind_a_read(tmp_path: Path):
    """Seeding from SMILES embeds molecules and writes them; a read must not do that.

    ``_seed_from_smiles_csv`` writes under ``output_dir/_seed`` and needs RDKit. Serving
    it here would make a GET that creates files and imports a heavy dependency, so it is
    refused with the way round it — and the refusal is checked to leave nothing behind.
    """
    (tmp_path / "seeds.csv").write_text("smiles\nCCO\n", encoding="utf-8")
    path = tmp_path / "input.yaml"
    path.write_text(
        yaml.safe_dump({"input": "seeds.csv", "steps": [{"step": 1, "engine": "orca"}]}), "utf-8"
    )
    with pytest.raises(ConfigError, match="seeds from SMILES"):
        agent_tools.get_structure(str(path))
    assert not (tmp_path / "outputs" / "_seed").exists()


def test_a_directory_is_a_directory_whatever_it_is_called(tmp_path: Path):
    """The suffix decides nothing until ``is_dir()`` has said no, as in ``bootstrap``.

    :func:`chemrefine.pipeline.bootstrap` takes a directory as a directory first and only
    then looks at the suffix, so a folder named ``batch.csv`` seeds from the ``.xyz`` files
    inside it. Testing the suffix first here refused that same folder as SMILES — the
    viewer disagreeing with the run about what the config means.
    """
    seeds = tmp_path / "batch.csv"
    seeds.mkdir()
    (seeds / "one.xyz").write_text("1\nfirst\nN 0 0 0\n", encoding="utf-8")
    path = tmp_path / "input.yaml"
    path.write_text(
        yaml.safe_dump({"input": "batch.csv", "steps": [{"step": 1, "engine": "orca"}]}), "utf-8"
    )
    assert agent_tools.get_structure(str(path))["text"].splitlines()[2][0] == "N"


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


def test_analyze_mode_gives_a_real_mode_its_frequency(tmp_path: Path):
    """Asked about a mode that is not imaginary, it used to hand back the index it was given.

    ``frequency_cm1`` was read from ``imaginary_freqs``, the only table anyone kept, so
    every real mode answered ``null`` — on the tool whose job is to say what a mode *does*,
    for a caller deciding whether 1465 cm⁻¹ is the coordinate they meant.
    """
    tree = str(_freq_tree(tmp_path))
    real = agent_tools.analyze_mode(tree, 1, "0", mode_index=7, top_atoms=1)
    assert real["frequency_cm1"] == pytest.approx(1464.97)
    assert real["is_imaginary"] is False
    # The whole table comes with it, so a caller can pick a mode without guessing an index.
    assert real["frequencies"] == {
        "6": pytest.approx(-820.38),
        "7": pytest.approx(1464.97),
        "8": pytest.approx(1465.36),
        "9": pytest.approx(3548.96),
        "10": pytest.approx(3765.14),
        "11": pytest.approx(3765.42),
    }
    # And the imaginary mode still reads as it did — the subset did not move.
    imaginary = agent_tools.analyze_mode(tree, 1, "0", mode_index=6, top_atoms=1)
    assert (imaginary["frequency_cm1"], imaginary["is_imaginary"]) == (
        pytest.approx(-820.38),
        True,
    )


def test_get_frequencies_serves_the_whole_table_from_the_cache(tmp_path: Path):
    """Persisted, so a laptop reading a finished tree has it without the ``.out`` files.

    ``analyze_mode`` re-parses and so always had access to the table; ``get_frequencies``
    reads only the cache, which is what a copied tree still has. A tree cached before the
    table was persisted answers ``null`` rather than ``{}`` — nothing known, not "no modes"
    — and ``rebuild-cache`` fills it in.
    """
    parsed = orca_coordinator.parse_text(_FREQ_OUT.read_text(encoding="utf-8"), "freq")[0]
    path = _cache_step(
        tmp_path,
        (
            Structure(
                id="0",
                atoms=Atoms(symbols=list(parsed.symbols), positions=parsed.positions),
                energy_hartree=parsed.energy_hartree,
                imaginary_freqs=parsed.imaginary_freqs,
                frequencies=parsed.frequencies,
            ),
        ),
    )
    served = agent_tools.get_frequencies(str(path), 1)["structures"][0]
    assert served["frequencies"] == {
        "6": pytest.approx(-820.38),
        "7": pytest.approx(1464.97),
        "8": pytest.approx(1465.36),
        "9": pytest.approx(3548.96),
        "10": pytest.approx(3765.14),
        "11": pytest.approx(3765.42),
    }
    assert served["imaginary_freqs"] == {"6": pytest.approx(-820.38)}
    assert served["imaginary_count"] == 1
    # And a structure cached before any of this existed says "unknown", not "no modes".
    older = tmp_path / "older"
    older.mkdir()
    before = _cache_step(older, (Structure(id="0", atoms=Atoms("H")),))
    assert agent_tools.get_frequencies(str(before), 1)["structures"][0]["frequencies"] is None


def test_list_structures_offers_the_ids_and_each_ones_modes(tmp_path: Path):
    """The enumeration ``get_structure`` lacks: what is there, rather than a guess refused.

    Both are read from the step cache, so this answers on a tree copied off a cluster with
    no output files and no ORCA — which is exactly where naming a structure by guesswork
    became a request that could only fail.
    """
    parsed = orca_coordinator.parse_text(_FREQ_OUT.read_text(encoding="utf-8"), "freq")[0]
    path = _cache_step(
        tmp_path,
        (
            Structure(
                id="0",
                atoms=Atoms(symbols=list(parsed.symbols), positions=parsed.positions),
                energy_hartree=parsed.energy_hartree,
                imaginary_freqs=parsed.imaginary_freqs,
                frequencies=parsed.frequencies,
            ),
            Structure(id="7", atoms=Atoms("H"), energy_hartree=-0.5),
        ),
    )
    listed = agent_tools.list_structures(str(path), 1)
    assert [row["id"] for row in listed["structures"]] == ["0", "7"]
    assert listed["structures"][0]["imaginary"] == [6]
    assert listed["structures"][0]["modes"]["7"] == pytest.approx(1464.97)
    # A structure with no frequency table says so, rather than offering an empty mode list
    # that reads as "this one has no modes" — unknown is not the same answer.
    assert listed["structures"][1]["modes"] is None
    assert listed["structures"][1]["imaginary"] == []


def test_list_structures_enumerates_the_seeds_and_refuses_to_embed_them(tmp_path: Path):
    """Seeds have ids and no modes; a SMILES ``input:`` is refused here as everywhere.

    Enumerating the seeds must go through the same guard the single-seed read does, or
    listing would embed the molecules that reading them refuses to — the classic way a
    second entry point loses a refusal.
    """
    (tmp_path / "two.xyz").write_text("1\na\nN 0 0 0\n1\nb\nO 0 0 0\n", encoding="utf-8")
    path = tmp_path / "input.yaml"
    path.write_text(
        yaml.safe_dump({"input": "two.xyz", "steps": [{"step": 1, "engine": "orca"}]}), "utf-8"
    )
    listed = agent_tools.list_structures(str(path))
    assert listed["step"] is None
    assert [row["id"] for row in listed["structures"]] == ["0", "1"]
    assert all(row["modes"] == {} for row in listed["structures"])  # seeds have no modes

    (tmp_path / "seeds.csv").write_text("smiles\nCCO\n", encoding="utf-8")
    smiles = tmp_path / "smiles.yaml"
    smiles.write_text(
        yaml.safe_dump({"input": "seeds.csv", "steps": [{"step": 1, "engine": "orca"}]}), "utf-8"
    )
    with pytest.raises(ConfigError, match="seeds from SMILES"):
        agent_tools.list_structures(str(smiles))
    assert not (tmp_path / "outputs" / "_seed").exists()


def test_list_structures_says_run_it_first_rather_than_offering_nothing(tmp_path: Path):
    """An empty list and "nothing has run" are different answers to "which structures?"."""
    with pytest.raises(ConfigError, match="run it"):
        agent_tools.list_structures(str(_write_config(tmp_path, {"step": 1, "engine": "orca"})), 1)


def test_a_structure_file_is_read_on_its_own_periodic_cell_and_all(tmp_path: Path):
    """No config, no step, no cache — the question is only "what is in this file".

    ASE reads about ninety formats and most are periodic, so this is the first path in
    ChemRefine that produces a structure with a cell: the pipeline accepts only ``.xyz``, a
    directory or a SMILES ``.csv``, no engine parser captures a cell, and the cache stores
    symbols and positions. Viewing a periodic file is not computing on one.
    """
    from ase.build import bulk

    bulk("Si", "diamond", a=5.43).write(tmp_path / "POSCAR", format="vasp")
    bulk("NaCl", "rocksalt", a=5.64).write(tmp_path / "salt.cif")
    (tmp_path / "water.xyz").write_text("1\nseed\nO 0 0 0\n", encoding="utf-8")

    for name, formula, periodic in [
        ("POSCAR", "Si2", True),
        ("salt.cif", "ClNa", True),
        ("water.xyz", "O", False),
    ]:
        served = agent_tools.read_structure_file(str(tmp_path / name))
        assert (served["formula"], served["periodic"]) == (formula, periodic)
        assert served["format"] == "extxyz"
        # The cell reaches the viewer as Lattice="…", which is what makes it draw a box.
        assert ('Lattice="' in served["text"]) is periodic


def test_a_dropped_file_is_read_from_its_contents_and_its_name(tmp_path: Path):
    """A file dropped on the page is on the *browser's* machine, not the server's.

    Over a forwarded port those are different machines, so there is no path to read — only
    contents and a basename. The name still decides the parser: ``POSCAR`` and ``.cif`` are
    not guessable from a first line.
    """
    from ase.build import bulk

    bulk("Si", "diamond", a=5.43).write(tmp_path / "POSCAR", format="vasp")
    text = (tmp_path / "POSCAR").read_text(encoding="utf-8")
    served = agent_tools.read_structure_text("POSCAR", text)
    assert served["formula"] == "Si2"
    assert served["periodic"] is True
    assert served["path"] == "POSCAR"  # what the user called it, not where it was staged

    # The same bytes under a name ASE cannot place are refused, rather than guessed at.
    with pytest.raises(ConfigError, match="cannot read a structure"):
        agent_tools.read_structure_text("notes.txt", text)


def test_a_dropped_name_never_becomes_part_of_a_path(tmp_path: Path):
    """That string came from a browser, and a browser is not a trusted source of paths.

    ``../../../etc/POSCAR`` must name the staged file and nothing else. It is the basename
    or nothing, and the staging directory is one this process made and then removes.
    """
    from ase.build import bulk

    bulk("Si", "diamond", a=5.43).write(tmp_path / "POSCAR", format="vasp")
    text = (tmp_path / "POSCAR").read_text(encoding="utf-8")
    # Directories stripped, the filename kept: still a POSCAR, read as one.
    for hostile in ("../../../etc/POSCAR", "/etc/POSCAR", "etc/../POSCAR"):
        assert agent_tools.read_structure_text(hostile, text)["path"] == "POSCAR"

    # And the names that are not filenames at all. `Path("..").name` is `".."` — pathlib
    # never claimed to sanitize — which resolved to the *parent* of the staging directory
    # and crashed on writing to a directory. They fall back to a name with no format to
    # guess, so the refusal is about the format and nothing has been written anywhere.
    for degenerate in ("..", ".", "", "a/.."):
        with pytest.raises(ConfigError, match="cannot read a structure from structure"):
            agent_tools.read_structure_text(degenerate, text)


def test_a_structure_file_read_refuses_what_it_cannot_show(tmp_path: Path):
    """Each refusal names the file and what was wrong, never a traceback.

    The size ceiling is the one worth stating: ASE will read a multi-gigabyte trajectory
    into memory, synchronously, inside a request.
    """
    with pytest.raises(ConfigError, match="not a file"):
        agent_tools.read_structure_file(str(tmp_path / "nothing.xyz"))
    with pytest.raises(ConfigError, match="not a file"):
        agent_tools.read_structure_file(str(tmp_path))  # a directory is not a structure

    (tmp_path / "junk.xyz").write_text("this is not a structure\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot read a structure"):
        agent_tools.read_structure_file(str(tmp_path / "junk.xyz"))

    (tmp_path / "empty.xyz").write_text("0\nnothing\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="holds no atoms"):
        agent_tools.read_structure_file(str(tmp_path / "empty.xyz"))

    huge = tmp_path / "huge.xyz"
    huge.write_bytes(b"x" * (agent_tools._MAX_STRUCTURE_BYTES + 1))
    with pytest.raises(ConfigError, match="not a trajectory"):
        agent_tools.read_structure_file(str(huge))
    with pytest.raises(ConfigError, match="not a trajectory"):
        agent_tools.read_structure_text("huge.xyz", "x" * (agent_tools._MAX_STRUCTURE_BYTES + 1))


def test_a_multi_frame_file_shows_its_last_frame(tmp_path: Path):
    """A trajectory's last frame is its result — as every other reader here takes the last."""
    (tmp_path / "two.xyz").write_text("1\nfirst\nN 0 0 0\n1\nsecond\nO 0 0 0\n", encoding="utf-8")
    assert agent_tools.read_structure_file(str(tmp_path / "two.xyz"))["formula"] == "O"


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
    with pytest.raises(ConfigError, match=r"cannot find step1_0\.out"):
        agent_tools.analyze_mode(str(tree), 1, "0", mode_index=0)


def test_a_relocated_tree_still_finds_its_outputs(tmp_path: Path):
    """A tree copied off the cluster keeps working, and this is the only place it did not.

    Everything about a tree is addressed relatively — ``step_dir`` derives from
    ``output_dir``, which resolves against the config file's own directory — except the
    manifest, which records the absolute path each output was *written* to. So a copied
    tree read its cache from the new root and looked for its outputs on the machine that
    is no longer there, and said "rerun the step to regenerate it" about a file sitting in
    the copy.
    """
    cluster = tmp_path / "cluster"
    cluster.mkdir()
    original = _freq_tree(cluster)
    moved = tmp_path / "laptop"
    shutil.copytree(cluster, moved)
    shutil.rmtree(cluster)  # the machine it ran on is gone
    relocated = moved / original.name

    stale = cache.load_manifest(load_config(relocated).step_dir(load_config(relocated).steps[0]))
    assert not stale.files[0][1].is_file()  # the manifest names the old machine

    assert agent_tools.analyze_mode(str(relocated), 1, "0", mode_index=6)["frequency_cm1"] == (
        pytest.approx(-820.38)
    )
    drawn = agent_tools.get_structure(str(relocated), 1, structure_id="0", mode_index=6)
    assert len(drawn["text"].splitlines()[2].split()) == 7  # displacement columns and all


def test_an_output_a_retry_moved_into_an_attempt_dir_is_found(tmp_path: Path):
    """Retries and NMS write into ``attemptN/``, so the search has to go below the id dir.

    The newest wins: a retried structure has the same basename in several attempt
    directories, and the latest is the one the manifest would have been rewritten to name.
    """
    tree = _freq_tree(tmp_path)
    config = load_config(tree)
    home = config.step_dir(config.steps[0]) / "0"
    recorded = home / "step1_0.out"
    for attempt in ("attempt1", "attempt2"):
        (home / attempt).mkdir()
        shutil.copy(recorded, home / attempt / "step1_0.out")
        time.sleep(0.01)  # attempt2 is the newer one
    recorded.unlink()  # the retry left nothing at the recorded path

    found = agent_tools._locate_output(config.step_dir(config.steps[0]), "0", recorded)
    assert found is not None
    assert found.parent.name == "attempt2"
    assert agent_tools.analyze_mode(str(tree), 1, "0", mode_index=6)["frequency_cm1"] == (
        pytest.approx(-820.38)
    )


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


def test_analyze_mode_names_the_registry_for_an_unregistered_engine(tmp_path: Path):
    """An engine name the registry has never heard of gets the registry's own error.

    `load_config` keeps `engine:` a free string, so a tree written elsewhere can name an
    engine this install lacks (a plugin not installed here). That is not "outputs carry
    no modes" — it is "no such engine", and `EngineNotFoundError` (exit 3) says so and
    lists what is registered. Pinned deliberately: capability dispatch resolves the
    engine first, and the truer error must not regress to the blanket refusal.
    """
    path = _write_config(tmp_path, {"step": 1, "engine": "not-installed-here"})
    config = load_config(path)
    step_dir = config.step_dir(config.steps[0])
    out = step_dir / "0" / "step1_0.out"
    out.parent.mkdir(parents=True)
    out.write_text("whatever", encoding="utf-8")
    cache.save_manifest(
        StepInputs(files=((out.with_suffix(".inp"), out, "0"),)),
        step_dir,
        operation=None,
        engine="not-installed-here",
    )
    with pytest.raises(EngineNotFoundError, match="not-installed-here"):
        agent_tools.analyze_mode(str(path), 1, "0", mode_index=0)


def test_step_lookup_failure_is_shared_by_both_tools(tmp_path: Path):
    path = _write_config(tmp_path)
    with pytest.raises(ConfigError, match="no step matches"):
        agent_tools.get_frequencies(str(path), 9)
