"""Tests for the per-step JSON cache, fingerprint, and manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine import cache
from chemrefine.cache import (
    CACHE_FORMAT_VERSION,
    RESULT_FORMAT_VERSION,
    ResolutionSpec,
    StepKey,
    invalidate,
    load,
    load_if_valid,
    load_manifest,
    manifest_path,
    option_file_digests,
    save,
    save_manifest,
    save_result_records,
    structure_from_record,
    structure_record,
)
from chemrefine.config import StepConfig
from chemrefine.errors import CacheError
from chemrefine.state import StepInputs, StepResults, Structure


def _results() -> StepResults:
    zero_force = np.zeros((1, 3))
    return StepResults(
        structures=(
            Structure(id="0", atoms=Atoms("H"), energy_hartree=-1.0, forces_ev_per_a=zero_force),
            Structure(id="1", atoms=Atoms("H"), energy_hartree=-1.5, forces_ev_per_a=zero_force),
        )
    )


def _cfg(**overrides) -> StepConfig:
    base = {"step": 1, "engine": "fake", "operation": "opt_sp"}
    base.update(overrides)
    return StepConfig(**base)


def _parents(*ids: str) -> tuple[Structure, ...]:
    """Seed structures with the given ids — what a step's key is derived over."""
    return tuple(Structure(id=i, atoms=Atoms("H")) for i in ids)


def _key(
    *ids: str,
    step_cfg: StepConfig | None = None,
    template: Path | None = None,
    charge: int = 0,
    multiplicity: int = 1,
    engine_options: dict | None = None,
    resolution: ResolutionSpec | None = None,
    aux_files: dict[str, Path] | None = None,
) -> StepKey:
    """The key a step over these parents would be written under."""
    return StepKey.of(
        step_cfg or _cfg(),
        _parents(*ids),
        template,
        charge=charge,
        multiplicity=multiplicity,
        engine_options=engine_options,
        resolution=resolution,
        aux_files=aux_files,
    )


def _resolution(
    target: str = "minimum", displacement: float = 1.0, seed: int = 42
) -> ResolutionSpec:
    """A resolution spec the way ``derive_step_key`` splits the NMS reading."""
    return ResolutionSpec(
        criterion={"target": target, "ts_mode_index": None},
        search={"displacement_value": displacement, "num_random_displacements": 1, "seed": seed},
    )


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_the_key_is_stable_for_identical_inputs():
    assert _key("0", "1").fingerprint == _key("0", "1").fingerprint
    assert len(_key("0").fingerprint) == 16


def test_the_key_moves_when_parents_change():
    assert _key("0").fingerprint != _key("0", "1").fingerprint


def test_the_effective_charge_is_the_identity_not_the_override():
    """Workflow ``charge: 0`` vs ``charge: 2`` must be two different steps.

    The old payload hashed the per-step *override* — ``None`` when inherited — so a
    workflow-level charge edit changed every job's physics while every fingerprint stood
    still (reproduced on the shipped payload before this model replaced it). The key now
    takes the effective value the jobs render.
    """
    assert _key("0", charge=0).fingerprint != _key("0", charge=2).fingerprint
    assert _key("0", multiplicity=1).fingerprint != _key("0", multiplicity=3).fingerprint


def test_an_undeclared_option_moves_no_key():
    """A key nothing reads cannot change a job — validate warns about it; the key ignores it.

    Engine options enter the key only *as the engine's declared model reads them*; a junk
    or typo'd key reaches no engine (script engines substitute declared fields only) and
    therefore no result.
    """
    plain = _key("0", step_cfg=_cfg(options={}))
    junk = _key("0", step_cfg=_cfg(options={"typo_knob": 7}))
    assert plain.fingerprint == junk.fingerprint
    assert plain.row_keys == junk.row_keys


def test_a_declared_option_moves_the_row_key():
    a = _key("0", engine_options={"cores": 4})
    b = _key("0", engine_options={"cores": 8})
    assert a.row_keys != b.row_keys
    assert a.fingerprint != b.fingerprint


def test_a_template_referenced_aux_file_is_part_of_the_row_key(tmp_path: Path):
    """Editing a file the template references must move the key — that is the re-run.

    The template digest covers only the path *string*: a ``%DOCKER GUEST`` edit to the
    guest geometry changed every docking result while every fingerprint stood still,
    and ``resume`` served the stale answers. The same rationale that pinned
    ``model_path`` bytes via ``option_file_digests``, applied to the enumeration the
    engine hands over. And the change is scoped: a step whose template names no aux
    files keys exactly as it did before the parameter existed.
    """
    guest = tmp_path / "cl.xyz"
    guest.write_text("1\nchloride\nCl 0.0 0.0 0.0\n", encoding="utf-8")

    before = _key("0", aux_files={"cl.xyz": guest})
    assert before == _key("0", aux_files={"cl.xyz": guest})  # stable while the file stands

    guest.write_text("1\nchloride moved\nCl 0.5 0.0 0.0\n", encoding="utf-8")
    after = _key("0", aux_files={"cl.xyz": guest})
    assert before.row_keys != after.row_keys
    assert before.fingerprint != after.fingerprint

    # The digest is keyed by the *written* reference, never the resolved path: a
    # relocated tree re-derives the same key from the same bytes, which is the
    # guarantee `rebuild-cache` on a moved project stands on.
    elsewhere = tmp_path / "moved" / "cl.xyz"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(guest.read_bytes())
    assert _key("0", aux_files={"cl.xyz": elsewhere}) == after

    # Scoped-change regression: no aux files means the pre-parameter key, bit for bit —
    # existing trees must not re-run over this feature's arrival.
    assert _key("0", aux_files={}) == _key("0")
    # A file that cannot be read contributes no entry (option_file_digests' escape):
    # the key moves only when the file appears.
    assert _key("0", aux_files={"missing.pc": tmp_path / "missing.pc"}) == _key("0")


def test_the_nms_family_moves_only_the_resolution_key():
    """The flag and its options steer post-round-1 resolution — never a job.

    Same rows either way is what makes flipping ``nms: true`` a served edit: every
    round-1 output on disk remains provably this configuration's.
    """
    plain = _key("0")
    resolving = _key("0", resolution=_resolution())
    assert plain.row_keys == resolving.row_keys
    assert plain.fingerprint != resolving.fingerprint
    assert plain.criterion_key == plain.search_key == ""
    assert resolving.criterion_key != "" and resolving.search_key != ""


def test_search_retunes_keep_the_criterion_key_criterion_changes_move_it():
    """The search/criterion split, structurally: search tunes the hunt, criterion the answer.

    The criterion key is what decides whether an ``attemptK/`` resolution on disk may be
    trusted; the rows are untouched by both, which is why neither edit ever re-runs
    round 1.
    """
    base = _key("0", resolution=_resolution())
    retuned = _key("0", resolution=_resolution(displacement=2.0, seed=7))
    recriterioned = _key("0", resolution=_resolution(target="ts"))
    assert base.row_keys == retuned.row_keys == recriterioned.row_keys
    assert base.fingerprint != retuned.fingerprint  # a different step...
    assert base.criterion_key == retuned.criterion_key  # ...same resolution verdict
    assert base.criterion_key != recriterioned.criterion_key


def test_manifest_provenance_round_trips(tmp_path: Path):
    """What save_manifest writes per row, load_manifest_provenance reads back — exactly."""
    from chemrefine.cache import load_manifest_provenance

    key = _key("0", "1", resolution=_resolution())
    inputs = StepInputs(
        files=(
            (tmp_path / "0.inp", tmp_path / "0.out", "0"),
            (tmp_path / "1.inp", tmp_path / "1.out", "1"),
        )
    )
    save_manifest(inputs, tmp_path, operation="opt_sp", engine="fake", **key.manifest_stamp())
    provenance = load_manifest_provenance(tmp_path)
    assert provenance.fingerprint == key.fingerprint
    assert provenance.criterion_key == key.criterion_key
    assert provenance.search_key == key.search_key
    assert provenance.rows == key.manifest_rows()
    # And the file layout is untouched by the extra keys.
    assert load_manifest(tmp_path) == inputs


def test_the_stamp_fills_every_provenance_slot_save_manifest_has():
    """The drift class, pinned in the direction it fired.

    ``save_manifest`` defaults every stamp field to ``""`` — unprovable, adoptable — so a
    slot added to it and not to :meth:`StepKey.manifest_stamp` (``search_key`` was, once,
    at every call site at once) would silently disarm the refusals built on it. The
    projection and the signature must name the same slots, and the projection must
    fill them from the key rather than with the default.
    """
    import inspect

    key = _key("0", "1", resolution=_resolution())
    stamp = key.manifest_stamp()
    slots = set(inspect.signature(save_manifest).parameters) - {
        "inputs",
        "step_dir",
        "operation",
        "engine",
    }
    assert set(stamp) == slots
    assert all(stamp.values()), "a stamped slot must never carry the default"


def test_the_manifest_spells_its_files_relative_to_the_step_directory(tmp_path: Path):
    """Everything about a tree is addressed relatively; the manifest was the one exception.

    Spelled absolute it named the machine the step ran on, and a copied tree's
    `rebuild-cache` / ledgered `rerun` read paths that were not there. A file outside the
    step directory has no relative spelling and stays absolute — the writer is total.
    """
    step_dir = tmp_path / "outputs" / "step1"
    elsewhere = tmp_path / "shared" / "seed.xyz"
    inputs = StepInputs(
        files=(
            (step_dir / "0" / "step1_0.inp", step_dir / "0" / "step1_0.out", "0"),
            (elsewhere, step_dir / "1" / "step1_1.out", "1"),
        )
    )
    save_manifest(inputs, step_dir, operation="opt_sp", engine="fake")
    records = json.loads(cache.manifest_path(step_dir).read_text(encoding="utf-8"))["files"]
    assert [(r["input"], r["output"]) for r in records] == [
        ("0/step1_0.inp", "0/step1_0.out"),
        (str(elsewhere), "1/step1_1.out"),
    ]
    # Read back, every path is the one the caller wrote.
    assert load_manifest(step_dir) == inputs


def test_a_moved_tree_reads_its_own_manifest(tmp_path: Path):
    """The point of the relative spelling: the manifest follows the tree it describes."""
    import shutil

    before = tmp_path / "here" / "step1"
    inputs = StepInputs(files=((before / "0" / "a.inp", before / "0" / "a.out", "0"),))
    save_manifest(inputs, before, operation="opt_sp", engine="fake")

    after = tmp_path / "there" / "step1"
    shutil.move(str(before.parent), str(after.parent))

    assert load_manifest(after) == StepInputs(
        files=((after / "0" / "a.inp", after / "0" / "a.out", "0"),)
    )


def test_a_manifest_with_absolute_paths_reads_them_verbatim(tmp_path: Path):
    """Every manifest written before the relative spelling, and every hand-written v1
    adoption manifest, names absolute paths — and reads exactly as it always did."""
    from chemrefine.cache import manifest_path

    step_dir = tmp_path / "step1"
    manifest_path(step_dir).parent.mkdir(parents=True)
    manifest_path(step_dir).write_text(
        json.dumps(
            {
                "operation": "goat",
                "engine": "orca",
                "fingerprint": "",
                "files": [
                    {"input": "/abs/run/step1/0/s.inp", "output": "/abs/run/s.out", "id": "0"}
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_manifest(step_dir) == StepInputs(
        files=((Path("/abs/run/step1/0/s.inp"), Path("/abs/run/s.out"), "0"),)
    )


def test_a_bare_manifest_reads_as_unprovenanced(tmp_path: Path):
    """A manifest without row keys — every pre-provenance tree — is empty provenance."""
    from chemrefine.cache import load_manifest_provenance

    save_manifest(
        StepInputs(files=((tmp_path / "0.inp", tmp_path / "0.out", "0"),)),
        tmp_path,
        operation="opt_sp",
        engine="fake",
        fingerprint="feedfacefeedface",
    )
    provenance = load_manifest_provenance(tmp_path)
    assert provenance.rows == {}
    assert provenance.fingerprint == "feedfacefeedface"


def test_a_manifest_that_is_not_a_mapping_reads_as_unprovenanced(tmp_path: Path):
    """Valid JSON of the wrong shape — a bare list — is unprovable, not a crash.

    ``read_json`` guarantees only that the file parsed; a hand-edited or foreign manifest
    can hold any JSON value, and provenance built on ``.get`` calls against a list would be
    a ``AttributeError`` three frames from the file that caused it. All-empty provenance
    routes the caller to the explicit ``rebuild-cache``, same as a pre-provenance tree.
    """
    from chemrefine.cache import load_manifest_provenance, manifest_path

    manifest_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    manifest_path(tmp_path).write_text('["not", "a", "mapping"]', encoding="utf-8")
    provenance = load_manifest_provenance(tmp_path)
    assert provenance.fingerprint == ""
    assert provenance.criterion_key == ""
    assert provenance.search_key == ""
    assert provenance.rows == {}


def test_manifest_rows_align_ids_keys_and_digests():
    key = _key("0", "1")
    rows = key.manifest_rows()
    assert set(rows) == {"0", "1"}
    assert rows["0"] == (key.row_keys[0], key.parent_digests[0])
    assert rows["1"] == (key.row_keys[1], key.parent_digests[1])


# ---------------------------------------------------------------------------
# Files an option names (option_file_digests)
# ---------------------------------------------------------------------------


def test_options_that_name_no_file_digest_to_nothing():
    """Only real files are read, so an ordinary step pays nothing for this."""
    assert option_file_digests(None) == {}
    assert option_file_digests({"task_name": "mace_off", "cores": 8, "device": "cuda"}) == {}


def test_a_path_that_does_not_exist_contributes_no_entry(tmp_path: Path):
    """ "Not a path" and "a path that is missing" are different claims.

    A missing file must contribute nothing rather than an empty digest, so that the key
    *changes* when the file later appears — which is the moment the step's inputs really did.
    """
    missing = tmp_path / "model.pt"
    assert option_file_digests({"model_path": str(missing)}) == {}
    missing.write_bytes(b"weights")
    assert set(option_file_digests({"model_path": str(missing)})) == {"model_path"}


def test_a_value_that_only_looks_like_a_path_is_not_a_reason_to_fail(tmp_path: Path):
    """Options are free-form text; most values are not paths and some are hostile to being asked.

    A name past the filesystem's length limit raises from `is_file()` rather than returning
    False. A step that never asked to be pinned to a file must not fail its whole run over
    one — the value simply contributes no digest.
    """
    assert option_file_digests({"note": "x" * 5000}) == {}


def test_a_named_file_is_digested_by_its_contents(tmp_path: Path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"first")
    before = option_file_digests({"model_path": str(model)})
    model.write_bytes(b"second")
    after = option_file_digests({"model_path": str(model)})

    assert before["model_path"] != after["model_path"]
    assert len(after["model_path"]) == 16


def test_the_option_digest_is_computable_where_sha1_is_policy_restricted(
    tmp_path: Path, monkeypatch
):
    """The digest must not need the code path a crypto policy can switch off.

    `file_digest` handed the *name* `"sha1"` builds its hash through `hashlib.new(...)` with
    `usedforsecurity` left at the default, which a host configured to allow SHA-1 only as a
    fingerprint refuses outright. This runs on every `StepKey.of` — every step, every run —
    and the `except OSError` beside it would not catch a `ValueError`, so the run would end
    in a traceback outside the exit-code contract. Passing the constructor keeps `new` out of
    it entirely, which is what this pins: `new` raising must not matter.
    """
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")

    def refuse(name: str, *args: object, **kwargs: object):
        raise AssertionError(f"hashlib.new({name!r}) is the path a crypto policy can refuse")

    monkeypatch.setattr(cache.hashlib, "new", refuse)
    assert len(option_file_digests({"model_path": str(model)})["model_path"]) == 16


def test_the_option_digest_value_is_the_plain_sha1_of_the_file(tmp_path: Path):
    """Pin the value, because it is a cache key.

    `usedforsecurity` is a policy hint to the backend, not an input to the hash — so marking
    the fingerprint must leave every stored key exactly where it was. Were that ever untrue,
    the symptom would be silent: every cached step in every existing output tree invalidated
    at once, which reads as a bug rather than as the one-line change that caused it.
    """
    model = tmp_path / "model.pt"
    model.write_bytes(b"chemrefine-fixed-test-vector")
    expected = hashlib.sha1(b"chemrefine-fixed-test-vector").hexdigest()[:16]
    assert option_file_digests({"model_path": str(model)}) == {"model_path": expected}


def test_retraining_a_model_re_runs_the_step_that_consumes_it(tmp_path: Path):
    """The point of the whole mechanism, at the level the pipeline actually uses.

    A training step passes its structures through unchanged, so retraining moves neither the
    consumer's parent ids nor their geometries — and the consumer names the model by a path
    string that did not change either. Without the content digest its fingerprint is identical
    and `resume` serves a cache computed with the *previous* weights.
    """
    model = tmp_path / "model.pt"
    model.write_bytes(b"first weights")
    cfg = _cfg(options={"model_path": str(model), "task_name": "mace_off"})

    before = StepKey.of(cfg, _parents("0"), None)
    model.write_bytes(b"retrained weights")
    after = StepKey.of(cfg, _parents("0"), None)

    assert before.fingerprint != after.fingerprint


def _h2(spacing: float = 0.74, energy: float | None = None) -> Structure:
    atoms = Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [spacing, 0, 0]])
    return Structure(id="0", atoms=atoms, energy_hartree=energy)


def test_structure_digest_stable_for_identical_content():
    from chemrefine.cache import structure_digest

    assert structure_digest(_h2()) == structure_digest(_h2())


def test_structure_digest_changes_when_geometry_changes():
    """Same ID, different coordinates — the digest is what catches an edited seed file."""
    from chemrefine.cache import structure_digest

    assert structure_digest(_h2(spacing=0.74)) != structure_digest(_h2(spacing=0.75))


def test_structure_digest_changes_when_energy_changes():
    from chemrefine.cache import structure_digest

    assert structure_digest(_h2(energy=-1.0)) != structure_digest(_h2(energy=-1.1))


def test_structure_digest_changes_when_id_changes():
    from chemrefine.cache import structure_digest

    a = _h2()
    b = Structure(id="7", atoms=a.atoms, energy_hartree=a.energy_hartree)
    assert structure_digest(a) != structure_digest(b)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_creates_single_json_document(tmp_path: Path):
    """The cache is one JSON document — no pickle is ever written."""
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        key=_key("0", "1", step_cfg=_cfg()),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    assert (step_dir / "_cache" / "step.json").is_file()
    assert not (step_dir / "_cache" / "step.pkl").exists()


def test_the_step_document_is_compact_but_the_records_beside_it_are_not(tmp_path: Path):
    """A deliberate asymmetry: the bulk document is machine-read, the small ones are not.

    Indentation costs 47% of ``step.json`` at 10⁴ structures — it is thousands of short
    numeric values each carrying a newline and a run of spaces — and no consumer benefits,
    because the loader parses the whole document rather than reading it by line or by eye.
    The per-structure ``.result.json`` is the opposite case: a few KB that a person opens.
    """
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        key=_key("0", "1", step_cfg=_cfg()),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )

    document = (step_dir / "_cache" / "step.json").read_text()
    assert "\n" not in document, "the step document carries no layout"
    assert '"step":' in document and '"step": ' not in document

    save_result_records(_results().structures, step_dir, step=1)
    records = list(step_dir.glob("*.result.json"))
    assert records and all("\n" in p.read_text() for p in records), "records stay readable"


def test_save_and_load_round_trip(tmp_path: Path):
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        key=_key("0", "1", step_cfg=_cfg()),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    loaded = load(step_dir)
    assert loaded is not None
    assert loaded.step == 1
    assert loaded.engine == "fake"
    assert [s.id for s in loaded.results.structures] == ["0", "1"]
    assert loaded.results.structures[1].energy_hartree == -1.5


def test_template_contents_round_trip_through_validity(tmp_path: Path):
    """Editing a template in place invalidates the step; the basename never changes."""
    step_dir = tmp_path / "step1"
    template = tmp_path / "step1.inp"
    template.write_text("! Opt\n", encoding="utf-8")
    save(
        step_cfg=_cfg(),
        key=_key("0", template=template),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    assert load_if_valid(key=_key("0", template=template), step_dir=step_dir)
    template.write_text("! Opt Freq\n", encoding="utf-8")
    assert not load_if_valid(key=_key("0", template=template), step_dir=step_dir)


def test_round_trip_preserves_positions_forces_and_flags(tmp_path: Path):
    """Coordinates, forces, lineage, and status flags survive the JSON store."""
    forces = np.array([[0.1234567890123456, -2.5e-7, 3.0]], dtype=np.float64)
    struct = Structure(
        id="0-1",
        atoms=Atoms("H", positions=[[0.7414213562373095, 0.0, -1.5e-9]]),
        parent_id="0",
        energy_hartree=-1.0000000000000002,
        forces_ev_per_a=forces,
        converged=True,
        terminated_normally=False,
    )
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        key=_key("0", step_cfg=_cfg()),
        results=StepResults(structures=(struct,)),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    loaded = load(step_dir).results.structures[0]
    assert loaded.parent_id == "0"
    assert loaded.energy_hartree == -1.0000000000000002
    assert loaded.converged is True and loaded.terminated_normally is False
    np.testing.assert_array_equal(loaded.atoms.get_positions(), struct.atoms.get_positions())
    np.testing.assert_array_equal(loaded.forces_ev_per_a, forces)
    assert loaded.forces_ev_per_a.dtype == np.float64


def test_round_trip_preserves_thermochemistry(tmp_path: Path):
    """Gibbs / enthalpy / electronic+ZPE survive the JSON store."""
    struct = Structure(
        id="0",
        atoms=Atoms("H"),
        energy_hartree=-76.40,
        gibbs_hartree=-76.41,
        enthalpy_hartree=-76.38,
        energy_zpe_hartree=-76.38,
    )
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        key=_key("0", step_cfg=_cfg()),
        results=StepResults(structures=(struct,)),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    loaded = load(step_dir).results.structures[0]
    assert loaded.gibbs_hartree == -76.41
    assert loaded.enthalpy_hartree == -76.38
    assert loaded.energy_zpe_hartree == -76.38


def test_load_tolerates_cache_without_thermochemistry(tmp_path: Path):
    """A cache written before thermochemistry existed loads with None fields."""
    import json

    from chemrefine.cache import _cache_path

    save(
        step_cfg=_cfg(),
        key=_key("0", step_cfg=_cfg()),
        results=_results(),
        step_dir=tmp_path / "step1",
        chemrefine_version="2.0.0",
    )
    path = _cache_path(tmp_path / "step1")
    doc = json.loads(path.read_text())
    for entry in doc["structures"]:  # simulate an older document
        entry.pop("gibbs_hartree", None)
        entry.pop("enthalpy_hartree", None)
        entry.pop("energy_zpe_hartree", None)
    path.write_text(json.dumps(doc))
    loaded = load(tmp_path / "step1").results.structures[0]
    assert loaded.gibbs_hartree is None
    assert loaded.enthalpy_hartree is None
    assert loaded.energy_zpe_hartree is None


def test_round_trip_keeps_the_structure_digest_stable(tmp_path: Path):
    """The load-bearing property: a JSON round-trip must not perturb the digest.

    Downstream steps key their rows against ``structure_digest`` of *loaded*
    structures (exact float64 bytes); any drift would invalidate every
    downstream cache on resume.
    """
    from chemrefine.cache import structure_digest

    struct = _h2(spacing=0.7414213562373095, energy=-1.1283791670955126)
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        key=_key("0", step_cfg=_cfg()),
        results=StepResults(structures=(struct,)),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    [loaded] = load(step_dir).results.structures
    assert structure_digest(loaded) == structure_digest(struct)


def test_load_missing_returns_none(tmp_path: Path):
    assert load(tmp_path / "step1") is None


def test_load_corrupt_json_raises(tmp_path: Path):
    cache_file = tmp_path / "step1" / "_cache" / "step.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"{not json")
    with pytest.raises(CacheError):
        load(tmp_path / "step1")


def test_load_rejects_old_cache_format(tmp_path: Path):
    import json

    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        key=_key("0", step_cfg=_cfg()),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    cache_file = step_dir / "_cache" / "step.json"
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    data["cache_format"] = "v9.9"
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CacheError, match=r"format v9\.9"):
        load(step_dir)


def test_load_rejects_legacy_summary_sidecar(tmp_path: Path):
    """A ``step.json`` that is a summary, carrying no ``structures``, is not a cache.

    Loading one must raise (→ ``load_if_valid`` None → clean rebuild), never
    misread the summary as a complete cache.
    """
    step_dir = tmp_path / "step1"
    cache_file = step_dir / "_cache" / "step.json"
    cache_file.parent.mkdir(parents=True)
    # Carries the *current* format, so what rejects it is the missing `structures` key
    # rather than the version check standing in front of it.
    cache_file.write_text(
        json.dumps(
            {
                "cache_format": CACHE_FORMAT_VERSION,
                "chemrefine_version": "2.0.0",
                "fingerprint": "abc",
                "step": 1,
                "name": None,
                "engine": "fake",
                "operation": "opt_sp",
                "structure_ids": ["0"],
                "parent_ids": [None],
                "energies_hartree": [-1.0],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CacheError, match="stale or corrupt"):
        load(step_dir)
    assert not load_if_valid(key=_key("0", step_cfg=_cfg()), step_dir=step_dir)


# ---------------------------------------------------------------------------
# load_if_valid
# ---------------------------------------------------------------------------


def test_load_if_valid_true_when_cache_matches(tmp_path: Path):
    step_dir = tmp_path / "step1"
    cfg = _cfg()
    save(
        step_cfg=cfg,
        key=_key("0", step_cfg=cfg),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    assert load_if_valid(key=_key("0", step_cfg=cfg), step_dir=step_dir)


def test_load_if_valid_false_when_config_changes(tmp_path: Path):
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        key=_key("0", charge=0),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    assert not load_if_valid(key=_key("0", charge=-1), step_dir=step_dir)


def test_load_if_valid_false_when_cache_is_corrupt(tmp_path: Path):
    """A corrupt pickle should make ``load_if_valid`` return None, not raise."""
    step_dir = tmp_path / "step1"
    cache_dir = step_dir / "_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "step.json").write_bytes(b"{not json")
    assert not load_if_valid(key=_key("0", step_cfg=_cfg()), step_dir=step_dir)


def _saved(tmp_path: Path) -> Path:
    """A step dir holding a freshly saved cache (document + coordinate sidecar)."""
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        key=_key("0", "1", step_cfg=_cfg()),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    return step_dir


def test_resolved_from_round_trips_and_is_optional(tmp_path: Path):
    """The NMS provenance survives save → load, and a record without it loads as ``None``.

    Additive by design: dropping the key is what a record written before the field existed
    looks like, so it must read back as "no promotion happened" rather than raising.
    """
    resolved = replace(_results().structures[0], resolved_from="0_m5_pos")
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        key=_key("0", step_cfg=_cfg()),
        results=StepResults(structures=(resolved,)),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    loaded = load(step_dir)
    assert loaded is not None
    assert loaded.results.structures[0].resolved_from == "0_m5_pos"

    record = structure_record(resolved)
    del record["resolved_from"]
    assert structure_from_record(record).resolved_from is None


def test_a_document_without_its_sidecar_is_an_error_not_a_fallback(tmp_path: Path):
    """The one direction the split format cannot enforce on its own.

    A ``step.json`` written before the arrays moved out still carries ``positions`` inline,
    and reading those would be the single way to end up with a cache half in each format —
    silently, since the records parse fine. So the reader takes arrays from the sidecar or
    refuses: no inline fallback, ever. ``load_if_valid`` turns that into a plain miss, which
    rebuilds the step.
    """
    step_dir = _saved(tmp_path)
    (step_dir / "_cache" / "arrays.npz").unlink()

    with pytest.raises(CacheError, match=r"arrays\.npz"):
        load(step_dir)
    assert not load_if_valid(key=_key("0", "1", step_cfg=_cfg()), step_dir=step_dir)


def test_the_sidecar_refuses_to_unpickle(tmp_path: Path):
    """``allow_pickle=False`` is the whole reason this is not a pickle — assert numpy enforces it.

    The cache promises that loading it can never execute code from the file. JSON gave that
    for free; ``.npy`` gives it only because the flag is spelled out, and numpy raises on an
    object array rather than running its reduce. A sidecar smuggling one in must be rejected.
    """
    step_dir = _saved(tmp_path)
    with (step_dir / "_cache" / "arrays.npz").open("wb") as fh:
        np.savez(fh, positions=np.array([{"payload": "code"}], dtype=object))

    with pytest.raises(CacheError, match="corrupt coordinate sidecar"):
        load(step_dir)


def _orphan_sidecar(step_dir: Path, *structures: Structure) -> None:
    """Write a sidecar for ``structures`` over ``step_dir``'s, leaving its document behind.

    A save killed between its two writes: the sidecar has landed, the document has not, so the
    previous one is still there. `save` writes the sidecar first, so this is that save stopped
    one statement early.
    """
    records = [cache.structure_record(s) for s in structures]
    arrays = cache._split_arrays(records)
    cache.atomic_write(cache._arrays_path(step_dir), cache._npz_bytes(arrays))


def test_a_sidecar_from_another_save_is_refused_not_read(tmp_path: Path):
    """A structure must never wear another structure's coordinates.

    The two cache files are separate atomic writes, so a save interrupted between them leaves
    a new ``arrays.npz`` beside the previous ``step.json`` — which `resume` produces whenever
    it repairs a step and is killed. Nothing upstream can catch it: the records parse, and the
    fingerprint matches because it covers the step's *inputs*, not what is on disk. Read, the
    pair returns each structure's old energy beside another's geometry, and
    `structure_digest` then carries those coordinates into every step computed from them.
    """
    step_dir = _saved(tmp_path)  # ids "0" and "1", both at the origin
    moved = Structure(id="0", atoms=Atoms("H", positions=[[9.0, 9.0, 9.0]]))
    _orphan_sidecar(step_dir, moved, moved, moved)

    with pytest.raises(CacheError, match="from different saves"):
        load(step_dir)
    assert not load_if_valid(key=_key("0", "1", step_cfg=_cfg()), step_dir=step_dir), (
        "a mismatched pair is a cache miss, so the step re-runs"
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_geometry_is_never_cached(tmp_path: Path, bad: float):
    """The half of the no-NaN promise `write_json` cannot make.

    `allow_nan=False` refuses a non-finite value in the *document*, but coordinates are the
    one part of a record that never reaches it: `_split_arrays` moves them into the
    `arrays.npz` sidecar, a raw buffer with no such check. What that cost is not a crash but
    a silence — the geometry round-trips save → load intact and `structure_digest` hashes it
    to a perfectly stable key, so every later step is computed from coordinates that are not
    numbers with nothing anywhere reporting a problem.
    """
    struct = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [bad, 0, 1.5]]))
    with pytest.raises(CacheError, match="non-finite coordinates"):
        save(
            step_cfg=_cfg(),
            key=_key("0", step_cfg=_cfg()),
            results=StepResults(structures=(struct,)),
            step_dir=tmp_path / "step1",
            chemrefine_version="2.0.0",
        )


def test_a_non_finite_force_is_never_cached(tmp_path: Path):
    """Forces ride the same sidecar, and are what an ``mlip-train`` step would go on to fit."""
    struct = Structure(
        id="0",
        atoms=Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        forces_ev_per_a=np.array([[float("nan"), 0.0, 0.0]]),
    )
    with pytest.raises(CacheError, match="non-finite forces"):
        save(
            step_cfg=_cfg(),
            key=_key("0", step_cfg=_cfg()),
            results=StepResults(structures=(struct,)),
            step_dir=tmp_path / "step1",
            chemrefine_version="2.0.0",
        )


def test_a_sidecar_of_the_same_length_is_still_refused(tmp_path: Path):
    """Matching the structure count is not evidence the two files belong together.

    The composition of a step changes between saves without changing its length: under
    ``on_failure: best`` a repaired structure moves out of the backfills and into the
    successes, which reorders the records. Counting them would pass that pair and hand back
    coordinates belonging to a different structure.
    """
    step_dir = _saved(tmp_path)
    _orphan_sidecar(
        step_dir,
        Structure(id="1", atoms=Atoms("H", positions=[[1.0, 0.0, 0.0]])),
        Structure(id="0", atoms=Atoms("H", positions=[[2.0, 0.0, 0.0]])),
    )

    with pytest.raises(CacheError, match="from different saves"):
        load(step_dir)


def test_ragged_structures_round_trip_through_the_sidecar(tmp_path: Path):
    """Structures in one step need not share an atom count, and need not all have forces.

    ``_seed_from_directory`` and ``_seed_from_smiles_csv`` both seed different molecules into
    a single step, so the arrays are concatenated with an offsets index rather than stacked —
    a stack raises outright on this input. Coordinates must survive **exactly**: the fingerprint
    hashes float64 bytes, so a lossy round-trip would invalidate every downstream step.
    """
    big = np.arange(9, dtype=np.float64).reshape(3, 3) / 7.0
    results = StepResults(
        structures=(
            Structure(id="0", atoms=Atoms("H"), forces_ev_per_a=np.array([[0.1, 0.2, 0.3]])),
            Structure(id="1", atoms=Atoms("H3", positions=big)),  # no forces
            Structure(id="2", atoms=Atoms("H2"), forces_ev_per_a=np.ones((2, 3)) / 3.0),
        )
    )
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        key=_key("0", step_cfg=_cfg()),
        results=results,
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )

    loaded = load(step_dir)
    assert loaded is not None
    out = loaded.results.structures
    assert [len(s.atoms) for s in out] == [1, 3, 2]
    assert np.array_equal(out[1].atoms.get_positions(), big), "exact, not merely close"
    assert out[1].forces_ev_per_a is None, "a structure without forces stays without them"
    assert np.array_equal(out[2].forces_ev_per_a, np.ones((2, 3)) / 3.0)


def test_load_if_valid_false_when_parents_change(tmp_path: Path):
    step_dir = tmp_path / "step1"
    cfg = _cfg()
    save(
        step_cfg=cfg,
        key=_key("0", step_cfg=cfg),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    assert not load_if_valid(key=_key("0", "1", step_cfg=cfg), step_dir=step_dir)


def test_load_if_valid_false_when_no_cache(tmp_path: Path):
    assert not load_if_valid(key=_key("0"), step_dir=tmp_path / "step1")


# ---------------------------------------------------------------------------
# load_if_valid — single load + fingerprint check (the cache-hit fast path)
# ---------------------------------------------------------------------------


def test_load_if_valid_returns_cache_on_match(tmp_path: Path):
    """A matching fingerprint returns the cached StepCache (not just a bool)."""
    step_dir = tmp_path / "step1"
    cfg = _cfg()
    save(
        step_cfg=cfg,
        key=_key("0", step_cfg=cfg),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    cached = load_if_valid(key=_key("0", step_cfg=cfg), step_dir=step_dir)
    assert cached is not None
    assert cached.fingerprint == _key("0", step_cfg=cfg).fingerprint


def test_load_if_valid_returns_none_on_mismatch(tmp_path: Path):
    """A changed config returns None (re-run), never a stale cache."""
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        key=_key("0", charge=0),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    assert load_if_valid(key=_key("0", charge=-1), step_dir=step_dir) is None


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------


def test_invalidate_removes_cache_and_legacy_pickle(tmp_path: Path):
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        key=_key("0", step_cfg=_cfg()),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    # A step.pkl beside the document is a stale binary; invalidate sweeps it too.
    (step_dir / "_cache" / "step.pkl").write_bytes(b"legacy")
    invalidate(step_dir)
    assert not (step_dir / "_cache" / "step.json").exists()
    assert not (step_dir / "_cache" / "step.pkl").exists()


def test_invalidate_missing_is_noop(tmp_path: Path):
    invalidate(tmp_path / "missing")  # must not raise


def test_cache_format_version_constant():
    """Bumping CACHE_FORMAT_VERSION is a public ABI break we want to notice."""
    # v2.0: a JSON document whose fingerprint excludes the sample config (filtering re-runs on
    # every load, so a filter-only edit must be a cache hit), which names the digest of the
    # sidecar it was written with, and whose fingerprint covers the *contents* of any file an
    # option points at (`option_digests`, so retraining a model re-runs its consumers).
    # Documents that predate a key are rejected structurally — a summary without `structures`,
    # or one without `arrays_digest`, raises out of `load` and rebuilds — and `option_digests`
    # joins the payload only for a step that names a file, so every existing key is unmoved.
    # None of the three needs a bump while 2.0.0 is unreleased.
    assert CACHE_FORMAT_VERSION == "v2.0"


# ---------------------------------------------------------------------------
# Per-step JSON manifest (save_manifest / load_manifest / manifest_path)
# ---------------------------------------------------------------------------


def _manifest_inputs(tmp_path: Path) -> StepInputs:
    return StepInputs(
        files=(
            (tmp_path / "in/step1_structure_0.inp", tmp_path / "out/step1_structure_0.out", "0"),
            (tmp_path / "in/step1_structure_1.inp", tmp_path / "out/step1_structure_1.out", "1"),
        )
    )


def test_manifest_save_writes_under_cache(tmp_path: Path):
    inputs = _manifest_inputs(tmp_path)
    path = save_manifest(inputs, tmp_path / "step1", operation="opt_sp", engine="fake")
    assert path == manifest_path(tmp_path / "step1")
    assert path.is_file()
    assert path.parent.name == "_cache"


def test_manifest_save_and_load_round_trip(tmp_path: Path):
    inputs = _manifest_inputs(tmp_path)
    save_manifest(inputs, tmp_path / "step1", operation="opt_sp", engine="fake")
    loaded = load_manifest(tmp_path / "step1")
    assert loaded is not None
    assert loaded.files == inputs.files


def test_manifest_load_missing_returns_none(tmp_path: Path):
    assert load_manifest(tmp_path / "step1") is None


def test_manifest_save_records_operation_and_engine(tmp_path: Path):
    import json

    inputs = _manifest_inputs(tmp_path)
    path = save_manifest(inputs, tmp_path / "step1", operation="goat", engine="orca")
    data = json.loads(path.read_text())
    assert data["operation"] == "goat"
    assert data["engine"] == "orca"


def test_manifest_load_corrupt_json_raises_cache_error(tmp_path: Path):
    path = manifest_path(tmp_path / "step1")
    path.parent.mkdir(parents=True)
    path.write_text("not valid json {", encoding="utf-8")
    with pytest.raises(CacheError):
        load_manifest(tmp_path / "step1")


def test_manifest_load_missing_files_key_raises_cache_error(tmp_path: Path):
    """A manifest with no ``files`` key must surface as a ``CacheError``."""
    path = manifest_path(tmp_path / "step1")
    path.parent.mkdir(parents=True)
    path.write_text('{"operation": "opt_sp", "engine": "orca"}', encoding="utf-8")
    with pytest.raises(CacheError):
        load_manifest(tmp_path / "step1")


# ---------------------------------------------------------------------------
# Canonical result records
# ---------------------------------------------------------------------------


def test_save_result_records_writes_versioned_canonical_records(tmp_path: Path):
    """Each structure gets a ``*.result.json`` wrapping its structure_record."""
    structure = Structure(
        id="0-1",
        atoms=Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        parent_id="0",
        energy_hartree=-75.5,
        gibbs_hartree=-75.4,
        converged=True,
        imaginary_freqs={3: -101.5},
    )
    save_result_records([structure], tmp_path, step=2)

    path = tmp_path / "step2_0-1.result.json"
    assert path.is_file()
    record = json.loads(path.read_text())
    assert record["result_format"] == RESULT_FORMAT_VERSION
    assert record["id"] == "0-1"
    assert record["parent_id"] == "0"
    assert record["energy_hartree"] == -75.5
    assert record["gibbs_hartree"] == -75.4
    assert record["imaginary_freqs"] == {"3": -101.5}
    assert record["symbols"] == ["O", "H"]


def test_result_record_round_trips_through_structure_from_record(tmp_path: Path):
    """The record body is the one canonical schema — the cache reader accepts it."""
    structure = Structure(
        id="0",
        atoms=Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        energy_hartree=-1.17,
        forces_ev_per_a=np.zeros((2, 3)),
        converged=True,
        terminated_normally=True,
    )
    save_result_records([structure], tmp_path, step=1)
    record = json.loads((tmp_path / "step1_0.result.json").read_text())
    record.pop("result_format")

    rebuilt = structure_from_record(record)

    assert rebuilt.id == structure.id
    assert rebuilt.energy_hartree == structure.energy_hartree
    assert rebuilt.converged is True
    assert rebuilt.atoms.get_chemical_symbols() == ["H", "H"]


def test_both_mode_tables_round_trip_and_neither_is_required(tmp_path: Path):
    """The whole table is persisted so a finished tree can be read without its outputs.

    Additive, like ``resolved_from``: a record written before the key existed loads as
    ``None`` and needs no :data:`RESULT_FORMAT_VERSION` bump. ``None`` must survive as
    ``None`` rather than flattening to ``{}`` — "no frequency table" and "a table with no
    imaginary modes" are the difference between "unknown" and "a verified minimum", which
    is what NMS's ``_is_resolved`` branches on.
    """
    structure = Structure(
        id="0",
        atoms=Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        energy_hartree=-1.17,
        imaginary_freqs={0: -512.4},
        frequencies={0: -512.4, 6: 284.9, 7: 1103.7},
    )
    record = structure_record(structure)
    assert record["frequencies"] == {"0": -512.4, "6": 284.9, "7": 1103.7}
    rebuilt = structure_from_record(record)
    assert rebuilt.frequencies == structure.frequencies
    assert rebuilt.imaginary_freqs == structure.imaginary_freqs

    older = {k: v for k, v in record.items() if k != "frequencies"}
    assert structure_from_record(older).frequencies is None

    empty = structure_record(Structure(id="1", atoms=Atoms("H"), imaginary_freqs={}))
    assert (empty["imaginary_freqs"], empty["frequencies"]) == ({}, None)
    assert structure_from_record(empty).imaginary_freqs == {}  # not None: a verified minimum


# --- cache: corrupt failed-jobs ledger --------------------------------------


def test_load_failure_records_raises_on_corrupt_ledger(tmp_path: Path):
    from chemrefine import cache

    path = cache.failed_jobs_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CacheError, match="corrupt failed-jobs ledger"):
        cache.load_failure_records(tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param('{"oops": 1}', id="an object where the record list belongs"),
        pytest.param('["a", "b"]', id="a list of strings instead of records"),
        pytest.param(
            '[{"structure_id": "0", "kind": "no-such-kind", "reason": ""}]',
            id="a record whose kind this version does not know",
        ),
    ],
)
def test_load_failure_records_raises_on_wrong_shape_ledger(tmp_path: Path, body: str):
    """Valid JSON of the wrong shape must become a CacheError naming the file.

    `read_json` only proves the file parsed; without the shape guard these escaped as a
    bare TypeError/ValueError — a traceback with the generic exit code, naming neither the
    file nor the fix — where every sibling `_cache/` reader raises CacheError (exit 7).
    """
    from chemrefine import cache

    path = cache.failed_jobs_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    with pytest.raises(CacheError, match="corrupt failed-jobs ledger") as excinfo:
        cache.load_failure_records(tmp_path)
    assert str(path) in str(excinfo.value)


def test_cache_documents_honour_the_umask(tmp_path: Path):
    """``mkstemp``'s 0600 must not survive into the documents a shared tree serves.

    The temp file is rightly private, but the rename preserved its mode — so a colleague
    handed an output tree could read every ``.out`` beside a cache they could not: the
    step document, the manifest and the failure ledger were all owner-only. The atomic
    writer re-modes to what a plain ``open()`` would have given, 0666 through the umask.
    (The server *token* sidecar keeps its 0600 — there the restriction is the point, and
    it does not go through this writer.)
    """
    import os

    previous = os.umask(0o022)
    try:
        save(
            step_cfg=_cfg(),
            key=_key("0", step_cfg=_cfg()),
            results=_results(),
            step_dir=tmp_path / "step1",
            chemrefine_version="2.0.0",
        )
    finally:
        os.umask(previous)

    for name in ("step.json", "arrays.npz"):
        mode = (tmp_path / "step1" / "_cache" / name).stat().st_mode & 0o777
        assert mode == 0o644, (
            f"{name} is {oct(mode)}, unreadable to the group the tree is shared with"
        )


def test_a_write_that_cannot_re_mode_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A failed re-mode must strand neither the temp file nor its descriptor.

    ``fchmod`` is the one step between ``mkstemp`` and ``fdopen`` that genuinely fails —
    shared filesystems and FUSE/CIFS mounts refuse it. Done before the ``fdopen`` it took
    both halves with it: nothing owned the descriptor yet, and the ``finally`` had not been
    entered, so a ``.tmp_*.part`` stayed in the directory the user was writing to. That is a
    nuisance once and an accumulation on the GUI's waitress threads, which write the user's
    config through this same writer.
    """
    import os

    monkeypatch.setattr(os, "fchmod", lambda *a: (_ for _ in ()).throw(OSError("EPERM")))
    target = tmp_path / "doc.json"
    with pytest.raises(OSError, match="EPERM"):
        cache.atomic_write(target, b'{"a": 1}')

    assert not target.exists(), "a failed write must not leave a partial document"
    assert list(tmp_path.iterdir()) == [], (
        f"temp residue survived a failed re-mode: {[p.name for p in tmp_path.iterdir()]}"
    )


@pytest.mark.skipif(
    not cache._PROC_STATUS.is_file(),
    reason="this kernel does not publish Umask; only the fallback probe exists here",
)
def test_the_umask_is_read_without_ever_being_written(monkeypatch: pytest.MonkeyPatch):
    """Reading the umask by setting it is a process-global write other threads see.

    ``os.umask`` reads by writing, so between the clear and the restore anything another
    thread creates is made with no mask at all and a ``mkdir`` lands 0777 — world-writable
    directories under an output tree that is routinely shared. The window is reachable:
    the GUI runs on four waitress threads whose handlers both write through this module
    and call ``Path.mkdir``. Linux publishes the value instead (``umask(2)``), so nothing
    here may call ``os.umask`` at all.
    """
    import os

    expected = next(
        int(line.split()[1], 8)
        for line in cache._PROC_STATUS.read_text(encoding="utf-8").splitlines()
        if line.startswith("Umask:")
    )
    monkeypatch.setattr(os, "umask", lambda _mask: pytest.fail("read the umask by setting it"))

    assert cache._umask() == expected


@pytest.mark.parametrize("published", ["no status file at all", "a status file without the field"])
def test_the_umask_falls_back_when_the_kernel_does_not_publish_it(
    published: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A kernel older than 4.7 has no ``Umask`` field, and must still get the right mode.

    Both ways the field can be absent take the probe, and the probe has to put back what
    it set — otherwise the first cache write of the run would leave the process at umask
    0 and every file after it world-writable, which is worse than the race it replaces.
    Asking twice is what proves the restore happened.
    """
    import os

    status = tmp_path / "status"
    if published == "a status file without the field":
        status.write_text("Name:\tpython\nPid:\t1\n", encoding="utf-8")
    monkeypatch.setattr(cache, "_PROC_STATUS", status)

    previous = os.umask(0o027)
    try:
        assert cache._umask() == 0o027
        assert cache._umask() == 0o027
    finally:
        os.umask(previous)
