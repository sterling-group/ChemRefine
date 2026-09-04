"""Tests for the shared template renderer + ``ScriptEngine`` helpers.

The per-engine ``test_engines_pyscf.py`` and ``test_engines_mlip.py``
exercise the lifecycle end-to-end with their backend labels. The
tests here exercise the *shared* surface area — the renderer
(``_template_render.build_input``) and the output-parsing helpers
(the contract's ``positions_from`` / ``forces_from_gradient`` converters) — once,
not twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines._options import EngineOptions
from chemrefine.engines._script import render as _template_render
from chemrefine.engines._script.contract import (
    SCRIPT_OUTPUT,
    OutputField,
    forces_from_gradient,
    positions_from,
)
from chemrefine.engines._script.engine import ScriptEngine
from chemrefine.engines._script.output import _load_output_json, parse_output
from chemrefine.errors import ChemRefineError, ConfigError, OutputParseError
from chemrefine.state import PipelineState, StepContext, Structure


def test_base_template_vars_default_is_empty():
    """The base exposes no placeholders; subclasses (mlip/pyscf) override ``_vars_from``.

    ``_template_vars`` itself is not overridden by anyone — reading the options through
    ``options_cls``, leniently, is the part that must not vary between engines.
    """
    assert ScriptEngine()._vars_from(EngineOptions()) == {}


# ---------------------------------------------------------------------------
# _template_render.build_input — renderer
# ---------------------------------------------------------------------------


def test_build_input_substitutes_geometry_placeholders(tmp_path: Path):
    template = tmp_path / "step1.py"
    template.write_text(
        'mol = gto.M(atom="$XYZ_PATH", charge=$CHARGE, spin=$MULTIPLICITY - 1)\n',
        encoding="utf-8",
    )
    out = tmp_path / "rendered.py"
    _template_render.build_input(
        xyz_path=tmp_path / "frame.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "frame.json",
        charge=-1,
        multiplicity=2,
    )
    text = out.read_text(encoding="utf-8")
    assert f'atom="{tmp_path / "frame.xyz"}"' in text
    assert "charge=-1" in text
    assert "spin=2 - 1" in text


def test_build_input_appends_output_footer(tmp_path: Path):
    """The rendered file must end with the canonical JSON-writing footer."""
    template = tmp_path / "step1.py"
    template.write_text("energy_hartree = -1.0\n", encoding="utf-8")
    out = tmp_path / "rendered.py"
    _template_render.build_input(
        xyz_path=tmp_path / "frame.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "step1_structure_0.json",
        charge=0,
        multiplicity=1,
    )
    text = out.read_text(encoding="utf-8")
    assert "energy_hartree = -1.0" in text
    assert "# --- ChemRefine output footer (generated; do not edit) ---" in text
    assert "_chemrefine_result" in text
    # Writes to the BASENAME so it lands in $WORK_DIR.
    assert 'with open(\'step1_structure_0.json\', "w", encoding="utf-8")' in text
    assert "$OUTPUT_JSON" not in text


def test_build_input_leaves_legacy_output_json_placeholder_alone(tmp_path: Path):
    """``$OUTPUT_JSON`` is not a known placeholder; ``safe_substitute`` leaves it intact."""
    template = tmp_path / "step1.py"
    template.write_text(
        "energy_hartree = -1.0\n# legacy: $OUTPUT_JSON\n",
        encoding="utf-8",
    )
    out = tmp_path / "rendered.py"
    _template_render.build_input(
        xyz_path=tmp_path / "frame.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "step1_structure_0.json",
        charge=0,
        multiplicity=1,
    )
    text = out.read_text(encoding="utf-8")
    assert "# legacy: $OUTPUT_JSON" in text
    assert 'with open(\'step1_structure_0.json\', "w", encoding="utf-8")' in text


def test_build_input_missing_template_raises_generic(tmp_path: Path):
    """Direct call (no engine layer above it) surfaces the generic message."""
    with pytest.raises(ConfigError, match="template not found"):
        _template_render.build_input(
            xyz_path=tmp_path / "x.xyz",
            template_path=tmp_path / "missing.py",
            output_path=tmp_path / "out.py",
            output_json_path=tmp_path / "out.json",
            charge=0,
            multiplicity=1,
        )


def test_build_input_preserves_python_braces(tmp_path: Path):
    """A real script uses Python ``{ ... }`` everywhere; those must survive intact."""
    template = tmp_path / "step1.py"
    template.write_text(
        'payload = {"key": float(mf.e_tot)}\n'
        "gradient = [(i, x) for i, x in enumerate(grad)]\n"
        'f = f"step{step}_done"\n'
        "energy_hartree = -1.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "rendered.py"
    _template_render.build_input(
        xyz_path=tmp_path / "x.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "y.json",
        charge=0,
        multiplicity=1,
    )
    text = out.read_text()
    assert 'payload = {"key": float(mf.e_tot)}' in text
    assert "gradient = [(i, x) for i, x in enumerate(grad)]" in text
    assert 'f = f"step{step}_done"' in text


def test_build_input_leaves_unknown_placeholders_intact(tmp_path: Path):
    """``safe_substitute`` should leave ``$UNKNOWN`` references alone."""
    template = tmp_path / "step1.py"
    template.write_text(
        'pyenv = "$VIRTUAL_ENV"\nenergy_hartree = -1.0\n',
        encoding="utf-8",
    )
    out = tmp_path / "rendered.py"
    _template_render.build_input(
        xyz_path=tmp_path / "x.xyz",
        template_path=template,
        output_path=out,
        output_json_path=tmp_path / "y.json",
        charge=0,
        multiplicity=1,
    )
    assert "$VIRTUAL_ENV" in out.read_text()


# ---------------------------------------------------------------------------
# positions_from / forces_from_gradient — the contract's own converters
# ---------------------------------------------------------------------------


def _ctx(tmp: Path) -> StepContext:
    """The smallest StepContext ``build_input`` reads — step number, charge, multiplicity."""
    return StepContext(
        step_cfg=StepConfig(step=1, engine="extended-probe", operation="sp", options={}),
        step_dir=tmp,
        template_dir=tmp,
        template=tmp / "step1.py",
        scratch_dir=None,
        prev_state=PipelineState(structures=()),
        charge=0,
        multiplicity=1,
        max_cores=1,
        slurm_template="cpu.slurm.header",
    )


def _parse(document: dict, seed: Atoms, *, fields=SCRIPT_OUTPUT, tmp: Path | None = None):
    """Round-trip one output document through ``parse_output`` and return its ParsedResult."""
    import json
    import tempfile

    out = Path(tmp or tempfile.mkdtemp()) / "step1_0.json"
    out.write_text(json.dumps(document), encoding="utf-8")
    return parse_output(out, label="MLIP", fallback=seed, fields=fields)[0]


def test_a_script_that_reports_no_geometry_keeps_the_seeds():
    """An absent optional field leaves the ParsedResult with the seed's own positions."""
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    parsed = _parse({"energy_hartree": -1.0}, seed)
    np.testing.assert_allclose(parsed.positions, seed.get_positions())


def test_positions_from_uses_what_the_script_reported():
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    moved = positions_from([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], seed)
    np.testing.assert_allclose(moved, [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])


def test_parse_output_raises_without_a_seed(tmp_path: Path):
    """The symbols come from the seed always, so a parse without one cannot produce one.

    The document itself is perfectly good — the refusal is about what it does not carry.
    """
    out = tmp_path / "step1_0.json"
    out.write_text('{"energy_hartree": -1.0}', encoding="utf-8")
    with pytest.raises(OutputParseError, match="no seed atoms"):
        parse_output(out, label="MLIP", fallback=None)


def test_forces_from_gradient_converts_units():
    from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A

    forces = forces_from_gradient([[1.0, 0.0, 0.0]], Atoms("H", positions=[[0, 0, 0]]))
    assert forces is not None
    np.testing.assert_allclose(forces[0], [-HARTREE_PER_BOHR_TO_EV_PER_A, 0.0, 0.0])


def test_forces_from_gradient_handles_none():
    assert forces_from_gradient(None, Atoms("H", positions=[[0, 0, 0]])) is None


def test_forces_from_gradient_handles_empty():
    assert forces_from_gradient([], Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])) is None


# ---------------------------------------------------------------------------
# Shape — a wrong-shaped array is one structure's failure, not the run's
# ---------------------------------------------------------------------------


def _write_output(tmp_path: Path, body: str) -> Path:
    out = tmp_path / "step1_0.json"
    out.write_text(body, encoding="utf-8")
    return out


@pytest.mark.parametrize(
    "positions",
    [
        pytest.param([0.0, 0.0, 0.0, 0.74, 0.0, 0.0], id="flat-3N-list"),
        pytest.param([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], id="wrong-atom-count"),
        pytest.param([[0.0, 0.0], [0.74, 0.0]], id="wrong-column-count"),
    ],
)
def test_positions_of_the_wrong_shape_are_refused(positions: list):
    """`positions_angstrom` must match the seed geometry, or be a parse failure.

    ASE raises a bare ValueError, which is outside the package hierarchy that
    `lifecycle._parse_job` and `cli._dispatch` catch — so a single structure's malformed
    output would end the whole run in a traceback instead of becoming its ledger entry.
    """
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    with pytest.raises(OutputParseError, match="positions_angstrom"):
        positions_from(positions, seed)


def test_positions_from_copies_rather_than_mutating_the_seed():
    """An optimised geometry must land on a copy of the seed, never on the seed.

    The fallback is the pipeline's own structure, shared by reference; written in place,
    the input geometry every later reader sees — including the `structure_digest` behind
    downstream cache keys — would silently become the output geometry. The `.copy()` is
    the whole protection (`Structure.atoms` cannot be write-locked the way the force
    arrays are), so this pins it.
    """
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    before = seed.get_positions().copy()
    moved = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
    updated = positions_from(moved, seed)
    assert np.array_equal(seed.get_positions(), before)
    assert np.array_equal(updated, np.asarray(moved))


def test_a_ragged_gradient_is_refused():
    """The other half of the same shape contract — `np.asarray` would raise bare, too."""
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    with pytest.raises(OutputParseError, match="gradient_hartree_per_bohr"):
        forces_from_gradient([[0.1, 0.2, 0.3], [0.1, 0.2]], seed)


@pytest.mark.parametrize(
    "gradient",
    [
        pytest.param([0.1, 0.2, 0.3, 0.1, 0.2, 0.3], id="flat-3N-list"),
        pytest.param([[0.1, 0.2, 0.3]], id="wrong-atom-count"),
        pytest.param([[0.1, 0.2], [0.3, 0.1]], id="wrong-column-count"),
    ],
)
def test_a_gradient_of_the_wrong_shape_is_refused(gradient: list):
    """The gradient is held to the positions guard's rule, by an explicit check.

    The positions path has ``set_positions`` as its shape oracle; a gradient has none, so
    the flat ``3N`` list that guard names as "the natural mistake" parsed as a valid
    ``(3N,)`` array and rode ``forces_ev_per_a`` — which declares no shape — through the
    cache and into any downstream ``mlip-train`` dataset, where the positions equivalent
    was an ordinary ledger entry naming the atom count.
    """
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    with pytest.raises(OutputParseError, match="gradient_hartree_per_bohr"):
        forces_from_gradient(gradient, seed)


@pytest.mark.parametrize(
    "body",
    [
        '{"energy_hartree": -1.0, "positions_angstrom": [0.0, 0.0, 0.0, 0.74, 0.0, 0.0]}',
        '{"energy_hartree": -1.0, "gradient_hartree_per_bohr": [[0.1, 0.2, 0.3], [0.1, 0.2]]}',
        '{"energy_hartree": -1.0, "gradient_hartree_per_bohr": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]}',
    ],
)
def test_a_malformed_shape_stays_inside_the_exit_code_contract(tmp_path: Path, body: str):
    """End to end: every failure `parse_output` can raise carries an `exit_code`.

    The guarantee the CLI depends on — it catches `ChemRefineError` and nothing else — so
    this asserts the base class rather than the leaf, which is what the contract is about.
    """
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    with pytest.raises(ChemRefineError):
        parse_output(_write_output(tmp_path, body), label="MLIP", fallback=seed)


def test_a_null_required_field_is_refused_by_name(tmp_path: Path):
    """`required` means present *and non-null* — key presence alone let null through.

    `{"energy_hartree": null}` passed the old presence check, was skipped by every later
    read, and surfaced as `TypeError: ParsedResult.__init__() missing 1 required
    positional argument` — outside the OutputParseError family `lifecycle._parse_job`
    contains, so one structure's odd output ended the whole run with a traceback instead
    of a ledgered failure.
    """
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    out = _write_output(tmp_path, '{"energy_hartree": null}')
    with pytest.raises(OutputParseError, match=r"energy_hartree.*absent or null"):
        parse_output(out, label="MLIP", fallback=seed)


@pytest.mark.parametrize("body", ["null", "[1, 2]", '"a string"'])
def test_a_non_mapping_document_is_refused_as_a_parse_failure(tmp_path: Path, body: str):
    """Valid JSON that is not an object must refuse like malformed JSON does.

    A top-level null raised `TypeError` at the `in` test and a list raised
    `AttributeError` at `.get` — the same escape route as the null field above. The
    refusal names the shape so the reader is sent to their document, not to a traceback.
    """
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    out = _write_output(tmp_path, body)
    with pytest.raises(OutputParseError, match="not a JSON mapping"):
        parse_output(out, label="MLIP", fallback=seed)


# ---------------------------------------------------------------------------
# _load_output_json — a diverged calculation must not read as a result
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_energy_is_refused(tmp_path: Path, literal: str):
    """A diverged calculation reports `nan`/`inf`; it must not rank as a real result.

    Nothing downstream would catch it: `succeeded` reads only an explicit False flag, and
    `filtering.apply` drops an energy that is None, not one that is NaN — so the structure
    would sort by list position (every NaN comparison is false) and displace a real survivor.
    """
    out = _write_output(tmp_path, f'{{"energy_hartree": {literal}}}')
    with pytest.raises(OutputParseError, match="non-finite"):
        _load_output_json(out, label="MLIP")


def test_a_non_finite_gradient_component_is_refused(tmp_path: Path):
    """The other half of the same diverged calculation — and what the trainer would fit."""
    out = _write_output(
        tmp_path, '{"energy_hartree": -1.0, "gradient_hartree_per_bohr": [[0.0, NaN, 0.0]]}'
    )
    with pytest.raises(OutputParseError, match=r"non-finite.*gradient_hartree_per_bohr"):
        _load_output_json(out, label="MLIP")


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_position_is_refused(tmp_path: Path, literal: str):
    """The third field of the same diverged calculation, and the one with two failure modes.

    Inline in a `.result.json`, a NaN coordinate meets `write_json`'s `allow_nan=False` and
    raises a bare `ValueError` that `lifecycle._parse_job` does not catch — the whole run
    ends over one structure. On the paths that write no record it goes to the `arrays.npz`
    sidecar instead, which has no such check, and is simply served to every later step.
    """
    out = _write_output(
        tmp_path, f'{{"energy_hartree": -1.0, "positions_angstrom": [[0.0, {literal}, 0.0]]}}'
    )
    with pytest.raises(OutputParseError, match=r"non-finite.*positions_angstrom"):
        _load_output_json(out, label="MLIP")


def test_a_finite_position_still_passes(tmp_path: Path):
    """The guard must not reject an ordinary optimised geometry."""
    out = _write_output(
        tmp_path, '{"energy_hartree": -1.5, "positions_angstrom": [[0.0, 0.0, 0.0]]}'
    )
    assert _load_output_json(out, label="MLIP")["positions_angstrom"] == [[0.0, 0.0, 0.0]]


def test_a_non_numeric_energy_is_refused(tmp_path: Path):
    """`float()` on a string would escape as a bare ValueError, past the exit-code contract."""
    out = _write_output(tmp_path, '{"energy_hartree": "diverged"}')
    with pytest.raises(OutputParseError, match="non-numeric"):
        _load_output_json(out, label="MLIP")


def test_a_finite_energy_and_gradient_still_pass(tmp_path: Path):
    """The guard must not reject the ordinary case."""
    out = _write_output(
        tmp_path, '{"energy_hartree": -1.5, "gradient_hartree_per_bohr": [[0.0, 1e-9, -2.0]]}'
    )
    assert _load_output_json(out, label="MLIP")["energy_hartree"] == -1.5


# ---------------------------------------------------------------------------
# The seam: an engine that reports more than the shared three
# ---------------------------------------------------------------------------


def test_an_engine_can_extend_the_output_contract_without_touching_a_building_block(
    tmp_path: Path,
):
    """The promise the declaration exists for, exercised end to end.

    A script engine that needs to report something beyond the shared set used to have no way
    to say so: the harvested names lived in a string literal inside ``_script/render.py`` and
    the mapping in ``_script/output.py``, so an extra quantity meant editing two building
    blocks that ``docs/developer/adding-an-engine.md`` says are never edited to add an engine.
    Here the whole extension is one ClassVar on the engine, and the footer, the JSON mapping
    and the ``ParsedResult`` follow from it.

    Thermochemistry is the example because the shared set does not carry it. ``converged``,
    once the motivating case, ships in :data:`SCRIPT_OUTPUT` now: the starters assign it, and
    an exhausted optimiser is a failure rather than a survivor.
    """

    class _Extended(ScriptEngine[EngineOptions]):
        name = "extended-probe"
        label = "Extended"
        output_fields = (
            *SCRIPT_OUTPUT,
            OutputField("gibbs_hartree", "gibbs_hartree"),
            OutputField("enthalpy_hartree", "enthalpy_hartree"),
        )

    engine = _Extended()

    # The footer it renders offers the template the extra names...
    template = tmp_path / "step1.py"
    template.write_text("energy_hartree = -1.0\n", encoding="utf-8")
    rendered = tmp_path / "step1_0.py"
    engine.build_input(
        xyz_path=tmp_path / "step1_0_inp.xyz",
        template_path=template,
        input_path=rendered,
        output_path=tmp_path / "step1_0.json",
        ctx=_ctx(tmp_path),
    )
    footer = rendered.read_text(encoding="utf-8")
    assert '"gibbs_hartree"' in footer and '"enthalpy_hartree"' in footer

    # ...and the reader lands them on the ParsedResult, with no edit to _script/.
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    parsed = _parse(
        {"energy_hartree": -1.0, "gibbs_hartree": -0.9, "enthalpy_hartree": -0.95},
        seed,
        fields=_Extended.output_fields,
        tmp=tmp_path,
    )
    assert parsed.gibbs_hartree == -0.9
    assert parsed.enthalpy_hartree == -0.95


def test_an_extended_field_is_swept_for_finiteness_like_every_other(tmp_path: Path):
    """The guard follows the declaration, which is the half a hand-kept roster drops.

    A quantity the footer harvests but a tuple in the reader does not know about is written by
    the script, read onto the structure, and never checked — so a diverged calculation reports
    ``nan`` and caches as a result. Declaring ``finite`` is the only thing that decides.
    """
    fields = (*SCRIPT_OUTPUT, OutputField("gibbs_hartree", "gibbs_hartree"))
    seed = Atoms("H", positions=[[0, 0, 0]])
    with pytest.raises(OutputParseError, match="non-finite 'gibbs_hartree'"):
        _parse({"energy_hartree": -1.0, "gibbs_hartree": float("nan")}, seed, fields=fields)


def test_a_flag_field_is_exempt_from_the_finiteness_sweep():
    """``finite=False`` is for a value the question does not apply to.

    ``converged`` in the shared contract is that value: ``True``/``False`` land as written, and
    a template that never assigns it leaves ``None`` — "not reported", the shape every engine
    that sets no flag has always had — rather than a failure.
    """
    seed = Atoms("H", positions=[[0, 0, 0]])
    assert _parse({"energy_hartree": -1.0, "converged": True}, seed).converged is True
    assert _parse({"energy_hartree": -1.0, "converged": False}, seed).converged is False
    assert _parse({"energy_hartree": -1.0}, seed).converged is None


def test_a_sidecar_field_is_harvested_but_lands_on_no_result_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``OutputField(field=None)`` is harvested into the raw JSON and read onto nothing.

    Diagnostics an engine wants kept — a solver's iteration history, the components a run
    resolved — have no ``ParsedResult`` home and should not grow one: the record is the
    canonical chemistry, the raw ``stepN_<id>.json`` is the engine's own. Declared with
    ``field=None`` the footer harvests the name, the finiteness rule still applies as
    declared, and the reader skips it — so the declaration stays on the engine, and the
    footer hard-codes nothing.
    """
    import json

    class _Sidecar(ScriptEngine[EngineOptions]):
        name = "sidecar-probe"
        label = "Sidecar"
        output_fields = (
            *SCRIPT_OUTPUT,
            OutputField("engine_metadata", None, finite=False),
            OutputField("residual", None),
        )

    engine = _Sidecar()
    template = tmp_path / "step1.py"
    template.write_text(
        'energy_hartree = -1.0\nengine_metadata = {"evaluations": 3}\nresidual = 1e-9\n',
        encoding="utf-8",
    )
    rendered = tmp_path / "step1_0.py"
    output = tmp_path / "step1_0.json"
    engine.build_input(
        xyz_path=tmp_path / "step1_0_inp.xyz",
        template_path=template,
        input_path=rendered,
        output_path=output,
        ctx=_ctx(tmp_path),
    )
    assert '"engine_metadata"' in rendered.read_text(encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # the footer writes its basename into the cwd
    exec(compile(rendered.read_text(encoding="utf-8"), str(rendered), "exec"), {})
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["engine_metadata"] == {"evaluations": 3}
    assert document["residual"] == 1e-9

    seed = Atoms("H", positions=[[0, 0, 0]])
    parsed = parse_output(output, label="Sidecar", fallback=seed, fields=_Sidecar.output_fields)[0]
    assert parsed.energy_hartree == -1.0
    assert not hasattr(parsed, "engine_metadata")

    # The finiteness rule follows the declaration even for a field nothing reads back.
    with pytest.raises(OutputParseError, match="non-finite 'residual'"):
        _parse(
            {"energy_hartree": -1.0, "residual": float("nan")},
            seed,
            fields=_Sidecar.output_fields,
        )


def test_an_unconverged_script_is_a_convergence_failure_with_a_geometry_to_retry_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The shared contract's ``converged`` reaches the lifecycle's verdict, end to end.

    The path a shipped starter takes: the template assigns ``converged = False`` (what
    ``MlipCalculator.optimize`` reports when it runs out of steps), the generated footer
    harvests it, the parser lands it, and ``lifecycle`` classifies the structure as
    ``NOT_CONVERGED`` with the last geometry as the one to retry from — instead of the
    survivor it used to be when nothing carried the verdict.
    """
    from chemrefine import lifecycle
    from chemrefine.state import FailureKind, StepInputs

    template = tmp_path / "step1.py"
    template.write_text(
        "energy_hartree = -1.0\npositions_angstrom = [[0.0, 0.0, 0.1]]\nconverged = False\n",
        encoding="utf-8",
    )
    rendered = tmp_path / "step1_0.py"
    output = tmp_path / "step1_0.json"

    class _Probe(ScriptEngine[EngineOptions]):
        name = "verdict-probe"
        label = "Probe"

    engine = _Probe()
    engine.build_input(
        xyz_path=tmp_path / "step1_0_inp.xyz",
        template_path=template,
        input_path=rendered,
        output_path=output,
        ctx=_ctx(tmp_path),
    )
    monkeypatch.chdir(tmp_path)  # the footer writes its basename into the cwd
    exec(compile(rendered.read_text(encoding="utf-8"), str(rendered), "exec"), {})

    seed = Structure(id="0", atoms=Atoms("H", positions=[[0, 0, 0]]))
    ctx = StepContext(
        step_cfg=StepConfig(step=1, engine="pyscf"),
        step_dir=tmp_path,
        template_dir=tmp_path,
        template=template,
        scratch_dir=None,
        prev_state=PipelineState(structures=(seed,)),
        charge=0,
        multiplicity=1,
        max_cores=1,
        slurm_template="cpu.slurm.header",
    )
    inputs = StepInputs(files=((rendered, output, "0"),))
    successes, failures = lifecycle.parse_with_failures(engine, inputs, ctx)

    assert successes == []
    assert [f.kind for f in failures] == [FailureKind.NOT_CONVERGED]
    best = lifecycle.retryable_best(failures[0])
    assert best is not None and best.converged is False
    assert best.atoms.get_positions().tolist() == [[0.0, 0.0, 0.1]]


def test_the_scaffold_starter_names_the_engines_own_contract():
    """The starter comment is generated from the engine's fields, not typed per starter."""
    from chemrefine.engines.api import get_engine
    from chemrefine.scaffold import _output_contract_comment

    comment = _output_contract_comment(get_engine("mlip"))
    for name in ("energy_hartree", "positions_angstrom", "gradient_hartree_per_bohr", "converged"):
        assert name in comment
    assert _output_contract_comment(get_engine("orca")) == "", (
        "a non-script engine has no such contract"
    )


@pytest.mark.parametrize(
    ("fields", "expected_tuple"),
    [
        pytest.param(
            SCRIPT_OUTPUT,
            '("positions_angstrom", "gradient_hartree_per_bohr", "converged")',
            id="three",
        ),
        pytest.param(SCRIPT_OUTPUT[:2], '("positions_angstrom",)', id="one"),
        pytest.param(SCRIPT_OUTPUT[:1], "()", id="none"),
    ],
)
def test_the_footers_optional_tuple_is_a_tuple_at_every_arity(
    fields, expected_tuple: str, tmp_path: Path
):
    """A one-element tuple needs its trailing comma; a zero-element one must not have one.

    ``("x")`` is a string, and the harvest loop would iterate its characters — looking up
    ``"p"``, ``"o"``, ``"s"`` in the template's locals and finding nothing, so a contract with
    exactly one optional field would silently report none of it.
    """
    # An absolute destination, because the footer's own `open()` really runs below and its
    # basename is relative to the process cwd by design — which for a test is the checkout.
    footer = _template_render._build_output_footer(str(tmp_path / "out.json"), fields)
    assert f"_chemrefine_optional = {expected_tuple}\n" in footer
    namespace: dict[str, object] = {"energy_hartree": -1.0, "positions_angstrom": [[0.0, 0.0, 0.0]]}
    exec(compile(footer, "<footer>", "exec"), namespace)
    assert isinstance(namespace["_chemrefine_optional"], tuple)
    assert (tmp_path / "out.json").is_file()


def test_a_required_only_contract_names_no_optional_fields():
    """The comment must not read "(optionally )" for a contract that has none."""
    from chemrefine.scaffold import _output_contract_comment

    class _RequiredOnly(ScriptEngine[EngineOptions]):
        name = "required-only-probe"
        label = "RequiredOnly"
        output_fields = (OutputField("energy_hartree", "energy_hartree", required=True),)

    assert _output_contract_comment(_RequiredOnly) == (
        "# Assign `energy_hartree` — the appended output footer harvests them.\n"
    )
