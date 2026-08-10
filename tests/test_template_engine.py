"""Tests for the shared template renderer + ``ScriptEngine`` helpers.

The per-engine ``test_engines_pyscf.py`` and ``test_engines_mlip.py``
exercise the lifecycle end-to-end with their backend labels. The
tests here exercise the *shared* surface area — the renderer
(``_template_render.build_input``) and the output-parsing helpers
(``_template_output._atoms_from_output`` / ``_forces_from_gradient``) — once,
not twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine.engines._options import EngineOptions
from chemrefine.engines._script import render as _template_render
from chemrefine.engines._script.engine import ScriptEngine
from chemrefine.engines._script.output import (
    _atoms_from_output,
    _forces_from_gradient,
    _load_output_json,
    parse_output,
)
from chemrefine.errors import ChemRefineError, ConfigError, OutputParseError


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
    assert "with open('step1_structure_0.json', \"w\")" in text
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
    assert "with open('step1_structure_0.json', \"w\")" in text


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
# _atoms_from_output / _forces_from_gradient — shared helpers
# ---------------------------------------------------------------------------


def test_atoms_from_output_falls_back_to_seed_atoms():
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms = _atoms_from_output({"energy_hartree": -1.0}, fallback=seed)
    np.testing.assert_allclose(atoms.get_positions(), seed.get_positions())


def test_atoms_from_output_uses_positions_when_present():
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms = _atoms_from_output(
        {"energy_hartree": -1.0, "positions_angstrom": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]},
        fallback=seed,
    )
    np.testing.assert_allclose(atoms.get_positions(), [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])


def test_atoms_from_output_raises_without_fallback_and_no_positions():
    with pytest.raises(OutputParseError, match="positions_angstrom"):
        _atoms_from_output({"energy_hartree": -1.0}, fallback=None)


def test_forces_from_gradient_converts_units():
    from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A

    forces = _forces_from_gradient([[1.0, 0.0, 0.0]])
    assert forces is not None
    np.testing.assert_allclose(forces[0], [-HARTREE_PER_BOHR_TO_EV_PER_A, 0.0, 0.0])


def test_forces_from_gradient_handles_none():
    assert _forces_from_gradient(None) is None


def test_forces_from_gradient_handles_empty():
    assert _forces_from_gradient([]) is None


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
        _atoms_from_output({"energy_hartree": -1.0, "positions_angstrom": positions}, fallback=seed)


def test_atoms_from_output_copies_rather_than_mutating_the_seed():
    """An optimised geometry must land on a copy of the seed, never on the seed.

    The fallback is the pipeline's own structure, shared by reference; written in place,
    the input geometry every later reader sees — including the `parents_digest` behind
    downstream cache keys — would silently become the output geometry. The `.copy()` is
    the whole protection (`Structure.atoms` cannot be write-locked the way the force
    arrays are), so this pins it.
    """
    seed = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    before = seed.get_positions().copy()
    moved = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
    updated = _atoms_from_output(
        {"energy_hartree": -1.0, "positions_angstrom": moved}, fallback=seed
    )
    assert np.array_equal(seed.get_positions(), before)
    assert np.array_equal(updated.get_positions(), np.asarray(moved))


def test_a_ragged_gradient_is_refused():
    """The other half of the same shape contract — `np.asarray` would raise bare, too."""
    with pytest.raises(OutputParseError, match="gradient_hartree_per_bohr"):
        _forces_from_gradient([[0.1, 0.2, 0.3], [0.1, 0.2]])


@pytest.mark.parametrize(
    "body",
    [
        '{"energy_hartree": -1.0, "positions_angstrom": [0.0, 0.0, 0.0, 0.74, 0.0, 0.0]}',
        '{"energy_hartree": -1.0, "gradient_hartree_per_bohr": [[0.1, 0.2, 0.3], [0.1, 0.2]]}',
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
