"""Shared base for direct template-driven engines (``pyscf`` and ``mlip``).

Both engines render a user-supplied ``step{N}.py`` template per
structure, submit each rendered script through the SLURM-or-local
machinery, and parse the JSON the appended footer writes. Only the
human-readable backend label differs between the two — every line of
logic below is identical.

Subclasses set two ClassVars and inherit the rest:

* ``name`` — registry / YAML tag (``"pyscf"``, ``"mlip"``). Becomes the
  ``@register`` argument and the trailing word in the
  ``normal_mode_sample`` ``NotImplementedError``.
* ``label`` — human backend label used in every parse / template
  error message (``"PySCF"``, ``"MLIP"``).

The helper functions :func:`_atoms_from_output` and
:func:`_forces_from_gradient` are top-level (not methods) because
neither path needs the backend label — the strings they raise are
generic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
from ase import Atoms

from chemrefine.engines import _template_render
from chemrefine.engines.base import SlurmBatchEngine
from chemrefine.errors import OutputParseError
from chemrefine.ids import structure_artifact_path
from chemrefine.io import write_xyz
from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A
from chemrefine.state import StepContext, StepInputs, StepResults, Structure


class TemplateScriptEngine(SlurmBatchEngine):
    """Direct template-driven engine satisfying :class:`CalculationEngine`."""

    name: ClassVar[str]
    label: ClassVar[str]
    supports_nms: ClassVar[bool] = False
    template_suffix: ClassVar[str] = "py"
    output_globs: ClassVar[tuple[str, ...]] = ("*.json", "*.xyz")

    # -- prepare -----------------------------------------------------------

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Render one ``.py`` + ``.xyz`` per seed structure."""
        ctx.step_dir.mkdir(parents=True, exist_ok=True)
        template = self._resolve_template(ctx)
        step = ctx.step_cfg.step
        extra_vars = self._template_vars(ctx)

        files: list[tuple[Path, Path, str]] = []
        for struct in ctx.prev_state.structures:
            xyz_paths = write_xyz(
                [struct.atoms],
                [struct.id],
                step_number=step,
                output_dir=ctx.step_dir,
            )
            xyz_path = xyz_paths[0]
            script_path = structure_artifact_path(ctx.step_dir, step, struct.id, "py")
            output_json = structure_artifact_path(ctx.step_dir, step, struct.id, "json")
            _template_render.build_input(
                xyz_path=xyz_path,
                template_path=template,
                output_path=script_path,
                output_json_path=output_json,
                charge=ctx.charge,
                multiplicity=ctx.multiplicity,
                extra_vars=extra_vars,
            )
            files.append((script_path, output_json, struct.id))
        return StepInputs(files=tuple(files))

    def _template_vars(self, ctx: StepContext) -> dict[str, object]:
        """Extra ``$VAR`` substitutions for the template (default: none).

        Subclasses override this to let the YAML ``step.options`` drive the
        rendered script — e.g. the MLIP engine injects ``$MODEL_NAME`` /
        ``$TASK_NAME`` / ``$DEVICE`` so a template need not hardcode the model.
        """
        return {}

    # -- submit (PAL + run_block hooks; the loop lives on SlurmBatchEngine) -

    def _pal(self, ctx: StepContext) -> int:
        """Direct scripts take their core count from ``options.cores`` (default 1)."""
        return int((ctx.step_cfg.options or {}).get("cores", 1))

    def _run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """Run the rendered Python script inside ``$WORK_DIR``."""
        return f"python {inp_path.name}"

    # -- parse -------------------------------------------------------------

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Read each output JSON into a :class:`Structure`."""
        prev_by_id = {s.id: s for s in ctx.prev_state.structures}
        out_structures: list[Structure] = []
        for _inp, out_path, sid in inputs.files:
            data = self._load_output_json(out_path)
            seed = prev_by_id.get(sid)
            atoms = _atoms_from_output(data, fallback=seed.atoms if seed else None)
            forces = _forces_from_gradient(data.get("gradient_hartree_per_bohr"))
            out_structures.append(
                Structure(
                    id=sid,
                    atoms=atoms,
                    parent_id=seed.parent_id if seed is not None else None,
                    energy_hartree=float(data["energy_hartree"]),
                    forces_ev_per_a=forces,
                )
            )
        return StepResults(structures=tuple(out_structures))

    def _load_output_json(self, out_path: Path) -> dict:
        """Read the user's script output JSON; raise :class:`OutputParseError` if malformed."""
        if not out_path.is_file():
            raise OutputParseError(f"{self.label} output not found: {out_path}")
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise OutputParseError(
                f"{self.label} output {out_path} is not valid JSON: {e}"
            ) from e
        if "energy_hartree" not in data:
            raise OutputParseError(
                f"{self.label} output {out_path} missing required 'energy_hartree' field"
            )
        return data

    def normal_mode_sample(self, results: StepResults, ctx: StepContext) -> StepResults:
        """Not supported — the orchestrator gates this on ``supports_nms``."""
        raise NotImplementedError(f"{self.name} does not support normal-mode sampling")


# ---------------------------------------------------------------------------
# Output parsing helpers (top-level for direct testability — neither needs
# the backend label)
# ---------------------------------------------------------------------------


def _atoms_from_output(data: dict, *, fallback: Atoms | None) -> Atoms:
    """Return ASE ``Atoms`` from the output JSON, falling back to the seed geometry.

    If the user's script wrote a ``positions_angstrom`` block (an
    optimised geometry, say), the returned ``Atoms`` carries those
    positions and the seed's symbols. Otherwise the seed atoms come
    back unchanged.
    """
    positions = data.get("positions_angstrom")
    if positions is None or fallback is None:
        if fallback is None:
            raise OutputParseError(
                "output lacks positions_angstrom and no seed atoms are available"
            )
        return fallback.copy()
    updated = fallback.copy()
    updated.set_positions(np.asarray(positions, dtype=float))
    return updated


def _forces_from_gradient(
    gradient: list[list[float]] | None,
) -> np.ndarray | None:
    """Convert template gradient (Hartree/Bohr) to ASE forces (eV/Å)."""
    if not gradient:
        return None
    return np.asarray(gradient, dtype=float) * (-HARTREE_PER_BOHR_TO_EV_PER_A)
