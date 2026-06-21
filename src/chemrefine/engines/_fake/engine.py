"""In-memory engine that drives the pipeline tests without any external tool.

Real engines invoke SLURM and a quantum-chemistry binary. The fake engine
short-circuits both: ``submit`` writes a synthetic output file with a
deterministic energy derived from the structure ID, and ``wait`` is a
no-op. ``parse`` reads the same file the engine just wrote, so it still
exercises the full prepare → submit → wait → parse → filter → cache
pipeline through realistic file I/O.

The energy formula is monotonic in the integer part of the structure ID
so tests can assert ordering without depending on Python's hash seed.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from ase import Atoms

from chemrefine.engines.base import register
from chemrefine.ids import structure_artifact_path
from chemrefine.state import (
    JobBatch,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)


def _fake_energy(structure_id: str) -> float:
    """Return a deterministic Hartree-scale energy for an ID.

    Pure numeric IDs map to ``-1.0 - n * 1e-4``; hierarchical IDs
    accumulate every digit they contain. The result is reproducible
    across Python runs (no :func:`hash`).
    """
    digits = [int(c) for c in structure_id if c.isdigit()]
    base = sum(digits) if digits else 0
    return -1.0 - base * 1e-4 - len(structure_id) * 1e-6


@register("fake")
class FakeEngine:
    """Test stub satisfying :class:`~chemrefine.engines.base.CalculationEngine`."""

    name: ClassVar[str] = "fake"
    supports_nms: ClassVar[bool] = False

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write one trivial ``.inp`` per seed structure."""
        ctx.step_dir.mkdir(parents=True, exist_ok=True)
        files: list[tuple] = []
        step = ctx.step_cfg.step
        for struct in ctx.prev_state.structures:
            inp = structure_artifact_path(ctx.step_dir, step, struct.id, "inp")
            out = structure_artifact_path(ctx.step_dir, step, struct.id, "out")
            inp.parent.mkdir(parents=True, exist_ok=True)  # per-structure dir
            inp.write_text(f"# fake input for {struct.id}\n", encoding="utf-8")
            files.append((inp, out, struct.id))
        return StepInputs(files=tuple(files))

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Run the fake "calculation" inline — write each output file now."""
        jobs: dict = {}
        for index, (inp, out, sid) in enumerate(inputs.files):
            energy = _fake_energy(sid)
            out.write_text(
                f"# fake output for {sid}\nFINAL ENERGY: {energy}\n",
                encoding="utf-8",
            )
            jobs[inp] = f"fake-{ctx.step_cfg.step}-{index}"
        return JobBatch(jobs=jobs)

    def wait(self, batch: JobBatch) -> None:
        """No-op — submit ran inline."""
        return None

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Read each output file's energy and return reconstructed structures."""
        seeds = {s.id: s for s in ctx.prev_state.structures}
        out_structures: list[Structure] = []
        for _inp, out, sid in inputs.files:
            text = out.read_text()
            energy = float(text.split("FINAL ENERGY:")[1].strip())
            seed = seeds.get(sid)
            atoms = seed.atoms if seed is not None else Atoms("H")
            out_structures.append(
                Structure(
                    id=sid,
                    atoms=atoms,
                    parent_id=seed.parent_id if seed is not None else None,
                    energy_hartree=energy,
                    forces_ev_per_a=np.zeros((len(atoms), 3)),
                )
            )
        return StepResults(structures=tuple(out_structures))

    def input_digest(self, ctx: StepContext) -> str:
        """No template to hash — the fake engine fabricates its results."""
        return ""
