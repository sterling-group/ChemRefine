"""Time the per-step bookkeeping from hundreds to tens of thousands of structures.

The audit flagged three structural limits without measuring any of them: the step cache
is one JSON document rewritten in full on every save, :func:`~chemrefine.cache.parents_digest`
re-hashes every parent's coordinates once per step, and the per-step CSV round-trips
through pandas. Each is a real property of the design; whether any is a *problem* depends
on numbers nobody had.

So this measures rather than asserts. It prints a table and checks only that the work
stays roughly linear in the structure count — the shape that would make a 10⁴-structure
step viable — instead of pinning wall-clock thresholds, which would fail on a loaded
machine and tell you nothing.

Run it with ``-s`` to see the table:

    pytest tests/test_perf_cache.py -m integration -s

Marked ``integration``: it builds 10⁴ structures and takes a few seconds, which does not
belong in the default loop.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine import cache, io
from chemrefine.config import StepConfig
from chemrefine.state import PipelineState, StepContext, StepResults, Structure

pytestmark = pytest.mark.integration

#: Structure counts to time. 200 is a normal conformer screen; 10 000 is well past
#: anything the tutorials do, and is the size the audit worried about.
SIZES = (200, 2_000, 10_000)

#: Atoms per structure — a mid-sized organic molecule, so the coordinate arrays that
#: dominate `parents_digest` and the cache document are realistic.
N_ATOMS = 30


def _structures(n: int) -> tuple[Structure, ...]:
    """``n`` structures with distinct geometries, energies and forces."""
    rng = np.random.default_rng(0)
    symbols = "C" * N_ATOMS
    return tuple(
        Structure(
            id=str(i),
            atoms=Atoms(symbols, positions=rng.random((N_ATOMS, 3)) * 10.0),
            energy_hartree=-100.0 - i * 1e-4,
            forces_ev_per_a=rng.random((N_ATOMS, 3)),
        )
        for i in range(n)
    )


def _ctx(step_dir: Path, structures: tuple[Structure, ...]) -> StepContext:
    return StepContext(
        step_cfg=StepConfig(step=1, engine="fake", operation="opt_sp"),
        step_dir=step_dir,
        template_dir=step_dir / "templates",
        scratch_dir=None,
        prev_state=PipelineState(structures=structures),
        charge=0,
        multiplicity=1,
        max_cores=1,
        slurm_template="cpu.slurm.header",
    )


class _timed:
    """Context manager recording the wall time of the block, in ``.seconds``."""

    def __enter__(self) -> _timed:
        self.seconds = 0.0
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.seconds = time.perf_counter() - self._start


def test_per_step_bookkeeping_scales_linearly(tmp_path: Path, capsys) -> None:
    """Time save / load / digest / CSV across the sizes and report the shape."""
    rows: list[tuple[int, float, float, float, float, float]] = []

    for n in SIZES:
        structures = _structures(n)
        results = StepResults(structures=structures)
        step_dir = tmp_path / f"step_{n}"
        step_dir.mkdir(parents=True)
        ctx = _ctx(step_dir, structures)

        with _timed() as digest:
            cache.parents_digest(structures)
        with _timed() as save:
            cache.save_step_results(
                step_cfg=ctx.step_cfg,
                parent_ids=tuple(s.id for s in structures),
                results=results,
                ctx=ctx,
                template_digest="",
                chemrefine_version="perf",
            )
        with _timed() as load:
            cache.load(step_dir)
        with _timed() as csv:
            io.save_step_csv(
                energies_hartree=[s.energy_hartree for s in structures],
                structure_ids=[s.id for s in structures],
                step_number=1,
                output_dir=step_dir,
            )
        size_mb = (step_dir / "_cache" / "step.json").stat().st_size / 1e6
        rows.append((n, digest.seconds, save.seconds, load.seconds, csv.seconds, size_mb))

    with capsys.disabled():
        print(f"\n  {N_ATOMS} atoms per structure\n")
        print(f"  {'n':>7}  {'digest':>8}  {'save':>8}  {'load':>8}  {'csv':>8}  {'step.json':>10}")
        for n, d, s, ld, c, mb in rows:
            print(f"  {n:>7}  {d:>7.3f}s  {s:>7.3f}s  {ld:>7.3f}s  {c:>7.3f}s  {mb:>9.1f}MB")
        print()

    # Linear-ish is the property that matters: a 50x jump in structures should not cost
    # dramatically more than 50x the time. The generous factor absorbs a loaded machine
    # and constant overheads at the small end — this is a shape check, not a stopwatch.
    small, large = rows[0], rows[-1]
    ratio_n = large[0] / small[0]
    for idx, label in ((1, "parents_digest"), (2, "cache.save"), (3, "cache.load")):
        if small[idx] < 1e-4:  # too fast to time meaningfully at the small end
            continue
        growth = large[idx] / small[idx]
        assert growth < ratio_n * 4, (
            f"{label} grew {growth:.1f}x for a {ratio_n:.0f}x increase in structures — "
            f"that is superlinear and worth fixing"
        )
