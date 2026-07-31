"""Time the per-step bookkeeping from hundreds to tens of thousands of structures.

Three properties of the design invite the question "does this scale": the step cache is
rewritten in full on every save, :func:`~chemrefine.cache.parents_digest` re-hashes every
parent's coordinates once per step, and the per-step CSV round-trips through pandas.
Whether any is a *problem* is a question about numbers, so this produces them.

So this measures rather than asserts. It prints a table and checks only that the work
stays roughly linear in the structure count — the shape that would make a 10⁴-structure
step viable — instead of pinning wall-clock thresholds, which would fail on a loaded
machine and tell you nothing.

Two things it reports beyond wall time:

* **Residency** — how much memory the live :class:`~chemrefine.state.PipelineState` holds.
  The pipeline keeps every structure's :class:`ase.Atoms` in memory for the whole run, so
  this, not disk, is what sets the ceiling. Most of the overhead is ASE's own object graph
  rather than the coordinates, which is why it is far above the array bytes and why
  reducing it would mean giving up ASE as the interop contract.
* **The format comparison** — the same structures serialized as one JSON document versus
  the shipped JSON-plus-``.npz`` split, so the reason for the split has a number behind it
  and `docs/concepts/caching.md` has somewhere to get one.

Run it with ``-s`` to see the table:

    pytest tests/test_perf_cache.py -m integration -s

Marked ``integration``: it builds 10⁴ structures and takes a few seconds, which does not
belong in the default loop.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine import cache, io
from chemrefine.config import StepConfig
from chemrefine.state import PipelineState, StepContext, StepResults, Structure

pytestmark = pytest.mark.integration

#: Structure counts to time. 200 is a normal conformer screen; 10 000 is well past
#: anything the tutorials do — the size at which these questions get interesting.
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


def _residency_mb(n: int) -> float:
    """Megabytes a live ``PipelineState`` of ``n`` structures retains.

    Measured by building it inside a :mod:`tracemalloc` window rather than by summing
    ``nbytes``: the coordinates are a minority of the cost, and what matters is the whole
    retained graph — ``Atoms``, its per-array wrappers, the symbol list, the ``Structure``.
    """
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    state = PipelineState(structures=_structures(n))
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    assert len(state.structures) == n  # keep it alive across the measurement
    return (after - before) / 1e6


def _ctx(step_dir: Path, structures: tuple[Structure, ...]) -> StepContext:
    return StepContext(
        step_cfg=StepConfig(step=1, engine="fake", operation="opt_sp"),
        step_dir=step_dir,
        template_dir=step_dir / "templates",
        template=None,
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
    rows: list[tuple[int, float, float, float, float, float, float]] = []

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
        size_mb = sum(p.stat().st_size for p in (step_dir / "_cache").iterdir()) / 1e6
        rows.append(
            (n, digest.seconds, save.seconds, load.seconds, csv.seconds, size_mb, _residency_mb(n))
        )

    with capsys.disabled():
        print(f"\n  {N_ATOMS} atoms per structure\n")
        print(
            f"  {'n':>7}  {'digest':>8}  {'save':>8}  {'load':>8}  {'csv':>8}"
            f"  {'_cache':>9}  {'residency':>10}"
        )
        for n, d, s, ld, c, mb, res in rows:
            print(
                f"  {n:>7}  {d:>7.3f}s  {s:>7.3f}s  {ld:>7.3f}s  {c:>7.3f}s"
                f"  {mb:>8.1f}MB  {res:>9.1f}MB"
            )
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


#: Atom counts for the ragged fixture — real steps mix molecules, because ``input:`` can
#: name a directory of ``.xyz`` files or a SMILES CSV. A format that only handles a uniform
#: atom count would not survive contact with either.
RAGGED_ATOM_RANGE = (10, 121)

#: Fraction of structures whose engine reported no forces (a plain single-point does not).
NO_FORCES_FRACTION = 0.30


def _ragged_structures(n: int) -> tuple[Structure, ...]:
    """``n`` structures of differing sizes, some without forces."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        count = int(rng.integers(*RAGGED_ATOM_RANGE))
        out.append(
            Structure(
                id=str(i),
                atoms=Atoms("C" * count, positions=rng.random((count, 3)) * 10.0),
                energy_hartree=-100.0 - i * 1e-4,
                forces_ev_per_a=(
                    None if rng.random() < NO_FORCES_FRACTION else rng.random((count, 3))
                ),
            )
        )
    return tuple(out)


def test_splitting_the_arrays_out_of_json_still_pays(tmp_path: Path, capsys) -> None:
    """Compare the shipped split against one JSON document, on the same structures.

    The split exists because coordinates are 93% of a record and JSON charges three times
    for each float64 — 18 bytes of decimal text, a ``strtod`` to parse it, 32 bytes to hold
    the resulting object — where a ``.npy`` member charges 8 bytes, a memcpy and 8 bytes.
    This keeps that claim measured. It asserts only the *direction*, since the margin is
    large and the exact numbers are the machine's, not the code's.
    """
    n = SIZES[-1]
    structures = _ragged_structures(n)
    step_dir = tmp_path / "split"
    step_dir.mkdir()

    step_cfg = StepConfig(step=1, engine="fake", operation="opt_sp")
    cache.save(
        step_cfg=step_cfg,
        key=cache.StepKey.of(step_cfg, structures[:1], None),
        results=StepResults(structures=structures),
        step_dir=step_dir,
        chemrefine_version="perf",
    )
    split_bytes = sum(p.stat().st_size for p in (step_dir / "_cache").iterdir())
    with _timed() as split_load:
        cache.load(step_dir)

    # The same records as one document, which is what this used to be.
    whole = json.dumps(
        {"structures": [cache.structure_record(s) for s in structures]}, separators=(",", ":")
    ).encode()
    with _timed() as whole_load:
        json.loads(whole)

    with capsys.disabled():
        print(f"\n  {n} structures, {RAGGED_ATOM_RANGE[0]}-{RAGGED_ATOM_RANGE[1] - 1} atoms each\n")
        print(f"  {'format':>18}  {'on disk':>9}  {'read':>8}")
        for label, size, seconds in (
            ("one JSON document", len(whole), whole_load.seconds),
            ("JSON + npz", split_bytes, split_load.seconds),
        ):
            print(f"  {label:>18}  {size / 1e6:>8.1f}MB  {seconds:>7.2f}s")
        print()

    assert split_bytes < len(whole), "the split must not cost disk space"
    # `cache.load` also rebuilds every `Atoms`, which `json.loads` alone does not — so beating
    # a bare parse is a conservative comparison, not a flattering one.
    assert split_load.seconds < whole_load.seconds, "the split must not cost read time"
