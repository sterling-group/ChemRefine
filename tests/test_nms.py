"""Tests for the engine-independent NMS coordinator (``chemrefine.nms``).

Everything here is driven by a **fake** ``NmsCapableEngine`` that implements only the
``nms_input_info`` hook + the standard lifecycle (attaching ``imaginary_freqs`` /
``normal_modes`` to each parsed structure, as the real engine does in its single parse) —
proving NMS is engine-agnostic. Covers the pure displacement maths, the reuse
fingerprint, and the unified "attempt" model (winner at the canonical id, no duplicate
minima, ``random`` fan-out, round-1/round-2 auto-retry, rebuild + reattempt).
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine import cache, nms
from chemrefine.config import Config, MinSample, StepConfig
from chemrefine.engines.api import NmsInputInfo
from chemrefine.ids import structure_artifact_path
from chemrefine.state import (
    Failure,
    FailureKind,
    JobBatch,
    PipelineState,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)

# Frequency data the fake engine attaches to each parsed structure (imaginary modes + the
# normal-mode tensor) — the real engine sets ``Structure.imaginary_freqs`` / ``normal_modes``
# in its single parse pass; NMS reads them off the structure (no ``read_frequencies`` hook).
_Freq = namedtuple("_Freq", ["imaginary", "modes"])


def _modes(n_modes: int = 6) -> np.ndarray:
    """(2 atoms, 3, n_modes): mode k moves atom0 +0.1(k+1) and atom1 -0.1(k+1) in x."""
    t = np.zeros((2, 3, n_modes))
    for k in range(n_modes):
        t[0, 0, k] = 0.1 * (k + 1)
        t[1, 0, k] = -0.1 * (k + 1)
    return t


def _h2(sid: str, *, energy: float | None = -1.0) -> Structure:
    return Structure(
        id=sid, atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]), energy_hartree=energy
    )


# ---------------------------------------------------------------------------
# Pure displacement maths + options (engine-independent)
# ---------------------------------------------------------------------------


def test_nms_options_default_and_from_raw():
    assert nms.NmsOptions().target == "minimum"
    o = nms.NmsOptions.from_raw({"target": "ts", "ts_mode_index": 7, "basis": "ignored"})
    assert o.target == "ts" and o.ts_mode_index == 7


def test_nms_options_rejects_bad_target():
    with pytest.raises(ValueError):
        nms.NmsOptions(target="saddle")


def test_target_imaginary_count():
    assert nms.target_imaginary_count(nms.NmsOptions(target="minimum")) == 0
    assert nms.target_imaginary_count(nms.NmsOptions(target="ts")) == 1
    assert nms.target_imaginary_count(nms.NmsOptions(target="random")) is None


def test_displace_along_mode_shapes_and_values():
    pos = np.zeros((2, 3))
    mode = np.ones((2, 3))
    plus, minus = nms.displace_along_mode(pos, mode, displacement=2.0)
    assert np.allclose(plus, 2.0) and np.allclose(minus, -2.0)


def test_displace_along_mode_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        nms.displace_along_mode(np.zeros((2, 3)), np.zeros((3, 3)), displacement=1.0)


def test_select_displacements_minimum_displaces_every_imaginary():
    out = nms.select_displacements(
        _h2("0"), {5: -42.0}, _modes(6), nms.NmsOptions(target="minimum"), np.random.default_rng(0)
    )
    assert [s for s, _ in out] == ["m5_pos", "m5_neg"]


def test_select_displacements_ts_keeps_reaction_coordinate():
    # two imaginary modes; ts keeps the largest-magnitude (mode 5) and displaces mode 4.
    out = nms.select_displacements(
        _h2("0"),
        {4: -10.0, 5: -99.0},
        _modes(6),
        nms.NmsOptions(target="ts"),
        np.random.default_rng(0),
    )
    assert [s for s, _ in out] == ["m4_pos", "m4_neg"]


def test_select_displacements_random_draws_from_modes():
    out = nms.select_displacements(
        _h2("0"),
        {},
        _modes(8),
        nms.NmsOptions(target="random", num_random_displacements=1),
        np.random.default_rng(1),
    )
    assert len(out) == 2  # one ± pair


def test_select_displacements_skips_out_of_range_mode():
    # imaginary mode index beyond the tensor's mode count is skipped, not crashed.
    out = nms.select_displacements(
        _h2("0"),
        {37: -118.0},
        _modes(6),
        nms.NmsOptions(target="minimum"),
        np.random.default_rng(0),
    )
    assert out == []


def test_select_displacements_ts_mode_index_overrides():
    out = nms.select_displacements(
        _h2("0"),
        {4: -10.0, 5: -99.0},
        _modes(6),
        nms.NmsOptions(target="ts", ts_mode_index=4),  # keep mode 4 explicitly
        np.random.default_rng(0),
    )
    assert [s for s, _ in out] == ["m5_pos", "m5_neg"]  # displaces the *other* imaginary


def test_select_displacements_minimum_no_imaginary_is_empty():
    out = nms.select_displacements(
        _h2("0"), {}, _modes(6), nms.NmsOptions(target="minimum"), np.random.default_rng(0)
    )
    assert out == []


def test_select_displacements_random_no_candidates_is_empty():
    out = nms.select_displacements(
        _h2("0"), {}, np.zeros((1, 3, 0)), nms.NmsOptions(target="random"), np.random.default_rng(0)
    )
    assert out == []


def test_select_displacements_skips_mode_shape_mismatch():
    # a 3-atom mode tensor against a 2-atom structure → per-mode shape mismatch, skipped.
    out = nms.select_displacements(
        _h2("0"),
        {0: -1.0},
        np.zeros((3, 3, 6)),
        nms.NmsOptions(target="minimum"),
        np.random.default_rng(0),
    )
    assert out == []


def test_best_returns_fallback_when_empty():
    fallback = _h2("fb")
    assert nms._best([], fallback, "energy_hartree") is fallback


def test_best_ranks_by_the_energy_the_step_filters_on():
    """A step sampling on Gibbs must promote the child its own filter would keep.

    The two children disagree: ``lo_elec`` wins on electronic energy, ``lo_gibbs`` on Gibbs.
    Ranking a ``gibbs`` step on electronic energy carried forward a structure the very next
    filter would have discarded — silently, because both children genuinely reached the target.
    """
    lo_elec = replace(_h2("lo_elec"), energy_hartree=-1.0, gibbs_hartree=-0.5)
    lo_gibbs = replace(_h2("lo_gibbs"), energy_hartree=-0.9, gibbs_hartree=-0.7)
    children = [lo_elec, lo_gibbs]

    assert nms._best(children, lo_elec, "energy_hartree") is lo_elec
    assert nms._best(children, lo_elec, "gibbs_hartree") is lo_gibbs


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        (None, "energy_hartree"),
        (MinSample(method="min", count=1), "energy_hartree"),
        (MinSample(method="min", count=1, energy_type="gibbs"), "gibbs_hartree"),
        (MinSample(method="min", count=1, energy_type="enthalpy"), "enthalpy_hartree"),
    ],
)
def test_energy_attr_follows_the_steps_sample_filter(sample, expected):
    """No ``sample`` means no declared preference — electronic, as the step CSV also reports."""
    step_cfg = StepConfig(step=1, engine="orca", operation="sp", sample=sample)
    assert nms._energy_attr(step_cfg) == expected


def test_is_resolved_false_for_non_terminated_child():
    child = Structure(id="c", atoms=Atoms("H"), terminated_normally=False)
    assert nms._is_resolved(child, 0) is False


# ---------------------------------------------------------------------------
# Reuse fingerprint
# ---------------------------------------------------------------------------


def _cfg(**over) -> StepConfig:
    base = {"step": 1, "engine": "orca", "operation": "freq", "nms": True}
    base.update(over)
    return StepConfig(**base)


def test_nms_reuse_fingerprint_ignores_search_params():
    base = _cfg(options={"target": "minimum", "displacement_value": 1.0})
    tuned = _cfg(options={"target": "minimum", "displacement_value": 2.0})
    assert cache.reuse_fingerprint(base, ("0",)) == cache.reuse_fingerprint(tuned, ("0",))


def test_nms_reuse_fingerprint_changes_on_criterion():
    mn = _cfg(options={"target": "minimum"})
    ts = _cfg(options={"target": "ts"})
    assert cache.reuse_fingerprint(mn, ("0",)) != cache.reuse_fingerprint(ts, ("0",))


def test_nms_reuse_fingerprint_empty_for_non_nms():
    assert cache.reuse_fingerprint(_cfg(nms=False), ("0",)) == ""


# ---------------------------------------------------------------------------
# Fake NmsCapableEngine + coordinator tests
# ---------------------------------------------------------------------------


class _FakeNms:
    """A minimal NMS-capable engine: lifecycle via files + the two NMS hooks.

    ``freqs`` maps a structure id → ``_Freq`` (imaginary, modes); ``converge_on_retry`` ids
    parse unconverged on the first pass and converged on the second (drives the retry);
    ``fail`` ids never converge.
    """

    name = "fake-nms"

    def __init__(
        self,
        *,
        freqs: dict[str, _Freq],
        is_ts: bool = False,
        computes_freq: bool = True,
        converge_on_retry: set[str] | None = None,
        fail: set[str] | None = None,
        fail_children: bool = False,
    ) -> None:
        self.freqs = freqs
        self.is_ts = is_ts
        self.computes_freq = computes_freq
        self.converge_on_retry = converge_on_retry or set()
        self.fail = fail or set()
        self.fail_children = fail_children  # every displaced child fails to converge
        self._parses: dict[str, int] = {}

    def prepare(self, ctx: StepContext) -> StepInputs:
        files = []
        for s in ctx.prev_state.structures:
            inp = structure_artifact_path(ctx.step_dir, ctx.step_cfg.step, s.id, "inp")
            out = structure_artifact_path(ctx.step_dir, ctx.step_cfg.step, s.id, "out")
            inp.parent.mkdir(parents=True, exist_ok=True)
            inp.write_text("in\n", encoding="utf-8")
            out.write_text("out\n", encoding="utf-8")
            files.append((inp, out, s.id))
        return StepInputs(files=tuple(files))

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        return JobBatch(jobs={})

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        seeds = {s.id: s for s in ctx.prev_state.structures}
        out = []
        for _inp, _o, sid in inputs.files:
            self._parses[sid] = self._parses.get(sid, 0) + 1
            if sid in self.fail or (self.fail_children and "_m" in sid):
                converged = False
            elif sid in self.converge_on_retry:
                converged = self._parses[sid] >= 2
            else:
                converged = True
            seed = seeds[sid]
            # The frequency values ride on the parsed structure (parsed in one pass with
            # geometry/energy in the real engine); NMS reads them off the structure.
            freq = self.freqs.get(sid, _Freq(imaginary=None, modes=None))
            out.append(
                replace(
                    seed,
                    id=sid,
                    converged=converged,
                    terminated_normally=True,
                    energy_hartree=-1.0,
                    imaginary_freqs=freq.imaginary,
                    normal_modes=freq.modes,
                )
            )
        return StepResults(structures=tuple(out))

    def input_digest(self, ctx: StepContext) -> str:
        return ""

    def nms_input_info(self, ctx: StepContext) -> NmsInputInfo:
        return NmsInputInfo(is_transition_state=self.is_ts, computes_frequencies=self.computes_freq)

    def artifact_paths(self, ctx: StepContext, structure_id: str) -> tuple[Path, Path]:
        return (
            structure_artifact_path(ctx.step_dir, ctx.step_cfg.step, structure_id, "inp"),
            structure_artifact_path(ctx.step_dir, ctx.step_cfg.step, structure_id, "out"),
        )


def _ctx(tmp_path: Path, structures: tuple[Structure, ...], **over) -> StepContext:
    cfg = Config(
        template_dir=tmp_path / "t",
        scratch_dir=tmp_path / "s",
        output_dir=tmp_path / "o",
        steps=[_cfg(**over)],
    )
    return StepContext(
        step_cfg=cfg.steps[0],
        step_dir=(tmp_path / "o" / "step1").resolve(),
        template_dir=cfg.template_dir,
        scratch_dir=cfg.scratch_dir,
        prev_state=PipelineState(structures=structures),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        executables={},
    )


def _seed_round1(engine: _FakeNms, ctx: StepContext) -> StepResults:
    """Run the fake's round-1 prepare so its outputs exist, return the parsed survivors."""
    inputs = engine.prepare(ctx)
    return engine.parse(inputs, ctx)


def test_run_nms_winner_at_canonical_id_stays(tmp_path: Path):
    """minimum: the best resolved child becomes the survivor *at the parent's id*, its
    geometry written to the canonical dir; the exploration is archived under attempt1/."""
    engine = _FakeNms(
        freqs={
            "0": _Freq(imaginary={5: -42.0}, modes=_modes(6)),  # round-1: one imaginary
            "0_m5_pos": _Freq(imaginary={}, modes=None),  # children resolve (0 imaginary)
            "0_m5_neg": _Freq(imaginary={}, modes=None),
        }
    )
    ctx = _ctx(tmp_path, (_h2("0"),))
    round1 = _seed_round1(engine, ctx)
    res = nms.run_nms(engine, round1, [], ctx)
    assert [s.id for s in res.survivors] == ["0"]  # stable id, ONE survivor (no ± duplicate)
    assert res.failures == ()
    assert (ctx.step_dir / "0" / "step1_0.xyz").is_file()  # winner geometry at canonical
    assert (ctx.step_dir / "0" / "attempt1").is_dir()  # exploration archived
    # The parent keeps its own id, so this is the only record of which child's calculation
    # the promoted artifacts came from.
    assert res.survivors[0].resolved_from == "0_m5_pos"


def test_run_nms_already_at_target_passes_through(tmp_path: Path):
    engine = _FakeNms(freqs={"0": _Freq(imaginary={}, modes=None)})  # already a minimum
    ctx = _ctx(tmp_path, (_h2("0"),))
    round1 = _seed_round1(engine, ctx)
    res = nms.run_nms(engine, round1, [], ctx)
    assert [s.id for s in res.survivors] == ["0"]
    assert res.survivors[0].converged is True
    assert not (ctx.step_dir / "0" / "attempt1").exists()  # no displacement needed
    assert res.survivors[0].resolved_from is None  # nothing was promoted


def test_run_nms_random_fanout_records_no_resolution(tmp_path: Path):
    """``random`` keeps the children as their own structures, so none is a promoted winner."""
    engine = _FakeNms(freqs={"0": _Freq(imaginary={}, modes=_modes(8))})
    ctx = _ctx(tmp_path, (_h2("0"),), options={"target": "random", "seed": 1})
    round1 = _seed_round1(engine, ctx)
    res = nms.run_nms(engine, round1, [], ctx)
    assert all(s.resolved_from is None for s in res.survivors)


def test_run_nms_random_fans_out_to_children(tmp_path: Path):
    engine = _FakeNms(
        freqs={"0": _Freq(imaginary={}, modes=_modes(8))},  # random ignores imaginary
    )
    ctx = _ctx(tmp_path, (_h2("0"),), options={"target": "random", "num_random_displacements": 1})
    round1 = _seed_round1(engine, ctx)
    res = nms.run_nms(engine, round1, [], ctx)
    assert {s.id for s in res.survivors} == {"0_m6_pos", "0_m6_neg"}  # fan-out, child ids
    assert res.failures == ()


def test_run_nms_ts_target_inferred_from_input(tmp_path: Path):
    """No options.target + a TS input → ts target (keep one imaginary, displace the rest)."""
    engine = _FakeNms(
        freqs={
            "0": _Freq(imaginary={4: -10.0, 5: -99.0}, modes=_modes(6)),
            "0_m4_pos": _Freq(imaginary={5: -50.0}, modes=None),  # one imaginary left = ts
            "0_m4_neg": _Freq(imaginary={5: -50.0}, modes=None),
        },
        is_ts=True,
    )
    ctx = _ctx(tmp_path, (_h2("0"),))  # options has no target → inferred
    round1 = _seed_round1(engine, ctx)
    res = nms.run_nms(engine, round1, [], ctx)
    assert [s.id for s in res.survivors] == ["0"]  # resolved to a TS, id kept


def test_run_nms_unresolved_becomes_failure(tmp_path: Path):
    engine = _FakeNms(
        freqs={
            "0": _Freq(imaginary={5: -42.0}, modes=_modes(6)),
            "0_m5_pos": _Freq(imaginary={3: -9.0}, modes=None),  # still imaginary
            "0_m5_neg": _Freq(imaginary={3: -9.0}, modes=None),
        }
    )
    ctx = _ctx(tmp_path, (_h2("0"),))
    round1 = _seed_round1(engine, ctx)
    res = nms.run_nms(engine, round1, [], ctx)
    assert res.survivors == ()
    assert [f.sid for f in res.failures] == ["0"]
    assert res.failures[0].reason == "NMS: target stationary point not reached"


def test_run_nms_no_modes_is_failure(tmp_path: Path):
    engine = _FakeNms(freqs={"0": _Freq(imaginary={5: -42.0}, modes=None)})
    ctx = _ctx(tmp_path, (_h2("0"),))
    round1 = _seed_round1(engine, ctx)
    res = nms.run_nms(engine, round1, [], ctx)
    assert [f.sid for f in res.failures] == ["0"]


def test_run_nms_no_displacements_is_failure(tmp_path: Path):
    """An imaginary mode beyond the tensor → no children → unresolved failure."""
    engine = _FakeNms(freqs={"0": _Freq(imaginary={37: -118.0}, modes=_modes(6))})
    ctx = _ctx(tmp_path, (_h2("0"),))
    round1 = _seed_round1(engine, ctx)
    res = nms.run_nms(engine, round1, [], ctx)
    assert [f.sid for f in res.failures] == ["0"]


def test_run_nms_random_all_children_fail_is_failure(tmp_path: Path):
    """random with every displaced child failing to converge → the parent is a failure."""
    engine = _FakeNms(freqs={"0": _Freq(imaginary={}, modes=_modes(8))}, fail_children=True)
    ctx = _ctx(tmp_path, (_h2("0"),), options={"target": "random", "num_random_displacements": 1})
    round1 = _seed_round1(engine, ctx)
    res = nms.run_nms(engine, round1, [], ctx)
    assert res.survivors == ()
    assert [f.sid for f in res.failures] == ["0"]


def test_run_nms_carries_round1_failures(tmp_path: Path):
    engine = _FakeNms(freqs={"0": _Freq(imaginary={}, modes=None)})
    ctx = _ctx(tmp_path, (_h2("0"),))
    round1 = _seed_round1(engine, ctx)
    prior = [Failure("9", FailureKind.MISSING_OUTPUT, None)]
    res = nms.run_nms(engine, round1, prior, ctx)
    assert "9" in {f.sid for f in res.failures}


def test_run_nms_retries_unconverged_child(tmp_path: Path):
    """An unconverged round-2 child is retried from best (its own attempt dir) and resolves."""
    engine = _FakeNms(
        freqs={
            "0": _Freq(imaginary={5: -42.0}, modes=_modes(6)),
            "0_m5_pos": _Freq(imaginary={}, modes=None),
            "0_m5_neg": _Freq(imaginary={}, modes=None),
        },
        converge_on_retry={"0_m5_pos", "0_m5_neg"},
    )
    ctx = _ctx(tmp_path, (_h2("0"),))
    round1 = _seed_round1(engine, ctx)
    res = nms.run_nms(engine, round1, [], ctx)
    assert [s.id for s in res.survivors] == ["0"]  # resolved after the child retry
    assert (ctx.step_dir / "0" / "attempt1" / "0_m5_pos" / "attempt1").is_dir()


def test_rebuild_nms_reads_existing_children(tmp_path: Path):
    """rebuild reuses the children on disk from a prior run (no new submission)."""
    engine = _FakeNms(
        freqs={
            "0": _Freq(imaginary={5: -42.0}, modes=_modes(6)),
            "0_m5_pos": _Freq(imaginary={}, modes=None),
            "0_m5_neg": _Freq(imaginary={}, modes=None),
        }
    )
    ctx = _ctx(tmp_path, (_h2("0"),))
    round1 = _seed_round1(engine, ctx)
    ran = nms.run_nms(engine, round1, [], ctx)  # populates attempt1/ on disk
    rebuilt = nms.rebuild_nms(engine, round1, [], ctx)

    # Comparing ids would pass while the two disagreed about energies, convergence or which
    # child won — the class of drift that shipped once already. Compare the whole record.
    assert [cache.structure_record(s) for s in rebuilt.survivors] == [
        cache.structure_record(s) for s in ran.survivors
    ]
    assert [(f.sid, f.kind, f.detail) for f in rebuilt.failures] == [
        (f.sid, f.kind, f.detail) for f in ran.failures
    ]


def test_rebuild_reproduces_the_resolution_after_the_winner_was_promoted(tmp_path: Path):
    """A rebuild of a resolved tree must report the same winner the run did.

    Promotion puts the winner's output at the parent's canonical path, so re-parsing round 1
    on a resolved tree yields a structure *already at the target* — the passthrough, which
    never re-derives children and so has no winner to name. The verdict has to come off disk.
    """
    engine = _FakeNms(
        freqs={
            "0": _Freq(imaginary={5: -42.0}, modes=_modes(6)),
            "0_m5_pos": _Freq(imaginary={}, modes=None),
            "0_m5_neg": _Freq(imaginary={}, modes=None),
        }
    )
    ctx = _ctx(tmp_path, (_h2("0"),))
    ran = nms.run_nms(engine, _seed_round1(engine, ctx), [], ctx)
    assert ran.survivors[0].resolved_from == "0_m5_pos"

    # What a re-parse of the promoted canonical output now yields: no imaginary modes.
    promoted = StepResults(structures=(replace(_h2("0"), imaginary_freqs={}),))
    rebuilt = nms.rebuild_nms(engine, promoted, [], ctx)

    assert [s.id for s in rebuilt.survivors] == ["0"]
    assert rebuilt.survivors[0].resolved_from == "0_m5_pos"


def test_rebuild_of_an_unresolved_tree_names_no_winner(tmp_path: Path):
    """No resolution on disk means none is claimed — an absent sidecar reads as ``None``."""
    engine = _FakeNms(freqs={"0": _Freq(imaginary={}, modes=None)})
    ctx = _ctx(tmp_path, (_h2("0"),))
    round1 = _seed_round1(engine, ctx)
    res = nms.rebuild_nms(engine, round1, [], ctx)
    assert res.survivors[0].resolved_from is None


def test_rebuild_nms_unresolved_when_no_attempt_on_disk(tmp_path: Path):
    engine = _FakeNms(freqs={"0": _Freq(imaginary={5: -42.0}, modes=_modes(6))})
    ctx = _ctx(tmp_path, (_h2("0"),))
    round1 = _seed_round1(engine, ctx)  # round-1 only, no attempt dir
    res = nms.rebuild_nms(engine, round1, [], ctx)
    assert [f.sid for f in res.failures] == ["0"]


# ---------------------------------------------------------------------------
# Mode selection must not depend on loop position
# ---------------------------------------------------------------------------


def test_random_mode_selection_is_per_structure_not_stream_order():
    """The same structure draws the same modes regardless of what ran before it.

    `run_nms` and `rebuild_nms` skip on different conditions -- only the rebuild
    skips a structure whose `attemptK/` is absent -- so a shared RNG stream made a
    skip shift every later structure's draw. `rebuild-cache` then looked for children
    that were never computed and reported a resolved structure as unresolved.
    """
    opts = nms.NmsOptions(target="random", num_random_displacements=1, seed=42)
    modes = np.zeros((3, 3, 9))
    modes[0, 0, :] = 1.0

    def drawn(structure_id: str) -> str:
        struct = Structure(id=structure_id, atoms=Atoms("H3", positions=np.zeros((3, 3))))
        rng = nms.rng_for(struct.id, opts.seed)
        return nms.select_displacements(struct, {}, modes, opts, rng)[0][0]

    visited_all = [drawn(sid) for sid in ("0", "1", "2")]
    skipped_one = [drawn(sid) for sid in ("0", "2")]

    assert visited_all[0] == skipped_one[0]
    assert visited_all[2] == skipped_one[1], "a skipped structure shifted a later one's draw"


def test_rng_for_is_deterministic_and_distinct_per_structure():
    """Same (id, seed) -> same stream; different ids -> different streams."""
    assert nms.rng_for("0", 42).integers(0, 1000, 5).tolist() == (
        nms.rng_for("0", 42).integers(0, 1000, 5).tolist()
    )
    assert nms.rng_for("0", 42).integers(0, 1000, 5).tolist() != (
        nms.rng_for("1", 42).integers(0, 1000, 5).tolist()
    )
    # The seed still matters -- it is not being ignored in favour of the id.
    assert nms.rng_for("0", 42).integers(0, 1000, 5).tolist() != (
        nms.rng_for("0", 43).integers(0, 1000, 5).tolist()
    )


def test_rebuilding_does_not_rewrite_the_children_it_reads(tmp_path: Path):
    """``rebuild_nms`` re-reads round-2 outputs to re-derive a decision, and writes nothing.

    It exists to reconstruct a cache from a finished tree. Re-recording each child's parsed
    result while doing so would make a read-only operation modify the very outputs a later
    comparison is meant to trust.
    """
    engine = _FakeNms(
        freqs={
            "0": _Freq(imaginary={5: -42.0}, modes=_modes(6)),
            "0_m5_pos": _Freq(imaginary={}, modes=None),
            "0_m5_neg": _Freq(imaginary={}, modes=None),
        }
    )
    ctx = _ctx(tmp_path, (_h2("0"),))
    round1 = _seed_round1(engine, ctx)
    nms.run_nms(engine, round1, [], ctx)

    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(ctx.step_dir.rglob("*"))
        if p.is_file()
    }
    nms.rebuild_nms(engine, round1, [], ctx)
    after = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(ctx.step_dir.rglob("*"))
        if p.is_file()
    }

    assert after.keys() == before.keys(), "a rebuild must not add or remove files"
    rewritten = [p.name for p in before if after[p] != before[p]]
    assert rewritten == [], f"a rebuild rewrote {rewritten}"


def test_rebuilding_cannot_install_a_winner():
    """The rebuild coordinator does not reach the code that writes — by construction.

    Choosing a winner and installing it on disk are separate functions, so a rebuild is not
    relying on an argument being right: it never calls the installer at all. This asserts the
    shape rather than the behaviour, so the byte-identical snapshot above cannot start passing
    for an accidental reason.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(nms.rebuild_nms))
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_select_survivors" in called, "the rebuild must still choose a winner"
    assert "_install_winner" not in called
