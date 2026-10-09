#!/usr/bin/env python
"""Assert that the suite still catches a deliberate break in each critical predicate.

100% branch coverage proves every line ran; it does not prove an assertion looked at the
result. A predicate can be executed by every test in the suite and checked by none of
them. Breaking it and demanding a red test is what closes that gap.

This is a standing gate, not a survey. The list is small and hand-held on purpose: each
entry is a predicate whose inversion is a *wrong scientific answer* rather than a crash —
a transition state reported for a minimum, a filter keeping less than it promises, a
cache serving results computed from different coordinates — so a survivor always deserves
a person's attention. Whole-package mutation testing is a different tool: thousands of
mutants, most of them log strings, and a survivor list to triage and then maintain. Use
that occasionally to *find* entries for this list; run this one to keep them.

Usage
-----
    python scripts/mutation_gate.py              # all mutations
    python scripts/mutation_gate.py --list       # show them without running
    python scripts/mutation_gate.py -k nms       # only ids containing "nms"

Exit status is 0 only if every mutation was caught.

How it runs
-----------
The tree is copied to a scratch directory and mutated *there*, never in place: a gate that
edits the working tree is one SIGKILL away from leaving a developer with a silently broken
checkout. ``PYTHONPATH`` puts the copy's ``src`` ahead of the editable install's ``.pth``
entry, and :func:`_assert_isolated` proves the copy is what got imported — a harness that
quietly tested the unmutated source would report everything caught, which is worse than
no gate at all.

Each mutation then runs the one test file it names (``Mutation.tests``) before it runs the
whole suite. That is an optimisation with no verdict attached: the baseline proves every
file green *before* any mutation, so a red target afterwards can only be the mutation's
doing — and a target that stays green always falls back to the full suite, so nothing is
called a survivor on a narrow run. Whole-suite-per-mutation cost 7m54s here; naming the
test costs 73s, because ``-x`` otherwise walks every alphabetically-earlier file first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: How long one run may take before it is assumed hung. A mutation that turns a budget
#: comparison into a strictly-smaller one makes a wait loop spin forever rather than fail,
#: so a timeout on the *named test file* counts as caught: that file did not complete,
#: which is a red build. The whole-suite fallback is a different case — it runs only after
#: the named file stayed green, and its timeout is as likely the suite outgrowing this
#: budget on a slow runner as a hang. Reported as caught, it would be a false green from the
#: one gate whose job is to refuse them; it is reported as *inconclusive* instead, and
#: fails the gate (see :class:`Verdict`).
TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class Mutation:
    """One deliberate break, and the wrong answer it would produce if it survived."""

    id: str
    path: str
    old: str
    new: str
    tests: str
    """The test file whose red is the proof — run before the whole suite.

    Named rather than derived from ``path``: the file that checks a predicate is often not
    the one named after the module holding it. ``boltzmann-cutoff`` mutates ``filtering.py``
    and a whole-suite run reports ``test_e2e_replay.py``, which merely sorts earlier under
    ``-x``. A wrong entry cannot hide — a fast path that stays green is re-run against the
    whole suite before anything is called a survivor — so it costs time, never a verdict.
    """
    breaks: str
    """What a user would get if this shipped — the reason the entry is on the list."""


_BUNDLED: tuple[Mutation, ...] = (
    Mutation(
        id="nms-is-resolved",
        path="src/chemrefine/nms.py",
        old="len(child.imaginary_freqs) == target",
        new="len(child.imaginary_freqs) <= target",
        tests="tests/test_nms.py",
        breaks="a `ts` step accepts a child that fell into a minimum and promotes it to "
        "the parent's id — the transition state replaced by the wrong stationary point",
    ),
    Mutation(
        id="nms-already-at-target",
        path="src/chemrefine/nms.py",
        old="len(structure.imaginary_freqs) == target",
        new="len(structure.imaginary_freqs) <= target",
        tests="tests/test_nms.py",
        breaks="a minimum passes through a `ts` step as though already resolved, so it is "
        "never displaced and never reaches the saddle point",
    ),
    Mutation(
        id="nms-no-frequency-evidence",
        path="src/chemrefine/nms.py",
        old="return child.imaginary_freqs is not None and len(child.imaginary_freqs) == target",
        new="return child.imaginary_freqs is None or len(child.imaginary_freqs) == target",
        tests="tests/test_nms.py",
        breaks="a child whose parse found no frequency table is promoted as a verified "
        "stationary point — a minimum claimed with no evidence that it is one",
    ),
    Mutation(
        id="nms-already-at-target-no-evidence",
        path="src/chemrefine/nms.py",
        old="        and structure.imaginary_freqs is not None\n"
        "        and len(structure.imaginary_freqs) == target",
        new="        and (structure.imaginary_freqs is None or "
        "len(structure.imaginary_freqs) == target)",
        tests="tests/test_nms.py",
        breaks="a structure with no frequency table passes through a `minimum` step as though "
        "already resolved, so it is never displaced and never verified",
    ),
    Mutation(
        id="nms-best-child",
        path="src/chemrefine/nms.py",
        old="    return min(\n        structures,\n"
        "        key=lambda s: (getattr(s, energy_attr) is None, getattr(s, energy_attr) or 0.0),",
        new="    return max(\n        structures,\n"
        "        key=lambda s: (getattr(s, energy_attr) is None, getattr(s, energy_attr) or 0.0),",
        tests="tests/test_nms.py",
        breaks="the highest-energy resolved child is promoted to the parent's canonical id — "
        "retry-best-frame's inversion, on the resolution side",
    ),
    Mutation(
        id="nms-requires-frequencies",
        path="src/chemrefine/nms.py",
        old="    if not engine.nms_input_info(ctx).computes_frequencies:",
        new="    if False:",
        tests="tests/test_step.py",
        breaks="an `nms: true` step whose input computes no frequencies runs to the end, and "
        "every structure it paid for is ledgered unresolved",
    ),
    Mutation(
        id="nms-record-before-promote",
        path="src/chemrefine/nms.py",
        old=(
            "    _write_resolution(attempt_dir, source_id, criterion_key)\n"
            "    attempts.seal(ctx.step_dir / parent_id, attempt_dir)\n"
            "    attempts.promote(attempt_dir, step=step, source_id=source_id, "
            "target_id=parent_id)\n"
        ),
        new=(
            "    attempts.seal(ctx.step_dir / parent_id, attempt_dir)\n"
            "    attempts.promote(attempt_dir, step=step, source_id=source_id, "
            "target_id=parent_id)\n"
            "    _write_resolution(attempt_dir, source_id, criterion_key)\n"
        ),
        tests="tests/test_recovery.py",
        breaks="an install stopped once the winner's output is at the canonical path leaves "
        "it unnamed, and the next resume caches the promoted child as an untouched round-1 "
        "result for good",
    ),
    Mutation(
        id="nms-record-worn-only-over-its-output",
        path="src/chemrefine/nms.py",
        old="    return child == canonical\n",
        new="    return True\n",
        tests="tests/test_nms.py",
        breaks="the record an install writes before its promote is worn by whatever reaches "
        "the canonical path instead — round 1 run again there after a stop is cached as "
        "resolved from a child whose calculation never arrived",
    ),
    Mutation(
        id="nms-winner-credited-only-over-its-output",
        path="src/chemrefine/nms.py",
        old="            if not _is_promoted(engine, ctx, s.id, attempt, winner_source):",
        new="            if False:",
        tests="tests/test_nms.py",
        breaks="a rebuild caches a child's energies and frequencies under its parent's id "
        "while the canonical output is round 1's — a saddle reported as the minimum a child "
        "reached, and a mode drawn from one calculation beside frequencies from another",
    ),
    Mutation(
        id="nms-record-worn-only-under-its-criterion",
        path="src/chemrefine/nms.py",
        old="    if resolved_under != criterion_key:",
        new="    if False:",
        tests="tests/test_nms.py",
        breaks="a structure that passes through under an edited criterion is cached as "
        "resolved from the child another criterion chose — a winner this configuration "
        "never judged, named as its own",
    ),
    Mutation(
        id="resubmit-skips-unconverged",
        path="src/chemrefine/lifecycle.py",
        old="_NOT_RESUBMITTED = frozenset({FailureKind.NOT_CONVERGED, FailureKind.UNREADABLE})",
        new="_NOT_RESUBMITTED = frozenset({FailureKind.UNREADABLE})",
        tests="tests/test_recovery.py",
        breaks="every resume re-runs each unconverged structure twice — once from the seed, "
        "discarding the geometry it had reached, and once more from best — so a structure "
        "that needs one more step costs two jobs per resume and converges a resume late",
    ),
    Mutation(
        id="resubmit-skips-unreadable",
        path="src/chemrefine/lifecycle.py",
        old="_NOT_RESUBMITTED = frozenset({FailureKind.NOT_CONVERGED, FailureKind.UNREADABLE})",
        new="_NOT_RESUBMITTED = frozenset({FailureKind.NOT_CONVERGED})",
        tests="tests/test_lifecycle.py",
        breaks="a resume archives every finished output it may not read and runs its job "
        "again, so a colleague's step is computed a second time by whoever resumes it "
        "without read access, where restoring the access would have read it",
    ),
    Mutation(
        id="convert-direction",
        path="src/chemrefine/quantities.py",
        old='    ("hartree", "ev"): HARTREE_TO_EV,\n    ("ev", "hartree"): 1.0 / HARTREE_TO_EV,',
        new='    ("hartree", "ev"): 1.0 / HARTREE_TO_EV,\n    ("ev", "hartree"): HARTREE_TO_EV,',
        tests="tests/test_quantities.py",
        breaks="every eV a caller converts is wrong by 27.2 squared while the round trip still "
        "passes — each direction is its own constant, so swapping both is invisible to it",
    ),
    Mutation(
        id="boltzmann-sign",
        path="src/chemrefine/quantities.py",
        old="    weights = np.exp(-arr / (R_KCALMOL_K * temperature_k))",
        new="    weights = np.exp(arr / (R_KCALMOL_K * temperature_k))",
        tests="tests/test_quantities.py",
        breaks="every Boltzmann weight, filter and steps.csv column prefers the higher-energy "
        "conformer",
    ),
    Mutation(
        id="boltzmann-shift",
        path="src/chemrefine/quantities.py",
        old="        arr = arr - arr.min()",
        new="        arr = arr - 0.0",
        tests="tests/test_quantities.py",
        breaks="raw absolute energies underflow every exponential to zero, so a step's weights "
        "are all 0 and its cumulative filter keeps nothing",
    ),
    Mutation(
        id="ranking-energy-fallback",
        path="src/chemrefine/filtering.py",
        old='    energy_type = "electronic" if sample is None else sample.energy_type',
        new='    energy_type = "electronic"',
        tests="tests/test_filtering.py",
        breaks="a step sampling on gibbs ranks, promotes, captions and reports on the "
        "electronic energy — four readers wrong at once",
    ),
    Mutation(
        id="min-count-zero-keeps-all",
        path="src/chemrefine/filtering.py",
        old="return list(sorted_structures) if sample.count <= 0 else",
        new="return list(sorted_structures) if sample.count < 0 else",
        tests="tests/test_filtering.py",
        breaks="`count: 0`, documented as keep everything, keeps nothing",
    ),
    Mutation(
        id="max-keeps-the-high-end",
        path="src/chemrefine/filtering.py",
        old="    high_first = list(reversed(sorted_structures))",
        new="    high_first = list(sorted_structures)",
        tests="tests/test_filtering.py",
        breaks="`method: max` keeps the lowest-energy structures — PES-style sampling inverted",
    ),
    Mutation(
        id="ts-reaction-coordinate",
        path="src/chemrefine/nms.py",
        old="        return max(imag_freqs, key=lambda i: abs(imag_freqs[i]))",
        new="        return min(imag_freqs, key=lambda i: abs(imag_freqs[i]))",
        tests="tests/test_nms.py",
        breaks="a `ts` step keeps the least imaginary mode and displaces along the reaction "
        "coordinate itself, destroying the saddle it was asked to keep",
    ),
    Mutation(
        id="nms-crashed-child-not-resolved",
        path="src/chemrefine/nms.py",
        old="    if child.terminated_normally is False:\n        return False\n"
        "    if target is None:",
        new="    if target is None:",
        tests="tests/test_nms.py",
        breaks="a child whose program died is promoted as resolved on whatever frequency "
        "table its truncated output still carried",
    ),
    Mutation(
        id="random-skips-trivial-modes",
        path="src/chemrefine/nms.py",
        old="    for idx in _selected_modes(imag_freqs, struct.frequencies or {}, opts, rng):",
        new="    for idx in _selected_modes(imag_freqs, dict.fromkeys(range(n_modes), 0.0), "
        "opts, rng):",
        tests="tests/test_nms.py",
        breaks="`random` sampling spends jobs displacing along translations and rotations — "
        "the same molecule moved, re-computed, and reported as an exploration",
    ),
    Mutation(
        id="boltzmann-cutoff",
        path="src/chemrefine/filtering.py",
        old="sorted_structures[: n_below + 1]",
        new="sorted_structures[:n_below]",
        tests="tests/test_filtering.py",
        breaks="the structure that crosses the cumulative-weight threshold is dropped, so "
        "a 99% filter silently keeps less than it promises",
    ),
    Mutation(
        id="min-window-boundary",
        path="src/chemrefine/filtering.py",
        old="if cast(float, getattr(s, energy_attr)) <= min_e + window_h",
        new="if cast(float, getattr(s, energy_attr)) < min_e + window_h",
        tests="tests/test_filtering.py",
        breaks="a structure exactly on the window edge is silently dropped, so the filter "
        "keeps less than it promises",
    ),
    Mutation(
        id="max-window-boundary",
        path="src/chemrefine/filtering.py",
        old=">= max_e - window_h]",
        new="> max_e - window_h]",
        tests="tests/test_filtering.py",
        breaks="a structure exactly on the window edge is silently dropped, so the filter "
        "keeps less than it promises — the high-energy mirror of min-window-boundary",
    ),
    Mutation(
        id="report-kcal-conversion",
        path="src/chemrefine/io.py",
        old='df["Energy (Hartree)"] * HARTREE_TO_KCALMOL',
        new='df["Energy (Hartree)"] / HARTREE_TO_KCALMOL',
        tests="tests/test_io.py",
        breaks="every kcal/mol column in steps.csv is wrong by a factor of ~627.5^2 while "
        "row order, headers and dE=0 all still hold — the user-facing report ships wrong "
        "numbers with nothing else red",
    ),
    Mutation(
        id="best-backfills-best-not-seed",
        path="src/chemrefine/lifecycle.py",
        old="    if failure.best is not None:\n        return failure.best\n",
        new="    if failure.best is not None:\n        return prev_by_id.get(failure.sid)\n",
        tests="tests/test_step.py",
        breaks="`on_failure: best` silently carries the submitted seed geometry downstream "
        "instead of the best geometry the failed run reached — ids and counts unchanged, "
        "so only a positions assertion can see it",
    ),
    Mutation(
        id="energy-pairs-with-its-geometry",
        path="src/chemrefine/engines/orca/output/coordinator.py",
        old="coords = geometry.parse_coordinates_from_text(evaluation)",
        new="coords = geometry.parse_coordinates_from_text(text)",
        tests="tests/test_engines_orca_output.py",
        breaks="an optimisation killed mid-cycle stores one cycle's energy on the next "
        "cycle's geometry, and `on_failure: best` carries that pair downstream — an "
        "energy for coordinates it was never computed at",
    ),
    Mutation(
        id="extopt-cycle-energy",
        path="src/chemrefine/engines/orca/output/coordinator.py",
        old="    if relaxed is None or (\n",
        new="    if True or (\n",
        tests="tests/test_engines_orca_output.py",
        breaks="an ExtOpt optimisation that stops at its cycle limit is read at the geometry "
        "it was submitted with, so its unconverged retry resubmits the identical job and "
        "`on_failure: best` carries the unoptimised input downstream",
    ),
    Mutation(
        id="hessian-of-its-own-geometry",
        path="src/chemrefine/engines/orca/output/coordinator.py",
        old="_parse_frequency_block(evaluation, n_atoms=len(symbols))",
        new="_parse_frequency_block(text, n_atoms=len(symbols))",
        tests="tests/test_engines_orca_output.py",
        breaks="a TS search's initial Hessian is reported as its optimised structure's, so "
        "the imaginary modes a `ts` step verifies and displaces along were computed at "
        "another geometry",
    ),
    Mutation(
        id="retry-best-frame",
        path="src/chemrefine/lifecycle.py",
        old="best = min(bad, key=lambda s: (s.energy_hartree is None, s.energy_hartree or 0.0))",
        new="best = max(bad, key=lambda s: (s.energy_hartree is None, s.energy_hartree or 0.0))",
        tests="tests/test_lifecycle.py",
        breaks="a fan-out job's retry restarts from the worst unconverged frame, and "
        "`on_failure: best` backfills the worst geometry obtained",
    ),
    Mutation(
        id="succeeded-ignores-convergence",
        path="src/chemrefine/lifecycle.py",
        old="return s.terminated_normally is not False and s.converged is not False",
        new="return s.terminated_normally is not False",
        tests="tests/test_lifecycle.py",
        breaks="an unconverged structure counts as a success, so it is never retried, "
        "never ledgered, and ranks against converged siblings",
    ),
    Mutation(
        id="structure-digest-ignores-geometry",
        path="src/chemrefine/cache.py",
        old="h.update(np.asarray(s.atoms.get_positions(), dtype=np.float64).tobytes())",
        new="pass",
        tests="tests/test_cache.py",
        breaks="a structure's coordinates leave its digest, so editing the seed geometry "
        "moves no row key and `resume` serves results computed from the old coordinates",
    ),
    Mutation(
        id="cache-pair-is-checked",
        path="src/chemrefine/cache.py",
        old='if found != document["arrays_digest"]:',
        new="if False:",
        tests="tests/test_cache.py",
        breaks="a `step.json` left by one save is read with the `arrays.npz` of another, so "
        "each structure keeps its own energy and adopts a different structure's geometry",
    ),
    Mutation(
        id="array-task-id-matching",
        path="src/chemrefine/slurm/dispatch.py",
        old='mine = {line for line in running if line == jid or line.startswith(f"{jid}_")}',
        new="mine = {line for line in running if line == jid}",
        tests="tests/test_slurm.py",
        breaks="a running array job reports finished — squeue prints its tasks as "
        "`12345_0`, never the bare parent id — so its outputs are parsed while it writes",
    ),
    Mutation(
        id="lease-batch-on-record-before-its-first-job",
        path="src/chemrefine/engines/_execution.py",
        old="batch = slurm.lease_batch(ctx.step_dir) if pending and not plan.local else None",
        new="batch = None",
        tests="tests/test_execution.py",
        breaks="a batch asks the scheduler for jobs under no recorded name, so one whose id "
        "never reaches the ledger — the driver killed or sbatch failing once the scheduler "
        "has the job — is invisible to the next run's fence, which resubmits the structures "
        "it is still writing",
    ),
    Mutation(
        id="fence-asks-every-accounts-queue",
        path="src/chemrefine/pipeline.py",
        old="queue = slurm.poll_jobs(tuple(scheduled), every_account=True)",
        new="queue = slurm.poll_jobs(tuple(scheduled))",
        tests="tests/test_pipeline.py",
        breaks="a resume by a second account asks for its own jobs alone, reads every job "
        "the first account's driver recorded as drained, sweeps the ledgers and resubmits "
        "beside them",
    ),
    Mutation(
        id="lease-batch-jobs-found-by-tag",
        path="src/chemrefine/slurm/dispatch.py",
        old="mine = {row[0] for row in queued if row[-1].endswith(_name_tail(jid))}",
        new="mine = {row[0] for row in queued if row[-1] == _name_tail(jid)}",
        tests="tests/test_slurm.py",
        breaks="no queued job is ever found under a batch's tag, so the fence passes beside "
        "every job the ledger names by tag alone and the run resubmits beside it",
    ),
    Mutation(
        id="mem-per-gpu-header-stands-alone",
        path="src/chemrefine/slurm/script.py",
        old="    if per_gpu is not None:\n        logger.warning(",
        new="    if False:\n        logger.warning(",
        tests="tests/test_slurm.py",
        breaks="a GPU header that grants --mem-per-gpu gets a derived --mem-per-cpu appended "
        "beside it for an input that declares its memory — a pair sbatch refuses as "
        "mutually exclusive, so the step cannot submit one job",
    ),
    Mutation(
        id="tree-write-names-the-refusal",
        path="src/chemrefine/io.py",
        old='        raise refusal(f"cannot write {path}: {reason}") from e\n',
        new="        raise\n",
        tests="tests/test_io.py",
        breaks="a full disk, a spent quota or a read-only mount under a run ends it in a "
        "traceback with exit 1, the code a halted step exits with, so a wrapper that resumes "
        "on 1 resubmits into the same disk, and the GUI and an agent get a crash where they "
        "would show a refusal naming the file",
    ),
    Mutation(
        id="absent-only-on-no-such-file",
        path="src/chemrefine/io.py",
        old="    except OSError:\n        return False\n    return False\n",
        new="    except OSError:\n        return True\n    return False\n",
        tests="tests/test_io.py",
        breaks="an output under a directory the run cannot search is ledgered as never "
        "written, so a resume takes the finished job for one that left nothing and runs it "
        "again",
    ),
    Mutation(
        id="unlistable-directory-is-not-empty",
        path="src/chemrefine/io.py",
        old='        raise CacheError(f"cannot read {directory}: {e.strerror or e}") from e\n',
        new="        return []\n",
        tests="tests/test_io.py",
        breaks="a structure directory that cannot be listed reads as one with no attempts, "
        "so a promoted NMS winner is cached as an untouched round 1 and the next attempt is "
        "numbered over the first",
    ),
    Mutation(
        id="cleanup-keeps-the-exception-in-flight",
        path="src/chemrefine/io.py",
        old="        if in_flight is None:\n",
        new="        if True:\n",
        tests="tests/test_io.py",
        breaks="a lock or lease release the output tree refuses while a run unwinds takes the "
        "place of what stopped it, so a SIGTERM's 143 leaves as exit 11 and a batch's own "
        "error as the release's",
    ),
    Mutation(
        id="build-structures-takes-back-a-partial-seed-set",
        path="src/chemrefine/agent_tools.py",
        old="        for done in written:\n            with io.cleanup():\n"
        "                io.remove_user_file(done)\n        raise\n",
        new="        raise\n",
        tests="tests/test_agent_tools_chemistry.py",
        breaks="a seed set a full disk or a molecule that will not embed cuts short stays in "
        "the directory, and a run whose `input:` is that directory computes on the molecules "
        "that landed — fewer than were asked for, with nothing saying so",
    ),
    Mutation(
        id="cache-only-may-submit",
        path="src/chemrefine/step.py",
        old="case StepMode.CACHE_ONLY | StepMode.REBUILD:\n                return False",
        new="case StepMode.CACHE_ONLY | StepMode.REBUILD:\n                return True",
        tests="tests/test_e2e_relocate.py",
        breaks="`rebuild-cache` and `rerun-errors` submit work for steps they were not "
        "pointed at, archiving the very outputs they were asked to read",
    ),
    Mutation(
        id="cache-only-may-halt",
        path="src/chemrefine/step.py",
        old=(
            "            case StepMode.CACHE_ONLY:\n                return False\n"
            "            case StepMode.EXECUTE | StepMode.RESUME | StepMode.REBUILD:\n"
            "                return True"
        ),
        new=(
            "            case StepMode.CACHE_ONLY:\n                return True\n"
            "            case StepMode.EXECUTE | StepMode.RESUME | StepMode.REBUILD:\n"
            "                return True"
        ),
        tests="tests/test_step.py",
        breaks="`rerun-errors N` and `rebuild-cache N` stop at an earlier `stop` step's "
        "pending failures and never reach step N — the scoped actions become unusable "
        "on exactly the trees they exist for",
    ),
    Mutation(
        id="mode-index-negative",
        path="src/chemrefine/agent_tools.py",
        old="    if not 0 <= mode_index < n_modes:\n",
        new="    if mode_index >= n_modes:\n",
        tests="tests/test_agent_tools_chemistry.py",
        breaks="`mode_index=-1` describes and animates the last mode instead of refusing, "
        "so a transition-state check reads a mode nobody asked for",
    ),
    Mutation(
        id="resume-row-adoption",
        path="src/chemrefine/step.py",
        old="if sid not in provenance.rows or provenance.rows[sid][0] != current[sid][0]",
        new="if False",
        tests="tests/test_step.py",
        breaks="resume adopts every row on disk whatever its provenance says, so a changed "
        "parent's stale output is served as current and the changed row never computes",
    ),
    Mutation(
        id="resume-artifact-adoption",
        path="src/chemrefine/step.py",
        old="    if provenance.fingerprint != key.fingerprint:\n        return None\n",
        new="    if False:\n        return None\n",
        tests="tests/test_step_artifact.py",
        breaks="resume adopts whatever trained model is on disk under any configuration, so "
        "a retuned training is never run and later steps compute with the previous model",
    ),
    Mutation(
        id="artifact-finished-by-its-jobs-verdict",
        path="src/chemrefine/step.py",
        old="        elif footer.exit_code != 0:\n",
        new="        elif False:\n",
        tests="tests/test_step_artifact.py",
        breaks="a training job that was stopped or died is cached as a trained model whenever "
        "it left a checkpoint at the product's path, and every later step computes with it",
    ),
    Mutation(
        id="resume-prepares-only-what-it-recomputes",
        path="src/chemrefine/step.py",
        old="        adopted=(job for job in manifest.files if job[2] in adopted),\n",
        new="        adopted=(),\n",
        tests="tests/test_step.py",
        breaks="resume prepares the rows it adopts again: a tree adopted under file names of "
        "its own is archived and recomputed, and a resolved structure's directory pairs the "
        "winning child's output with round 1's input and starting geometry",
    ),
    Mutation(
        id="rebuild-cache-provenance",
        path="src/chemrefine/step.py",
        old="if foreign or unproven:",
        new="if False:",
        tests="tests/test_step.py",
        breaks="`rebuild-cache` adopts rows whose provenance disagrees with the current "
        "key, caching results this configuration never produced — and the next `resume` "
        "serves that instead of computing what was asked for",
    ),
    Mutation(
        id="rebuild-cache-unproven-rows",
        path="src/chemrefine/step.py",
        old="if foreign or unproven:",
        new="if foreign:",
        tests="tests/test_step.py",
        breaks="`rebuild-cache` adopts a tree whose rows do not cover a grown parent set, "
        "caching a survivor set one parent short — and the next `resume` serves it "
        "(this exact mutant survived the whole suite before its test existed)",
    ),
    Mutation(
        id="resume-resolution-stamp-waits",
        path="src/chemrefine/step.py",
        old='stamp["search_key"] = provenance.search_key',
        new='stamp["search_key"] = key.search_key',
        tests="tests/test_step.py",
        breaks="the incremental resume stamps the current search key before the "
        "resolution that earns it exists, so a driver killed mid-resume leaves the "
        "previous search's attempts adoptable by `rebuild-cache`",
    ),
    Mutation(
        id="policy-change-over-cache",
        path="src/chemrefine/step.py",
        old="return policy_stores_backfills(stored) != current.stores_backfills",
        new="return False",
        tests="tests/test_step.py",
        breaks="editing on_failure over a cached step silently serves the previous "
        "policy's survivor set — a step switched to `best` quietly behaves as `skip`, "
        "and one switched away from `best` keeps carrying its backfills",
    ),
    Mutation(
        id="dead-job-vs-unreadable-file",
        path="src/chemrefine/engines/orca/output/coordinator.py",
        old="if status.parse_terminated_normally(text):\n        return OutputParseError",
        new="if True:\n        return OutputParseError",
        tests="tests/test_engines_orca_output.py",
        breaks="a job that died is ledgered as an unreadable file, sending a reader to the "
        "parser for a cluster or input problem",
    ),
    Mutation(
        id="mlip-optimize-verdict",
        path="src/chemrefine/engines/mlip/calculator.py",
        old="converged = bool(LBFGS(atoms, logfile=os.devnull).run(fmax=fmax, steps=steps))",
        new="converged = bool(LBFGS(atoms, logfile=os.devnull).run(fmax=fmax, steps=steps))"
        " or True",
        tests="tests/test_engines_mlip_calculator.py",
        breaks="an optimiser that ran out of steps reports its last geometry as converged, and "
        "a non-stationary point ranks as a survivor against converged siblings",
    ),
    Mutation(
        id="finite-energy-guard",
        path="src/chemrefine/engines/_script/output.py",
        old="if not np.isfinite(number):",
        new="if False:",
        tests="tests/test_properties.py",
        breaks="a diverged calculation's NaN energy ranks as a real result and, because "
        "every NaN comparison is false, displaces a genuine survivor by list position",
    ),
    Mutation(
        id="finite-geometry-guard",
        path="src/chemrefine/cache.py",
        old="if array is not None and not np.isfinite(np.asarray(array, dtype=np.float64)).all():",
        new="if False:",
        tests="tests/test_cache.py",
        breaks="a NaN geometry is written to arrays.npz — the half of a record `write_json`'s "
        "`allow_nan=False` never sees — and served to every downstream step; it round-trips "
        "the cache and `structure_digest` hashes it to a stable key, so a run reports results "
        "computed from coordinates that are not numbers, and nothing anywhere says so",
    ),
    Mutation(
        id="pyscf-mol-unit",
        path="src/chemrefine/engines/pyscf/_runtime.py",
        old='mol.atom = atom\n    mol.unit = "Bohr"',
        new="mol.atom = atom",
        tests="tests/test_engines_pyscf_extopt_calc.py",
        breaks="PySCF's default unit is Angstrom, so the coordinates — already converted "
        "to Bohr — are read 1.889x too large: silently wrong energies and gradients on "
        "every PySCF job",
    ),
    Mutation(
        id="pyscf-df-applied",
        path="src/chemrefine/engines/pyscf/_runtime.py",
        old="mf = mf.density_fit()",
        new="pass",
        tests="tests/test_engines_pyscf_extopt_calc.py",
        breaks="density fitting silently never applied on the shipped-default path — the "
        "RI approximation the user configured, priced, and did not get, with a different "
        "energy and a much longer runtime",
    ),
    Mutation(
        id="boltzmann-temperature",
        path="src/chemrefine/filtering.py",
        old="sorted_structures, sample.percent_cumulative, sample.temperature_k, energy_attr",
        new="sorted_structures, sample.percent_cumulative, 298.15, energy_attr",
        tests="tests/test_filtering.py",
        breaks="a `temperature_k: 77` step filters at room temperature — a different "
        "survivor set at every non-default temperature, with nothing anywhere saying so "
        "(this exact mutant survived the full suite before the temperature test existed)",
    ),
    Mutation(
        id="report-temperature",
        path="src/chemrefine/pipeline.py",
        old="temperature_k = sample.temperature_k if sample is not None else DEFAULT_TEMPERATURE_K",
        new="temperature_k = DEFAULT_TEMPERATURE_K",
        tests="tests/test_pipeline.py",
        breaks="steps.csv's Boltzmann columns are computed at the default temperature "
        "whatever the step configured — a report that silently contradicts the survivor "
        "set the filter produced (this exact mutant survived the full suite)",
    ),
    Mutation(
        id="wait-for-jobs-full-drain",
        path="src/chemrefine/slurm/dispatch.py",
        old="remaining = pending - state.finished",
        new="remaining = set()",
        tests="tests/test_slurm.py",
        breaks="the array-path wait declares victory after its first poll while every "
        "task still runs — outputs are parsed mid-write and structures ledgered "
        "MISSING_OUTPUT while the jobs finish into an archived tree (this exact mutant "
        "survived while the partial-drain tests held no assertions)",
    ),
    Mutation(
        id="pes-atom-count",
        path="src/chemrefine/engines/orca/output/ensembles.py",
        old="        elif len(atoms) != n_atoms:",
        new="        elif False:",
        tests="tests/test_engines_orca_output.py",
        breaks="a scan point that lost a coordinate row ships as a smaller molecule "
        "carrying the whole molecule's energy — the next step optimises the wrong species",
    ),
    Mutation(
        id="best-backfill-is-a-geometry",
        path="src/chemrefine/lifecycle.py",
        old=(
            "    return Structure(id=submitted.id, atoms=submitted.atoms, "
            "parent_id=submitted.parent_id)"
        ),
        new="    return submitted",
        tests="tests/test_step.py",
        breaks="an `on_failure: best` backfill carries the previous step's energy into this "
        "step's filter and steps.csv — ranked and reported on a number this step never "
        "computed",
    ),
    Mutation(
        id="finalize-orders-what-it-stores",
        path="src/chemrefine/lifecycle.py",
        old=(
            "    kept = apply_failure_policy(successes, failures, ctx, step_cfg)\n"
            "    results = StepResults(structures=_in_parent_order(kept, ctx.prev_state))\n"
        ),
        new=(
            "    kept = apply_failure_policy(\n"
            "        _in_parent_order(successes, ctx.prev_state), failures, ctx, step_cfg\n"
            "    )\n"
            "    results = StepResults(structures=kept)\n"
        ),
        tests="tests/test_lifecycle.py",
        breaks="`on_failure: best` stores its backfills after the survivors, in whatever "
        "order the recovery path failed them — a resume and a rebuild-cache of one step "
        "re-key every step after it, and an mlip-train step retrains on a new split",
    ),
    Mutation(
        id="ledger-settles-after-the-cache",
        path="src/chemrefine/lifecycle.py",
        old=(
            "    cache.save(\n"
            "        step_cfg=step_cfg,\n"
            "        key=key,\n"
            "        results=results,\n"
            "        step_dir=ctx.step_dir,\n"
            "        chemrefine_version=__version__,\n"
            "    )\n"
            "    if not records:\n"
            "        cache.clear_failed_jobs(ctx.step_dir)\n"
            "    elif ledgered != records:\n"
            "        cache.save_failure_records(ctx.step_dir, records)\n"
        ),
        new=(
            "    if not records:\n"
            "        cache.clear_failed_jobs(ctx.step_dir)\n"
            "    elif ledgered != records:\n"
            "        cache.save_failure_records(ctx.step_dir, records)\n"
            "    cache.save(\n"
            "        step_cfg=step_cfg,\n"
            "        key=key,\n"
            "        results=results,\n"
            "        step_dir=ctx.step_dir,\n"
            "        chemrefine_version=__version__,\n"
            "    )\n"
        ),
        tests="tests/test_recovery.py",
        breaks="a resume stopped at its cache write has already taken what it repaired off "
        "the ledger — the earlier cache still matches its key, so the next resume serves it "
        "whole and the repaired structure is gone from every later step, with nothing halted",
    ),
    Mutation(
        id="failures-ledgered-before-the-cache",
        path="src/chemrefine/lifecycle.py",
        old="            cache.save_failure_records(ctx.step_dir, ledgered)\n",
        new="            ledgered = None\n",
        tests="tests/test_recovery.py",
        breaks="a step's failures reach the ledger only after its cache — killed between the "
        "two, the cache holds the successes under a key that matches, so the next resume is "
        "a clean hit and the failed structures are never run again",
    ),
    Mutation(
        id="ledger-added-to-before-the-cache",
        path="src/chemrefine/lifecycle.py",
        old=(
            "            ledgered = [*(r for r in earlier if r.structure_id not in failed), "
            "*records]\n"
        ),
        new="            ledgered = list(records)\n",
        tests="tests/test_lifecycle.py",
        breaks="a run that repairs one structure and fails another drops the repaired one "
        "from the ledger before its cache is saved — stopped there, the earlier cache still "
        "matches its key and the structure is on neither document: an NMS re-attempt, which "
        "reads only what the ledger names, and every skip or best step serve the cache "
        "without it",
    ),
    Mutation(
        id="unreadable-ledger-does-not-stop-a-step",
        path="src/chemrefine/lifecycle.py",
        old="        except CacheError:\n            pass  # not a ledger this can add to",
        new="        except KeyError:\n            pass  # not a ledger this can add to",
        tests="tests/test_lifecycle.py",
        breaks="a step that ends with a failure beside a ledger this version cannot read — a "
        "v1.3.1 tree's, a corrupt one — stops on that ledger after its jobs have run, and "
        "`rebuild-cache`, the documented repair for both, stops the same way",
    ),
    Mutation(
        id="exit-status-answers-for-no-verdict",
        path="src/chemrefine/lifecycle.py",
        old='    elif footer.exit_code == 0:\n        return parsed, ""\n',
        new='    elif True:\n        return parsed, ""\n',
        tests="tests/test_lifecycle.py",
        breaks="a program that died after printing an energy and a geometry is cached as a "
        "success wherever its engine reads no verdict off the output — ranked and handed to "
        "the next step, with no ledger entry and exit 0",
    ),
    Mutation(
        id="a-run-cut-short-is-no-success",
        path="src/chemrefine/lifecycle.py",
        old='        if not job_log.cut_short(runlog):\n            return parsed, ""\n',
        new='        if True:\n            return parsed, ""\n',
        tests="tests/test_lifecycle.py",
        breaks="a job stopped and cut down before its handler finished — an interrupt past "
        "the kill grace, a walltime, a node that went down — leaves an output written up to "
        "the moment it died, and a resume adopts it as a success wherever its engine reads "
        "no verdict",
    ),
    Mutation(
        id="exit-status-leaves-an-engines-fault-alone",
        path="src/chemrefine/lifecycle.py",
        old="        return s.terminated_normally is None and succeeded(s)\n",
        new="        return s.terminated_normally is None\n",
        tests="tests/test_lifecycle.py",
        breaks="an unconverged structure whose script also exited non-zero is filed as not "
        "terminated — the next resume resubmits it from its original input, where the "
        "convergence retry would have restarted it from the geometry it reached",
    ),
    Mutation(
        id="orca-scan-block-bound",
        path="src/chemrefine/engines/orca/inspect.py",
        old='_GEOM_BLOCK_RE = re.compile(r"%geom\\b([^%*!]*)", re.IGNORECASE)',
        new='_GEOM_BLOCK_RE = re.compile(r"%geom\\b([\\s\\S]*)", re.IGNORECASE)',
        tests="tests/test_engines_orca_inspect.py",
        breaks="an `Opt Freq` template naming `scan` anywhere after its `%geom` block is read "
        "by the scan parser — the step keeps its energies and silently loses its frequency "
        "table and every imaginary mode",
    ),
    Mutation(
        id="legacy-spellings-clash-refused",
        path="src/chemrefine/config_legacy.py",
        old="    if target[key] != value:\n",
        new="    if False:\n",
        tests="tests/test_config.py",
        breaks="a half-migrated config that sets a v1 key and its current spelling to different "
        "values runs the value already in place — the current key's where the file sets one, "
        "else the first legacy spelling's — with nothing said",
    ),
    Mutation(
        id="legacy-unit-read-only-with-energy",
        path="src/chemrefine/config_legacy.py",
        old='        if key == "unit" and "energy" in sample:\n',
        new='        if key == "unit":\n',
        tests="tests/test_config.py",
        breaks="a hand-migrated `window_kcalmol: 0.005, unit: hartree` loses its unit, reported "
        "as read by a conversion that never ran, and filters with a window 627 times narrower "
        "than its author meant",
    ),
    Mutation(
        id="anchored-paths-judged-by-the-model",
        path="src/chemrefine/config.py",
        old="    return type(cfg).model_validate(cfg.model_copy(update=updates)) "
        "if updates else cfg\n",
        new="    return cfg.model_copy(update=updates) if updates else cfg\n",
        tests="tests/test_config.py",
        breaks="a config in a directory named `$(…)` passes `load_config` and `validate` and is "
        "refused only when its first job script is built, and a `scratch_dir` naming the "
        "directory `output_dir` anchors to loads until a budget override refuses it in the "
        "budget's name",
    ),
    Mutation(
        id="cores-left-out-of-the-key",
        path="src/chemrefine/engines/_options.py",
        old="    cores: Annotated[int, UNKEYED] = Field(1, ge=1)\n",
        new="    cores: int = Field(1, ge=1)\n",
        tests="tests/test_step.py",
        breaks="raising a step's core count to finish faster recomputes every row it already "
        "finished on `resume`, and `rebuild-cache` refuses the tree as another configuration's",
    ),
    Mutation(
        id="step-dir-refused-at-run-start",
        path="src/chemrefine/step.py",
        old="        check_step_dir(engine, ctx)\n",
        new="        # check_step_dir(engine, ctx)\n",
        tests="tests/test_pipeline.py",
        breaks="a step whose engine cannot express a path with whitespace is refused only when "
        "it writes its first input, after every step ahead of it has run and been paid for",
    ),
    Mutation(
        id="no-validation-set-refused-at-run-start",
        path="src/chemrefine/engines/mlip/train/engine.py",
        old="            trainer_for(opts.task_name)()"
        ".require_validation(opts.valid_fraction > 0)\n",
        new="            pass\n",
        tests="tests/test_engines_mlip_train_engine.py",
        breaks="a training step whose library needs a validation set is refused for "
        "`valid_fraction: 0` only when it writes its dataset, after the labelling steps ahead "
        "of it have run and been paid for",
    ),
    Mutation(
        id="start-run-hands-on-the-resolved-config",
        path="src/chemrefine/agent_tools.py",
        old="    argv.append(str(path))\n",
        new="    argv.append(config_path)\n",
        tests="tests/test_agent_tools.py",
        breaks="a run started on `~/proj/input.yaml` hands the child that text, which no shell "
        "expands, so it exits 2 into a log nobody watches after a pid was already returned",
    ),
    Mutation(
        id="cache-sha1-asked-for-a-fingerprint",
        path="src/chemrefine/cache.py",
        old="    return hashlib.sha1(usedforsecurity=False)\n",
        new="    return hashlib.sha1()\n",
        tests="tests/test_cache.py",
        breaks="on a host whose crypto policy allows SHA-1 only as a fingerprint, every step "
        "whose options name a file dies in a traceback while its key is derived",
    ),
)

#: What an engine files beside its contract fixture — ``tests/data/engines/<name>/`` — to
#: put its own guards on the list: a JSON array of objects with :class:`Mutation`'s fields.
#: Read here so an engine ships its mutations without an edit to this script.
ENGINE_MUTATIONS_NAME = "mutations.json"


def engine_mutations(root: Path) -> tuple[Mutation, ...]:
    """Every mutation an engine files beside its fixture under ``root``, in path order."""
    found: list[Mutation] = []
    fixtures = root / "tests" / "data" / "engines"
    for path in sorted(fixtures.glob(f"*/{ENGINE_MUTATIONS_NAME}")):
        entries = json.loads(path.read_text(encoding="utf-8"))
        found.extend(Mutation(**entry) for entry in entries)
    return tuple(found)


MUTATIONS: tuple[Mutation, ...] = (*_BUNDLED, *engine_mutations(REPO))
"""Every mutation the gate runs: the bundled ones above, then each engine's own."""


#: Everything a pytest run reads. ``examples`` belongs here because the shipped tutorials
#: are living documentation and several tests resolve their templates; ``README.md`` and
#: ``docs`` because ``test_docs_examples`` validates every full config their prose shows.
#: The last four are ``test_docs_urls``'s: it reads the canonical ``site_url``/``repo_url``
#: out of ``mkdocs.yml`` at *import* time — so its absence would be a collection error
#: rather than a failing test, and the whole gate would refuse to run — and it resolves every
#: self-referential ``blob/main/…`` URL against the tree, which is what names the three
#: top-level files the prose links to.
#: An incomplete copy fails on its own, which :func:`_assert_baseline_is_green` catches
#: whatever the missing input turns out to be — it is how each of these earned its entry.
_INPUTS = (
    "src",
    "tests",
    "examples",
    "docs",
    "README.md",
    "pyproject.toml",
    "mkdocs.yml",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
)


def _copy_tree(dest: Path) -> None:
    """Copy the parts of the repo a pytest run needs, and nothing else."""
    for name in _INPUTS:
        source = REPO / name
        if source.is_dir():
            shutil.copytree(source, dest / name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, dest / name)


def _assert_isolated(work: Path, env: dict[str, str]) -> None:
    """Fail unless the *copy* is what an import resolves to.

    Without this the gate degrades silently into testing the pristine source, where every
    mutation is trivially "caught" and the report is a green light for nothing.
    """
    resolved = subprocess.run(
        [sys.executable, "-c", "import chemrefine; print(chemrefine.__file__)"],
        capture_output=True,
        text=True,
        cwd=work,
        env=env,
        check=True,
    ).stdout.strip()
    if not resolved.startswith(str(work)):
        raise SystemExit(
            f"mutation gate is not isolated: `import chemrefine` resolved to {resolved},\n"
            f"outside the scratch copy at {work}. Refusing to report on the real source."
        )


def _assert_baseline_is_green(work: Path, env: dict[str, str]) -> None:
    """Fail unless the *unmutated* copy passes.

    The check that makes every verdict below mean something. If the copy is broken for a
    reason of its own — a missing input, a stale path — then every mutation "fails the
    suite" and the gate reports all-caught while testing nothing. A gate whose failure mode
    is a false green has to prove it can be green for the right reason first.
    """
    verdict = _run_suite(work, env)
    if verdict.caught or verdict.inconclusive:
        raise SystemExit(
            f"mutation gate baseline is not green: {verdict.why}\n"
            f"The unmutated copy at {work} already fails, so every mutation would look\n"
            f"'caught' regardless. Fix the copy (see _INPUTS) before trusting a verdict."
        )


def stale_anchors(root: Path, mutations: Sequence[Mutation]) -> list[str]:
    """One report line per mutation whose ``old`` is not unique, or whose ``tests`` is gone.

    Every anchor is checked before any is applied, and every stale one is reported
    together. Failing on the first would let a single moved line hide the state of the
    whole gate: the run would stop there, the mutations after it would never execute, and
    the exit code would say "gate failed" without saying that most of it had not run.

    Checked against the repository rather than the scratch copy, so a stale anchor costs
    milliseconds instead of a tree copy plus a baseline suite. Public because
    ``tests/test_mutation_gate.py`` asks the same question in the edit loop — which is
    where the developer who moves a line will actually see the answer.
    """
    problems: list[str] = []
    for mutation in mutations:
        source = root / mutation.path
        if not source.is_file():
            # The same report line the tests half gets: a `git mv` of a mutated source
            # file would otherwise raise FileNotFoundError out of main() — the one moved file
            # hiding the state of the whole gate, which is the failure this function's
            # aggregate report exists to prevent.
            problems.append(
                f"[{mutation.id}] names {mutation.path}, which is not a file; "
                f"the code moved or was renamed."
            )
        else:
            found = source.read_text(encoding="utf-8").count(mutation.old)
            if found != 1:
                problems.append(
                    f"[{mutation.id}] expected exactly one occurrence of\n    {mutation.old}\n"
                    f"in {mutation.path}, found {found}."
                )
        if not (root / mutation.tests).is_file():
            problems.append(
                f"[{mutation.id}] names {mutation.tests}, which is not a file; "
                f"the test moved or was renamed."
            )
    return problems


def _prepended_path(environ: Mapping[str, str], first: Path) -> str:
    """``PYTHONPATH`` with *first* ahead of whatever the caller already had on it.

    Replacing the variable outright would drop a developer's own entries — on a login node
    whose dependencies arrive through ``PYTHONPATH`` the baseline would fail to import
    them, and the gate would report its copy "not green" with a message pointing at the copy.
    """
    existing = environ.get("PYTHONPATH", "")
    return os.pathsep.join([str(first), existing]) if existing else str(first)


def _apply(work: Path, mutation: Mutation) -> None:
    """Rewrite the single occurrence of ``old`` in the copy.

    Uniqueness is :func:`stale_anchors`' job, proven for every selected mutation before
    this runs — so the rule lives in one place rather than in two that could drift.
    """
    target = work / mutation.path
    text = target.read_text(encoding="utf-8")
    mutated = text.replace(mutation.old, mutation.new)
    if text.count(mutation.old) != 1 or mutated == text:
        # A copy that drifted from the checked tree, or an entry whose `new` is its `old`:
        # either way the suite would run over unmutated code and every verdict from here
        # would be about nothing. Stop rather than report a survivor that does not exist.
        raise SystemExit(
            f"[{mutation.id}] would not change {mutation.path}: the anchor occurs "
            f"{text.count(mutation.old)} time(s) and the replacement "
            f"{'is identical' if mutation.new == mutation.old else 'differs'}."
        )
    target.write_text(mutated, encoding="utf-8")


@dataclass(frozen=True)
class Verdict:
    """What one run of the suite said about a mutation.

    ``caught`` is the suite going red. ``inconclusive`` is the suite proving nothing either
    way — not finishing at all on a run whose timeout cannot be read as a catch (the
    whole-suite fallback, see :data:`TIMEOUT_SECONDS`), or collecting no tests at all; it
    never comes with ``caught``. Neither set is a survivor: the suite finished green with
    the mutation in.
    """

    caught: bool
    why: str
    inconclusive: bool = False


def _run_suite(work: Path, env: dict[str, str], target: str | None = None) -> Verdict:
    """The :class:`Verdict` of the suite — or of one file of it — as it stands in ``work``.

    ``-x`` so a caught mutation stops at the first red test. ``target`` narrows the run to
    the file a mutation names: a red file is a red suite, so the verdict is the one the
    whole suite would give, reached in seconds instead of after ``-x`` has walked every
    alphabetically-earlier file first.
    """
    argv = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"]
    if target is not None:
        argv.append(target)
    try:
        # The only non-literal entry is `target`, which comes from this file's own
        # MUTATIONS table and is proven to name a real file by `stale_anchors`.
        completed = subprocess.run(  # noqa: S603
            argv,
            capture_output=True,
            text=True,
            cwd=work,
            env=env,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        if target is not None:
            return Verdict(True, f"{target} hung (>{TIMEOUT_SECONDS}s) — a red build either way")
        return Verdict(
            False,
            f"the whole suite did not finish within {TIMEOUT_SECONDS}s — a hang or a slow "
            "runner, and the gate cannot tell which",
            inconclusive=True,
        )
    if completed.returncode == 5:
        # pytest's "no tests collected": the named file was emptied, or everything in it
        # is deselected by the default addopts. Nothing ran, so nothing was caught — and
        # "non-zero means red" would report it caught, the false green this gate exists to
        # refuse. `stale_anchors` cannot see it: `is_file()` is true of a 0-byte file.
        # A named file that ran nothing is simply not a catch, and `main` then asks the
        # whole suite, which is where the verdict really comes from; the whole suite
        # collecting nothing is the same "nothing proven" as its timeout.
        if target is not None:
            return Verdict(False, f"{target} collected no tests — nothing ran")
        return Verdict(False, "the whole suite collected no tests — nothing ran", inconclusive=True)
    if completed.returncode != 0:
        first = next(
            (ln for ln in completed.stdout.splitlines() if ln.startswith("FAILED")),
            "suite failed",
        )
        return Verdict(True, first)
    return Verdict(False, "suite passed unchanged")


def main(argv: list[str] | None = None) -> int:
    """Run every selected mutation against a scratch copy; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-k", dest="pattern", default="", help="only ids containing this")
    parser.add_argument("--list", action="store_true", help="list the mutations and exit")
    args = parser.parse_args(argv)

    selected = [m for m in MUTATIONS if args.pattern in m.id]
    if args.list:
        for mutation in selected:
            print(
                f"{mutation.id:32} {mutation.path}\n"
                f"{'':32} checked by: {mutation.tests}\n"
                f"{'':32} if it survived: {mutation.breaks}"
            )
        return 0
    if not selected:
        raise SystemExit(f"no mutation id contains {args.pattern!r}")
    if stale := stale_anchors(REPO, selected):
        raise SystemExit("\n".join([*stale, "The code moved; update the mutation(s)."]))

    survivors: list[Mutation] = []
    inconclusive: list[tuple[Mutation, str]] = []
    with tempfile.TemporaryDirectory(prefix="chemrefine-mutation-") as tmp:
        work = Path(tmp)
        _copy_tree(work)
        env = {**os.environ, "PYTHONPATH": _prepended_path(os.environ, work / "src")}
        _assert_isolated(work, env)
        _assert_baseline_is_green(work, env)
        print(f"baseline green; {len(selected)} mutation(s) to check\n", flush=True)

        for mutation in selected:
            pristine = (work / mutation.path).read_text(encoding="utf-8")
            _apply(work, mutation)
            try:
                verdict = _run_suite(work, env, mutation.tests)
                if not verdict.caught:
                    # The named file missed it. Before calling anything a survivor, ask the
                    # whole suite — the entry may simply name the wrong file, and `why` then
                    # reports the test that did catch it.
                    verdict = _run_suite(work, env)
            finally:
                (work / mutation.path).write_text(pristine, encoding="utf-8")
            if verdict.inconclusive:
                label = "INCONCLUSIVE"
                inconclusive.append((mutation, verdict.why))
            elif verdict.caught:
                label = "caught"
            else:
                label = "SURVIVED"
                survivors.append(mutation)
            print(f"{label:12} {mutation.id:32} {verdict.why}", flush=True)

    if inconclusive:
        print(
            f"\n{len(inconclusive)} mutation(s) inconclusive — the suite did not finish, so "
            "nothing is proven about these:"
        )
        for mutation, why in inconclusive:
            print(f"  {mutation.id} ({mutation.path})\n    {why}")
    if survivors:
        print(f"\n{len(survivors)} mutation(s) survived — the suite does not check these:")
        for mutation in survivors:
            print(f"  {mutation.id} ({mutation.path})\n    if it shipped: {mutation.breaks}")
        return 1
    if inconclusive:
        return 2
    print(f"\nall {len(selected)} mutation(s) caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
