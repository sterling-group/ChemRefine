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
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: How long one mutated run may take before it is assumed hung. A mutation that turns a
#: budget comparison into a strictly-smaller one makes a wait loop spin forever rather than
#: fail, so a timeout counts as *caught*: the suite did not complete, which is a red build.
TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class Mutation:
    """One deliberate break, and the wrong answer it would produce if it survived."""

    id: str
    path: str
    old: str
    new: str
    breaks: str
    """What a user would get if this shipped — the reason the entry is on the list."""


MUTATIONS = (
    Mutation(
        id="nms-is-resolved",
        path="src/chemrefine/nms.py",
        old="len(child.imaginary_freqs) == target",
        new="len(child.imaginary_freqs) <= target",
        breaks="a `ts` step accepts a child that fell into a minimum and promotes it to "
        "the parent's id — the transition state replaced by the wrong stationary point",
    ),
    Mutation(
        id="nms-already-at-target",
        path="src/chemrefine/nms.py",
        old="len(structure.imaginary_freqs) == target",
        new="len(structure.imaginary_freqs) <= target",
        breaks="a minimum passes through a `ts` step as though already resolved, so it is "
        "never displaced and never reaches the saddle point",
    ),
    Mutation(
        id="boltzmann-cutoff",
        path="src/chemrefine/filtering.py",
        old="sorted_structures[: n_below + 1]",
        new="sorted_structures[:n_below]",
        breaks="the structure that crosses the cumulative-weight threshold is dropped, so "
        "a 99% filter silently keeps less than it promises",
    ),
    Mutation(
        id="min-window-boundary",
        path="src/chemrefine/filtering.py",
        old="if cast(float, getattr(s, energy_attr)) <= min_e + window_h",
        new="if cast(float, getattr(s, energy_attr)) < min_e + window_h",
        breaks="a structure exactly on the window edge is silently dropped, so the filter "
        "keeps less than it promises",
    ),
    Mutation(
        id="max-window-boundary",
        path="src/chemrefine/filtering.py",
        old=">= max_e - window_h]",
        new="> max_e - window_h]",
        breaks="a structure exactly on the window edge is silently dropped, so the filter "
        "keeps less than it promises — the high-energy mirror of min-window-boundary",
    ),
    Mutation(
        id="retry-best-frame",
        path="src/chemrefine/lifecycle.py",
        old="best = min(bad, key=lambda s: (s.energy_hartree is None, s.energy_hartree or 0.0))",
        new="best = max(bad, key=lambda s: (s.energy_hartree is None, s.energy_hartree or 0.0))",
        breaks="a fan-out job's retry restarts from the worst unconverged frame, and "
        "`on_failure: best` backfills the worst geometry obtained",
    ),
    Mutation(
        id="succeeded-ignores-convergence",
        path="src/chemrefine/lifecycle.py",
        old="return s.terminated_normally is not False and s.converged is not False",
        new="return s.terminated_normally is not False",
        breaks="an unconverged structure counts as a success, so it is never retried, "
        "never ledgered, and ranks against converged siblings",
    ),
    Mutation(
        id="parents-digest-ignores-geometry",
        path="src/chemrefine/cache.py",
        old="h.update(np.asarray(s.atoms.get_positions(), dtype=np.float64).tobytes())",
        new="pass",
        breaks="editing the seed geometry no longer invalidates the cache, so `resume` "
        "serves results computed from the old coordinates",
    ),
    Mutation(
        id="cache-pair-is-checked",
        path="src/chemrefine/cache.py",
        old='if found != document["arrays_digest"]:',
        new="if False:",
        breaks="a `step.json` left by one save is read with the `arrays.npz` of another, so "
        "each structure keeps its own energy and adopts a different structure's geometry",
    ),
    Mutation(
        id="array-task-id-matching",
        path="src/chemrefine/slurm/dispatch.py",
        old='mine = {line for line in running if line == jid or line.startswith(f"{jid}_")}',
        new="mine = {line for line in running if line == jid}",
        breaks="a running array job reports finished — squeue prints its tasks as "
        "`12345_0`, never the bare parent id — so its outputs are parsed while it writes",
    ),
    Mutation(
        id="cache-only-may-submit",
        path="src/chemrefine/step.py",
        old="case StepMode.CACHE_ONLY | StepMode.REBUILD:\n                return False",
        new="case StepMode.CACHE_ONLY | StepMode.REBUILD:\n                return True",
        breaks="`rebuild-cache` and `rerun-errors` submit work for steps they were not "
        "pointed at, archiving the very outputs they were asked to read",
    ),
    Mutation(
        id="rebuild-cache-provenance",
        path="src/chemrefine/step.py",
        old="if stamped and stamped != key.fingerprint:",
        new="if False:",
        breaks="`rebuild-cache` caches results under a configuration that never produced "
        "them, and the next `resume` serves that instead of computing what was asked for",
    ),
    Mutation(
        id="policy-change-over-cache",
        path="src/chemrefine/step.py",
        old='return (stored == "best") != (current == "best")',
        new="return False",
        breaks="editing on_failure over a cached step silently serves the previous "
        "policy's survivor set — a step switched to `best` quietly behaves as `skip`, "
        "and one switched away from `best` keeps carrying its backfills",
    ),
    Mutation(
        id="dead-job-vs-unreadable-file",
        path="src/chemrefine/engines/orca/output/coordinator.py",
        old="if status.parse_terminated_normally(text):\n        return OutputParseError",
        new="if True:\n        return OutputParseError",
        breaks="a job that died is ledgered as an unreadable file, sending a reader to the "
        "parser for a cluster or input problem",
    ),
    Mutation(
        id="qiskit-particle-sector-filter",
        path="src/chemrefine/engines/qiskit/components/algorithms.py",
        old="not np.isclose(particle_number[0], expected_particles)",
        new="np.isclose(particle_number[0], expected_particles)",
        breaks="the exact solver discards eigenstates with the requested electron count and "
        "can report a plausible eigenvalue from the wrong particle-number sector",
    ),
    Mutation(
        id="qiskit-spin-sector-filter",
        path="src/chemrefine/engines/qiskit/components/algorithms.py",
        old="np.isclose(angular_momentum[0], expected_angular_momentum)",
        new="not np.isclose(angular_momentum[0], expected_angular_momentum)",
        breaks="the exact solver rejects the requested spin sector and can report a state "
        "with the wrong multiplicity",
    ),
    Mutation(
        id="qiskit-aer-noise-is-applied",
        path="src/chemrefine/engines/qiskit/components/estimators.py",
        old="if noise_model is not None:",
        new="if noise_model is None:",
        breaks="an explicitly requested Aer noise model is silently ignored, producing an "
        "ideal-simulator energy that looks valid but models a different experiment",
    ),
    Mutation(
        id="finite-energy-guard",
        path="src/chemrefine/engines/_script/output.py",
        old="if not np.isfinite(number):",
        new="if False:",
        breaks="a diverged calculation's NaN energy ranks as a real result and, because "
        "every NaN comparison is false, displaces a genuine survivor by list position",
    ),
)


#: Everything a pytest run reads. ``examples`` belongs here because the shipped tutorials
#: are living documentation and several tests resolve their templates; ``README.md`` and
#: ``docs`` because ``test_docs_examples`` validates every full config their prose shows.
#: An incomplete copy fails on its own, which :func:`_assert_baseline_is_green` catches
#: whatever the missing input turns out to be — it is how these two earned their entries.
_INPUTS = ("src", "tests", "examples", "docs", "README.md", "pyproject.toml")


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
    caught, why = _run_suite(work, env)
    if caught:
        raise SystemExit(
            f"mutation gate baseline is not green: {why}\n"
            f"The unmutated copy at {work} already fails, so every mutation would look\n"
            f"'caught' regardless. Fix the copy (see _INPUTS) before trusting a verdict."
        )


def stale_anchors(root: Path, mutations: Sequence[Mutation]) -> list[str]:
    """One report line per mutation whose ``old`` is not unique under ``root``.

    Every anchor is checked before any is applied, and every stale one is reported
    together. Failing on the first made a single moved line hide the state of the whole
    gate: the run stopped there, the mutations after it never executed, and the exit code
    said "gate failed" without saying that most of it had not run.

    Checked against the repository rather than the scratch copy, so a stale anchor costs
    milliseconds instead of a tree copy plus a baseline suite. Public because
    ``tests/test_mutation_gate.py`` asks the same question in the edit loop — which is
    where the developer who moves a line will actually see the answer.
    """
    problems: list[str] = []
    for mutation in mutations:
        found = (root / mutation.path).read_text(encoding="utf-8").count(mutation.old)
        if found != 1:
            problems.append(
                f"[{mutation.id}] expected exactly one occurrence of\n    {mutation.old}\n"
                f"in {mutation.path}, found {found}."
            )
    return problems


def _apply(work: Path, mutation: Mutation) -> None:
    """Rewrite the single occurrence of ``old`` in the copy.

    Uniqueness is :func:`stale_anchors`' job, proven for every selected mutation before
    this runs — so the rule lives in one place rather than in two that could drift.
    """
    target = work / mutation.path
    text = target.read_text(encoding="utf-8")
    target.write_text(text.replace(mutation.old, mutation.new), encoding="utf-8")


def _run_suite(work: Path, env: dict[str, str]) -> tuple[bool, str]:
    """Return ``(caught, why)`` for the suite as it stands in ``work``.

    ``-x`` so a caught mutation stops at the first red test: the healthy case is fast, and
    only a survivor pays for the whole suite.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
            capture_output=True,
            text=True,
            cwd=work,
            env=env,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return True, f"suite hung (>{TIMEOUT_SECONDS}s) — a red build either way"
    if completed.returncode != 0:
        first = next(
            (ln for ln in completed.stdout.splitlines() if ln.startswith("FAILED")),
            "suite failed",
        )
        return True, first
    return False, "suite passed unchanged"


def main(argv: list[str] | None = None) -> int:
    """Run every selected mutation against a scratch copy; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-k", dest="pattern", default="", help="only ids containing this")
    parser.add_argument("--list", action="store_true", help="list the mutations and exit")
    args = parser.parse_args(argv)

    selected = [m for m in MUTATIONS if args.pattern in m.id]
    if args.list:
        for mutation in selected:
            print(f"{mutation.id:32} {mutation.path}\n{'':32} if it survived: {mutation.breaks}")
        return 0
    if not selected:
        raise SystemExit(f"no mutation id contains {args.pattern!r}")
    if stale := stale_anchors(REPO, selected):
        raise SystemExit("\n".join([*stale, "The code moved; update the mutation(s)."]))

    survivors: list[Mutation] = []
    with tempfile.TemporaryDirectory(prefix="chemrefine-mutation-") as tmp:
        work = Path(tmp)
        _copy_tree(work)
        env = {**os.environ, "PYTHONPATH": str(work / "src")}
        _assert_isolated(work, env)
        _assert_baseline_is_green(work, env)
        print(f"baseline green; {len(selected)} mutation(s) to check\n")

        for mutation in selected:
            pristine = (work / mutation.path).read_text(encoding="utf-8")
            _apply(work, mutation)
            try:
                caught, why = _run_suite(work, env)
            finally:
                (work / mutation.path).write_text(pristine, encoding="utf-8")
            print(f"{'caught ' if caught else 'SURVIVED'}  {mutation.id:32} {why}")
            if not caught:
                survivors.append(mutation)

    if survivors:
        print(f"\n{len(survivors)} mutation(s) survived — the suite does not check these:")
        for mutation in survivors:
            print(f"  {mutation.id} ({mutation.path})\n    if it shipped: {mutation.breaks}")
        return 1
    print(f"\nall {len(selected)} mutation(s) caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
