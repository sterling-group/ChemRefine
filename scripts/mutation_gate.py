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


MUTATIONS = (
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
        old="if (fallback := (f.best if f.best is not None else prev_by_id.get(f.sid)))",
        new="if (fallback := prev_by_id.get(f.sid))",
        tests="tests/test_step.py",
        breaks="`on_failure: best` silently carries the submitted seed geometry downstream "
        "instead of the best geometry the failed run reached — ids and counts unchanged, "
        "so only a positions assertion can see it",
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
        id="cache-only-may-submit",
        path="src/chemrefine/step.py",
        old="case StepMode.CACHE_ONLY | StepMode.REBUILD:\n                return False",
        new="case StepMode.CACHE_ONLY | StepMode.REBUILD:\n                return True",
        tests="tests/test_e2e_relocate.py",
        breaks="`rebuild-cache` and `rerun-errors` submit work for steps they were not "
        "pointed at, archiving the very outputs they were asked to read",
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
        old='return (stored == "best") != (current == "best")',
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
)


#: Everything a pytest run reads. ``examples`` belongs here because the shipped tutorials
#: are living documentation and several tests resolve their templates; ``README.md`` and
#: ``docs`` because ``test_docs_examples`` validates every full config their prose shows.
#: The last four are ``test_docs_urls``'s: it reads the canonical ``site_url``/``repo_url``
#: out of ``mkdocs.yml`` at *import* time — so that one's absence was a collection error
#: rather than a failing test, and the whole gate refused to run — and it resolves every
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
    caught, why = _run_suite(work, env)
    if caught:
        raise SystemExit(
            f"mutation gate baseline is not green: {why}\n"
            f"The unmutated copy at {work} already fails, so every mutation would look\n"
            f"'caught' regardless. Fix the copy (see _INPUTS) before trusting a verdict."
        )


def stale_anchors(root: Path, mutations: Sequence[Mutation]) -> list[str]:
    """One report line per mutation whose ``old`` is not unique, or whose ``tests`` is gone.

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
        source = root / mutation.path
        if not source.is_file():
            # The same report line the tests half gets: a `git mv` of a mutated source
            # file otherwise raised FileNotFoundError out of main() — the one moved file
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


def _apply(work: Path, mutation: Mutation) -> None:
    """Rewrite the single occurrence of ``old`` in the copy.

    Uniqueness is :func:`stale_anchors`' job, proven for every selected mutation before
    this runs — so the rule lives in one place rather than in two that could drift.
    """
    target = work / mutation.path
    text = target.read_text(encoding="utf-8")
    target.write_text(text.replace(mutation.old, mutation.new), encoding="utf-8")


def _run_suite(work: Path, env: dict[str, str], target: str | None = None) -> tuple[bool, str]:
    """Return ``(caught, why)`` for the suite — or for one file of it — as it stands in ``work``.

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
    with tempfile.TemporaryDirectory(prefix="chemrefine-mutation-") as tmp:
        work = Path(tmp)
        _copy_tree(work)
        env = {**os.environ, "PYTHONPATH": str(work / "src")}
        _assert_isolated(work, env)
        _assert_baseline_is_green(work, env)
        print(f"baseline green; {len(selected)} mutation(s) to check\n", flush=True)

        for mutation in selected:
            pristine = (work / mutation.path).read_text(encoding="utf-8")
            _apply(work, mutation)
            try:
                caught, why = _run_suite(work, env, mutation.tests)
                if not caught:
                    # The named file missed it. Before calling anything a survivor, ask the
                    # whole suite — the entry may simply name the wrong file, and `why` then
                    # reports the test that did catch it.
                    caught, why = _run_suite(work, env)
            finally:
                (work / mutation.path).write_text(pristine, encoding="utf-8")
            print(f"{'caught ' if caught else 'SURVIVED'}  {mutation.id:32} {why}", flush=True)
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
