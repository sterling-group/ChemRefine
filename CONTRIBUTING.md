# Contributing to ChemRefine

## Dev setup

```bash
git clone https://github.com/sterling-group/ChemRefine.git
cd ChemRefine
pip install -e ".[dev]"     # test + docs tooling, ruff, pre-commit
pre-commit install          # REQUIRED — CI runs these same hooks
```

`pre-commit install` is not optional: CI runs `ruff check`, `ruff format`,
and `interrogate` through the pinned pre-commit hooks, so a clone that
skips the hook install drifts out of format and fails its first PR.

## The gates

Every PR must pass all of these — run them locally before pushing:

```bash
pre-commit run --all-files                    # lint + format + docstring coverage
pytest --cov=chemrefine --cov-fail-under=100  # 100% coverage; live tier deselected
mypy                                          # type check (config in pyproject.toml)
mkdocs build --strict                         # docs build with no warnings
```

The coverage gate is real: new code ships with tests that cover every
line and branch, and every module/class/function carries a docstring
(`interrogate --fail-under=100`). The suite is fast (< 10 s) — run it often.

### What 100% coverage does not prove

It proves every line ran. It does not prove two components agree, and that
is where this project's defects have actually lived: a `device` knob whose
options model defaulted to `cuda` while the scheduler read the raw dict and
booked a CPU job; an executable quoted where it was run but not where it was
logged; a backend requirement that preflight accepted and `prepare` refused.
Every one of those sat in fully covered code.

So a change that spans two modules lands with an **invariant** test, not only
a unit test — one that asserts the two readers agree, over every engine rather
than the one you touched. `tests/test_engines_invariants.py` is where they go;
the worked examples there are the device/GPU agreement, the `bash -n` check on
every generated run block, and the scan that fails if a knob a model declares
gets a second reader.

### If you change what a parser produces

Regenerate the end-to-end recordings. `tests/data/e2e/recordings/` stores what
the parsers *used to* produce, and
`test_rebuilt_records_match_the_archived_ones_field_for_field` will fail when
that drifts. Rebuilding is parse-only — it re-derives the cached records from
the archived outputs already in the tree, so it needs no ORCA and no MLIP
stack. An archive that no longer matches the code is a fossil, not a fixture.

## Conventions worth knowing

- **Engines** are plugins. Read the
  [Adding an Engine recipe](docs/developer/adding-an-engine.md) before
  adding one — it names the base class to subclass for each shape of
  backend, and the contract fixture every engine must ship.
- **Legacy YAML/CLI vocabulary** lives in exactly two quarantine zones:
  `config_legacy.py` (YAML keys) and `cli_legacy.py` (v1 flag-style argv).
  New legacy spellings go there, nowhere else. Both are scheduled for removal in
  3.0 — see `docs/migrating-v1-to-v2.md`.
- **Tests are tiered.** Unit tests mirror `src/` one file per module
  (`tests/test_<module>.py`, `tests/test_engines_<name>*.py`); shared
  synthetic ORCA snippets live in `tests/synthetic.py`. The recorded
  end-to-end tier replays real outputs with no binaries installed:
  per-engine parse contracts under `tests/data/engines/` (every
  registered engine must ship one; regenerate goldens with
  `pytest tests/test_engines_contract.py --update-goldens`) and workflow
  recordings under `tests/data/e2e/recordings/`. The live tier
  (`pytest -m integration`, deselected by default) runs the case
  definitions in `tests/data/e2e/cases/` against real ORCA/MLIP
  binaries; add `--record` to re-pack the recordings from a passing run.
- **Commits** are short, present-tense, and prefixed
  (`feat:`/`fix:`/`refactor:`/`docs:`/`ci:`/`test:`/`harden:`), matching `git log`.
  No `Co-Authored-By:` trailers and no generated-by/AI attribution footers —
  `git log` is clean of them today, keep it that way.

## Landing a change

`main` is protected: branch off it, open a PR, and let CI go green — the
`required-checks-pass` job aggregates the required checks. PRs are
squash-merged, so one PR is one commit on `main`, and each merge redeploys
the docs.

## GitHub Actions

If you add or edit a workflow step, pin it to a full commit SHA with the
release tag in a trailing comment:

```yaml
- uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
```

Tags are mutable — re-pointing `v1` at a malicious commit is how the
`tj-actions/changed-files` attack leaked CI secrets from every repo tracking a
tag — and two actions we use publish only a moving `release/v1` branch, so a SHA
is the only way to pin them at all. Dependabot reads the trailing comment and
bumps both, so pinning costs nothing in freshness. Resolve a tag with:

```bash
sha=$(gh api repos/$REPO/git/ref/tags/$TAG --jq '.object.sha')
gh api repos/$REPO/git/tags/$sha --jq '.object.sha' 2>/dev/null || echo "$sha"
```

Set `persist-credentials: false` on `actions/checkout` unless the job actually
pushes, so the token isn't left behind in `.git/config`.

## Releases

Pushing a `vX.Y.Z` tag builds the package, creates a GitHub Release, and
publishes to PyPI; a `vX.Y.Z.devN` tag publishes to TestPyPI instead
(see `.github/workflows/publish.yml`). The version lives only in
`pyproject.toml`, and the publish jobs refuse a tag that doesn't match it.
Before tagging: bump the version, retitle the Unreleased section in
`CHANGELOG.md`, and start a fresh Unreleased section.

**Then run the release gate — this step is required:**

```bash
scripts/release-check.sh
```

It runs everything CI runs, then installs the built wheel into a throwaway
venv and runs the tier-3 suite against a real ORCA. CI cannot do that last
part: GitHub-hosted runners have no ORCA, and automating it would mean a
self-hosted runner, which is unsafe on a public repository — a pull request
from a fork can execute arbitrary code on it.

That matters because the defects worth catching before a release are the
ones every structural gate passes. A parser that misreads real output, or a
step that returns the previous run's results, produces a plausible number
rather than an error; 100 % coverage and a clean mypy say nothing about it.
Running the real thing is the only check that does.

The script refuses to run if `orca` on PATH is not the quantum-chemistry
ORCA — desktop Linux ships `/usr/bin/orca`, the GNOME screen reader, and
without that check the live cases would skip and the gate would pass having
tested nothing.
