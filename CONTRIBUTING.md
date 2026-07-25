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

## Conventions worth knowing

- **Engines** are plugins. Read the
  [Adding an Engine recipe](docs/developer/adding-an-engine.md) before
  adding one — it names the base class to subclass for each shape of
  backend, and the contract fixture every engine must ship.
- **Legacy YAML/CLI vocabulary** lives in exactly two quarantine zones:
  the config normalizer (`config.py`) and `cli._translate_legacy_argv`.
  New legacy spellings go there, nowhere else.
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
