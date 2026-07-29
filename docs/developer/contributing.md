# Contributing

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

- **Engines** are plugins. Read [Adding an Engine](adding-an-engine.md) before
  adding one — it names the base class to subclass for each shape of backend,
  and the contract fixture every engine must ship.
- **Legacy YAML/CLI vocabulary** lives in exactly two quarantine zones:
  `config_legacy.py` (YAML keys) and `cli_legacy.py` (v1 flag-style argv).
  New legacy spellings go there, nowhere else. Both are scheduled for removal in
  3.0 — see [migrating-v1-to-v2](../migrating-v1-to-v2.md).
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

## Releases

Pushing a `vX.Y.Z` tag builds the package, creates a GitHub Release, and
publishes to PyPI; a `vX.Y.Z.devN` tag publishes to TestPyPI instead
(see `.github/workflows/publish.yml`). The version lives only in
`pyproject.toml`, and the publish jobs refuse a tag that doesn't match it.
Before tagging: bump the version, retitle the Unreleased section in
`CHANGELOG.md`, and start a fresh Unreleased section.
