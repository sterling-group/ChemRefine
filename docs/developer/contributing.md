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
pytest --cov=chemrefine --cov-fail-under=100  # 100% line AND branch coverage
mypy                                          # type check (config in pyproject.toml)
mkdocs build --strict                         # docs build with no warnings
```

The coverage gate is real: new code ships with tests that cover every
line and branch, and every module/class/function carries a docstring
(`interrogate --fail-under=100`). The suite is fast (< 10 s) — run it often.

## Conventions worth knowing

- **Engines** are plugins. Read [Adding an Engine](adding-an-engine.md) before
  adding one — it names the base class to subclass for each shape of backend.
- **Legacy YAML/CLI vocabulary** lives in exactly two quarantine zones:
  the config normalizer (`config.py`) and `cli._translate_legacy_argv`.
  New legacy spellings go there, nowhere else.
- **Tests** mirror `src/` one file per module (`tests/test_<module>.py`,
  `tests/test_engines_<name>*.py`); shared synthetic ORCA snippets live
  in `tests/synthetic.py`, real trimmed fixtures in `tests/data/`.
- **Commits** are short, present-tense, and prefixed
  (`feat:`/`fix:`/`refactor:`/`docs:`/`ci:`/`test:`/`harden:`), matching `git log`.

## Releases

Pushing a `vX.Y.Z` tag builds the package, creates a GitHub Release, and
publishes to PyPI; a `vX.Y.Z.devN` tag publishes to TestPyPI instead
(see `.github/workflows/publish.yml`). The version lives only in
`pyproject.toml`, and the publish jobs refuse a tag that doesn't match it.
Before tagging: bump the version, retitle the Unreleased section in
`CHANGELOG.md`, and start a fresh Unreleased section.
