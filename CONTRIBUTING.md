# Contributing to ChemRefine

## Dev setup

```bash
git clone https://github.com/sterling-group/ChemRefine.git
cd ChemRefine
pip install -e ".[dev]"     # every tool the gates call: test + docs tooling,
                            # ruff, pre-commit, build, pip-audit
pre-commit install          # REQUIRED — CI runs these same hooks
```

`pre-commit install` is not optional, and the line above does not cover it:
that one installs the pre-commit *tool*, this one writes the hook into this
clone's `.git/hooks`, which is what makes it run on a commit. CI runs
`ruff check`, `ruff format` and `interrogate` through the pinned hooks, so a
clone that skips it drifts out of format and fails its first PR.

## The gates

One command runs every gate your pull request will face:

```bash
scripts/release-check.sh --pr
```

It needs nothing but a `[dev]` install — no ORCA, no MLIP stack — and gates
the checkout you are standing in, refusing to run against a `chemrefine`
installed anywhere else. If it passes, CI will too, apart from the four
things no workstation can do (named at the end of the run, and listed under
[Releases](#releases)).

For the inner loop, the individual gates are still the fastest way round:

```bash
pre-commit run --all-files                    # lint + format + docstring coverage
pytest --cov=chemrefine --cov-fail-under=100  # 100% coverage; live tier deselected
mypy                                          # type check (config in pyproject.toml)
mkdocs build --strict                         # docs build with no warnings
python scripts/mutation_gate.py               # critical predicates are *checked*, not just run
```

The coverage gate is real: new code ships with tests that cover every
line and branch, and every module/class/function carries a docstring
(`interrogate --fail-under=100`). The suite takes under a minute — run it often.
The mutation gate is the slow one (~1.5 min) because it runs a suite per mutation;
it only needs re-running when you touch one of the predicates it lists
(`python scripts/mutation_gate.py --list`).

**The GUI's JavaScript is checked with a Node, and `[test]` installs one.**
`tests/test_gui_assets.py` parses the builder's static assets and executes its
pure form logic — the one layer coverage cannot see, where a stray brace blanks
the whole page and an `@click` naming a deleted method fails silently. It uses
any `node` on `PATH` first and falls back to the one `nodejs-wheel-binaries`
ships, so there is nothing to install and nothing to remember.

Those seven cases used to **skip** without a Node, which meant a machine with
none ran every gate green while checking the frontend not at all. Setting
`CHEMREFINE_REQUIRE_NODE=1` turns that skip into a failure; `ci.yml` and
`scripts/release-check.sh` both set it, so a skip can no longer reach a release.

**Biome is the frontend's ruff.** `pre-commit` runs `biome check --write` over
`src/chemrefine/gui/static/`: it formats the JavaScript and CSS and lints all
three languages, with the settings and their reasons in `biome.jsonc`. It is
the one hook that is not `language: python`, so on a machine with no Node the
first `pre-commit` run downloads one into `~/.cache/pre-commit`. The vendored
bundles under `static/vendor/` are excluded — they must stay byte-identical to
what upstream published.

### The gates above do not cover the `integration` tier

`pytest` deselects `-m integration` by default and so does CI, which is deliberate — those
tests build 10⁴ structures or need real ORCA. But it means **a signature change can break one
of them and every gate above will still be green**: `mypy` will not catch it either, because
`attr-defined` is disabled for tests (see the rationale in `pyproject.toml`), so a call to a
function that no longer exists reads as ordinary noise.

If you change a public signature in `cache`, `step`, `lifecycle` or `nms`, run the tier that
uses it:

```bash
pytest -m perf tests/test_perf_cache.py   # ~12 s, no external binaries
```

That is its own marker because it is deselected for cost, not because it reaches for
anything: `integration` (`tests/test_e2e_live.py`) does need real ORCA / an MLIP stack, and
`slow` still runs by default. See the e2e section below.

### What 100% coverage does not prove

It proves every line ran. It does not prove an assertion looked at the result — a
predicate can be executed by every test in the suite and checked by none of them.
`scripts/mutation_gate.py` closes that gap for the predicates where it matters
most: the handful whose inversion produces a *wrong scientific answer* rather
than a crash, such as the NMS resolution test that decides whether a structure
reached the stationary point that was asked for. It breaks each one and requires
a red test. Add an entry when you write such a predicate; when one survives, the
fix is an assertion, not a smaller list.

Whole-package mutation testing is a different tool — thousands of mutants, mostly
log strings, and a survivor list to triage and then maintain. Use it occasionally
to *find* entries for that list; the gate is what keeps them.

Nor does coverage prove two components agree, and that is the other shape a
covered defect takes: a `device` knob whose options model defaults
to `cuda` while the scheduler reads the raw dict and books a CPU job; an
executable quoted where it is run but not where it is logged; a backend
requirement preflight accepts and `prepare` refuses. Each half passes its own
tests.

So a change that spans two modules lands with an **invariant** test, not only
a unit test — one that asserts the two readers agree, over every engine rather
than the one you touched. `tests/test_engines_invariants.py` is where they go;
the worked examples there are the device/GPU agreement, the `bash -n` check on
every generated run block, and the scan that fails if a knob a model declares
gets a second reader.

### If you change what a parser produces

Regenerate the end-to-end recordings. `tests/data/e2e/recordings/` stores what
the parsers produce today, and
`test_rebuilt_records_match_the_archived_ones_field_for_field` will fail when
that drifts. Rebuilding is parse-only — it re-derives the cached records from
the archived outputs already in the tree, so it needs no ORCA and no MLIP
stack. An archive that no longer matches the code is a fossil, not a fixture.

```bash
pytest 'tests/test_e2e_relocate.py::test_rebuilt_records_match_the_archived_ones_field_for_field' \
    --update-recordings
```

Review the re-packed archives in the git diff like any golden. This covers a
*parser* change only — a behavioural change (different jobs, different
survivors, a new operation) is not a re-parse, and needs the live tier:
`pytest -m integration --record` on a machine with the binaries.

## Conventions

- **Engines** are plugins. Read the
  [Adding an Engine recipe](https://sterling-group.github.io/ChemRefine/developer/adding-an-engine/)
  before adding one — it names the base class to subclass for each shape of
  backend, and the contract fixture every engine must ship.
- **Legacy YAML/CLI vocabulary** lives in exactly two quarantine zones:
  `config_legacy.py` (YAML keys) and `cli_legacy.py` (v1 flag-style argv).
  New legacy spellings go there, nowhere else. Both are scheduled for removal in
  3.0 — see `docs/get-started/upgrading-from-v1.md`.
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
  After a parser-only change, re-pack offline instead — no binaries needed:
  run the drift-detector test
  (`tests/test_e2e_relocate.py::test_rebuilt_records_match_the_archived_ones_field_for_field`)
  with `--update-recordings`.
- **Commits** are short, present-tense, and prefixed with a type from the fixed
  set (`build`, `chore`, `ci`, `docs`, `feat`, `fix`, `harden`, `perf`,
  `refactor`, `revert`, `style`, `test`) — the conventional-commits vocabulary
  plus `harden`, this repo's name for a change that closes a hole without
  altering behaviour. An area goes in parentheses, never in place of the type:
  `feat(gui):`, not `gui:`. Keep the subject under 72 characters.

## Landing a change

`main` is protected. Branch off it and name the branch for what it does —
`type/short-description`, with the same type vocabulary the commits use:
`fix/nms-imaginary-mode-count`, `docs/cluster-forwarding`. Open a PR and let
CI go green — the `required-checks-pass` job aggregates the required checks.

PRs are squash-merged, which has a consequence worth stating outright: **the
PR title becomes the one commit on `main`, and the messages on your branch
are discarded.** So the title carries the same `type(scope): subject` format
the commits do, and `.github/workflows/pr-title.yml` checks it there — on the
text that lands, rather than on the text that never does. Each merge
redeploys the docs and deletes the branch.

## GitHub Actions

If you add or edit a workflow step, pin it to a full commit SHA with the
release tag in a trailing comment:

```yaml
- uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
```

Tags are mutable — re-pointing `v1` at a malicious commit is how the
`tj-actions/changed-files` attack leaked CI secrets from every repo tracking a
tag — and two of the actions used here publish only a moving `release/v1` branch, so a SHA
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

That is the `--pr` gate above plus the two halves only a workstation has:
`pip-audit` (installed by `[dev]`), whose CI job is a weekly sweep rather
than a per-tag one, and the tier-3 suite against a real ORCA and the
managed MLIP/PySCF envs — the one thing here you install yourself. CI
cannot do that last part — GitHub-hosted runners have no ORCA, and
automating it would mean a self-hosted runner, which is unsafe on a public
repository: a pull request from a fork can execute arbitrary code on it.

Four CI jobs are deliberately **not** replicated locally, because doing so
would cost more than it covers: the dependency-floors job and the 3.11–3.14
matrix both need interpreters most machines lack, and CodeQL, Scorecard and
dependency-review are GitHub-hosted analyses with no local equivalent. All
four run on the tag anyway — `publish.yml` calls `ci.yml`, so nothing
reaches PyPI without the full matrix having passed.

Nothing machine-specific is baked into the script: it gates with whichever
venv or conda env is active. Put per-machine paths (`PY`, `LIVE_PY`, `ORCA`)
in an untracked `scripts/release-check.env` beside it — `LIVE_PY` is the one
to set when the MLIP and PySCF stacks live in an environment of their own.

That matters because the defects worth catching before a release are the
ones every structural gate passes. A parser that misreads real output, or a
step that returns the previous run's results, produces a plausible number
rather than an error; 100 % coverage and a clean mypy say nothing about it.
Running the real thing is the only check that does.

The script refuses to run if `orca` on PATH is not the quantum-chemistry
ORCA — desktop Linux ships `/usr/bin/orca`, the GNOME screen reader, and
without that check the live cases would skip and the gate would pass having
tested nothing.
