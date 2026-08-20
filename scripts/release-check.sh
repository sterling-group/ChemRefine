#!/usr/bin/env bash
#
# The local gate. Two modes:
#
#   scripts/release-check.sh --pr     everything CI gates that needs no special hardware
#   scripts/release-check.sh          the above, plus the tier-3 live suite (release)
#
# `--pr` is the one a contributor runs before opening a pull request: if it passes, CI
# will too, bar the things no workstation can do (below). It needs nothing but a `[dev]`
# install — no ORCA, no MLIP stack, no second interpreter.
#
# The default mode adds the half CI *cannot* run at all. GitHub-hosted runners have no
# ORCA and no MLIP stack, so the tier-3 suite — the only tests that drive real binaries
# end to end — never runs there. Automating it would mean a self-hosted runner, and
# ChemRefine is a public repository: a pull request from a fork can execute arbitrary code
# on it, which here would be a workstation holding an ORCA licence and the maintainer's
# keys. So this is a local gate, named in CONTRIBUTING.md as a required release step.
#
# The defects it exists to catch are the ones that pass every structural gate: a parser
# that reads real output wrongly, a step that returns the previous run's results. Coverage
# and mypy cannot see those; running the real thing can.
#
# What this deliberately does NOT replicate, because doing so costs more than it is worth:
#
#   · the dependency-floors job — it needs a Python 3.11, which most machines lack, and
#     CI runs it on every pull request as well as every tag;
#   · the 3.11-3.14 matrix, for the same reason;
#   · CodeQL, Scorecard and dependency-review, which are GitHub-hosted analyses with no
#     local equivalent.
#
# All four are covered on the tag regardless: `publish.yml` calls `ci.yml`, so nothing
# reaches PyPI without the full matrix having passed.
#
# Environment. Nothing machine-specific is baked in: the interpreter is whichever venv or
# conda env is active. Override with the variables below, or put them in an untracked
# `scripts/release-check.env` beside this file, which is sourced when present.
#
#   PY        the Python to gate with          (default: active venv/conda, else python3)
#   LIVE_PY   the Python for the tier-3 suite  (default: $PY — set it when the MLIP and
#             PySCF stacks live in an environment of their own)
#   ORCA      the ORCA binary                  (default: whatever is on PATH)

set -euo pipefail

mode="release"
case "${1:-}" in
    --pr)      mode="pr" ;;
    ""|--release) ;;
    -h|--help) sed -n '2,42p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         printf 'unknown option %s (try --pr, --release, --help)\n' "$1" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Machine-specific paths live here, outside the tracked script.
# shellcheck source=/dev/null
[ -f scripts/release-check.env ] && . scripts/release-check.env

fail() { printf '\n\033[31m✗ %s\033[0m\n' "$1" >&2; exit 1; }
step() { printf '\n\033[36m▸ %s\033[0m\n' "$1"; }

# -- resolve the interpreter ------------------------------------------------

if [ -z "${PY:-}" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        PY="$VIRTUAL_ENV/bin/python"
    elif [ -n "${CONDA_PREFIX:-}" ]; then
        PY="$CONDA_PREFIX/bin/python"
    else
        PY="$(command -v python3 || command -v python || true)"
    fi
fi
[ -x "$PY" ] || fail "no Python found — activate your environment, or set PY"
bin="$(dirname "$PY")"

# The gate must test *this* checkout, not some other copy that happens to be installed.
# `mutation_gate.py` proves the same thing for the same reason: a harness quietly checking
# the wrong source reports everything green and means nothing.
resolved="$("$PY" -c 'import chemrefine, pathlib; print(pathlib.Path(chemrefine.__file__).parent.parent)' 2>/dev/null || true)"
[ -n "$resolved" ] || fail "chemrefine is not importable by $PY — run: pip install -e '.[dev]'"
[ "$resolved" = "$repo_root/src" ] || fail "$PY imports chemrefine from $resolved, not $repo_root/src — run: pip install -e '.[dev]'"

# `pyproject-build` is the console script the `build` distribution ships (there is no
# `build` binary); it stands here for the `$PY -m build` further down. Without it that
# step failed several minutes in — after pre-commit, mypy, two whole suites and every
# extras resolve — on a tool this loop exists to catch in the first second.
for tool in pre-commit mypy pytest mkdocs pyproject-build; do
    [ -x "$bin/$tool" ] || fail "$tool is not in $bin — run: pip install -e '.[dev]'"
done

if [ -n "$(git status --porcelain)" ]; then
    fail "working tree is dirty — gate a clean tree"
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

printf 'repo    %s\n' "$repo_root"
printf 'python  %s (%s)\n' "$PY" "$("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
printf 'mode    %s\n' "$mode"

# -- the PR gate: every CI job that needs no special hardware ----------------

step "lint + format + docstrings (CI: pre-commit)"
# `pre-commit run --all-files`, not the three tools by hand: the hooks are what CI runs,
# and a hook added to .pre-commit-config.yaml would otherwise never reach this gate. The
# ruff hook carries --fix, so a violation is *written* and the run then fails on the
# modified file — which the clean-tree precondition above makes visible.
"$bin/pre-commit" run --all-files

step "types (CI: type-check)"
"$bin/mypy"

# The GUI's JavaScript is executed by exactly seven tests, and without a Node they *skip* —
# so on a machine with none this gate has been passing while checking the frontend not at
# all, the same hole the ORCA precondition below closes for tier-3. `[test]` now installs a
# Node, so a skip here means the lookup broke, not that the machine is bare. Exported once:
# the coverage run, the provisioned-backend run, the sdist run and the mutation gate all
# inherit it.
export CHEMREFINE_REQUIRE_NODE=1

step "the suite, with the coverage gate (CI: test)"
"$bin/pytest" --cov=chemrefine --cov-fail-under=100 -q

step "the suite is independent of provisioned backends (CI: provisioned-backend)"
# Runners are bare, so CI only ever exercises the not-provisioned half — while a developer
# who has run `chemrefine backends install` is in the other. Faking an env per extra
# reaches the same branch a real 3 GB torch install would, in a second. The registry is
# asked for the extras so a new backend is covered by existing.
fake_home="$work/fake-home"
for extra in $("$PY" -c \
        "from chemrefine.engines._provision import known_backend_extras as k; print(' '.join(sorted(k())))"); do
    mkdir -p "$fake_home/backends/$extra/bin"
    ln -sf "$PY" "$fake_home/backends/$extra/bin/python"
done
CHEMREFINE_HOME="$fake_home" "$bin/pytest" -q

step "every declared extra still resolves (CI: extras-resolve)"
# Metadata only — no torch wheel is downloaded. Read out of the metadata that declares
# them rather than restated here, which is how CI's own hand-written list came to be
# missing `mlff` and `pyscf-gpu`.
for extra in $("$PY" -c \
        "import tomllib;print(' '.join(sorted(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies'])))"); do
    printf '  [%s]\n' "$extra"
    "$bin/pip" install --disable-pip-version-check --dry-run --quiet ".[$extra]" >/dev/null
done

step "docs build (CI: docs-check)"
"$bin/mkdocs" build --strict

step "the package builds, validates, installs and runs (CI: build-package, install-smoke-test)"
rm -rf dist
"$PY" -m build --sdist --wheel >/dev/null

# The metadata half of CI's build-and-inspect. twine is not a project dependency, so it
# lives in a throwaway venv of its own.
"$PY" -m venv "$work/twine"
"$work/twine/bin/pip" install --disable-pip-version-check --quiet twine
"$work/twine/bin/twine" check dist/*

# The sdist is the second artifact PyPI serves and a distro packager builds from it, so it
# gets the same proof the wheel does. It used to get none: this script built it and then
# installed `dist/*.whl`, which never matched the tarball — and that is how `examples/`
# fell out of the sdist with every gate green (5246d7a).
"$PY" -m venv "$work/smoke"
"$work/smoke/bin/pip" install --disable-pip-version-check --quiet dist/*.whl
"$work/smoke/bin/chemrefine" --version
"$work/smoke/bin/chemrefine" --help > /dev/null
"$work/smoke/bin/python" - <<'PY'
import importlib.util, pathlib, sys
spec = importlib.util.find_spec("chemrefine")
marker = pathlib.Path(spec.submodule_search_locations[0]) / "py.typed"
sys.exit("py.typed missing from the wheel" if not marker.is_file() else 0)
PY
"$work/smoke/bin/pip" uninstall --disable-pip-version-check --quiet -y chemrefine
"$work/smoke/bin/pip" install --disable-pip-version-check --quiet --no-binary chemrefine dist/*.tar.gz
"$work/smoke/bin/chemrefine" --version
"$work/smoke/bin/chemrefine" --help > /dev/null

step "the suite passes from the unpacked sdist (CI: install-smoke-test)"
# The sdist ships tests and examples precisely so the suite can be run from it, and nothing
# else executes that suite — which is how docs/ and examples/ silently fell out of it. The
# repo-only guards skip themselves there with named reasons.
tar -xzf dist/*.tar.gz -C "$work"
( cd "$work"/chemrefine-*/ \
  && "$PY" -m venv "$work/sdist-venv" \
  && "$work/sdist-venv/bin/pip" install --disable-pip-version-check --quiet ".[test]" \
  && "$work/sdist-venv/bin/python" -m pytest -q -p no:cacheprovider )

step "critical predicates are checked, not just covered (CI: mutation-gate)"
# 100% branch coverage proves every line ran; it does not prove an assertion looked at the
# result. This breaks each predicate whose inversion would be a wrong scientific answer and
# requires a red test. It copies the tree to a scratch dir and never touches this one.
"$PY" scripts/mutation_gate.py

if [ "$mode" = "pr" ]; then
    printf '\n\033[32m✓ PR gate passed\033[0m\n'
    printf '  Still CI-only: the dependency floors, the 3.11-3.14 matrix, CodeQL,\n'
    printf '  Scorecard and dependency-review. All run on your pull request.\n'
    exit 0
fi

# -- the release half: what CI cannot run at all ----------------------------

LIVE_PY="${LIVE_PY:-$PY}"
[ -x "$LIVE_PY" ] || fail "LIVE_PY=$LIVE_PY is not executable"

ORCA="${ORCA:-$(command -v orca || true)}"
# `which orca` is not enough: desktop Linux ships /usr/bin/orca, the GNOME screen reader.
# A real ORCA keeps its helpers beside the main binary. Without this check the tier-3 suite
# skips every live case and the gate passes having tested nothing.
[ -n "$ORCA" ] || fail "ORCA is not on PATH, so the live tier would test nothing.
  For a pre-PR check that needs no ORCA, run: scripts/release-check.sh --pr"
orca_dir="$(cd "$(dirname "$(readlink -f "$ORCA")")" && pwd)"
[ -x "$orca_dir/orca_2json" ] || fail "$ORCA is not the quantum-chemistry ORCA (no orca_2json beside it)"
printf '\norca    %s\n' "$orca_dir"

step "known vulnerabilities (CI: security.yml)"
# A scheduled weekly sweep in CI, so between two Mondays a release can ship against an
# advisory nothing has looked at yet. --skip-editable: the local checkout is not on PyPI.
if [ -x "$bin/pip-audit" ]; then
    "$bin/pip-audit" --skip-editable
else
    fail "pip-audit is not in $bin — it ships with the dev extras, so this checkout is
  installed without them or predates them: pip install -e '.[dev]' (or use --pr)"
fi

# CHEMREFINE_REQUIRE_LIVE turns a missing backend into a failure instead of a skip. The
# ORCA precondition above exists because a silently skipped tier leaves this gate passing
# having tested nothing — but that reasoning covers every backend, not just ORCA, and
# pytest reports "N passed, M skipped" with exit 0 either way. With the variable set, a
# missing MACE or PySCF stack fails here by name.
#
# Memory discipline: the UMA single point alone peaks at 5.1 GB RSS (measured), and on a
# desktop this tier has OOM-killed VS Code instead of the test — the snap ships the editor
# with oom_score_adj=300, which makes it the kernel's *preferred* victim over any plain
# process. So: refuse to start without headroom, and run the tier inside a memory-capped
# scope sized below what is available, so a shortage kills the tier with this script's
# message rather than whatever the kernel fancies.
step "tier-3: the real binaries, end to end"
tier3=(env CHEMREFINE_REQUIRE_LIVE=1 PATH="$orca_dir:$PATH"
       "$(dirname "$LIVE_PY")/pytest" -m 'integration or perf' -q)
if command -v systemd-run >/dev/null 2>&1; then
    avail_g="$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)"
    if [ "$avail_g" -lt 7 ]; then
        fail "only ${avail_g}G of memory available — tier-3 needs ~7G free (the UMA load alone peaks at 5.1G); close some applications first"
    fi
    systemd-run --user --scope --collect --quiet \
        -p MemoryMax="$((avail_g - 1))G" -p MemorySwapMax=512M \
        "${tier3[@]}"
else
    "${tier3[@]}"
fi

version="$("$PY" -c 'import chemrefine; print(chemrefine.__version__)')"
printf '\n\033[32m✓ ready to tag v%s\033[0m\n' "$version"
printf '  Still CI-only: the dependency floors, the 3.11-3.14 matrix, CodeQL,\n'
printf '  Scorecard and dependency-review — all run on the tag, since publish.yml\n'
printf '  calls ci.yml before anything is published.\n'
printf '  remaining by hand: CHANGELOG Unreleased section retitled to %s\n' "$version"
