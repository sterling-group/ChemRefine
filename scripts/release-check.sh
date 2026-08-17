#!/usr/bin/env bash
#
# Everything that can be checked on this machine, run in one command before tagging.
#
# Two halves. The first mirrors the CI gate job for job, so a failure is found before a
# tag exists rather than after — `publish.yml` calls `ci.yml`, so a tag does run the full
# matrix, but discovering `floors` or the mutation gate red at that point means an
# abandoned tag. The second is the part CI *cannot* run at all: GitHub-hosted runners have
# no ORCA and no MLIP stack, so the tier-3 suite — the only tests that drive real binaries
# end to end — never runs there. Automating it would mean a self-hosted runner, and
# ChemRefine is a public repository: a pull request from a fork can execute arbitrary code
# on it, which here would be a workstation holding an ORCA licence and the maintainer's
# keys. So this is a local gate, named in CONTRIBUTING.md as a required step.
#
# The defects this exists to catch are the ones that pass every structural gate: a parser
# that reads real output wrongly, a step that returns the previous run's results. Coverage
# and mypy cannot see those; running the real thing can.
#
# **Nothing is skipped silently.** A step whose prerequisite is missing is recorded and
# reprinted at the end, because a gate that quietly shrinks is worse than no gate — the
# ORCA precondition below exists for exactly that reason. Where a prerequisite can be
# detected (a 3.11 interpreter for the dependency floors), the step runs the moment it
# appears rather than staying off.
#
# Usage:  scripts/release-check.sh
# Env:    DEV_ENV / E2E_ENV  — override the conda prefixes (defaults below)
#         ORCA               — override the ORCA binary to test against
#         FLOORS_PY          — a Python 3.11 for the dependency-floors job

set -euo pipefail

DEV_ENV="${DEV_ENV:-$HOME/opt/envs/chemrefine-dev}"
E2E_ENV="${E2E_ENV:-$HOME/opt/envs/chemrefine-e2e}"
ORCA="${ORCA:-$(command -v orca || true)}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fail() { printf '\n\033[31m✗ %s\033[0m\n' "$1" >&2; exit 1; }
step() { printf '\n\033[36m▸ %s\033[0m\n' "$1"; }

# Steps that could not run, with the reason. Reprinted at the end — see the header.
skipped=()
skip() { printf '\033[33m  skipped: %s\033[0m\n' "$1"; skipped+=("$1"); }

# -- preconditions ----------------------------------------------------------

[ -x "$DEV_ENV/bin/python" ] || fail "no dev env at $DEV_ENV (set DEV_ENV)"
[ -x "$E2E_ENV/bin/python" ] || fail "no e2e env at $E2E_ENV (set E2E_ENV)"

# `which orca` is not enough: desktop Linux ships /usr/bin/orca, the GNOME screen reader.
# A real ORCA keeps its helpers beside the main binary. Without this check the tier-3
# suite skips every live case and the gate passes having tested nothing.
[ -n "$ORCA" ] || fail "ORCA is not on PATH — the live suite would skip silently"
orca_dir="$(cd "$(dirname "$(readlink -f "$ORCA")")" && pwd)"
[ -x "$orca_dir/orca_2json" ] || fail "$ORCA is not the quantum-chemistry ORCA (no orca_2json beside it)"

if [ -n "$(git status --porcelain)" ]; then
    fail "working tree is dirty — release from a clean tree"
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

printf 'repo    %s\n' "$repo_root"
printf 'orca    %s\n' "$orca_dir"
printf 'version %s\n' "$("$DEV_ENV/bin/python" -c 'import chemrefine; print(chemrefine.__version__)')"

# -- the gate, cheapest first so a failure surfaces early --------------------

step "lint + format + docstrings (the pre-commit job)"
# `pre-commit run --all-files`, not the three tools by hand: the hooks are what CI runs,
# and a hook added to .pre-commit-config.yaml would otherwise never reach this gate. The
# ruff hook carries --fix, so a violation is *written* and the run then fails on the
# modified file — which is what the clean-tree precondition above makes visible.
"$DEV_ENV/bin/pre-commit" run --all-files

step "types"
"$DEV_ENV/bin/mypy"

step "unit + contract + replay, with the coverage gate"
"$DEV_ENV/bin/pytest" --cov=chemrefine --cov-fail-under=100 -q

step "the suite on a second interpreter"
# CI runs 3.11-3.14; a workstation has whichever it has. Running the e2e env's Python as
# well is the difference between "passes on one interpreter" and "passes on two", and it
# costs a minute. The versions are printed so the summary says what was actually covered.
dev_py="$("$DEV_ENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
e2e_py="$("$E2E_ENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$dev_py" = "$e2e_py" ]; then
    skip "second interpreter: dev and e2e are both $dev_py (CI covers 3.11-3.14)"
else
    printf '  dev %s, e2e %s\n' "$dev_py" "$e2e_py"
    "$E2E_ENV/bin/pytest" -q
fi

step "the suite is independent of provisioned backends"
# CI's provisioned-backend job. Runners are bare, so the default matrix only ever exercises
# the not-provisioned half — while a developer who has run `chemrefine backends install` is
# in the other. The registry is asked for the extras so a new backend is covered by
# existing rather than by remembering to edit this.
fake_home="$work/fake-home"
for extra in $("$DEV_ENV/bin/python" -c \
        "from chemrefine.engines._provision import known_backend_extras as k; print(' '.join(sorted(k())))"); do
    mkdir -p "$fake_home/backends/$extra/bin"
    ln -sf "$DEV_ENV/bin/python" "$fake_home/backends/$extra/bin/python"
done
CHEMREFINE_HOME="$fake_home" "$DEV_ENV/bin/pytest" -q

step "every declared extra still resolves"
# CI's extras-resolve. Metadata only — no torch wheel is downloaded. Read out of the
# metadata that declares them rather than restated here, which is how CI's own hand-written
# list came to be missing `mlff` and `pyscf-gpu`.
extras="$("$DEV_ENV/bin/python" -c \
    "import tomllib;print(' '.join(sorted(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies'])))")"
for extra in $extras; do
    printf '  [%s]\n' "$extra"
    "$DEV_ENV/bin/pip" install --disable-pip-version-check --dry-run --quiet ".[$extra]" >/dev/null
done

step "the dependency floors still run"
# CI's floors job. Every other step tests whatever pip picks today — always the newest
# allowed — so a floor could sit years behind the first version the code actually needs.
# The floors bind on the oldest supported Python, so this needs a 3.11 and says so when
# there is none rather than pretending on 3.13.
floors_py="${FLOORS_PY:-$(command -v python3.11 || true)}"
if [ -z "$floors_py" ]; then
    skip "dependency floors: no Python 3.11 on this machine (set FLOORS_PY; CI's floors job covers it)"
else
    "$floors_py" -m venv "$work/floors"
    "$DEV_ENV/bin/python" - > "$work/floors.txt" <<'PY'
import re, tomllib
deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]
print("\n".join(f"{m.group(1)}=={m.group(2)}"
                for dep in deps
                if (m := re.match(r"^([A-Za-z0-9_.-]+)\s*>=\s*([^,;]+?)\s*$", dep))))
PY
    "$work/floors/bin/pip" install --disable-pip-version-check --quiet -e ".[test,mcp,gui,agent]" -c "$work/floors.txt"
    "$work/floors/bin/pytest" -q
    # CI additionally asserts the mlip-orb marker excludes orb-models on 3.11 — without it
    # pip backtracks through every orb release instead of saying "requires a different
    # Python", and nothing else would notice the marker regressing.
    "$work/floors/bin/pip" install --disable-pip-version-check --dry-run --quiet --report "$work/orb.json" ".[mlip-orb]" >/dev/null
    "$DEV_ENV/bin/python" - "$work/orb.json" <<'PY'
import json, sys
names = {p["metadata"]["name"].lower() for p in json.load(open(sys.argv[1]))["install"]}
sys.exit("orb-models resolved on 3.11 — the version marker is not holding"
         if "orb-models" in names else 0)
PY
fi

step "docs build"
"$DEV_ENV/bin/mkdocs" build --strict

step "package builds, validates, installs, and runs — wheel *and* sdist"
rm -rf dist
"$DEV_ENV/bin/python" -m build --sdist --wheel >/dev/null

# The metadata half of CI's build-and-inspect. twine is not a project dependency, so it
# lives in a throwaway venv of its own rather than in the dev env.
"$DEV_ENV/bin/python" -m venv "$work/twine"
"$work/twine/bin/pip" install --disable-pip-version-check --quiet twine
"$work/twine/bin/twine" check dist/*

# The sdist is the second artifact PyPI serves and a distro packager builds from it, so it
# gets the same proof the wheel does. It used to get none: this script built it and then
# installed `dist/*.whl`, which the glob never matched to the tarball — which is how
# `examples/` fell out of the sdist while every gate stayed green (5246d7a).
"$DEV_ENV/bin/python" -m venv "$work/smoke"
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

step "the suite passes from the unpacked sdist"
# The sdist ships tests and examples precisely so the suite can be run from it, and nothing
# else executes that suite — which is exactly how docs/ and examples/ silently fell out of
# it. The repo-only guards skip themselves there with named reasons.
tar -xzf dist/*.tar.gz -C "$work"
( cd "$work"/chemrefine-*/ \
  && "$DEV_ENV/bin/python" -m venv "$work/sdist-venv" \
  && "$work/sdist-venv/bin/pip" install --disable-pip-version-check --quiet ".[test,mcp,gui,agent]" \
  && "$work/sdist-venv/bin/python" -m pytest -q -p no:cacheprovider )

step "known vulnerabilities"
# security.yml's pip-audit, which is a scheduled sweep — so between two Mondays a release
# can ship against an advisory nothing has looked at yet. --skip-editable: the local
# checkout is not on PyPI.
if [ -x "$DEV_ENV/bin/pip-audit" ]; then
    "$DEV_ENV/bin/pip-audit" --skip-editable
else
    skip "pip-audit: not installed in $DEV_ENV (pip install pip-audit)"
fi

step "critical predicates are checked, not just covered"
# CI's mutation-gate. 100% branch coverage proves every line ran; it does not prove an
# assertion looked at the result. This breaks each predicate whose inversion would be a
# wrong scientific answer and requires a red test. It copies the tree to a scratch dir and
# never touches the working one.
"$DEV_ENV/bin/python" scripts/mutation_gate.py

# CHEMREFINE_REQUIRE_LIVE turns a missing backend into a failure instead of a skip. The
# ORCA precondition above exists because a silently skipped tier leaves this gate passing
# having tested nothing — but that reasoning covers every backend, not just ORCA, and
# pytest reports "N passed, M skipped" with exit 0 either way. With the variable set, a
# missing MACE or PySCF stack fails here by name.
#
# Memory discipline: the UMA single point alone peaks at 5.1 GB RSS (measured), and on a
# desktop this tier has OOM-killed VS Code instead of the test — the snap ships the
# editor with oom_score_adj=300, which makes it the kernel's *preferred* victim over any
# plain process. So: refuse to start without headroom, and run the tier inside a
# memory-capped scope sized below what is available, so a shortage kills the tier with
# this script's message rather than whatever the kernel fancies.
step "tier-3: the real binaries, end to end"
tier3=(env CHEMREFINE_REQUIRE_LIVE=1 PATH="$orca_dir:$PATH"
       "$E2E_ENV/bin/pytest" -m 'integration or perf' -q)
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

# -- what ran, and what did not ---------------------------------------------

step "the tag will match the version"
version="$("$DEV_ENV/bin/python" -c 'import chemrefine; print(chemrefine.__version__)')"

if [ "${#skipped[@]}" -gt 0 ]; then
    printf '\n\033[33m%d step(s) could not run here:\033[0m\n' "${#skipped[@]}"
    printf '  · %s\n' "${skipped[@]}"
    printf '  These are covered by CI on the tag — publish.yml calls ci.yml — but they are\n'
    printf '  not covered by this run.\n'
fi

printf '\n\033[32m✓ ready to tag v%s\033[0m\n' "$version"
printf '  Not checkable anywhere but GitHub: CodeQL, Scorecard, dependency-review.\n'
printf '  remaining by hand: CHANGELOG Unreleased section retitled to %s\n' "$version"
