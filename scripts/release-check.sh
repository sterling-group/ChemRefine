#!/usr/bin/env bash
#
# Everything CI cannot check, run in one command before tagging a release.
#
# GitHub-hosted runners have no ORCA and no MLIP stack, so the tier-3 suite — the only
# tests that drive real binaries end to end — never runs there. Automating it would mean
# a self-hosted runner, and ChemRefine is a public repository: a pull request from a fork
# can execute arbitrary code on a self-hosted runner, which here would be a workstation
# holding an ORCA licence and the maintainer's keys. So this is a local gate instead,
# named in CONTRIBUTING.md as a required step, rather than a workflow.
#
# The defects this exists to catch are the ones that pass every structural gate: a parser
# that reads real output wrongly, a step that returns the previous run's results. Coverage
# and mypy cannot see those; running the real thing can.
#
# Usage:  scripts/release-check.sh
# Env:    DEV_ENV / E2E_ENV  — override the conda prefixes (defaults below)
#         ORCA               — override the ORCA binary to test against

set -euo pipefail

DEV_ENV="${DEV_ENV:-$HOME/opt/envs/chemrefine-dev}"
E2E_ENV="${E2E_ENV:-$HOME/opt/envs/chemrefine-e2e}"
ORCA="${ORCA:-$(command -v orca || true)}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fail() { printf '\n\033[31m✗ %s\033[0m\n' "$1" >&2; exit 1; }
step() { printf '\n\033[36m▸ %s\033[0m\n' "$1"; }

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

printf 'repo    %s\n' "$repo_root"
printf 'orca    %s\n' "$orca_dir"
printf 'version %s\n' "$("$DEV_ENV/bin/python" -c 'import chemrefine; print(chemrefine.__version__)')"

# -- the gate ---------------------------------------------------------------

step "lint + format"
"$DEV_ENV/bin/ruff" check src tests
"$DEV_ENV/bin/ruff" format --check src tests

step "types"
"$DEV_ENV/bin/mypy"

step "docstrings"
"$DEV_ENV/bin/interrogate" -c pyproject.toml src/chemrefine/

step "unit + contract + replay, with the coverage gate"
"$DEV_ENV/bin/pytest" --cov=chemrefine --cov-fail-under=100 -q

step "docs build"
"$DEV_ENV/bin/mkdocs" build --strict

step "package builds, installs, and runs"
rm -rf dist
"$DEV_ENV/bin/python" -m build --sdist --wheel >/dev/null
smoke="$(mktemp -d)"
trap 'rm -rf "$smoke"' EXIT
"$DEV_ENV/bin/python" -m venv "$smoke/venv"
"$smoke/venv/bin/pip" install --quiet dist/*.whl
"$smoke/venv/bin/chemrefine" --version
"$smoke/venv/bin/python" -c "
import importlib.util, pathlib, sys
spec = importlib.util.find_spec('chemrefine')
marker = pathlib.Path(spec.submodule_search_locations[0]) / 'py.typed'
sys.exit('py.typed missing from the wheel' if not marker.is_file() else 0)
"

step "tier-3: the real binaries, end to end"
PATH="$orca_dir:$PATH" "$E2E_ENV/bin/pytest" -m integration -q

step "the tag will match the version"
version="$("$DEV_ENV/bin/python" -c 'import chemrefine; print(chemrefine.__version__)')"
printf '\n\033[32m✓ ready to tag v%s\033[0m\n' "$version"
printf '  remaining by hand: CHANGELOG Unreleased section retitled to %s\n' "$version"
