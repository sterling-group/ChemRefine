#!/usr/bin/env bash
#
# The local gate. Two modes:
#
#   scripts/release-check.sh --pr     everything CI gates that needs no special hardware
#   scripts/release-check.sh          the above, plus the tier-3 live suite (release)
#
# `--pr` is the one a contributor runs before opening a pull request: if it passes, CI
# will too, bar the things no workstation can do (below). It needs nothing but a `[dev]`
# install — no ORCA, no MLIP stack, no second interpreter (though it uses every supported
# one it can find).
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
#   · the matrix legs this machine has no interpreter for — the ones it has are run, and
#     the ones it lacks are named at the end, where every ending prints this list
#     (`left_to_ci`);
#   · CodeQL, Scorecard and dependency-review, which are GitHub-hosted analyses with no
#     local equivalent.
#
# All of them are covered on the tag regardless: `publish.yml` calls `ci.yml`, so nothing
# reaches PyPI without the full matrix having passed.
#
# Sections. Every step is a named section, and any of them runs on its own:
#
#   --list                 name the sections and the tier-3 cases, marking what the other
#                          options would run — nothing is run
#   --only NAME[,NAME...]  run just these sections, in gate order
#   --skip NAME[,NAME...]  run the mode's sections but these
#   --case NAME[,NAME...]  run just these tier-3 cases (and so only the live section)
#   --no-heavy             leave the heavy cases out of tier-3
#   -- ARGS...             hand ARGS to tier-3's pytest: `-- --record`, `-- -x`, `-- -k mlip`
#
#   e.g.  scripts/release-check.sh --only types,docs
#         scripts/release-check.sh --only live --no-heavy
#         scripts/release-check.sh --case mlip -- --record
#
# A section that needs the built package builds it first. A selection that leaves
# anything out passes as that selection, never as the gate — it names what ran and does
# not say "ready to tag" — so it needs a clean tree only when it runs lint, whose ruff
# hook writes its fixes.
#
# Environment. Nothing machine-specific is baked in: the interpreter is whichever venv or
# conda env is active. Override with the variables below, or put them in an untracked
# `scripts/release-check.env` beside this file, which is sourced when present.
#
#   PY        the Python to gate with          (default: active venv/conda, else python3)
#   LIVE_PY   the Python for the tier-3 suite  (default: $PY — set it when the MLIP and
#             PySCF stacks live in an environment of their own)
#   ORCA      the ORCA binary                  (default: whatever is on PATH)
#   MATRIX_PYTHONS  interpreters for the other supported versions, space-separated, when
#             they are not on PATH as `python3.X` (e.g. one living in a conda env)

set -euo pipefail

self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
repo_root="$(dirname "$(dirname "$self")")"
cd "$repo_root"

# -- the sections: one table, read by --list, the selection and the run loop -----------

# name, whether --pr runs it, and the heading the run prints. Each has a section_<name>.
sections() {
    cat <<'EOF'
lint      pr       lint + format + docstrings (CI: pre-commit)
types     pr       types (CI: type-check)
suite     pr       the suite, with the coverage gate (CI: test)
extras    pr       every declared extra still resolves (CI: extras-resolve)
docs      pr       docs build (CI: docs-check)
package   pr       the package builds, validates, installs and runs (CI: build-package, install-smoke-test)
sdist     pr       the suite passes from the unpacked sdist (CI: install-smoke-test)
matrix    pr       the sdist suite on every other supported Python found here (CI: test matrix)
mutation  pr       critical predicates are checked, not just covered (CI: mutation-gate)
audit     release  known vulnerabilities (CI: security.yml)
live      release  tier-3: the real binaries, end to end
EOF
}
all_sections() { sections | awk '{ print $1 }'; }
pr_sections() { sections | awk '$2 == "pr" { print $1 }'; }
heading() { sections | awk -v name="$1" '$1 == name { $1 = $2 = ""; sub(/^ +/, ""); print }'; }

# A case is a directory holding an input.yaml — the rule tests/test_e2e_live.py collects by.
case_names() {
    local dir
    for dir in tests/data/e2e/cases/*/; do
        if [ -f "$dir/input.yaml" ]; then basename "$dir"; fi
    done
}

usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$self"; }
usage_error() { printf '%s (try --help)\n' "$1" >&2; exit 2; }

# words LIST... — true when the first word is one of the rest.
among() {
    local word="$1" other
    shift
    for other in "$@"; do
        if [ "$word" = "$other" ]; then return 0; fi
    done
    return 1
}

# known WHAT VALID NAMES... — refuse a name VALID does not hold, listing what it does.
known() {
    local what="$1" valid="$2" name
    shift 2
    for name in "$@"; do
        # shellcheck disable=SC2086
        among "$name" $valid || usage_error "no $what '$name' — they are: $(echo $valid)"
    done
}

# -- options ----------------------------------------------------------------------------

mode="" only="" skip="" cases="" no_heavy="" list=""
pytest_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --only=*|--skip=*|--case=*) set -- "${1%%=*}" "${1#*=}" "${@:2}" ;;
    esac
    case "$1" in
        --pr|--release)
            [ -z "$mode" ] || usage_error "give one mode, not --$mode and $1"
            mode="${1#--}" ;;
        --only|--skip|--case)
            [ $# -ge 2 ] && [ -n "$2" ] || usage_error "$1 needs a name"
            case "$1" in
                --only) only="$only ${2//,/ }" ;;
                --skip) skip="$skip ${2//,/ }" ;;
                --case) cases="$cases ${2//,/ }" ;;
            esac
            shift ;;
        --no-heavy) no_heavy=1 ;;
        --list)     list=1 ;;
        -h|--help)  usage; exit 0 ;;
        --)         shift; pytest_args=("$@"); break ;;
        *)          usage_error "unknown option $1" ;;
    esac
    shift
done

# shellcheck disable=SC2086
{
    known section "$(all_sections)" $only $skip
    known case "$(case_names)" $cases
}
if [ -n "$only" ] && [ -n "$mode" ]; then
    usage_error "--only picks the sections itself; drop --$mode"
fi
# --case names tier-3 cases, so on its own it means the live section alone.
if [ -n "$cases" ] && [ -z "$only" ]; then
    [ "$mode" != pr ] || usage_error "--pr has no live section for --case to narrow"
    only="live"
fi
mode="${mode:-release}"
if [ "$mode" = pr ]; then full="$(pr_sections)"; else full="$(all_sections)"; fi

# The selection, always in gate order: a section never runs before one it reads from.
selection=""
for name in $(all_sections); do
    # shellcheck disable=SC2086
    if among "$name" ${only:-$full} && ! among "$name" $skip; then
        selection="$selection $name"
    fi
done
selection="${selection# }"
[ -n "$selection" ] || usage_error "that selection runs no section"
selected() {
    # shellcheck disable=SC2086
    among "$1" $selection
}

if [ -n "$cases$no_heavy" ] || [ ${#pytest_args[@]} -gt 0 ]; then
    selected live || usage_error "--case, --no-heavy and pytest arguments narrow the live section, which this selection does not run"
fi
# Named cases leave --no-heavy nothing to decide.
if [ -n "$no_heavy" ]; then
    [ -z "$cases" ] || usage_error "--case names the cases outright; drop --no-heavy"
fi

# Anything short of the mode's whole gate is partial, and says so instead of passing.
partial=""
# shellcheck disable=SC2086
if [ "$selection" != "$(echo $full)" ] || [ -n "$cases$no_heavy" ] || [ ${#pytest_args[@]} -gt 0 ]; then
    partial=1
fi
# What a run of this selection passes as: `selection`, `pr` or `release`. --list says it and
# the run's ending acts on it, so the preview and the verdict cannot come apart.
verdict() { if [ -n "$partial" ]; then echo selection; else echo "$mode"; fi; }

# What the run left to CI, as every ending prints it, so no ending can drop a clause.
left_to_ci() {
    local legs=""
    if selected matrix && [ "$unmatched" != none ]; then
        legs=", the matrix legs with no interpreter here ($unmatched)"
    fi
    printf '  Still CI-only: the dependency floors%s, CodeQL, Scorecard and dependency-review.\n' \
        "$legs"
}

if [ -n "$list" ]; then
    printf 'sections (* runs with these options):\n'
    while read -r name modes title; do
        mark=" "
        if selected "$name"; then mark="*"; fi
        where="--pr and release"
        if [ "$modes" = release ]; then where="release only"; fi
        printf '%s %-9s %-17s %s\n' "$mark" "$name" "$where" "$title"
    done < <(sections)
    if selected sdist || selected matrix; then
        if ! selected package; then
            printf '  (the sdist and wheel are built first: sdist and matrix run from them)\n'
        fi
    fi
    # shellcheck disable=SC2086
    printf 'tier-3 cases (--case): %s\n' "$(echo $(case_names))"
    if selected live; then
        # shellcheck disable=SC2086
        printf 'live runs: %s%s%s\n' "$(if [ -n "$cases" ]; then echo $cases; else echo every case; fi)" \
            "${no_heavy:+, the heavy ones left out}" "${pytest_args[*]:+, pytest given: ${pytest_args[*]}}"
    fi
    # The order the run loop takes, from the variable it iterates.
    printf 'runs, in order: %s\n' "$selection"
    case "$(verdict)" in
        selection) printf 'passes as: a selection, not the gate\n' ;;
        *) printf 'passes as: the %s gate\n' "$mode" ;;
    esac
    exit 0
fi

# -- preconditions: everything the selection needs, checked before any of it runs -------

# Machine-specific paths live here, outside the tracked script.
# shellcheck source=/dev/null
[ -f scripts/release-check.env ] && . scripts/release-check.env

fail() { printf '\n\033[31m✗ %s\033[0m\n' "$1" >&2; exit 1; }
step() { printf '\n\033[36m▸ %s\033[0m\n' "$1"; }

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
# the wrong source reports everything green and means nothing. It holds for the tier-3
# interpreter as much as for this one.
imports_this_checkout() {
    local resolved
    resolved="$("$1" -c 'import chemrefine, pathlib; print(pathlib.Path(chemrefine.__file__).parent.parent)' 2>/dev/null || true)"
    [ -n "$resolved" ] || fail "chemrefine is not importable by $1 — run: pip install -e '.[dev]'"
    [ "$resolved" = "$repo_root/src" ] || fail "$1 imports chemrefine from $resolved, not $repo_root/src — run: pip install -e '.[dev]'"
}
imports_this_checkout "$PY"

# Each section's own tool, checked before anything runs and only when that section does.
# `pyproject-build` is the console script the `build` distribution ships (there is no
# `build` binary); it stands here for the `$PY -m build` the packaging sections run.
# Without it that step failed several minutes in — after pre-commit, mypy, two whole
# suites and every extras resolve — on a tool this check exists to catch in the first
# second.
need_tool() { [ -x "$bin/$1" ] || fail "$1 is not in $bin — run: pip install -e '.[dev]'"; }
if selected lint; then need_tool pre-commit; fi
if selected types; then need_tool mypy; fi
if selected suite || selected mutation; then need_tool pytest; fi
if selected docs || selected sdist; then need_tool mkdocs; fi
if selected package || selected sdist || selected matrix; then need_tool pyproject-build; fi

status="$(git status --porcelain)" ||
    fail "$repo_root is not a git checkout git can read — the gate certifies a commit"
if [ -n "$status" ]; then
    if [ -z "$partial" ] || selected lint; then
        fail "working tree is dirty — gate a clean tree"
    fi
    dirty=" (dirty tree: a partial run certifies nothing)"
fi

if selected audit && [ ! -x "$bin/pip-audit" ]; then
    fail "pip-audit is not in $bin — it ships with the dev extras, so this checkout is
  installed without them or predates them: pip install -e '.[dev]' (or use --pr)"
fi

if selected live; then
    LIVE_PY="${LIVE_PY:-$PY}"
    [ -x "$LIVE_PY" ] || fail "LIVE_PY=$LIVE_PY is not executable"
    imports_this_checkout "$LIVE_PY"
    live_pytest="$(dirname "$LIVE_PY")/pytest"
    [ -x "$live_pytest" ] || fail "pytest is not in $(dirname "$LIVE_PY") — the tier-3 env needs '.[test]'"

    # `which orca` is not enough: desktop Linux ships /usr/bin/orca, the GNOME screen reader.
    # A real ORCA keeps its helpers beside the main binary. Without this check the tier-3 suite
    # skips every live case and the gate passes having tested nothing. Named cases may need
    # no ORCA at all, and one that does fails by name inside the tier, which runs under
    # CHEMREFINE_REQUIRE_LIVE and rejects a binary with no helpers the same way — so only
    # the whole tier refuses to start without a real ORCA, and an ORCA named in the
    # environment must be the real one whatever runs.
    orca_dir="" found="${ORCA:-$(command -v orca || true)}"
    if [ -n "$found" ]; then
        orca_dir="$(cd "$(dirname "$(readlink -f "$found")")" && pwd)"
        if [ ! -x "$orca_dir/orca_2json" ]; then
            orca_dir=""
            if [ -n "${ORCA:-}" ] || [ -z "$cases" ]; then
                fail "$found is not the quantum-chemistry ORCA (no orca_2json beside it)"
            fi
        fi
    elif [ -z "$cases" ]; then
        fail "ORCA is not on PATH, so the live tier would test nothing.
  For a pre-PR check that needs no ORCA, run: scripts/release-check.sh --pr"
    fi

    # The heavy cases, by their own mark, asked of pytest under an argv this script owns, so
    # `-q`'s one node id per line holds whatever the caller passes through after `--`.
    rc=0
    heavy="$(env -u PYTEST_ADDOPTS "$live_pytest" -q -m heavy --collect-only tests/test_e2e_live.py 2>&1)" || rc=$?
    case "$rc" in
        0|5) ;;
        *) printf '%s\n' "$heavy" >&2; fail "collecting the heavy tier-3 cases failed (pytest exit $rc)" ;;
    esac
    heavy_ids=()
    while IFS= read -r id; do heavy_ids+=("$id"); done < <(grep '::' <<<"$heavy")

    # The tier's own argv, built once: the live section runs it. --no-heavy deselects each
    # heavy case by id, which pytest applies after any -m or -k, so no argument passed
    # through can bring one back.
    live_targets=()
    for case_name in $cases; do
        live_targets+=("tests/test_e2e_live.py::test_live_case[$case_name]")
    done
    leave_out=()
    if [ -n "$no_heavy" ]; then
        for id in ${heavy_ids[@]+"${heavy_ids[@]}"}; do leave_out+=(--deselect "$id"); done
    fi
    tier3_args=(-m "integration or perf" -q ${live_targets[@]+"${live_targets[@]}"}
        ${leave_out[@]+"${leave_out[@]}"} ${pytest_args[@]+"${pytest_args[@]}"})
    tier3=(env CHEMREFINE_REQUIRE_LIVE=1 PATH="${orca_dir:+$orca_dir:}$PATH" "$live_pytest"
        "${tier3_args[@]}")

    # What the live section needs free to start: 7G while a heavy case can run, 4G once
    # none can — --no-heavy, or named cases none of which is heavy, with nothing passed
    # through that could add one.
    names_heavy() {
        local target id
        for target in ${live_targets[@]+"${live_targets[@]}"}; do
            for id in ${heavy_ids[@]+"${heavy_ids[@]}"}; do
                [ "$target" != "$id" ] || return 0
            done
        done
        return 1
    }
    need_g=7 why="the UMA load alone peaks at 5.1G; --no-heavy leaves it out"
    if [ -n "$no_heavy" ] || { [ -n "$cases" ] && [ ${#pytest_args[@]} -eq 0 ] && ! names_heavy; }; then
        need_g=4 why="no heavy case in the run"
    fi
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

printf 'repo    %s\n' "$repo_root"
printf 'python  %s (%s)\n' "$PY" "$("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
printf 'mode    %s\n' "$mode"
if [ -n "$partial" ]; then printf 'runs    %s%s\n' "$selection" "${dirty:-}"; fi
if selected live; then
    printf 'orca    %s\n' "${orca_dir:-none (a case that needs it fails by name)}"
    printf 'tier-3  pytest %s\n' "${tier3_args[*]}"
    printf 'memory  ~%sG free (%s)\n' "$need_g" "$why"
fi

# The GUI's JavaScript is executed by exactly seven tests, and without a Node they *skip* —
# so on a machine with none this gate has been passing while checking the frontend not at
# all, the same hole the ORCA precondition above closes for tier-3. `[test]` now installs a
# Node, so a skip here means the lookup broke, not that the machine is bare. Exported once:
# the coverage run, the sdist run and the mutation gate all inherit it.
export CHEMREFINE_REQUIRE_NODE=1

# -- the sections ---------------------------------------------------------------------

section_lint() {
    # `pre-commit run --all-files`, not the three tools by hand: the hooks are what CI runs,
    # and a hook added to .pre-commit-config.yaml would otherwise never reach this gate. The
    # ruff hook carries --fix, so a violation is *written* and the run then fails on the
    # modified file — which the clean-tree precondition above makes visible.
    "$bin/pre-commit" run --all-files
}

section_types() {
    "$bin/mypy"
}

section_suite() {
    "$bin/pytest" --cov=chemrefine --cov-fail-under=100 -q
}

section_extras() {
    # Metadata only — no torch wheel is downloaded. Read out of the metadata that declares
    # them rather than restated here, which is how CI's own hand-written list came to be
    # missing `mlff` and `pyscf-gpu`.
    local extra
    for extra in $("$PY" -c \
            "import tomllib;print(' '.join(sorted(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies'])))"); do
        printf '  [%s]\n' "$extra"
        "$bin/pip" install --disable-pip-version-check --dry-run --quiet ".[$extra]" >/dev/null
    done
}

section_docs() {
    "$bin/mkdocs" build --strict
}

# The sdist and wheel every packaging section reads, built once per run and always fresh:
# a dist/ left by an earlier run is another commit's package, and gating it would be the
# wrong-source trap the import check above closes.
built=""
build_package() {
    if [ -n "$built" ]; then return 0; fi
    printf '  building dist/\n'
    rm -rf dist
    "$PY" -m build --sdist --wheel >/dev/null
    tar -xzf dist/*.tar.gz -C "$work"
    built=1
}

section_package() {
    build_package
    # The metadata half of CI's build-and-inspect. twine is not a project dependency, so it
    # lives in a throwaway venv of its own.
    "$PY" -m venv "$work/twine"
    "$work/twine/bin/pip" install --disable-pip-version-check --quiet twine
    "$work/twine/bin/twine" check dist/*

    # The sdist is the second artifact PyPI serves and a distro packager builds from it, so it
    # gets the same proof the wheel does. It used to get none: this script built it and then
    # installed `dist/*.whl`, which never matched the tarball — and that is how `examples/`
    # fell out of the sdist with every gate green (5246d7a). `--help` proves the entry point
    # imports; `validate` on the shipped quickstart (its ORCA path is a placeholder, so it
    # warns and exits 0), `schema` and `engines --json` prove the installed package works.
    "$PY" -m venv "$work/smoke"
    "$work/smoke/bin/pip" install --disable-pip-version-check --quiet dist/*.whl
    "$work/smoke/bin/chemrefine" --version
    "$work/smoke/bin/chemrefine" --help > /dev/null
    "$work/smoke/bin/chemrefine" validate examples/first_run/input.yaml
    "$work/smoke/bin/chemrefine" schema > /dev/null
    "$work/smoke/bin/chemrefine" engines --json > /dev/null
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
    "$work/smoke/bin/chemrefine" validate examples/first_run/input.yaml
    "$work/smoke/bin/chemrefine" schema > /dev/null
    "$work/smoke/bin/chemrefine" engines --json > /dev/null
}

section_sdist() {
    build_package
    # The sdist ships tests and examples precisely so the suite can be run from it, and nothing
    # else executes that suite — which is how docs/ and examples/ silently fell out of it. The
    # few remaining repo-only guards (.github/) skip themselves there with named reasons.
    ( cd "$work"/chemrefine-*/ \
      && "$PY" -m venv "$work/sdist-venv" \
      && "$work/sdist-venv/bin/pip" install --disable-pip-version-check --quiet ".[test]" \
      && "$work/sdist-venv/bin/python" -m pytest -q -p no:cacheprovider )

    # The sdist also promises it can rebuild the site, which is a promise about *contents*:
    # mkdocs.yml, the hooks, and the two root files two pages `--8<--` are all things that
    # would fall out of the include list without a word, exactly as `examples/` did. Four
    # seconds, and run with this environment's mkdocs rather than a second install — that
    # proves the tarball's own docs sources are complete, which is what can rot here.
    "$bin/mkdocs" build --strict -f "$work"/chemrefine-*/mkdocs.yml > /dev/null
}

unmatched=""
section_matrix() {
    build_package
    # A deprecation only the newest interpreter raises is invisible from every other, so each
    # supported version this machine has runs the same sdist suite as above. The versions are
    # the classifiers' — the claim the matrix exists to test — and the ones the machine lacks
    # are named at the end rather than dropped. A candidate counts only if it answers with that
    # version: a name on PATH can be a shim for a version that is not selected.
    local versions this_py version other candidate missing_pythons=()
    versions="$("$PY" - <<'PY'
import re, tomllib
classifiers = tomllib.load(open("pyproject.toml", "rb"))["project"]["classifiers"]
print(*(m[1] for c in classifiers if (m := re.fullmatch(r"Programming Language :: Python :: (3\.\d+)", c))))
PY
)"
    [ -n "$versions" ] || fail "no 'Programming Language :: Python :: 3.x' classifiers in pyproject.toml"
    this_py="$(pyver "$PY")"
    for version in $versions; do
        [ "$version" = "$this_py" ] && continue
        other=""
        for candidate in "python$version" ${MATRIX_PYTHONS:-}; do
            [ "$(pyver "$candidate")" = "$version" ] && other="$candidate"
        done
        if [ -z "$other" ]; then
            missing_pythons+=("$version")
            continue
        fi
        printf '  [%s] %s\n' "$version" "$other"
        "$other" -m venv "$work/py$version"
        ( cd "$work"/chemrefine-*/ \
          && "$work/py$version/bin/pip" install --disable-pip-version-check --quiet ".[test]" \
          && "$work/py$version/bin/python" -m pytest -q -p no:cacheprovider )
    done
    unmatched="${missing_pythons[*]:-none}"
}
pyver() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true; }

section_mutation() {
    # 100% branch coverage proves every line ran; it does not prove an assertion looked at the
    # result. This breaks each predicate whose inversion would be a wrong scientific answer and
    # requires a red test. It copies the tree to a scratch dir and never touches this one.
    "$PY" scripts/mutation_gate.py
}

section_audit() {
    # A scheduled weekly sweep in CI, so between two Mondays a release can ship against an
    # advisory nothing has looked at yet. --skip-editable: the local checkout is not on PyPI.
    "$bin/pip-audit" --skip-editable
}

section_live() {
    # CHEMREFINE_REQUIRE_LIVE turns a missing backend into a failure instead of a skip. The
    # ORCA precondition above exists because a silently skipped tier leaves this gate passing
    # having tested nothing — but that reasoning covers every backend, not just ORCA, and
    # pytest reports "N passed, M skipped" with exit 0 either way. With the variable set, a
    # missing MACE or PySCF stack fails here by name. The argv is the live preconditions'
    # (`tier3`), built once.

    # Memory discipline: the UMA single point alone peaks at 5.1 GB RSS (measured), and on a
    # desktop this tier has OOM-killed VS Code instead of the test — the snap ships the editor
    # with oom_score_adj=300, which makes it the kernel's *preferred* victim over any plain
    # process. So: refuse to start without headroom, and run the tier inside a memory-capped
    # scope sized below what is available, so a shortage kills the tier with this script's
    # message rather than whatever the kernel fancies. The cap is what is available less 1G,
    # so the need is the peak plus 2G: 7G with a heavy case in the selection, 4G without —
    # every other case peaks near 1G of its own memory. The live preconditions decide which
    # (`need_g`), and the header prints it.
    if command -v systemd-run >/dev/null 2>&1; then
        local avail_g
        avail_g="$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)"
        if [ "$avail_g" -lt "$need_g" ]; then
            fail "only ${avail_g}G of memory available — this tier-3 selection needs ~${need_g}G free ($why); close some applications first"
        fi
        # Low CPU weight and niceness too, so the desktop stays responsive while it runs.
        systemd-run --user --scope --collect --quiet \
            -p MemoryMax="$((avail_g - 1))G" -p MemorySwapMax=512M -p CPUWeight=20 \
            nice -n 19 "${tier3[@]}"
    else
        "${tier3[@]}"
    fi
}

# -- the run ----------------------------------------------------------------------------

# Each section runs as a plain command, never in a condition: bash suspends `set -e` inside
# a function called from an `if`, `||` or `&&`, and a failing line would then be ignored.
for name in $selection; do
    step "$(heading "$name")"
    "section_$name"
done

case "$(verdict)" in
    selection)
        printf '\n\033[32m✓ passed: %s\033[0m\n' "$selection"
        left_to_ci
        printf '  A selection, not the gate: before %s, run it whole.\n' \
            "$([ "$mode" = pr ] && echo 'opening a pull request' || echo 'tagging')"
        exit 0 ;;
    pr)
        printf '\n\033[32m✓ PR gate passed\033[0m\n'
        left_to_ci
        printf '  All run on your pull request.\n'
        exit 0 ;;
esac

version="$("$PY" -c 'import chemrefine; print(chemrefine.__version__)')"
printf '\n\033[32m✓ ready to tag v%s\033[0m\n' "$version"
left_to_ci
printf '  All run on the tag, since publish.yml calls ci.yml before anything is published.\n'
printf '  remaining by hand: CHANGELOG Unreleased section retitled to %s\n' "$version"
