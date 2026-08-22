"""The commit types CONTRIBUTING states and the ones CI enforces must be one list.

The vocabulary has two readers: the Commits bullet in ``CONTRIBUTING.md``, which is what a
contributor is told, and ``TYPES`` in ``.github/workflows/pr-title.yml``, which is what
actually rejects a pull request. Nothing connected them, and the failure is the quiet kind
both halves pass their own review — a type added to the workflow is enforced but
undocumented, so nobody uses it; a type added to the prose is documented but rejected, so
whoever follows the documentation is told their correct title is wrong.

This is the same defect class ``CONTRIBUTING`` requires an invariant test for, and the same
one the other guards here cover for other pairs: ``test_docs_drift`` holds the configuration
reference to the schema's fields, ``test_docs_metadata`` holds the prose to the packaging
metadata, ``test_docs_urls`` holds every self-referential URL to the tree. This holds the
documented vocabulary to the enforced one.

Both sides are parsed as text rather than imported: the workflow is YAML wrapping a heredoc,
so its ``TYPES`` tuple is not reachable any other way, and the prose is prose.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONTRIBUTING = _REPO_ROOT / "CONTRIBUTING.md"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "pr-title.yml"

pytestmark = pytest.mark.skipif(
    not _WORKFLOW.is_file(),
    reason=".github/ is a repository artifact and does not ship in the sdist",
)

#: The tuple the workflow matches titles against, spread over however many lines `ruff`
#: would wrap it to.
_WORKFLOW_TYPES_RE = re.compile(r"^\s*TYPES = \(([^)]*)\)", re.M)

#: The parenthesised run in CONTRIBUTING's Commits bullet. Anchored on "fixed set" so the
#: guard cannot silently start reading some other list of backticked words if the bullet is
#: reworded around it.
_PROSE_TYPES_RE = re.compile(r"fixed\s+set\s+\(([^)]*)\)", re.S)

#: A backticked word inside either run.
_QUOTED_RE = re.compile(r"`([a-z]+)`|\"([a-z]+)\"")


def _types(pattern: re.Pattern[str], path: Path, what: str) -> set[str]:
    """Every type named in the one run ``pattern`` captures out of ``path``."""
    m = pattern.search(path.read_text(encoding="utf-8"))
    assert m is not None, f"{path.name} no longer states {what} where this guard looks for it"
    found = {a or b for a, b in _QUOTED_RE.findall(m.group(1))}
    assert found, f"{path.name} states {what} but the list came out empty"
    return found


def test_the_documented_types_are_the_enforced_types():
    """A type added on one side fails here until it reaches the other."""
    documented = _types(_PROSE_TYPES_RE, _CONTRIBUTING, "the commit types")
    enforced = _types(_WORKFLOW_TYPES_RE, _WORKFLOW, "TYPES")
    assert documented == enforced, (
        "CONTRIBUTING.md and pr-title.yml disagree about which commit types exist.\n"
        f"  documented but not enforced: {sorted(documented - enforced) or 'none'}\n"
        f"  enforced but not documented: {sorted(enforced - documented) or 'none'}"
    )


def test_harden_is_among_them():
    """The one type outside the conventional-commits set, and the one worth losing.

    ``harden`` is this repository's own — a change that closes a hole without altering
    behaviour — so it is exactly the entry a future edit that "restores the standard list"
    would drop from both sides at once, which the agreement test above cannot see.
    """
    assert "harden" in _types(_WORKFLOW_TYPES_RE, _WORKFLOW, "TYPES")
