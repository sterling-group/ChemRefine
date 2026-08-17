"""Drift guard: the Python range the prose promises must be the one the package declares.

Three docs state a supported Python range in words, and nothing connected them to the
metadata that decides it. When the matrix gained 3.14 (``f7c16fc``) the classifiers and
``ci.yml`` moved together and the prose did not, so for as long as that took to notice a
user on 3.14 was told the package did not support their interpreter while PyPI's own
metadata said it did — the two halves of the same promise contradicting each other.

This is the fourth guard of its kind, and it covers the class the other three leave open:
``test_docs_drift`` holds the configuration reference to the schema's field names,
``test_docs_examples`` holds every YAML fence to the real models, ``test_docs_xrefs``
holds every docstring role to a real object. Claims about *packaging metadata* had
nothing watching them.

The classifiers are the source, not ``requires-python``: the floor alone cannot say where
the range ends, and the ceiling is the half that goes stale. ``requires-python`` is
checked against the lowest classifier instead, so the two cannot drift from each other
either.

Skipping is **per file**, not module-wide: the sdist ships ``/README.md`` and
``/pyproject.toml`` but deliberately not ``/docs``, so a distro packager's run must still
check the README rather than skip the whole module the way ``test_docs_drift`` does.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

#: Every file whose prose states the supported Python range, and must therefore track it.
_PROSE = ("README.md", "docs/index.md", "docs/user-guide/installation.md")

_CLASSIFIER_RE = re.compile(r"^Programming Language :: Python :: (3\.\d+)$")

_EN_DASH = "\N{EN DASH}"
"""Named rather than typed, because the docs use an en dash and a hyphen looks identical.

``ruff``'s RUF001 flags the literal for exactly that reason: a guard that silently matched
the wrong dash would pass while checking nothing."""

_RANGE_RE = re.compile(r"Python[^\n]{0,4}\*{0,2}\s*(3\.\d+" + _EN_DASH + r"3\.\d+)")

pytestmark = pytest.mark.skipif(
    not _PYPROJECT.is_file(), reason="pyproject.toml is the source of truth and is absent"
)


def _supported() -> list[tuple[int, int]]:
    """Every ``3.x`` the classifiers name, ascending — the range the package claims."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    found = [
        m.group(1)
        for line in data["project"]["classifiers"]
        if (m := _CLASSIFIER_RE.match(line))
    ]
    return sorted(tuple(int(part) for part in v.split(".")) for v in found)  # type: ignore[misc]


def _documented_range() -> str:
    """The range string the prose must contain, spelled with the dash the docs use."""
    versions = _supported()
    lowest = ".".join(str(part) for part in versions[0])
    highest = ".".join(str(part) for part in versions[-1])
    return f"{lowest}{_EN_DASH}{highest}"


def test_the_classifiers_name_a_contiguous_range():
    """A gap would make "3.11-3.14" a lie in the middle rather than at the end.

    The prose states a range, so the classifiers have to *be* one — a package that
    supported 3.11 and 3.14 but not 3.12 could not be described this way at all, and the
    guard below would pass while saying something false.
    """
    versions = _supported()
    assert versions, "no `Programming Language :: Python :: 3.x` classifiers to read"
    minor = [v[1] for v in versions]
    assert minor == list(range(minor[0], minor[-1] + 1)), (
        f"classifiers name {minor}, which is not contiguous; the docs describe a range"
    )


def test_requires_python_floor_matches_the_lowest_classifier():
    """The two halves of the same promise, kept in step.

    ``requires-python`` is what pip enforces and the classifiers are what a human reads
    on PyPI; a floor that disagrees with them installs on an interpreter the project does
    not claim, or refuses one it does.
    """
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    floor = data["project"]["requires-python"]
    lowest = ".".join(str(part) for part in _supported()[0])
    assert floor == f">={lowest}", (
        f"requires-python is {floor!r} but the lowest classifier is {lowest}"
    )


@pytest.mark.parametrize("relative", _PROSE)
def test_the_documented_python_range_matches_the_package(relative: str):
    """Every doc that states the range must state the one the classifiers declare.

    Written as a substring check rather than a rewrite of the sentence: each of these
    says it differently — a sentence in the README, a bolded bullet in the install guide
    — and what has to hold is the numbers, not the phrasing around them.
    """
    path = _REPO_ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} is a repository artifact and does not ship in the sdist")
    text = path.read_text(encoding="utf-8")
    wanted = _documented_range()
    stated = _RANGE_RE.findall(text)
    assert stated, f"{relative} states no Python range; it used to, and the guard needs one"
    assert all(found == wanted for found in stated), (
        f"{relative} says Python {set(stated)} but the classifiers declare {wanted}; "
        f"the matrix moved and the prose did not"
    )
