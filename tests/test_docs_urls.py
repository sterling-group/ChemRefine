"""Every URL that points back at this project must still point at something real.

Four files are rendered in more than one place — ``README.md``, ``CONTRIBUTING.md``,
``SECURITY.md`` and ``CHANGELOG.md`` are read at the repository root on GitHub, shipped to
PyPI, *and* pulled into the documentation site with ``--8<--``. No relative path is correct
in both contexts, so their cross-references have to be absolute URLs — and an absolute URL
is the one link form ``mkdocs build --strict`` cannot see inside. ``test_docs_xrefs`` covers
``docs/…md`` paths named in prose; nothing covered the same page spelled as a URL.

Three had already rotted when this test landed, all from one rename: ``CHANGELOG.md`` twice
and ``SECURITY.md`` once still pointed at ``/migrating-v1-to-v2/`` and ``/concepts/security/``.
The very same rename *was* caught in ``src/`` by ``test_docs_xrefs`` — which is the evidence
that the URL spelling needs the check the path spelling already gets.

A self-referential URL rots two ways and both are checked here:

* **the page moves, the prefix stays** — the case above, caught by resolving the URL back to
  a file in the tree. ``use_directory_urls`` makes a site URL a pure function of a path
  under ``docs/``, and ``blob|tree/main/…`` a pure function of a repository path, so this
  needs no network and stays as deterministic as the rest of the suite.
* **the repository, organisation or domain moves, the page stays** — caught by taking the
  canonical prefixes from ``mkdocs.yml`` rather than hardcoding them. ``site_url`` and
  ``repo_url`` must be updated on any move or the built site links to the old address, so
  the moment they change every stale URL still naming the old owner fails here at once.
  Hardcoding them would leave this passing after a rename while every link in the wild was
  dead, which is the worse of the two failures and the harder one to notice.

What no guard can reach: URLs inside an **already-published** sdist or wheel. A release is
immutable, so the only defence there is the form of the URL — ``github.com/…`` survives a
rename or an organisation transfer on GitHub's permanent redirects, a ``…github.io/…``
address has none. Prefer the former for anything a published artifact carries.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MKDOCS = _REPO_ROOT / "mkdocs.yml"
_DOCS = _REPO_ROOT / "docs"

# The sibling docs guards say this with `pytestmark = pytest.mark.skipif(...)`, and for them
# that is enough: they only touch `docs/` from inside a test body. This module cannot — every
# constant below is a function of `mkdocs.yml`, read while the module is still importing, and
# a mark is not consulted until after the import it is written in has finished. So in the
# unpacked sdist, where `mkdocs.yml` deliberately does not ship, the read raised first and the
# skip never got a turn: one FileNotFoundError, collection interrupted, the whole suite red on
# a guard that had already decided it had nothing to check. A module-level skip is the form
# that runs early enough to keep that promise.
if not _MKDOCS.is_file():
    pytest.skip(
        "mkdocs.yml declares the canonical URLs and does not ship in the sdist",
        allow_module_level=True,
    )


def _declared(key: str) -> str:
    """The value ``mkdocs.yml`` gives ``key``, read as text rather than parsed.

    ``yaml.safe_load`` cannot read this file at all: the superfences configuration carries a
    ``!!python/name:`` tag, which ``safe_load`` refuses by design. Two top-level scalars do
    not need a parser, and reading them as text keeps the guard free of a YAML dependency it
    would otherwise need only here.
    """
    m = re.search(rf"^{key}:\s*(\S+)\s*$", _MKDOCS.read_text(encoding="utf-8"), re.M)
    assert m is not None, f"mkdocs.yml declares no {key}, so the canonical URL is unknown"
    return m.group(1).rstrip("/")


#: The two addresses this project answers on, from the file that has to be right about them.
_SITE = _declared("site_url")
_REPO = _declared("repo_url")

#: ``sterling-group``, ``ChemRefine`` — the pair a rename changes.
_ORG, _PROJECT = _REPO.rsplit("/", 2)[-2:]

#: Whoever a URL claims owns this project, however it spells the claim: the organisation in
#: a Pages host (``<org>.github.io/ChemRefine``), or the one in a path segment before the
#: project name (``github.com/<org>/ChemRefine``, and the shields.io badges that embed the
#: same pair). The Pages branch is first because the generic one would otherwise capture
#: ``io`` out of the host.
_OWNER_RE = re.compile(
    r"https://(?:([A-Za-z0-9-]+)\.github\.io|[^\s)>\"']*?/([A-Za-z0-9._-]+))/"
    + re.escape(_PROJECT)
    + r"\b"
)

#: A URL pointing back at this project, trimmed of the punctuation that ends a sentence or
#: closes a markdown link.
_SELF_RE = re.compile(r"(?:" + re.escape(_SITE) + r"|" + re.escape(_REPO) + r")[^\s)>\"'\]]*")

#: ``blob``/``tree`` are the two repository-browsing routes that name a path; every other
#: route on the repository host (``/issues``, ``/releases``, ``/security/advisories/new``,
#: a bare clone URL) addresses GitHub's own UI and resolves to no file here.
_BLOB_RE = re.compile(r"(?:blob|tree)/[^/]+/(.+)$")

#: Where a self-reference can be written. ``site/`` is a build artifact and is not scanned.
_SCANNED = (
    "*.md",
    "docs/**/*.md",
    "examples/**/*.md",
    ".github/**/*.md",
    ".github/**/*.yml",
    "pyproject.toml",
    "CITATION.cff",
    "mkdocs.yml",
)


def _iter_text() -> list[tuple[Path, str]]:
    """Every scanned file with its text, each file once however many patterns match it."""
    seen: set[Path] = set()
    out: list[tuple[Path, str]] = []
    for pattern in _SCANNED:
        for path in sorted(_REPO_ROOT.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                out.append((path, path.read_text(encoding="utf-8")))
    return out


def _where(path: Path, text: str, offset: int) -> str:
    """``file:line`` for a match, so a failure names the place to edit."""
    line = text.count("\n", 0, offset) + 1
    return f"{path.relative_to(_REPO_ROOT)}:{line}"


def _candidates(url: str) -> list[Path] | None:
    """Every tree path that would satisfy ``url``, or ``None`` when it names no file.

    A site URL may be written by either mkdocs convention — ``docs/<page>.md`` or
    ``docs/<page>/index.md`` — so both are offered and either one satisfies it.
    """
    url = url.split("#", 1)[0].rstrip(".,;:")
    if url.startswith(_SITE):
        page = url[len(_SITE) :].strip("/")
        if not page:
            return None  # the site root is the home page, not a path to resolve
        return [_DOCS / f"{page}.md", _DOCS / page / "index.md"]
    m = _BLOB_RE.match(url[len(_REPO) :].lstrip("/"))
    return [_REPO_ROOT / m.group(1)] if m else None


def test_every_self_referential_url_names_the_canonical_owner():
    """A rename leaves the old owner behind in prose; this is where that surfaces.

    Checked against ``mkdocs.yml`` rather than a literal, so the assertion is "the docs and
    the prose agree on who owns this project" rather than "the prose says what it said when
    this test was written" — the first survives a move, the second silently outlives one.
    """
    found, wrong = 0, []
    for path, text in _iter_text():
        for m in _OWNER_RE.finditer(text):
            found += 1
            owner = m.group(1) or m.group(2)
            if owner != _ORG:
                wrong.append(f"{_where(path, text, m.start())} -> {owner}/{_PROJECT}")
    assert found, "no URLs naming this project were found — the scanner itself has broken"
    assert wrong == [], (
        f"these URLs name an owner that is not {_ORG!r}, which is what mkdocs.yml declares; "
        "update them together or the published links point at the old address:\n" + "\n".join(wrong)
    )


def test_every_self_referential_url_resolves_to_something_in_the_tree():
    """A page renamed under ``docs/`` leaves these pointing at a 404, and nothing else looks.

    ``--strict`` validates the inside of a *relative* link only, so it walks straight past
    every absolute one — which is the only form the dual-context files can use.
    """
    if not _DOCS.is_dir():
        pytest.skip("docs/ is a repository artifact and does not ship in the sdist")

    found, missing = 0, []
    for path, text in _iter_text():
        for m in _SELF_RE.finditer(text):
            candidates = _candidates(m.group(0))
            if candidates is None:
                continue  # addresses GitHub's UI or the site root: no file to check
            found += 1
            if not any(c.exists() for c in candidates):
                missing.append(f"{_where(path, text, m.start())} -> {m.group(0)}")
    assert found, "no resolvable self-references were found — the scanner itself has broken"
    assert missing == [], (
        "these URLs point back at this project but name nothing that exists; the target was "
        "renamed or removed:\n" + "\n".join(missing)
    )
