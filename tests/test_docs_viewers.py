"""Guard: a 3D figure must be a viewer with data in it, and a caption must agree with it.

``docs/hooks/viewers.py`` draws every 3D pane on this site from a file in this tree —
a tutorial's starting geometry, or one imaginary mode of a frequency run. Both used to
work another way and both were broken by it: the mode figures were 13.6 MB of animated
GIF with nothing tying them to a run, and the structure viewers fetched their geometry
from ``raw.githubusercontent.com/…/main/`` at page load, which rendered six empty panes
because ``main`` is still the layout this branch replaces.

``mkdocs build --strict`` fails if the hook raises, which covers a missing or malformed
file. What it cannot see is the failure that put those six panes on the site in the first
place: a viewer that is *emitted* perfectly and simply has no molecule in it. That is what
these tests are, along with the caption's numbers agreeing with the file they describe.

``docs/`` ships in the repository but not in the sdist, so every test skips rather than
fails when the directory is absent — the same rule as ``test_docs_tables``.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HOOK = _REPO_ROOT / "docs" / "hooks" / "viewers.py"
_MODES = _REPO_ROOT / "docs" / "tutorials" / "modes"
_DOCS = _REPO_ROOT / "docs"

pytestmark = pytest.mark.skipif(
    not _HOOK.is_file(), reason="docs/ is a repository artifact and does not ship in the sdist"
)


def _hook() -> Any:
    """Import the hook the way MkDocs does — by path, not as a package module."""
    spec = importlib.util.spec_from_file_location("chemrefine_docs_viewers", _HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _page(url: str = "tutorials/transition-state/") -> Any:
    """What mkdocs hands the hook alongside the markdown — the page, for its own URL.

    The URL is what anchors the vendored 3Dmol path to the page's depth, so the default
    here is a realistic two-level one rather than the site root.
    """
    import types

    return types.SimpleNamespace(url=url)


def _render(directive: str) -> str:
    """The markdown one directive expands to, as a page would receive it."""
    return str(_hook().on_page_markdown(f"<!-- chemrefine:{directive} -->", page=_page()))


def _modes() -> list[str]:
    """Every mode file's directive name, so adding one adds its coverage."""
    return sorted(path.stem.replace("_", "-") for path in _MODES.glob("*.xyz"))


def _structures() -> list[str]:
    """Every structure a page asks for, so adding a tutorial adds its coverage."""
    found: set[str] = set()
    for page in sorted(_DOCS.rglob("*.md")):
        text = page.read_text(encoding="utf-8")
        found.update(re.findall(r"<!--\s*chemrefine:structure\s+(\S+)\s*-->", text))
    return sorted(found)


def test_there_are_viewers_to_check():
    """Every parametrised test below would pass vacuously on an empty roster."""
    assert _modes(), "no mode files found — the tutorial's figures have gone missing"
    assert _structures(), "no page shows a structure — the viewers have gone missing"


@pytest.mark.parametrize("relative", _structures())
def test_a_structure_viewer_carries_the_geometry_it_names(relative: str):
    """The molecule is in the page, not behind a request that can fail silently.

    This is the whole reason the fetch is gone: six panes on this site render empty
    because the file they ask for is not on ``main`` yet, and a blank rectangle looks
    exactly like a slow load. Resolving the path here is also the cheapest half of that
    guard — no network needed, and none wanted in a test.
    """
    path = _REPO_ROOT / relative
    assert path.is_file(), f"{relative} is named by a page and is not in the tree"
    rendered = _render(f"structure {relative}")
    assert "raw.githubusercontent" not in rendered
    for atom_line in path.read_text(encoding="utf-8").splitlines()[2:]:
        if atom_line.strip():
            assert atom_line in rendered, f"{relative} renders without its line {atom_line!r}"


@pytest.mark.parametrize("relative", _structures())
def test_a_structure_viewer_does_not_animate(relative: str):
    """A still geometry has no displacement columns, so vibrate() would shake nothing."""
    assert "vibrate" not in _render(f"structure {relative}")


@pytest.mark.parametrize("name", _modes())
def test_a_mode_viewer_carries_its_own_coordinates(name: str):
    """Same inlining as a structure, for the same reason, with three more columns."""
    rendered = _render(f"mode {name}")
    lines = (_MODES / f"{name.replace('-', '_')}.xyz").read_text(encoding="utf-8").splitlines()
    assert f'id="chemrefine-mode-{name}"' in rendered
    assert "raw.githubusercontent" not in rendered
    for atom_line in lines[2:]:
        assert atom_line in rendered, f"{name} renders without its own line {atom_line!r}"


@pytest.mark.parametrize("name", _modes())
def test_a_mode_viewer_animates_the_mode_it_loaded(name: str):
    """``vibrate()`` builds the frames and ``animate()`` plays them — neither alone moves.

    Pinned because a viewer missing the second call renders a still molecule that looks
    like a deliberate choice rather than a broken figure.
    """
    rendered = _render(f"mode {name}")
    assert "model.vibrate(10, 1, true);" in rendered
    assert 'viewer.animate({ loop: "backAndForth", interval: 60 });' in rendered


@pytest.mark.parametrize("name", _modes())
def test_the_caption_states_what_the_file_states(name: str):
    """Every number under a viewer comes out of the file, which is why none is typed in prose.

    The caption is generated for exactly this: a wavenumber written into the markdown beside
    a data file is a second source for one number, and it is the prose copy that survives a
    regenerated file and starts lying.
    """
    hook = _hook()
    _, info = hook._read_mode(name)
    rendered = _render(f"mode {name}")
    assert f"Mode {info['mode']} at {info['frequency_cm1']} cm⁻¹" in rendered
    assert info["method"] in rendered
    assert info["source"] in rendered
    entries = info["imaginary"].split(",")
    if len(entries) == 1:
        assert "only imaginary mode" in rendered
    else:
        for entry in entries:
            index, frequency = entry.split(":")
            assert f"mode {index} at {frequency}" in rendered


def test_the_recipe_is_the_function_it_claims_to_be():
    """Rendered from the live source, so the how-to cannot drift from what made the files."""
    hook = _hook()
    rendered = str(hook.on_page_markdown("<!-- chemrefine:mode recipe -->", page=_page()))
    assert rendered.startswith("```python")
    assert inspect.getsource(hook.viewer_file).rstrip() in rendered


def test_the_good_mode_is_drawn_on_the_geometry_the_tutorial_ships():
    """The figure and the example must be the same structure, or the page teaches two of them.

    ``examples/tutorials/transition_state/step1.xyz`` *is* the converged TS this run
    produced; the good-mode file is that geometry with three displacement columns added. A
    figure quietly drawn on some other candidate is the kind of thing only a reader with
    both files open would ever catch.
    """
    example = _REPO_ROOT / "examples" / "tutorials" / "transition_state" / "step1.xyz"
    if not example.is_file():
        pytest.skip("examples/ is absent")
    shipped = [line.split()[:4] for line in example.read_text().splitlines()[2:] if line.strip()]
    figure = [
        line.split()[:4]
        for line in (_MODES / "ts_good.xyz").read_text().splitlines()[2:]
        if line.strip()
    ]
    assert [row[0] for row in figure] == [row[0] for row in shipped]
    for drawn, ships in zip(figure, shipped, strict=True):
        for a, b in zip(drawn[1:], ships[1:], strict=True):
            assert abs(float(a) - float(b)) < 1e-6, f"{drawn} is not {ships}"


@pytest.mark.parametrize(
    ("broken", "complaint"),
    [
        ("1\ncomment\nC 0.0 0.0 0.0 0.0\n", "columns"),
        ("31\nmethod=x mode=0 frequency_cm1=-1 imaginary=0:-1 source=s\nC 0 0 0 0 0 0\n", "atoms"),
        ("1\ncomment\nC 0.0 0.0 0.0 0.1 0.2 0.3\n", "comment line"),
    ],
)
def test_a_malformed_mode_file_fails_the_build(tmp_path: Path, broken: str, complaint: str):
    """Each of these renders as a blank rectangle if it is allowed through.

    A viewer that fails at run time fails in the reader's browser, where no gate is
    watching. The build is the last place it can still be a loud failure.
    """
    hook = _hook()
    hook._MODES = tmp_path
    (tmp_path / "wrong.xyz").write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError, match=complaint):
        hook.on_page_markdown("<!-- chemrefine:mode wrong -->", page=_page())


def test_a_directive_naming_nothing_fails_the_build():
    """A typo would otherwise render as an HTML comment — invisible on the page and in review."""
    with pytest.raises(ValueError, match="no mode file"):
        _render("mode ts-goof")
    with pytest.raises(ValueError, match="names no file in the tree"):
        _render("structure examples/tutorials/nowhere/step1.xyz")


def test_the_library_is_loaded_once_per_page():
    """Several viewers on one page must not load the 3Dmol bundle several times."""
    rendered = str(
        _hook().on_page_markdown(
            "<!-- chemrefine:structure examples/tutorials/transition_state/step1.xyz -->\n\n"
            "<!-- chemrefine:mode ts-bad -->\n\n<!-- chemrefine:mode ts-good -->\n",
            page=_page(),
        )
    )
    assert rendered.count("3dmol.min.js") == 1


def test_the_library_is_the_sites_own_vendored_bundle():
    """The 3Dmol tag points into the site itself, at this page's own depth — never a CDN.

    Inlining the geometries bought "the built docs work with no network at all —
    including from the unpacked sdist", and a CDN library tag quietly took it back:
    offline, every viewer was the blank rectangle the hook's docstring calls the worst
    possible failure. The vendored bundle is already published into the site on every
    build (the playground hook copies STATIC_DIR, vendor included), so the tag has a
    local target at a page-relative path — which is what still works under ``file://``,
    where no site root exists to anchor an absolute one.
    """
    rendered = str(
        _hook().on_page_markdown(
            "<!-- chemrefine:mode ts-good -->", page=_page("tutorials/transition-state/")
        )
    )
    assert 'src="../../playground/static/vendor/3dmol.min.js"' in rendered
    assert "3Dmol.org" not in rendered
    shallow = str(_hook().on_page_markdown("<!-- chemrefine:mode ts-good -->", page=_page("")))
    assert 'src="playground/static/vendor/3dmol.min.js"' in shallow


def test_no_page_still_fetches_a_structure_over_the_network():
    """The fetch is what put six empty panes on the site; nothing should reintroduce it.

    ``data-xyz`` was the attribute the old include read, so its absence is the check —
    a page that grows a new viewer has to go through the hook, where the file is resolved
    at build time and the build fails if it is not there.
    """
    fetching = [
        str(page.relative_to(_REPO_ROOT))
        for page in sorted(_DOCS.rglob("*.md"))
        if "data-xyz" in page.read_text(encoding="utf-8")
    ]
    assert fetching == [], (
        "these pages name a structure for a run-time fetch instead of the build-time "
        "directive, so they render an empty pane wherever the fetch fails:\n" + "\n".join(fetching)
    )
