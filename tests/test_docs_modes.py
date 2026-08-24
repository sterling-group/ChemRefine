"""Guard: a normal-mode figure must be a viewer with data in it, and the caption must agree.

``docs/hooks/modes.py`` replaced two 13.6 MB screen recordings with two 3 KB frequency
extracts and a viewer built from them at docs-build time. ``mkdocs build --strict`` fails
if the hook raises, which covers a missing or malformed file — but not the two ways this
can be quietly wrong: a hook that *succeeds* and emits a viewer with no coordinates in it
(a blank rectangle nobody sees in review), and a caption whose numbers no longer match the
file it claims to describe. Those are what these tests are.

The last one here has nothing to do with modes and is the reason the whole family exists:
every viewer on this site names its structure in a ``data-xyz`` attribute, and until now
nothing checked that any of those paths led anywhere. All six of the fetch-based ones
currently resolve against ``main``, where this branch's layout does not exist yet.

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
_HOOK = _REPO_ROOT / "docs" / "hooks" / "modes.py"
_MODES = _REPO_ROOT / "docs" / "tutorials" / "modes"
_DOCS = _REPO_ROOT / "docs"

pytestmark = pytest.mark.skipif(
    not _HOOK.is_file(), reason="docs/ is a repository artifact and does not ship in the sdist"
)


def _hook() -> Any:
    """Import the hook the way MkDocs does — by path, not as a package module."""
    spec = importlib.util.spec_from_file_location("chemrefine_docs_modes", _HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _render(name: str) -> str:
    """The markdown one directive expands to, as a page would receive it."""
    return str(_hook().on_page_markdown(f"<!-- chemrefine:mode {name} -->"))


def _shipped() -> list[str]:
    """Every mode file's directive name, so adding one adds its coverage."""
    return sorted(path.stem.replace("_", "-") for path in _MODES.glob("*.xyz"))


def test_there_are_mode_files_to_check():
    """The parametrised tests below would all pass vacuously on an empty directory."""
    assert _shipped(), "no mode files found — the tutorial's figures have gone missing"


@pytest.mark.parametrize("name", _shipped())
def test_a_viewer_carries_its_own_coordinates(name: str):
    """The data is inlined, not fetched — that is the whole point of the file being small.

    Fetching is what every other viewer here does, and it makes a figure depend on a branch
    having been merged, on the reader having a network, and on the file never moving.
    """
    rendered = _render(name)
    lines = (_MODES / f"{name.replace('-', '_')}.xyz").read_text(encoding="utf-8").splitlines()
    assert f'id="chemrefine-mode-{name}"' in rendered
    assert "raw.githubusercontent" not in rendered
    for atom_line in lines[2:]:
        assert atom_line in rendered, f"{name} renders without its own line {atom_line!r}"


@pytest.mark.parametrize("name", _shipped())
def test_a_viewer_animates_the_mode_it_loaded(name: str):
    """``vibrate()`` builds the frames and ``animate()`` plays them — neither alone shows one.

    Pinned because a viewer missing the second call renders a still molecule that looks
    like a deliberate choice rather than a broken figure.
    """
    rendered = _render(name)
    assert "model.vibrate(10, 1, true);" in rendered
    assert 'viewer.animate({ loop: "backAndForth", interval: 60 });' in rendered


@pytest.mark.parametrize("name", _shipped())
def test_the_caption_states_what_the_file_states(name: str):
    """Every number under a viewer comes out of the file, which is why none is typed in prose.

    The caption is generated for exactly this: a wavenumber written into the markdown beside
    a data file is a second source for one number, and it is the prose copy that survives a
    regenerated file and starts lying.
    """
    hook = _hook()
    _, info = hook._read(name)
    rendered = _render(name)
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
    rendered = str(hook.on_page_markdown("<!-- chemrefine:mode recipe -->"))
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
        hook.on_page_markdown("<!-- chemrefine:mode wrong -->")


def test_a_directive_naming_no_mode_file_fails_the_build():
    """A typo would otherwise render as an HTML comment — invisible on the page and in review."""
    with pytest.raises(ValueError, match="no mode file"):
        _hook().on_page_markdown("<!-- chemrefine:mode ts-goof -->")


def test_the_library_is_loaded_once_per_page():
    """Two viewers on one page must not fetch the 3Dmol bundle twice."""
    rendered = str(
        _hook().on_page_markdown(
            "<!-- chemrefine:mode ts-bad -->\n\n<!-- chemrefine:mode ts-good -->\n"
        )
    )
    assert rendered.count("3Dmol-min.js") == 1


def test_every_viewer_on_the_site_names_a_structure_that_exists():
    """A ``data-xyz`` path that leads nowhere is a blank pane, and nothing else looks at these.

    The attribute is resolved by ``docs/_includes/viewer.md`` against the repository root at
    read time, so the tree is exactly where the answer lives — no network needed, and none
    wanted in a test.
    """
    missing = []
    for page in sorted(_DOCS.rglob("*.md")):
        for match in re.finditer(r'data-xyz="([^"]+)"', page.read_text(encoding="utf-8")):
            if not (_REPO_ROOT / match.group(1)).is_file():
                missing.append(f"{page.relative_to(_REPO_ROOT)} -> {match.group(1)}")
    assert missing == [], (
        "these viewers name a structure that is not in the tree, so they render an empty "
        "pane:\n" + "\n".join(missing)
    )
