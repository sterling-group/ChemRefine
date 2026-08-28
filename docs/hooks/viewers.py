"""MkDocs hook: the 3D viewers, drawn from files in this repository rather than fetched.

Two kinds of figure, one mechanism. A **structure** viewer shows a tutorial's starting
geometry; a **mode** viewer animates one imaginary mode of a frequency run. Both are a
``.xyz`` from this tree, inlined into the page, handed to 3Dmol.

Neither used to work this way and both were broken by it. The mode figures were two
animated GIFs — 13.6 MB, 90% of ``docs/``, and the only reason this directory could not
ship in the sdist — with no geometry, no frequency and no tie to the run behind them. The
structure viewers fetched from ``raw.githubusercontent.com/…/main/`` at page load, which
makes a figure depend on a branch having been merged, on the reader having a network, and
on a path never moving: all six of them render an empty pane today, because ``main`` is
still the layout this branch replaced. A blank rectangle is the worst possible failure —
it looks like a slow load, it survives every gate, and nothing in the build can see it.

Inlined, a structure costs about 2 KB in the page and a mode about 3 KB. That is cheaper
than the fetch it replaces in every way that matters, and it is what lets the built docs
work with no network at all — including from the unpacked sdist.

**Transform, never compute.** The Hessian is not run here and must never be: ORCA is
licensed, a docs build has to work in CI and from the sdist, and ``mkdocs build`` is 4
seconds. What runs at build time is the parse of a committed file — the same discipline as
:mod:`tables`, where the generated thing is generated because a second hand-typed copy is
what goes stale. The caption obeys the same rule: a frequency typed into prose beside a
data file is two sources for one number, and the prose copy is the one that will still say
``-295`` after the file is regenerated.

Directives, all argument-bearing so that :mod:`tables`'s bare ``<!-- chemrefine:name -->``
pattern cannot match them (it raises on names it does not know, and it runs first):

``<!-- chemrefine:structure examples/tutorials/redox/dimethylaniline/step1.xyz -->``
    A still viewer for one geometry, named by its path from the repository root.
``<!-- chemrefine:mode ts-good -->``
    The animated viewer for ``docs/tutorials/modes/ts_good.xyz``, plus its caption.
``<!-- chemrefine:mode recipe -->``
    :func:`viewer_file`'s own source, so the page teaches the call that made those files
    and cannot drift from them.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODES = _REPO_ROOT / "docs" / "tutorials" / "modes"

#: ``<!-- chemrefine:mode <name> -->`` / ``<!-- chemrefine:structure <path> -->`` alone on a
#: line. The argument is what keeps these out of :mod:`tables`'s pattern, which matches a
#: directive whose name runs straight into ``-->`` and raises on any name it does not know.
_DIRECTIVE = re.compile(
    r"^[ \t]*<!--[ \t]*chemrefine:(mode|structure)[ \t]+(\S+)[ \t]*-->[ \t]*$", re.MULTILINE
)

#: One ``key=value`` or ``key="value with spaces"`` off the extended-XYZ comment line.
_INFO = re.compile(r'(\w+)=("(?:[^"]*)"|\S+)')

#: The keys :func:`viewer_file` writes and a mode caption needs. A file without them is a
#: file whose numbers nobody can trace, which is the state this hook exists to leave behind.
_REQUIRED = ("method", "mode", "frequency_cm1", "imaginary", "source")

#: The vendored 3Dmol bundle, as the playground hook publishes it into the site — the same
#: bytes the GUI serves and ``tests/test_gui_assets.py`` parses.
_LIBRARY_ASSET = "playground/static/vendor/3dmol.min.js"


def _library(page_url: str) -> str:
    """The 3Dmol ``<script src>`` for one page, pointing at the site's own vendored copy.

    Emitted once per page; a classic ``<script src>`` blocks until it has run, so every
    init below it can rely on ``$3Dmol`` being there. The *vendored* bundle rather than a
    CDN, because the CDN tag quietly took back what inlining the geometries bought: the
    module's whole claim is that the built docs work with no network at all — including
    from the unpacked sdist — and offline, a CDN library is the blank rectangle the
    docstring above calls the worst possible failure. The copy is already in the site:
    :mod:`playground` publishes ``STATIC_DIR`` (vendor bundle included) on every build.

    Relative to the page rather than site-absolute, because ``file://`` browsing — the
    sdist case — has no site root to anchor an absolute path. mkdocs URLs are
    ``dir/page/`` (or ``dir/page.html``), so the segment count is the depth either way.
    """
    return f'<script src="{"../" * page_url.count("/")}{_LIBRARY_ASSET}"></script>'


def viewer_file(out_path: Path, mode_index: int, *, method: str) -> str:
    """One frequency output's mode, as the extended XYZ a 3Dmol viewer animates.

    The geometry and the displacement come from the same parsed frame, never one of each:
    a mode drawn onto positions it was not computed for is a picture of the wrong molecule
    moving. ``atoms.info`` rides out on the comment line, so the file states which mode it
    holds, what produced it and what else was imaginary in that structure — the caption
    under the viewer is read back out of it rather than typed beside it.
    """
    import hashlib

    from ase import Atoms

    from chemrefine.engines.orca.output import parse_output
    from chemrefine.io import extended_xyz_text

    frame = parse_output(out_path, "freq")[-1]
    if frame.normal_modes is None or frame.frequencies is None:
        raise ValueError(f"{out_path.name} carries no normal modes — was it a FREQ run?")
    atoms = Atoms(symbols=list(frame.symbols), positions=frame.positions)
    atoms.info = {
        "method": method,
        "mode": mode_index,
        "frequency_cm1": frame.frequencies[mode_index],
        "imaginary": ",".join(f"{i}:{v:.2f}" for i, v in sorted(frame.imaginary_freqs.items())),
        "source": out_path.name,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest()[:16],
    }
    return extended_xyz_text(atoms, displacements=frame.normal_modes[:, :, mode_index])


def _atom_lines(text: str, source: str, *, columns: int) -> list[str]:
    """A frame's atom lines, or a build failure saying which of the two ways it is wrong.

    Every failure here is deliberate. The alternative — a viewer div whose data never
    arrives or never parses — renders as a blank rectangle that no test and no reviewer
    sees, which is exactly how six viewers came to point at paths that 404.
    """
    lines = text.splitlines()
    declared = int(lines[0])
    atoms = [line for line in lines[2 : 2 + declared] if line.strip()]
    if len(atoms) != declared:
        raise ValueError(f"{source} declares {declared} atoms and carries {len(atoms)}")
    for row in atoms:
        if len(row.split()) < columns:
            raise ValueError(
                f"{source} has an atom line with {len(row.split())} columns and needs "
                f"{columns}: {row.strip()!r}"
            )
    return atoms


def _read_mode(name: str) -> tuple[str, dict[str, str]]:
    """A mode file's text and its comment-line metadata, both validated."""
    path = _MODES / f"{name.replace('-', '_')}.xyz"
    if not path.is_file():
        known = sorted(p.stem.replace("_", "-") for p in _MODES.glob("*.xyz"))
        raise ValueError(f"no mode file for <!-- chemrefine:mode {name} -->; known: {known}")
    text = path.read_text(encoding="utf-8")
    # Seven columns: symbol, x, y, z and the three displacements a viewer reads as dx/dy/dz.
    _atom_lines(text, path.name, columns=7)
    info = {k: v.strip('"') for k, v in _INFO.findall(text.splitlines()[1])}
    if missing := [key for key in _REQUIRED if key not in info]:
        raise ValueError(f"{path.name} states no {', '.join(missing)} on its comment line")
    return text, info


def _caption(info: dict[str, str]) -> str:
    """What a mode file says about itself, as the sentence under its viewer."""
    imaginary = info["imaginary"].split(",") if info["imaginary"] else []
    if len(imaginary) == 1:
        which = "the structure's only imaginary mode"
    else:
        spelled = ", ".join(f"mode {entry.replace(':', ' at ')}" for entry in imaginary)
        which = f"one of this structure's {len(imaginary)} imaginary modes ({spelled})"
    return (
        f"Mode {info['mode']} at {info['frequency_cm1']} cm⁻¹ — {which}. "
        f"{info['method']}, from `{info['source']}`."
    )


def _block(element: str, xyz: str, *, animate: bool) -> str:
    """One viewer: its pane, its structure inlined, and the calls that draw it.

    ``vibrate()`` builds the frames from the displacement columns and ``animate()`` plays
    them — the same pair the GUI's structure pane makes, and neither alone shows a mode.
    """
    motion = (
        "    // The three displacement columns are what 3Dmol reads as dx/dy/dz; vibrate()\n"
        "    // only builds the frames, animate() plays them. Same two calls as the GUI.\n"
        "    model.vibrate(10, 1, true);\n"
        '    viewer.animate({ loop: "backAndForth", interval: 60 });\n'
    )
    return (
        f'<div id="{element}" class="chemrefine-viewer"'
        ' style="width: 100%; height: 400px; position: relative;"></div>\n'
        "<script>\n"
        "  (() => {\n"
        f"    const xyz = {json.dumps(xyz)};\n"
        f'    const viewer = $3Dmol.createViewer(document.getElementById("{element}"),'
        ' { backgroundColor: "white" });\n'
        '    const model = viewer.addModel(xyz, "xyz");\n'
        "    viewer.setStyle({}, { stick: { radius: 0.15 }, sphere: { scale: 0.25 } });\n"
        "    viewer.zoomTo();\n"
        f"{motion if animate else ''}"
        "    viewer.render();\n"
        "  })();\n"
        "</script>"
    )


def _structure(relative: str) -> str:
    """A still viewer for one geometry in this tree, named from the repository root.

    From the root rather than from the page, because these files live in ``examples/`` and
    the pages that show them live in ``docs/`` — the same spelling the fetch used, so the
    directive reads as the attribute it replaces.
    """
    path = _REPO_ROOT / relative
    if not path.is_file():
        raise ValueError(f"<!-- chemrefine:structure {relative} --> names no file in the tree")
    text = path.read_text(encoding="utf-8")
    _atom_lines(text, relative, columns=4)
    element = "chemrefine-structure-" + re.sub(r"[^a-z0-9]+", "-", relative.lower()).strip("-")
    return _block(element, text, animate=False)


def _mode(name: str) -> str:
    """The animated viewer for one mode file, with the caption its own numbers make."""
    text, info = _read_mode(name)
    return f"{_block(f'chemrefine-mode-{name}', text, animate=True)}\n\n{_caption(info)}"


def _recipe() -> str:
    """:func:`viewer_file`'s source, fenced — the recipe the committed mode files were made by.

    Rendered from the live function rather than transcribed beside it, for the reason every
    other generated block on this site exists: a snippet that merely *describes* the call
    is a second copy, and the copy is what goes stale.
    """
    return f"```python\n{inspect.getsource(viewer_file).rstrip()}\n```"


def on_page_markdown(markdown: str, **kwargs: Any) -> str:
    """Expand every ``<!-- chemrefine:structure … -->`` / ``<!-- chemrefine:mode … -->``.

    Raises
    ------
    ValueError
        If a directive names no file, or names one that is malformed. Both would otherwise
        reach the built page as a silently empty viewer.
    """
    seen_library = False
    # mkdocs hands the Page alongside the markdown; its site-relative URL is what anchors
    # the vendored-library path to this page's own depth.
    page_url = str(kwargs["page"].url)

    def expand(match: re.Match[str]) -> str:
        nonlocal seen_library
        kind, argument = match.groups()
        if kind == "mode" and argument == "recipe":
            return _recipe()
        block = _structure(argument) if kind == "structure" else _mode(argument)
        if not seen_library:
            seen_library = True
            return f"{_library(page_url)}\n{block}"
        return block

    return _DIRECTIVE.sub(expand, markdown)
