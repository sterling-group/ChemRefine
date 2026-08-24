"""MkDocs hook: normal-mode viewers, drawn from the frequency output that produced them.

A vibration is a thing to watch, and the tutorial used to show one as two screen recordings
— 13.6 MB of animated GIF, which was 90% of ``docs/`` and the only reason it could not ship
in the sdist. Worse, they were pixels: no geometry, no frequency, no way to fix a wrong
camera angle or restyle a figure, and nothing tying either picture to the run it came from.

The data behind such a picture is 3 KB. ChemRefine already parses an ORCA frequency output
into a ``(n_atoms, 3, n_modes)`` tensor, already writes one mode's column as the three extra
XYZ columns a viewer reads as ``dx/dy/dz``, and already animates exactly that file in the
GUI's structure pane (``vibrate()`` builds the frames, ``animate()`` plays them). This hook
is the last link: a page carries a directive, and the file's own numbers become the viewer
and the caption under it.

**Transform, never compute.** The Hessian is not run here and must never be: ORCA is
licensed, a docs build has to work in CI and from the unpacked sdist, and ``mkdocs build``
is 4 seconds. What runs at build time is the parse of a committed 3 KB file — the same
discipline as :mod:`tables`, where the generated thing is generated because a second
hand-typed copy is what goes stale.

The caption is generated for that reason and no other. A frequency typed into prose beside
a data file is two sources for one number, and the one in the prose is the one that will
still say ``-295`` after the file is regenerated.

Directives, all argument-bearing so that :mod:`tables`'s bare ``<!-- chemrefine:name -->``
pattern cannot match them (it raises on names it does not know, and it runs first):

``<!-- chemrefine:mode ts-good -->``
    The viewer for ``docs/tutorials/modes/ts_good.xyz``, plus its caption.
``<!-- chemrefine:mode recipe -->``
    :func:`viewer_file`'s own source, so the page teaches the call that made the files
    above it and cannot drift from them.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODES = _REPO_ROOT / "docs" / "tutorials" / "modes"

#: ``<!-- chemrefine:mode <name> -->`` alone on a line. The argument is what keeps this out
#: of :mod:`tables`'s pattern, which matches a directive whose name runs straight into
#: ``-->`` and raises on any name it does not recognise.
_DIRECTIVE = re.compile(
    r"^[ \t]*<!--[ \t]*chemrefine:mode[ \t]+([a-z][a-z0-9-]*)[ \t]*-->[ \t]*$", re.MULTILINE
)

#: One ``key=value`` or ``key="value with spaces"`` off the extended-XYZ comment line.
_INFO = re.compile(r'(\w+)=("(?:[^"]*)"|\S+)')

#: The keys :func:`viewer_file` writes and a viewer block needs. A file without them is a
#: file whose numbers nobody can trace, which is the state this hook exists to leave behind.
_REQUIRED = ("method", "mode", "frequency_cm1", "imaginary", "source")

#: 3Dmol, from the same address ``docs/_includes/viewer.md`` uses. Emitted once per page:
#: a classic ``<script src>`` blocks until it has run, so the init below it can rely on it.
_LIBRARY = '<script src="https://3Dmol.org/build/3Dmol-min.js"></script>'


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


def _read(name: str) -> tuple[str, dict[str, str]]:
    """A mode file's text and its comment-line metadata, both validated.

    Every failure here is a build failure on purpose. The alternative — a viewer div whose
    data never arrives — renders as a blank rectangle that no test and no reviewer sees,
    which is how the six ``data-xyz`` viewers on this site came to point at a path that
    404s without anything noticing.
    """
    path = _MODES / f"{name.replace('-', '_')}.xyz"
    if not path.is_file():
        known = sorted(p.stem.replace("_", "-") for p in _MODES.glob("*.xyz"))
        raise ValueError(f"no mode file for <!-- chemrefine:mode {name} -->; known: {known}")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    declared = int(lines[0])
    atoms = [line for line in lines[2 : 2 + declared] if line.strip()]
    if len(atoms) != declared:
        raise ValueError(f"{path.name} declares {declared} atoms and carries {len(atoms)}")
    for row in atoms:
        if len(row.split()) != 7:
            raise ValueError(
                f"{path.name} has an atom line with {len(row.split())} columns; a mode file "
                "needs 7 — symbol, x, y, z and the three displacement columns a viewer "
                f"reads as dx/dy/dz: {row.strip()!r}"
            )
    info = {k: v.strip('"') for k, v in _INFO.findall(lines[1])}
    if missing := [key for key in _REQUIRED if key not in info]:
        raise ValueError(f"{path.name} states no {', '.join(missing)} on its comment line")
    return text, info


def _caption(info: dict[str, str]) -> str:
    """What the file says about itself, as the sentence under its viewer."""
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


def _viewer(name: str) -> str:
    """The viewer block for one mode file: coordinates inlined, caption generated.

    Inlined rather than fetched. Every other viewer on this site pulls its structure from
    ``raw.githubusercontent.com/…/main/``, which makes a figure depend on a branch having
    been merged, on the reader having a network, and on the file never moving. Three
    kilobytes in the page costs less than any of that, and it is what lets the built docs
    work from the sdist with no network at all.
    """
    text, info = _read(name)
    element = f"chemrefine-mode-{name}"
    return (
        f'<div id="{element}" class="chemrefine-mode"'
        ' style="width: 100%; height: 400px; position: relative;"></div>\n'
        "<script>\n"
        "  (() => {\n"
        f"    const xyz = {json.dumps(text)};\n"
        f'    const viewer = $3Dmol.createViewer(document.getElementById("{element}"),'
        ' { backgroundColor: "white" });\n'
        '    const model = viewer.addModel(xyz, "xyz");\n'
        "    viewer.setStyle({}, { stick: { radius: 0.15 }, sphere: { scale: 0.25 } });\n"
        "    viewer.zoomTo();\n"
        "    // The three displacement columns are what 3Dmol reads as dx/dy/dz; vibrate()\n"
        "    // only builds the frames, animate() plays them. Same two calls as the GUI.\n"
        "    model.vibrate(10, 1, true);\n"
        '    viewer.animate({ loop: "backAndForth", interval: 60 });\n'
        "    viewer.render();\n"
        "  })();\n"
        "</script>\n\n"
        f"{_caption(info)}"
    )


def _recipe() -> str:
    """:func:`viewer_file`'s source, fenced — the recipe the committed files were made by.

    Rendered from the live function rather than transcribed beside it, for the reason every
    other generated block on this site exists: a snippet that merely *describes* the call
    is a second copy, and the copy is what goes stale.
    """
    return f"```python\n{inspect.getsource(viewer_file).rstrip()}\n```"


def on_page_markdown(markdown: str, **kwargs: Any) -> str:
    """Expand every ``<!-- chemrefine:mode … -->`` directive on one page.

    Raises
    ------
    ValueError
        If a directive names no mode file, or names one that is malformed. Both would
        otherwise reach the built page as a silently empty viewer.
    """
    seen_library = False

    def expand(match: re.Match[str]) -> str:
        nonlocal seen_library
        name = match.group(1)
        if name == "recipe":
            return _recipe()
        block = _viewer(name)
        if not seen_library:
            seen_library = True
            return f"{_LIBRARY}\n{block}"
        return block

    return _DIRECTIVE.sub(expand, markdown)
