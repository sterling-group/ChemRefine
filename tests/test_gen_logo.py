"""``tools/gen_logo.py`` takes no answer on trust and replaces no file until the kit is staged.

The generator is a developer tool, not the package, and CI never runs inkscape: its renders
are byte-stable only on the authoring toolchain. These cases pin what holds everywhere. The
SVGs Python draws alone are the committed ones, and no stroke is cut into a piece that draws
nothing. A program's answer counts as a product only once its own bytes say so, since an
exit status of 0 is no evidence of success. And publishing replaces no file until every one
is staged. Fake programs on ``PATH`` and throwaway roots stand in for the toolchain, so none
of the cases starts inkscape.
"""

from __future__ import annotations

import errno
import importlib.util
import os
import re
import shlex
import stat
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parent.parent
_TOOL = REPO / "tools" / "gen_logo.py"

# The same reason as tests/test_mutation_gate.py: neither the sdist nor the mutation gate's
# scratch copy carries tools/.
pytestmark = pytest.mark.skipif(
    not _TOOL.is_file(),
    reason="no tools/ — neither the sdist nor the gate's own scratch copy carries it",
)

#: A whole SVG document, the shape inkscape answers an SVG export with.
_SVG = b'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg"></svg>\n'
#: A real 16 px render: the committed one.
_PNG16 = REPO / "docs" / "assets" / "favicon-16.png"


def _load() -> ModuleType:
    """Import ``tools/gen_logo.py`` by path; ``tools/`` is not a package."""
    spec = importlib.util.spec_from_file_location("gen_logo", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gen_logo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The generator with both of its roots under *tmp_path*."""
    module = _load()
    monkeypatch.setattr(module, "ASSETS", tmp_path / "assets")
    monkeypatch.setattr(module, "GUI_BRAND", tmp_path / "brand")
    return module


def _program(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, script: str) -> None:
    """Put a shell script called *name* first on ``PATH``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    program = bin_dir / name
    program.write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
    program.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir), prepend=os.pathsep)


def _seed(gen_logo: ModuleType) -> dict[str, bytes]:
    """Fill both roots with a previous kit; return a new kit's bytes by name."""
    names = (*gen_logo.GUI_SUBSET, "logo-dark.svg", "favicon.ico")
    for root, kept in ((gen_logo.ASSETS, names), (gen_logo.GUI_BRAND, gen_logo.GUI_SUBSET)):
        root.mkdir(parents=True)
        for name in kept:
            (root / name).write_bytes(b"old " + name.encode())
    return {name: b"new " + name.encode() for name in names}


def _tree(gen_logo: ModuleType) -> dict[str, bytes]:
    """Every file in both roots by ``root/name``, staged ones included."""
    return {
        f"{root.name}/{path.name}": path.read_bytes()
        for root in (gen_logo.ASSETS, gen_logo.GUI_BRAND)
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def test_the_svgs_python_draws_alone_are_the_committed_ones():
    """A drawing edit committed without a regeneration fails here, with no inkscape.

    The GUI's copies are held to these by ``tests/test_gui_assets.py``.
    """
    gen_logo = _load()
    drawn = gen_logo.drawn_svgs()
    assert drawn, "drawn_svgs() names no file, so this compares nothing"
    stale = sorted(
        name
        for name, text in drawn.items()
        if (gen_logo.ASSETS / name).read_bytes() != text.encode("utf-8")
    )
    assert stale == [], (
        f"committed SVGs that are not what the generator draws: {stale} — rerun tools/gen_logo.py"
    )


def test_a_stroke_cut_where_a_segment_starts_keeps_none_of_that_segment():
    """Cut at its own start, a segment leaves a piece that starts and ends at one point.

    The flank face's outline cuts A at the apex, where A's second segment starts, so such a
    piece would go into every SVG that draws the face, as a curve that goes nowhere.
    """
    gen_logo = _load()
    wrong = [
        f"{name} at x={seg[0][0]}"
        for name, segs in (("A", gen_logo.A_SEGS), ("B", gen_logo.B_SEGS))
        for i, seg in enumerate(segs)
        if gen_logo.split_chain_at_x(segs, seg[0][0]) != list(segs[:i])
    ]
    assert wrong == [], (
        f"cuts at a segment's start that return more or less than the segments before it: {wrong}"
    )


def test_inkscape_reads_the_svg_on_stdin_and_is_handed_no_path(gen_logo, tmp_path, monkeypatch):
    """A confined inkscape cannot open a path in another /tmp, and can always read a pipe."""
    _program(
        tmp_path,
        monkeypatch,
        "inkscape",
        'for arg; do [ -e "$arg" ] && { echo "handed a path: $arg" >&2; exit 0; }; done\n'
        '[ "$1" = --pipe ] && exec cat',
    )
    assert gen_logo.text_to_path(_SVG) == _SVG


@pytest.mark.parametrize(
    "answer",
    ["", '<?xml version="1.0"?>\\n<svg', "<svg></svg>"],
    ids=["nothing", "torn-document", "no-prologue"],
)
def test_an_svg_counts_only_whole_and_the_refusal_quotes_inkscape(
    gen_logo, tmp_path, monkeypatch, answer
):
    """inkscape exits 0 on input it cannot parse, and says why only on stderr."""
    _program(
        tmp_path,
        monkeypatch,
        "inkscape",
        f"printf '{answer}'; echo 'parser error : Premature end of data' >&2",
    )
    with pytest.raises(gen_logo.KitError) as refused:
        gen_logo.text_to_path(_SVG)
    assert "`inkscape --pipe --export-text-to-path" in str(refused.value)
    assert str(refused.value).endswith("on stderr:\nparser error : Premature end of data")


def test_a_png_counts_only_at_the_size_it_was_asked_for(gen_logo, tmp_path, monkeypatch):
    """The header is read, not trusted: a 16 px render passes as 16 px and as nothing else."""
    _program(tmp_path, monkeypatch, "inkscape", f"exec cat {shlex.quote(str(_PNG16))}")
    png = _PNG16.read_bytes()
    assert gen_logo.export_png(b"<svg/>", 16, 16) == png
    assert gen_logo.export_png(b"<svg/>", 16) == png
    for dims, asked in (
        ((32, 32), "-w 32 -h 32"),
        ((16, 32), "-w 16 -h 32"),
        ((32,), "-w 32 --export-type=png"),
    ):
        with pytest.raises(gen_logo.KitError, match=asked):
            gen_logo.export_png(b"<svg/>", *dims)


@pytest.mark.parametrize(
    "script",
    [
        "head -c 100 {png}",
        "cat {png}; echo done",
        "printf XXXXXXXX; tail -c +9 {png}",
        "head -c 12 {png}; printf XXXX; tail -c +17 {png}",
    ],
    ids=["cut-short", "followed-by-more", "no-signature", "no-header-tag"],
)
def test_a_png_counts_only_whole(gen_logo, tmp_path, monkeypatch, script):
    """With ``-o -`` the product is all of stdout: it opens as a PNG, carries its header
    where a PNG does, and ends where a PNG ends."""
    _program(tmp_path, monkeypatch, "inkscape", script.format(png=shlex.quote(str(_PNG16))))
    with pytest.raises(gen_logo.KitError, match="exited 0 with"):
        gen_logo.export_png(b"<svg/>", 16, 16)


def test_a_failed_exit_is_refused_even_with_a_good_answer(gen_logo, tmp_path, monkeypatch):
    """The answer is judged as well as the exit status, not instead of it."""
    _program(tmp_path, monkeypatch, "inkscape", f"cat {shlex.quote(str(_PNG16))}; exit 1")
    with pytest.raises(gen_logo.KitError, match="exited 1"):
        gen_logo.export_png(b"<svg/>", 16, 16)


@pytest.mark.parametrize("answer", ["\\n\\n", "a,b,c,d,e\\n"], ids=["blank-lines", "not-numbers"])
def test_a_query_with_no_boxes_in_it_is_no_measurement(gen_logo, tmp_path, monkeypatch, answer):
    """Malformed SVG gets blank lines and exit 0 from --query-all: not empty, and not a box.

    Neither is a line of five fields that are not numbers.
    """
    _program(
        tmp_path,
        monkeypatch,
        "inkscape",
        f"printf '{answer}'; echo 'parser error : Premature end of data' >&2",
    )
    with pytest.raises(gen_logo.KitError, match="Premature end of data"):
        gen_logo.fit_wordmark()


def test_a_program_that_never_answers_fails_the_run(gen_logo, tmp_path, monkeypatch):
    """A hung inkscape is a refusal naming the command, not a traceback from subprocess."""
    monkeypatch.setattr(gen_logo, "SUBPROCESS_TIMEOUT_SECONDS", 0.5)
    _program(tmp_path, monkeypatch, "inkscape", "echo stuck >&2; exec sleep 30")
    with pytest.raises(
        gen_logo.KitError, match=r"export-text-to-path.* gave no answer in 0\.5 s"
    ) as refused:
        gen_logo.text_to_path(_SVG)
    assert str(refused.value).endswith("on stderr:\nstuck")


def test_a_program_that_is_not_installed_is_named(gen_logo, tmp_path, monkeypatch):
    """No inkscape on PATH is a refusal that says so, not a FileNotFoundError traceback."""
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(gen_logo.KitError, match="inkscape is required and is not on PATH"):
        gen_logo.text_to_path(_SVG)


def test_only_arial_in_bold_is_the_wordmarks_font(gen_logo, tmp_path, monkeypatch):
    """The preflight refuses fontconfig's match for Arial Bold unless it is Arial, and Bold."""
    _program(tmp_path, monkeypatch, "inkscape", "echo 'Inkscape 1.4.4 (fake)'")
    monkeypatch.setitem(sys.modules, "PIL.Image", ModuleType("PIL.Image"))
    for answer, found in (
        ("Arial|Regular,Normal", "Arial Regular"),
        ("Liberation Sans|Bold", "Liberation Sans Bold"),
    ):
        _program(tmp_path, monkeypatch, "fc-match", f"printf '{answer}'")
        with pytest.raises(gen_logo.KitError, match=f"to {found}, not Arial Bold"):
            gen_logo._preflight()
    _program(tmp_path, monkeypatch, "fc-match", "printf 'Arial|Bold,Negreta'")
    assert gen_logo._preflight() == "Inkscape 1.4.4 (fake)"


def test_pillow_is_required_whatever_else_the_host_has(gen_logo, tmp_path, monkeypatch):
    """The same host passes with Pillow and is refused without it: favicon.ico is not optional."""
    _program(tmp_path, monkeypatch, "fc-match", "printf 'Arial|Bold'")
    _program(tmp_path, monkeypatch, "inkscape", "echo 'Inkscape 1.4.4 (fake)'")
    monkeypatch.setitem(sys.modules, "PIL.Image", ModuleType("PIL.Image"))
    assert gen_logo._preflight() == "Inkscape 1.4.4 (fake)"
    monkeypatch.setitem(sys.modules, "PIL.Image", None)
    with pytest.raises(gen_logo.KitError, match="Pillow is required") as refused:
        gen_logo._preflight()
    assert sys.executable in str(refused.value)


def test_main_refuses_before_anything_renders_and_says_why_on_stderr(gen_logo, monkeypatch, capsys):
    """A refused preflight is exit 1 with its reason on stderr, and nothing is rendered."""

    def preflight():
        raise gen_logo.KitError("inkscape is required and is not on PATH")

    def build_kit():
        pytest.fail("the kit was built after the preflight refused")

    monkeypatch.setattr(gen_logo, "_preflight", preflight)
    monkeypatch.setattr(gen_logo, "build_kit", build_kit)
    assert gen_logo.main() == 1
    assert capsys.readouterr() == ("", "error: inkscape is required and is not on PATH\n")


def test_main_reports_a_refused_render_and_replaces_nothing(gen_logo, monkeypatch, capsys):
    """A refused render is exit 1 with its reason on stderr, and both roots stay as they were."""
    _seed(gen_logo)
    before = _tree(gen_logo)

    def build_kit():
        raise gen_logo.KitError("`inkscape --pipe --query-all` exited 0 with 2 bytes of output")

    monkeypatch.setattr(gen_logo, "_preflight", lambda: "Inkscape 1.4.4 (fake)")
    monkeypatch.setattr(gen_logo, "build_kit", build_kit)
    assert gen_logo.main() == 1
    assert capsys.readouterr() == (
        "rendering with Inkscape 1.4.4 (fake)\n",
        "error: `inkscape --pipe --query-all` exited 0 with 2 bytes of output\n",
    )
    assert _tree(gen_logo) == before


def test_main_publishes_what_it_built_to_both_roots(gen_logo, tmp_path, monkeypatch, capsys):
    """Both roots end up holding exactly the kit, in the mode any new file gets.

    What a killed run left staged is gone, and nothing this run staged is left behind.
    """
    kit = _seed(gen_logo)
    (gen_logo.GUI_BRAND / f"{gen_logo.STAGE_PREFIX}logo-dark.svg").write_bytes(b"killed")
    (gen_logo.ASSETS / f"{gen_logo.STAGE_PREFIX}dropped.png").write_bytes(b"killed")
    monkeypatch.setattr(gen_logo, "_preflight", lambda: "Inkscape 1.4.4 (fake)")
    monkeypatch.setattr(gen_logo, "build_kit", lambda: kit)
    assert gen_logo.main() == 0
    assert capsys.readouterr() == (
        "rendering with Inkscape 1.4.4 (fake)\n"
        f"brand kit written to {gen_logo.ASSETS} ({len(kit)} files)\n"
        f"GUI subset written to {gen_logo.GUI_BRAND} ({len(gen_logo.GUI_SUBSET)} files)\n",
        "",
    )
    assert _tree(gen_logo) == {f"assets/{name}": data for name, data in kit.items()} | {
        f"brand/{name}": kit[name] for name in gen_logo.GUI_SUBSET
    }
    fresh = tmp_path / "fresh"
    fresh.touch()
    modes = {
        stat.S_IMODE(path.stat().st_mode)
        for root in (gen_logo.ASSETS, gen_logo.GUI_BRAND)
        for path in root.iterdir()
    }
    assert modes == {stat.S_IMODE(fresh.stat().st_mode)}


def test_a_root_that_cannot_be_staged_leaves_the_previous_kit(gen_logo, monkeypatch, capsys):
    """The disk fills while the package root is staged, after all of the docs root.

    The run exits 1 with its reason on stderr; nothing has been replaced, and what was staged
    is gone again.
    """
    kit = _seed(gen_logo)
    before = _tree(gen_logo)
    write_bytes = Path.write_bytes

    def full_in_the_package(path, data):
        if path.parent == gen_logo.GUI_BRAND:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(path))
        return write_bytes(path, data)

    monkeypatch.setattr(gen_logo, "_preflight", lambda: "Inkscape 1.4.4 (fake)")
    monkeypatch.setattr(gen_logo, "build_kit", lambda: kit)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", full_in_the_package)
        assert gen_logo.main() == 1
    assert capsys.readouterr().err.rstrip().endswith("; nothing was replaced")
    assert _tree(gen_logo) == before


def test_a_failed_rename_leaves_every_file_whole_and_names_the_new_ones(gen_logo):
    """A rename refused partway leaves each file old or new, never torn, and says which."""
    kit = _seed(gen_logo)
    blocked = gen_logo.GUI_BRAND / "icon-192.png"
    blocked.unlink()
    blocked.mkdir()
    with pytest.raises(gen_logo.KitError) as refused:
        gen_logo.publish(kit)
    tree = _tree(gen_logo)
    new = {key for key, data in tree.items() if data == b"new " + key.split("/")[1].encode()}
    old = {key for key, data in tree.items() if data == b"old " + key.split("/")[1].encode()}
    assert new and old, "the refusal did not fall partway through the renames"
    assert new | old == set(tree), f"neither old nor new: {sorted(set(tree) - new - old)}"
    named = re.search(r"files are new: (.*)\. Every other", str(refused.value))
    assert named is not None, str(refused.value)
    assert sorted(named[1].split(", ")) == sorted(new)
