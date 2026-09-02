#!/usr/bin/env python3
"""ChemRefine brand kit generator.

Draws the logo from measured cubic-segment tables and writes every asset the
project needs into docs/assets/.

Architecture
------------
The mark is a drawing, and the drawing is data: every curve below is a chain
of cubic Bezier segments whose control points were MEASURED from the original
raster (provenance: REFERENCE_URL) - the outlines by silhouette extraction,
the seven mesh curves by cell decomposition (the mesh partitions the sheet
into fill cells; each curve is the medial line between adjacent cells, so
junctions can never be confused), each then least-squares fitted with a
minimal number of segments. Generation is pure emission: no fitting, no
image processing, no randomness at build time. check_logo.py audits the
tables (G1 joints, attachment of mesh ends to the strokes, family order,
gradient monotonicity) instead of re-deriving them.

Anatomy (paint order)
---------------------
  1. sheet fill: the region between stroke A (top) and stroke B, closed over
     the shared left ascent; navy -> teal -> green ramp
  2. the seven mesh curves (4 sweeping "verticals", 3 flowing "horizontals"),
     each a single cubic stroked with its own measured ink ramp that begins
     and ends in the host stroke's exact ink (no seam at any junction)
  3. the flank face: the dark seen-from-below triangle bounded by the ascent,
     stroke B's dive, and the ground line; lighter blue up, navy down
  4. the ground line: the sheet's straight back edge, left tip -> T onto B,
     painted with the SAME face gradient (extended to TONGUE_INK at the back
     edge), so face and tongue are one seamless field, not a bordered line
  5. strokes A and B on top, round caps, measured ink ramps; B wears A's
     exact ink over the apex until the strokes visibly separate

Stroke A runs left tip -> apex -> saddle -> crest -> tail tip with every
extremum pinned to a horizontal-tangent anchor. Stroke B's first anchor IS
A's apex (same point, both exactly 0 deg) and its last IS A's tail tip with
the tied tangent - the strokes genuinely divide and remerge.

One drawing, two optical sizes: favicon.svg boosts stroke widths for 16-48 px
legibility only, and only the 16/32/48 rasters come from it - the 180 px
apple-touch icon and the 192/512 launcher tiles rasterize from the display
mark, or they wear tab-sized stroke weights at tile size.
logo-header.svg is the white mono line-art for the docs header bar.
Requires inkscape (text->path + PNG export; authored with Inkscape 1.4.4),
Pillow (favicon.ico) and real Arial Bold for the wordmark.
Regenerated assets are byte-stable only on the authoring toolchain; CI never
regenerates.

Two output roots, one writer
----------------------------
The kit lands in docs/assets/ AND in the GUI package (GUI_BRAND below). The
second root is not a convenience: the wheel ships src/chemrefine and nothing
else, so docs/ is not on the served path, and the builder can only reach a
brand file that lives inside the package. The GUI subset is GUI_SUBSET - the
colored mark for its white header bar (the white mono line-art is the same
framing for the docs' teal one), the lockup its stylesheet washes over the
page ground, and
the icon family. No dark variant: the GUI has no dark mode. Both roots are
written from the same values in the same run, so they are byte-identical by
construction and a test holds them that way. That is a different claim from
"a rebuild reproduces the committed bytes", which is only true on the
authoring toolchain and is why CI never rebuilds.

Regenerating
------------
    python3 tools/check_logo.py   # audit the tables after ANY edit
    python3 tools/gen_logo.py     # rebuild every asset (runs from anywhere)
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
ASSETS = _ROOT / "docs" / "assets"
GUI_BRAND = _ROOT / "src" / "chemrefine" / "gui" / "static" / "brand"
# What the builder page and its stylesheet reference. Beside the assets, not inside
# vendor/, which means "third-party, byte-identical to upstream" and is excluded
# from formatting.
GUI_SUBSET = (
    "logo.svg",
    "logo-wordmark.svg",
    "favicon.svg",
    "favicon-32.png",
    "apple-touch-icon.png",
    "icon-192.png",
    "icon-512.png",
)
REFERENCE_URL = "https://github.com/user-attachments/assets/ae7b1ad5-0d90-445c-be83-ddcb76fa85c3"

# --------------------------------------------------------------- palette ----
# All values measured from the reference raster (cell medians for the fill,
# windowed stroke-core medians for the inks, in-face samples for the flank).
CHEM_NAVY, REFINE_TEAL = "#25567d", "#3f9b93"
WHITE = "#ffffff"

FILL_STOPS = (
    (0.00, "#1d4a75"),
    (0.10, "#28618d"),
    (0.25, "#4295a8"),
    (0.45, "#4aa79c"),
    (0.62, "#54b69b"),
    (0.80, "#5abf9c"),
    (1.00, "#5ec69e"),
)

FACE_STOPS = ((0.00, "#3478a3"), (0.50, "#28608a"), (1.00, "#1d4a73"))

# Dark-scheme remap: raise only what melts into a slate page.
DARK_MAP = {"#1d4a75": "#2b689c", "#25567d": "#7fb0d8", "#1d4a73": "#2a659a"}

FONT = "Arial"  # real Arial Bold required; _require_font() refuses lookalikes
FONT_WEIGHT = "bold"
LS = "-0.01em"

GRAD_RISE = 140.0  # fill-ramp axis climb tip-to-tip (lightens up-right)

SW_MESH, SW_OUTLINE, SW_GROUND = 16.0, 21.0, 19.0
SW_MESH_BOOST, SW_OUTLINE_BOOST = 26.0, 36.0

# -------------------------------------------------------------- geometry ----
# Measured anchor tables: (x, y, tangent deg, len_in, len_out) per anchor,
# one tangent per anchor (G1 by construction); _build() emits the cubics.
# fmt: off
OUTLINE_A_STOPS = (
    (0.021, '#1e4871'), (0.05, '#1a426c'), (0.069, '#163964'), (0.084, '#153663'),
    (0.102, '#173a67'), (0.186, '#193f6a'), (0.559, '#1e496d'), (0.763, '#295f75'),
    (0.808, '#357981'), (0.848, '#3c8789'), (0.889, '#3f8e8a'),
)

OUTLINE_B_STOPS = (
    (0.0, '#183e69'), (0.164, '#183e69'), (0.201, '#1b466e'), (0.237, '#1c466f'),
    (0.28, '#1c4770'), (0.306, '#1e4b72'), (0.339, '#205074'), (0.377, '#225476'),
    (0.42, '#245878'), (0.467, '#275d79'), (0.513, '#29617a'), (0.56, '#2c667c'),
    (0.603, '#2e6c7e'), (0.686, '#327381'), (0.76, '#367c84'), (0.843, '#3a8588'),
    (0.89, '#3f8e8c'),
)

MESH_STOPS = {  # measured ink ramps, offsets on each line's own axis
    "V1": (
        (0.0, '#1a416b'), (0.134, '#1a416b'), (0.171, '#18406c'), (0.245, '#18406c'),
        (0.392, '#183f6d'), (0.467, '#18406e'), (0.541, '#1a436f'), (0.617, '#1d4871'),
        (0.766, '#204e74'), (0.841, '#225376'), (0.913, '#245678'), (0.964, '#29627a'),
        (1.0, '#29627a'),
    ),
    "V2": (
        (0.0, '#1b436b'), (0.15, '#1b436b'), (0.316, '#183f6c'), (0.39, '#18406d'),
        (0.464, '#19416e'), (0.539, '#1c4771'), (0.687, '#204e74'), (0.761, '#225376'),
        (0.835, '#245878'), (0.907, '#275d79'), (0.968, '#2f6e7f'), (1.0, '#2f6e7f'),
    ),
    "V3": (
        (0.0, '#1c456c'), (0.132, '#1c456c'), (0.244, '#163c6a'), (0.392, '#19416e'),
        (0.466, '#1b4570'), (0.614, '#1d4a72'), (0.688, '#204f74'), (0.763, '#235676'),
        (0.837, '#265b78'), (0.91, '#28607a'), (0.976, '#337682'), (1.0, '#337682'),
    ),
    "V4": (
        (0.0, '#1d486d'), (0.185, '#1d486d'), (0.239, '#173e6b'), (0.388, '#18406d'),
        (0.465, '#1c4670'), (0.626, '#204e74'), (0.703, '#235676'), (0.778, '#275e7a'),
        (0.85, '#2b667d'), (0.917, '#306f80'), (0.979, '#377f85'), (1.0, '#377f85'),
    ),
    "H1": (
        (0.0, '#1c466f'), (0.039, '#1c466f'), (0.094, '#18406e'), (0.168, '#183f6c'),
        (0.315, '#18406c'), (0.39, '#18406d'), (0.543, '#183e6c'), (0.696, '#183e6a'),
        (0.753, '#1d476c'), (1.0, '#1d476c'),
    ),
    "H2": (
        (0.0, '#1c4770'), (0.03, '#1c4770'), (0.089, '#183f6c'), (0.16, '#18406e'),
        (0.232, '#19416e'), (0.309, '#1a426e'), (0.461, '#1a436f'), (0.612, '#1a436f'),
        (0.685, '#1a436f'), (0.829, '#1a426f'), (0.871, '#214e6f'), (1.0, '#214e6f'),
    ),
    "H3": (
        (0.0, '#205074'), (0.026, '#205074'), (0.088, '#1c4670'), (0.238, '#1e4b72'),
        (0.316, '#1f4d74'), (0.392, '#1f4c73'), (0.467, '#1e4b73'), (0.617, '#1e4a72'),
        (0.691, '#1e4a72'), (0.839, '#1e4b73'), (0.928, '#265973'), (1.0, '#265973'),
    ),
}

TONGUE_INK = "#1e4b75"  # face gradient end at the back edge

WEDGE_X = 2168.0  # the fill between the strokes closes here

A_ANCHORS = (  # (x, y, tangent deg, len_in, len_out): left tip, apex, saddle, crest, tail tip
    (262.0, 517.5, -24.65, 0.0, 183.3),
    (548.0, 17.0, 0.00, 124.1, 111.4),
    (1090.0, 329.0, 0.00, 184.5, 244.0),
    (1658.0, 152.0, 0.00, 229.1, 329.8),
    (2327.0, 520.0, 17.89, 335.2, 0.0),
)

B_ANCHORS = (  # split off A, dive shoulder, valley, wing, tail tip
    (548.0, 17.0, 0.00, 0.0, 144.8),
    (1240.6, 820.4, 0.00, 494.8, 397.0),
    (2088.3, 451.2, 0.00, 249.6, 82.8),
    (2327.0, 520.0, 17.89, 121.4, 0.0),
)

V1_ANCHORS = (
    (729.3, 123.2, 38.51, 0.0, 240.7),
    (1365.4, 807.9, 20.31, 420.0, 0.0),
)

V2_ANCHORS = (
    (862.6, 227.2, 35.95, 0.0, 185.0),
    (1575.4, 728.0, 23.67, 471.4, 0.0),
)

V3_ANCHORS = (
    (1020.9, 316.5, 18.50, 0.0, 275.9),
    (1741.3, 620.5, 42.07, 258.8, 0.0),
)

V4_ANCHORS = (
    (1165.8, 323.0, -9.37, 0.0, 300.8),
    (1909.1, 507.3, 51.47, 449.4, 0.0),
)

H1_ANCHORS = (
    (715.2, 308.0, 14.12, 0.0, 476.1),
    (1388.3, 235.8, -28.09, 272.5, 0.0),
)

H2_ANCHORS = (
    (806.8, 505.9, 6.56, 0.0, 522.4),
    (1624.4, 153.3, -2.65, 199.5, 0.0),
)

H3_ANCHORS = (
    (938.2, 689.1, 0.64, 0.0, 511.6),
    (1806.6, 174.3, 18.11, 249.9, 0.0),
)

GROUND_ANCHORS = (
    (262.0, 518.5, -3.31, 0.0, 178.7),
    (797.2, 488.0, -3.31, 178.7, 0.0),
)
# fmt: on


def _build(anchors):
    """Cubic segments from PyA3EDA-style anchors (x, y, tangent deg, li, lo).

    One tangent per anchor makes every chain G1-continuous by construction."""
    segs = []
    for (x1, y1, a1, _li1, lo1), (x2, y2, a2, li2, _lo2) in pairwise(anchors):
        r1, r2 = math.radians(a1), math.radians(a2)
        segs.append(
            (
                (x1, y1),
                (x1 + lo1 * math.cos(r1), y1 + lo1 * math.sin(r1)),
                (x2 - li2 * math.cos(r2), y2 - li2 * math.sin(r2)),
                (x2, y2),
            )
        )
    return tuple(segs)


A_SEGS = _build(A_ANCHORS)
B_SEGS = _build(B_ANCHORS)
GROUND_SEGS = _build(GROUND_ANCHORS)
MESH_ORDER = ("V1", "V2", "V3", "V4", "H1", "H2", "H3")
MESH = {
    "V1": _build(V1_ANCHORS),
    "V2": _build(V2_ANCHORS),
    "V3": _build(V3_ANCHORS),
    "V4": _build(V4_ANCHORS),
    "H1": _build(H1_ANCHORS),
    "H2": _build(H2_ANCHORS),
    "H3": _build(H3_ANCHORS),
}

LTIP = A_SEGS[0][0]
APEX = A_SEGS[0][3]
RTIP = A_SEGS[-1][3]


def _bez(seg, t):
    """Point on one cubic segment at parameter t."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = seg
    u = 1 - t
    return (
        u**3 * x0 + 3 * u**2 * t * x1 + 3 * u * t**2 * x2 + t**3 * x3,
        u**3 * y0 + 3 * u**2 * t * y1 + 3 * u * t**2 * y2 + t**3 * y3,
    )


def sample_chain(segs, n=60):
    """Dense polyline of a segment chain (n points per segment)."""
    pts = []
    for seg in segs:
        pts += [_bez(seg, i / n) for i in range(n)]
    pts.append(segs[-1][3])
    return pts


def _rev(seg):
    """The same cubic travelled backwards."""
    p0, c1, c2, p3 = seg
    return (p3, c2, c1, p0)


def split_at_x(seg, x_target, lo=0.0, hi=1.0):
    """De Casteljau split of one cubic at the parameter where x = x_target."""
    for _ in range(48):
        mid = (lo + hi) / 2
        if _bez(seg, mid)[0] < x_target:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = seg

    def lerp(a, b):
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    p01 = lerp((x0, y0), (x1, y1))
    p12 = lerp((x1, y1), (x2, y2))
    p23 = lerp((x2, y2), (x3, y3))
    p012 = lerp(p01, p12)
    p123 = lerp(p12, p23)
    pm = lerp(p012, p123)
    return ((x0, y0), p01, p012, pm), (pm, p123, p23, (x3, y3))


def _d(segs, close=False, move=True):
    """SVG path data for a chain of cubic segments."""
    parts = []
    if move:
        parts.append(f"M {segs[0][0][0]:.1f},{segs[0][0][1]:.1f}")
    for _, c1, c2, p3 in segs:
        parts.append(f"C {c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}")
    if close:
        parts.append("Z")
    return " ".join(parts)


def _line_seg(p, q):
    """A straight connector expressed as a degenerate cubic."""
    mx1 = (2 * p[0] + q[0]) / 3, (2 * p[1] + q[1]) / 3
    mx2 = (p[0] + 2 * q[0]) / 3, (p[1] + 2 * q[1]) / 3
    return (p, mx1, mx2, q)


def split_chain_at_x(segs, x_target):
    """The sub-chain of *segs* left of x_target (splitting the containing segment)."""
    left = []
    for seg in segs:
        if seg[3][0] <= x_target:
            left.append(seg)
        else:
            l_, _ = split_at_x(seg, x_target)
            left.append(l_)
            break
    return left


def _subchain(segs, xa, xb):
    """The part of a chain between screen x = xa and x = xb."""
    left = split_chain_at_x(segs, xb)
    kept = []
    for seg in left:
        if seg[3][0] <= xa:
            continue
        if seg[0][0] < xa:
            _l, r_ = split_at_x(seg, xa)
            kept.append(r_)
        else:
            kept.append(seg)
    return kept


def sheet_path():
    """The closed sheet region between the strokes.

    B splits from A at SPLIT (its cap sits on A's stroke) and the fill closes
    again where the centerlines come within one stroke width (WEDGE_X);
    outside that span the sheet is bounded by single merged lines."""
    sx = B_SEGS[0][0][0]
    a_chain = _subchain(A_SEGS, sx, WEDGE_X)
    b_chain = split_chain_at_x(B_SEGS, WEDGE_X)
    segs = (
        a_chain + [_line_seg(a_chain[-1][3], b_chain[-1][3])] + [_rev(s) for s in reversed(b_chain)]
    )
    return _d(segs, close=True)


def face_path():
    """The flank face: ascent, A over the apex to the split, B's crest+dive to
    the junction, ground line home."""
    sx = B_SEGS[0][0][0]
    a_top = _subchain(A_SEGS, A_SEGS[0][3][0], sx)
    b_left = split_chain_at_x(B_SEGS, GROUND_SEGS[-1][3][0])
    segs = [A_SEGS[0]] + a_top + b_left + [_rev(s) for s in reversed(GROUND_SEGS)]
    return _d(segs, close=True)


def mark_extents(sw=SW_OUTLINE):
    """Ink bounding box (x0, y0, x1, y1) of the mark including its outline."""
    pts = sample_chain(A_SEGS) + sample_chain(B_SEGS)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    r = sw / 2
    return min(xs) - r, min(ys) - r, max(xs) + r, max(ys) + r


def _color(c, dark):
    """The palette color *c*, remapped for the dark scheme when asked."""
    return DARK_MAP.get(c, c) if dark else c


def gradients(dark=False):
    """All linearGradient defs, in mark coordinates (userSpaceOnUse)."""
    (x1, y1), (x2, y2) = LTIP, RTIP
    y2 = y2 - GRAD_RISE
    axis = f'gradientUnits="userSpaceOnUse" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
    defs = []
    for gid, table in (
        ("gfill", FILL_STOPS),
        ("glineA", OUTLINE_A_STOPS),
        ("glineB", OUTLINE_B_STOPS),
    ):
        rows = "".join(f'<stop offset="{o:g}" stop-color="{_color(c, dark)}"/>' for o, c in table)
        defs.append(f'<linearGradient id="{gid}" {axis}>{rows}</linearGradient>')
    for name in MESH_ORDER:
        (sx, sy) = MESH[name][0][0]
        (ex, ey) = MESH[name][-1][3]
        rows = "".join(
            f'<stop offset="{o:g}" stop-color="{_color(c, dark)}"/>' for o, c in MESH_STOPS[name]
        )
        defs.append(
            f'<linearGradient id="gm{name}" gradientUnits="userSpaceOnUse" '
            f'x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}">{rows}</linearGradient>'
        )
    # the face gradient runs on to the back edge: the tongue is stroked with
    # the SAME gradient, so face and tongue are one seamless paint field
    face_pts = sample_chain([A_SEGS[0], B_SEGS[0]])
    fy0 = min(p[1] for p in face_pts)
    fy1 = max(p[1] for p in face_pts)
    gy1 = max(p[1] for p in sample_chain(GROUND_SEGS)) + SW_GROUND / 2
    scale = (fy1 - fy0) / (gy1 - fy0)
    rows = "".join(
        f'<stop offset="{o * scale:.3f}" stop-color="{_color(c, dark)}"/>' for o, c in FACE_STOPS
    )
    rows += f'<stop offset="1" stop-color="{_color(TONGUE_INK, dark)}"/>'
    defs.append(
        f'<linearGradient id="gface" gradientUnits="userSpaceOnUse" '
        f'x1="0" y1="{fy0:.1f}" x2="0" y2="{gy1:.1f}">{rows}</linearGradient>'
    )
    return "<defs>" + "".join(defs) + "</defs>"


def mark_group(dark=False, mono=False, boost=False):
    """The full mark as defs + paths, in measured coordinates.

    mono=True: white line-art (docs header bar). boost=True: micro-size
    stroke widths for favicons."""
    sw_mesh, sw_line = (SW_MESH_BOOST, SW_OUTLINE_BOOST) if boost else (SW_MESH, SW_OUTLINE)
    sw_ground = sw_mesh if boost else SW_GROUND
    if mono:
        defs, fill, face_fill = "", "none", "none"
        ink_a = ink_b = WHITE
        mesh_ref = dict.fromkeys(MESH_ORDER, WHITE)
    else:
        defs = gradients(dark)
        fill, face_fill = "url(#gfill)", "url(#gface)"
        ink_a, ink_b = "url(#glineA)", "url(#glineB)"
        mesh_ref = {name: f"url(#gm{name})" for name in MESH_ORDER}
    stroke = 'fill="none" stroke-linecap="round" stroke-linejoin="round"'
    parts = [defs, f'<path d="{sheet_path()}" fill="{fill}" stroke="none"/>']
    parts += [
        f'<path d="{_d(MESH[name])}" {stroke} stroke="{mesh_ref[name]}" stroke-width="{sw_mesh}"/>'
        for name in MESH_ORDER
    ]
    if face_fill != "none":
        parts.append(f'<path d="{face_path()}" fill="{face_fill}" stroke="none"/>')
    ink_g = ink_b if (mono or boost) else "url(#gface)"
    parts.append(
        f'<path d="{_d(GROUND_SEGS)}" {stroke} stroke="{ink_g}" stroke-width="{sw_ground}"/>'
    )
    for segs, ink in ((B_SEGS, ink_b), (A_SEGS, ink_a)):
        parts.append(f'<path d="{_d(segs)}" {stroke} stroke="{ink}" stroke-width="{sw_line}"/>')
    return "\n".join(parts)


def icon_svg(dark=False, boost=False, pad=20):
    """Square 512-box mark: logo.svg, logo-dark.svg and (boosted) favicon.svg."""
    sw = SW_OUTLINE_BOOST if boost else SW_OUTLINE
    x0, y0, x1, y1 = mark_extents(sw)
    s = (512 - 2 * pad) / (x1 - x0)
    tx = pad - x0 * s
    ty = (512 - (y1 - y0) * s) / 2 - y0 * s
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" '
        f'width="512" height="512">\n<g transform="translate({tx:.2f},{ty:.2f}) '
        f'scale({s:.4f})">\n{mark_group(dark=dark, boost=boost)}\n</g>\n</svg>'
    )


def _wide_svg(sw, group, pad):
    """The mark in its own box: viewBox is the ink extents plus pad, nothing else.

    The counterpart to icon_svg's square canvas. An icon has to be square, so that one
    centres a 2.5:1 drawing in a 512 box and leaves ~32% of the height empty above and
    below. Set inline in a layout at a given height, that box renders the mark at about
    a third of the size the height implies, so anything that is not an icon takes this.
    """
    x0, y0, x1, y1 = mark_extents(sw)
    w = math.ceil(x1 - x0 + 2 * pad)
    h = math.ceil(y1 - y0 + 2 * pad)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}">\n<g transform="translate({pad - x0:.2f},{pad - y0:.2f})">\n'
        f"{group}\n</g>\n</svg>"
    )


def header_svg(pad=8):
    """White mono boosted line-art in a tight wide viewBox for the docs header bar."""
    return _wide_svg(SW_OUTLINE_BOOST, mark_group(mono=True, boost=True), pad)


def mark_svg(dark=False, pad=8):
    """The colored mark in the same tight box: logo.svg and logo-dark.svg."""
    return _wide_svg(SW_OUTLINE, mark_group(dark=dark), pad)


# -------------------------------------------------------------- documents ---
# Lockup composition, measured from the reference raster: the wordmark runs
# 1.24x the mark's width, the gap under the mark is 0.225x the mark's height.
WORD_TO_MARK = 1.24
GAP_FRAC = 0.225
MARGIN_FRAC = 0.006


def _wordmark_text(x, y, size, dark=False):
    """The "ChemRefine" wordmark as a live two-tspan `<text>` element."""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-weight="{FONT_WEIGHT}" '
        f'font-size="{size:.2f}" letter-spacing="{LS}">'
        f'<tspan fill="{_color(CHEM_NAVY, dark)}">Chem</tspan>'
        f'<tspan fill="{_color(REFINE_TEAL, dark)}">Refine</tspan></text>'
    )


def fit_wordmark(wd: Path):
    """Measure 'ChemRefine' at size 100 and fit it under the mark."""
    probe = wd / "probe.svg"
    probe.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2000 600">'
        + _wordmark_text(0, 300, 100)
        + "</svg>"
    )
    out = subprocess.run(  # noqa: S603 - fixed args, probe file we just wrote
        ["inkscape", str(probe), "--query-all"],  # noqa: S607 - snap-managed, not on a fixed path
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    boxes = []
    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) == 5:
            _, x, y, w, h = parts
            boxes.append((float(x), float(y), float(w), float(h)))
    px0 = min(b[0] for b in boxes)
    py0 = min(b[1] for b in boxes)
    px1 = max(b[0] + b[2] for b in boxes)
    py1 = max(b[1] + b[3] for b in boxes)
    mx0, my0, mx1, my1 = mark_extents()
    mark_w, mark_h = mx1 - mx0, my1 - my0
    word_w = WORD_TO_MARK * mark_w
    size = 100 * word_w / (px1 - px0)
    scale = size / 100
    margin = MARGIN_FRAC * word_w
    canvas_w = word_w + 2 * margin
    x = margin - px0 * scale
    word_top = mark_h + GAP_FRAC * mark_h
    baseline = word_top - (py0 - 300) * scale
    canvas_h = baseline + (py1 - 300) * scale + margin
    return x, baseline, size, math.ceil(canvas_w), math.ceil(canvas_h)


def lockup_svg(fit, dark=False):
    """The full mark-over-wordmark lockup for the *fit* from fit_wordmark()."""
    x, baseline, size, canvas_w, canvas_h = fit
    mx0, my0, mx1, _my1 = mark_extents()
    tx = (canvas_w - (mx1 - mx0)) / 2 - mx0
    ty = -my0
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas_w} {canvas_h}" '
        f'width="{canvas_w}" height="{canvas_h}">\n'
        f'<g transform="translate({tx:.2f},{ty:.2f})">\n{mark_group(dark=dark)}\n</g>\n'
        f"{_wordmark_text(x, baseline, size, dark)}\n</svg>"
    )


def text_to_path(src: Path, dst: Path):
    """Flatten *src*'s live text to outlines via inkscape, so no font ships."""
    subprocess.run(  # noqa: S603 - fixed args over files this run created
        ["inkscape", str(src), "--export-text-to-path", "--export-plain-svg", "-o", str(dst)],  # noqa: S607 - snap-managed, not on a fixed path
        check=True,
        capture_output=True,
    )
    if not dst.exists():
        raise RuntimeError(f"inkscape produced no output: {dst}")


def export_png(svg: Path, png: Path, width: int, height: int | None = None):
    """Rasterize *svg* to PNG via inkscape: square by default, else by width.

    height=None keeps the SVG's own aspect (used for the lockup raster the
    README serves to PyPI, where raw SVG does not render)."""
    dims = ["-w", str(width)] + ([] if height is None else ["-h", str(height)])
    subprocess.run(  # noqa: S603 - fixed args over files this run created
        ["inkscape", str(svg), *dims, "-o", str(png)],  # noqa: S607 - snap-managed, not on a fixed path
        check=True,
        capture_output=True,
    )
    if not png.exists():
        raise RuntimeError(f"inkscape produced no output: {png}")


def _require_font():
    """Abort unless fontconfig resolves Arial Bold to real Arial."""
    out = subprocess.run(  # noqa: S603 - fixed args, diagnostic query
        ["fc-match", f"{FONT}:{FONT_WEIGHT}"],  # noqa: S607 - fontconfig on PATH by design
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if f'"{FONT}"' not in out:
        raise RuntimeError(f"fontconfig resolves {FONT} {FONT_WEIGHT} to: {out.strip()}")


def main() -> int:
    """Regenerate the whole brand kit into docs/assets/, mirroring GUI_SUBSET to the package."""
    if shutil.which("inkscape") is None:
        print("error: inkscape is required (text->path, PNG export)")
        return 1
    _require_font()
    ASSETS.mkdir(parents=True, exist_ok=True)
    GUI_BRAND.mkdir(parents=True, exist_ok=True)
    wd = Path.home() / "chemrefine_logo_build"
    wd.mkdir(exist_ok=True)
    try:
        (ASSETS / "logo.svg").write_text(mark_svg())
        (ASSETS / "logo-dark.svg").write_text(mark_svg(dark=True))
        (ASSETS / "favicon.svg").write_text(icon_svg(boost=True))
        (ASSETS / "logo-header.svg").write_text(header_svg())
        (wd / "favicon.svg").write_text(icon_svg(boost=True))
        # The square display canvas is a raster source and nothing else - no page or
        # stylesheet ever wanted a mark with a third of its height empty - so it lives
        # here for the export and is not part of the kit.
        (wd / "icon.svg").write_text(icon_svg())

        fit = fit_wordmark(wd)
        (ASSETS / "logo-wordmark-src.svg").write_text(lockup_svg(fit))
        (wd / "light.svg").write_text(lockup_svg(fit))
        (wd / "dark.svg").write_text(lockup_svg(fit, dark=True))
        text_to_path(wd / "light.svg", wd / "logo-wordmark.svg")
        text_to_path(wd / "dark.svg", wd / "logo-wordmark-dark.svg")
        shutil.copy2(wd / "logo-wordmark.svg", ASSETS / "logo-wordmark.svg")
        shutil.copy2(wd / "logo-wordmark-dark.svg", ASSETS / "logo-wordmark-dark.svg")

        # Two optical sizes, so two sources. The boost is a 16-48 px treatment and the
        # docstring says so; rasterizing everything from it put stroke widths meant for a
        # 16 px tab onto a 512 px launcher tile, where the drawing reads as a heavier mark
        # than the one on every other surface. Above the boosted range the display mark is
        # the source, which is what makes "for 16-48 px legibility only" true of the code.
        sizes = {
            "favicon-16.png": (16, "favicon.svg"),
            "favicon-32.png": (32, "favicon.svg"),
            "favicon-48.png": (48, "favicon.svg"),
            "apple-touch-icon.png": (180, "icon.svg"),
            "icon-192.png": (192, "icon.svg"),
            "icon-512.png": (512, "icon.svg"),
        }
        for name, (px, source) in sizes.items():
            export_png(wd / source, wd / name, px, px)
            shutil.copy2(wd / name, ASSETS / name)

        export_png(wd / "logo-wordmark.svg", wd / "logo-wordmark-1200.png", 1200)
        shutil.copy2(wd / "logo-wordmark-1200.png", ASSETS / "logo-wordmark-1200.png")

        try:
            from PIL import Image

            img = Image.open(ASSETS / "favicon-48.png")
            img.save(ASSETS / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
            print("wrote favicon.ico (16/32/48)")
        except ImportError:
            print("Pillow not found - skipped favicon.ico")

        # Last, from the files just written rather than from a second render: the two
        # roots have to be the same bytes, and copying is the only way to say that
        # without depending on the export being deterministic.
        for name in GUI_SUBSET:
            shutil.copy2(ASSETS / name, GUI_BRAND / name)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    print(f"brand kit written to {ASSETS}")
    print(f"GUI subset mirrored to {GUI_BRAND} ({len(GUI_SUBSET)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
