#!/usr/bin/env python3
"""Mathematical audit of the measured curve tables in gen_logo.py.

Run after ANY edit to the tables, before regenerating the kit:

    python3 tools/check_logo.py

The generator only emits; this file carries the constraints that make the
tables a sound drawing rather than a pile of numbers:
  1. every chain is contiguous (segments share endpoints exactly) and G1
     (tangents collinear at interior joints)
  2. stroke B splits from A exactly at the apex and merges back ONTO A's
     tail tangentially - beyond the merge the tail is one single line
  3. every mesh curve's ends lie ON the stroke it fans out from; the ground
     line runs from the left tip to its junction on B
  4. each mesh family is ordered (attachment x monotonically increasing) and
     free of self-intersections - the families come from one surface grid,
     so same-family curves must never cross
  5. the silhouette polygon is simple; the flank face closes
  6. gradient stops ascend in offset, the fill ramp ascends in luminance,
     boosted strokes are thicker than display strokes
  7. (optional) ink-IoU resemblance gate against the original raster when
     CHEMREFINE_LOGO_REF points at it - the "looks almost the same" number
"""

import math
import os
import sys
from itertools import pairwise
from pathlib import Path

import gen_logo as g

ATTACH_TOL = 3.5  # px: mesh ends must sit on their stroke
G1_TOL_DEG = 2.5
FAMILY_MIN_DIST = 8.0


def fail(msg):
    """Report one failed constraint."""
    print(f"FAIL  {msg}")
    return 1


def ok(msg):
    """Report one satisfied constraint."""
    print(f"  ok  {msg}")
    return 0


def tangents(seg):
    """(incoming, outgoing) tangent angles of one cubic segment."""
    p0, c1, c2, p3 = seg
    a_in = math.atan2(c1[1] - p0[1], c1[0] - p0[0])
    a_out = math.atan2(p3[1] - c2[1], p3[0] - c2[0])
    return a_in, a_out


def angdiff(a, b):
    """Absolute angle difference in degrees, wrapped."""
    d = abs(a - b) % (2 * math.pi)
    return math.degrees(min(d, 2 * math.pi - d))


def poly_dist(p, pts):
    """Distance from point p to a polyline given as a point list."""
    return min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in pts)


def _orient(p, q, r):
    """Twice the signed area of triangle pqr."""
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def polygon_simple(pts):
    """True when the closed polygon through *pts* has no self-intersections."""
    n = len(pts)
    segs = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            a, b = segs[i]
            c, d = segs[j]
            d1, d2 = _orient(c, d, a), _orient(c, d, b)
            d3, d4 = _orient(a, b, c), _orient(a, b, d)
            if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
                return False
    return True


def rel_luminance(color):
    """WCAG relative luminance of an #rrggbb color."""
    chans = []
    for i in (1, 3, 5):
        c = int(color[i : i + 2], 16) / 255
        chans.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g_, b = chans
    return 0.2126 * r + 0.7152 * g_ + 0.0722 * b


def main() -> int:
    """Run every constraint; exit non-zero if any fails."""
    bad = 0

    # 1. chain contiguity + G1
    for name, segs in (("A", g.A_SEGS), ("B", g.B_SEGS)):
        contiguous = all(s1[3] == s2[0] for s1, s2 in pairwise(segs))
        bad += (
            ok(f"stroke {name}: segments contiguous")
            if contiguous
            else fail(f"stroke {name}: segment chain has a gap")
        )
        worst = 0.0
        for s1, s2 in pairwise(segs):
            worst = max(worst, angdiff(tangents(s1)[1], tangents(s2)[0]))
        if worst <= G1_TOL_DEG:
            bad += ok(f"stroke {name}: G1 at joints (worst {worst:.2f} deg)")
        else:
            bad += fail(f"stroke {name}: tangent break {worst:.1f} deg at a joint")

    # 2. shared apex split and shared tail
    bs = g.B_SEGS[0][0]
    a_pts = g.sample_chain(g.A_SEGS, 400)
    b_pts = g.sample_chain(g.B_SEGS, 400)
    dsplit = poly_dist(bs, a_pts)
    i0 = min(range(len(a_pts)), key=lambda k: math.hypot(a_pts[k][0] - bs[0], a_pts[k][1] - bs[1]))
    j0 = max(1, min(i0, len(a_pts) - 2))
    a_dir0 = math.atan2(a_pts[j0 + 1][1] - a_pts[j0 - 1][1], a_pts[j0 + 1][0] - a_pts[j0 - 1][0])
    dang = angdiff(a_dir0, tangents(g.B_SEGS[0])[0])
    if dsplit <= 2.0 and dang <= 4.0:
        bad += ok(
            f"B splits off A tangentially at {tuple(round(v) for v in bs)} "
            f"({dsplit:.1f}px, {dang:.1f} deg)"
        )
    else:
        bad += fail(f"B's split is off: {dsplit:.1f}px from A, {dang:.1f} deg")
    if g.A_SEGS[-1][3] == g.B_SEGS[-1][3]:
        bad += ok(f"A and B share the tail tip {g.A_SEGS[-1][3]}")
    else:
        bad += fail("A and B end at different points")
    d = angdiff(tangents(g.A_SEGS[-1])[1], tangents(g.B_SEGS[-1])[1])
    if d <= 8.0:
        bad += ok(f"tail tangents agree within {d:.1f} deg (one merged line)")
    else:
        bad += fail(f"tail tangents differ by {d:.1f} deg")
    aw = g.split_chain_at_x(g.A_SEGS, g.WEDGE_X)[-1]
    bw = g.split_chain_at_x(g.B_SEGS, g.WEDGE_X)[-1]
    gap = math.hypot(aw[3][0] - bw[3][0], aw[3][1] - bw[3][1])
    if abs(gap - g.SW_OUTLINE) <= 4.0:
        bad += ok(
            f"fill wedge closes at x={g.WEDGE_X:.0f} (stroke gap {gap:.1f} = one stroke width)"
        )
    else:
        bad += fail(f"WEDGE_X inconsistent: stroke gap there is {gap:.1f}px")

    # 3. attachments (H1 peels off A and merges back into A; H2/H3 run B->A)
    HOSTS = {
        "H1": (b_pts, a_pts, "B->A"),
        "H2": (b_pts, a_pts, "B->A"),
        "H3": (b_pts, a_pts, "B->A"),
    }
    for name in g.MESH_ORDER:
        segs = g.MESH[name]
        s_, e_ = segs[0][0], segs[-1][3]
        h1, h2, what = HOSTS.get(name, (a_pts, b_pts, "A->B"))
        d1, d2 = poly_dist(s_, h1), poly_dist(e_, h2)
        if max(d1, d2) <= ATTACH_TOL:
            bad += ok(f"{name} attached {what} ({d1:.1f}/{d2:.1f}px)")
        else:
            bad += fail(f"{name} detached: {d1:.1f}/{d2:.1f}px off its strokes")

    # 3b. tangential peels: every V top leaves A, and H1's merge rejoins A,
    # along A's own direction (the fused-ramp construction, like B's split)
    def a_dir_near(p):
        i = min(range(len(a_pts)), key=lambda k: math.hypot(a_pts[k][0] - p[0], a_pts[k][1] - p[1]))
        j = max(1, min(i, len(a_pts) - 2))
        return math.atan2(a_pts[j + 1][1] - a_pts[j - 1][1], a_pts[j + 1][0] - a_pts[j - 1][0])

    for name, seg_pt in (
        ("V1", 0),
        ("V2", 0),
        ("V3", 0),
        ("V4", 0),
        ("H1", -1),
    ):
        seg = g.MESH[name][0 if seg_pt == 0 else -1]
        p, c = (seg[0], seg[1]) if seg_pt == 0 else (seg[3], seg[2])
        tang = math.atan2(c[1] - p[1], c[0] - p[0])
        da = abs((math.degrees(tang - a_dir_near(p)) + 180) % 360 - 180)
        da = min(da, 180 - da)
        side = "start" if seg_pt == 0 else "end"
        if da <= 6.0:
            bad += ok(f"{name} {side} peels tangentially off A ({da:.1f} deg)")
        else:
            bad += fail(f"{name} {side} not tangential to A ({da:.1f} deg)")
    gseg = g.GROUND_SEGS
    flush = (gseg[0][0][1] + g.SW_GROUND / 2) - (g.LTIP[1] + g.SW_OUTLINE / 2)
    if abs(flush) <= 0.1:
        bad += ok("ground and stroke A end flush at the left tip (same bottom edge)")
    else:
        bad += fail(f"ground tip not flush with A: bottom edges differ {flush:+.1f}px")
    d_start = math.hypot(
        gseg[0][0][0] - g.LTIP[0], gseg[0][0][1] - g.LTIP[1] - (g.SW_OUTLINE - g.SW_GROUND) / 2
    )
    d_end = poly_dist(gseg[-1][3], b_pts)
    if d_start <= 0.2 and d_end <= ATTACH_TOL:
        bad += ok(f"ground line runs left tip -> junction on B ({d_end:.1f}px)")
    else:
        bad += fail(
            f"ground line endpoints wrong (start {d_start:.2f}px off ltip, end {d_end:.1f}px off B)"
        )

    # 3b. mesh containment: no curve may poke outside the sheet
    def ytop_at(x):
        return min(a_pts, key=lambda q: abs(q[0] - x))[1]

    def ybot_at(x):
        return min(b_pts, key=lambda q: abs(q[0] - x))[1]

    pokes = []
    for name in g.MESH_ORDER:
        for q in g.sample_chain(g.MESH[name], 40):
            # anything within the outline stroke's own footprint is covered
            if q[1] < ytop_at(q[0]) - g.SW_OUTLINE / 2 or q[1] > ybot_at(q[0]) + g.SW_OUTLINE / 2:
                pokes.append(name)
                break
    if not pokes:
        bad += ok("every mesh curve stays inside the sheet")
    else:
        bad += fail(f"mesh curves poke outside the sheet: {pokes}")

    # 4. family order + no same-family crossings
    for fam, names in (
        ("V", [n for n in g.MESH_ORDER if n.startswith("V")]),
        ("H", [n for n in g.MESH_ORDER if n.startswith("H")]),
    ):
        starts = [g.MESH[n][0][0][0] for n in names]
        ends = [g.MESH[n][-1][3][0] for n in names]
        if all(b > a for a, b in pairwise(starts)) and all(b > a for a, b in pairwise(ends)):
            bad += ok(f"{fam}-family attachment order monotone")
        else:
            bad += fail(f"{fam}-family attachments out of order: {starts} / {ends}")
        polys = [g.sample_chain(g.MESH[n], 80) for n in names]
        worst = 1e9
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                worst = min(worst, min(poly_dist(p, polys[j]) for p in polys[i][::4]))
        if worst >= FAMILY_MIN_DIST:
            bad += ok(f"{fam}-family curves never touch (min gap {worst:.0f}px)")
        else:
            bad += fail(f"{fam}-family curves cross or touch (gap {worst:.1f}px)")

    # 5. silhouette + face. The fill region's boundary is A from the apex to
    # the wedge, then B back to the apex - the shared ascent is not part of
    # the enclosed region's boundary (both strokes traverse it together).
    a_body = [p for p in a_pts[121:] if p[0] <= g.WEDGE_X]
    b_body = [p for p in b_pts if p[0] <= g.WEDGE_X]
    # B rides within a hair of A just after the split (under the stroke
    # overlap); those coincident points would zigzag the polygon - drop them
    b_body = [p for p in b_body if min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in a_body) > 2.5]
    sil = a_body + b_body[::-1]
    if polygon_simple(sil[::6]):
        bad += ok("silhouette is simple")
    else:
        bad += fail("silhouette self-intersects")
    x0, y0, x1, y1 = g.mark_extents()
    if g.face_path().endswith("Z"):
        bad += ok("flank face path closes")
    else:
        bad += fail("flank face path does not close")

    # 6. gradients + stroke discipline
    for nm, stops in (
        ("fill", g.FILL_STOPS),
        ("outline A", g.OUTLINE_A_STOPS),
        ("outline B", g.OUTLINE_B_STOPS),
        ("face", g.FACE_STOPS),
    ):
        offs = [o for o, _ in stops]
        if all(b > a for a, b in pairwise(offs)):
            bad += ok(f"{nm} gradient offsets ascend")
        else:
            bad += fail(f"{nm} gradient offsets broken: {offs}")

    def grad_color(stops, off):
        if off <= stops[0][0]:
            return stops[0][1]
        for (o1, c1), (o2, c2) in pairwise(stops):
            if off <= o2:
                f = (off - o1) / (o2 - o1)
                v1 = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
                v2 = [int(c2[i : i + 2], 16) for i in (1, 3, 5)]
                r_, g_, b2_ = (round(a_ + f * (b_ - a_)) for a_, b_ in zip(v1, v2, strict=True))
                return f"#{r_:02x}{g_:02x}{b2_:02x}"
        return stops[-1][1]

    x_lo, x_hi = g.LTIP[0], g.RTIP[0]
    worst_dc = 0
    HOST_STOPS = {
        "H1": (g.OUTLINE_B_STOPS, g.OUTLINE_A_STOPS),
        "H2": (g.OUTLINE_B_STOPS, g.OUTLINE_A_STOPS),
        "H3": (g.OUTLINE_B_STOPS, g.OUTLINE_A_STOPS),
    }
    for name in g.MESH_ORDER:
        (sx, _sy), (ex, _ey) = g.MESH[name][0][0], g.MESH[name][-1][3]
        so, eo = (sx - x_lo) / (x_hi - x_lo), (ex - x_lo) / (x_hi - x_lo)
        s_host, e_host = HOST_STOPS.get(name, (g.OUTLINE_A_STOPS, g.OUTLINE_B_STOPS))
        ends = (g.MESH_STOPS[name][0][1], g.MESH_STOPS[name][-1][1])
        for stops, off, have in ((s_host, so, ends[0]), (e_host, eo, ends[1])):
            expect = grad_color(stops, off)
            dc = max(abs(int(expect[i : i + 2], 16) - int(have[i : i + 2], 16)) for i in (1, 3, 5))
            worst_dc = max(worst_dc, dc)
    # measured end stops sit ~22px inboard of the junctions and the original
    # itself lets H-line ink run lighter than stroke B at a joint (delta 20
    # measured) - the strokes paint over the mesh ends anyway
    if worst_dc <= 24:
        bad += ok(
            f"mesh ink stays close to its outline at every junction "
            f"(worst channel delta {worst_dc})"
        )
    else:
        bad += fail(f"mesh ink breaks at a junction (channel delta {worst_dc})")

    if all(all(b_ > a_ for (a_, _), (b_, _) in pairwise(g.MESH_STOPS[n])) for n in g.MESH_ORDER):
        bad += ok("mesh gradient offsets ascend")
    else:
        bad += fail("mesh gradient offsets not ascending")
    if (
        g.OUTLINE_B_STOPS[0][1]
        == g.OUTLINE_B_STOPS[1][1]
        == grad_color(g.OUTLINE_A_STOPS, g.OUTLINE_B_STOPS[1][0])
    ):
        bad += ok("B wears A's exact ink over the apex split")
    else:
        bad += fail("B's ink differs from A's at the split")

    # every drawn extremum of A and B sits AT an anchor with an exactly
    # horizontal tangent (no angle at the extremum), and B's tip tangent is
    # tied to A's so the merged tail leaves as one line
    for nm, segs, tab in (("A", g.A_SEGS, g.A_ANCHORS), ("B", g.B_SEGS, g.B_ANCHORS)):
        pts = g.sample_chain(segs, 400)
        problems = []
        prev = 0.0
        for i in range(1, len(pts)):
            dy = pts[i][1] - pts[i - 1][1]
            if abs(dy) < 1e-9:
                continue
            if prev and (dy > 0) != (prev > 0):
                p = pts[i - 1]
                near = min(tab, key=lambda r: math.hypot(r[0] - p[0], r[1] - p[1]))
                d = math.hypot(near[0] - p[0], near[1] - p[1])
                if d > 3.0 or abs(near[2]) > 0.05:
                    problems.append(f"({p[0]:.0f},{p[1]:.0f}) d={d:.1f} ang={near[2]:.2f}")
            prev = dy
        if not problems:
            bad += ok(f"{nm}: every extremum sits at a horizontal-tangent anchor")
        else:
            bad += fail(f"{nm}: extremum off-anchor or angled: {'; '.join(problems)}")
    if abs(g.A_ANCHORS[-1][2] - g.B_ANCHORS[-1][2]) <= 0.1:
        bad += ok("tail tip tangents are tied (A and B leave as one line)")
    else:
        bad += fail(
            f"tail tip tangents differ: A {g.A_ANCHORS[-1][2]:.2f} vs "
            f"B {g.B_ANCHORS[-1][2]:.2f} deg"
        )

    def seg_flips(seg):
        pts = g.sample_chain([seg], 60)
        flips, prev = 0, 0.0
        for i in range(1, len(pts) - 1, 2):
            ax_, ay_ = (pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            bx_, by_ = (pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            cr = ax_ * by_ - ay_ * bx_
            if abs(cr) < 1e-3:
                continue
            if prev and (cr > 0) != (prev > 0):
                flips += 1
            prev = cr
        return flips

    # a cubic has at most ONE inflection; more within a segment means a
    # degenerate loop or control-point blowup - a visible "bounce"
    chains = {"A": g.A_SEGS, "B": g.B_SEGS, "GROUND": g.GROUND_SEGS}
    chains.update({n: g.MESH[n] for n in g.MESH_ORDER})
    worst = [
        f"{name} seg{si}:{seg_flips(seg)}"
        for name, segs in chains.items()
        for si, seg in enumerate(segs)
        if seg_flips(seg) > 1
    ]
    if not worst:
        bad += ok("no curve bounces: every segment holds one inflection at most")
    else:
        bad += fail(f"segment wobbles detected: {', '.join(worst)}")

    # joint-level wobble: the heading along B's dive may only turn one way
    cp = g.sample_chain(g.B_SEGS[:1], 80)
    head = [
        math.atan2(cp[i + 1][1] - cp[i][1], cp[i + 1][0] - cp[i][0]) for i in range(len(cp) - 1)
    ]
    changes, prev = 0, 0.0
    for i in range(1, len(head)):
        dh = (head[i] - head[i - 1] + math.pi) % (2 * math.pi) - math.pi
        if abs(dh) < 0.002:
            continue
        if prev and (dh > 0) != (prev > 0):
            changes += 1
        prev = dh
    if changes <= 1:
        bad += ok(f"crest curvature changes sign {changes}x (steepen, then flatten)")
    else:
        bad += fail(f"crest wiggles: {changes} curvature sign changes")

    lums = [rel_luminance(c) for _, c in g.FILL_STOPS]
    if all(b > a for a, b in pairwise(lums)):
        bad += ok("fill ramp luminance strictly ascends navy -> green")
    else:
        bad += fail("fill ramp luminance not ascending")
    if (
        g.SW_OUTLINE_BOOST > g.SW_OUTLINE
        and g.SW_MESH_BOOST > g.SW_MESH
        and g.SW_OUTLINE >= g.SW_MESH
    ):
        bad += ok("stroke-width discipline holds")
    else:
        bad += fail("stroke-width discipline broken")

    # 7. report + optional resemblance gate
    print(f"mark ink extents {x1 - x0:.0f} x {y1 - y0:.0f} (aspect {(x1 - x0) / (y1 - y0):.2f})")
    ref = os.environ.get("CHEMREFINE_LOGO_REF", "")
    if ref and Path(ref).is_file():
        try:
            import numpy as np
            from PIL import Image, ImageDraw

            rim = Image.open(ref).convert("RGB")
            ra = np.asarray(rim).astype(int)[:850]
            rmask = ra.sum(2) < 690
            canvas = Image.new("L", (rim.width, rim.height), 0)
            dd = ImageDraw.Draw(canvas)
            dd.polygon([tuple(p) for p in sil], fill=255)
            face = (
                g.sample_chain([g.A_SEGS[0]], 60)
                + g.sample_chain([g.B_SEGS[0]], 60)
                + g.sample_chain(g.split_chain_at_x(g.B_SEGS, g.GROUND_SEGS[-1][3][0]), 60)
                + g.sample_chain(g.GROUND_SEGS, 60)[::-1]
            )
            dd.polygon([tuple(q) for q in face], fill=255)
            for name in g.MESH_ORDER:
                dd.line(
                    [tuple(p) for p in g.sample_chain(g.MESH[name], 60)],
                    fill=255,
                    width=int(g.SW_MESH),
                )
            for segs, w in (
                (g.A_SEGS, g.SW_OUTLINE),
                (g.B_SEGS, g.SW_OUTLINE),
                (g.GROUND_SEGS, g.SW_GROUND),
            ):
                dd.line([tuple(p) for p in g.sample_chain(segs, 60)], fill=255, width=int(w))
            om = np.asarray(canvas)[:850] > 0
            iou = (om & rmask).sum() / (om | rmask).sum()
            if iou >= 0.96:
                bad += ok(f"resemblance gate: ink IoU vs reference {iou:.3f} (>= 0.96)")
            else:
                bad += fail(f"resemblance gate: ink IoU {iou:.3f} < 0.96")

            # mesh-course gate: our centerlines rasterized at mesh width vs the
            # reference's own mesh ink (darker than fill, inside the sheet,
            # away from the strokes) - geometric fit of the weave itself
            dom = Image.new("L", (rim.width, rim.height), 0)
            ImageDraw.Draw(dom).polygon([tuple(p) for p in sil], fill=255)
            excl = Image.new("L", (rim.width, rim.height), 0)
            de = ImageDraw.Draw(excl)
            for segs, w in (
                (g.A_SEGS, g.SW_OUTLINE),
                (g.B_SEGS, g.SW_OUTLINE),
                (g.GROUND_SEGS, g.SW_GROUND),
            ):
                de.line([tuple(p) for p in g.sample_chain(segs, 120)], fill=255, width=int(w) + 26)
            ours_m = Image.new("L", (rim.width, rim.height), 0)
            dm_ = ImageDraw.Draw(ours_m)
            for name in g.MESH_ORDER:
                dm_.line(
                    [tuple(p) for p in g.sample_chain(g.MESH[name], 120)],
                    fill=255,
                    width=int(g.SW_MESH),
                )
            dom_a = np.asarray(dom)[:850] > 0
            ok_zone = dom_a & (np.asarray(excl)[:850] == 0)
            ref_mesh = (ra.sum(2) < 300) & ok_zone
            our_mesh = (np.asarray(ours_m)[:850] > 0) & ok_zone
            miou = (ref_mesh & our_mesh).sum() / max((ref_mesh | our_mesh).sum(), 1)
            if miou >= 0.70:
                bad += ok(f"mesh-course gate: weave IoU vs reference {miou:.3f} (>= 0.70)")
            else:
                bad += fail(f"mesh-course gate: weave IoU {miou:.3f} < 0.70")

            # per-line ink-color gate: reference core ink along each curve vs
            # the color our gradient tables produce at the same spot
            def lab(c):
                v = np.array(c, float) / 255.0
                v = np.where(v > 0.04045, ((v + 0.055) / 1.055) ** 2.4, v / 12.92)
                M = np.array(
                    [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
                )
                xyz = M @ v / np.array([0.95047, 1.0, 1.08883])
                f_ = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
                return np.array([116 * f_[1] - 16, 500 * (f_[0] - f_[1]), 200 * (f_[1] - f_[2])])

            def hex_rgb(h):
                return [int(h[i : i + 2], 16) for i in (1, 3, 5)]

            all_chains = {
                "A": g.sample_chain(g.A_SEGS, 400),
                "B": g.sample_chain(g.B_SEGS, 400),
            }  # ground is painted with the face gradient (audited above)
            all_chains.update({n: g.sample_chain(g.MESH[n], 400) for n in g.MESH_ORDER})
            worst_line, worst_med = "", 0.0
            for name, pts in all_chains.items():
                others = np.vstack([np.array(p) for k, p in all_chains.items() if k != name])
                arr = np.array(pts)
                arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(arr, axis=0).T))])
                des = []
                for t in np.linspace(0.15, 0.85, 8):
                    i = int(np.searchsorted(arc, t * arc[-1]).clip(0, len(arr) - 1))
                    p = arr[i]
                    if np.min(np.hypot(*(others - p).T)) < 22:
                        continue
                    win = ra[
                        max(0, int(p[1]) - 7) : int(p[1]) + 7, max(0, int(p[0]) - 7) : int(p[0]) + 7
                    ].reshape(-1, 3)
                    core = np.median(win[np.argsort(win.sum(1))[: max(6, len(win) // 3)]], 0)
                    if name in ("A", "B"):
                        off = (p[0] - g.LTIP[0]) / (g.RTIP[0] - g.LTIP[0])
                        stops = g.OUTLINE_A_STOPS if name == "A" else g.OUTLINE_B_STOPS
                    else:
                        P0, P1 = arr[0], arr[-1]
                        d_ = P1 - P0
                        off = float(np.clip((p - P0) @ d_ / (d_ @ d_), 0, 1))
                        stops = g.MESH_STOPS[name]
                    have = hex_rgb(grad_color(stops, off))
                    des.append(float(np.linalg.norm(lab(core) - lab(have))))
                med = float(np.median(des)) if des else 0.0
                if med > worst_med:
                    worst_med, worst_line = med, name
            if worst_med <= 5.0:
                bad += ok(
                    f"ink-color gate: worst per-line median dE {worst_med:.1f} ({worst_line}) <= 5"
                )
            else:
                bad += fail(f"ink-color gate: {worst_line} median dE {worst_med:.1f} > 5")
        except ImportError:
            print("      (numpy/Pillow missing - resemblance gate skipped)")
    else:
        print("      (set CHEMREFINE_LOGO_REF to the original raster to run the resemblance gate)")

    print("all checks passed" if bad == 0 else f"{bad} check(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
