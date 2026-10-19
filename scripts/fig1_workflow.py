#!/usr/bin/env python3
"""Fig. 1: study workflow, from nozzle geometry to operating window.

Numbers on the cards are read from the grid when the figure is drawn.

    python scripts/fig1_workflow.py
"""
from __future__ import annotations
import csv
import itertools
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch             # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l8_style as S                                                        # noqa: E402
from l8_paths import CANONICAL_GRID, require, save_figure                   # noqa: E402

FIGNAME = "Fig1_workflow.png"
GRID = require(CANONICAL_GRID, "the canonical 32-case grid")
DPI = 600

W = S.PAGE_W_MM
PAD = 1.5                  # canvas margin
PP = 2.0                   # padding inside a phase panel
GAP = 6.0                  # between cards in a phase, room for an arrow
GGAP = 7.0                 # between the two phases
CARD_H = 40.0              # room for the three-line name of the third card
LABEL_H = 5.0              # the phase label above the cards
CW = (W - 2 * PAD - 4 * PP - 3 * GAP - GGAP) / 5          # five cards in the row
H = PAD + PP + CARD_H + 1.5 + LABEL_H + PAD

SIM_C = S.MESH_C           # simulation phase accent
SUR_C = S.ANNOT            # surrogate phase accent
TYPE = dict(title=8.0, sub=6.8, phase=7.2)


def grid_facts():
    rows = list(csv.DictReader(open(GRID)))
    dias = sorted({int(round(float(r["needle_um"]))) for r in rows})
    pres = sorted({float(r["pressure_kPa"]) for r in rows})
    return len(rows), dias, pres


def draw_icon(ax, kind, cx, cy, w, h, col):
    """A small schematic glyph per stage, which makes the diagram readable at a glance where
    bare text boxes are not. Drawn, not imported, so the figure stays a single self-contained
    script."""
    import numpy as np
    from matplotlib.patches import Polygon, Rectangle
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    lw = 1.0
    if kind == "needle":                      # cartridge, funnel and tip, in section
        # Axial proportions are the real ones, taken from l8_geometry rather than chosen: the
        # cartridge, the funnel and the needle occupy z1, z2 - z1 and z3 - z2 of the z3 total.
        from l8_geometry import GEOM
        f1 = GEOM["z1"] / GEOM["z3"]                      # end of the straight cartridge
        f2 = GEOM["z2"] / GEOM["z3"]                      # end of the funnel, needle entry
        r_tip = .07 * h                                   # exaggerated, see above
        ax.add_patch(Polygon([(x0, y1), (x0 + f1 * w, y1), (x0 + f2 * w, cy + r_tip),
                              (x1, cy + r_tip), (x1, cy - r_tip),
                              (x0 + f2 * w, cy - r_tip), (x0 + f1 * w, y0), (x0, y0)],
                             closed=True, facecolor=col, alpha=0.18, edgecolor=col,
                             lw=lw, zorder=5))
    elif kind == "mesh":                      # the axisymmetric half-section, triangulated
        # This draws the domain that is actually solved: the axisymmetric half-section, axis at the
        # bottom on a dashed centerline, triangulated and graded into the needle where the elements
        # really are smallest (M12 = 12 elements across the tip radius).
        from l8_geometry import GEOM
        f1 = GEOM["z1"] / GEOM["z3"]                # end of the straight cartridge
        f2 = GEOM["z2"] / GEOM["z3"]                # end of the funnel, needle entry
        R_IN, R_MID, R_TIP = h, 0.46 * h, 0.21 * h  # exaggerated, see above

        def wall(f):
            """Half-section wall height at axial fraction f, in icon units."""
            if f <= f1:
                return R_IN
            if f < f2:
                return R_IN + (R_MID - R_IN) * (f - f1) / (f2 - f1)
            return R_TIP

        # Axial stations, clustered into the contraction and the needle.
        fs = sorted({0.0, f2 - 1e-4, f2 + 1e-4, 1.0}
                    | {f1 * (i / 5) for i in range(6)}
                    | {f1 + (f2 - f1) * (i / 3) for i in range(1, 3)}
                    | {f2 + (1 - f2) * (i / 4) ** 0.8 for i in range(1, 5)})
        nr = 4
        cols = [[(x0 + f * w, y0 + wall(f) * j / nr) for j in range(nr + 1)] for f in fs]

        for c in cols:                                    # radial lines, wall to axis
            ax.plot([p[0] for p in c], [p[1] for p in c], color=col, lw=0.35, zorder=5)
        for j in range(nr + 1):                           # axial lines
            ax.plot([c[j][0] for c in cols], [c[j][1] for c in cols],
                    color=col, lw=0.35, zorder=5)
        for a, b in zip(cols, cols[1:]):                  # one diagonal per cell
            if abs(a[0][0] - b[0][0]) < 1e-9:
                continue                                  # the two stations at the step
            for j in range(nr):
                ax.plot([a[j][0], b[j + 1][0]], [a[j][1], b[j + 1][1]],
                        color=col, lw=0.28, alpha=0.75, zorder=5)
        # The symmetry axis, dashed, the standard mark for a half-section.
        ax.plot([x0, x1], [y0, y0], color=col, lw=0.7, ls=(0, (5, 2)), zorder=6)
    elif kind == "damage":                    # a saturating damage curve
        t = np.linspace(0, 1, 60)
        ax.plot(x0 + t * w, y0 + h * (1 - np.exp(-3.2 * t)), color=col, lw=1.4, zorder=5)
        ax.plot([x0, x1], [y0, y0], color=col, lw=0.6, zorder=5)
        ax.plot([x0, x0], [y0, y1], color=col, lw=0.6, zorder=5)
    elif kind == "tracers":                   # paths converging into the tip
        for k, off in enumerate(np.linspace(-0.42, 0.42, 5)):
            t = np.linspace(0, 1, 40)
            ax.plot(x0 + t * w, cy + off * h * (1 - t) ** 1.7,
                    color=col, lw=0.8, alpha=0.85, zorder=5)
            ax.plot([x0 + w], [cy], marker="o", ms=1.6, color=col, zorder=6)
    elif kind == "gp":                        # a smooth response surface as nested contours
        for k, f in enumerate((1.0, 0.72, 0.44)):
            t = np.linspace(0, np.pi, 40)
            ax.plot(cx + f * .48 * w * np.cos(t), y0 + h * (0.15 + f * 0.7 * np.sin(t)),
                    color=col, lw=0.8, alpha=0.5 + 0.2 * k, zorder=5)
        ax.plot([x0, x1], [y0, y0], color=col, lw=0.6, zorder=5)
    elif kind == "window":                    # a feasible region inside a plane
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=col,
                               lw=0.6, zorder=5))
        t = np.linspace(0, 1, 40)
        ax.fill_between(x0 + t * w, y0, y0 + h * (0.85 - 0.7 * t ** 1.6),
                        color=col, alpha=0.18, zorder=5)
        ax.plot(x0 + t * w, y0 + h * (0.85 - 0.7 * t ** 1.6), color=col, lw=1.2, zorder=6)



def card(ax, x, y, w, h, col, kind, title, sub, horizontal=False):
    """One stage: white card, accent border, schematic, bold name, caption line(s).
    Returns the text artists so the guards can measure them against the card."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                lw=0.9, edgecolor=col, facecolor="white", zorder=3))
    if horizontal:                                     # icon left, text right
        draw_icon(ax, kind, x + 9.0, y + h / 2, 12.0, h - 6.0, col)
        t = ax.text(x + 18.0, y + h / 2 + 0.6, title, ha="left", va="bottom",
                    fontsize=TYPE["title"], fontweight="bold", color=S.INK, zorder=6)
        s = ax.text(x + 18.0, y + h / 2 - 0.6, sub, ha="left", va="top",
                    fontsize=TYPE["sub"], color=S.MUTE, zorder=6, linespacing=1.3)
        return [t, s]
    draw_icon(ax, kind, x + w / 2, y + h - 9.5, w * 0.62, 11.0, col)
    t = ax.text(x + w / 2, y + h - 17.5, title, ha="center", va="top", fontsize=TYPE["title"],
                fontweight="bold", color=S.INK, zorder=6, linespacing=1.15)
    s = ax.text(x + w / 2, y + 2.2, sub, ha="center", va="bottom", fontsize=TYPE["sub"],
                color=S.MUTE, zorder=6, linespacing=1.3)
    return [t, s]


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.9,
                                 color=S.MUTE, zorder=4, shrinkA=0, shrinkB=0))


def main():
    n_cases, dias, pres = grid_facts()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]})
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()

    y_card = PAD + PP
    xs = []
    x = PAD + PP
    for i in range(5):
        xs.append(x)
        x += CW + (GAP if i not in (2,) else PP + GGAP + PP)

    # Phase panels: a pale field behind each group, its name in the top left corner.
    p1 = (xs[0] - PP, xs[2] + CW + PP)
    p2 = (xs[3] - PP, xs[4] + CW + PP)
    y_panel = y_card - PP
    h_panel = PP + CARD_H + 1.5 + LABEL_H
    texts = []
    for (a, b), label in ((p1, f"Simulation, each of the {n_cases} conditions "
                                f"({len(dias)} nozzles × {len(pres)} pressures)"),
                          (p2, f"Surrogate, fitted once over the {n_cases} cases")):
        ax.add_patch(FancyBboxPatch((a, y_panel), b - a, h_panel,
                                    boxstyle="round,pad=0,rounding_size=2.0", lw=0,
                                    facecolor=S.PANEL, zorder=1))
        texts.append(ax.text(a + 2.0, y_panel + h_panel - 1.2, label, ha="left", va="top",
                             fontsize=TYPE["phase"], fontweight="bold", color=S.MUTE, zorder=2))

    stages = [
        ("needle", "Nozzle and\nbioink", f"cartridge and nozzle\nØ{dias[0]}–{dias[-1]} µm\n"
                                         f"power-law bioink", SIM_C),
        ("mesh", "Axisymmetric\nFEM", f"Navier–Stokes\nwith inertia\n"
                                      f"{int(pres[0])}–{int(pres[-1])} kPa", SIM_C),
        ("damage", "Damage\nlaw", "entrance extension\nand nozzle shear", SIM_C),
        ("gp", "Gaussian\nprocess", f"{n_cases} solved cases\n< 1 ms per query", SUR_C),
        # The two limits are the window's definition (Methods, Operating-window definition), set before any case was
        # solved; no result value is drawn.
        ("window", "Operating\nwindow", "$D_t$ < 20 %\n$Q$ ≥ 1 µL/s", SUR_C),
    ]
    boxes = []
    for (kind, title, sub, col), x0 in zip(stages, xs):
        texts_card = card(ax, x0, y_card, CW, CARD_H, col, kind, title, sub)
        boxes.append(((x0, y_card, CW, CARD_H), texts_card))
    for i in range(4):
        arrow(ax, (xs[i] + CW + 0.6, y_card + CARD_H / 2), (xs[i + 1] - 0.6, y_card + CARD_H / 2))


    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for (x0, y0, w, h), arts in boxes:
        for t in arts:
            bb = t.get_window_extent(rend)
            (a0, b0), (a1, b1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
            assert x0 + 0.6 <= a0 and a1 <= x0 + w - 0.6 and y0 + 0.6 <= b0 and b1 <= y0 + h - 0.6, (
                f"{t.get_text().splitlines()[0]!r} spans x {a0:.1f}-{a1:.1f}, y {b0:.1f}-{b1:.1f} "
                f"outside its card x {x0:.1f}-{x0 + w:.1f}, y {y0:.1f}-{y0 + h:.1f}")
        tb = arts[0].get_window_extent(rend)
        sb = arts[1].get_window_extent(rend)
        assert sb.y1 <= tb.y0 - 1, f"{arts[0].get_text()!r}: name and caption line overlap"
    for p, (a, b) in zip(texts, (p1, p2)):
        bb = p.get_window_extent(rend)
        assert inv.transform((bb.x1, 0))[0] <= b - 1.0, f"phase label {p.get_text()!r} overflows"
    alltext = texts + [t for _, arts in boxes for t in arts]
    for ta, tb in itertools.combinations(alltext, 2):
        a, b = ta.get_window_extent(rend), tb.get_window_extent(rend)
        assert not (min(a.x1, b.x1) - max(a.x0, b.x0) > 0.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 0.5), (
            f"{ta.get_text()!r} overlaps {tb.get_text()!r}")

    for w_ in save_figure(fig, FIGNAME, DPI):          # save_figure checks the canvas margins
        print(f"  wrote {w_}")
    plt.close(fig)
    from PIL import Image
    wpx, hpx = Image.open(HERE.parent / "results" / "figures" / FIGNAME).size
    assert abs(wpx / DPI * 25.4 - W) < 0.2, f"width {wpx / DPI * 25.4:.1f} mm"
    print(f"  {wpx}x{hpx} px = {wpx / DPI * 25.4:.0f} x {hpx / DPI * 25.4:.1f} mm at {DPI} dpi; "
          f"cards {CW:.1f} mm wide")
    return 0


if __name__ == "__main__":
    sys.exit(main())
