#!/usr/bin/env python3
"""Values read off the figures of Ning et al. (2018) that ning2018_data.py marks "digitised".

Ning L, Betancourt N, Schreyer DJ, Chen X. Characterization of cell damage and proliferative ability
during and after bioprinting. ACS Biomater Sci Eng. 2018;4:3906-18. doi:10.1021/acsbiomaterials.8b00714

The figures are not redistributed here. Render the page of the figure from the published PDF (400 dpi
for Figs. 4 and 7, 500 dpi for Supporting Information Fig. S4), crop it to one panel with its axes, and
pass the image:

    python scripts/digitise_ning_figures.py fig4  PANEL.png   # Fig. 4C/4D, mass flow against nozzle diameter at 200 kPa
    python scripts/digitise_ning_figures.py fig7  PANEL.png   # Fig. 7A/7B, damage against nozzle diameter at fixed pressure
    python scripts/digitise_ning_figures.py figS4 PANEL.png   # Fig. S4, RSC96 damage at a matched flow rate

Axes are calibrated from the tick marks or gridlines detected in the image; markers are located by
colour (and, for Fig. 4, by template matching against a ring). Each command prints the mean and the ends
of the error bar for every nozzle diameter.
"""
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, signal  # noqa: F401  (ndimage is used by fig7 and figS4)


def fig4(path):
    """Fig. 4C/4D: red ring markers by template matching (ring radius from 7 to 11 px), y from the dotted
    gridlines in 0.05 g/s steps, x from the 100-350 um ticks; the error bar is the red extent above and below the ring."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    red = ((R > 170) & (G < 150) & (B < 150)).astype(float)
    dark = (R < 120) & (G < 120) & (B < 120)
    gray = (abs(R - G) < 25) & (abs(G - B) < 25) & (R < 200)
    h, w = red.shape
    # x axis = longest dark row; y axis = longest dark column
    xrow = int(np.argmax(dark.sum(1))); ycol = int(np.argmax(dark.sum(0)))
    # gridlines: rows above the axis where gray pixels cover > 25 % of the plot width (dotted lines)
    cov = gray[:, ycol + 5:].sum(1) / (w - ycol - 5)
    rows = np.where(cov > 0.25)[0]; rows = rows[rows < xrow - 3]
    grid = [int(np.mean(g)) for g in np.split(rows, np.where(np.diff(rows) > 3)[0] + 1) if len(g)]
    top = min(grid); exp = np.linspace(top, xrow, 7)               # 0.30 ... 0.00, equally spaced
    ys = [min(grid + [xrow], key=lambda g: abs(g - e)) for e in exp]
    assert max(abs(np.array(ys) - exp)) < 6, (ys, exp)
    vals = np.linspace(0.30, 0.0, 7)
    print("gridline rows", ys)
    ky = np.polyfit(ys, vals, 1)
    # x ticks below the axis
    below = dark[xrow + 3:xrow + 14, :].sum(0); idx = np.where(below >= 6)[0]
    tx = [int(np.mean(g)) for g in np.split(idx, np.where(np.diff(idx) > 2)[0] + 1) if len(g)]
    tx = [t for t in tx if t >= ycol - 3]
    print("x ticks", tx)
    kx = np.polyfit(tx, np.linspace(100, 350, len(tx)), 1)
    yy, xx = np.mgrid[-14:15, -14:15]; rr = np.hypot(yy, xx)
    best = None
    for r0 in (7, 8, 9, 10, 11):
        tpl = ((rr > r0 - 1.5) & (rr < r0 + 1.5)).astype(float); tpl -= tpl.mean()
        c = signal.correlate(red - red.mean(), tpl, mode="same")
        if best is None or c.max() > best[0]: best = (c.max(), r0, c)
    _, r0, corr = best; print("ring radius px", r0)
    for d in (100, 200, 250, 330):
        xc = np.polyval(np.polyfit(np.linspace(100, 350, len(tx)), tx, 1), d)
        x0, x1 = int(xc) - 12, int(xc) + 13
        sub = corr[:xrow + 12, x0:x1]; iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        cy, cx = iy, x0 + ix
        col = red[:xrow + 12, cx - 1:cx + 2].any(1)
        top = cy - r0 - 1
        while top - 1 >= 0 and col[top - 1]: top -= 1
        bot = cy + r0 + 1
        while bot + 1 < len(col) and col[bot + 1]: bot += 1
        print(f"D={d}: mean {np.polyval(ky, cy):.4f} g/s, red column extent {np.polyval(ky, bot):.4f}-{np.polyval(ky, top):.4f} g/s (x check {np.polyval(kx, cx):.0f} um)")


def fig7(path):
    """Fig. 7A/7B: red circles are the measured mean, red bars the error bars; y from the 0/10/20/30 % ticks,
    x from the 50-400 um ticks."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    dark = (R < 110) & (G < 110) & (B < 110)
    red = (R > 170) & (G < 140) & (B < 140)
    # y axis: the column with the longest dark vertical run; x axis: the row with the longest dark horizontal run
    colsum = dark.sum(0); xaxis_col = int(np.argmax(colsum))
    rowsum = dark.sum(1); yaxis_row = int(np.argmax(rowsum))
    # y ticks: short dark runs just left of the y axis
    left = dark[:, xaxis_col - 12:xaxis_col - 2].sum(1)
    ty = [int(np.mean(g)) for g in np.split(np.where(left >= 6)[0], np.where(np.diff(np.where(left >= 6)[0]) > 2)[0] + 1) if len(g)]
    # x ticks: short dark runs just below the x axis
    below = dark[yaxis_row + 3:yaxis_row + 12, :].sum(0)
    idx = np.where(below >= 5)[0]
    tx = [int(np.mean(g)) for g in np.split(idx, np.where(np.diff(idx) > 2)[0] + 1) if len(g)]
    ty = [t for t in ty if t <= yaxis_row + 2]; tx = [t for t in tx if t >= xaxis_col - 3]
    print("y ticks px", ty, "x ticks px", tx)
    assert len(ty) == 4 and len(tx) == 8, (ty, tx)
    ky = np.polyfit(ty, [30, 20, 10, 0], 1)              # value = ky0*px + ky1
    kx = np.polyfit(tx, np.linspace(50, 400, len(tx)), 1)
    lab, n = ndimage.label(red)
    out = []
    for d in (100, 200, 250, 330):
        px = np.polyval(np.polyfit(np.linspace(50, 400, len(tx)), tx, 1), d)
        band = red[:, int(px) - 22:int(px) + 23]
        ys = np.where(band.any(1))[0]
        # marker = ring: rows where the red run is wide (> 8 px); bar = narrow rows
        widths = band.sum(1)
        ring = np.where(widths > 8)[0]
        yc = ring.mean()
        out.append((d, np.polyval(ky, yc), np.polyval(ky, ys.max()), np.polyval(ky, ys.min())))
    for d, m, lo, hi in out:
        print(f"D={d} um: mean {m:.2f} %, bar {lo:.2f}-{hi:.2f} % (half-width {(hi-lo)/2:.2f})")


def figS4(path):
    """Fig. S4: red squares are the measured mean with red error bars, blue circles the simulation of the
    same study; y from the 0-25 % ticks, the four conditions located by the red marker columns."""
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    dark = (R < 90) & (G < 90) & (B < 90)
    red = (R > 180) & (G < 90) & (B < 90)
    blue = (B > 150) & (R < 90) & (G < 120)
    cs = dark.sum(0); band = np.where(cs > 0.5 * cs.max())[0]
    band = band[band < dark.shape[1] // 2]            # the y axis is the thick vertical line on the left
    ycol = int(band.min())                            # its left edge
    left = dark[:, max(0, ycol - 28):ycol - 3].sum(1)
    idx = np.where(left >= 6)[0]
    ty = [int(np.mean(g)) for g in np.split(idx, np.where(np.diff(idx) > 2)[0] + 1) if len(g)]
    ty = ty[:6]                                       # 25..0; rows below are the rotated x labels
    print("y tick rows", ty)
    assert len(ty) == 6, ty
    ky = np.polyfit(ty, [25, 20, 15, 10, 5, 0], 1)
    lab, n = ndimage.label(red)
    labels = ["430 kPa, 100 um", "100 kPa, 200 um", "60 kPa, 250 um", "40 kPa, 330 um"]
    found = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if xs.min() <= ycol + 20 or ys.min() < ty[0] + 60 or len(xs) < 200:
            continue                                   # axis, legend swatch, specks
        rows = np.arange(ys.min(), ys.max() + 1)
        width = np.array([(lab[r] == i).sum() for r in rows])
        marker = rows[width >= 0.6 * width.max()]      # the filled square, wider than the bar
        found.append((xs.mean(), marker.mean(), ys.min(), ys.max(), width.max()))
    # a blue circle drawn over a red bar splits it: merge the pieces of each category (same x)
    found.sort()
    groups = []
    for f in found:
        if groups and abs(groups[-1][-1][0] - f[0]) < 15:
            groups[-1].append(f)
        else:
            groups.append([f])
    assert len(groups) == 4, groups
    found = []
    for g in groups:
        m = max(g, key=lambda t: t[4])                 # the piece holding the square marker
        found.append((m[0], m[1], min(t[2] for t in g), max(t[3] for t in g)))
    for (cx, cy, top, bot), name in zip(found, labels):
        bl = blue[:, int(cx) - 8:int(cx) + 9]
        byr = np.where(bl.any(1))[0]
        byr = byr[(byr > ty[0] + 60)]
        sim = np.polyval(ky, byr.mean()) if len(byr) else float("nan")
        print(f"{name}: experimental {np.polyval(ky, cy):.2f} %, bar {np.polyval(ky, bot):.2f}-{np.polyval(ky, top):.2f} %, simulated {sim:.2f} %")


if __name__ == "__main__":
    {"fig4": fig4, "fig7": fig7, "figS4": figS4}[sys.argv[1]](sys.argv[2])
