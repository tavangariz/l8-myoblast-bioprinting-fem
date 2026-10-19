#!/usr/bin/env python3
"""Fig. 7: operating window and the effect of the applied constraint.

(A) Total damage over the pressure-diameter plane with the two criteria, Dt = 20 % and
Q = 1 µL/s, the 32 grid cases and the eleven FEM checks. (B) Total damage against flow rate for
each nozzle.

    python scripts/fig7_operating_window.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l8_style as S
import l8_surrogate as SUR

from l8_paths import save_figure

FIGNAME = "Fig7_operating_window.png"

# The excluded region's hatch.
HATCH = "///"
HATCH_C = "white"

D_LIMIT = 20.0        # % predicted total damage; a screening cutoff, NOT a
                      # measured viability limit; nothing here measures survival
Q_FLOOR = 1.0         # µL/s, below this it is too slow to print with
# The reference throughput is looked up in the grid, never written here as a constant.
REF_DIA_UM, REF_P_KPA = 200, 200.0     # the paper's reference operating point


def reference_flow(rows):
    """Q at the paper's reference operating point, from the canonical grid."""
    hits = [r for r in rows if int(r["needle_um"]) == REF_DIA_UM
            and float(r["pressure_kPa"]) == REF_P_KPA]
    assert len(hits) == 1, f"{len(hits)} rows at Ø{REF_DIA_UM}/{REF_P_KPA:.0f} kPa"
    return float(hits[0]["Q_uLs"])


def main():
    rows = SUR.load_grid()
    Q_REF = reference_flow(rows)
    gp_dt, gp_logq, z = SUR.fit_all(rows)

    by = defaultdict(list)
    for r in rows:
        by[int(r["needle_um"])].append(r)
    for d in by:
        by[d].sort(key=lambda r: r["pressure_kPa"])
    dias = sorted(by)

    accept = [r for r in rows if r["Dt_pct"] < D_LIMIT and r["Q_uLs"] >= Q_FLOOR]
    print(f"  inside the window: {len(accept)} of {len(rows)}  "
          f"(D_t < {D_LIMIT:.0f} % and Q >= {Q_FLOOR:.0f} µL/s)")

    plt.rcParams.update(S.rc(S.PAPER))
    # Hatch stroke weight is a global, not a per-artist keyword, so it is set with the rest
    # of the style. At 600 dpi the default 1.0 pt prints as a hairline over a filled contour.
    plt.rcParams["hatch.linewidth"] = 1.5
    P = S.PAPER
    # The height includes room for panel (a)'s key below the axes: a key outside costs vertical space
    # rather than covering data.
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(S.PAGE_W_IN, 3.6000), dpi=P["dpi"],
                                   gridspec_kw=dict(wspace=0.56))
    cax = axa.inset_axes([1.035, 0.0, 0.042, 1.0], transform=axa.transAxes)

    pp = np.linspace(25, 400, 260)
    dd = np.linspace(min(dias), max(dias), 240)
    PG, DG = np.meshgrid(pp, dd)
    DT = gp_dt.predict(z(PG, DG)).reshape(PG.shape)
    LQ = gp_logq.predict(z(PG, DG)).reshape(PG.shape)

    # A rectangular colour bar, not an arrow.
    lo, hi = float(np.floor(DT.min())), float(np.ceil(DT.max()))
    levels = np.arange(lo, hi + 1.0, 1.0)
    assert DT.min() >= levels[0] and DT.max() <= levels[-1], (
        f"surface spans {DT.min():.2f}-{DT.max():.2f} outside levels "
        f"{levels[0]:.0f}-{levels[-1]:.0f}; the bar would need an extend cap")
    cf = axa.contourf(PG, DG, DT, levels=levels, cmap=S.CMAP_DT, zorder=1,
                      extend="neither")
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label(S.LBL["damage_t"], fontsize=P["label"],
                 fontweight="bold")
    cb.ax.tick_params(labelsize=P["tick"])

    # The two constraints, drawn as constraints
    c1 = axa.contour(PG, DG, DT, levels=[D_LIMIT], colors="white", linewidths=2.6)
    # The throughput constraint is drawn as an explicit contour, dashed to distinguish it from the
    # solid damage limit, rather than left for the reader to infer from the edge of an unfilled
    # region.
    axa.contour(PG, DG, 10.0 ** LQ, levels=[Q_FLOOR], colors="white",
                linewidths=2.2, linestyles="--")
    # The two constraint lines are named in the legend, not on the field.


    # The excluded area is a region, not thirty-two markers.
    infeasible = (DT >= D_LIMIT) | (10.0 ** LQ < Q_FLOOR)
    _frac = float(infeasible.mean())
    assert 0.02 < _frac < 0.98, (
        f"the excluded region covers {_frac*100:.1f} % of the plane; at that fraction it is "
        f"a background, not a region, and the constraints need re-reading")
    hatched = axa.contourf(PG, DG, infeasible.astype(float), levels=[0.5, 1.5],
                           colors="none", hatches=[HATCH], zorder=3)
    hatched.set_edgecolor(HATCH_C)
    hatched.set_linewidth(0.0)          # the boundary is the constraint contour, not this

    # The 32 solved conditions are drawn identically, because the caption counts them and a reader
    # must be able to count them off the panel.
    for r in rows:
        axa.plot(r["pressure_kPa"], r["needle_um"], marker="o", ms=P["ms"],
                 mfc="white", mec=S.INK, mew=1.1, ls="none", zorder=6, clip_on=True)

    # Independent FEM checks.
    checks_csv = (Path(__file__).resolve().parents[1] / "results" / "fem" / "runs"
                  / "2026-09-23_independent-checks" / "checks_vs_surrogate.csv")
    checks = list(csv.DictReader(open(checks_csv))) if checks_csv.exists() else []
    # Sized to fit inside the white of a grid circle: four checks fall within 2.5 kPa of a
    # solved condition, and a full-size diamond there would hide the circle the caption counts.
    ms_check = 0.6 * P["ms"]
    for c in checks:
        axa.plot(float(c["pressure_kPa"]), float(c["needle_um"]), marker="D", ms=ms_check,
                 mfc=S.ANNOT, mec="white", mew=0.5, ls="none", zorder=7, clip_on=True)
    print(f"  independent FEM checks drawn: {len(checks)}")

    axa.set_xlabel(S.LBL["pressure"])
    axa.set_ylabel(S.LBL["diameter"])
    S.panel_label(axa, "A")
    # Padded like the diameter axis below, so the 25 and 400 kPa columns print as whole
    # markers; the field itself still spans only the trained 25 to 400 kPa.
    _xpad = 0.045 * (400 - 25)
    axa.set_xlim(25 - _xpad, 400 + _xpad)
    # Pad the limits, so the Ø100 and Ø330 marker rows print as whole markers rather than as half-
    # circles on the spines, and every condition the caption counts can be counted off the figure.
    _pad = 0.045 * (max(dias) - min(dias))
    axa.set_ylim(min(dias) - _pad, max(dias) + _pad)
    axa.set_yticks(sorted(dias))   # the four SOLVED diameters, not 100/150/.../300
    axa.grid(False)
    # The two constraint contours are named here, in the key, instead of in boxes on the field.
    leg_a = axa.legend(handles=[
        Line2D([], [], marker="o", mfc="white", mec=S.INK, mew=1.1, ls="none",
               ms=P["ms"], label="FEM condition"),
    ] + ([Line2D([], [], marker="D", mfc=S.ANNOT, mec="white", mew=0.5, ls="none",
                 ms=ms_check, label="independent FEM check")] if checks else []) + [
        # The swatch is filled before it is hatched.
        Patch(facecolor=plt.get_cmap(S.CMAP_DT)(0.45), edgecolor=HATCH_C,
              hatch=HATCH, linewidth=0.6,
              label="outside the window"),
        Line2D([], [], color=S.INK, lw=2.6, label=f"$D_t$ = {D_LIMIT:.0f} %"),
        Line2D([], [], color=S.INK, lw=2.2, ls="--", label=f"$Q$ = {Q_FLOOR:.0f} µL/s"),
    ] + [
    # Two by two, not one row of four.
    ], loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=2, frameon=False,
        fontsize=P["legend"] - 0.5, handlelength=1.9, columnspacing=1.3,
        borderaxespad=0.0)

    for d in dias:
        rr = by[d]
        axb.semilogx([r["Q_uLs"] for r in rr], [r["Dt_pct"] for r in rr],
                     marker=S.dia_marker(d), ls="-",
                     color=S.dia_color(d), lw=P["lw_data"], ms=P["ms"],
                     mec="white", mew=0.8, label=S.dia_label(d), zorder=3)
    axb.axhline(D_LIMIT, color=S.THRESHOLD, lw=1.2, ls="--", zorder=1,
                label=f"$D_t$ = {D_LIMIT:.0f} % limit")
    # Named in the legend, not printed across the panel

    # The read-out is restricted to tips that actually reach Q_REF.
    at_ref, out_of_range = {}, []
    for d in dias:
        s = by[d]
        qs = [r["Q_uLs"] for r in s]
        if not (min(qs) <= Q_REF <= max(qs)):
            out_of_range.append(d)
            continue
        at_ref[d] = float(np.interp(np.log10(Q_REF), np.log10(qs), [r["Dt_pct"] for r in s]))
    # Named in the key.
    axb.axvline(Q_REF, ymax=0.975, color=S.ANNOT, lw=1.4, ls=":", zorder=2,
                label=f"$Q$ = {Q_REF:.1f} µL/s, reference point")
    # Journal rule: figure titles and legends go in the main manuscript, never inside the graphic
    # file.

    print("  at matched flow rate "
          + " · ".join(f"Ø{d} {at_ref[d]:.2f} %" for d in sorted(at_ref)))
    if out_of_range:
        print(f"  NOT quoted at Q = {Q_REF:.2f} µL/s (outside the computed range): "
              + ", ".join(f"Ø{d}" for d in out_of_range))

    axb.set_xlabel(S.LBL["flow_q"])
    axb.set_ylabel(S.LBL["damage_t"])
    S.panel_label(axb, "B")
    # Outside the axes, under the panel, like panel (a)'s key.
    leg_b = axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.150), ncol=2,
                       frameon=False, fontsize=P["legend"] - 0.5, handlelength=1.9,
                       columnspacing=1.3, borderaxespad=0.0)

    # No tight_layout: it re-solves the column widths and would undo the equal split, and it cannot
    # see an inset colour bar as something needing room.
    fig.subplots_adjust(left=0.085, right=0.975, top=0.9333, bottom=0.2895)
    S.snap_panel_labels(fig)

    # Centre panel (a)'s key under panel (a) and its colour bar. Anchored to the axes alone it
    # would sit left of the block it describes.
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _ab, _cb = axa.get_window_extent(_r), cax.get_window_extent(_r)
    # Measure the panel without its key.
    leg_a.set_visible(False)
    _bottom = axa.get_tightbbox(_r).y0
    leg_a.set_visible(True)
    leg_a.set_bbox_to_anchor(((0.5 * (_ab.x0 + _cb.x1)) / fig.bbox.width,
                              (_bottom - 0.018 * fig.bbox.height) / fig.bbox.height),
                             transform=fig.transFigure)

    # Measured, not assumed.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fig_bb = fig.bbox

    # Clamp panel (b)'s key to the canvas, measured.
    leg_b.set_visible(False)
    _bottom_b = axb.get_tightbbox(rend).y0
    leg_b.set_visible(True)
    leg_b.set_bbox_to_anchor((0.5 * (axb.get_window_extent(rend).x0 + axb.get_window_extent(rend).x1)
                              / fig.bbox.width,
                              (_bottom_b - 0.018 * fig.bbox.height) / fig.bbox.height),
                             transform=fig.transFigure)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    # The clamp leaves 1.2 mm clear; it left 6 px (0.25 mm), inside save_figure's canvas guard.
    _pad = 1.2 / 25.4 * fig.dpi
    _lb = leg_b.get_window_extent(rend)
    _over = max(0.0, _lb.x1 - (fig_bb.x1 - _pad)) - max(0.0, (fig_bb.x0 + _pad) - _lb.x0)
    if _over:
        _c = 0.5 * (_lb.x0 + _lb.x1) - _over
        leg_b.set_bbox_to_anchor((_c / fig.bbox.width, _lb.y0 / fig.bbox.height),
                                 transform=fig.transFigure)
        leg_b.set_loc("lower center")
        fig.canvas.draw()
        print(f"  key (b) shifted {_over:.0f} px left to clear the canvas edge")

    for name, leg, ax in (("(a)", leg_a, axa), ("(b)", leg_b, axb)):
        lb = leg.get_window_extent(rend)
        assert (lb.x0 >= fig_bb.x0 - 0.5 and lb.x1 <= fig_bb.x1 + 0.5
                and lb.y0 >= fig_bb.y0 - 0.5 and lb.y1 <= fig_bb.y1 + 0.5), (
            f"panel {name} legend runs off the canvas: {lb} vs {fig_bb}")
    # And clear of the axis title, not just of the axes: at least 1 mm between the title's
    # descenders and the key's first row.
    for name, leg, ax in (("(a)", leg_a, axa), ("(b)", leg_b, axb)):
        _gap = (ax.xaxis.label.get_window_extent(rend).y0 - leg.get_window_extent(rend).y1) / fig.dpi * 25.4
        assert _gap >= 1.0, f"panel {name} key sits {_gap:.2f} mm under its axis title; need 1 mm"
    for name, leg, ax in (("(a)", leg_a, axa), ("(b)", leg_b, axb)):
        lb, ab = leg.get_window_extent(rend), ax.get_window_extent(rend)
        assert lb.y1 <= ab.y0 + 0.5, (
            f"panel {name} key at y {lb.y0:.0f}-{lb.y1:.0f} still overlaps its axes "
            f"({ab.y0:.0f}-{ab.y1:.0f})")
        print(f"  key {name} sits {ab.y0 - lb.y1:.0f} px below the axes, "
              f"{lb.x1 - lb.x0:.0f} px wide on a {fig_bb.x1 - fig_bb.x0:.0f} px canvas")

    written = save_figure(fig, FIGNAME, P["dpi"])
    plt.close(fig)
    for w in written:
        print(f"  wrote {w}")


if __name__ == "__main__":
    main()
