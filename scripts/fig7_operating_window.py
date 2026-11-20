#!/usr/bin/env python3
"""Fig. 7: operating window and the effect of the applied constraint.

(A) Total damage over the pressure-diameter plane with the two criteria, Dt = 20 % and
Q = 1 µL/s, the 32 grid cases, the ranges of entrance stress (440-9251 Pa) and wall shear stress
(100-1000 Pa) on which the two damage branches were calibrated, the eleven FEM checks of the
published damage law and the four of the 20 % limit on the total damage (Additional file 1,
Table S5). (B) Total damage against flow rate for each nozzle.

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
from scipy.interpolate import PchipInterpolator
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


# The column layout, in mm, computed from measured ink (B under A).
PAD_MM = 1.5          # canvas margin around the outermost ink
PANEL_H_MM = 50.0     # plot-box height of each panel (side by side they were about 52 mm)
CB_GAP_MM = 2.5       # panel (a) to its colour bar, as in the side-by-side figure
CB_W_MM = 3.5         # colour-bar width
KEY_GAP_MM = 1.2      # axis title to the first row of its key
ROW_GAP_MM = 5.0      # panel (a)'s key to panel (b)'s letter


def layout_column(fig, axa, cax, axb, leg_a, leg_b):
    """Place (a) over (b) on the 170 mm canvas from measured ink; set the canvas height to fit.

    Ink extents are in points (type, ticks, bar), so they barely move when the boxes are
    resized; three passes land on a fixed point. The keys are measured but hidden while the
    panel ink is measured, because a key is a child of its axes and would otherwise count as
    the panel's own text, and the panel letters are snapped with the keys hidden for the same
    reason.
    """
    W = S.PAGE_W_MM
    for _ in range(3):
        for leg in (leg_a, leg_b):
            leg.set_visible(False)
        S.snap_panel_labels(fig)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        mm = 25.4 / fig.dpi

        def ext(a):
            box, ink = a.get_window_extent(r), a.get_tightbbox(r)
            return ((box.x0 - ink.x0) * mm, (ink.x1 - box.x1) * mm,
                    (box.y0 - ink.y0) * mm, (ink.y1 - box.y1) * mm)   # left, right, below, above

        ea, eb, ec = ext(axa), ext(axb), ext(cax)
        for leg in (leg_a, leg_b):
            leg.set_visible(True)
        fig.canvas.draw()
        ka = leg_a.get_window_extent(r).height * mm
        kb = leg_b.get_window_extent(r).height * mm
        left = PAD_MM + max(ea[0], eb[0])
        wa = W - PAD_MM - left - CB_GAP_MM - CB_W_MM - ec[1]
        wb = W - PAD_MM - left - eb[1]
        H = (PAD_MM + ea[3] + PANEL_H_MM + ea[2] + KEY_GAP_MM + ka + ROW_GAP_MM
             + eb[3] + PANEL_H_MM + eb[2] + KEY_GAP_MM + kb + PAD_MM)
        fig.set_size_inches(W / 25.4, H / 25.4)
        ya = H - PAD_MM - ea[3] - PANEL_H_MM
        yb = PAD_MM + kb + KEY_GAP_MM + eb[2]
        axa.set_position([left / W, ya / H, wa / W, PANEL_H_MM / H])
        cax.set_position([(left + wa + CB_GAP_MM) / W, ya / H, CB_W_MM / W, PANEL_H_MM / H])
        axb.set_position([left / W, yb / H, wb / W, PANEL_H_MM / H])
        # Each key hangs KEY_GAP_MM under its own axis title. (a)'s is centred under the panel
        # and its colour bar, the block it describes; (b)'s under its panel.
        leg_a.set_loc("upper center")
        leg_a.set_bbox_to_anchor(((left + 0.5 * (wa + CB_GAP_MM + CB_W_MM)) / W,
                                  (ya - ea[2] - KEY_GAP_MM) / H), transform=fig.transFigure)
        leg_b.set_loc("upper center")
        leg_b.set_bbox_to_anchor(((left + 0.5 * wb) / W, (yb - eb[2] - KEY_GAP_MM) / H),
                                 transform=fig.transFigure)
    for leg in (leg_a, leg_b):
        leg.set_visible(False)
    S.snap_panel_labels(fig)
    for leg in (leg_a, leg_b):
        leg.set_visible(True)
    print(f"  layout: panel (a) {wa:.1f} x {PANEL_H_MM:.0f} mm, panel (b) {wb:.1f} x "
          f"{PANEL_H_MM:.0f} mm, on a {W:.0f} x {H:.1f} mm canvas")


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
    # One column, B under A, so that both panels are larger and easier to read.
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(S.PAGE_W_IN, 7.0), dpi=P["dpi"])
    # The colour bar is its own axes, placed by layout_column() at a fixed gap and width in mm
    # beside panel (a), so it never takes width from (a) and does not stretch when (a) widens.
    cax = fig.add_axes([0.9, 0.6, 0.02, 0.3])

    pp = np.linspace(25, 400, 260)
    dd = np.linspace(min(dias), max(dias), 240)
    PG, DG = np.meshgrid(pp, dd)
    # The surrogate emulates Dn; the window is drawn on the total Dt = D0 + (1 - D0) Dn (Eq. 25)
    DT = SUR.total_from_dn(gp_dt.predict(z(PG, DG))).reshape(PG.shape)
    LQ = gp_logq.predict(z(PG, DG)).reshape(PG.shape)
    # Calibration ranges: where each damage branch is interpolated and where it is extrapolated, drawn
    # on the same plane as the window.
    gp_te, _ = SUR.fit_field("tau_ext_Pa", rows, log10=True)
    gp_tw, _ = SUR.fit_field("tau_w_Pa", rows, log10=True)
    LTE = gp_te.predict(z(PG, DG)).reshape(PG.shape)
    LTW = gp_tw.predict(z(PG, DG)).reshape(PG.shape)

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
    # Light grey on the field, like the white constraint lines (black in the key, as those are),
    # so the ranges stay readable on the dark end of the damage colormap.
    axa.contour(PG, DG, LTE, levels=[np.log10(440.0), np.log10(9251.0)], colors="#E6E6E6",
                linewidths=1.5, linestyles=":", zorder=4)
    axa.contour(PG, DG, LTW, levels=[np.log10(100.0), np.log10(1000.0)], colors="#E6E6E6",
                linewidths=1.3, linestyles="-.", zorder=4)
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
    # The four checks fixed on the recalibrated damage limit (prespecify_boundary_checks.py), drawn as
    # squares so the two sets stay apart.
    d0_csv = checks_csv.parents[1] / "2026-10-31_boundary-checks" / "checks_vs_surrogate.csv"
    checks_d0 = list(csv.DictReader(open(d0_csv))) if d0_csv.exists() else []
    for c in checks_d0:
        axa.plot(float(c["pressure_kPa"]), float(c["needle_um"]), marker="s", ms=ms_check,
                 mfc=S.ANNOT, mec="white", mew=0.5, ls="none", zorder=7, clip_on=True)
    print(f"  FEM checks drawn: {len(checks)} + {len(checks_d0)}")

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
                 ms=ms_check, label="check, published law")] if checks else []) + (
        [Line2D([], [], marker="s", mfc=S.ANNOT, mec="white", mew=0.5, ls="none",
                ms=ms_check, label="check, recalibrated limit")] if checks_d0 else []) + [
        # The swatch is filled before it is hatched.
        Patch(facecolor=plt.get_cmap(S.CMAP_DT)(0.45), edgecolor=HATCH_C,
              hatch=HATCH, linewidth=0.6,
              label="outside the window"),
        Line2D([], [], color=S.INK, lw=2.6, label=f"$D_t$ = {D_LIMIT:.0f}%"),
        Line2D([], [], color=S.INK, lw=2.2, ls="--", label=f"$Q$ = {Q_FLOOR:.0f} µL/s"),
        Line2D([], [], color=S.INK, lw=1.1, ls=":", label="$\\tau_e$ = 440 and 9251 Pa"),
        Line2D([], [], color=S.INK, lw=1.0, ls="-.", label="$\\tau_w$ = 100 and 1000 Pa"),
    ] + [
    # Three columns under the full-width panel: markers, then region and constraints, then the
    # calibration ranges.
    ], loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=3, frameon=False,
        fontsize=P["legend"], handlelength=1.9, columnspacing=2.0,
        borderaxespad=0.0)

    for d in dias:
        rr = by[d]
        axb.semilogx([r["Q_uLs"] for r in rr], [r["Dt_pct"] for r in rr],
                     marker=S.dia_marker(d), ls="-",
                     color=S.dia_color(d), lw=P["lw_data"], ms=P["ms"],
                     mec="white", mew=0.8, label=S.dia_label(d), zorder=3)
    axb.axhline(D_LIMIT, color=S.THRESHOLD, lw=1.2, ls="--", zorder=1,
                label=f"$D_t$ = {D_LIMIT:.0f}% damage limit")
    # Named in the legend, not printed across the panel

    # The read-out is restricted to tips that actually reach Q_REF.
    at_ref, out_of_range = {}, []
    for d in dias:
        s = by[d]
        qs = [r["Q_uLs"] for r in s]
        if not (min(qs) <= Q_REF <= max(qs)):
            out_of_range.append(d)
            continue
        # monotone in log10 Q (PCHIP), the method the text states and scripts/matched_flow.py uses; panel B draws
        # straight segments between the solved cases, which differ from it by up to 0.12 pp at Q_REF (250 µm)
        at_ref[d] = float(PchipInterpolator(np.log10(qs), [r["Dt_pct"] for r in s])(np.log10(Q_REF)))
    # Named in the key.
    axb.axvline(Q_REF, ymax=0.975, color=S.ANNOT, lw=1.4, ls=":", zorder=2,
                label=f"$Q$ = {Q_REF:.1f} µL/s, reference condition")
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
    leg_b = axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.150), ncol=3,
                       frameon=False, fontsize=P["legend"], handlelength=1.9,
                       columnspacing=2.0, borderaxespad=0.0)

    # An inset magnifies the empty upper left of (b), where the curves meet the reference line.
    INSET_QSPAN, INSET_DPAD = 2.2, 1.0
    drawn_at_ref = {}
    for d in at_ref:
        s = by[d]
        drawn_at_ref[d] = float(np.interp(np.log10(Q_REF), np.log10([r["Q_uLs"] for r in s]),
                                          [r["Dt_pct"] for r in s]))
    i_x = (Q_REF / INSET_QSPAN, Q_REF * INSET_QSPAN)
    i_y = (np.floor(2 * (min(drawn_at_ref.values()) - INSET_DPAD)) / 2,
           np.ceil(2 * (max(drawn_at_ref.values()) + INSET_DPAD)) / 2)
    # Upper left of (b), the empty region: above the 20 % limit line, left of the dotted reference
    # line.
    axins = axb.inset_axes([0.075, 0.43, 0.43, 0.52])
    axins.set_facecolor("white")
    for d in dias:
        rr = by[d]
        axins.semilogx([r["Q_uLs"] for r in rr], [r["Dt_pct"] for r in rr],
                       marker=S.dia_marker(d), ls="-",
                       color=S.dia_color(d), lw=P["lw_data"], ms=P["ms"],
                       mec="white", mew=0.8, label="_nolegend_", zorder=3)
    axins.axvline(Q_REF, color=S.ANNOT, lw=1.4, ls=":", zorder=2, label="_nolegend_")
    axins.set_xlim(*i_x)
    axins.set_ylim(*i_y)
    # Few, readable ticks: three flow rates in plain numbers and whole percentage points.
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter, MultipleLocator
    axins.xaxis.set_major_locator(FixedLocator([10, 20, 30]))
    axins.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axins.xaxis.set_minor_locator(NullLocator())
    axins.yaxis.set_major_locator(MultipleLocator(1.0))
    axins.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axins.tick_params(labelsize=P["tick"])
    _ind = axb.indicate_inset_zoom(axins, edgecolor=S.MUTE, linewidth=0.8, alpha=1.0)
    # The connectors run from the frame's top corners to the inset's bottom corners.
    from matplotlib.patches import ConnectionPatch
    for _c in _ind.connectors:
        _c.set_visible(False)
    zoom_links = []
    for _ins_xy, _main_xy in (((0, 0), (i_x[0], i_y[1])), ((1, 0), (i_x[1], i_y[1]))):
        _cp = ConnectionPatch(xyA=_ins_xy, coordsA=axins.transAxes, xyB=_main_xy,
                              coordsB=axb.transData, arrowstyle="-", color=S.MUTE,
                              linewidth=0.8, zorder=2.5)
        axb.add_artist(_cp)
        zoom_links.append(_cp)
    print(f"  inset limits: Q {i_x[0]:.2f} to {i_x[1]:.2f} µL/s, D_t {i_y[0]:.1f} to {i_y[1]:.1f} %")
    print(f"  inset at Q_REF = {Q_REF:.2f} µL/s, drawn curve (linear in log10 Q) vs PCHIP (text): "
          + " · ".join(f"Ø{d} {drawn_at_ref[d]:.2f} / {at_ref[d]:.2f} %" for d in sorted(at_ref)))
    _ordered = [drawn_at_ref[d] for d in sorted(drawn_at_ref)]
    assert all(a > b for a, b in zip(_ordered, _ordered[1:])), (
        f"the inset was drawn to show D_t falling with diameter at Q_REF, and it does not: "
        f"{drawn_at_ref}")
    assert all(i_y[0] < v < i_y[1] for v in drawn_at_ref.values()), "a curve leaves the inset at Q_REF"

    # No tight_layout: it cannot see a colour bar placed as its own axes, or a key hung below a
    # panel, as something needing room. The layout is computed from measured ink instead.
    layout_column(fig, axa, cax, axb, leg_a, leg_b)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fig_bb = fig.bbox

    # The inset hides no data and no key.
    _ib = axins.get_tightbbox(rend)
    for line in axb.lines:
        xy = axb.transData.transform(line.get_xydata())
        xy = xy[np.isfinite(xy).all(axis=1)]
        if line.get_label().startswith("Ø") and len(xy) > 1:
            _d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            _t = np.linspace(0, _d[-1], 2000)
            xy = np.column_stack([np.interp(_t, _d, xy[:, 0]), np.interp(_t, _d, xy[:, 1])])
        elif not line.get_label().startswith("Ø"):
            # Axhline / axvline: their x or y data are in axes fraction, so take the window extent
            _lb = line.get_window_extent(rend)
            xy = np.array([[_lb.x0, _lb.y0], [_lb.x1, _lb.y1],
                           [0.5 * (_lb.x0 + _lb.x1), 0.5 * (_lb.y0 + _lb.y1)]])
            if line.get_xdata()[0] == line.get_xdata()[-1]:          # vertical: test its x
                assert not (_ib.x0 <= _lb.x0 <= _ib.x1), f"the inset covers {line.get_label()!r}"
            else:                                                     # horizontal: test its y
                assert not (_ib.y0 <= _lb.y0 <= _ib.y1), f"the inset covers {line.get_label()!r}"
            continue
        _n = int(((xy[:, 0] >= _ib.x0) & (xy[:, 0] <= _ib.x1)
                  & (xy[:, 1] >= _ib.y0) & (xy[:, 1] <= _ib.y1)).sum())
        assert _n == 0, f"the inset covers {_n} sampled points of {line.get_label()!r}"
    _lb = leg_b.get_window_extent(rend)
    assert _lb.y1 < _ib.y0, "the inset overlaps the key of (b)"
    _ab = axb.get_window_extent(rend)
    assert _ab.x0 < _ib.x0 and _ib.x1 < _ab.x1 and _ab.y0 < _ib.y0 and _ib.y1 < _ab.y1, (
        "the inset's ink leaves panel (b)")
    # The two connector lines may cross a curve, but never a solved marker: every marker of (b)
    # must stay at least its own radius plus 0.5 mm from both lines.
    _px_mm = fig.dpi / 25.4
    for _cp in zoom_links:
        _a = axins.transAxes.transform(_cp.xy1)
        _b = axb.transData.transform(_cp.xy2)
        _seg = np.linspace(_a, _b, 400)
        for line in axb.lines:
            if not line.get_label().startswith("Ø"):
                continue
            _m = axb.transData.transform(line.get_xydata())
            _dist = np.min(np.linalg.norm(_seg[:, None, :] - _m[None, :, :], axis=2))
            _need = (0.5 * line.get_markersize() / 72 * 25.4 + 0.5) * _px_mm
            assert _dist >= _need, (f"a connector passes {_dist / _px_mm:.2f} mm from a "
                                    f"{line.get_label()} marker; need {_need / _px_mm:.2f} mm")
    print("  inset: clear of every data curve, both reference lines and the key; "
          "connectors clear of every marker")
    S.assert_labels_clear(fig, [axa, axb, axins])

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
