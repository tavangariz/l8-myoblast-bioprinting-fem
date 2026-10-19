"""Fig. 6: Gaussian process response surfaces over applied pressure and nozzle diameter.

(A) log10 Q. (B) log10 wall shear stress. (C) Total damage, with iso-flow contours at 1, 10 and
100 µL/s. (D) Posterior standard deviation of the total damage. The surrogate is the one in
l8_surrogate.py; markers are the 32 training cases.

    python scripts/fig6_surrogate_surfaces.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import l8_style as S
import l8_surrogate as SUR
from l8_paths import save_figure

FIGNAME = "Fig6_surrogate_surfaces.png"
NGRID = 160
# 1 µL/s is the flow-rate criterion of the operating window (Fig. 7).
Q_CONTOURS = (1.0, 10.0, 100.0)   # µL/s, the throughput lines the operating window uses

# Where each iso-throughput label sits, chosen rather than left to matplotlib.
Q_LABEL_AT = ((161.8, 137.0), (253.8, 162.2), (298.6, 222.6))   # kPa, µm

# One spelling of each axis, used once per row and once per column rather than once per panel.
XLABEL = S.LBL["pressure"]
# The layout is computed, not tuned.
FIG_W_MM = 170.0
PAD_MM = 1.5             # canvas margin around the outermost ink, on every side
COL_GAP_MM = 5.0         # left column's colour-bar caption to the right column's ink
ROW_GAP_MM = 3.0         # top row's lowest ink to the bottom row's panel letter
EMPTY_BAND_MM = 3.0      # the most blank canvas the guard allows on any side
YLABEL = S.LBL["diameter"]


def surface(ax, P, D, Z, cmap, letter, cbar_label, levels=24):
    """Draw one field. Axis labels are NOT set here: all four panels share both axes, so the
    labels are applied once per outer row and column by the loop in main()."""
    m = ax.contourf(P, D, Z, levels=levels, cmap=cmap)
    S.panel_label(ax, letter)
    ax.grid(False)
    # Pad 0.03 and labelpad 7 give the colorbar the room the shared axes free up.
    cb = ax.figure.colorbar(m, cax=ax.inset_axes([1.045, 0.0, 0.055, 1.0]))
    cb.set_label(cbar_label, fontsize=S.PAPER["label"], fontweight="bold", labelpad=7)
    cb.ax.tick_params(labelsize=S.PAPER["tick"])
    return m


# The training grid is sampled to the edge of the plotted rectangle in both directions, so with the
# axis limits set to the data's own span the boundary cases land exactly on the spines and render as
# half-markers: the whole Ø100 row would be bisected by the bottom axis and the 400 kPa column by the
# right one.
AXIS_MARGIN = 0.025


def mark_cases(ax, rows):
    ax.plot([r["pressure_kPa"] for r in rows], [r["needle_um"] for r in rows],
            "o", markersize=5.5, markerfacecolor="white", markeredgecolor=S.INK,
            markeredgewidth=0.9, linestyle="none", zorder=5)
    ps = [r["pressure_kPa"] for r in rows]
    ds = [r["needle_um"] for r in rows]
    dx = (max(ps) - min(ps)) * AXIS_MARGIN
    dy = (max(ds) - min(ds)) * AXIS_MARGIN
    ax.set_xlim(min(ps) - dx, max(ps) + dx)
    ax.set_ylim(min(ds) - dy, max(ds) + dy)


def layout_squares(fig, ax):
    """Place the 2 x 2 square panels from measured ink extents; set the figure height to fit.

    Two passes: the extents are in points (type, ticks, bar width), so they barely move when
    the squares are resized, and the second pass lands on the first one's numbers."""
    for _ in range(2):
        S.snap_panel_labels(fig)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        mm = 25.4 / fig.dpi

        def ext(a):
            box, ink = a.get_window_extent(r), a.get_tightbbox(r)
            return ((box.x0 - ink.x0) * mm, (ink.x1 - box.x1) * mm,
                    (box.y0 - ink.y0) * mm, (ink.y1 - box.y1) * mm)   # left, right, below, above

        E = [[ext(ax[i, j]) for j in range(2)] for i in range(2)]
        L = [max(E[i][j][0] for i in range(2)) for j in range(2)]      # per column
        Rr = [max(E[i][j][1] for i in range(2)) for j in range(2)]
        B = [max(E[i][j][2] for j in range(2)) for i in range(2)]      # per row
        A = [max(E[i][j][3] for j in range(2)) for i in range(2)]
        side = (FIG_W_MM - 2 * PAD_MM - L[0] - Rr[0] - COL_GAP_MM - L[1] - Rr[1]) / 2
        H = PAD_MM + A[0] + side + B[0] + ROW_GAP_MM + A[1] + side + B[1] + PAD_MM
        fig.set_size_inches(FIG_W_MM / 25.4, H / 25.4)
        x = [PAD_MM + L[0], PAD_MM + L[0] + side + Rr[0] + COL_GAP_MM + L[1]]
        y = [PAD_MM + B[1] + side + A[1] + ROW_GAP_MM + B[0], PAD_MM + B[1]]   # row 0 on top
        for i in range(2):
            for j in range(2):
                ax[i, j].set_position([x[j] / FIG_W_MM, y[i] / H, side / FIG_W_MM, side / H])
    S.snap_panel_labels(fig)
    print(f"  layout: four {side:.1f} mm squares on a {FIG_W_MM:.0f} x {H:.1f} mm canvas")


def main():
    rows = SUR.load_grid()
    gp_dt, gp_logq, z = SUR.fit_all(rows)
    gp_tau, _ = SUR.fit_field("tau_w_Pa", rows, log10=True)

    ps = np.array([r["pressure_kPa"] for r in rows])
    ds = np.array([r["needle_um"] for r in rows])
    P, D = np.meshgrid(np.linspace(ps.min(), ps.max(), NGRID),
                       np.linspace(ds.min(), ds.max(), NGRID))
    Xz = z(P, D)

    logq = gp_logq.predict(Xz).reshape(P.shape)
    dt, sd = gp_dt.predict(Xz, return_std=True)
    dt, sd = dt.reshape(P.shape), sd.reshape(P.shape)
    logtau = gp_tau.predict(Xz).reshape(P.shape)

    print(f"  D_t over the trained rectangle: {dt.min():.2f} to {dt.max():.2f} %")
    print(f"  GP posterior s.d.: {sd.min():.3f} to {sd.max():.3f} pp")

    # PAPER, not SLIDE.
    with plt.rc_context(S.rc(S.PAPER)):
        # Shared axes.
        fig, ax = plt.subplots(2, 2, figsize=(FIG_W_MM / 25.4, FIG_W_MM / 25.4), sharex=True, sharey=True)
        for _a in ax.flat:
            _a.set_box_aspect(1)

        surface(ax[0, 0], P, D, logq, S.CMAP_Q, "A",
                "log$_{10}$($Q$ / (µL/s))")   # the unit as the text writes it
        mark_cases(ax[0, 0], rows)

        surface(ax[0, 1], P, D, logtau, S.CMAP_TAU, "B",
                "log$_{10}$($τ_w$ / Pa)")
        mark_cases(ax[0, 1], rows)

        surface(ax[1, 0], P, D, dt, S.CMAP_DT, "C",
                S.LBL["damage_t"])
        cs = ax[1, 0].contour(P, D, 10.0 ** logq, levels=Q_CONTOURS,
                              colors="white", linewidths=2.0, linestyles="--")
        # Inline=True.
        q_labels = ax[1, 0].clabel(cs, fmt=lambda v: f"{v:g} µL/s", fontsize=S.PAPER["annot"],
                                   inline=True, inline_spacing=7, colors="white",
                                   manual=Q_LABEL_AT)
        # White text is right on the dark end of viridis and marginal on the light end, and the 100
        # µL/s label sits in the panel's mid-green.
        for _t in q_labels:
            _t.set_path_effects([pe.withStroke(linewidth=1.6, foreground=S.INK)])
        mark_cases(ax[1, 0], rows)

        surface(ax[1, 1], P, D, sd, S.CMAP_SD, "D",
                "Posterior s.d. of  $D_t$ (pp)")
        mark_cases(ax[1, 1], rows)

        # The four solved diameters as ticks.
        for _r in range(2):
            for _c in range(2):
                _ax = ax[_r, _c]
                _bottom, _left = _r == 1, _c == 0
                _ax.set_yticks([100, 200, 250, 330])
                _ax.set_xlabel(XLABEL if _bottom else "")
                _ax.set_ylabel(YLABEL if _left else "")
                _ax.tick_params(labelbottom=_bottom, labelleft=_left)

        # Journal rule: figure titles and legends go in the main manuscript, never inside the
        # graphic file, so the facts the caption states are printed to stdout.
        print(f"  for the caption: {len(set(ds)):d} tip diameters x "
              f"{len(set(ps)):d} pressures, {len(rows)} FEM cases marked")
        layout_squares(fig, ax)

        # No contour label may leave panel (c) or land on another one.
        fig.canvas.draw()
        _rend = fig.canvas.get_renderer()
        assert len(q_labels) == len(Q_CONTOURS), (
            f"{len(q_labels)} of {len(Q_CONTOURS)} iso-throughput contours carry a label; an "
            f"unlabelled iso-line is worse than no iso-line, because the reader cannot tell "
            f"which throughput it is reading the operating window against")
        _panel = ax[1, 0].get_window_extent(_rend)
        for _t in q_labels:
            _b = _t.get_window_extent(_rend)
            assert (_b.x0 >= _panel.x0 and _b.x1 <= _panel.x1
                    and _b.y0 >= _panel.y0 and _b.y1 <= _panel.y1), (
                f"contour label {_t.get_text()!r} leaves panel (c): label "
                f"[{_b.x0:.0f},{_b.x1:.0f}]x[{_b.y0:.0f},{_b.y1:.0f}] against panel "
                f"[{_panel.x0:.0f},{_panel.x1:.0f}]x[{_panel.y0:.0f},{_panel.y1:.0f}]; "
                f"move its Q_LABEL_AT entry further inside")
        for _i in range(len(q_labels)):
            for _j in range(_i + 1, len(q_labels)):
                _a = q_labels[_i].get_window_extent(_rend)
                _b = q_labels[_j].get_window_extent(_rend)
                _dx = min(_a.x1, _b.x1) - max(_a.x0, _b.x0)
                _dy = min(_a.y1, _b.y1) - max(_a.y0, _b.y0)
                assert not (_dx > 0 and _dy > 0), (
                    f"contour labels {q_labels[_i].get_text()!r} and "
                    f"{q_labels[_j].get_text()!r} overlap by {_dx:.0f} x {_dy:.0f} px; "
                    f"stagger their Q_LABEL_AT entries along their own contours")
        print(f"  contour labels: {len(q_labels)} inside panel (c), none overlapping")

        # Every panel is square, measured on the rendered axes rather than trusted to the aspect
        # setting, and every colour-scale caption fits the height of its own bar: a square panel is
        # short, and a long caption would overhang its bar.
        for _a in ax.flat:
            _b = _a.get_window_extent(_rend)
            assert abs(_b.width / _b.height - 1) < 0.01, (
                f"panel is {_b.width:.0f} x {_b.height:.0f} px, not square")
            _cax = _a.child_axes[0]
            _lab = _cax.yaxis.label.get_window_extent(_rend)
            _bar = _cax.get_window_extent(_rend)
            assert _lab.height <= _bar.height + 1, (
                f"colour-scale caption {_cax.yaxis.label.get_text()!r} is {_lab.height:.0f} px "
                f"tall against a {_bar.height:.0f} px bar; shorten it")
        # All INK on the canvas, and no band of empty canvas.
        _ink = fig.get_tightbbox(_rend)             # inches
        _fw, _fh = fig.get_size_inches()
        _bands = {"left": _ink.x0, "right": _fw - _ink.x1, "bottom": _ink.y0, "top": _fh - _ink.y1}
        for _side, _v in _bands.items():
            _mm = _v * 25.4
            assert 0 <= _mm <= EMPTY_BAND_MM, (
                f"{_side}: {_mm:.1f} mm between the ink and the canvas edge "
                f"({'ink is cut off' if _mm < 0 else 'blank band'}); allowed 0 to {EMPTY_BAND_MM} mm")
        print("  four square panels; canvas margins " +
              ", ".join(f"{k} {v * 25.4:.1f} mm" for k, v in _bands.items()))

        for w in save_figure(fig, FIGNAME, 600):   # 600 dpi: the PAPER profile carries a screen dpi, and 170 mm at 200 dpi is below the journal floor of 300
            print(f"  wrote {w}")


if __name__ == "__main__":
    main()
