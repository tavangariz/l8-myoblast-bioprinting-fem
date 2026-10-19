#!/usr/bin/env python3
"""Fig. 4: flow rate, wall shear stress and predicted damage against applied pressure.

(A) Mass flow rate, with the flow rates weighed by Ning et al. (2018). (B) Nozzle wall shear
stress, with the values they computed. (C) Total damage, with the 20 % damage limit. (D) Entrance
and shear terms for the 200 µm nozzle. Reads the grid only, and stops if the grid no longer gives
the values reported in the article.

    python scripts/fig4_pressure_dependence.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l8_style as S

from l8_paths import CANONICAL_GRID, require, save_figure

GRID = require(CANONICAL_GRID, "the canonical FEM grid")
FIGNAME = "Fig4_pressure_dependence.png"

# Ning et al. 2018, transcribed from the published tables (p.
NING_Q = {50: 2.6e-4, 100: 1.8e-3, 200: 1.7e-2, 300: 4.1e-2, 400: 9.8e-2}
NING_TW = {50: 246, 100: 491, 200: 981, 400: 1909}

# Bioink density, used only to put our Q in the mass units Ning weighed in.
RHO_G_PER_UL = 0.99e-3          # g per microlitre
BREAKDOWN_TIP = 200             # panel (d) fixes one tip, so no diameter colour is in play

# The grid this figure is allowed to draw. If the CSV ever stops reproducing these, the
# manuscript's own numbers have moved and the figure must not be regenerated silently.
CHECK = {(200, 200.0): 8.956, (250, 200.0): 11.161, (330, 200.0): 18.000}
TOL = 0.002


def _maybe_float(v):
    """Numeric columns become floats; provenance columns stay text.

    The canonical grid carries a text `source` column naming the run each row came
    from, so every column cannot simply be coerced with float(). Same helper as
    l8_surrogate.load_grid.
    """
    if v in ("", None):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return v


def load():
    rows = []
    with open(GRID) as f:
        for r in csv.DictReader(f):
            rows.append({k: _maybe_float(v) for k, v in r.items()})
    # The row count is checked against the diameters and pressures the file itself
    # contains, never a constant, so the check follows the grid when it is widened.
    dias = {round(r["needle_um"]) for r in rows}
    press = {r["pressure_kPa"] for r in rows}
    if len(rows) != len(dias) * len(press):
        sys.exit(f"REFUSING: {len(rows)} rows is not a full "
                 f"{len(dias)} x {len(press)} grid")

    by = defaultdict(list)
    for r in rows:
        by[int(r["needle_um"])].append(r)
    for d in by:
        by[d].sort(key=lambda r: r["pressure_kPa"])

    for (d, p), want in CHECK.items():
        got = next(r["Dt_pct"] for r in by[d] if r["pressure_kPa"] == p)
        if abs(got - want) > TOL:
            sys.exit(f"REFUSING: Ø{d} at {p:.0f} kPa is {got:.3f} %, the article reports "
                     f"{want:.3f} %. The grid has changed.")
    print(f"  canonical grid verified: {sorted(by)} µm, "
          f"{len(by[200])} pressures, mesh M{int(rows[0]['mesh_M'])}")
    return by


def main():
    by = load()
    dias = sorted(by)

    plt.rcParams.update(S.rc(S.PAPER))
    # 170 mm canvas.
    fig, axes = plt.subplots(2, 2, figsize=(S.PAGE_W_IN, 6.0065), dpi=S.PAPER["dpi"])
    P = S.PAPER

    ax = axes[0, 0]
    for d in dias:
        rr = by[d]
        ax.semilogy([r["pressure_kPa"] for r in rr],
                    [r["Q_uLs"] * RHO_G_PER_UL * 1e3 for r in rr],
                    marker=S.dia_marker(d), ls="-", color=S.dia_color(d), lw=P["lw_data"], ms=P["ms"],
                    mec="white", mew=0.6, label=S.dia_label(d), zorder=3)
    ax.plot(sorted(NING_Q), [NING_Q[p] * 1e3 for p in sorted(NING_Q)],
            label="Ning 2018, weighed $Q$", zorder=4, **S.NING_MARKER)
    ax.set_xlabel(S.LBL["pressure"])
    ax.set_ylabel("Mass flow rate  ($\\times10^{-3}$ g s$^{-1}$)")
    S.panel_label(ax, "A")

    ax = axes[0, 1]
    for d in dias:
        rr = by[d]
        ax.plot([r["pressure_kPa"] for r in rr], [r["tau_w_Pa"] for r in rr],
                marker=S.dia_marker(d), ls="-", color=S.dia_color(d), lw=P["lw_data"], ms=P["ms"],
                mec="white", mew=0.6, label=S.dia_label(d), zorder=3)
    ax.plot(sorted(NING_TW), [NING_TW[p] for p in sorted(NING_TW)],
            label="Ning 2018, Table 2 $\\tau_w$", zorder=4, **S.NING_MARKER)
    ax.set_xlabel(S.LBL["pressure"])
    ax.set_ylabel("Nozzle wall shear stress  $\\tau_w$  (Pa)")
    S.panel_label(ax, "B")

    ax = axes[1, 0]
    for d in dias:
        rr = by[d]
        xs = [r["pressure_kPa"] for r in rr]
        ys = [r["Dt_pct"] for r in rr]
        ax.plot(xs, ys, marker=S.dia_marker(d), ls="-", color=S.dia_color(d), lw=P["lw_data"], ms=P["ms"],
                mec="white", mew=0.6, zorder=3, label=S.dia_label(d))
        # Labelled by the key, not by boxes on the plot.
    ax.axhline(20, color=S.THRESHOLD, lw=1.1, ls="--", zorder=1,
               label="$D_t$ = 20 % screening threshold")
    # The threshold is named in the key rather than printed across the panel.
    ax.set_xlabel(S.LBL["pressure"])
    ax.set_ylabel(S.LBL["damage_t"])
    S.panel_label(ax, "C")
    # The axis ends where the data ends.
    ax.set_xlim(0, 420)

    ax = axes[1, 1]
    rr = by[BREAKDOWN_TIP]
    x = np.arange(len(rr))
    dst = np.array([r["Dst_pct"] for r in rr])
    det = np.array([r["Det_pct"] for r in rr])
    ax.bar(x, dst, 0.62, color=S.COMP_C["shear"], edgecolor="white", linewidth=0.5,
           label="shear  $D_{st}$", zorder=2)
    ax.bar(x, det, 0.62, bottom=dst, color=S.COMP_C["extensional"], edgecolor="white",
           linewidth=0.5, label="extensional  $D_{et}$", zorder=2)
    ax.plot(x, [r["Dt_pct"] for r in rr], "o-", color=S.COMP_C["total"],
            lw=P["lw_data"], ms=P["ms"] - 0.5, label="total  $D_t$", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r['pressure_kPa'])}" for r in rr])
    ax.set_xlabel(S.LBL["pressure"])
    ax.set_ylabel("Damage (%)")
    S.panel_label(ax, "D")
    ax.legend(loc="upper left")

    # Reported as D_et / D_t, which is how the manuscript states it, not D_et/(D_st+D_et).
    dt = np.array([r["Dt_pct"] for r in rr])
    share = det / dt * 100
    print(f"  extensional share at Ø{BREAKDOWN_TIP} (D_et/D_t): "
          f"{share.min():.1f} % to {share.max():.1f} %")

    # One key for the whole figure, below the panels.
    _handles, _labels = [], []
    for _ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        for _h, _l in zip(*_ax.get_legend_handles_labels()):
            if _l not in _labels:
                _handles.append(_h); _labels.append(_l)
    # Four columns.
    _leg = fig.legend(_handles, _labels, loc="lower center", ncol=4,
                      frameon=False, fontsize=P["legend"] - 0.5,
                      bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handlelength=1.9)
    fig.tight_layout(rect=(0, 0.085, 1, 0.99))   # headroom for the snapped panel letters
    # Panel letters snap to each panel's real top-left corner once the layout is fixed.
    S.snap_panel_labels(fig)

    # Nothing in the key may sit on the data. Measured from the renderer, because a check on
    # values or hashes cannot see one drawn object covering another.
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    for _ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        assert _ax.get_legend() is None, "a per-panel legend survived; it will sit on the data"
    _bb = _leg.get_window_extent(_r)
    _fw, _fh = fig.canvas.get_width_height()
    assert _bb.x0 >= -0.5 and _bb.x1 <= _fw + 0.5, (
        f"the shared key runs off the canvas: x {_bb.x0:.0f}..{_bb.x1:.0f} of {_fw}")
    assert _bb.y0 >= -0.5, f"the shared key is cut off at the bottom: y0 {_bb.y0:.0f}"
    _lowest = min(a.get_window_extent(_r).y0 for a in axes.ravel())
    assert _bb.y1 <= _lowest + 0.5, "the shared key overlaps the bottom row of panels"
    print(f"  key: one shared legend below the panels, {len(_labels)} entries in 4 columns, "
          f"inside the canvas and clear of the axes")
    written = save_figure(fig, FIGNAME, P["dpi"])
    plt.close(fig)
    for w in written:
        print(f"  wrote {w}")


if __name__ == "__main__":
    main()
