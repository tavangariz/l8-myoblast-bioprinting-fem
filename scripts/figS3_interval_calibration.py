#!/usr/bin/env python3
"""Additional file 1, Fig. S3: leave-one-out error against the nominal 95 % interval of the surrogate.

Each of the 32 predictions of total damage is plotted at its nominal 95 % half-width (1.96
posterior standard deviations) against its actual error; a point below the line lies inside its
own interval.

    python scripts/figS3_interval_calibration.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import l8_style as S
import l8_surrogate as SUR
from l8_paths import save_figure
from surrogate_diagnostics import gp, leave_one_out, panel_calibration

FIGNAME = "FigS3_interval_calibration.png"


def main():
    rows = SUR.load_grid()
    X = np.array([[r["pressure_kPa"], r["needle_um"]] for r in rows])
    y = np.array([r["Dt_pct"] for r in rows])
    dia = np.array([int(r["needle_um"]) for r in rows])

    pred, sd = leave_one_out(X, y, gp, with_sd=True)
    mae = float(np.mean(np.abs(pred - y)))
    if abs(mae - SUR.LOO_DT_REPORTED) > SUR.LOO_TOL:
        raise SystemExit(f"REFUSING: LOO MAE {mae:.3f} pp is not the reported "
                         f"{SUR.LOO_DT_REPORTED:.3f} pp")

    with plt.rc_context(S.rc(S.PAPER)):
        # One square panel, about 85 mm wide on the page once trimmed, so canvas points are
        # page points when Additional file 1 places it at its own trimmed width.
        fig, ax = plt.subplots(figsize=(3.45, 3.3))
        inside = panel_calibration(ax, y, pred, sd, dia, letter=None)
        # The diameter key that panel A carries in the four-panel diagnostics. Upper left is
        # empty: every point lies below an error of 2.5 pp.
        handles = [plt.Line2D([], [], marker=S.dia_marker(d), color=S.dia_color(d),
                              markeredgecolor="white", markersize=S.PAPER["ms"],
                              linestyle="none", label=S.dia_label(d))
                   for d in sorted(set(dia))]
        ax.legend(handles=handles, loc="upper left", title="Nozzle diameter",
                  fontsize=S.PAPER["legend"] - 1.5)
        fig.tight_layout(pad=0.6)
        print(f"  leave-one-out MAE {mae:.3f} pp; {inside} of {len(y)} inside "
              f"the nominal 95 % interval ({100 * inside / len(y):.1f} %)")
        for w in save_figure(fig, FIGNAME, S.PAPER["dpi"], tight=True):
            print(f"  wrote {w}")


if __name__ == "__main__":
    main()
