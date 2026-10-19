#!/usr/bin/env python3
"""Additional file 1, Fig. S1: surrogate error with a whole nozzle diameter or applied pressure held out.

Reads results/surrogate/ (written by surrogate_validation.py) and fits nothing.

    python scripts/figS1_blocked_holdouts.py
"""
from __future__ import annotations

import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import l8_style as S
from l8_paths import save_figure

FIGNAME = "FigS1_blocked_holdouts.png"
TABLE = pathlib.Path(__file__).resolve().parents[1] / "results" / "surrogate" / "blocked_holdouts.csv"
FILL = {"interior": S.MUTE, "edge": S.ANNOT}


def load():
    if not TABLE.exists():
        raise SystemExit(f"{TABLE} not found: run scripts/surrogate_validation.py first")
    rows = list(csv.DictReader(open(TABLE)))
    return {h: [r for r in rows if r["holdout"] == h] for h in ("diameter", "pressure")}


def panel(ax, rows, mean_key, max_key, letter, xlabel, ylabel):
    x = np.arange(len(rows))
    mean = np.array([float(r[mean_key]) for r in rows])
    worst = np.array([float(r[max_key]) for r in rows])
    kinds = [r["kind"] for r in rows]
    ax.bar(x, mean, width=0.62, color=[FILL[k] for k in kinds], edgecolor="none", zorder=2)
    # The worst case as a plain whisker and cap. errorbar() caps are markers, and the house
    # rc styles markers in a way that draws them white on this background.
    lw = S.PAPER["lw_axis"]
    ax.vlines(x, mean, worst, color=S.INK, lw=lw, zorder=3)
    ax.hlines(worst, x - 0.14, x + 0.14, color=S.INK, lw=lw, zorder=3)
    ax.set_xticks(x, [f"{float(r['held_out']):g}" for r in rows])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, worst.max() * 1.12)
    ax.grid(False, axis="x")
    ax.grid(axis="y", color=S.GRID_C, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    S.panel_label(ax, letter)


def main():
    g = load()
    dt_lbl = "Held-out error  $|\\Delta D_t|$  (pp)"
    q_lbl = "Held-out error  $|\\Delta \\log_{10} Q|$"
    with plt.rc_context(S.rc(S.PAPER)):
        # 170 mm on the page after the tight crop, as every manuscript figure.
        fig, ax = plt.subplots(2, 2, figsize=(6.9191, 5.0),
                               gridspec_kw={"width_ratios": [1, 1.7]})
        panel(ax[0, 0], g["diameter"], "D_t_mae_pp", "D_t_max_pp", "A", S.LBL["diameter"], dt_lbl)
        panel(ax[0, 1], g["pressure"], "D_t_mae_pp", "D_t_max_pp", "B", S.LBL["pressure"], dt_lbl)
        panel(ax[1, 0], g["diameter"], "log10_Q_mae_dex", "log10_Q_max_dex", "C",
              S.LBL["diameter"], q_lbl)
        panel(ax[1, 1], g["pressure"], "log10_Q_mae_dex", "log10_Q_max_dex", "D",
              S.LBL["pressure"], q_lbl)
        # in D, where the bars leave the upper left empty
        ax[1, 1].legend(handles=[Patch(color=FILL["interior"], label="Interior holdout"),
                                 Patch(color=FILL["edge"], label="Edge holdout")],
                        loc="upper left", frameon=False)
        fig.tight_layout(pad=1.1, w_pad=2.5, h_pad=2.0)
        S.snap_panel_labels(fig)
        S.assert_keys_clear_of_data(fig)
        for w in save_figure(fig, FIGNAME, S.PAPER["dpi"], tight=True):
            print(f"  wrote {w}")


if __name__ == "__main__":
    main()
