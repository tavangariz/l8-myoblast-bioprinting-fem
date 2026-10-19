#!/usr/bin/env python3
"""Fig. 3: predicted damage against nozzle diameter at 200 kPa.

(A) Total damage and (B) its entrance and nozzle-shear terms, read from the grid. (C) Shear rate at
nozzle entry against entry radius for the 168 tracers at the reference condition (200 kPa, 200 µm),
integrated through the solved field in results/fem/checkpoints/.

    python scripts/fig3_damage_vs_diameter.py
"""
# No numeric result is printed into an axes; the numbers the caption quotes go to stdout.

from __future__ import annotations
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l8_style as S                     # noqa: E402
import l8_supporting_data as supporting  # noqa: E402

from l8_paths import CANONICAL_GRID, CHECKPOINT, require   # noqa: E402
OUT = HERE.parent / "results/figures/Fig3_damage_vs_diameter.png"
GRID = require(CANONICAL_GRID, "the canonical 32-case grid")
MM = 1 / 25.4
W_IN, H_IN, DPI = 170 * MM, 72 * MM, 400
TIPS = (100, 200, 250, 330)
P_REF = 200.0


def main():
    rows = [r for r in csv.DictReader(open(GRID))
            if abs(float(r["pressure_kPa"]) - P_REF) < 1e-6]
    by = {int(round(float(r["needle_um"]))): r for r in rows}
    dt = {d: float(by[d]["Dt_pct"]) for d in TIPS}
    det = {d: float(by[d]["Det_pct"]) for d in TIPS}
    dst = {d: float(by[d]["Dst_pct"]) for d in TIPS}

    # Through rc(), not A hand-built rcParams dict, so weight, box, line width and tick colour come
    # from the house style exactly as in every other figure.
    plt.rcParams.update(S.rc(S.PAPER))
    plt.rcParams["axes.linewidth"] = 0.8
    fig, axes = plt.subplots(1, 3, figsize=(W_IN, H_IN), dpi=DPI)
    # Keys at 8 pt, the PAPER legend size. Three panels across 170 mm leave a 36 mm plot box,
    # and 8 pt fits inside it once the tick density is thinned below.
    FS_G = S.PAPER["legend"]
    for ax in axes:
        ax.tick_params(length=2.5, width=0.7)
        ax.grid(axis="y", which="major", color="#D9D9D9", lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # (a) the headline, isolated
    a = axes[0]
    x = np.arange(len(TIPS))
    a.bar(x, [dt[d] for d in TIPS], 0.62, color=[S.DIA_C[d] for d in TIPS],
          edgecolor="#333333", lw=0.7)
    # No bar-top values and no arrow marking the rise: the headline is the caption's to state,
    # so the rise is printed to stdout for the caption to quote exactly.
    rise = (dt[330] / dt[200] - 1) * 100
    # No "Ø" prefix. At 8 pt four prefixed labels crowd into one
    # string, and the axis label already says these are nozzle diameters.
    a.set_xticks(list(x)); a.set_xticklabels([str(d) for d in TIPS])
    a.set_xlim(-0.70, len(TIPS) - 0.30)
    a.set_ylim(0, max(dt.values()) * 1.10)
    a.set_xlabel(S.LBL["diameter"])
    # The pressure is in the caption, not in this y label: a bold label long enough to carry
    # it grows taller than the 39 mm panel and collides with the panel letter.
    a.set_ylabel(S.LBL["damage_t"])
    S.panel_label(a, "A")

    # (b) the budget, tip by tip
    b = axes[1]
    # The same two components as figure 4(d), so they get the same two COLOURS and the same two names:
    # S.COMP_C, "shear" and "extensional".
    b.bar(x, [dst[d] for d in TIPS], 0.58, color=S.COMP_C["shear"], edgecolor="#333333",
          lw=0.7, label="shear  $D_{st}$")
    b.bar(x, [det[d] for d in TIPS], 0.58, bottom=[dst[d] for d in TIPS],
          color=S.COMP_C["extensional"], edgecolor="#333333", lw=0.7,
          label="extensional  $D_{et}$")
    b.set_xticks(list(x)); b.set_xticklabels([str(d) for d in TIPS])
    b.set_ylim(0, max(dt.values()) * 1.30)
    b.set_ylabel(f"Damage at {P_REF:.0f} kPa (%)")
    b.set_xlabel(S.LBL["diameter"])
    S.panel_label(b, "B")
    print("  caption numbers  ·  " + "  ".join(f"Ø{d}: Dt={dt[d]:.2f} % "
          f"(Det share {100*det[d]/dt[d]:.1f} %)" for d in TIPS))
    print(f"  caption numbers  ·  Ø200 -> Ø330 rise = +{rise:.1f} %")
    # The key sits inside, upper left: the bars rise to the right, so that corner is the only
    # empty one, and the 30 % headroom above the tallest bar is there to hold it.
    b.legend(frameon=False, fontsize=FS_G, loc="upper left", handlelength=1.1,
             labelspacing=0.25, borderpad=0.1)
    b.grid(axis="y", which="major", color="#D9D9D9", lw=0.5)

    # (c) the shear dose per tracer at tip entry
    require(CHECKPOINT, "the solver checkpoint")
    r_um, gdot = supporting.tracer_entry()
    # Tracer_entry() returns the entry radius in metres; the axis is labelled µm.
    r_um = np.asarray(r_um, float)
    if float(np.nanmax(r_um)) < 1e-3:
        r_um = r_um * 1e6
    c = axes[2]
    # One colour.
    c.scatter(r_um, gdot, s=5.5, color=S.DIA_C[200], edgecolors="white", linewidths=0.15)
    med = float(np.median(gdot))
    # Named in a key, so the grey dashed rule across the scatter says what it is.
    c.axhline(med, color=S.MUTE, lw=0.9, ls=(0, (4, 3)), label="median")
    c.legend(frameon=False, fontsize=FS_G, loc="upper left", handlelength=1.4,
             labelspacing=0.25, borderpad=0.1)
    # The median line stays; its value is a caption sentence
    c.set_yscale("log")
    c.set_xlabel(S.LBL["entry_r"])
    # The x axis already says "Nozzle entry radius", so the y axis need not repeat "nozzle",
    # and the shorter label fits the 36 mm panel.
    c.set_ylabel("Shear rate at entry (s$^{-1}$)")
    S.panel_label(c, "C")

    # Rect leaves the top strip for the panel letters, which sit above the whole panel.
    fig.tight_layout(w_pad=1.9, rect=(0, 0, 1, 0.935))
    # Panel letters snap to each panel's real top-left corner once the layout is fixed.
    S.snap_panel_labels(fig)
    # After the layout is final: tick label positions are not known before it.
    S.assert_labels_clear(fig, axes)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=DPI, facecolor="white")
    from l8_paths import flatten_png; flatten_png(OUT)
    plt.close(fig)
    from PIL import Image
    px = Image.open(OUT).size
    print(f"wrote {OUT.name}  {px[0]}x{px[1]} px = {px[0]/DPI*25.4:.0f} x "
          f"{px[1]/DPI*25.4:.0f} mm at {DPI} dpi")
    print(f"  (a) Ø200 {dt[200]:.2f} % -> Ø330 {dt[330]:.2f} % = {rise:+.1f} %")
    print(f"  (b) entrance share: " + " · ".join(f"Ø{d} {100*det[d]/dt[d]:.1f} %" for d in TIPS))
    print(f"  (c) {len(r_um)} tracers, median {med:,.0f} 1/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
