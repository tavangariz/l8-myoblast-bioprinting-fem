#!/usr/bin/env python3
"""Additional file 1, Fig. S2: three checks on the model.

(A) Change in predicted total damage between meshes M12 and M16. (B) Entrance share of the
predicted damage against applied pressure. (C) Entrance pressure against a Bagley extrapolation
to zero nozzle length.

    python scripts/figS2_model_checks.py
"""
from __future__ import annotations
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l8_style as S                         # noqa: E402
import l8_supporting_data as supporting      # noqa: E402

from l8_paths import CANONICAL_GRID, require          # noqa: E402
OUT = HERE.parent / "results/figures/FigS2_model_checks.png"
MM = 1 / 25.4
W_IN, H_IN, DPI = 170 * MM, 72 * MM, 400
TIPS = (100, 200, 250, 330)


def main():
    series, dt_all, dq_all, head = supporting.read_pairs()
    rows = supporting.residual_rows()

    import csv
    canon = require(CANONICAL_GRID, "the canonical 32-case grid")
    by_tip = {d: [] for d in TIPS}
    for r in csv.DictReader(open(canon)):
        d = int(round(float(r["needle_um"])))
        by_tip[d].append((float(r["pressure_kPa"]),
                          100 * float(r["Det_pct"]) / float(r["Dt_pct"])))
    for d in by_tip:
        by_tip[d].sort()

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
        ax.grid(True, which="major", color="#D9D9D9", lw=0.5)
        ax.set_axisbelow(True)

    # (a) mesh convergence
    a = axes[0]
    for d in TIPS:
        p, ddt = series[d]
        a.plot(p, ddt, ls="-", marker=S.dia_marker(d), color=S.DIA_C[d], lw=1.1, ms=2.6,
               mec="white", mew=0.5,
               label=f"Ø{d}")
    worst = max(v[1].max() for v in series.values())
    # The worst difference, drawn as a rule and labelled inline below.
    a.axhline(worst, color=S.MUTE, lw=0.9, ls=(0, (4, 3)))
    print(f"  caption (a)  ·  worst mesh difference {worst:.3f} pp")
    a.set_xscale("log"); a.set_yscale("log")
    # A decade of headroom above the worst difference. A log axis pays for this in almost no
    # ink, and the panel spans nearly three decades either way.
    a.set_ylim(top=worst * 10.0)
    # Three labels, not five. At 8 pt "100", "200" and "400" overlap in a 36 mm box; the
    # decades are what the reader needs off a log axis, not every sampled pressure.
    a.set_xticks([25, 100, 400]); a.set_xticklabels(["25", "100", "400"])
    a.minorticks_off()
    a.set_xlabel(S.LBL["pressure"])
    # Short, because a longer label grows taller than this 39 mm panel and meets the panel
    # letter. The caption names the two meshes, so the axis label does not have to repeat it.
    a.set_ylabel("|Δ$D_t$| between meshes (pp)")
    S.panel_label(a, "A")
    # No key inside this panel.
    a.annotate("worst difference", xy=(0.03, worst), xycoords=("axes fraction", "data"),
               xytext=(0, -3), textcoords="offset points", fontsize=FS_G, color=S.MUTE,
               ha="left", va="top")

    # Measured, not eyeballed. With no key in the panel, the one thing that can cover a sample is
    # the inline rule label, so that is what is tested.
    fig.canvas.draw()
    _rend = fig.canvas.get_renderer()
    _wl = [t for t in a.texts if t.get_text() == "worst difference"][0]
    _lb = _wl.get_window_extent(_rend)
    _hit = 0
    for d in TIPS:
        _p, _ddt = series[d]
        _pts = a.transData.transform(np.column_stack([_p, _ddt]))
        _hit += int((((_pts[:, 0] >= _lb.x0) & (_pts[:, 0] <= _lb.x1)
                      & (_pts[:, 1] >= _lb.y0) & (_pts[:, 1] <= _lb.y1))).sum())
    assert _hit == 0, f"panel (a) rule label covers {_hit} plotted samples"

    # (b) entrance share
    b = axes[1]
    for d in TIPS:
        xs = [x for x, _ in by_tip[d]]; ys = [y for _, y in by_tip[d]]
        b.plot(xs, ys, ls="-", marker=S.dia_marker(d), color=S.DIA_C[d], lw=1.1, ms=2.6,
               mec="white", mew=0.5)
    # At 50 % the entrance term is half of the total, which is the reference the four curves are read
    # against.
    b.axhline(50, color=S.MUTE, lw=0.9, ls=(0, (4, 3)))
    b.annotate("half of the total", xy=(0.97, 50), xycoords=("axes fraction", "data"),
               xytext=(0, -3), textcoords="offset points", fontsize=FS_G, color=S.MUTE,
               ha="right", va="top")
    b.set_xlabel(S.LBL["pressure"])
    # The caption defines the quantity: "(B) The entrance share of the dose, D_et/D_t,
    # against applied pressure". The axis does not have to carry the ratio as well.
    b.set_ylabel("Entrance share (%)")
    S.panel_label(b, "B")

    # (c) reported entrance pressure against the Bagley intercept
    c = axes[2]
    x = np.arange(len(rows)); w = 0.36
    # Not a diameter colour.
    c.bar(x - w / 2, [r[1] for r in rows], w, color=S.ANNOT, label="Bagley intercept")
    c.bar(x + w / 2, [r[4] for r in rows], w, color="#E2E8F0", edgecolor="#333333", lw=0.7,
          label="reported $p_{en}$")
    # The result value belongs in the caption, not on the axes; it is printed to stdout so the
    # caption stays in step with the data.
    print("  caption (c)  ·  " + "  ".join(f"Ø{n}: {100*(rep-ic)/ic:+.1f} %"
                                          for n, ic, pr, fr, rep in rows))
    # No "Ø" prefix. At 8 pt four prefixed labels run together into "Ø100Ø200Ø250Ø330"; the axis
    # label already says what these numbers are.
    c.set_xticks(list(x)); c.set_xticklabels([str(n) for n, *_ in rows])
    # Headroom for the key. At 1.10 x the tallest bar the two-row key sat on the Ø100 and Ø200
    # bars and hid the "Bagley intercept" entry; save_figure now refuses a key over data.
    c.set_ylim(0, max(max(r[1] for r in rows), max(r[4] for r in rows)) * 1.45)
    c.set_xlabel(S.LBL["diameter"])
    c.set_ylabel("Entrance pressure (kPa)")
    S.panel_label(c, "C")
    c.legend(frameon=False, fontsize=FS_G, loc="upper left", handlelength=1.2,
             labelspacing=0.25, borderpad=0.1)

    # One key for the whole figure, below it, naming the four diameters that panels (A) and (B) share.
    fig.tight_layout(w_pad=1.9, rect=(0, 0.085, 1, 0.935))
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=S.DIA_C[d], marker=S.dia_marker(d), ms=3.0, lw=1.1,
                               mec="white", mew=0.5, label=f"Ø{d} µm") for d in TIPS],
               loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=4, frameon=False,
               fontsize=FS_G, handlelength=1.6, columnspacing=2.0, borderaxespad=0.0)
    # Panel letters snap to each panel's real top-left corner once the layout is fixed.
    S.snap_panel_labels(fig)
    # After the layout is final: tick label positions are not known before it.
    S.assert_labels_clear(fig, axes)
    # Through save_figure, like every other figure, so the key-over-data and canvas-edge guards
    # run here too (this figure saved directly, which is how panel (c)'s key sat on its bars).
    from l8_paths import save_figure
    save_figure(fig, OUT.name, DPI)
    plt.close(fig)
    from PIL import Image
    px = Image.open(OUT).size
    print(f"wrote {OUT.name}  {px[0]}x{px[1]} px = {px[0]/DPI*25.4:.0f} x "
          f"{px[1]/DPI*25.4:.0f} mm at {DPI} dpi")
    print(f"  (a) worst mesh difference {worst:.4f} pp")
    print(f"  (b) entrance share at 200 kPa: "
          + " · ".join(f"Ø{d} {dict(by_tip[d])[200.0]:.1f} %" for d in TIPS))
    print(f"  (c) reported vs Bagley: "
          + " · ".join(f"Ø{n} {100*(rep-ic)/ic:+.2f} %" for n, ic, pr, fr, rep in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
